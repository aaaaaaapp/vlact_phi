"""RoboTwin evaluator that never substitutes a requested scene seed."""

import json
import os
import subprocess
import traceback
from pathlib import Path


def parse_fixed_test_seeds(value: str, expected_count: int) -> list[int]:
    seeds = json.loads(value)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("ROBOTWIN_FIXED_TEST_SEEDS must be a non-empty JSON list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("every fixed RoboTwin scene seed must be an integer")
    if len(seeds) != expected_count:
        raise ValueError(
            f"fixed seed count {len(seeds)} does not match test_num {expected_count}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("fixed RoboTwin scene seeds must be unique")
    return seeds


def validate_video_file(
    video_path: str | os.PathLike[str], ffmpeg_returncode: int | None
) -> dict[str, object]:
    path = Path(video_path)
    if ffmpeg_returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {ffmpeg_returncode}: {path}"
        )
    size_bytes = path.stat().st_size if path.is_file() else 0
    if size_bytes < 1024:
        raise RuntimeError(
            f"evaluation video is missing or too small ({size_bytes} bytes): {path}"
        )
    return {"file": path.name, "size_bytes": size_bytes}


def label_video_file(
    video_path: str | os.PathLike[str], succeeded: bool
) -> Path:
    path = Path(video_path)
    outcome = "success" if succeeded else "failure"
    labeled_path = path.with_name(f"{path.stem}_{outcome}{path.suffix}")
    path.replace(labeled_path)
    return labeled_path


def eval_policy_fixed(
    base,
    task_name,
    task_env,
    args,
    model,
    st_seed,
    test_num=100,
    video_size=None,
    instruction_type=None,
):
    del st_seed
    seeds = parse_fixed_test_seeds(
        os.environ["ROBOTWIN_FIXED_TEST_SEEDS"], int(test_num)
    )
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")
    print(f"[fixed-seeds] requested scene seeds: {seeds}", flush=True)

    task_env.suc = 0
    task_env.test_num = 0
    args["eval_mode"] = True
    policy_name = args["policy_name"]
    eval_func = base.eval_function_decorator(policy_name, "eval")
    reset_func = base.eval_function_decorator(policy_name, "reset_model")
    clear_cache_freq = args["clear_cache_freq"]
    result_dir = None
    episode_records = []
    skipped_seeds = []
    skip_unstable = os.environ.get("ROBOTWIN_SKIP_UNSTABLE_SEEDS", "0") == "1"

    for episode_index, scene_seed in enumerate(seeds):
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        try:
            task_env.setup_demo(
                now_ep_num=episode_index, seed=scene_seed, is_test=True, **args
            )
            episode_info = task_env.play_once()
            stable = bool(task_env.plan_success and task_env.check_success())
        except Exception as exc:
            task_env.close_env()
            if skip_unstable:
                skipped_seeds.append({"seed": scene_seed, "reason": "expert_precheck_exception", "error": repr(exc)})
                args["render_freq"] = render_freq
                print(f"[fixed-seeds] skipping seed {scene_seed}: expert precheck exception", flush=True)
                continue
            raise RuntimeError(
                f"fixed scene seed {scene_seed} failed expert precheck"
            ) from exc
        task_env.close_env()
        if not stable:
            if skip_unstable:
                skipped_seeds.append({"seed": scene_seed, "reason": "not_expert_stable"})
                args["render_freq"] = render_freq
                print(f"[fixed-seeds] skipping seed {scene_seed}: not expert-stable", flush=True)
                continue
            raise RuntimeError(
                f"fixed scene seed {scene_seed} is not expert-stable; refusing substitution"
            )
        args["render_freq"] = render_freq

        task_env.setup_demo(
            now_ep_num=episode_index, seed=scene_seed, is_test=True, **args
        )
        descriptions = base.generate_episode_descriptions(
            args["task_name"], [episode_info["info"]], test_num
        )
        instruction = base.np.random.choice(descriptions[0][instruction_type])
        task_env.set_instruction(instruction=instruction)

        video_path = None
        ffmpeg = None
        if task_env.eval_video_path is not None:
            result_dir = task_env.eval_video_path
            video_path = Path(task_env.eval_video_path) / f"episode{task_env.test_num}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", video_size,
                    "-framerate", "10", "-i", "-", "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264", "-crf", "23",
                    str(video_path),
                ],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)

        succeeded = False
        reset_func(model)
        try:
            while task_env.take_action_cnt < task_env.step_lim:
                observation = task_env.get_obs()
                eval_func(task_env, model, observation)
                if task_env.eval_success:
                    succeeded = True
                    break
        except Exception:
            traceback.print_exc()
            raise
        finally:
            if task_env.eval_video_path is not None:
                task_env._del_eval_video_ffmpeg()

        video_record = None
        if video_path is not None and ffmpeg is not None:
            validate_video_file(video_path, ffmpeg.returncode)
            video_path = label_video_file(video_path, succeeded)
            video_record = validate_video_file(video_path, 0)

        if succeeded:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        task_env.close_env(
            clear_cache=((episode_index + 2) % clear_cache_freq == 0)
        )
        if task_env.render_freq:
            task_env.viewer.close()
        task_env.test_num += 1
        episode_records.append(
            {
                "episode": episode_index,
                "seed": scene_seed,
                "success": succeeded,
                "video": video_record,
            }
        )
        print(
            f"{task_name} | {policy_name} | {args['task_config']} | "
            f"{args['ckpt_setting']}\nSuccess rate: {task_env.suc}/{task_env.test_num}, "
            f"fixed seed: {scene_seed}\n",
            flush=True,
        )

    if result_dir is not None:
        manifest_path = os.path.join(str(result_dir), "_fixed_seeds.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seeds": seeds,
                    "successes": int(task_env.suc),
                    "episodes": int(task_env.test_num),
                    "skipped_seeds": skipped_seeds,
                    "episode_records": episode_records,
                },
                handle,
                indent=2,
            )
        print(f"[fixed-seeds] wrote {manifest_path}", flush=True)
    return max(seeds) + 1, task_env.suc


def main():
    # This file is executed by absolute path, so Python's import path does not
    # necessarily contain the RoboTwin checkout that owns ``script``.
    robotwin_path = os.environ.get("ROBOTWIN_PATH")
    starvla_path = os.environ.get("STARVLA_PATH")
    import sys
    for candidate in (robotwin_path, starvla_path):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        import script  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name == "script":
            raise RuntimeError(
                "RoboTwin script package is unavailable; set ROBOTWIN_PATH "
                "to the RoboTwin checkout (the directory containing script/)."
            ) from exc
        raise
    from script import eval_policy as base
    from script.test_render import Sapien_TEST

    Sapien_TEST()
    user_args = base.parse_args_and_config()

    def patched_eval_policy(*args, **kwargs):
        return eval_policy_fixed(base, *args, **kwargs)

    base.eval_policy = patched_eval_policy
    base.main(user_args)


if __name__ == "__main__":
    main()
