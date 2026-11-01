"""Find expert-stable RoboTwin scenes without loading a policy checkpoint."""

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    robotwin_path = os.environ["ROBOTWIN_PATH"]
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)
    from script import eval_policy as base
    from script.test_render import Sapien_TEST

    output = Path(os.environ["ROBOTWIN_DISCOVERY_OUTPUT"])
    target = int(os.environ.get("ROBOTWIN_DISCOVERY_COUNT", "100"))
    max_candidates = int(os.environ.get("ROBOTWIN_DISCOVERY_MAX_CANDIDATES", "300"))
    start_candidate = os.environ.get("ROBOTWIN_DISCOVERY_START_SEED")
    if target <= 0 or max_candidates < target:
        raise ValueError("invalid target or candidate limit")

    original_decorator = base.eval_function_decorator

    def decorator(policy_name, model_name):
        if model_name == "get_model":
            return lambda args: None
        return original_decorator(policy_name, model_name)

    def discover(task_name, task_env, args, model, st_seed, **kwargs):
        del model, kwargs
        seeds = []
        rejected = []
        args["eval_mode"] = True
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        try:
            for offset in range(max_candidates):
                seed = (int(start_candidate) if start_candidate is not None else st_seed) + offset
                try:
                    task_env.setup_demo(now_ep_num=offset, seed=seed, is_test=True, **args)
                    task_env.play_once()
                    if task_env.plan_success and task_env.check_success():
                        seeds.append(seed)
                    else:
                        rejected.append({"seed": seed, "reason": "not_expert_stable"})
                except Exception as exc:
                    rejected.append({"seed": seed, "reason": repr(exc)})
                    if len(rejected) == 1:
                        traceback.print_exc()
                finally:
                    task_env.close_env()
                if (offset + 1) % 10 == 0:
                    print(
                        f"[discovery] {task_name}: checked={offset + 1} "
                        f"stable={len(seeds)} rejected={len(rejected)}",
                        flush=True,
                    )
                if len(seeds) == target:
                    break
        finally:
            args["render_freq"] = render_freq
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "task": task_name,
            "task_config": args["task_config"],
            "seeds": seeds,
            "rejected": rejected,
            "candidates_checked": len(seeds) + len(rejected),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"[discovery] {task_name}: {len(seeds)}/{target} stable seeds -> {output}", flush=True)
        if len(seeds) != target:
            raise RuntimeError(f"only {len(seeds)} stable seeds found for {task_name}")
        return st_seed + len(seeds) + len(rejected), 0

    base.eval_function_decorator = decorator
    base.eval_policy = discover
    Sapien_TEST()
    base.main(base.parse_args_and_config())


if __name__ == "__main__":
    main()
