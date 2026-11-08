#!/usr/bin/env python3
"""Prepare Randomized token targets with GPU/task/episode parallelism, then train."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import socket
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "artifacts/robotwin_full50_study"
PREPARED = STUDY / "clean_randomized_32x64_v1"
TMP = Path("/tmp/robotwin_randomized_32x64_parallel_v2")
DATA_ROOT = Path("/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin")
ACTION_CKPT = STUDY / "action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/step-00020000.pt"
PHYSICS_CKPT = STUDY / "physics_tokenizers/ema_t64_k2048_steps20000/step-00020000.pt"
VISUAL_CKPT = Path("/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt")
CLEAN_CACHE = STUDY / "dual_ar_32action_64physics_targets.npz"
COMBINED = PREPARED / "clean_randomized_dual_targets.npz"
TASK_LIST = ROOT / "docs/robotwin_50task_list.txt"
RUN_ID = "qwenpi_v4_dualar_32x64_clean_randomized20k_m8_ga4_v1"
STATUS = STUDY / "node002_logs" / f"{RUN_ID}.status"
VISUAL_SHA256 = "5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d"
MERGER = ROOT / "scripts/physical_tokens/merge_robotwin_clean_randomized_dual_targets.py"


def manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def complete_pair(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and manifest_path(path).is_file()


def run(command: list[str], log: Path, *, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + " ".join(command[:4]) + " ...\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def episode_shards(task: str, shard_count: int) -> list[tuple[list[int], int]]:
    dataset = DATA_ROOT / "Randomized" / task
    path = dataset / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(episodes) != 500 or len({row["episode_index"] for row in episodes}) != 500:
        raise ValueError(f"{task}: expected 500 distinct Randomized episodes")
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))
    chunk_size = int(info["chunks_size"])
    video_keys = [
        (key, value.get("original_key") or key)
        for key, value in modality.get("video", {}).items()
    ]
    valid_episodes = []
    for row in episodes:
        episode_id = int(row["episode_index"])
        chunk_index = episode_id // chunk_size
        data_path = dataset / info["data_path"].format(
            episode_chunk=chunk_index, episode_index=episode_id
        )
        video_paths = [
            dataset / info["video_path"].format(
                episode_chunk=chunk_index,
                episode_index=episode_id,
                video_key=original_key,
            )
            for _, original_key in video_keys
        ]
        if data_path.is_file() and all(video_path.is_file() for video_path in video_paths):
            valid_episodes.append(row)
    if len(valid_episodes) != len(episodes):
        print(
            f"Randomized/{task}: using {len(valid_episodes)}/{len(episodes)} episodes "
            "with complete parquet/video files",
            flush=True,
        )
    if not valid_episodes:
        print(
            f"Randomized/{task}: no trainable episodes (required camera videos are missing)",
            flush=True,
        )
        return []
    bins: list[list[int]] = [[] for _ in range(shard_count)]
    sizes = [0] * shard_count
    for row in sorted(valid_episodes, key=lambda value: value["length"], reverse=True):
        index = min(range(shard_count), key=lambda shard: sizes[shard])
        bins[index].append(int(row["episode_index"]))
        sizes[index] += int(row["length"])
    return [(sorted(ids), sizes[index]) for index, ids in enumerate(bins)]


def ensure_actions(task: str, gpu: int, env: dict[str, str], log: Path) -> Path:
    path = PREPARED / "action_ids" / f"{task}.npz"
    # QuarkFS may reject concurrent flock() with EAGAIN even for a blocking
    # request. All preparers run on node002, so a node-local lock is sufficient.
    lock = TMP / "action_locks" / f"{task}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not complete_pair(path):
            raw = TMP / "actions" / f"{task}.npz"
            if not complete_pair(raw):
                run([sys.executable, "-u", "scripts/physical_tokens/export_robotwin_action_cache.py",
                     "--data-root", str(DATA_ROOT), "--data-name", f"Randomized/{task}",
                     "--robot-type", "robotwin_wrap32", "--output", str(raw), "--stride", "4"],
                    log, env=env)
            run([sys.executable, "scripts/physical_tokens/export_action_token_ids.py",
                 "--input", str(raw), "--checkpoint", str(ACTION_CKPT), "--output", str(path),
                 "--batch-size", "512", "--num-workers", "0", "--device", "cuda:0"], log, env=env)
            if not complete_pair(path):
                raise RuntimeError(f"action cache was not completed for {task}")
            raw.unlink(missing_ok=True)
            manifest_path(raw).unlink(missing_ok=True)
    return path


def prepare_shard(task: str, shard: int, shard_count: int, episode_ids: list[int],
                  gpu: int, base_env: dict[str, str]) -> None:
    env = {**base_env, "CUDA_VISIBLE_DEVICES": str(gpu)}
    label = f"{task}.shard{shard}"
    log = PREPARED / "logs/parallel" / f"{label}.log"
    feature = TMP / "features" / f"{label}.npz"
    physics = TMP / "physics" / f"{label}.npz"
    dual = PREPARED / "shards/dual" / f"{label}.npz"
    if complete_pair(dual):
        record = json.loads(manifest_path(dual).read_text(encoding="utf-8"))
        if (record.get("task") != f"Randomized/{task}" or record.get("shard_index") != shard
                or record.get("shard_count") != shard_count):
            raise ValueError(f"stale or mismatched shard cache {dual}")
        return
    if not complete_pair(feature):
        run([sys.executable, "-u", "scripts/physical_tokens/export_robotwin_spatiotemporal_features_sequential.py",
             "--data-root", str(DATA_ROOT), "--data-name", f"Randomized/{task}",
             "--robot-type", "robotwin_wrap32", "--camera", "video.cam_high",
             "--offsets", "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32",
             "--spatial-grid-size", "4", "--base-vlm",
             "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
             "--pretrained-checkpoint", str(VISUAL_CKPT),
             "--pretrained-checkpoint-sha256", VISUAL_SHA256,
             "--output", str(feature), "--include-episode-ids",
             ",".join(map(str, episode_ids)), "--stride", "4", "--batch-size", "16",
             "--device", "cuda:0"], log, env=env)
    feature_manifest = json.loads(manifest_path(feature).read_text(encoding="utf-8"))
    if (feature_manifest.get("data_name") != f"Randomized/{task}"
            or feature_manifest.get("included_episode_ids") != episode_ids):
        raise ValueError(f"stale or mismatched feature shard {feature}")
    if not complete_pair(physics):
        run([sys.executable, "scripts/physical_tokens/export_spatiotemporal_physics_token_ids.py",
             "--input", str(feature), "--checkpoint", str(PHYSICS_CKPT),
             "--output", str(physics), "--batch-size", "128", "--num-workers", "0",
             "--device", "cuda:0"], log, env=env)
    actions = ensure_actions(task, gpu, env, log)
    run([sys.executable, str(MERGER), "join-task", "--task", task,
         "--action", str(actions), "--physics", str(physics),
         "--feature", str(feature), "--output", str(dual),
         "--shard-index", str(shard), "--shard-count", str(shard_count)], log, env=env)
    if not complete_pair(dual):
        raise RuntimeError(f"dual shard was not completed: {dual}")
    for path in (feature, physics):
        path.unlink(missing_ok=True)
        manifest_path(path).unlink(missing_ok=True)


def gpu_free_mib(gpu: int) -> int:
    result = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], text=True,
    )
    return int(result.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-per-task", type=int, default=4)
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--min-free-mib", type=int, default=60000)
    parser.add_argument("--allow-current-node", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if "node002" not in socket.gethostname() and not args.allow_current_node:
        raise RuntimeError("run only on node002")
    if args.shards_per_task < 1 or args.slots_per_gpu < 1:
        raise ValueError("shards and slots must be positive")
    gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 or gpu > 7 for gpu in gpus):
        raise ValueError("--gpus must be a unique comma-separated list from 0 through 7")
    if (ROOT / "results/Checkpoints" / RUN_ID).exists():
        raise FileExistsError(f"refusing to reuse training output {RUN_ID}")
    for path in (ACTION_CKPT, PHYSICS_CKPT, VISUAL_CKPT, CLEAN_CACHE):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    tasks = [value.strip() for value in TASK_LIST.read_text().splitlines() if value.strip()]
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise ValueError("expected 50 unique task names")
    for gpu in gpus:
        if gpu_free_mib(gpu) < args.min_free_mib:
            raise RuntimeError(
                f"GPU {gpu} has less than {args.min_free_mib} MiB free for preparation"
            )
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{ROOT}:/mnt/jfzn/liujn/.local/lib/python3.10/site-packages:" + base_env.get("PYTHONPATH", "")
    base_env["PYTHONDONTWRITEBYTECODE"] = "1"
    base_env["WANDB_MODE"] = "disabled"
    base_env["PATH"] = str(Path(sys.executable).parent) + ":" + base_env.get("PATH", "")
    # Each visual exporter otherwise creates ~80 PyTorch/OpenMP threads.
    # Twelve such processes oversubscribe the 128 CPU cores and starve GPUs.
    prepare_env = {**base_env, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                   "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
                   "TOKENIZERS_PARALLELISM": "false"}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text("PREPARING parallel\n", encoding="utf-8")
    try:
        queue: Queue[tuple[str, int, list[int], int]] = Queue()
        jobs = []
        incomplete_tasks = []
        for task in tasks:
            full = PREPARED / "dual" / f"{task}.npz"
            if complete_pair(full):
                continue
            task_shards = episode_shards(task, args.shards_per_task)
            if not task_shards:
                incomplete_tasks.append(task)
                continue
            for shard, (episode_ids, size) in enumerate(task_shards):
                jobs.append((task, shard, episode_ids, size))
        # Longest remaining clips first; workers draw from one shared queue so
        # an unusually long task does not strand a fixed GPU partition.
        for job in sorted(jobs, key=lambda value: value[3], reverse=True):
            queue.put(job)
        stop = threading.Event()

        def worker(gpu: int) -> None:
            while not stop.is_set():
                try:
                    task, shard, episode_ids, size = queue.get_nowait()
                except Empty:
                    return
                try:
                    print(f"START GPU={gpu} task={task} shard={shard} frames={size}", flush=True)
                    prepare_shard(task, shard, args.shards_per_task, episode_ids, gpu, prepare_env)
                    print(f"READY GPU={gpu} task={task} shard={shard}", flush=True)
                except Exception:
                    stop.set()
                    raise
                finally:
                    queue.task_done()

        with ThreadPoolExecutor(max_workers=len(gpus) * args.slots_per_gpu) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus
                       for _ in range(args.slots_per_gpu)]
            for future in futures:
                future.result()
        if not queue.empty():
            raise RuntimeError(f"{queue.qsize()} preparation jobs were not completed")
        for task in tasks:
            if task in incomplete_tasks:
                continue
            full = PREPARED / "dual" / f"{task}.npz"
            if complete_pair(full):
                continue
            run([sys.executable, str(MERGER), "combine-shards", "--task", task,
                 "--shard-dir", str(PREPARED / "shards/dual"), "--shard-count",
                 str(args.shards_per_task), "--output", str(full)],
                PREPARED / "logs/parallel" / f"{task}.merge.log", env=base_env)
        if incomplete_tasks:
            detail = ",".join(incomplete_tasks)
            STATUS.write_text(f"PREPARED_INCOMPLETE parallel {detail}\n", encoding="utf-8")
            print(f"Prepared available tasks; no combined cache because tasks are incomplete: {detail}",
                  flush=True)
            return
        if not complete_pair(COMBINED):
            run([sys.executable, str(MERGER), "combine", "--clean", str(CLEAN_CACHE),
                 "--random-dir", str(PREPARED / "dual"), "--task-list", str(TASK_LIST),
                 "--output", str(COMBINED)],
                PREPARED / "logs/parallel/combine.log", env=base_env)
        if args.prepare_only:
            STATUS.write_text("PREPARED parallel\n", encoding="utf-8")
            return
        for gpu in (4, 5, 6, 7):
            if gpu_free_mib(gpu) < 60000:
                raise RuntimeError(f"GPU {gpu} has less than 60000 MiB free before training")
        STATUS.write_text("TRAINING parallel\n", encoding="utf-8")
        train_env = {**base_env, "CUDA_VISIBLE_DEVICES": "4,5,6,7"}
        run([str(Path(sys.executable).parent / "accelerate"), "launch", "--config_file",
             "starVLA/config/deepseeds/deepspeed_zero2.yaml", "--num_processes", "4",
             "--main_process_port", "29717", "starVLA/training/train_starvla.py",
             "--config_yaml", "scripts/physical_tokens/qwenpi_v4_dualar_32x64_concat_20k.yaml",
             "--run_id", RUN_ID, "--datasets.vla_data.data_mix",
             "robotwin_clean_randomized_wrap_32", "--datasets.vla_data.discrete_token_cache",
             str(COMBINED)], STUDY / "node002_logs" / f"{RUN_ID}.log", env=train_env)
        STATUS.write_text("COMPLETE parallel\n", encoding="utf-8")
    except BaseException:
        STATUS.write_text("FAILED parallel\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
