"""Stream the Clean+Randomized QwenPi run's JSONL metrics to SwanLab."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import time
from pathlib import Path

import swanlab


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "qwenpi_v4_dualar_32x64_clean_randomized20k_m8_ga4_v1"
RUN_DIR = ROOT / "results/Checkpoints" / RUN_ID
METRICS = RUN_DIR / "train_metrics.jsonl"
STATUS = ROOT / "artifacts/robotwin_full50_study/node002_logs" / f"{RUN_ID}.status"
LOG_DIR = ROOT / "artifacts/robotwin_full50_study/swanlab_qwenpi_clean_randomized_20k"


def check_training_status(status: str, last_step: int) -> bool:
    """Return whether idle timeout applies; raise on a terminal early exit."""
    phase = status.split(maxsplit=1)[0] if status else ""
    if phase in {"FAILED", "EXIT"}:
        raise RuntimeError(f"Training stopped before step 20000: {status}; synced={last_step}")
    if phase == "COMPLETE" and last_step < 20000:
        raise RuntimeError(f"Training completed without step 20000 metrics: {status}; synced={last_step}")
    return phase == "TRAINING"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--idle-timeout-seconds", type=int, default=1800)
    parser.add_argument("--connectivity-retry-seconds", type=int, default=60)
    args = parser.parse_args()

    api_key = os.environ.get("SWANLAB_API_KEY") or getpass.getpass("SwanLab API key: ")
    while True:
        try:
            with socket.create_connection(("api.swanlab.cn", 443), timeout=5):
                break
        except OSError as error:
            print(f"SWANLAB_WAITING_FOR_NETWORK={error}", flush=True)
            time.sleep(args.connectivity_retry_seconds)
    if not swanlab.login(api_key=api_key, save=False, timeout=15):
        raise RuntimeError("SwanLab authentication failed")
    del api_key

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run = swanlab.init(
        workspace="jianingliu",
        project="vlact-physical-tokens",
        name="qwenpi-dualar32x64-clean-randomized-20k-m8ga4",
        id="qpi20kcr01",
        resume="allow",
        group="qwenpi-dualar32x64",
        job_type="train",
        description="Node002 GPU 4-7; Clean+Randomized RoboTwin, fresh dual-AR 40k LoRA + mature Pi; 20k joint action-expert training; global batch 128.",
        config={
            "source_run_id": RUN_ID,
            "max_train_steps": 20000,
            "global_batch_size": 128,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": False,
            "data_mix": "robotwin_clean_randomized_wrap_32",
            "metric_source": str(METRICS),
        },
        mode="online",
        log_dir=str(LOG_DIR),
    )
    print(f"SWANLAB_URL={run.url}", flush=True)

    last_step = 0
    offset = 0
    last_update = time.monotonic()
    was_training = False
    try:
        while last_step < 20000:
            if METRICS.exists():
                with METRICS.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    while True:
                        start = handle.tell()
                        line = handle.readline()
                        if not line:
                            break
                        if not line.endswith("\n"):
                            handle.seek(start)
                            break
                        record = json.loads(line)
                        step = int(record["step"])
                        if step > last_step:
                            payload = {f"train/{key}": value for key, value in record["metrics"].items()}
                            payload["optimizer_step"] = step
                            run.log(payload, step=step)
                            last_step = step
                            last_update = time.monotonic()
                            print(f"SWANLAB_SYNCED_STEP={step}", flush=True)
                    offset = handle.tell()
            if last_step >= 20000:
                break
            status = STATUS.read_text(encoding="utf-8").strip() if STATUS.exists() else ""
            is_training = check_training_status(status, last_step)
            if is_training and not was_training:
                last_update = time.monotonic()
            was_training = is_training
            if is_training and time.monotonic() - last_update > args.idle_timeout_seconds:
                raise TimeoutError(f"No metrics for {args.idle_timeout_seconds}s; synced={last_step}")
            time.sleep(args.poll_seconds)
    finally:
        run.finish()
    print(f"SWANLAB_COMPLETE step={last_step} url={run.url}", flush=True)


if __name__ == "__main__":
    main()
