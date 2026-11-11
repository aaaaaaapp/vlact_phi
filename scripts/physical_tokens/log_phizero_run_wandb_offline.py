#!/usr/bin/env python3
"""Mirror an existing tqdm training log into an offline W&B run."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import wandb


PROGRESS_RE = re.compile(
    r"PhiZero fsq training:\s*\d+%\|.*?\|\s*(\d+)/\d+.*?"
    r"loss=([-+0-9.eE]+),\s*lr=([-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-pid", type=int, default=0)
    parser.add_argument("--entity", default="jianingliu")
    parser.add_argument("--project", default="phizero-robotwin-quantizer-comparison")
    parser.add_argument("--name", default="phizero-fsq64-gbs16-gradckpt")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--idle-timeout", type=float, default=900.0)
    return parser.parse_args()


def train_process_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    command_line = Path(f"/proc/{pid}/cmdline")
    try:
        return b"accelerate.commands.launch" in command_line.read_bytes()
    except FileNotFoundError:
        return False


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "training_args.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.name,
        dir=str(args.run_dir),
        mode="offline",
        config=config,
        tags=["phizero", "fsq64", "robotwin", "offline-log-mirror"],
        settings=wandb.Settings(_disable_stats=True),
    )
    offset = 0
    partial = b""
    last_logged_step = 0
    logged_count = 0
    pending: tuple[int, float, float] | None = None
    last_change = time.monotonic()
    last_size = -1
    train_finished_at: float | None = None

    def accept_line(raw_line: bytes) -> None:
        nonlocal pending, last_logged_step, logged_count
        line = raw_line.replace(b"\x00", b"").decode("utf-8", errors="ignore")
        match = PROGRESS_RE.search(line)
        if not match:
            return
        step = int(match.group(1))
        try:
            loss = float(match.group(2))
            learning_rate = float(match.group(3))
        except ValueError:
            return
        if step <= last_logged_step:
            return
        if pending is not None:
            pending_step, pending_loss, pending_lr = pending
            if step > pending_step:
                run.log(
                    {
                        "optimizer_step": pending_step,
                        "train/loss": pending_loss,
                        "train/lr": pending_lr,
                    },
                    step=pending_step,
                )
                last_logged_step = pending_step
                logged_count += 1
                if logged_count % 100 == 0:
                    print(f"offline W&B logged {logged_count} steps (latest={last_logged_step})", flush=True)
                pending = None
        if pending is None or step >= pending[0]:
            # tqdm writes the step number and postfix as separate carriage-return
            # updates; keep the latest metric for a step before logging it.
            pending = (step, loss, learning_rate)

    print(f"offline W&B run: {run.id}; backfilling {args.log}", flush=True)
    while True:
        try:
            current_size = args.log.stat().st_size
        except FileNotFoundError:
            current_size = 0

        if current_size < offset:
            offset = 0
            partial = b""
        if current_size != last_size:
            last_change = time.monotonic()
            last_size = current_size

        if current_size > offset:
            with args.log.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
            parts = (partial + chunk).splitlines(keepends=True)
            partial = b""
            for part in parts:
                if part.endswith((b"\r", b"\n")):
                    accept_line(part.rstrip(b"\r\n"))
                else:
                    partial = part

        alive = train_process_alive(args.train_pid)
        now = time.monotonic()
        if not alive:
            if train_finished_at is None:
                train_finished_at = now
            if now - train_finished_at >= 10:
                if partial:
                    accept_line(partial)
                break
        elif now - last_change >= args.idle_timeout:
            if partial:
                accept_line(partial)
            print("training log idle timeout reached; closing offline run", flush=True)
            break

        time.sleep(max(args.poll_seconds, 0.2))

    if pending is not None and pending[0] > last_logged_step:
        step, loss, learning_rate = pending
        run.log(
            {
                "optimizer_step": step,
                "train/loss": loss,
                "train/lr": learning_rate,
            },
            step=step,
        )
        last_logged_step = step
        logged_count += 1
    run.summary["train/last_logged_step"] = last_logged_step
    run.summary["metric_records"] = logged_count
    run.finish()
    print(f"offline W&B run closed at optimizer step {last_logged_step}", flush=True)


if __name__ == "__main__":
    main()
