"""Upload complete PhiZero console metric histories to SwanLab."""

import argparse
import getpass
import json
import math
import os
import re
from pathlib import Path

import swanlab


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "artifacts/robotwin_full50_study/phizero_quantizer_comparison/runs_20261030"
OUTPUT_ROOT = ROOT / "artifacts/robotwin_full50_study/phizero_quantizer_comparison/swanlab_log_import"
PROGRESS = re.compile(
    r"^Training:.*?\b(\d+)/(\d+)\b.*?loss=([0-9.eE+-]+), lr=([0-9.eE+-]+)"
)
RUNS = (
    ("fsq_levels8_5x5_k25000_seed42_bs8", "fsq_bs8", "phifsq08", "NCCL timeout"),
    ("ema_vq_k25000_seed42_bs8", "ema_vq_bs8", "phiema08", "SIGTERM"),
)


def parse_history(log_path: Path) -> tuple[dict[int, tuple[float, float]], int]:
    history = {}
    target_steps = None
    for segment in re.split(r"[\r\n]+", log_path.read_text(errors="replace")):
        match = PROGRESS.search(segment)
        if match is None:
            continue
        step, target, loss, lr = match.groups()
        step, target = int(step), int(target)
        loss, lr = float(loss), float(lr)
        if not math.isfinite(loss) or not math.isfinite(lr):
            raise ValueError(f"Non-finite metric at step {step} in {log_path}")
        if target_steps is not None and target != target_steps:
            raise ValueError(f"Inconsistent target steps in {log_path}")
        target_steps = target
        history[step] = (loss, lr)  # tqdm can print a step more than once.
    if not history or set(history) != set(range(1, max(history) + 1)):
        raise ValueError(f"Missing or non-contiguous metric steps in {log_path}")
    return history, target_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="jianingliu")
    parser.add_argument("--project", default="phizero-robotwin-quantizer-comparison")
    parser.add_argument("--only", choices=("fsq_bs8", "ema_vq_bs8"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = [run for run in RUNS if args.only is None or run[1] == args.only]
    prepared = []
    for dirname, name, run_id, stop_reason in selected:
        run_dir = RUNS_ROOT / dirname
        history, target_steps = parse_history(run_dir / "training.log")
        config = {
            **json.loads((run_dir / "training_args.json").read_text()),
            "metric_source": "reconstructed from training.log tqdm output",
            "source_run_dir": str(run_dir),
            "source_stop_reason": stop_reason,
            "source_last_step": max(history),
            "source_target_steps": target_steps,
        }
        prepared.append((name, run_id, history, config))
        print(f"Validated {name}: {len(history)} contiguous steps, last={max(history)}", flush=True)

    if args.dry_run:
        return

    api_key = os.environ.get("SWANLAB_API_KEY") or getpass.getpass("SwanLab API key: ")
    if not swanlab.login(api_key=api_key, save=False):
        raise RuntimeError("SwanLab authentication failed")
    del api_key

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, run_id, history, config in prepared:
        run = swanlab.init(
            workspace=args.workspace,
            project=args.project,
            name=name,
            id=run_id,
            resume="allow",
            group="phizero-robotwin-bs8",
            job_type="historical-log-import",
            description="Imported from console logs; training did not record native SwanLab runs.",
            config=config,
            mode="online",
            log_dir=str(OUTPUT_ROOT),
        )
        for step, (loss, lr) in sorted(history.items()):
            run.log({"optimizer_step": step, "train/loss": loss, "train/lr": lr}, step=step)
            if step % 1000 == 0 or step == len(history):
                print(f"{name}: queued {step}/{len(history)}", flush=True)
        run.finish()
        print(f"Finished {name}: {len(history)} steps, url={run.url}", flush=True)


if __name__ == "__main__":
    main()
