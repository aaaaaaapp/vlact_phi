"""Import complete PhiZero console metric histories into a separate W&B project."""

import argparse
import json
import math
import re
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "artifacts/robotwin_full50_study/phizero_quantizer_comparison/runs_20261030"
IMPORT_ROOT = ROOT / "artifacts/robotwin_full50_study/phizero_quantizer_comparison/wandb_log_import"
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
        step = int(step)
        target = int(target)
        loss = float(loss)
        lr = float(lr)
        if not math.isfinite(loss) or not math.isfinite(lr):
            raise ValueError(f"Non-finite metric at step {step} in {log_path}")
        if target_steps is not None and target != target_steps:
            raise ValueError(f"Inconsistent target steps in {log_path}")
        target_steps = target
        history[step] = (loss, lr)  # tqdm prints twice per step; the final line is current.
    if not history or set(history) != set(range(1, max(history) + 1)):
        raise ValueError(f"Missing or non-contiguous metric steps in {log_path}")
    return history, target_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--entity", default="jianingliu")
    parser.add_argument("--project", default="phizero-robotwin-quantizer-comparison")
    parser.add_argument("--output-dir", type=Path, default=IMPORT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dirname, run_name, run_id, stop_reason in RUNS:
        run_dir = RUNS_ROOT / dirname
        history, target_steps = parse_history(run_dir / "training.log")
        training_args = json.loads((run_dir / "training_args.json").read_text())
        config = {
            **training_args,
            "metric_source": "reconstructed from training.log tqdm output",
            "source_run_dir": str(run_dir),
            "source_stop_reason": stop_reason,
            "source_last_step": max(history),
            "source_target_steps": target_steps,
        }
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            id=run_id,
            name=run_name,
            group="phizero-robotwin-bs8",
            job_type="historical-log-import",
            notes="Imported from console logs; these training jobs did not record native W&B runs.",
            config=config,
            mode=args.mode,
            dir=str(args.output_dir),
            resume="allow" if args.mode == "online" else None,
            settings=wandb.Settings(console="off"),
        )
        for step, (loss, lr) in sorted(history.items()):
            run.log({"optimizer_step": step, "train/loss": loss, "train/lr": lr}, step=step)
        run.summary["metric_records"] = len(history)
        run.summary["last_optimizer_step"] = max(history)
        run.finish()
        print(f"{run_name}: {len(history)} steps, last={max(history)}, id={run_id}, url={run.url}")


if __name__ == "__main__":
    main()
