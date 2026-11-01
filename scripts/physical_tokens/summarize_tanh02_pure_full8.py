"""Summarize paired fixed-seed rollouts from the node2 eight-task panel."""

import argparse
import json
from pathlib import Path


TASKS = (
    "click_alarmclock", "beat_block_hammer", "open_microwave", "shake_bottle",
    "move_playingcard_away", "pick_dual_bottles", "handover_block", "place_object_basket",
)


def load_mode(run_dir: Path, task: str, mode: str) -> tuple[dict[int, bool], list[int]]:
    records = {}
    skipped = []
    for path_file in sorted((run_dir / task / mode).glob("*/result_path.txt")):
        result_dir = Path(path_file.read_text(encoding="utf-8").strip())
        manifest = json.loads((result_dir / "_fixed_seeds.json").read_text(encoding="utf-8"))
        for record in manifest["episode_records"]:
            seed = int(record["seed"])
            if seed in records:
                raise ValueError(f"duplicate result: {task} {mode} {seed}")
            records[seed] = bool(record["success"])
        skipped.extend(int(record["seed"]) for record in manifest["skipped_seeds"])
    return records, skipped


def summarize(run_dir: Path) -> dict:
    rows = []
    for task in TASKS:
        discovery_path = run_dir / task / "discovery.json"
        if not discovery_path.is_file():
            continue
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        planned = set(discovery["seeds"])
        pure, pure_skipped = load_mode(run_dir, task, "pure")
        tanh, tanh_skipped = load_mode(run_dir, task, "tanh")
        if not (set(pure) | set(pure_skipped)).issubset(planned) or not (set(tanh) | set(tanh_skipped)).issubset(planned):
            raise ValueError(f"unexpected seed in {task}")
        common = sorted(set(pure) & set(tanh))
        wins = sum(tanh[seed] and not pure[seed] for seed in common)
        losses = sum(pure[seed] and not tanh[seed] for seed in common)
        rows.append({
            "task": task,
            "planned": len(planned),
            "pure_evaluated": len(pure),
            "tanh_evaluated": len(tanh),
            "pure_precheck_skipped": len(pure_skipped),
            "tanh_precheck_skipped": len(tanh_skipped),
            "paired_episodes": len(common),
            "pure_successes_on_pair": sum(pure[seed] for seed in common),
            "tanh_successes_on_pair": sum(tanh[seed] for seed in common),
            "tanh_only_wins": wins,
            "pure_only_wins": losses,
            "ties": len(common) - wins - losses,
            "paired_seeds": common,
        })
    return {
        "run_dir": str(run_dir),
        "completed_tasks": sum(row["planned"] == row["paired_episodes"] for row in rows),
        "tasks": rows,
        "total_paired_episodes": sum(row["paired_episodes"] for row in rows),
        "total_pure_successes_on_pair": sum(row["pure_successes_on_pair"] for row in rows),
        "total_tanh_successes_on_pair": sum(row["tanh_successes_on_pair"] for row in rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    destination = args.run_dir / "summary.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}, indent=2))
