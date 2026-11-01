"""Track supplemental paired RoboTwin episodes without changing the original run."""

import argparse
import json
import time
from pathlib import Path


ROOT = Path("/mnt/jfzn/liujn/code/VLAct-PhysToken")
SIM = Path("/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013")
ORIGINAL = ROOT / "logs/tanh02_pure_full8_node2_20261029"
SUPPLEMENT = ROOT / "logs/tanh02_pure_full8_supplement_20261030"
TASKS = (
    "click_alarmclock", "beat_block_hammer", "open_microwave", "shake_bottle",
    "move_playingcard_away", "pick_dual_bottles", "handover_block", "place_object_basket",
)


def read_records(run_dir: Path, task: str, mode: str) -> dict[int, dict]:
    records = {}
    for pointer in sorted((run_dir / task / mode).glob("*/result_path.txt")):
        result_dir = Path(pointer.read_text().strip())
        if not result_dir.is_absolute():
            result_dir = SIM / result_dir
        manifest = json.loads((result_dir / "_fixed_seeds.json").read_text())
        for record in manifest["episode_records"]:
            seed = int(record["seed"])
            if seed in records:
                raise ValueError(f"duplicate executed seed: {task}/{mode}/{seed}")
            records[seed] = {"success": bool(record["success"]), "video": record["video"],
                             "result_dir": str(result_dir)}
    return records


def paired(task: str) -> tuple[dict, dict, list[int]]:
    original = {mode: read_records(ORIGINAL, task, mode) for mode in ("pure", "tanh")}
    extra = {mode: read_records(SUPPLEMENT, task, mode) for mode in ("pure", "tanh")}
    for mode in ("pure", "tanh"):
        overlap = set(original[mode]) & set(extra[mode])
        if overlap:
            raise ValueError(f"supplement duplicated original {task}/{mode}: {overlap}")
    ordered = sorted(set(original["pure"]) & set(original["tanh"]))
    ordered += sorted(set(extra["pure"]) & set(extra["tanh"]))
    return original, extra, ordered


def status(task: str) -> int:
    _, extra, ordered = paired(task)
    print(json.dumps({"task": task, "paired": len(ordered), "needed": max(0, 100 - len(ordered)),
                      "supplement_pure": len(extra["pure"]), "supplement_tanh": len(extra["tanh"])}))
    return max(0, 100 - len(ordered))


def finalize() -> None:
    result = {"description": "100 actually executed paired seeds per task, original plus supplement",
              "task_config": "demo_clean", "complete": True, "tasks": {}}
    rows = []
    for task in TASKS:
        original, extra, ordered = paired(task)
        if len(ordered) < 100:
            result["complete"] = False
        selected = ordered[:100]
        merged = {mode: original[mode] | extra[mode] for mode in ("pure", "tanh")}
        pure_success = sum(merged["pure"][seed]["success"] for seed in selected)
        tanh_success = sum(merged["tanh"][seed]["success"] for seed in selected)
        tanh_only = sum(not merged["pure"][seed]["success"] and merged["tanh"][seed]["success"] for seed in selected)
        pure_only = sum(merged["pure"][seed]["success"] and not merged["tanh"][seed]["success"] for seed in selected)
        result["tasks"][task] = {"seeds": selected, "paired": len(selected),
                                 "pure_success": pure_success, "tanh_success": tanh_success,
                                 "tanh_only": tanh_only, "pure_only": pure_only}
        rows.append((task, len(selected), pure_success, tanh_success, tanh_only, pure_only))
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    (SUPPLEMENT / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["complete"]:
        docs = ROOT / "docs"
        (docs / "robotwin_tanh02_pure_full8_executed_800_seeds.json").write_text(json.dumps(result, indent=2) + "\n")
        lines = ["# Pure PI / tanh 八任务共同执行 800 seed", "",
                 "本清单为原始测评加补测后两模型均实际执行的 seed；每任务 100 个。原始预筛清单保留不变。",
                 "", "| 任务 | 共同执行 | Pure PI 成功 | tanh 成功 | tanh 独赢 | Pure PI 独赢 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: |"]
        lines += [f"| `{t}` | {n} | {p} | {h} | {w} | {l} |" for t, n, p, h, w, l in rows]
        totals = [sum(row[i] for row in rows) for i in range(1, 6)]
        lines += [f"| **合计** | **{totals[0]}** | **{totals[1]}** | **{totals[2]}** | **{totals[3]}** | **{totals[4]}** |",
                  "", "机器清单：`docs/robotwin_tanh02_pure_full8_executed_800_seeds.json`。",
                  "补测原始记录：`logs/tanh02_pure_full8_supplement_20261030/`。", ""]
        (docs / "ROBOTWIN_TANH02_PURE_FULL8_EXECUTED_800.md").write_text("\n".join(lines))
    print(json.dumps({"complete": result["complete"], "rows": rows}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("status", "finalize", "monitor"))
    parser.add_argument("task", nargs="?", choices=TASKS)
    args = parser.parse_args()
    if args.command == "status":
        if not args.task:
            parser.error("status needs task")
        status(args.task)
    elif args.command == "finalize":
        finalize()
    else:
        while True:
            finalize()
            progress = json.loads((SUPPLEMENT / "progress.json").read_text())
            if progress["complete"]:
                break
            time.sleep(60)
