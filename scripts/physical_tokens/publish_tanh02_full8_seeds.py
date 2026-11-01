"""Publish one reusable seed manifest and readable document for all eight tasks."""

import argparse
import json
import os
from pathlib import Path


TASKS = (
    "click_alarmclock", "beat_block_hammer", "open_microwave", "shake_bottle",
    "move_playingcard_away", "pick_dual_bottles", "handover_block", "place_object_basket",
)


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def publish(run_dir: Path, docs_dir: Path, count: int) -> int:
    entries = {}
    for task in TASKS:
        source = run_dir / task / "discovery.json"
        if not source.is_file():
            continue
        entry = json.loads(source.read_text(encoding="utf-8"))
        seeds = entry["seeds"]
        if entry["task"] != task or entry["task_config"] != "demo_clean":
            raise ValueError(f"wrong task/config in {source}")
        if len(seeds) != count or len(set(seeds)) != count or any(type(seed) is not int for seed in seeds):
            raise ValueError(f"invalid seed count or values in {source}")
        entries[task] = {
            "seeds": seeds,
            "candidates_checked": entry["candidates_checked"],
            "rejected_count": len(entry["rejected"]),
        }

    complete = len(entries) == len(TASKS)
    payload = {
        "description": "Expert-precheck-stable RoboTwin scene seeds for matched Pure PI/tanh02 rollout",
        "task_config": "demo_clean",
        "simulator": "/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013",
        "seed_start": 100000,
        "target_seeds_per_task": count,
        "complete": complete,
        "task_order": list(TASKS),
        "tasks": entries,
    }
    lines = [
        "# RoboTwin 八任务固定 scene seeds（Pure PI / 训练 tanh）",
        "",
        f"状态：{'八任务已齐' if complete else f'已完成 {len(entries)}/8 任务，筛选仍在运行'}。",
        "",
        "这些 seed 在 node002 的恢复版旧 RoboTwin、`demo_clean` 配置下，"
        "从 `100000` 起顺序检查并通过一次专家预检。每任务按下面的顺序取 100 个，"
        "Pure PI 与训练 tanh 使用同一清单。再次运行时仍应执行专家预检；"
        "若场景偶发不稳定，应记为 `skipped_seeds`，不可计为模型失败。",
        "",
        "机器可读取的同一清单：[robotwin_tanh02_pure_full8_seeds.json](robotwin_tanh02_pure_full8_seeds.json)。",
        "八任务齐备后，后续测评给 `run_tanh02_pure_full8_node2.sh` 设置 "
        "`SEED_MANIFEST=/mnt/jfzn/liujn/code/VLAct-PhysToken/docs/robotwin_tanh02_pure_full8_seeds.json` "
        "即可直接复用，不再执行场景筛选。",
        "原始筛选及拒绝原因保存在"
        "[本次运行目录](../logs/tanh02_pure_full8_node2_20261029/)。",
        "每批闭环结果的 `_fixed_seeds.json` 中，`episode_records` 逐 seed 记录 `success` "
        "和视频文件名；视频以 `_success.mp4` 或 `_failure.mp4` 结尾。"
        "未执行策略的场景单列于 `skipped_seeds`。",
        "",
    ]
    for task in TASKS:
        lines.extend((f"## {task}", ""))
        if task not in entries:
            lines.extend(("筛选中。", ""))
            continue
        item = entries[task]
        lines.append(
            f"100 个 seed；检查 {item['candidates_checked']} 个候选，"
            f"拒绝 {item['rejected_count']} 个。"
        )
        lines.extend(("", "```json", json.dumps(item["seeds"], ensure_ascii=False), "```", ""))

    docs_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(docs_dir / "robotwin_tanh02_pure_full8_seeds.json", json.dumps(payload, indent=2) + "\n")
    write_atomic(docs_dir / "ROBOTWIN_TANH02_PURE_FULL8_SEEDS.md", "\n".join(lines))
    print(f"published {len(entries)}/8 task seed lists; complete={complete}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("docs_dir", type=Path)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    raise SystemExit(publish(args.run_dir, args.docs_dir, args.count))
