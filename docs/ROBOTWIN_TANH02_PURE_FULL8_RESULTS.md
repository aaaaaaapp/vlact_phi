# 训练 tanh 与 Pure PI：RoboTwin 八任务固定 seed 闭环测评

状态：2026-10-30 07:39（北京时间）完成，80/80 个 20-seed 模型分片全部落盘。这里的 tanh 是训练时启用 `pi_gate_bound_mode=tanh, pi_gate_alpha_max=0.2` 的 20k 权重，不是给其他权重仅在推理时加 tanh。

补测状态（2026-10-30）：已完成。原始 734 个配对 episode 保留为历史结果；补测后每任务双方均实际执行 100 个，严格共同分母为 800。最终共同执行 seed 见 `docs/robotwin_tanh02_pure_full8_executed_800_seeds.json`，任务表见 `docs/ROBOTWIN_TANH02_PURE_FULL8_EXECUTED_800.md`。

## 权重与协议

- Pure PI：`/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`
- 训练 tanh：`/mnt/jfzn/liujn/code/VLAct-PhysToken/results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt`
- 恢复版旧 RoboTwin、`demo_clean`、RT 渲染、state 输入、wrap 动作解码。八任务各预筛 100 个专家有效 scene seed，两模型各在相同 seed 上测一次。评测时再次被专家预检跳过的 episode 不计为模型失败；表中只统计两模型都实际执行的 seed。

## 配对结果

| 任务 | 共同执行 | Pure PI 成功 | tanh 成功 | tanh 独赢 | Pure PI 独赢 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `click_alarmclock` | 99 | 94 | 71 | 5 | 28 |
| `beat_block_hammer` | 84 | 76 | 64 | 6 | 18 |
| `open_microwave` | 82 | 82 | 70 | 0 | 12 |
| `shake_bottle` | 99 | 99 | 99 | 0 | 0 |
| `move_playingcard_away` | 100 | 100 | 77 | 0 | 23 |
| `pick_dual_bottles` | 95 | 86 | 58 | 5 | 33 |
| `handover_block` | 98 | 94 | 79 | 2 | 17 |
| `place_object_basket` | 77 | 71 | 61 | 5 | 15 |
| **合计** | **734** | **702（95.64%）** | **579（78.88%）** | **23** | **146** |

按 Pure PI 的结果分层，tanh 的保持与救回情况为：

| Pure PI 结果 | tanh 成功 | tanh 失败 | 行分母 |
| --- | ---: | ---: | ---: |
| 成功 | 556（79.20%） | 146（20.80%） | 702 |
| 失败 | 23（71.88%） | 9（28.12%） | 32 |

tanh 净少 123 个成功 episode。它救回 Pure PI 失败场景的能力是否稳定，仍需在同一 scene seed 上重复 rollout，并控制扩散初始噪声；本轮只固定了场景 seed，每个模型各运行一次。因此不能把 23 次独赢全部归因于物理 token 的稳定收益。不过 146 次回归远多于 23 次救回，整体退化不能只用少量 episode 的随机波动解释。

## 补测后严格 800 配对结果

每任务补至双方均实际执行的 100 个 seed 后，Pure PI 为 **759/800（94.88%）**，训练 tanh 为 **631/800（78.88%）**；tanh 独赢 27 个、Pure PI 独赢 155 个，tanh 净少 128 个成功 episode。逐任务 800 清单与结果表位于 `docs/ROBOTWIN_TANH02_PURE_FULL8_EXECUTED_800.md`。

| Pure PI 结果 | tanh 成功 | tanh 失败 | 行分母 |
| --- | ---: | ---: | ---: |
| 成功 | 604（79.58%） | 155（20.42%） | 759 |
| 失败 | 27（65.85%） | 14（34.15%） | 41 |

这仍是每模型每 scene seed 一次 rollout，故 27 个救回不能直接量化为确定的物理 token 收益；但 155 个回归显著更多，且与原始 734 配对结论一致。

## Seed 与结果路径

- 人类可读、包含八任务各 100 个 seed 的清单：`/mnt/jfzn/liujn/code/VLAct-PhysToken/docs/ROBOTWIN_TANH02_PURE_FULL8_SEEDS.md`
- 可直接复用的机器清单：`/mnt/jfzn/liujn/code/VLAct-PhysToken/docs/robotwin_tanh02_pure_full8_seeds.json`；后续运行可设置 `SEED_MANIFEST` 为此绝对路径，跳过重新筛选。复用时仍需对偶发不稳定场景做专家预检。
- 严格共同执行 800 episode 的最终机器清单：`/mnt/jfzn/liujn/code/VLAct-PhysToken/docs/robotwin_tanh02_pure_full8_executed_800_seeds.json`；后续两模型 matched 重跑应以此为准，不需重新筛选。
- 严格共同执行 800 episode 的人类可读任务表：`/mnt/jfzn/liujn/code/VLAct-PhysToken/docs/ROBOTWIN_TANH02_PURE_FULL8_EXECUTED_800.md`。
- 本轮汇总（各任务分母、预检跳过数、配对 seed 与独赢数）：`/mnt/jfzn/liujn/code/VLAct-PhysToken/logs/tanh02_pure_full8_node2_20261029/summary.json`
- 原始预筛结果及拒绝原因：同一运行目录下的 `<task>/discovery.json`。
- 每批结果路径：同一运行目录下的 `<task>/<pure|tanh>/<start>/result_path.txt`，其中指向 `_fixed_seeds.json` 所在目录。该 manifest 的 `episode_records` 逐 seed 记录成功/失败及视频文件名；`skipped_seeds` 单列预检跳过。视频文件名以 `_success.mp4` 或 `_failure.mp4` 结尾。

本轮曾有一个分片因 WebSocket 心跳超时中断，重试后完成；通信故障没有计入模型失败。此前检查未发现 checkpoint、state 输入或动作反归一化配置不一致，但现有结果尚不能把退化定位为某个确定的代码错误。
