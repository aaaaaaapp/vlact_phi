# 蒸馏 20k 权重的推理 gate 消融（2026-10-14）

## 结论

同一份 P64/M32 蒸馏 20k 权重，在恢复的旧版 RoboTwin、RT 渲染、**有 state 输入＋wrap 解码**的协议下，直接在推理时把无界 residual gate 改为 `0.2 × tanh(gate)`，没有改善本轮 rollout。四任务相同的 35 个有效 seed 上，无界版 **29/35**，推理 tanh 版 **21/35**；逐 seed 有 11 个仅无界版成功、3 个仅 tanh 版成功、21 个同结果。困难集共同有效的 7 个 seed 上是 **4/7 对 2/7**。

这是对**无界 gate 训练所得同一 checkpoint 的推理消融**，不是 tanh gate 重新训练实验。两轮运行在不同节点且每个 seed 只跑一次，策略初始噪声及仿真非确定性未消除，因此不把单个 seed 翻转解释为严格因果；但本轮没有证据支持“推理端补 tanh 可以修复退化”。此前在另一评测链路得到的 handover 0/10 也不能与本轮恢复旧版＋RT 的 7/9 混作同一协议结果。

## 固定协议

- checkpoint：`results/Checkpoints/qwenpi_v4_p64_m32_distill_medium_seed42_steps20000/checkpoints/steps_20000_pytorch_model.pt`，训练时未启用 gate 边界；checkpoint 的 `config.yaml` 没有 `pi_gate_bound_mode`，模型默认 `none`。
- 仿真器：`/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013/`，`SAPIEN_DISABLE_RAYTRACING=0`。该目录由缓存恢复，不能保证与历史 checkout 字节完全相同。
- 输入和动作：`ROBOTWIN_SEND_STATE=1`、`ROBOTWIN_ACTION_UNNORM_MODE=wrap`。评测日志确认 `send_state_to_server: True`；服务端元信息确认 `action_unnorm_mode: wrap`。
- node003 八卡：原始无界 gate。node004 八卡：同一权重，服务端仅设置 `PI_GATE_BOUND_MODE=tanh`、`PI_GATE_ALPHA_MAX=0.2`，日志确认 `mode=tanh alpha_max=0.2`。
- 困难集用 `demo_randomized`，四任务用 `demo_clean`。每个四任务 seed 使用独立端口和 tag。两个最终四任务队列均 `ALL_DONE`，`progress.log` 无 `ERROR`。
- 有效性只由 `_fixed_seeds.json` 中 `episodes=1`、`skipped_seeds=[]` 的真实 rollout 判定。专家预检跳过不算模型失败。

## 结果

| 数据集 / 任务 | node003 无界 gate | node004 推理 tanh 0.2 | 共同有效 seed |
| --- | ---: | ---: | ---: |
| 困难 8 episode | 4/8 | 2/7 | 无界 4/7；tanh 2/7 |
| `handover_block` | 7/9 | 2/9 | 9 |
| `beat_block_hammer` | 8/9 | 6/9 | 9 |
| `click_bell` | 9/10 | 7/10 | 10 |
| `place_phone_stand` | 5/7 | 6/7 | 7 |
| 四任务合计 | **29/35** | **21/35** | **35** |

困难集中，node004 的 `hanging_mug/100002` 被专家预检跳过；共同有效 7 个 seed 的两个无界独有成功是 `click_alarmclock/100017` 和 `100022`。四任务两轮预检跳过的 5 个 seed 相同：`handover_block/100100005`、`beat_block_hammer/100150006`、`place_phone_stand/100100007`、`100100010`、`100100013`。四任务 40 个计划 seed 中因此只有 35 个配对有效 rollout。

四任务仅无界版成功的 seed：handover `100100006/009/015/023/027`；beat `100150005/008/011`；bell `100100006/007/009`。仅 tanh 版成功：beat `100150010`、bell `100100003`、phone `100100003`。不要把 40 个计划 seed 直接写成 29/40 或 21/40 成功率。

## 可复核位置

- node003 无界困难集：`logs/distill_medium_state_wrap_recovered_rt_node3_20261013/`。
- node004 tanh 困难集：`logs/distill_medium_state_wrap_tanh02_recovered_rt_node4_20261013/`。
- node003 无界四任务：`logs/distill_medium_state_wrap_recovered_rt_node3_four_uniqueports_20261013/`。
- node004 tanh 四任务：`logs/distill_medium_state_wrap_tanh02_recovered_rt_node4_four_uniqueports_20261013/`。
- 结果清单：恢复版 RoboTwin 的 `eval_result/<task>/model2robotwin_interface/<demo_clean|demo_randomized>/<tag>/<timestamp>/_fixed_seeds.json`。
- 四任务队列脚本：`scripts/physical_tokens/run_distill_medium_four_unique_ports.sh`；单 seed runner：`scripts/physical_tokens/run_policy_single_seed.sh`。

这轮结果仅说明“对已经用无界 gate 训练的权重，推理端硬加 tanh”不合适。若要判断受限 gate 的训练策略是否更好，需要从训练开始启用该参数化、在同一评测协议及更多重复 rollout 下另行比较。
