# Pure PI RoboTwin 旧评测链路恢复记录（2026-10-13）

## 结论与状态

历史 `handover_block / demo_clean` 的 Pure PI 基线是 8/8 个**有效** episode；其中
`100100009`、`100100023`、`100100028` 三个固定 scene seed 均成功。历史评测与本次
复测使用同一路径的 100k checkpoint：
`/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`。
路径一致不等于已经核实两个日期的文件内容哈希相同。

本次只在 node003 跑 GPU 评测。重建的旧版环境采用 RT 渲染后，历史 8 个有效
scene seed 全部完成策略 rollout 并成功，**8/8**；8 份独立 `_fixed_seeds.json` 均为
`successes=1, episodes=1`，并有 8 个非空成功视频。这是在历史相同 seed、任务和
checkpoint 路径下对 Pure PI handover 基线的复现，不是对其他任务的成功率保证。

其中前三个配对 seed 在同一重建环境、同一权重、state 和动作解码下，仅把 RT 切换为
默认 shader，就从 **3/3** 降为 **0/3**。首帧视频显示 RT 的粉色调与历史成功视频
相近，默认 shader 的灰白画面明显不同。这证明渲染/图像输入协议足以影响 Pure PI
rollout。RoboTwin 2.0 即使使用 RT，修复动作步数后这三个 seed 仍只有 1/3，说明
还存在旧/新仿真器或评测链路差异；不能把 2.0 的差距全归因于 shader。

| handover seed | 历史旧链路 | 当前 RoboTwin 2.0（修复动作步数后） | 重建旧版＋默认 shader | 重建旧版＋RT |
| --- | ---: | ---: | ---: | ---: |
| `100100009` | 1/1 | 0/1 | 0/1 | 1/1 |
| `100100023` | 1/1 | 1/1 | 0/1 | 1/1 |
| `100100028` | 1/1 | 0/1 | 0/1 | 1/1 |

其余五个有效 seed `100100000/015/027/036/039` 在 node003 分别为 1/1，
与表中三个 seed 合计 8/8；每个 seed 使用不同 tag，未发生输出目录碰撞。
历史请求中的 `100100005/006` 被专家预检跳过，不计入 8 个有效分母。

## 缓存恢复与运行路径

`VLAct-main.zip` 只有 VLAct 评测适配器，没有 RoboTwin 仿真器。历史使用的
`/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval/` 在恢复前只剩
`envs_overlay/` 和 `eval_result/`。本机另存有旧版核心组件：

- 任务/机器人代码：`/mnt/jfzn/liujn/robotwin_envs_overlay/`；
- 旧 `script/eval_policy.py`：`/mnt/jfzn/liujn/robotwin_script_overlay/`；
- 配置：`/mnt/jfzn/liujn/robotwin_task_config/`；
- 描述：`/mnt/jfzn/liujn/robotwin_description/`；
- 资源：`/mnt/jfzn/liujn/robotwin_assets/`。

这些缓存以符号链接组装在独立目录
`/mnt/jfzn/liujn/code/Starvla/Starvla/playground/RoboTwin_eval_recovered_20261013/`，
该目录有自己的 `eval_result/`；原有历史结果没有被覆盖。缓存的 aloha-agilex cuRobo
YAML 仍硬编码原 `RoboTwin_eval/assets/` 绝对路径，因此仅在原目录补了一个指向现存
资源缓存的 `assets` 符号链接。恢复目录的 `RECOVERY_NOTES.md` 记录了各链接来源。

旧版 `envs` 可以在 node003 导入，旧 `script/eval_policy.py` 与固定 seed evaluator
可以实际完成专家预检及策略 rollout；这比只找到日志/视频更接近可运行的恢复。
但是缓存源码未证明与 10 月 9 日 checkout **字节完全一致**：缓存中的 `_base_task.py`
会响应 `SAPIEN_DISABLE_RAYTRACING=1` 使用默认 shader；历史启动脚本虽设置了该变量，
历史视频却呈现与 RT 复测一致的成像。因此应以实际图像和同 seed rollout 判定协议，
不能只凭脚本环境变量推断历史 renderer 状态。本次成功复测显式设置
`SAPIEN_DISABLE_RAYTRACING=0`，采用当前缓存代码中的 RT 分支；没有修改任务成功阈值。

## 排除项与注意事项

- 2.0 兼容层此前未递增动作段 `step`，导致每步重取动作段第一步；已修。修复后
  2.0 对上述 seed 仍仅 1/3，所以该 bug 不是全部原因。
- `VLAct-main.zip` 的适配器会删除 state；但历史 8/8 用的是已有改动的评测链路，
  历史日志未保存实际请求载荷。不能把压缩包行为直接等同历史输入协议。
- 当前服务端可切换 `wrap`/统计量解码。无 state＋wrap 和无 state＋统计量在 node003
  对这三个 seed 均为 0/3；没有证据把这两者设为恢复基线的默认组合。历史日志记录
  `joint_action_unnorm_mode: wrap`。
- 较早的 `VLAct-main` `QwenPI_v4.py` 已用同一 checkpoint 在重建旧版＋默认 shader
  上对 `100100009` 复测，仍是 0/1；仅换这份较早模型源码未修复该 seed。
- 一次 `023/028` 并行试跑误用了相同 tag，导致视频输出目录碰撞；该轮**作废**，
  不纳入上表。上表采用独立 tag 的重跑。

## 证据位置

- 历史 8/8：`logs/last_four_handover_block_matrix_pure_20261009_165117_gpu0/`，
  及旧 `RoboTwin_eval/eval_result/handover_block/model2robotwin_interface/demo_clean/matrix_pure/2026-10-09 16:53:47/`；
- 重建旧版＋默认 shader：`logs/matched_handover_block_pure_recovered_legacy_retry_node3_20261013_seed_100100009_20261013_202443/`，
  以及独立 tag 的 `023/028` 日志；
- 重建旧版＋RT：`logs/matched_handover_block_pure_recovered_rt_node3_20261013_seed_100100009_20261013_205256/`，
  以及独立 tag 的 `023/028/000/015/027/036/039` 日志；
- 各 seed 的结果清单在恢复目录的 `eval_result/.../_fixed_seeds.json`，记录实际
  `successes` 和 `episodes`。策略 Python 为 `miniconda3/envs/vlact/bin/python`，
  仿真 Python 为 `robotwin_env/bin/python`。

本次恢复与协议验证针对 Pure PI handover，不把预检拒绝、建场失败或视频输出冲突算作
策略失败。接下来若比较蒸馏 checkpoint，必须在相同的重建旧版＋RT 协议、相同有效
seed 上做配对；不能直接拿 RoboTwin 2.0 的零分与此 8/8 相减。

## 补充：无 state＋wrap 配对复测（2026-10-13）

在 node003 上用同一 Pure PI 100k checkpoint、同一重建旧版环境、RT 渲染、
`handover_block/demo_clean` 和上述 8 个固定 seed 复测。服务端显式设置
`ROBOTWIN_ACTION_UNNORM_MODE=wrap`，客户端显式设置 `ROBOTWIN_SEND_STATE=0`；
各日志确认 `send_state_to_server: False` 和 `action_unnorm_mode: wrap`。只改变
state 请求输入这一项（并行 seed 使用 node003 不同 GPU）。

| scene seed | 有 state＋wrap＋RT | 无 state＋wrap＋RT |
| --- | ---: | ---: |
| `100100000` | 1/1 | **0/1** |
| `100100009` | 1/1 | 1/1 |
| `100100015` | 1/1 | 1/1 |
| `100100023` | 1/1 | 1/1 |
| `100100027` | 1/1 | 1/1 |
| `100100028` | 1/1 | 1/1 |
| `100100036` | 1/1 | 1/1 |
| `100100039` | 1/1 | 1/1 |
| 合计 | **8/8** | **7/8** |

无 state 轮的 8 份 `_fixed_seeds.json` 均为 `episodes=1`、`skipped_seeds=[]`，
且有对应非空成功/失败视频；`100100000` 是完成 rollout 的真实失败，不是预检零分。
但每颗 seed 仅跑一次，QwenPI_v4 推理含随机初始噪声，因此这组结果不能把
`100100000` 的单次失败严格归因于 state 缺失。它证明的是：在这套 RT 条件下，
无 state＋wrap 不是此前默认 shader 条件下的 0/3，而是本次 7/8。

复现实验脚本：`scripts/physical_tokens/run_pure_no_state_wrap_recovered.sh`。
日志：`logs/pure_no_state_wrap_rt_recovered_node3_20261013/`（009/023/028）和
`logs/pure_no_state_wrap_rt_parallel_node3_20261013_seed_*/`（其余 5 个 seed）。
结果清单位于恢复目录 `eval_result/handover_block/model2robotwin_interface/demo_clean/`
下对应 tag 的时间戳目录中。GPU 0 在完成 028 后短暂启动了重复 000，随后停止；
该未完成的重复试跑不计入结果，000 采用 GPU 1 的完整 rollout。

## 补充：无 state＋stats 配对复测（2026-10-13）

仍用上述 8 个 `handover_block/demo_clean` 固定 seed、同一 Pure PI 100k 权重、
重建旧版 RoboTwin 和 RT 渲染，在 node003 的 GPU 0–7 各跑一个 seed。客户端显式
`ROBOTWIN_SEND_STATE=0`，服务端和客户端显式 `ROBOTWIN_ACTION_UNNORM_MODE=stats`；
8 份运行日志分别确认 `send_state_to_server: False`、`action_unnorm_mode: stats`。
8 份 `_fixed_seeds.json` 均为 `episodes=1`、`successes=0`、`skipped_seeds=[]`，
并有非空失败视频。因此这次是 **0/8 个有效策略 rollout**，不是预检零分。

| 配对协议 | 8 个有效 handover seed |
| --- | ---: |
| 有 state＋wrap＋RT | 8/8 |
| 无 state＋wrap＋RT | 7/8 |
| 无 state＋stats＋RT | **0/8** |

原因是训练和解码尺度不匹配，而非 `stats` 一般不可用。checkpoint 的
`config.yaml` 指定 `data_mix: robotwin_all_wrap_32`；对应 `AgilexWrapData32Config`
将关节动作设为 `action_joint_mode="wrap"`，其训练归一化公式是将原弧度角取模到
`[-π, π)`，没有映射到 `[-1,1]`。推理的 wrap 分支也按这个周期还原。stats 分支则
先把输出裁到 `[-1,1]`，再按 dataset_statistics 的逐关节 min/max 线性展开。
例如第 2 个关节的统计范围约 `[-0.364,4.179]` rad：模型输出 0 时，wrap 得到
0 rad，stats 却得到约 1.91 rad；输出幅度超过 1 rad 时还会被 stats 截断。
故 stats 会把本应为弧度角的预测解释成另一套数值协议，造成系统性的错误关节目标。

本轮使用 `ACTION_UNNORM_MODE=stats` 调用
`scripts/physical_tokens/run_pure_no_state_wrap_recovered.sh`，各 seed 的日志在
`logs/pure_no_state_stats_rt_node3_20261013_seed_*/`；结果清单在恢复目录
`eval_result/handover_block/model2robotwin_interface/demo_clean/` 下对应 tag 的
时间戳目录。单次 rollout 仍有随机性，但 0/8 与训练/解码公式的差异方向一致；
不应把 stats 结果当作此 wrap 训练 checkpoint 的默认性能。
