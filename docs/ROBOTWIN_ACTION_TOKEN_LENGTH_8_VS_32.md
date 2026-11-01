# RoboTwin 动作 tokenizer：32 步编码为 8 或 32 个 token

## 目的与边界

用户要求试验把每段 **32 步 × 14 维**动作从当前 **8 个离散 token** 改为 **32 个离散 token**。本实验只训练/评估动作 tokenizer，**暂不更改**正在提交的 VLM 自回归模型，它仍监督 8 动作 + 64 物理 token；若决定切换到 32 动作 token，必须另行重导真值 ID、修改输出长度并重新训练 AR 模型，旧 AR checkpoint 不能当作同一目标继续训。

沿用全 50 任务的 `action_all50_wrap32.npz`（549,787 段）、固定 episode 留出 `docs/robotwin_tokenizer_episode_split.json`、EMA-VQ K525、EMA decay 0.95、dead-code steps 100、同 seed 42、同网络宽度/损失权重/学习率。新做两组匹配实验，均单卡全局 batch **128**、20k 优化步：8-token 对照与 32-token 候选。原因是旧 8-token 与 16-token 训练使用 batch 256，直接拿它们和新 batch 128 对照会混入训练采样量差异。脚本：`scripts/physical_tokens/run_full50_action_8vs32_bs128_node1.sh`，使用 `vlact` 环境，GPU4/7 与其他任务共享，不停止任何别人的进程。两组在同一留出集上报告 reconstruction Smooth-L1/RMSE、关节 wrap MAE、关节速度 MAE、夹爪准确率、码本 active 数/比例、entropy、perplexity、top10 比例与各任务指标。

## 历史参考（batch 256；不作为匹配实验最终结论）

| 32 步动作压缩后长度 | 重建 RMSE ↓ | 关节 wrap MAE（度）↓ | 速度 MAE（弧度）↓ | 夹爪准确率 ↑ | 有效码数 / 525 | perplexity ↑ | top10 比例 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 token | 0.09999 | 3.2473 | 0.009291 | 0.998656 | 508/525 | 363.36 | 0.08569 |
| 16 token | 0.09612 | 3.1306 | 0.009491 | 0.999108 | 507/525 | 376.63 | 0.07653 |

历史 16-token 重建误差较低，但关节速度误差略高；更长 token 序列也会增加后续 VLM 生成长度和错误累积风险。32-token 是否值得采用，不能只看 RMSE，还需同时看动作动态、码本使用和后续 AR/rollout 成本。上表原始报告位于对应 `artifacts/robotwin_full50_study/action_tokenizers/ema_{8,16}x525_decay095_dead100_steps20000/heldout_step20000.json`。

## 新匹配实验状态

2026-10-27 22:13 在 node001 启动：

- 8 token、batch 128：`artifacts/robotwin_full50_study/action_tokenizers/ema_8x525_decay095_dead100_bs128_steps20000/`
- 32 token、batch 128：`artifacts/robotwin_full50_study/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/`

每 2k 步做验证、每 5k 步保存。两组 20k checkpoint 和完整留出集评估均已完成，脚本于 2026-10-27 22:36:29 正常退出（exit code 0）。下表均为**同一批 55,797 段 episode 留出动作**，不是训练 batch 的瞬时统计；原始数据在两个实验目录的 `heldout_step20000.json`。

| 指标 | 8 token | 32 token | 方向/解释 |
|---|---:|---:|---|
| 重建 Smooth-L1 ↓ | 0.005173 | **0.004836** | 32 token 低约 6.5% |
| 全局重建 RMSE ↓ | **0.104830** | 0.106263 | 32 token 高约 1.4% |
| 关节 wrap MAE（度）↓ | 3.3379 | **3.0994** | 32 token 低约 7.1% |
| 关节速度 MAE（弧度）↓ | **0.009422** | 0.009668 | 32 token 高约 2.6% |
| 夹爪准确率 ↑ | 0.998790 | **0.998996** | 32 token 略优 |
| 全量有效码数 / 525 ↑ | **511 / 525** | 504 / 525 | 两者均接近全利用；不是 batch active ratio |
| 全量码本 entropy（nats）↑ | 5.9310 | **5.9792** | 32 token 的使用分布更均匀 |
| 全量 perplexity ↑ | 376.53 | **395.13** | `exp(entropy)`，不等于有效码数 |
| 全量 top10 占比 ↓ | 0.07553 | **0.06835** | 32 token 更分散 |
| 50 任务宏平均 RMSE ↓ | 0.103212 | **0.101251** | 32 token 在大多数任务更好 |
| 50 任务宏平均关节 MAE（度）↓ | 3.6914 | **3.4662** | 32 token 更好 |
| 50 任务宏平均速度 MAE ↓ | **0.009928** | 0.010234 | 32 token 更差 |

逐任务看，32-token 的 RMSE 在 **45/50** 个任务下降，关节 wrap MAE 在 **47/50** 个任务下降；但速度 MAE 仅在 **19/50** 个任务下降。全局 RMSE 与宏平均方向相反，主要有少数退化明显的任务：`press_stapler` 的 RMSE 从 0.1747 增至 0.3395，`stack_bowls_three` 从 0.0912 增至 0.1855。这些是完整留出集实际评估结果，不应仅用宏平均掩盖。

**判断：**32-token tokenizer 在多数任务的静态角度重建上有可见收益，码本分布也略均匀，但动态误差和全局 RMSE 未改善，且存在明显长尾任务退化。把动作输出从 8 增至 32，会使后续每段自回归目标由 `8 动作 + 64 物理 = 72` 个增至 `32 + 64 = 96` 个离散 token（总长度 +33.3%，动作部分 4 倍），还可能增加生成误差累积。本次 tokenizer 离线实验**不足以证明端到端控制性能更好**；建议暂保留已有 8-token AR 主线，把 32-token checkpoint 作为候选，只有完成匹配的 AR 训练和 rollout 成功率比较后再决定切换。两组的 checkpoint 均保留，未修改当前 8+64 AR 目标。

## 后续决定：采用 32-token 目标并重新提交 ACP（2026-10-27）

用户随后明确决定把动作 tokenizer 改成这组 32-token 候选，故以上建议成为**历史评估结论**，新正式候选改为每段 32 动作 + 64 物理 token；尚无端到端收益数据，不能把离线重建改进表述成 rollout 成功率改进。

- 动作 tokenizer：`artifacts/robotwin_full50_study/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/step-00020000.pt`，即前文同条件对比中的 20k EMA-VQ K525 checkpoint。
- 用 `vlact` 环境从该 checkpoint 导出 `action_32x525_ids_vlact_for_dual_ar.npz`：549,787 段，`[549787,32]`，manifest 记录 checkpoint SHA256。与原 64/K2048 物理 ID 按 `(task_id, episode_id, base_index)` 精确合并为 `dual_ar_32action_64physics_targets.npz`：50 任务、469,787 个完整未来 clip，动作 `[469787,32]`、物理 `[469787,64]`；原 8-token 缓存不覆盖。
- `QwenDualAutoregressiveReasoner` 的目标构造、loss mask、推理切换点和生成长度现在按配置支持 8 或 32 动作 token；8-token 默认保持不变。新配置 `scripts/physical_tokens/qwen_strict_lora_r32_32action_64physics_full50_robotwin_finetune.yaml` 保持 RoboTwin-All 100k 基座、冻结视觉、严格分模态绑权、LoRA rank/alpha 32、K525/K2048、50 任务与原 episode 划分；仅改动作长度、prompt 和目标缓存。每段预测 96 个码，动作/物理 CE 仍分别取均值相加。
- 新 ACP 启动脚本 `scripts/physical_tokens/run_acp_dual_ar_32action_robotwin_ft_20k.sh`：4 张 A100 × 每卡 8 × 梯度累积 4 = 全局 batch 128，最多 20k 步，每 1k 保存；独立 run ID `qwen_dual_ar_32a64p_full50_seed42_steps20000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_acp_bs128_r32_32action_v1`，持久日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_robotwin_ft_20k_bs128_r32_v1.log`。
- `vlact` 下 29 项相关测试通过，脚本语法与缓存形状预检通过。ACP 从本方旧任务 `pt-0b3hrb0l` 复制相同的 4 卡规格、镜像和挂载，仅覆盖启动命令；**新任务 `pt-z9vz3h39`**（显示名 `ljn-robotwin-dualar-32a64p-r32a32-bs128`）于 22:46 提交成功，提交后查询状态为 `QUEUEING`，**尚未开训**。旧 8-token 任务仍保留，不停止任何作业。新任务是否真正开跑、显存是否够、损失及留出表现，都必须以后续 ACP 日志和 checkpoint 核验。

2026-10-28 17:28 更新：按用户明确要求，旧 8-token ACP 任务 **`pt-0b3hrb0l`** 已先停止（确认 `SUSPENDED`），再删除（CLI 返回 `successfully deleted jobs`，重新查询 `No jobs found`）。只删除 ACP 作业记录；8/32-token 本地缓存、checkpoint、代码及报告均保留。新的 32-token ACP 任务 `pt-z9vz3h39` 仍为 `STARTING`，没有停止或删除。

## node003 后四卡并行试跑（2026-10-28）

用户希望 ACP 排队时同时在本地后四卡运行 32-token 任务，随后立即纠正目标机器为 **node003**（不是 node002）。误启的 node002 试跑只运行几十秒；核对专属进程组 `822148` 仅含本次进程后发送 SIGTERM，确认无残留；没有停止别人的任务。其专属启动日志和未完成目录保留作审计，不作为训练结果。

node003 GPU4–7 启动前分别约有 55/42/44/51 GiB 空闲显存；四卡都被其他任务共享使用。新入口 `scripts/physical_tokens/run_node003_dual_ar_32action_robotwin_ft_20k.sh` 只使用 GPU4–7，维持与 ACP 相同的 32 动作 + 64 物理目标、RoboTwin-All 100k 基座、冻结视觉、严格绑权、LoRA rank/alpha 32、**4 卡 × 每卡 8 × 累积 4 = 全局 batch 128**、20k 步；采用独立的端口、日志和 run ID，不能覆盖 ACP checkpoint。

第一次 node003 尝试 `..._m8_v1` 在 step 0 失败，**不是 OOM**：Accelerate 1.5.2 在梯度累积的中间微批调用 `DeepSpeedEngine.no_sync()`，而 ZeRO-2 明确不支持梯度分片时的 `no_sync`，报 `AssertionError: no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2`。已在 `starVLA/training/train_starvla.py` 给 Accelerate 的 `GradientAccumulationPlugin` 设置 `sync_each_batch=True`（累积步数大于 1 时）；DeepSpeed 继续负责真正的梯度累积和更新，Accelerate 仍追踪优化步边界。这一共享入口修复也适用于尚未启动的 ACP 作业。第一次失败目录不复用。

第二次 `..._m8_v2` 独立启动后，strict warm-start、50 任务数据集、ZeRO-2 优化器均通过；截至 17:51 已连续完成至少 **3 个优化步**，无 OOM，最低可用显存约 25 GiB。持久日志 `artifacts/robotwin_full50_study/node003_logs/dual_ar_32action_bs128_r32_m8_v2.log`；checkpoint 目录 `results/Checkpoints/qwen_dual_ar_32a64p_full50_seed42_steps20000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_node003_bs128_r32_32action_m8_v2/`。此时每个优化步约 20–24 秒，受共享 GPU 负载影响，不能据最初几步稳定估计全程耗时；首个 1k checkpoint 仍未产生。ACP `pt-z9vz3h39` 查询仍为 `STARTING`，没有触碰。

## ACP 单卡 A100-80GB 同目标试跑（2026-10-28）

按用户要求，在不停止 node003 试跑和原四卡 ACP `pt-z9vz3h39` 的情况下，新增 ACP 单卡任务 **`pt-dift3y5o`**（显示名 `ljn-robotwin-dualar-32a64p-1gpu-bs128`）。从本方四卡 ACP 任务复制镜像和 AFS 挂载，只把规格改为 `N2lS.Ii.I60a.1` 并覆盖启动命令。入口 `scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_20k.sh`，目标/基座/LoRA/冻结策略不变；**1 卡 × 每卡 8 × 累积 16 = 全局 batch 128**，最多 20k 步，每 1k 步保存，独立 run ID `qwen_dual_ar_32a64p_full50_seed42_steps20000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_acp_1gpu_bs128_r32_32action_v1`，持久日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_20k_bs128_r32_v1.log`。

17:56 提交后 ACP worker 已分配到单张 A100-80GB，worker 状态 `Running`，持久日志确认 DeepSpeed/NCCL 单进程启动、全 50 任务数据集和 714 个 Qwen 张量 strict warm-start 正常；截至随后核验已完成至少 **7 个优化步，无 OOM**。早期速度约 15 秒/优化步，只是初始观察；20k 步的真实耗时、训练损失和效果均需后续日志与 checkpoint 验证。作业列表短暂显示 `QUEUEING`，之后也已同步为 `RUNNING`。

## 单卡微批与梯度检查点调优（2026-10-28）

单卡原方案 `pt-dift3y5o`（微批 8、累积 16、全局 batch 128）在 100 步左右测得约 **14.0 秒/优化步**；ACP worker 的 `nvidia-smi` 读数为显存 **14,815/81,920 MiB（18.1%）**、GPU 利用率瞬时约 94%。原配置 `trainer.enable_gradient_checkpointing: true`，Qwen 加载日志明确确认启用。只修改正在运行的启动脚本/配置不会热更新 worker，因此调优使用独立 ACP 作业、run ID 与日志。

运行器 `run_full50_dual_autoregressive_reasoner.sh` 新增可选 `DUAL_AR_GRADIENT_CHECKPOINTING=0/1` CLI 覆盖，默认不改变原 YAML；`false` 的 OmegaConf 布尔解析已核对。调优结果：

| 单卡方案 | 累积 | 检查点 | 全局 batch | 观察结果 |
|---|---:|---|---:|---|
| 微批 8（旧 `pt-dift3y5o`） | 16 | 开 | 128 | 约 14.0 秒/步；显存 14,815/81,920 MiB（约 14.5/80 GiB） |
| 微批 32（`pt-b3798zfu`） | 4 | 关 | 128 | **第 0 步 OOM**；进程用约 79.28 GiB，GPU 只剩 21 MiB；保留失败日志，不继续此档 |
| 微批 32（`pt-6qy26zim`，20 步短测） | 4 | 开 | 128 | **完成 20/20 步，无 OOM**；后期约 **12.3–12.6 秒/步**；每秒 `nvidia-smi` 采样峰值 **20,117/81,920 MiB（24.6%）**；最终权重已保存。短测速度比当前微批 16、关检查点主线慢约 18%（以 12.3 与 10.4 秒/步比较） |
| 微批 64（`pt-gyg8caxr`，20 步短测） | 2 | 开 | 128 | **完成 20/20 步，无 OOM**；训练后期约 **11.9 秒/步**；每秒 `nvidia-smi` 采样峰值 **26,493/81,920 MiB（32.3%）**；最终权重已保存。比微批 32、开检查点略快，但比微批 16、关检查点主线慢约 14%（以 11.9 与 10.4 秒/步比较） |
| 微批 16（`pt-my24ff1g`） | 8 | 关 | 128 | 已持续完成数十步，约 **10.4 秒/步**；显存约 **71,299/81,920 MiB（87%）**，约 9,923 MiB 空闲；当前单卡主线 |
| 微批 128（`pt-86ie5stn`，**1 步烟测**） | 1 | 开 | 128 | **可以跑通** 1 个优化步并保存 `steps_1` 与 final checkpoint，无 OOM；ACP 状态 `SUCCEEDED`；第一步 model time 13.665 秒，另有冷启动取数 10.205 秒，不能用 1 步估计稳态速度 |
| 微批 128（`pt-pudtphxw`，20 步短测） | 1 | 开 | 128 | **完成 20/20 步，无 OOM**；训练后期约 **11.9 秒/步**，与微批 64×2、开检查点几乎相同；每秒 `nvidia-smi` 采样峰值 **43,303/81,920 MiB（52.9%）**；最终权重已保存。比当前微批 16×8、关检查点主线慢约 14% |

微批 128 **不使用累积并非不可能**，但必须保留梯度检查点；微批 32 在关闭检查点时都已 OOM，因此不能认为微批 128 也能关。微批 16 关闭检查点在当前实测中显存利用率明显更高、优化步更快，故保留为长跑方案。主线入口 `scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_tuned_20k.sh`，持久日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m16_gc0_bs128_r32_v1.log`；仍维持 32 动作 + 64 物理、LoRA rank/alpha 32、全局 batch 128、上限 20k 步。旧单卡 `pt-dift3y5o` 已按本次优化请求停止并确认 `SUSPENDED`，其尚未产生首个 1k checkpoint，旧日志与作业记录保留；node003 和四卡 ACP 未触碰。单步累积=1 试验限定 `max_steps=1`，不用于长训；它的两样本、两任务 teacher-forced 留出诊断已运行，但一步权重不足以比较模型质量。单卡新方案仍须在首个 1k checkpoint 时验证保存阶段的显存峰值与耗时。

微批 32、开检查点短测使用独立入口 `scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_m32_gc1_bench.sh`，限制 20 步，不替换长跑主线；训练日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m32_gc1_bs128_r32_v1.log`，显存采样 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m32_gc1_bs128_gpu_memory.csv`。这里的 20,117 MiB 是 1 秒采样峰值，不等于框架内部瞬时最大显存；两个配置也并非完全相同长度的基准测试，因此速度差值只作当前吞吐判断。

微批 64、开检查点短测同样限制 20 步，入口 `scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_m64_gc1_bench.sh`，日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m64_gc1_bs128_r32_v1.log`，显存采样 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m64_gc1_bs128_gpu_memory.csv`。每个“秒/步”均为**一次优化器更新**，单卡情况下每次包含 128 个训练样本：微批 16×累积 8、32×4、64×2 或 128×1。20 步测试不足以比较模型最终质量；目前吞吐最好的仍是微批 16、累积 8、关检查点。

微批 128、开检查点的 20 步短测入口为 `scripts/physical_tokens/run_acp_dual_ar_32action_1gpu_m128_gc1_bench.sh`，显存采样 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m128_gc1_bs128_gpu_memory.csv`。训练日志为 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_1gpu_m128_gc1_bs128_r32_v1.log`；由于启动器按微批与检查点状态命名日志，这份日志**先有 1 步烟测、后追加了 20 步短测**，两个任务的独立 run ID 分别含 `steps1` 与 `steps20`，分析速度时只看后半段 `20/20` 进度。128×1 无累积技术上可行，但未带来可测得的吞吐优势，且显存使用比 64×2 多约 16,810 MiB。
