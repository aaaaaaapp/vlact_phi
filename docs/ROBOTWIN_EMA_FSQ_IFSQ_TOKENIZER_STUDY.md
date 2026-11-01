# RoboTwin 全训练集动作与物理 tokenizer：EMA-VQ / FSQ / IFSQ 对比

状态：进行中。本文只把实际完成并可复现的运行标为结果；待跑实验和历史基线分开记录。

**最新摘要（2026-10-27）**：全 50 任务动作与物理 tokenizer 的 EMA-VQ/FSQ/iFSQ 首轮离线对比、物理 64/128/256 长度对照、动作 8/16 长度对照及同 levels 的 FSQ 对称/非对称公式复核均已完成。用户确定下一阶段的**动作长度为 8 token/32-step chunk、物理长度为 64 token/17-frame clip，并明确物理码本 K2048**；当前相应已训模型为动作 EMA-VQ K525 与物理 `spatial_aligned` EMA-VQ K2048。这是人为选型，尚未自动切换 VLM 配方、导出新缓存或完成策略训练。动作 16-token 虽改善静态重建，却使动作 ID 位置翻倍且关节相邻步增量 MAE 略差；物理 64-token 相比 128/256 损失部分离线重建精度，但降低后续序列长度。没有相同条件的 VLM 预测或 rollout 证据，不把选型写成下游性能赢家。

### 当前物理/动作 token 方案（决策记录，不等于已完成下游接入）

| 模态 | 用户选择 | 对应已训 checkpoint | 选择时的主要取舍 |
|---|---|---|---|
| 动作 | **8 ID / 32 步 chunk** | 当前全任务 EMA-VQ K525：`ema_8x525_decay095_dead100_steps20000/step-00020000.pt` | 16 ID 静态重建更好，但速度增量 MAE 高 2.15%，ID 位置和固定符号预算翻倍；下游收益未验证 |
| 物理 | **64 ID / 17 帧 clip，EMA-VQ K2048，空间对齐** | `ema_t64_k2048_steps20000/step-00020000.pt` | 同为 64 ID 时 K2048 RMSE `0.085326`，比 K1024 `0.085883` 略低；更大词表是否更易由 VLM 预测未验证 |

物理选定 checkpoint 为 `artifacts/robotwin_full50_study/physics_tokenizers/ema_t64_k2048_steps20000/step-00020000.pt`，SHA256 `182cafcfb42a86d3c5604842610c3f47d9b0e117818cdeaa40887c11417754f6`；训练配置已核对为 `emitted_tokens=64`、`queries_per_transition=4`、`codebook_size=2048`、`token_routing=spatial_aligned`、seed42、20,000 steps。独立完整留出评估为 47,797 clips，差分 RMSE `0.085326`、未来特征 RMSE `0.151243`、有效码 `2048/2048`、perplexity `1959.52`。K1024→2048 在相同 64-token 长度上降低 RMSE 约 0.65%，但把每个 ID 的二进制固定码长由 10 增至 11 bit；这是离线权衡，**选 K2048 是用户的人为决定，不是已证 VLM 最优**。

若下一阶段同时生成两类离散 ID，名义输出位置为 8+64=72（不含边界/特殊 token）；动作与物理的时间采样和解码器不同，不能把这 72 个位置当作同一种 token。未来 VLM 配方须显式匹配 token 数、词表大小、checkpoint 和导出缓存；本记录固定的是**方案决策**，不宣称配置已经切换或策略已经验证。

## 1. 范围与数据

- **全任务目标**：源目录 `/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin` 有 50 个有效任务目录（排除 `.smoke_randomized_press_stapler`），每任务 50 条 episode，共 2,500 条、549,787 帧。每个任务的 `meta/info.json` 都标为 train `0:50`。这才是本研究的“RoboTwin 全部训练集”范围；每任务仍按固定 episode 留出验证，不能把验证 episode 混进训练。
- **当前全任务缓存已齐备**：动作 `artifacts/robotwin_full50_study/action_all50_wrap32.npz` 与物理 `artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz` 均覆盖 50 任务、2,500 条 episode、549,787 个 key；物理 manifest 标记 `exact_action_key_alignment: true`。动作留出 55,797 chunks；物理因仅纳入完整 17 帧 clips，训练 421,990、留出 47,797 clips。
- **历史 8 任务先导缓存**仍单独保留：动作 `artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz`，物理 `artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz`。先导物理训练 64,589、留出 7,709 个完整 clips，action auxiliary weight 0.25；全任务物理实验 aux=0，二者数值不可直接当作同协议对照。

## 2. 比较协议

1. 三种量化器分别用于**动作**与**物理** tokenizer。每一模态内部固定同一输入、episode 划分、encoder/decoder、token 位置数、近似离散容量（bits/token）、训练步数与评估样本；仅替换量化器及其必要的专有超参数。动作与物理的损失量纲不同，不做跨模态 loss 排名。
2. 先复现共同预算，再对每种方法分别调优（EMA decay/commitment/dead-code；FSQ levels/投影/分布正则；IFSQ 按原论文机制和建议范围）。调优只看验证集，最终选择应另用固定测试集或明确注明属于验证集选择结果；不从训练 loss 单点选优。
3. 每次完整评估保存 checkpoint hash、配置、数据 hash、seed、训练时长、训练/验证损失分项、重建误差、按任务宏平均和总体微平均。动作包括关节 wrapped MAE、速度 MAE、夹爪准确率；物理包括 quantized/continuous/no-motion RMSE、action auxiliary loss（若启用）。
4. 码分布在**相同的完整验证集**累计 ID 直方图，再计算 active codes/K、Shannon entropy `H=-Σ p log p`（nats）、`exp(H)` perplexity、top-10 ID 总频率、归一化 entropy `H/log K`。另外保存逐任务与逐位置统计。不能把 batch active ratio 当成全局码本利用率；FSQ/IFSQ 的 ID 是离散组合，EMA-VQ 的 ID 是 learned codebook entry。
5. 选优顺序：先保证无数值问题和可接受的 held-out 重建，再比较码本分布；最后通过相同 predictor/下游策略预算比较可预测性和动作效果。若指标冲突，分别报告 Pareto 关系，不用单个 perplexity 宣称赢家。

## 3. 当前证据与待解决问题

| 项目 | 已有证据 | 本轮状态 |
|---|---|---|
| 动作 EMA-VQ | 历史单任务 `click_alarmclock` 的 8×512、5000 step 验证：关节 MAE 1.64°、active 128/512、perplexity 55.19。不是这次全任务结果。 | 50 任务两组 20,000-step 匹配训练与完整留出集评估已完成，见下文 |
| 动作 FSQ | 同一历史单任务 8×8×8：关节 MAE 2.39°、active 80/512、perplexity 31.79。 | 50 任务四组 20,000-step 匹配训练与完整留出集评估已完成，见下文 |
| 物理 EMA-VQ | 50 任务 K256、20,000-step 完整留出评估，pooled RMSE 0.082253。 | 已完成首轮；见第 5 节 |
| 物理 FSQ | 同协议 K256 两组及奇数 levels 成对 K255/245/275 三组。 | 已完成首轮；见第 5 节 |
| 动作/物理 IFSQ | 用户提供官方代码 `/mnt/jfzn/liujn/code/iFSQ-main`；关键替换为 `2·sigmoid(1.6x)−1`，simple bound 要求奇数 levels。 | 动作三 seed K525 与物理三组匹配 levels 的首轮均已完成 |

## 4. 运行记录

2026-10-25：检查了仓库、D15/D28 缓存 manifest、既有训练/评估脚本和本机 GPU 占用。8 张 A100 均有其他任务，占用约 46–75 GiB/卡；不能把剩余显存视为独占资源。访问 arXiv PDF、摘要页及其他镜像均连接超时。已向用户询问论文 PDF 与“RoboTwin 全部训练集”的任务范围。IFSQ 尚未实现或训练，因此**无三方法最终赢家结论**。

同日更新（**8 任务先导实验**）：动作 `8 tokens × 512 IDs` 两组已用固定 manifest `docs/robotwin_tokenizer_episode_split.json` 完成 5000-step 重训。训练样本为 76,109 个动作 chunk；留出 8,989 个 chunk，合计 71,912 个离散 ID。以下码指标均由**完整留出集累计**，不是训练日志中的 batch 平均。两者 seed 42、batch 256、网络及 loss 权重相同，FSQ levels `8,8,8`。

| 动作方法 | 关节 wrapped MAE ↓ | 速度 MAE(rad) ↓ | 夹爪准确率 ↑ | 有效码数 ↑ | perplexity ↑ | entropy(nats) ↑ | top10 占比 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMA-VQ | 2.3417° | 0.007393 | 99.7528% | 380/512 | 229.75 | 5.4370 | 13.50% |
| FSQ `8,8,8` | 4.0068° | 0.013493 | 99.5967% | 307/512 | 123.82 | 4.8188 | 24.83% |
| FSQ `4,4,4,8` | 3.7713° | 0.013482 | 99.6458% | 269/512 | 124.09 | 4.8210 | 22.96% |

本轮动作配置中 EMA-VQ 的重建与码分布都更好；这不是三方法最终结论。EMA checkpoint SHA256 `2191032759b1565af1f25e7c4befac1790f09682d93ef26ff92d07c7b246bbca`；FSQ `8,8,8` 为 `de941ecc6d38b350d56f4b65b7ba03b31184d1ab5efc6e45c1384b8808b41c6f`。完整 JSON 在各自目录的 `heldout_evaluation.json`，checkpoint 在同目录 `step-00005000.pt`。将 FSQ levels 改成 `4,4,4,8` 后关节 MAE 下降约 5.9%，但有效码数由 307 降为 269，说明 levels 需要联看重建和码分布。

物理 tokenizer 的 EMA/FSQ 同配置重训已启动：256 个位置、256 种 ID、spatial-aligned routing、action-token-residual auxiliary weight 0.25、batch 32、8000 steps。为避免反复缓慢解压 6.1 GB 的共享 NPZ，将原缓存逐字节复制到本机临时盘并核对 SHA256 `386b2154dd7c1bbdf6be253bcbfd7a66ab2894aee62598f75325f240f3f3d519`；从这份临时 NPZ 解出 `frame_features.npy`，供两个方法只读 memory-map。训练配置仍以原仓库 NPZ 为 `--input`，临时 `.npy` 仅是同内容的读取加速缓存，不是新数据。临时路径 `/tmp/robotwin_d28_extract.I57Ix3/frame_features.npy` 不保证跨机器/重启存在；复现时可从原 NPZ 的 `frame_features.npy` 成员重新解出。

物理 FSQ 的同容量 `4,4,4,4` levels 变体已启动，用来测试增加一个量化标量维数的效果；与原 `4,8,8` 共用其余训练设置和留出集。

8 任务物理 tokenizer 的第一组匹配实验已跑满 8000 step。以下是 7,709 个完整留出 clips × 256 位置的**微平均**；逐任务宏平均和位置统计的独立评估正在进行。

| 物理方法 | quantized RMSE ↓ | continuous RMSE ↓ | action aux loss ↓ | 有效码数 ↑ | perplexity ↑ | entropy(nats) ↑ | top10 占比 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMA-VQ K256 | 0.078861 | 0.071492 | 0.004335 | 256/256 | 245.12 | 5.5017 | 7.32% |
| FSQ `4,8,8` | 0.081762 | 0.081491 | 0.004485 | 256/256 | 193.86 | 5.2671 | 11.00% |

这个固定容量的首轮对照中，EMA quantized RMSE 低约 3.55%，码分布更均衡；FSQ `4,4,4,4` 尚未收敛，暂不下最终方法结论。EMA 的总训练 loss 包含 commitment，FSQ 不含同一项，因此不能仅用总 loss 数字排名；表中相同定义的 RMSE、action auxiliary 和码统计才可直接比较。

独立评估已生成 `heldout_evaluation.json`，检查点 SHA256：EMA `9149754715611b157b812d223d9ac83ce7cc0477bc37c62c4f5c258c1ee33027`，FSQ `77e875c1554104618c463a961f64615345bd0e530f2c870b2e685440c5520841`。8 任务**宏平均** quantized RMSE 为 EMA `0.083219`、FSQ `0.086307`；EMA 在 8/8 个任务的 RMSE 都更低。宏平均 perplexity 为 `208.19/190.56`，active ratio 为 `98.58%/98.83%`，top10 占比为 `12.55%/11.52%`（EMA/FSQ）。因此全局 pooled top10 的 EMA 优势**不能推广为每任务都更均衡**：例如 `place_object_basket` 的 top10 分别为 `28.1%/16.0%`。全局和逐任务码分布两套口径必须同时保留。

FSQ `4,4,4,4` 也完成了相同 8000-step 对照。验证微平均 quantized RMSE `0.080879`，宏平均 `0.085378`，比 FSQ `4,8,8` 的 `0.081762/0.086307` 更好；全局 perplexity `210.45`、active `256/256`，但 top10 占比增至 `13.35%`。它仍弱于 EMA 的 `0.078861/0.083219`。其 checkpoint SHA256 `f927f19736dc6a352f38c76d2120896b73d043db1fcd2a09a1e2de23b6c13d63`，逐任务及逐位置详情见其 `heldout_evaluation.json`。

全 50 任务导出准备：已审计 50 个 `meta/info.json`，动作合计 549,787 帧；确认现有 8-task D15/D28 缓存只是子集。`adjust_bottle` 的动作缓存作为第一个新增任务正在按 D15 契约导出。全任务新缓存必须使用单独路径，保留 8-task 历史缓存不覆盖；视觉导出需复用同一个 Qwen3-VL checkpoint、4×4 网格和 17 个 offsets，并在合并后逐 key 对齐动作缓存。

全任务动作导出由 `scripts/physical_tokens/run_full50_action_cache_export.sh` 管理：按 `docs/robotwin_50task_list.txt` 固定任务顺序，复用原 8 个已验证 parts，补导其余 42 个任务，合并到 `artifacts/robotwin_full50_study/action_all50_wrap32.npz`。首个新增 `adjust_bottle` 已成功导出 7,188 chunks；其余任务并行导出中。物理特征的可恢复导出入口为 `run_full50_physics_cache_export.sh`，沿用 D28 的视觉编码器和 checkpoint；11 GB checkpoint 已复制到本机临时盘，并核对 SHA256 `5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d`。临时副本同样不保证跨机器/重启存在。

全 50 任务动作缓存现已合并并校验：50 个唯一任务、2,500 条 episode、549,787 个 `[32,14]` chunks，qualified keys 无重复；缓存 SHA256 `3be69db04833df756680d9e62847e0be497324238f6cc4c9eafd0b4fb9a29865`。全任务 EMA-VQ 与 FSQ `4,4,4,8` 动作 tokenizer 已在相同 split、8×512 容量、batch 256、20,000 step 预算下分别启动。物理特征新增任务在 node001/node004 分片导出，明确排除任务重叠；合并前必须与全任务动作缓存逐 key 核对。

本次全任务动作训练实际使用 493,990 个 chunks，固定留出 55,797 个 chunks（每任务 episode `5,20,25,39,49`）；总数与合并缓存的 549,787 一致。共享 split 清单已把原“8-task”描述修正为同时适用于 8 任务先导和 50 任务正式实验，episode ID 本身没有改变。

因此“全训练集”目前准确含义是**覆盖全部 50 种任务**并从每种任务保留同一组 episode 作可比较验证，而不是把 2,500 条 episode 全部用于梯度更新。待用留出集选定方法与超参数后，可另做一次 2,500 条 episode 的最终 refit；那次模型不能再用同一留出 episode 给出无偏指标。物理特征提取所用 Qwen3-VL 预训练 checkpoint 的上游训练样本范围未在本研究重新审计，故这里的 held-out 仅保证 tokenizer 训练层面的 episode 隔离，不声称端到端预训练完全隔离。

继续推进：为防止把 8 任务先导实验的参数选择直接当作全任务最优，另在 node002 启动相同 20,000-step 预算的 FSQ `8,8,8` 及 EMA-VQ `decay=0.95, dead_code_steps=100` 对照，均沿用 seed 42、固定 episode 划分、网络、batch 与 8×512 容量。它们分别写入 `artifacts/robotwin_full50_study/action_tokenizers/fsq_8x512_levels888_steps20000/` 和 `ema_8x512_decay095_dead100_steps20000/`。两组原配置仍在运行，未有全任务最终结果。

另为 FSQ 增加 `2,4,8,8` 和 `4,4,4,4,2` 两个同为 512 组合码的维数/levels 候选，使用完全相同的 20,000-step 动作训练预算，输出分别在 `action_tokenizers/fsq_8x512_levels2488_steps20000/` 与 `action_tokenizers/fsq_8x512_levels44442_steps20000/`。这些是方法内调优，不等同于对 IFSQ 的替代实现。

全 50 任务动作训练的**5,000-step 早期检查点**已在相同的 55,797 个留出 chunks（446,376 个 token）上完成首次完整累计评估。训练仍在继续，以下不是最终排序；码分布数字不是 batch 平均。

| 早期动作方法 | Smooth L1 ↓ | RMSE ↓ | 关节 wrapped MAE ↓ | 有效码数 ↑ | perplexity ↑ | entropy(nats) ↑ | top10 占比 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| EMA-VQ K512 | 0.006601 | 0.11770 | 3.8546° | 501/512 | 382.89 | 5.9477 | 7.37% |
| FSQ `4,4,4,8` | 0.012909 | 0.17034 | 5.7767° | 437/512 | 197.45 | 5.2855 | 17.32% |

两份早期 JSON 分别在各自全任务实验目录的 `heldout_step05000.json`，含逐任务与逐位置码统计。此时 EMA 的重建与 pooled 码分布同时领先，但只有两个候选的早期点，不能推断充分调优后的 FSQ，更不能推断 IFSQ 或物理 tokenizer 的最终胜者。

六组动作候选全部完成 20,000-step checkpoint 后，可运行 `scripts/physical_tokens/run_full50_action_evaluation.sh` 对全部候选顺序做同一完整留出集评估；脚本要求六个 checkpoint 全齐，并拒绝覆盖已有评估 JSON。此入口已通过 `bash -n`，目前因训练未全部结束而未执行。

动作 FSQ `4,4,4,8` 已先完成 20,000 steps，checkpoint 存在；训练器的验证 batch 平均重建损失继续降到 step 20,000 的 `0.011506`。其余候选的匹配步数评估尚未齐备，因此这里不与它们的较早 step 直接排名。

该 FSQ `4,4,4,8` 的 20,000-step **完整留出集评估现已完成**：Smooth L1 `0.011422`、RMSE `0.16020`、关节 wrapped MAE `5.0527°`、有效码 `490/512`、perplexity `254.09`、entropy `5.5377 nats`、top10 `12.90%`；留出集确为 250 个 `(task_id, episode_id)`、55,797 个 chunks。与自身 5,000-step 相比，重建和码覆盖都有改善。详情在 `artifacts/robotwin_full50_study/action_tokenizers/fsq_8x512_levels4448_steps20000/heldout_step20000.json`；EMA 的 20,000-step 完整评估尚未齐备，不能把上述数值直接与 EMA 5,000-step 当成最终公平对照。

四组 FSQ 的全任务 20,000-step 留出集评估随后全部完成；同一 55,797 chunks、8×512 token 预算、seed 42：

| FSQ levels | Smooth L1 ↓ | 关节 wrapped MAE ↓ | 有效码数 ↑ | perplexity ↑ | entropy(nats) ↑ | top10 占比 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| `4,4,4,8` | 0.011422 | 5.0527° | 490/512 | 254.09 | 5.5377 | 12.90% |
| `8,8,8` | **0.010672** | **4.9610°** | 486/512 | 256.30 | 5.5464 | 13.80% |
| `2,4,8,8` | 0.011833 | 5.1687° | 492/512 | 256.20 | 5.5460 | 13.86% |
| `4,4,4,4,2` | 0.011926 | 5.1652° | **497/512** | **268.02** | **5.5910** | **12.81%** |

因此 FSQ 内部的最小重建损失来自 `8,8,8`，最均衡的 pooled 码覆盖/entropy 来自 `4,4,4,4,2`；不存在所有指标都赢的单个 FSQ 配置。各目录的 `heldout_step20000.json` 保存逐任务与逐位置明细。两组 EMA 的匹配评估见下文。

EMA-VQ `decay=0.95, dead_code_steps=100` 的 20,000-step 完整留出集评估也已完成：Smooth L1 `0.004784`、RMSE `0.09965`、关节 wrapped MAE `3.2291°`、有效码 `493/512`、perplexity `369.95`、entropy `5.9134 nats`、top10 `7.23%`，留出集同为 250 个 task-episode、55,797 chunks。相比 FSQ `8,8,8`（重建最优的 FSQ），它的重建、关节误差、perplexity 和 top10 均更好；相比 FSQ `4,4,4,4,2`，只有“非零出现过的码数”略少（493 对 497），却有更高 entropy/perplexity 和更低 top10。这个**动作 tokenizer 层面的领先**已有完整同预算证据，但不能外推到物理 tokenizer、IFSQ 或下游 rollout。

默认 EMA-VQ 也已完成 20,000 steps，并且六个动作候选全部通过相同完整留出集评估。以下把两种 EMA 配置与 FSQ 各有代表性的两种配置并列；另外两种 FSQ 的全部数字见上表。所有模型共用 50 任务动作缓存 SHA256 `3be69db04833df756680d9e62847e0be497324238f6cc4c9eafd0b4fb9a29865`、493,990 训练 chunks、55,797 验证 chunks、250 个留出 task-episode、8×512 ID、seed 42、batch 256、20,000 steps。

| 全任务动作配置 | Smooth L1 ↓ | RMSE ↓ | 关节 MAE ↓ | 速度 MAE(rad) ↓ | 夹爪准确率 ↑ | 有效码 ↑ | perplexity ↑ | entropy(nats) ↑ | top10 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA `decay=.95, dead=100` | **0.004784** | **0.09965** | **3.2291°** | **0.009546** | **99.8595%** | 493/512 | 369.95 | 5.9134 | **7.23%** |
| EMA `decay=.99, dead=200` | 0.005600 | 0.10792 | 3.3794° | 0.009606 | 99.8259% | **500/512** | **380.71** | **5.9420** | 7.57% |
| FSQ `8,8,8`（重建最优） | 0.010672 | 0.15761 | 4.9610° | 0.018946 | 99.7972% | 486/512 | 256.30 | 5.5464 | 13.80% |
| FSQ `4,4,4,4,2`（分布最均衡） | 0.011926 | 0.16424 | 5.1652° | 0.021002 | 99.6142% | 497/512 | 268.02 | 5.5910 | 12.81% |

**动作 tokenizer 阶段性结论**：在本次同数据/同预算的六个候选里，EMA-VQ 优于已调过四种 levels 的 FSQ；若优先重建精度和低 top10，选 `decay=.95, dead=100`，若优先最多被访问过的 ID 与 entropy/perplexity，选默认 `decay=.99, dead=200`。perplexity 是 `exp(entropy)`，不是独立的第二条证据。表中的“有效码”只表示至少使用一次；默认 EMA 的 500/512 不意味着每个码都常用。此结论只覆盖 tokenizer 离线动作重建；iFSQ 尚无完成的可比结果，物理缓存未合并，未有 rollout/策略测试，因此**不能据此宣布三方法总体赢家**。

任务均衡口径也支持该动作结论：EMA `.95` 的 50 任务宏平均 Smooth L1 `0.005779`、关节 MAE `3.6054°`、速度 MAE `0.010054`；重建最优的 FSQ `8,8,8` 分别为 `0.012072`、`5.3142°`、`0.019625`。更严格地逐任务与**四个 FSQ 候选中该任务最好的数值**相比，EMA `.95` 在重建 Smooth L1、关节 MAE、速度 MAE 三项均为 50/50 任务更低；但夹爪准确率不是每任务都赢（对 FSQ `8,8,8` 为 42/50 任务更高）。这些仍是同一 seed 的验证集结果，不等于独立测试或 rollout 成功率。

六个动作最终 checkpoint 的 SHA256（与上文配置同名目录）：EMA `.95` `5b31d7dc873b5311e6db520d72a16526a63e789e9ed9fe6b8fc8dfc1987c7ebd`，EMA `.99` `8bf80a8ce1d5e1428189d598a6d41eb7c1c16791fe817597512856f8b7dd50a6`，FSQ `8,8,8` `06d1848df5c09056c18687bf8b8cd7d8b324755d05d208cb26beda5d5532f4e4`，FSQ `4,4,4,8` `857ff2941fd35b253943744d3dd74fdf14546f1369f5f78f17cfb808f93f4720`，FSQ `2,4,8,8` `1baf1b565f0dc91b27e0b71f0f76c4b2c34ac4616eac711da1f6a3ee35ae7bd6`，FSQ `4,4,4,4,2` `1a3d6fc173aeaea863812992bb61a70b9206eb4654af1c985a747579f5904501`。各目录的 `heldout_step20000.json` 保存逐任务与逐位置明细。

动作独立评估现额外逐元素累计完整留出集的 Smooth L1 重建损失和 RMSE，并保存每任务及宏平均；只对有效时间步计数。此处 14 个动作维度的 `action_dim_mask` 经检查均为 true。原有 wrapped joint MAE、velocity MAE、gripper accuracy、全局/逐任务/逐位置码分布继续保留；重建损失与 EMA 特有 commitment loss 分开，避免用不可比的总训练 loss 排名。

评估 JSON 的旧字段 `validation_episodes` 表示去重后的 **episode ID 数**（固定为 5），而不是 50 任务合计的 episode 实例数；新增 `validation_task_episodes` 显式计数 `(task_id, episode_id)`，全任务应为 250，避免误读。

该评估器先在 8 任务 EMA checkpoint 上回归验证：原关节 MAE `2.341658°` 与 perplexity `229.7507` 保持不变，新增留出集 Smooth L1 `0.00606337`、RMSE `0.127160`；JSON 位于 `artifacts/action_tokenizer_quantizer_study/ema_8x512_steps5000/heldout_evaluation_with_loss.json`。这只是评估代码校验，仍属 8 任务先导。

相同的 8 任务留出集上，FSQ `8,8,8` 的 Smooth L1 / RMSE 为 `0.00861482 / 0.147445`，FSQ `4,4,4,8` 为 `0.00822770 / 0.144288`，对应 JSON 均位于各自实验目录的 `heldout_evaluation_with_loss.json`。因此先导实验的共同定义重建损失排序与关节 MAE 排序一致：EMA 最低、FSQ `4,4,4,8` 次之；全 50 任务仍须重新评估。

物理缓存新增 `index % 4 == 2` 分片在 node001 GPU6 开始导出；与此前 node001 GPU1 的 `%4==0`、node004 GPU0/1 的奇数任务分片不重叠。`adjust_bottle` 单任务在 node001 GPU3 单独导出，`click_bell` 已完成，两个任务均从分片任务列表排除。当前导出尚未完毕，不应启动全任务物理训练。全任务物理的可比首轮将不使用仅存在于 8 任务的 action auxiliary sidecar（aux weight 0）；若再加入辅助目标，须先生成逐 key 对齐的 50 任务 sidecar，且所有方法使用同一版本。这样 8 任务先导的 aux=0.25 数值与全任务 aux=0 的结果不能直接比较。

导出进度更新：`adjust_bottle` 单任务已成功写出 7,188 样本并生成 manifest；`blocks_ranking_size`、`grab_roller`、`lift_pot` 与 `click_bell` 等新增任务的分片也已完成。其余任务仍在导出；只有全部 50 个任务的 part 及 manifest 齐备，才运行 `ROBOTWIN_PHYSICS_MERGE_ONLY=1 bash scripts/physical_tokens/run_full50_physics_cache_export.sh` 进行合并和与动作缓存逐 key 对齐。

最新检查已有 **13 个新增物理分片**完成，另有原先 8 个可复用分片，合计 21/50 个任务具备特征 part；导出日志未见 traceback 或 CUDA OOM。剩余 29 个任务尚待完成，物理 cache 尚未合并，不能声称已完成物理 tokenizer 全任务训练。

后续物理导出持续推进；为缩短 `%4==2` 大任务 `blocks_ranking_rgb` 对后续任务的串行等待，在 node001 GPU4/GPU5 分别提前导出该分片队列尾部的 `stack_bowls_three` 与 `place_object_stand`/`put_bottles_dustbin`。两次补充导出均用显式 skip list 排除了原队列当前及更早任务，原队列之后会自动跳过已齐备的 part+manifest。合并仍以逐任务 manifest、唯一 key 和动作缓存逐 key 对齐为准，任何任务失败或重复写入都不得当作完成。

原先 node004 的奇数任务分片已成功结束。它腾出的 GPU 上又启动了与当前 node001 队列**任务名称不重叠**的尾部补充导出：`stamp_seal`、`rotate_qrcode`、`place_cans_plasticbox`、`place_fan`、`place_bread_basket`、`pick_diverse_bottles`、`place_dual_shoes`、`place_shoe`。它们均在原队列当前任务之后；必须在原队列抵达相应任务前写出完整 part+manifest，否则须停下并避免并发写同一路径。当前仍未合并物理缓存或启动物理训练。

全任务物理 EMA-VQ 与两种 FSQ 候选的匹配训练入口已备好：`scripts/physical_tokens/run_full50_physics_tokenizer_baselines.sh`。它要求合并缓存及 manifest 先存在，默认 256 个 token、K256、batch 32、20,000 steps、三组共用固定验证 episode，并拒绝覆盖部分运行目录；可用 `ROBOTWIN_PHYSICS_TRAIN_GPUS` 指定三卡，`ROBOTWIN_FULL50_FRAME_FEATURES_NPY` 指定从**该合并 NPZ**解出的只读 NPY，以避免每个进程重复解压大体量特征。脚本已通过 `bash -n`，但因全任务物理缓存未合并，**尚未启动训练**。训练 20,000 steps 是先固定相同更新预算；最终仍需检查验证曲线与训练覆盖率，再决定是否延长所有候选或作方法内调优。

全任务物理训练的 step 1 验证会扫描约 5 万个完整留出 clips，开销很大且模型尚未收敛；EMA/FSQ/iFSQ 正式脚本统一加 `--skip-initial-eval`，只从 step 5,000 开始按同一间隔验证，最终 step 20,000 必评估。该开关不改变训练数据、参数更新或最终评估口径。

为物理缓存合并后的多组训练准备了 `scripts/physical_tokens/prepare_full50_physics_frame_array.sh`：按合并 NPZ 的 SHA256 命名本机临时盘 `.npy`，流式解出 `frame_features.npy`，用合并 manifest 核验 shape/dtype/50 任务后原子落盘；同一节点可复用只读 memory-map。该 `.npy` 只作读取加速，不能替代原 NPZ 与其 manifest；脚本因合并缓存尚未生成而未执行。

此前 IFSQ 论文 `arxiv:2601.17124` 的 arXiv、export.arXiv、ar5iv 与 Hugging Face Papers 地址在训练节点均连接超时，因此当时没有实现或训练 IFSQ。随后用户提供了官方开源代码，见下一段；这一阻塞已解除，但仍没有三方法最终胜者。

上述网络阻塞已由用户提供的**官方开源实现**解除。代码位于 `/mnt/jfzn/liujn/code/iFSQ-main`；官方 README SHA256 `eed6d8d9a775b251a72f57645f0ed704358a01964ad8d7dacc419d0e260997cd`，关键源码 `ifsq/src/model/modules/fsq.py` SHA256 `817a6dcd15e6e90460327c6e460996edac15c818f6dcea277704fff08d04c554`。在 `do_simple_bound=true, act_fun=scale_sigmoid_16` 下，官方 `simple_bound(z)=(2·sigmoid(1.6z)−1)·(L−1)/2`，随后 round-STE；官方实现显式要求 levels 为奇数。当前项目的普通 FSQ 使用 `tanh(z)`；对奇数 levels 将其替换为上述激活后，混合进制 ID、归一化码值、解码与直通梯度仍保持原契约。用相同投影权重与随机输入直接对照官方 `GroupFSQ`，输出最大绝对差 `5.96e-8`、ID 完全一致、输入梯度最大绝对差 `4.66e-10`；相关项目测试 29 项通过（包括物理 tokenizer 反向梯度和动作 checkpoint round-trip）。

由于奇数 levels 的乘积不可能等于 `512=2^9`，严格官方 iFSQ 无法在单个混合进制码上做到与原 K512 **完全**等容量。动作实验选择 K513 `3,3,3,19`（+0.20%）、K525 `3,5,5,7`（+2.54%）与 K539 `7,7,11`（+5.27%）三组奇数 levels；每组均另跑**相同 levels 的普通 FSQ**，从而隔离激活函数的贡献；K525 另跑 EMA-VQ 做严格容量匹配。七组均为 50 任务相同 split、8 tokens、batch 256、seed 42、20,000 steps，入口 `scripts/physical_tokens/run_full50_action_ifsq_study.sh`，输出各自位于 `artifacts/robotwin_full50_study/action_tokenizers/`。训练已启动，尚无 20,000-step 结果；不能提前宣布 iFSQ 胜负。与 K512 原对照跨容量比较时，还需报告 active ratio、归一化 entropy 和重建/码分布的 Pareto 关系。

官方 README 报告约每个标量 4 bits 的效果平衡点，其图像配置用 `17,17,17,17`（K83521），远大于本任务固定的约 9 bits/token。故这里优先满足机器人 tokenizer 的近等容量公平对比；若 iFSQ 在 K≈512 下效果不佳，也不能直接推断其在论文的大容量图像设定下无效。

各 FSQ/iFSQ 成对实验不只共享超参数、数据划分和 seed：两者的可学习投影及其初始化路径也相同，量化器类只替换了 bounded activation，因此成对差异能更直接归因于论文的一行改动。全部动作检查点写出后，执行 `ROBOTWIN_ACTION_EVAL_GROUP=ifsq bash scripts/physical_tokens/run_full50_action_evaluation.sh` 对七组做完整留出集累计评估；当前检查点未齐，评估入口已验证会拒绝不完整对比。

若评估节点有七张可用卡，可给该入口设置 `ROBOTWIN_ACTION_EVAL_GPUS=0,1,2,3,4,5,6` 并行评估；不设置时顺序评估。脚本已通过 `bash -n`，检查点齐备前不会产生部分对比结果。

七组 20,000-step 动作检查点现均已完成训练，并用上述入口在相同的 55,797 个留出 chunks、250 个 task-episode、446,376 个离散 ID 上完成完整累计评估。下表每对 FSQ/iFSQ 具有**完全相同**的 levels、K、初始化路径、step 与训练数据；EMA K525 与 `3,5,5,7` 一对的容量严格一致。码指标为完整留出集 pooled 统计，不是 batch 平均。

| 方法 / levels | K | Smooth L1 ↓ | 关节 MAE ↓ | 有效码数 ↑ | perplexity ↑ | entropy(nats) ↑ | top10 占比 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSQ `3,3,3,19` | 513 | 0.012643 | 5.3189° | 501/513 | 255.16 | 5.5419 | 14.92% |
| iFSQ `3,3,3,19` | 513 | **0.012535** | 5.3673° | **502/513** | **269.20** | **5.5954** | **12.74%** |
| FSQ `3,5,5,7` | 525 | 0.011424 | **5.0677°** | 490/525 | 260.47 | 5.5625 | **13.41%** |
| iFSQ `3,5,5,7` | 525 | **0.011342** | 5.0826° | **505/525** | **266.86** | **5.5867** | 13.49% |
| FSQ `7,7,11` | 539 | 0.011219 | **5.0564°** | 510/539 | 263.50 | 5.5741 | 13.77% |
| iFSQ `7,7,11` | 539 | **0.010974** | 5.0845° | **520/539** | **271.68** | **5.6046** | **12.77%** |
| EMA-VQ `decay=.95, dead=100` | 525 | **0.004821** | **3.2473°** | 508/525 | **363.36** | **5.8954** | **8.57%** |

**成对结论**：iFSQ 在三个相同 levels 的对照中 pooled Smooth L1 均稍低（约 0.7%–2.2%），perplexity/entropy 与有效码数均稍高；但关节 wrapped MAE 三组均略高，top10 在 K525 反而略高。逐任务宏平均和胜场也表明收益并不均匀：iFSQ 的 Smooth L1 逐任务胜场分别为 25/50、25/50、29/50，关节 MAE 均为 24/50。因此不能说 iFSQ 在该机器人动作任务里全面优于同容量 FSQ；目前只可说这一行激活改动改善了 pooled 码分布，并带来较小的微平均重建收益。验证仅一个 seed，微小差距有待重复实验。

**三方法动作结论**：在容量严格匹配的 K525 对照里，EMA-VQ 的重建、关节误差，以及**全局 pooled** perplexity/top10 都明显优于 FSQ/iFSQ；与本轮所有六个 FSQ/iFSQ 候选逐任务取最优数值相比，EMA K525 的重建 Smooth L1、关节 MAE、速度 MAE 均在 50/50 个任务更低。这个结论限于离线动作 tokenizer 与当前模型/预算；不能外推到尚未完成的物理 tokenizer 或下游策略成功率。尤其 K539 iFSQ 的有效码 520 高于 K525 EMA 的 508，是不同分母，不能直接把 520 当作更好的码本利用率；相应比率为 96.47% 与 96.76%。

但**码分布存在 pooled 与逐任务口径反转**，不能只看上表的全局分布。对严格同 K525 的三组，50 任务码分布指标的等权宏平均如下（每个任务先在其全部留出 chunks 上统计，再平均）：

| K525 动作方法 | 逐任务有效码比率 ↑ | 逐任务 perplexity ↑ | 逐任务 top10 ↓ |
|---|---:|---:|---:|
| EMA-VQ | 10.77% | 28.67 | 64.37% |
| FSQ `3,5,5,7` | 15.21% | 33.91 | 59.42% |
| iFSQ `3,5,5,7` | 15.46% | 34.02 | 59.70% |

EMA 的 pooled perplexity `363.36` 虽明显高于 iFSQ 的 `266.86`，但逐任务 perplexity 却更低。这说明其全局码分布的均衡有一部分来自不同任务使用不同码（`H(C)=H(C|T)+I(C;T)`），不能解读为每个任务内部都更均衡。EMA 的重建优势在 50/50 任务成立；**码利用率优势只在 pooled/逐位置口径成立，不在逐任务宏平均口径成立**。逐位置 pooled 平均 perplexity 为 EMA `356.32`、FSQ `249.86`、iFSQ `256.74`。

七组的完整结果分别保存在各自 `artifacts/robotwin_full50_study/action_tokenizers/<run>/heldout_step20000.json`，含 RMSE、速度 MAE、夹爪准确率、归一化 entropy、逐任务与逐位置统计。最终 checkpoint SHA256：iFSQ K513 `744e23888606cfd4cb507976a38bb1c376bd0a679b88192f21a8d56f8327c5e7`、K525 `23f58564ac5f2a4d83d1c2fe831f03ced2f5293b7f6f79f9a2a0d7a464e5a47c`、K539 `c48dcbd0ded86c866e5d37290c68881ddbe286dd38cfe23767384f6466b92c9a`；FSQ K513 `6cb32c45030ee35bb001d7e36275ba0338f2363f99662819c1a91dc9143d110f`、K525 `1fecbee052b3c8367c8df2337946d18da77df9336633d605ba2e0a6912a6c81b`、K539 `cda49c632b9f8e4df93cd9fb022972bf63136c5c4f8e797cb7f6c3d227f23705`；EMA K525 `4acd681da24aeaf0ef76c6d5c137dda82623380b6b008fd476a46e32649f43f8`。

物理 iFSQ 的匹配训练入口也已准备：`scripts/physical_tokens/run_full50_physics_ifsq_study.sh`，在相同 256 token、batch 32、20,000-step、aux=0 协议下比较 iFSQ/FSQ 成对 levels `3,5,17`（K255）、`5,7,7`（K245）、`5,5,11`（K275）。它与此前 K256 的 EMA/FSQ baseline 脚本互不覆盖。该脚本必须等全 50 任务物理缓存合并并通过对齐校验后才可运行；目前**未启动物理 iFSQ**。

2026-10-25 后续检查：8 任务、2-step 的物理 iFSQ **入口冒烟测试**已成功写出 `ifsq_smoke_8tasks_levels3517_steps2/step-00000002.pt`（约 32 MB）及两条验证指标；它只证明模型前向、反向、评估和 checkpoint 保存可运行，不能作为收敛或全任务性能结论。全任务物理分片目前新增 27 个任务，加上原 D28 可复用的 8 个，合计 **35/50**；尚缺 15 个任务，导出进程仍在运行。`run_full50_physics_tokenizer_baselines.sh` 和 `run_full50_physics_ifsq_study.sh` 均已启用 `--skip-initial-eval`，并通过 `bash -n`；待合并缓存逐 key 验证后才启动正式训练。

物理九组正式模型的完整留出集评估入口已新增 `scripts/physical_tokens/run_full50_physics_evaluation.sh`：先确认九份最终 checkpoint 全齐，再顺序写各自 `heldout_step20000.json`，保留 pooled、50 任务宏平均/逐任务及 256 个位置的码分布；已有报告不覆盖。脚本语法检查通过，当前按预期因全任务合并缓存不存在而拒绝运行。相关定向测试 29 项通过；仅有 pytest 缓存目录无写权限的警告，不影响测试结果。

2026-10-25 23:49（北京时间）进度复查：仍为 35/50 个完整物理分片；剩余 15 个中，多数导出日志在持续增加，`hanging_mug`、`put_bottles_dustbin`、`stack_blocks_three` 尚未开始。当前节点多卡显存/算力被其他训练与特征导出共享，故不能用历史独占 GPU 速度作硬截止时间。首轮九组全量物理离线结果预计还需约 8–16 小时（包含分片完成、合并校验、训练、完整留出评估），这是有资源竞争和数据导出失败风险的粗估；额外调优、重复 seed 或下游 rollout 不包含在内。

用户确认 node1/node2/node4 全部 GPU 均可用于本研究后，复查已有分片提升至 **37/50**。剩余大多数任务已有独立导出进程，占用 node1/node4 多卡；继续添加相同任务进程会竞争同一个输出文件，故不重复启动。node2 八卡各有约 29–32 GiB 空闲显存，因此在 node2 GPU0–5 启动动作 K525 的第二、第三随机种子（43/44）复验：每个 seed 各跑 EMA-VQ、FSQ `3,5,5,7`、iFSQ `3,5,5,7`，六组均使用固定原留出 episode、20,000 steps，并在训练成功后自动做完整留出评估。入口 `scripts/physical_tokens/run_full50_action_k525_seed_repeats.sh`，启动日志 `artifacts/robotwin_full50_study/action_k525_seed_repeats_node002.log`；此处只是已启动，**不能把重复 seed 的结果当成已完成**。物理全量训练仍须等 50/50 分片合并通过 key 对齐，届时优先利用 node2 空闲卡并按显存余量分批启动。

2026-10-26 动作重复 seed 的六组训练及完整留出评估均已完成。K525、相同划分和 20,000-step 下，seed43/44 的 pooled Smooth L1 分别为：EMA `.005093/.004925`，FSQ `.011885/.011256`，iFSQ `.011882/.011655`；关节 wrapped MAE（度）依次为 EMA `3.308/3.268`，FSQ `5.125/5.081`，iFSQ `5.129/5.188`。配合 seed42，可确认 **EMA 动作重建优势在三种种子均明显**；iFSQ 对同 levels FSQ 的微小重建收益却不稳定（seed44 反而更差），不能把 seed42 的提升视作稳定方法效应。各自 `heldout_step20000.json` 保留 perplexity、active codes、entropy、top10、逐任务和逐位置统计。

同日 10:28（北京时间）物理导出达到 **50/50 完整分片**，所有导出进程结束，未见 traceback/OOM；已在 node1 启动 `ROBOTWIN_PHYSICS_MERGE_ONLY=1`，读取 50 分片并与动作缓存逐 key 对齐，尚未完成合并。为避免合并结束后空等，node2 启动有 360 分钟超时的只读就绪等待器 `scripts/physical_tokens/run_full50_physics_after_merge_node2.sh`；仅在合并 NPZ+manifest 存在且 manifest 确认 50 任务和精确动作 key 对齐后，才会于 node2 本机解出验证过的只读 NPY，然后顺序执行三组 EMA/FSQ baseline、六组成对 FSQ/iFSQ、九组完整留出集评估。运行日志为 `artifacts/robotwin_full50_study/physics_after_merge_node002.log`。**当前只能说自动流水线已排队，不能声称物理模型已经开始训练或完成。**

10:37 复查：合并进程仍正常读取分片，约 9 分钟才完成首个 `adjust_bottle`（520 MB 压缩文件），因此原 6 小时等待上限可能不足。已明确停止仅处于等待状态的旧 node2 watcher（PID 311334），重新启动唯一的 24 小时等待器（PID 311958，日志 `artifacts/robotwin_full50_study/physics_after_merge_node002_longwait.log`）；训练入口、数据校验和输出路径不变。此时合并仍未完成，不能沿用此前“上午完成”的估计；需按后续分片吞吐重新更新 ETA。

合并慢的原因之一已定位为 `merge_multitask_visual_caches.py` 对同一 NPZ 最大成员 `frame_features` 在 shape、dtype 与 `.copy()` 三处重复读取/解压。已改为每分片仅解压一次、复用该独立 NumPy 数组，不改变特征值或 key 校验契约；语法与 `git diff --check` 通过。确认旧合并进程尚未写出任何目标/临时文件后，只终止该旧进程（PID 3563110），并用同一 `MERGE_ONLY` 入口重启优化版。node2 的唯一等待器继续等待。因这次为节省多小时 I/O 而重启，ETA 仍需等优化版实际处理速度再校准。

## 5. 全 50 任务物理 tokenizer 首轮完整结果（2026-10-26）

50 个分片已合并为 `artifacts/robotwin_full50_study/physics_all50_spatial4x4_17frames.npz`（39 GB）；manifest 明确记录 50 任务、549,787 个视觉/动作精确对齐的 key、`[549787,16,2560]` 的独立帧特征数组及动作缓存 SHA256 `3be69db04833df756680d9e62847e0be497324238f6cc4c9eafd0b4fb9a29865`。每个训练/验证样本取 17 帧中的 16 个相邻采样帧视觉特征差，完整 clips 共 421,990 训练 + 47,797 留出；其余 key 因 episode 尾部未来帧不完整，不进入物理 tokenizer 的训练/评估。所有九组：256 个物理 token、spatial-aligned routing、batch 32、seed 42、20,000 steps、相同留出 episode `5,20,25,39,49`、action auxiliary weight 0。node2 流水线于 16:15（北京时间）完成九个最终 checkpoint 的完整留出评估。以下 pooled 指标均由相同 47,797 clips 计算；RMSE 与 standardized Smooth L1 越低越好，perplexity/entropy 越高、top10 越低一般表示码更均衡。

| 物理方法 / levels | K | RMSE ↓ | 逐任务宏 RMSE ↓ | standardized Smooth L1 ↓ | 有效码 | perplexity ↑ | entropy(nats) ↑ | top10 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA-VQ | 256 | **0.082253** | **0.083273** | **0.327602** | 256/256 | **245.16** | **5.5019** | **7.16%** |
| FSQ `4,8,8` | 256 | 0.085139 | 0.086141 | 0.345447 | 256/256 | 183.95 | 5.2147 | 10.64% |
| FSQ `4,4,4,4` | 256 | **0.084305** | **0.085282** | **0.339994** | 256/256 | 208.05 | 5.3378 | 13.51% |
| FSQ `3,5,17` | 255 | 0.085211 | 0.086232 | 0.346151 | 255/255 | 183.03 | 5.2097 | 11.88% |
| iFSQ `3,5,17` | 255 | 0.085220 | 0.086239 | 0.346180 | 255/255 | 183.29 | 5.2111 | 11.76% |
| FSQ `5,7,7` | 245 | 0.085141 | 0.086140 | 0.345384 | 245/245 | 182.51 | 5.2068 | 10.50% |
| iFSQ `5,7,7` | 245 | 0.085141 | 0.086140 | 0.345384 | 245/245 | 182.33 | 5.2058 | 10.51% |
| FSQ `5,5,11` | 275 | 0.085052 | 0.086063 | 0.345052 | 275/275 | 196.29 | 5.2796 | 9.92% |
| iFSQ `5,5,11` | 275 | 0.085063 | 0.086072 | 0.345093 | 275/275 | 196.73 | 5.2818 | 9.88% |

同 K256 的最好 FSQ (`4,4,4,4`) 相比，EMA-VQ pooled RMSE 低 `0.002052`（约 2.43%），宏 RMSE 低 `0.002009`；连续瓶颈 RMSE 也为 EMA `0.074885` 对 FSQ `0.083982`。EMA 对表中每个 FSQ/iFSQ 候选，逐任务 RMSE 均在 **50/50** 个任务更低；即便每任务单独取八个非 EMA 候选的最好值，EMA 仍在 50/50 任务更低。所有模型在 pooled 留出集均用到全部组合码，因此“有效码/K”此时没有区分力；pooled perplexity、entropy、top10 与逐任务/逐位置统计更有意义。EMA 的逐任务宏 perplexity 为 `221.49`，FSQ `4,4,4,4` 为 `202.33`；逐位置平均 perplexity 分别为 `215.72/197.09`，也支持 EMA 在本轮物理实验的码分布优势。表中的 standardized Smooth L1 与 RMSE 定义相同，可同模态比较；**总 loss 不应直接排名**，因为 EMA 额外包含 commitment 项。

三对**完全同 levels、同 K**的 FSQ/iFSQ 中，iFSQ 的 pooled RMSE 差（iFSQ−FSQ）分别为 `+0.00000816`（K255）、约 `−0.00000008`（K245）、`+0.00001075`（K275）；iFSQ 逐任务 RMSE 胜场分别为 `10/50`、`29/50`、`8/50`。差异太小且方向不一致，不能宣称 iFSQ 在这个物理特征差分任务中优于同配置 FSQ。`perplexity=exp(entropy)` 不是独立第二份证据；不同 K 的 active 数量不直接互比。九份原始报告在 `artifacts/robotwin_full50_study/physics_tokenizers/<run>_steps20000/heldout_step20000.json`，包括 checkpoint SHA256、pooled、逐任务宏平均、50 个任务和 256 个位置的全部指标；训练配置与 5k/10k/15k/20k 验证曲线在相同目录。所有候选的重建曲线总体仍有小幅改善，因此 20,000 step 是共同首轮预算，不应等同“充分训练至每法最优”。

**阶段性判断**：在本轮已训练的动作、物理 tokenizer 离线重建上，EMA-VQ 都领先；动作 K525 的三随机种子优势稳定，物理目前只有一个 seed，且尚未做方法内更充分调优、下游 token predictor 或策略 rollout，所以不把 tokenizer 离线领先改写成“最终机器人效果最好”。

### 码本扩容与评价口径复核（2026-10-26）

用户指出物理 K 主要约 256 且 pooled 有效码数/K 为 100%。这是扩容的合理动机，但 **100% 只表示 47,797 clips × 256 token 中每个 ID 至少出现一次，不足以证明码本饱和或 K 越大越好**。例如 EMA K256 的 perplexity 245.16 接近上限，FSQ `4,4,4,4` 为 208.05；还须看 K 增大后重建误差是否下降、每位置及每任务的码分布是否均衡，以及实际编码率是否值得。容量不同的 perplexity 原值不直接排序：同时报告 normalized entropy、`log2(K)` 固定长 bits/token 和 `H/log(2)` 经验熵 bits/token；后者不等于已实现的实际熵编码文件大小。

扩容面板入口为 `scripts/physical_tokens/run_full50_physics_large_codebooks_node1.sh`，独立输出目录且拒绝覆盖。约 K512 档：EMA K512、FSQ `4,4,4,8` K512、FSQ/iFSQ 同 levels `3,3,3,19` K513；约 K1024 档：EMA K1024、FSQ `4,4,4,4,4` K1024、FSQ/iFSQ 同 levels `3,7,7,7` K1029。每档 FSQ/iFSQ 逐 levels 成对比较，EMA 与标准 FSQ 则同 K；两档均沿用全 50 任务同缓存、同完整 episode 留出、256 tokens/clip、seed42、batch32、20,000 steps 和相同网络。面板在 node1 GPU4–7 上等待已有 levels 扫描释放资源，先跑 512 再跑 1024；**在最终留出 JSON 落盘前，这些只是排队实验，不能写成扩容有收益**。后续如有容量效应，再对优选大 K 做重复种子与训练预算复核。

资源重新安排（同日）：用户允许 node1–4 各卡与其他任务共享。node1 GPU4–7 已用于 levels 扫描，原大 K 等待器仍未创建任一输出目录，因此终止了**仅排队的等待器**（PID 3702638），没有中断训练或删除结果。新分节点面板入口 `scripts/physical_tokens/run_full50_physics_large_k_panel.sh`，保持上述模型配置、目录名、数据划分与评估完全不变；只改变执行节点和并行时序。node2 最初尝试 K≈512，但启动前 GPU0 可用显存从约 16 GiB 降至 7 GiB，12 GiB 安全门槛自动拒绝，未创建模型输出。随后将 K≈512 安排到 node4 GPU1–4（日志 `artifacts/robotwin_full50_study/physics_large_k512_node004.log`），K≈1024 安排到 node3 GPU0–3（日志 `.../physics_large_k1024_node003.log`）。两节点原先都无本地 NPY，现先通过同一 SHA 命名的只读特征数组准备脚本从已验证合并 NPZ 解出缓存；**截至该记录，两组进程已进入缓存准备，尚未看到训练启动，更没有新 K 的验证结果**。node2 原 seed43/44 复验、node1 levels/EMA 调优保持不变；node4/node3 使用不同新目录，不与它们冲突。启动前每张卡要求至少 12 GiB 空闲显存，若资源继续波动导致失败，应看面板日志及各训练日志，不把排队视为完成。

17:28（北京时间）node3 的 K≈1024 四组、17:32 node4 的 K≈512 四组均完成 42 GB 本地 NPY 准备和形状/类型校验，面板日志已逐组给出训练 PID；复查进程仍存活、GPU 显存有余量。此处是**训练已启动**，不是已收敛或已有完整留出结果；20,000-step checkpoint 与 `heldout_step20000.json` 仍须等待。node1 和 node2 原实验保持运行。

PhiZero 视频解码器下载提示：官方 `PhiZero-main/README.md` 要求 `ShuyaoShang/PhiZero` 的 `ckpt/tokenizer/tokenizer.safetensors` 与同目录 `model_config.json`（内含 Physical Language 模块、inverse-FSQ conditioner 和渲染 LoRA），再加 `Wan-AI/Wan2.2-TI2V-5B` 三个 `diffusion_pytorch_model-0000*-of-00003.safetensors` DiT 分片、`DiffSynth-Studio/Wan-Series-Converted-Safetensors` 的 `Wan2.2_VAE.safetensors`。但该渲染器与 PhiZero 自己的 token ID、levels、conditioner、Wan 视觉 latent 契约绑定；**只下载权重不能直接解码本研究 Qwen3-VL 特征差分 tokenizer 的 EMA/FSQ/iFSQ ID**。若要用生成视频公平比较三方法，需要为各自训练或适配兼容的视频 conditioner/decoder，并固定生成与评估协议；这是单独的下游实验，不得把 PhiZero 预训练渲染器直接套用后声称模型优劣。

对照官方开源实现：`iFSQ-main/ifsq/eval_ddp.py` 调用 `train_ddp.py::valid`，图像重建报告 L1、PSNR、LPIPS、SSIM、FID；`src/model/losses/perceptual_loss.py::measure_perplexity` 还给出码 ID 的 perplexity 和访问过的 cluster 数。这里重建对象是 Qwen3-VL `[17,16,2560]` 多帧特征的 16 组**差分**，没有把 token 解码回 RGB 图像的模型。因此 PSNR/SSIM/LPIPS/FID 的像素定义不适用；不把特征 MSE 伪称图像 PSNR。保留原差分单位 RMSE、标准化 Smooth L1、cosine、无运动基线、quantized-vs-continuous gap、全局/逐任务/逐位置 active/entropy/perplexity/top10；新增“差分累积后未来特征 RMSE”“末帧特征 RMSE”和起始帧冻结的未来特征基线。后两类检验逐步预测误差是否累积漂移，仍是**特征空间离线指标**，不是像素保真度或控制成功率。上述完整评估只使用 `valid_mask.all()` 的 clips，故所有帧有效；pooled 重建 MSE 先按元素/clip 累计再开根号，任务宏 RMSE 是 50 个任务各自 RMSE 的算术平均，不能混用。`perplexity=exp(entropy)` 仍不是独立证据。新指标在评估 schema v3 的 JSON 中，旧报告为 v2，比较新旧配置时须以同一版评估器重评。

#### PhiZero 开源实现的进一步指标核对

用户指定的 `/mnt/jfzn/liujn/code/PhiZero-main` 已检查其**本项目代码**，而非只看打包在仓库中的通用 DiffSynth/MS-Swift 指标模块。关键位置：`DiffSynth-Studio/diffsynth/models/phizero/tokenizer.py`、`diffsynth/training/phizero.py`、`DiffSynth-Studio/examples/phizero/train.py`、`benchmarks/physical_iq/run.py`。PhiZero 官方 tokenizer 接收视频，经相邻帧 QFormer 与 FSQ 输出固定 256 个物理符号；其 reasoner 词表符号范围是 `0..24999`（README 的 SFT 约束），与本实验的 256 **token 位置**、约 256 **ID 数**是两个不同轴，不能由 PhiZero 的 25,000 直接推出本项目必须用 25,000 个码。

PhiZero 发布的训练脚本报告的主要标量是 **首帧给定时未来视频 latent 的 flow-matching MSE**：`prediction[:, :, 1:]` 对 `target[:, :, 1:]`，乘 timestep 权重；启用 `--train-tokenizer --fsq-entropy-weight` 后再加 FSQ 的 soft entropy 正则。该正则在 `tokenizer.py::_FiniteScalarCodeEncoder.soft_entropy_loss` 中约为 `(逐样本 soft entropy 之和 − gamma × batch joint entropy) / log(K)`，是**训练目标/可诊断量**，不是留出集离散 ID 的经验 entropy/perplexity，二者不可互换。PhiZero 的最小发布训练器只显示总 loss，未见逐 batch 输出 perplexity、有效码率、top10 或视频 PSNR 等 tokenizer 验证报表；不能把第三方库里存在的通用图像指标误写成 PhiZero 正式物理 tokenizer 指标。

PhiZero 的最终视频级评估另走 `benchmarks/physical_iq/run.py` 调用外部官方 Physics-IQ Verified scorer，输入是 reasoner+视频渲染器的生成视频，输出 `final_score_view`。这是有物理意义的下游评估，但**当前 RoboTwin 特征差分 tokenizer 没有对应视频生成解码器，也没有接入 Physics-IQ 案例**，故不能直接产出可比的 Physics-IQ 分数。现阶段的主指标仍应是同一验证划分上的差分 RMSE + 累积未来特征误差，配合固定容量/经验码率与逐任务/逐位置码分布；最终方法选择仍须 token predictor/机器人策略 rollout。PhiZero 的熵正则值得作为 FSQ/iFSQ 之后的独立超参数消融，而非把其训练 loss 与 EMA commitment loss 直接相互比较；正在运行的已固定配置实验不因这次审计而改动。

为检验物理结果的 seed 稳定性，16:54（北京时间）在 node2 启动第二、第三随机种子复验，入口 `scripts/physical_tokens/run_full50_physics_seed_repeats_node2.sh`：每个 seed 各重训 EMA K256、首轮最佳 FSQ `4,4,4,4` K256、以及严格同 levels/K 的 FSQ/iFSQ `3,5,17` K255；仍为相同 50 任务、相同固定留出 episode、256 tokens、batch 32、20,000 step、aux=0，并在每轮训练后对全部留出 clips 做逐任务/逐位置评估。输出目录增加 `_seed43`/`_seed44`，不会覆盖首轮；进度日志为 `artifacts/robotwin_full50_study/physics_seed_repeats_node002.log`。**截至此记录，复验只是已启动，尚无可报告的正式结果。**

为了不让 iFSQ 全部只有 3 个量化标量、却把 4 维 FSQ 当作最优对照，node1 另启动相同 seed42/20,000-step/数据划分的物理 levels 扫描：FSQ `2,2,2,2,2,2,2,2` K256、FSQ `2,2,4,4,4` K256，以及两对相同 levels/K 的 FSQ/iFSQ `3,3,3,9` K243 与 `3,3,3,3,3` K243。六组 levels 均已通过量化前向、ID 范围和反向有限梯度的定向检查。入口 `scripts/physical_tokens/run_full50_physics_level_sweep_node1.sh`，进度日志 `artifacts/robotwin_full50_study/physics_level_sweep_node001.log`；先在 node1 本地解出经源 SHA/shape 校验的只读 frame NPY，再分两波训练、评估。这个扫描扩充了方法内配置，不改变先前九组已完成结果；**新六组仍在运行，不能提前用它们排名。**

物理 EMA 的专有超参数也需公平调优：node1 已另排两组 seed42、同 20,000-step 预算的 EMA K256 候选，`decay=.95, dead=100, commitment=.25`（动作端领先设置）与 `decay=.99, dead=200, commitment=.5`（提高 commitment 权重）。入口 `scripts/physical_tokens/run_full50_physics_ema_tuning_node1.sh`，日志 `artifacts/robotwin_full50_study/physics_ema_tuning_node001.log`；它只会在 node1 的同源 SHA 命名、验证后的本地 NPY 原子落盘后开始 GPU1/3 训练，避免重复解压合并缓存。这两组尚无结果。

### 17:49 进度核验：扩容早期点与重复 seed（北京时间）

node1 的 FSQ `2,2,2,2,2,2,2,2`、`2,2,4,4,4` 及 FSQ/iFSQ `3,3,3,9` 四组第一波均已完成 20,000 step，正在执行完整留出集逐任务/逐位置评估；第二波 `3,3,3,3,3` 尚未开始。node1 两组 EMA K256 专有调优也都已完成 20,000 step、正在独立评估。以下新组的 step20k 数值当前仅取训练器的全留出集 pooled 验证，待独立 JSON 落盘再确认。EMA `decay=.95, dead=100` 的 RMSE `0.082219`，默认 EMA 为 `0.082253`，改善仅 `0.000034`，不应过度解读；EMA `commitment=.5` 为 `0.082307`。FSQ `2,2,4,4,4` 为 `0.083900`，比此前 K256 最佳 FSQ `4,4,4,4` 的 `0.084305` 低；需核对独立评估和逐任务分布。

node2 seed43 的四组 **20,000-step 完整独立留出集**已落盘：EMA K256 RMSE `0.082227`、FSQ `4,4,4,4` K256 `0.084298`、FSQ/iFSQ `3,5,17` K255 分别 `0.085223/0.085225`。配合 seed42 (`0.082253/0.084305/0.085211/0.085220`)，EMA 重建领先在这两个 seed 均成立；同 levels iFSQ 未显示实质改善。seed44 四组正训练，未加入稳健性结论。seed43 的未来特征 RMSE 依次为 `0.148835/0.150624/0.152238/0.152337`；它是新 schema v3，旧 seed42 独立 JSON 尚无此字段。

node3 K≈1024 与 node4 K≈512 各四组正在训练。为避免把不同 step 当最终排名，先列**共同 step5,000 全留出集 pooled 验证**（不是独立逐任务评估，也不是最终 20,000-step 结论）：

| 方法/容量 | K | 5k 差分 RMSE ↓ | 5k perplexity | 5k top10 |
|---|---:|---:|---:|---:|
| EMA 基线 | 256 | 0.082467 | 247.05 | 6.78% |
| EMA 扩容 | 512 | 0.081448 | 488.08 | 3.99% |
| EMA 扩容 | 1024 | 0.080660 | 973.02 | 2.21% |
| FSQ `4,4,4,4` | 256 | 0.084468 | 205.83 | 13.87% |
| FSQ `4,4,4,8` | 512 | 0.083922 | 364.40 | 8.27% |
| FSQ `4,4,4,4,4` | 1024 | 0.083012 | 742.50 | 5.94% |

EMA 在相同网络/步数下随 K 增大，早期差分 RMSE 明确下降；FSQ 行同时改变标量维数与 levels，不能把所有收益纯归因于 K。perplexity 原值随 K 上限变大，不可当作独立容量收益。对应同 levels 的奇数码 FSQ/iFSQ：K513 step5k RMSE `0.084231/0.084235`，K1029 `0.083659/0.083732`；iFSQ 早期并未更低，仍须等完整 20k。未来特征 RMSE 在新的扩容模型 step5k 可见（EMA K512/K1024 为 `0.147181/0.145299`），但历史 K256 5k JSON 是旧 schema，缺该字段，不能将缺值当 0 或用不同训练步数冒充公平比较。

为补齐基线的新指标，17:53 在 node3 GPU5/7 启动旧 seed42 EMA K256 与 FSQ `4,4,4,4` K256 最终 checkpoint 的 **schema v3 独立重评**，入口 `scripts/physical_tokens/reevaluate_full50_physics_k256_schema3_node3.sh`，输出到原目录的新文件 `heldout_step20000_schema3.json`，不覆盖 v2 报告或 checkpoint。已核实两个评估 PID 存活，仍待 JSON 落盘；两组沿用相同 47,797 留出 clips，可以与最终扩容模型的未来特征指标按相同步数比较。评估期间 node3 GPU0–3 的 K≈1024 训练继续运行。

### 全部扩容面板和复验的最终结果（2026-10-26，18:39 后核对）

node3 K≈1024 面板于 18:33 完成，node4 K≈512 于 18:15 完成；node1 levels 扫描于 18:38 完成，node2 seed43/44 于 18:13 完成，旧 K256 两个 checkpoint 的 schema v3 独立重评于 18:01 完成。两组 EMA K256 调优的独立 JSON 亦已落盘。已核对以下全为同一 50 任务、47,797 个完整留出 clips、256 tokens/clip、seed42（除注明重复 seed）、20,000-step checkpoint 的**独立逐任务评估**，非训练日志的中间点。

| 物理配置 | K | 差分 RMSE ↓ | 任务宏 RMSE ↓ | 累积未来特征 RMSE ↓ | perplexity | normalized entropy | 有效码 | top10 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA-VQ 基线 | 256 | 0.082253 | 0.083273 | 0.148810 | 245.16 | 0.9922 | 256/256 | 7.16% |
| EMA-VQ | 512 | 0.081241 | 0.082286 | 0.146777 | 488.48 | 0.9925 | 512/512 | 4.09% |
| EMA-VQ | 1024 | **0.080381** | **0.081446** | **0.144942** | 973.21 | 0.9927 | 1024/1024 | **2.19%** |
| FSQ `2,2,4,4,4` | 256 | 0.083900 | 0.084875 | 0.150375 | 214.60 | 0.9682 | 256/256 | 10.73% |
| FSQ `4,4,4,8` | 512 | 0.083726 | 0.084717 | 0.147966 | 362.24 | 0.9445 | 512/512 | 8.09% |
| FSQ `4,4,4,4,4` | 1024 | **0.082878** | **0.083861** | **0.146952** | 735.07 | 0.9522 | 1024/1024 | 5.93% |
| FSQ / iFSQ `3,3,3,19` | 513 | 0.084143 / 0.084146 | 0.085145 / 0.085147 | 0.149732 / 0.149764 | 362.35 / 361.58 | 0.9443 / 0.9439 | 503/513 各 | 7.46% / 7.52% |
| FSQ / iFSQ `3,7,7,7` | 1029 | 0.083500 / 0.083529 | 0.084496 / 0.084526 | 0.148282 / 0.148626 | 658.81 / 665.22 | 0.9357 / 0.9371 | 1029/1029 各 | 4.88% / 4.81% |

同 EMA 架构容量效应清晰：K512 的 RMSE 低于 K256，K1024 又低于 K512，**每一步都在 50/50 个任务降低 RMSE**；K256→1024 pooled 差分 RMSE 降约 2.28%，但固定码长预算增加 2 bits/token（8→10），需要按质量–码率权衡，不能只看最大 perplexity。与同 K1024 的最佳已测 FSQ 相比，EMA 差分 RMSE 低约 3.01%，且 50/50 任务更低。FSQ 的 K1024 也优于自身 K512，在 50/50 任务更低；K256 最优 levels 已变为 `2,2,4,4,4`，但 K256 与 K512 的 FSQ 同时改变 levels 结构，不能把全部差异归因于 K。K1024 的 EMA 逐位置有效码平均约 1014.6/1024、逐位置平均 perplexity 约 801.8；“pooled 100% active”不能误说每位置全满。FSQ/iFSQ 的同 levels 配对在 K513 基本持平，K1029 上 FSQ 的 RMSE 在 50/50 任务略低，但绝对差仅 `0.000029`，不应当作跨种子稳定方法效应。

码率字段注意：当前 schema v3 JSON 的 `fixed_bits_per_token` 实际计算 `log2(K)`，这是容量信息下界，**非二次幂 K 时不是真正逐 token 固定长度位数**；实际固定长度应为 `ceil(log2(K))`（例如 K513 为 10 bit、K1029 为 11 bit）。`effective_bits_per_token=H/log(2)` 则是单 token 边际经验熵，不是已经测得的压缩文件大小，也忽略时间/空间相关。上文 8→10 bit 仅用于 K256→K1024 的二次幂容量对照，定义无歧义；修订评估器和报告版本之前不将非二次幂的旧 `fixed_bits_per_token` 当成实际存储成本。

K256 的 EMA/FSQ `4,4,4,4` 在 seed42/43/44 上差分 RMSE 分别为 `0.082253/0.082227/0.082216` 对 `0.084305/0.084298/0.084299`，三 seed 的 EMA 优势稳定；FSQ/iFSQ `3,5,17` 的 seed43/44 RMSE 分别为 `0.085223/0.085233` 对 `0.085225/0.085222`，差异接近零且方向变化。EMA 专有调优 `.95/dead100` 的 RMSE `0.082219`、commitment `.5` 为 `0.082307`，相对默认 `.99/dead200` 的 `0.082253` 改善/退化都很小。新 levels 中 FSQ `3,3,3,3,3` K243 RMSE `0.084142`，iFSQ `0.084152`；FSQ `3,3,3,9` `0.084348`，iFSQ `0.084345`。以上进一步支持目前已测离线配置中 **EMA-VQ 的重建优势**，但不是策略 rollout 或视频生成效果的证明。

原始结果均在 `artifacts/robotwin_full50_study/physics_tokenizers/<run>_steps20000/heldout_step20000.json`；最初 K256 EMA 和 FSQ `4,4,4,4` 的新指标在同目录 `heldout_step20000_schema3.json`，旧 v2 JSON 未覆盖。当前 K1024 EMA pooled/per-position 码分布仍较接近满使用，说明可探索更大 K，但每增加一档至少多约 1 bit/token；下一档应作为质量–码率扫描，不能预设越大越好。物理 tokenizer 尚没有 RGB 视频重建损失、动作监督（本轮 aux=0）或策略 rollout，离线排名不得称为最终机器人效果排名。

20:16（北京时间）据上述 K1024 仍有较高有效使用且 RMSE 在 50/50 任务优于 K512，启动新一档**探索性** K≈2048（node3 GPU0–3，日志 `artifacts/robotwin_full50_study/physics_large_k2048_node003.log`）：EMA K2048、标准 FSQ `4,4,4,4,4,2` K2048，以及严格同 levels 的 FSQ/iFSQ `3,3,3,3,5,5` K2025。沿用相同 50 任务、训练/验证划分、seed42、batch32、256 token、20,000 step 和 schema v3 独立评估，四组各有独立新目录、拒绝覆盖。节点已复用此前同源 SHA 命名并通过 manifest 形状/类型校验的本地特征 NPY，已核实四个训练 PID 启动；**这不是已完成结果**。K2025 与 K2048 的真实固定长度都是 11 bit/token，但 FSQ/iFSQ 的 levels 不同，跨组需明确近容量而非同构比较；优先看同 levels FSQ/iFSQ 成对差异及 K2048 EMA 对 K1024 的收益是否递减。

用户要求先找到“全局有效码/K 不再是 100%”的容量。**现成已完成样例**：FSQ 和 iFSQ 的 `3,3,3,19` K513 均为 `503/513=98.05%`；这仅是该 levels 的观察，不是所有方法或所有 K 的统一阈值。EMA 已完成的 K256/512/1024 全部 100%；EMA K1024 的逐位置有效码平均 `1014.6/1024`，说明换成逐位置口径其实已不是 100%。K2025/2048 的 FSQ/iFSQ 在 step5,000 pooled 仍 100%，说明“非满利用”并不随 K 单调变化，受 levels 和训练状态影响。为找 EMA 的全局非满利用区间，20:32–20:33 在 node4 GPU1–4 另启动 K4096/8192/16384/32768 四组相同 seed42、完整 20,000-step **容量扫描**，入口 `scripts/physical_tokens/run_full50_physics_ema_utilization_scan_node4.sh`，日志 `artifacts/robotwin_full50_study/physics_ema_utilization_scan{,_high}_node004.log`；每组新目录拒绝覆盖，训练成功后自动独立评估同一完整留出集。已核实四个训练进程启动；**尚无这四档的有效码结果**。码本越来越大时计算和显存开销也增大，先用 K4096/8192/16384/32768 定位已测区间，若仍 100% 则只报告“在已测范围内未找到”，不臆造阈值。即使出现 99.99%，也不应把这个任意采样阈值当作最优码本；仍须联看失真、normalized entropy、码率和逐位置分布。

随后 K2048 EMA 的 step5,000 pooled 验证也达到 `2048/2048`，同阶段 K2048 标准 FSQ 和 K2025 FSQ/iFSQ 均为各自 `K/K`；这些是训练中间点，须等最终 20,000-step 独立评估确认。K513 的 `503/513` 已说明 FSQ/iFSQ 存在非满利用配置，但不能由此推出 EMA 的临界 K。

20:42 再核查：node3 的 K2048 EMA、K2048 FSQ、K2025 FSQ/iFSQ 四组均已保存 `step-00010000.pt`，四个主训练进程仍运行；step10,000 pooled 验证有效码分别为 `2048/2048`、`2048/2048`、`2025/2025`、`2025/2025`。其中 EMA K2048 step10,000 差分 RMSE `0.079628`、perplexity `1937.78`、future-feature RMSE `0.143535`；仍是**中途验证，不是最终完整留出评估**。node4 的 EMA K4096/8192/16384/32768 也均已保存 `step-00010000.pt`、四个主训练进程仍运行；这些实验设为只在 step20,000 训练验证，之后再做独立完整留出评估，故当前没有可信的这四档 pooled 利用率或重建指标，不能据空白 `train.log` 推断停滞。现阶段“EMA 首个非满利用 K”仍未确定。

### 20,000-step 高 K 扫描最终结果（21:21 北京时间核对）

上述八组现均已完成独立完整留出集评估：相同 50 任务、47,797 clips、每 clip 256 tokens（总计 12,236,032 token），评估 schema v3。下表的有效码和 perplexity 都是将全部任务、全部位置 pooled 后计算；RMSE 是冻结视觉特征的差分重建误差，非像素或视频质量。前面的“运行中”记录保留作时间线，以本表最终结果为准。

| 方法 | K | 差分 RMSE ↓ | 未来特征 RMSE ↓ | 重建 Smooth L1 ↓ | 有效码/K | Perplexity | 归一化 entropy | Top10 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EMA-VQ | 2048 | 0.079519 | 0.143288 | 0.311395 | 2048/2048 | 1933.75 | 0.99247 | 1.258% |
| EMA-VQ | 4096 | 0.078682 | 0.141841 | 0.306468 | 4096/4096 | 3830.60 | 0.99195 | 0.719% |
| EMA-VQ | 8192 | 0.077955 | 0.140458 | 0.302225 | 8192/8192 | 7557.19 | 0.99105 | 0.418% |
| EMA-VQ | 16384 | 0.077277 | 0.139447 | 0.298279 | 16384/16384 | 14621.08 | 0.98827 | 0.251% |
| EMA-VQ | 32768 | **0.076681** | **0.138612** | 0.294807 | **32629/32768 = 99.576%** | 27095.47 | 0.98172 | 0.168% |
| FSQ `4,4,4,4,4,2` | 2048 | 0.082129 | 0.145141 | 0.326948 | 2048/2048 | 1418.88 | 0.95187 | 3.701% |
| FSQ `3,3,3,3,5,5` | 2025 | 0.082157 | 0.145249 | 0.327032 | 2025/2025 | 1525.78 | 0.96282 | 3.621% |
| iFSQ `3,3,3,3,5,5` | 2025 | 0.082155 | 0.145035 | 0.327037 | 2025/2025 | 1514.31 | 0.96183 | 3.495% |

**在已测 EMA 档位中**，K32768 是首次 pooled 利用率不足 100% 的 K，缺 139 个码；这只把首次非满利用区间夹在 K16384 与 K32768 之间，未对中间整数 K 做穷举，不能称 K32768 是数学上的最小阈值。K32768 的 256 个位置平均各只激活约 18,249.5/32,768 个码（K16384 为 12,147.0/16,384）；pooled 与逐位置利用率不可混为一谈。此前 FSQ/iFSQ 的 K513 503/513 是 levels 结构造成的另一种非满样例，不能与 EMA 阈值等同。

EMA K1024→2048→4096→8192→16384→32768 的相邻扩容，在每对的 50/50 任务上差分 RMSE 都降低；各次 pooled 相对降幅约 1.07%、1.05%、0.92%、0.87%、0.77%。K32768 对 K16384 的 RMSE 改善仅约 0.77%，固定码长却由 14 增至 15 bit/token，故不应仅凭“终于不满 100%”将 K32768 选为最优。与严格同 K2048 的 FSQ 相比，EMA 的 RMSE `0.079519 < 0.082129`，50/50 任务均更低；同 levels 的 K2025 FSQ/iFSQ pooled RMSE 仅差约 `0.000002`，iFSQ 仅在 21/50 任务更低，不能凭该差异宣称优势。以上均为 seed42 的离线视觉特征重建，不是策略 rollout 结果；高 K 的重建优势仍需按实际码率、训练/推理成本和重复种子复核。

## 6. 后续选择与已完成消融记录

### 全 50 任务 query routing 严格对照（最终结果与后续选择）

为回答 PhiZero 风格全局可学习 query 是否优于当前空间对齐 query，node3 GPU4/5 新启动 EMA-VQ K1024、K2048 的 `global_qformer` 路由训练，入口 `scripts/physical_tokens/run_full50_physics_routing_ablation_node3.sh`，总日志 `artifacts/robotwin_full50_study/physics_routing_ablation_node003.log`。与已完成 `ema_k1024_steps20000` 和 `ema_k2048_steps20000` 的 `spatial_aligned` 对照保持相同 50 任务、完整 17 帧 clips、256 token/clip、seed42、batch32、20,000 steps、同一留出 episode 和 EMA 超参数；**唯一预设模型变量是 encoder/decoder attention routing mask**。新输出使用 `ema_global_k{1024,2048}_steps20000` 独立目录，拒绝覆盖已有结果，训练完成后自动运行同一完整留出集评估。已确认两个主训练进程在 node3 正常运行；结果未落盘前不得宣称 global 更好或更差。

主要比较差分 RMSE、未来特征 RMSE、逐任务差异、pooled/逐位置码分布；另用 `analyze_spatiotemporal_token_structure.py` 检查同一留出集上的同一时间段 token 多样性。为避免反复解压 39 GB NPZ，结构分析器新增可选 `--feature-array-cache` 参数，不改变指标定义；node3 GPU6/7 已启动已完成的 spatial-aligned K1024/K2048 基线结构分析，入口 `scripts/physical_tokens/run_full50_physics_routing_structure_node3.sh`，日志 `artifacts/robotwin_full50_study/physics_routing_structure_aligned_node003.log`，待 global 训练/评估完成再以完全相同脚本分析 global 一侧。这个对照只能检验 tokenizer 表征与结构，**不能**直接证明哪一路由更易由 VLM 预测；后者需要相同条件的冻结-tokenizer 预测器/策略对照，单看边际 perplexity 不足以回答。

随后两组均完成 20,000-step 独立完整留出评估；与同 K 的 spatial-aligned 对照完全相同的 47,797 clips、256 tokens/clip：

| K | spatial-aligned 差分 RMSE ↓ | global-qformer 差分 RMSE ↓ | spatial / global 未来特征 RMSE ↓ | global 较 spatial RMSE 增幅 | spatial 优于 global 的任务数 |
|---:|---:|---:|---:|---:|---:|
| 1024 | **0.080381** | 0.089124 | **0.144942 / 0.158946** | 10.88% | 50/50 |
| 2048 | **0.079519** | 0.088842 | **0.143288 / 0.158163** | 11.72% | 50/50 |

两个 global 模型 pooled 有效码也均为 K/K，说明“码本用满”不能替代重建质量或 token 结构验证。用户据此确定：**后续物理 tokenizer 的常规实验显式使用 `token_routing=spatial_aligned`；`global_qformer` 仅作已完成的消融/历史 checkpoint 兼容。**这与 64/256 token 长度选择是两个独立维度，不擅自撤销用户此前确定的 64-token 方案。代码构造器旧默认 `global_qformer` 继续保留，以免缺少路由字段的历史 checkpoint 被错误解释；所有新实验入口必须显式传 `spatial_aligned`。同段 token 是否同质化仍待结构分析完成，不能把当前 RMSE 差异单独当成机制证明。

### 全 50 任务 64-token 长度对照（最终离线重建结果）

用户确定先采用 64 个物理 token/clip，不再把 256 当成唯一方案。node1 GPU4/5 已启动新的 EMA-VQ K1024/K2048 训练，入口 `scripts/physical_tokens/run_full50_physics_64token_node1.sh`，总日志 `artifacts/robotwin_full50_study/physics_64token_node001.log`；输出 `ema_t64_k{1024,2048}_steps20000` 为独立目录，不覆盖 256-token 对照。仍使用相同全 50 任务、17 帧输入、训练/留出划分、seed42、batch32、20,000 steps、EMA 与 `spatial_aligned`，**只将每 transition query 数从 16 改为 4**。因此输出由 `16 transitions × 16 queries = 256` 变为 `16 × 4 = 64`；4 个 query 按 2×2 空间组分别读写原 4×4 特征网格，嵌套网格单元测试通过。训练结束后自动进行相同完整留出集评估，重点比较同 K 的差分 RMSE、未来特征误差、码分布以及每 clip 序列成本；目前仅确认主训练进程启动，尚无可报告的最终结果。

VLM 端若用现有固定长度 tokenizer/decoder，标签必须为完整 64 个编号；自回归模型可学 `<eom>`，但任意提前终止或额外生成不能直接喂给要求 `[B,64]` 的 decoder。真正的可变长度需要另设计长度条件、mask/padding 或变长 codec，并重新训练/验证；不能仅在推理时让 VLM 自由停止而声称当前 64-token 表示支持变长。

两组 64-token 模型随后完成相同 20,000-step 和同一 47,797-clips 完整留出评估。K1024 的 64/256-token RMSE 为 `0.085883/0.080381`，K2048 为 `0.085326/0.079519`；64-token 分别高约 `6.84%/7.30%`，对应 50/50 任务均高。64-token 的未来特征 RMSE 为 K1024 `0.152478`、K2048 `0.151243`，也高于各自 256-token 基线 `0.144942/0.143288`。两组 pooled 有效码都为 K/K。当前只可说 64-token 将序列长度降为四分之一且损失部分离线重建精度；尚无相同 VLM 训练/rollout 证据决定 64 与 256 在下游哪一方更优。用户的 64-token 选择保留，路由仍为 `spatial_aligned`。

### 全 50 任务 128-token 长度对照（启动记录及最终结果）

按用户要求增加中间长度，入口为 `scripts/physical_tokens/run_full50_physics_128token_node1.sh`，总日志 `artifacts/robotwin_full50_study/physics_128token_node001.log`，独立输出 `physics_tokenizers/ema_t128_k{1024,2048}_steps20000`。2026-10-26 23:01（北京时间）已核实 node1 GPU4/5 两个训练进程启动。沿用相同 50 任务、17 帧、训练/留出划分、EMA-VQ、seed42、batch32、20,000 steps、aux=0、`spatial_aligned` 和 K1024/K2048；将每 transition query 数由 64-token 的 4 或 256-token 的 16 改为 8，故每 clip 为 `16×8=128` 个 ID。现有 4×4 特征网格按 **2×4 query 网格**路由：每个 query 负责上下相邻两个特征位置；保留水平方向 4 格分辨率，存在方向性偏置。已新增 128-token 形状、精确路由 mask、解码及反向传播测试，相关测试 `22 passed`。训练结束后脚本会自动在相同完整 47,797 clips 留出集上独立评估。**当前不能用训练中 loss 代替最终 RMSE，也不能由离线重建指标断定 VLM/策略效果。** 最终应同 K 比较 64/128/256 的差分 RMSE、未来特征 RMSE、逐任务误差、active codes、perplexity、entropy、top10 占比与每 clip 的 ID 数；是否更适合 VLM 还需冻结 tokenizer 后的同条件下游验证。原 64-token 方案在最终对照出炉前不自动改动。

23:07 再次检查：两个 `training_config.json` 均记录 `emitted_tokens=128`、`num_tasks=50`、`train_complete_clips=421990`、`validation_complete_clips=47797`、`seed=42`、`max_steps=20000` 和正确 K/路由，两个主训练进程仍存活。训练入口设置 `--skip-initial-eval --eval-interval 5000`，所以首次 `metrics.jsonl` 和 `train.log` 指标要到 5,000 step 才出现；现在尚无可报告的中间/最终质量数据。

同日 23:47 两组 128-token 模型均完成 20,000-step checkpoint 与相同 **47,797 clips** 的独立完整留出评估；上一段只是当时的进度。最终结果与同 K 的已有 64/256-token 对照如下：

| EMA-VQ K | 64-token 差分 RMSE ↓ | 128-token 差分 RMSE ↓ | 256-token 差分 RMSE ↓ | 64 / 128 / 256 的未来特征 RMSE ↓ |
|---:|---:|---:|---:|---|
| 1024 | 0.085883 | 0.083388 | **0.080381** | 0.152478 / 0.149273 / **0.144942** |
| 2048 | 0.085326 | 0.082686 | **0.079519** | 0.151243 / 0.147667 / **0.143288** |

每个 K 的 128-token 模型逐任务在 **50/50** 任务优于 64-token、在 **0/50** 任务优于 256-token；有效码均为 K/K。128-token 以 2×4 query 网格取得中间重建质量，但采用更多自回归输出位置。用户明确选择后续物理长度为 **64 token/clip**；这是序列成本优先的工程选择，不改写上述重建排名，也不单独确定 K 或证明下游 rollout 更好。原始 JSON：`physics_tokenizers/ema_t128_k1024_steps20000/heldout_step20000.json`、`physics_tokenizers/ema_t128_k2048_steps20000/heldout_step20000.json`。

### 官方默认 FSQ 公式 K2048 复核（启动记录；最终结果见后文）

检查 `vector-quantize-pytorch-master/vector_quantize_pytorch/finite_scalar_quantization.py` 后发现本项目旧 `fsq` 使用与上游 `preserve_symmetry=True` 对应的对称网格；上游默认 `preserve_symmetry=False` 的偶数 levels 则是非对称网格。例如 L4：旧实现 `[-1,-1/3,1/3,1]`，上游默认 `[-1,-1/2,0,1/2]`。旧 K2048 levels `4,4,4,4,4,2` 含 L2，上游明确要求 L2 开启 `preserve_symmetry`，不能在保留旧 levels 的同时试验其默认公式。因此新增精确 K2048 且无 L2 的 levels `8,4,4,4,4`，两组并行：`fsq_official_levels84444_k2048` 为上游默认 bound，`fsq_symmetric_levels84444_k2048` 为本项目旧对称 bound。二者唯一预设差异为 bound/码格点；旧 `fsq_levels444442_k2048` 与 EMA K2048 作为跨 levels/方法参考，不当作单变量证据。

官方公式接入为新的 `fsq_official` 类型，旧 `fsq` 和已有 checkpoint 的行为保持不变；其 `half_l=(levels-1)*(1+1e-3)/2`、偶数 offset=0.5、atanh shift、tanh bound、round-STE、mixed-radix ID 和 decode 均按上述上游默认实现，且拒绝 L2。K2048 前向 ID、解码回环和非零有限梯度测试通过；相关 action/spatiotemporal tokenizer 测试 `27 passed`。node3 GPU4/6 两组相同 50 任务、256 tokens/clip、seed42、batch32、20,000 steps、同留出 episode、同空间路由，入口 `scripts/physical_tokens/run_full50_physics_official_fsq_k2048_node3.sh`，日志 `artifacts/robotwin_full50_study/physics_official_fsq_k2048_node003.log`；训练后自动独立完整留出评估。两个主进程已启动，最终结果未落盘前不宣布官方默认 FSQ 是否缩小与 EMA 的差距。

- K2048 及更大容量的 256-token 扫描已完成，原“是否继续 K2048”待办不再适用。用户已在人为权衡后将物理方案固定为 64-token EMA-VQ K2048；它不是相同 VLM 预测/策略预算下的赢家证明。更大 K 的少量离线收益仍需重复种子与码率复核。
- 若要回答下游机器人任务效果，须在相同 predictor/策略预算下比较选定的动作 8-token、物理 64-token 配方与必要对照，并报告 token 准确率、动作误差和成功率；本研究现有结论只覆盖离线 tokenizer 指标。

### 全 50 任务动作 tokenizer：8 vs 16 token/chunk 对照（2026-10-27，最终结果）

用户将后续物理 tokenizer 长度选为 **64 token/clip**，同时要求比较动作长度。当前全 50 任务动作基线为 EMA-VQ `8 tokens/chunk × K525`，每个 chunk 对应 `[32,14]` 动作（12 个关节维、2 个夹爪维），而非历史 VLM 试验中的 4-token 动作版本。为单独测试动作序列长度，node1 GPU0 于 12:09（北京时间）启动 EMA-VQ `16 tokens/chunk × K525`，入口 `scripts/physical_tokens/run_full50_action_16token_node1.sh`，总日志 `artifacts/robotwin_full50_study/action_16token_node001.log`，独立输出 `action_tokenizers/ema_16x525_decay095_dead100_steps20000`。新脚本拒绝覆盖旧输出，检查既有 8-token 最终评估和 GPU 可用显存，训练完成后自动运行相同完整留出评估。16-token 编码/解码形状冒烟测试已通过，训练 PID 已核实启动。

唯一预设长度变量为 `--num-tokens 8 → 16`；其余保持相同动作缓存 `action_all50_wrap32.npz`、固定 episode split、EMA-VQ K525、`ema_decay=0.95`、`dead_code_steps=100`、seed42、batch256、20,000 steps、相同网络维度、学习率和评估协议。8-token 基线最终 JSON：`action_tokenizers/ema_8x525_decay095_dead100_steps20000/heldout_step20000.json`。正式比较必须核对 55,797 个留出 chunks、Smooth L1、重建 RMSE、关节 wrapped MAE、关节速度 MAE、夹爪准确率、逐任务胜负、active codes、perplexity、entropy、top10 与每 chunk 的输出长度。16-token 若只改善离线重建，仍不能自动替代 8-token：VLM 输出位置和预测负担翻倍，需另作冻结 tokenizer 的下游同条件对照。当前**尚无 16-token 最终结果**。

12:20（北京时间）16-token 已完成 20,000-step checkpoint 与同一 55,797 chunks、250 个 task-episode 的独立完整留出评估；完整结果为 `action_tokenizers/ema_16x525_decay095_dead100_steps20000/heldout_step20000.json`。上一段保留为启动时状态，以下为**最终结果**。

| 动作 EMA-VQ K525，seed42 | 8 token/chunk | 16 token/chunk | 16 相对 8 |
|---|---:|---:|---:|
| 留出 chunks / 总 ID | 55,797 / 446,376 | 55,797 / 892,752 | ID 数 ×2 |
| 重建 Smooth L1 ↓ | 0.004821 | **0.004503** | −6.58% |
| 重建 RMSE ↓ | 0.099993 | **0.096116** | −3.88% |
| 关节 wrapped MAE ↓ | 3.2473° | **3.1306°** | −3.59% |
| 关节相邻步增量 MAE ↓ | **0.009291 rad/步** | 0.009491 rad/步 | +2.15%（变差） |
| 夹爪准确率 ↑ | 99.8656% | **99.9108%** | +0.0451 百分点 |
| 有效码 / K | 508/525 | 507/525 | 基本持平 |
| Perplexity ↑ | 363.36 | 376.63 | +3.65% |
| Entropy (nats) ↑ | 5.8954 | 5.9313 | +0.0359 |
| Top10 频率 ↓ | 8.57% | 7.65% | −0.92 百分点 |

逐任务的 16-token 胜出数：Smooth L1 与 RMSE 各 **46/50**，关节 wrapped MAE **43/50**，关节相邻步增量 MAE 仅 **19/50**；RMSE 变差的四任务为 `grab_roller`、`place_bread_skillet`、`place_dual_shoes`、`turn_switch`。逐任务等权宏平均 RMSE 为 `0.101218 → 0.097005`，与 pooled 方向一致。所谓评估器的 `joint_velocity_mae_rad` 实际计算每个有效相邻步的 `|[(q̂[t+1]−q̂[t])−(q[t+1]−q[t])]|`，仅在 12 个关节维上平均；它没有除以时间间隔，严格说单位为 **rad/步**，不是物理速度 rad/s，也不是额外的速度预测头。训练里的 `velocity_loss` 是权重 0.25 的相邻差分 Smooth L1，作用于动作各维；它与评估时仅关节维的绝对误差口径并不完全相同。单步角度误差变小不保证相邻步误差也变小：若误差逐步交替，差分会放大；这只是可能机制，本实验没有证明 16-token 的退步一定来自抖动。

16-token 的更细时间表示能降低静态重建误差，但并未改善相邻步变化保真；不能把“重建更好”概括为所有动作质量都更好。若用二进制固定长度编码 K525 的 ID，单 ID 需 10 bit，因此符号预算由 80→160 bit/chunk（未计封装或熵编码），VLM 需要预测的动作位置也翻倍。**用户决定后续动作采用 8-token/chunk；16-token 保留为质量–长度 Pareto 对照**。这是在尚无同条件 VLM 预测/策略收益时的长度取舍，不是“8-token 的离线重建更好”。两组只有 seed42 的长度对照，细小差异需种子复核。

### 官方默认与旧对称 FSQ 的同 levels K2048 最终复核

2026-10-26 23:44 两组均完成相同 50 任务、47,797 clips、256 tokens/clip、seed42、20,000-step 和独立完整留出评估；详细 JSON 分别是 `physics_tokenizers/fsq_official_levels84444_k2048_steps20000/heldout_step20000.json` 与 `physics_tokenizers/fsq_symmetric_levels84444_k2048_steps20000/heldout_step20000.json`。二者均为 levels `8,4,4,4,4`、K2048，唯一预设量化差别为官方默认非对称 bound vs 旧对称 bound。

| 指标 | 官方默认非对称 FSQ | 旧对称 FSQ |
|---|---:|---:|
| 差分 RMSE ↓ | 0.082405 | **0.082399** |
| 未来特征 RMSE ↓ | **0.145099** | 0.145100 |
| 有效码 / K | 2048/2048 | 2048/2048 |
| Perplexity ↑ | 1324.48 | 1321.55 |
| Entropy (nats) ↑ | 7.1888 | 7.1866 |
| Top10 频率 ↓ | 3.52% | **3.48%** |

官方默认的差分 RMSE 相对旧对称高约 **0.006%**，逐任务胜负为 25/50；这远小于可据以选择公式的幅度。旧 `4,4,4,4,4,2` K2048 FSQ RMSE 为 `0.082129`，但 levels 不同，不是公式的单变量对照。同 K2048 的 EMA-VQ、256-token RMSE `0.079519`，比官方 FSQ 低约 **3.50%**（以官方 FSQ 为分母；反向以 EMA 为分母，FSQ 高约 3.63%），且 50/50 任务均低。故“旧 FSQ 因公式写错导致 EMA 领先”的假设在这次同 levels 复核中未获支持；但这里只检验一个 levels/K/seed，不等于任何 FSQ 配置都无法追上。

### PhiZero 预训练 tokenizer 视频续训准备（2026-10-27）

用户提出在 RoboTwin 上从 `/mnt/jfzn/liujn/models/phizero_model/tokenizer/tokenizer.safetensors` 续训。该 checkpoint 与旁边的 `model_config.json` 均存在；官方 PhiZero 代码的正式 `--train-tokenizer` 路线用 33 帧/8 FPS 视频做未来潜变量 flow-matching 重建，同时更新 motion tokenizer、inverse-FSQ conditioner 与 DiT LoRA，Wan2.2 基础 DiT 和 VAE 冻结。预训练 tokenizer 的 FSQ levels 为 `[8,5,5,5,5,5]`，对应 K25000、256 token/clip；它应作为**独立 PhiZero 视频路线**，不能直接替换已选的空间对齐 EMA-VQ K2048/64-token ID，也不能把此前的 Qwen 特征差分 RMSE 与视频 flow-matching loss 直接比较。

已新增 `scripts/physical_tokens/prepare_phizero_robotwin_videos.py`，按固定 episode split 从 RoboTwin `Randomized` 的 `observation.images.cam_high` 视频制备 33 帧/8 FPS 的 MP4 与 `train.csv`、`val.csv`；少于 4.125 秒的原片会跳过，不补帧。单 episode smoke 已输出到 `artifacts/robotwin_full50_study/phizero_video_smoke`，其 `preparation_report.json` 记录 `train_clips=1`；官方 `VideoPathDataset` 已读出 `(3,33,512,896)`、像素范围 `[-1,1]`。PhiZero 预训练 tokenizer 本身也已通过官方 `load_physical_language_tokenizer(..., strict=True)` 加载（152,129,614 参数）。标准 RoboTwin 示例是 50 FPS、143 帧，约 2.86 秒；Randomized 示例是 15 FPS、139 帧，约 9.27 秒。因此当前优先对 Randomized 做视频路线，不能把短片直接交给官方 loader 冒充 4 秒样本。全量视频预处理和训练**尚未启动**。

目前在 `/mnt/jfzn/liujn/models` 及 PhiZero 仓库中未找到正式训练还需的四个冻结基础权重：`Wan-AI/Wan2.2-TI2V-5B` 的 `diffusion_pytorch_model-00001-of-00003.safetensors`、`00002-of-00003.safetensors`、`00003-of-00003.safetensors`（三者约 20 GB）及 `DiffSynth-Studio/Wan-Series-Converted-Safetensors` 的 `Wan2.2_VAE.safetensors`（约 1.4 GB）。PhiZero 仓库 README 的下载命令和 `PHIZERO_DIT_PATHS`、`PHIZERO_VAE_PATH` 设置可直接使用。现有 `starvla_qwen` 环境能导入官方训练模块，但这不等于已验证完整训练；获得基础权重后，需先跑单卡短训及 checkpoint 导出，再扩大到 50 任务并在固定留出集评价。

### PhiZero tokenizer 续训与公平方法比较（2026-10-30 更新）

上段是 2026-10-27 的准备记录，现由本次只读核验更新：本机已有 PhiZero tokenizer checkpoint `/mnt/jfzn/liujn/models/phizero_model/tokenizer/tokenizer.safetensors`（496 MB，Safetensors 可读取，749 tensors）、配套 `model_config.json`、Wan2.2 VAE（1.4 GB）及 Wan2.2 TI2V-5B 的三个 DiT 分片（约 20 GB）。Safetensors header/key 索引分别可读：DiT 三分片为 413/407/5 keys，VAE 为 196 keys；这不是完整模型 forward 的替代验证。tokenizer 配置为 8 维 FSQ code、levels `[8,5,5,5,5,5]`、K=25,000、每段 32 个 learned queries，33 帧输入输出 256 IDs。此前“缺少基础权重”的判断已过期。

本次采用 PhiZero 官方实现而不是只借用其结构：输入为实际 RGB 视频，经 PhiZero 3D 视频编码器和相邻帧 Q-Former 量化；训练目标是给定首帧后预测未来 Wan VAE latent 的 flow-matching MSE，并联合优化 tokenizer、inverse-FSQ/context conditioner 和 DiT LoRA，冻结 Wan base DiT 与 VAE。它和当前 spatial-aligned Qwen 特征差分 tokenizer 的输入、输出语义及损失空间不同，**不可将 flow-matching loss 与特征 RMSE 放在同一数值排行榜**。

全量视频已准备到独立目录 `artifacts/robotwin_full50_study/phizero_robotwin_videos/`，来源为 RoboTwin `Randomized` 的 `observation.images.cam_high`，固定复用 `docs/robotwin_tokenizer_episode_split.json`。ffprobe 确认 2,469 个输出片段均为 33 帧、8 FPS；CSV 为 2,222 train / 247 validation，覆盖全部 50 任务且 split 无交集。49 个任务各有 45/5 clips；`place_shoe` 源数据只有 19 个目标 episode，故仅有 17/2，解释了与计划 2,250/250 的差额。因提前中断留下的无效 MP4 已自动重编码。准备脚本 `prepare_phizero_robotwin_videos.py` 增加 `--workers` 并发和已有 MP4 校验/续跑；本次用 4 workers、ultrafast preset，视频总量约 1.42 GiB。当前没有闲置 GPU：GPU0–5 各约 28–34 GiB 可用显存但仍由其他作业占用，GPU6–7 仅约 9.3/3.3 GiB；因此未启动 PhiZero 训练，也未影响其他任务。

公平比较分两层进行：

1. **像素视频 tokenizer 主比较**：固定同一 33 帧视频、数据划分、PhiZero encoder/Q-Former/conditioner/DiT-LoRA 架构、256 输出位置、相同训练步数和 held-out flow-matching 评估。候选至少包括现成 PhiZero FSQ checkpoint 同域续训，以及同结构下的 FSQ、iFSQ、EMA-VQ 对照。方法间需尽可能复用相同的非量化初始化与训练预算；EMA-VQ 的码向量初始化规则、dead-code 处理、iFSQ 的准确公式和 conditioner 对离散码/连续 code 的接口，须在实现审计及短 smoke 后锁定，不能直接把不同 tokenizer 的旧 checkpoint loss 当作结果。初轮统一 256 IDs/clip；若量化器容量无法严格同 K，明确报告实际 vocab 与每 token bit 数，并做匹配码率复核。
2. **VLAct 下游价值比较**：冻结各 tokenizer，在完全相同的 RoboTwin train/validation episodes 上训练相同预算的物理-token predictor，并进一步比较同一策略/rollout 设置。至少报告 token CE/accuracy（同时对比 position-majority baseline）、预测 token 覆盖/perplexity、动作 MAE/速度 MAE、任务成功率。PhiZero 的 256 IDs 不能直接替换用户已选的 64-token、K2048 VLAct 物理配方；若要比较部署收益，必须单独统一 token 长度/预测预算或把不同长度作为明确的 Pareto 轴。

实现接口审计补充：PhiZero 当前 `FiniteScalarCodeDecoder` 根据 FSQ levels/mixed-radix ID 还原 code，不能原样服务任意量化器。iFSQ 需确认 ID 与 8 维码点的严格一致性；EMA-VQ 需引入共享的 codebook lookup，使训练 `from_codes(codes)` 和推理 `forward(indices)` 使用同一组 code vectors，并确保该表参与训练和 checkpoint 导出。比较实现前先写 round-trip、ID range、梯度和 save/load 一致性测试；否则不能把方法差异归因于量化算法。

每个方法分别报告：同目标 held-out flow loss 及确定性固定噪声/时间步重建曲线、离散 ID pooled 与逐位置 active count、经验 entropy、normalized entropy、perplexity、top10 质量占比、训练/验证 gap、训练吞吐和显存；最终以重复种子和下游预测/rollout共同决策。Flow loss 是主训练目标，离散码分布是 tokenizer 诊断，二者不可互相替代。执行顺序为：完整数据与权重校验 → tokenizer/模型加载和单步 forward/backward/checkpoint smoke → 小预算共同验证调参 → 锁定设置跑完整训练 → 统一 held-out 评估 → 进入下游 predictor。任何 GPU 任务只会在资源可用时启动。

### 2026-10-30：量化器兼容性审计与 node3 全量训练

用户要求三种 tokenizer 后续均可正常加载/推理，并指定使用 node3 8 卡。node3 (`10.120.154.141`) 已通过 SSH 确认为 8 张 A100-80GB 空闲；node1 的 GPU 有其他工作负载，本实验没有占用 node1。`vlact` 环境缺少 PhiZero 训练依赖 `peft`，已通过用户提供的 HTTP(S) 代理和清华 PyPI 镜像安装 `peft 0.21.2`，未改动 Torch/CUDA。

兼容审计发现并修正了两类问题：

- checkpoint 迁移原先只允许从 FSQ 转到 EMA-VQ，EMA-VQ 到 FSQ/iFSQ 或量化维数改变仍可能因量化器权重形状不符而加载失败。现在迁移时只对 `quantizer.*` / `code_decoder.*` 忽略缺失或不兼容形状，主干参数仍严格校验；形状兼容的投影会保留。
- EMA-VQ 现在复用 PhiZero 的 `8→6` quantizer input projection 和 `6→8` decoder output projection，在 tanh 归一化的 6 维码空间聚类；量化器与 conditioner 共享同一码本。EMA 统计保持 FP32，训练 ID/连续码与导出 checkpoint 的推理查表使用相同码本；straight-through 前向精确返回码本值。

验证结果：量化器单测 `9 passed`；原始 PhiZero FSQ checkpoint 的 tokenizer/conditioner 严格加载成功（tokenizer 130 个 state tensors，conditioner 14 个）；iFSQ (K=30,625) 与 EMA-VQ (K=25,000) smoke checkpoint 也通过严格 tokenizer/conditioner 加载和 ID→向量 decode round-trip，EMA conditioner 与 tokenizer 确认共享同一码本。修正结构后的 8 卡 EMA 两步 smoke 完成，峰值约 29 GiB/GPU，训练 loss 约 `0.20`；更早绕过原有投影的 EMA smoke 不作为实验结果。iFSQ 的 8 卡两步 smoke 也完成，levels `[5,5,5,5,7,7]`，迁移时成功复用同形状投影。短 smoke loss 使用不同随机视频噪声/时间步，不能用于方法排名。

完整训练顺序已在 node3 启动：每个方法从同一个现成 PhiZero tokenizer 初始化，使用相同 RoboTwin train split（2,222 clips）、8 卡 BF16 DDP、每卡 batch 1（global batch 8）、seed 42、20,000 optimizer steps、LR `2e-5` / 1,000 warmup / cosine 到 `1e-6`、gradient checkpointing、每 5,000 步保存；三种方法顺序为 FSQ K25,000、iFSQ K30,625、EMA-VQ K25,000。FSQ/iFSQ 配置 `fsq_entropy_weight=0.05`；EMA-VQ 不使用标量熵项，改由 EMA 更新和 dead-code 重置维护码本。此差异会在报告中注明，不把总训练 loss 当作完全同构的算法比较指标。

launcher：`PhiZero-main/examples/run_robotwin_quantizer_comparison.sh`。训练输出根目录：`artifacts/robotwin_full50_study/phizero_quantizer_comparison/runs_20261030/`，每个方法目录含 `training_args.json`、`model_config.json`、`training.log` 和 step checkpoints；编排日志为 `runs_20261030_orchestrator.log`。首个 FSQ 任务在 22:28 启动；最近一次核对约 77/20,000 步、2.54 秒/步，loss 在 `0.02–0.24` 范围波动，尚无完整训练结果。node3 开始时 8 卡空闲，之后另一项 `ovon_gdn` 任务开始共用 GPU 1、3、5、6（每卡约 32 GiB）；本实验 worker 在这些卡约占 30 GiB，当前总占用约 64/80 GiB，未停止或干扰该任务。新增 `PhiZero-main/DiffSynth-Studio/examples/phizero/evaluate_robotwin_checkpoint.py`，会在同一 validation CSV 上固定 seed 计算 held-out flow loss、任务宏平均、pooled/逐位置 token entropy、normalized entropy、perplexity、top10 质量占比、全留出 active count 和 mean per-clip active ratio。评估结果待训练 checkpoint 产出后运行；不会把 PhiZero flow loss 和旧 Qwen 特征 RMSE 混在一个排行榜中。

### 2026-10-30：按用户要求调整为 FSQ + EMA-VQ 并行

用户决定不再运行 iFSQ，并允许 node4 与其他任务共享 GPU 并行训练 EMA-VQ。node3 的 FSQ 训练继续运行；为防止原顺序 launcher 在 FSQ 完成后自动启动 iFSQ，已暂停其编排 shell（PID 41001），没有暂停 FSQ 的 accelerate/训练 worker。node3 最近核对为约 348/20,000 步、2.51 秒/步。

EMA-VQ 已在 node4（`10.120.154.142`）以 8 卡 BF16 DDP 启动，配置保持原实验的 20,000 步、每卡 batch 1、seed 42、LR/warmup/cosine、gradient checkpointing 与每 5,000 步保存；使用 K=25,000、EMA decay 0.99、dead-code threshold 0.1。输出目录为 `artifacts/robotwin_full50_study/phizero_quantizer_comparison/runs_20261030/ema_vq_k25000_seed42/`，启动日志为 `ema_vq_node004_launcher.log`。启动时 node4 已有其他任务共享 GPU；本实验成功完成前向/反向并推进至 29/20,000 步，约 2.51 秒/步，loss 有限且未见 OOM。显存读数约 80 GiB/卡，余量极小，仍需监控；没有停止或修改 node4 上已有任务。

因此后续正式对比只包含 FSQ 与 EMA-VQ；先前关于 iFSQ 的兼容性和 smoke 测试仍是代码验证记录，不构成这轮完整训练结果。两种方法结束后在同一 held-out split 上运行评估器；若 node4 EMA 初始化/训练失败，保留日志并报告失败原因，不影响 node3 FSQ。

用户进一步要求尝试 ACP 四卡 EMA-VQ。已新增 `PhiZero-main/examples/run_robotwin_ema_acp4.sh`，采用 ACP 推荐的可挂载模板 `pt-0crwzvee` 复制提交，规格 `N2lS.Ii.I60a.4`、单节点、显示名 `ljn-phizero-ema-acp4-20261030`，资源名 `pt-q81z5c0c`。配置为 4 卡、每卡 batch 1、gradient accumulation 2，global batch 8；输出路径独立于 node4 EMA。ACP 接受后一直处于 `STARTING`、Worker replicas=0，未启动训练。用户要求取消后已停止并删除该 ACP 任务；因此没有 ACP EMA checkpoint，也不将其计入实验比较。提交脚本保留，但不代表当前有 ACP 任务运行。

### 2026-10-30：尝试将 PhiZero FSQ/EMA-VQ 改为 global batch 32

用户要求两项训练改为 gradient accumulation 4。旧 FSQ/EMA 进程无法热改，且训练脚本只保存 tokenizer 权重、不保存 optimizer/scheduler state；当时约为 FSQ 775/20,000、EMA 382/20,000，尚未到 step 5,000 checkpoint。按要求保留旧日志并停止旧进程，新增 `PhiZero-main/examples/run_robotwin_quantizer_method.sh`，配置 8 卡×per-device batch 1×accumulation 4 = global batch 32，并分别写入 `_bs32` 新目录。

首次和 `bs32_retry1` 两轮启动均在第一个 optimizer step附近 OOM，两个方法都没有形成训练 checkpoint。失败日志分别位于 `fsq_levels8_5x5_k25000_seed42_bs32/training.log`、`ema_vq_k25000_seed42_bs32/training.log` 及同名 `_bs32_retry1/` 目录。日志显示训练进程每卡约需 29.4 GiB，而 node3/node4 的共租进程及其他 GPU context 使首次 backward 时只剩约 80–199 MiB；gradient accumulation 不增加单个 microbatch 的前向激活，但不能解决底层显存容量不足。之后用户决定恢复 global batch 8。

### 2026-10-30：恢复 global batch 8 并行重启 FSQ / EMA-VQ

按用户最新决定，两项训练都设置为 8 卡 × 每卡 batch 1 × gradient accumulation 1，即 global batch 8。此前 global batch 8 部分运行没有到达 5,000-step checkpoint，无法恢复模型/优化器状态；因此新运行从共同的 PhiZero tokenizer 初始化权重重新开始，独立保留旧日志，不覆盖旧目录。运行入口为 `PhiZero-main/examples/run_robotwin_quantizer_method.sh`，其 `PHIZERO_GRAD_ACCUMULATION_STEPS` 默认为 1，`PHIZERO_RUN_TAG=bs8`。

- FSQ：node3 (`10.120.154.141`)，输出 `artifacts/robotwin_full50_study/phizero_quantizer_comparison/runs_20261030/fsq_levels8_5x5_k25000_seed42_bs8/`，K=25,000，levels `[8,5,5,5,5,5]`。
- EMA-VQ：node4 (`10.120.154.142`)，输出 `artifacts/robotwin_full50_study/phizero_quantizer_comparison/runs_20261030/ema_vq_k25000_seed42_bs8/`，K=25,000，EMA decay 0.99，dead-code threshold 0.1。

两组均为 seed 42、20,000 optimizer steps、BF16、gradient checkpointing、每 5,000 步保存，训练集 2,222 clips；学习率配置保持 `2e-5`、1,000 步 warmup、cosine decay 到 `1e-6`。2026-10-30 最近一次只读核验：FSQ 已到约 31/20,000、EMA-VQ 约 26/20,000；日志 loss 均有限（最近分别约 0.149、0.316），进程已越过首个 optimizer step。node3/node4 GPU 显存分别约 80.3–80.9/80 GiB、利用率 100%，余量极小。故目前仅确认训练已启动且早期步骤在推进，不能称长程稳定；继续观察是否发生 OOM/挂起。独立完整评估需等 checkpoint 产出后再进行，当前无方法优劣结论。

### 2026-11-01：训练日志导入 W&B 的状态

FSQ `bs8` 在 4,686/20,000 步附近遇到 NCCL all-reduce 超时，未达到首次 5,000-step checkpoint；EMA-VQ `bs8` 在 18,872/20,000 步收到 SIGTERM，保留 5k、10k、15k tokenizer checkpoint。两项训练原本没有调用 `wandb.init`，因此没有原生 W&B 历史。`scripts/physical_tokens/import_phizero_training_logs_to_wandb.py` 按 tqdm 每个进度片段的最后一条 loss/lr 记录重建历史，并验证步号连续、数值有限。已生成两个 W&B offline runs：

- `wandb_log_import/wandb/offline-run-20261101_131507-phifsq08`：FSQ，共 4,686 条，W&B ID `phifsq08`。
- `wandb_log_import/wandb/offline-run-20261101_131510-phiema08`：EMA-VQ，共 18,872 条，W&B ID `phiema08`。

两条 run 的 entity 设为 `jianingliu`，待新建项目为 `phizero-robotwin-quantizer-comparison`。记录 `optimizer_step`、`train/loss`、`train/lr` 和原训练配置，名称/备注明确标注由控制台日志重建。W&B 的前期成功日志曾显示登录身份 `app123 (jianingliu)`，并在 2026-10-30 成功上传过其他项目的离线历史；现有 `.netrc` 凭据仍能被 SDK 读取。

两条 offline runs 已一并打包为 `artifacts/robotwin_full50_study/phizero_quantizer_comparison/wandb_log_import/phizero-robotwin-wandb-offline-runs.tar.gz`（约 1.5 MB），供另一个具备 W&B API 出口的机器下载、解包后分别运行 `wandb sync offline-run-...`；归档已校验包含两个 `run-*.wandb` 数据文件。

当前**尚未同步到网页端，也未创建云端项目**。用 `10.119.18.5:3128`、本地 `127.0.0.1:17897`、node1–4 以及现有 ACP worker 试验：服务器代理可以完成 `CONNECT api.wandb.ai:443`，TLS 证书匹配，但 W&B/Google 前端对 `/graphql` 返回 HTML 403；ACP worker 直连和同代理均超时。带 `.netrc` 认证的 GraphQL 请求、`wandb.Api()`、在线 `wandb.init()`、以及对 FSQ offline run 的正式 `wandb sync` 都因该 403 失败。反向 SSH 隧道复用同一个代理出口，结果也为 403；不能把本次失败归因为项目名或 entity 权限。需要一个当前能成功访问 `api.wandb.ai/graphql` 的出口，或由 W&B/网络侧恢复现有出口访问。

恢复网络后，使用 `vlact` 环境从仓库根目录同步上述两个 offline run 目录，例如：

```bash
export http_proxy=http://10.119.18.5:3128 https_proxy=http://10.119.18.5:3128
/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/wandb sync artifacts/robotwin_full50_study/phizero_quantizer_comparison/wandb_log_import/wandb/offline-run-20261101_131507-phifsq08
/mnt/jfzn/liujn/miniconda3/envs/vlact/bin/wandb sync artifacts/robotwin_full50_study/phizero_quantizer_comparison/wandb_log_import/wandb/offline-run-20261101_131510-phiema08
```

同步完成后应核对网页 run `phifsq08` / `phiema08` 的 history 末步分别为 4,686 / 18,872。不能把日志重建的 total training loss 当作相同验证集上的 flow loss 排名；FSQ 训练 loss 还包含熵正则项。

### 2026-11-01：完整训练历史已上传 SwanLab

W&B API 仍返回 HTML 403；已改用 SwanLab 的 `jianingliu/phizero-robotwin-quantizer-comparison` 项目上传同一批原始训练日志。导入脚本为 `scripts/physical_tokens/import_phizero_training_logs_to_swanlab.py`，逐步记录 `optimizer_step`、`train/loss`、`train/lr`，保留训练配置、日志来源及停止原因。SwanLab 客户端安装在独立的 `swanlab_upload_env` 中，未改动 `vlact` 训练环境；上传密钥未写入脚本或文档。

- [FSQ bs8](https://swanlab.cn/@jianingliu/phizero-robotwin-quantizer-comparison/runs/phifsq08)：云端状态 `FINISHED`，完整 4,686 步；API 核验末步 `optimizer_step=4686`，`train/loss=0.2062`，`train/lr=1.83e-5`。
- [EMA-VQ bs8](https://swanlab.cn/@jianingliu/phizero-robotwin-quantizer-comparison/runs/phiema08)：云端状态 `FINISHED`，完整 18,872 步；API 核验末步 `optimizer_step=18872`，`train/loss=0.2228`，`train/lr=1.16e-6`。

这些是两个**不同停止步数**的训练曲线，不是同一验证集上的方法排名。W&B offline runs 仍保留，尚未上传 W&B 网页端。

### 2026-11-01：EMA-VQ bs8 15k checkpoint 留出集评估

用 `runs_20261030/ema_vq_k25000_seed42_bs8/step-15000.safetensors` 和固定的 `phizero_robotwin_videos/val.csv` 评估全部 247 个 clip、50 个任务。结果文件为 `artifacts/robotwin_full50_study/phizero_quantizer_comparison/ema_vq_bs8_step15000_eval_full.json`。评估脚本 `PhiZero-main/DiffSynth-Studio/examples/phizero/evaluate_robotwin_checkpoint.py` 的 `normalized_entropy` 原先对 Python 整数调用 `.log()`，已修正并先通过单 clip 测试，再完成全量测试。

| 指标 | 15k EMA-VQ |
| --- | ---: |
| 留出集 flow loss（clip 平均） | 0.138319 |
| 留出集 flow loss（任务宏平均） | 0.136714 |
| 每 clip token 数 | 256 |
| 码本大小 | 25,000 |
| 全留出集有效码数 / 利用率 | 21,558 / 86.23% |
| pooled perplexity / normalized entropy | 18,029.91 / 0.9677 |
| pooled top-10 质量占比 | 0.2530% |
| 每 clip 平均有效码数 / perplexity | 250.13 / 247.91 |
| 每位置平均有效码数 / perplexity | 104.91 / 85.91 |

这些结果表明 15k 权重能正常推理，且在本验证集上码分布较分散；不能仅据训练 loss 的单步波动判断已发散。不过当前 flow loss 是每 clip 抽一次随机扩散时间步和噪声、再做时间步加权的估计，任务级极值尤其容易受抽样影响。尚无同一评估设置下的预训练/5k/10k 对照，也没有多时间步重复评估，因此不能据此断言 15k 一定优于其他 checkpoint 或量化方法。

### 2026-11-01：EMA-VQ 5k/10k/15k checkpoint 对照

随后用同一评估脚本、同一 247 个验证 clip、seed 42 和相同 Wan/VAE 权重完成 5k、10k 评估；每个 clip 的随机时间步和噪声按固定 seed 配对。结果 JSON 分别为 `ema_vq_bs8_step5000_eval_full.json`、`ema_vq_bs8_step10000_eval_full.json`、`ema_vq_bs8_step15000_eval_full.json`。

| Checkpoint | Flow loss（clip 平均） | Flow loss（任务宏平均） | 全集有效码 | 利用率 | Pooled perplexity | Top-10 占比 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5k | 0.139228 | 0.137619 | 21,158 / 25,000 | 84.63% | 17,443.98 | 0.3954% |
| 10k | 0.138445 | 0.136842 | 21,488 / 25,000 | 85.95% | 17,904.85 | 0.3321% |
| 15k | 0.138319 | 0.136714 | 21,558 / 25,000 | 86.23% | 18,029.91 | 0.2530% |

15k 相比 5k 的 flow loss 下降约 0.65%，且 50 个任务的任务均值全部更低；相比 10k 仅下降约 0.09%。因此这组固定采样评估显示训练到 15k 有小幅、方向一致的留出集改善，之后趋于平台，同时码本使用率仍缓慢上升。每个 clip 只抽一个扩散时间步/噪声，且没有预训练基线，故改善幅度应视为有限证据，不能外推为显著的生成质量提升或跨方法优胜。
