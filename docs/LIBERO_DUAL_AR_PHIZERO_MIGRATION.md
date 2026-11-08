# 双流物理/动作 token 方法迁移到 LIBERO：实现说明与落地清单

本文供接手 LIBERO 实验的同学使用。以仓库当前代码为准，目标是在 LIBERO 上训练「图像/任务语言 → VLM 自回归生成动作与物理 token → 条件化 DiT 生成连续动作」的策略。RoboTwin 的实验配置可用作实现参考，数据统计、动作语义、物理码及 checkpoint 不能默认跨 benchmark 通用。

**本轮锁定的 LIBERO 主方案**：从 PhiZero **FSQ** 预训练权重出发，在 LIBERO 视频上续训 **64 个物理 token/33 帧 clip，目标采样频率 8 FPS**；物理 tokenizer 训练对完整窗口使用 **8 个原始控制帧（0.4 秒）**的起点 stride，并为**每个 episode 最多额外保留 1 个末帧重复补齐窗口**，训练预算固定为 **全局 batch 8、10,000 个 optimizer steps**。其后的**独立双流 VLM LoRA 阶段与 VLM+DiT 联合阶段各训练 10,000 个 optimizer steps**；这两个阶段的全局 batch 尚待显存 smoke 确定。FSQ levels 沿用 `[8,5,5,5,5,5]`，因此物理词表为 **K=25,000**。VLM 训练采用 **teacher forcing** 计算动作/物理双流 CE；推理仍必须自回归生成，不使用未来真值码。LIBERO 动作 token 数 `N_a` 待动作 tokenizer 的 8/32 码实验确定，不把 RoboTwin 的 32 码当成既定结论。256 码和 EMA-VQ 仅作可选对照，不是主方案。

**实现状态**：双流 AR、分模态码表、concat fuser 和 RoboTwin 联合训练配置在仓库里有代码；`QwenPIv4DualAR32x64.py` 与对应 concat YAML 目前还是工作区中未提交的新文件。现已加入 PhiZero 32→8 query 权重迁移、精确 10k-step 入口、LIBERO 视频窗口与 64 码 ID 导出，以及双流 K25000 配置支持；**尚未实际训练 64 码 PhiZero FSQ，也尚未完成 LIBERO 动作 tokenizer、双流缓存对齐、联合模型实测或 RGB 重建/闭环评估**。此前的 15k EMA-VQ 可视化和其他 256 码 FSQ 图不能当作本方案的实测质量。开始正式实验前先将这些文件纳入可复现的版本记录。

## 1. 五项改动分别是什么

| 改动 | 仓库当前实现/参考 | 迁移到 LIBERO 时的决定 |
| --- | --- | --- |
| PhiZero 路线物理 tokenizer | 原版 PhiZero FSQ 为 33 帧、256 码、K25,000；`train_phizero_global_batch.py` 已支持 `--queries-per-transition 8` 和 `--max-train-steps`；RoboTwin 的 EMA-VQ K2048 launcher **不是**本方案 | 锁定真正的 PhiZero 视频 FSQ：保持 33 帧/8 FPS、FSQ levels `[8,5,5,5,5,5]` 与 K25,000，将每个编码后 transition 的 query 数 **32→8**，得到 `8×8=64` 码；同域续训并导出 LIBERO ID。不能截断预训练 256 码来代替训练 |
| 新动作 tokenizer | `docs/ROBOTWIN_ACTION_TOKEN_LENGTH_8_VS_32.md`：32 步×14 维 RoboTwin 动作用 EMA-VQ，32 ID/chunk、K525、decay 0.95、dead-code 100、20k 步、batch 128 | LIBERO 为 7 维相对末端执行器动作，常用 8 步 chunk；应在 LIBERO 动作上**重新训练**，沿用量化超参数可作初始候选，32 个 ID/32 步这个长度不能直接搬过去 |
| 模态自己的 head | `starVLA/model/framework/QwenDualAutoregressiveReasoner.py` 的 `separate_heads=true`、`strict_modality_tying=true`；联合模型在 `QwenPIv4DualAR32x64.py` | 文字表、动作表、物理表按模态路由；动作 `E_a∈R^(K_a×H)`、物理 `E_p∈R^(K_p×H)` 分别承担该模态输入 embedding 和输出分类矩阵。仅换输出 head 与严格输入/输出绑权是两种不同配置，checkpoint 不互换 |
| VLM 自回归 | RoboTwin 参考目标顺序为 32 动作 ID → 64 物理 ID；训练 teacher forcing 按流计算 CE，推理逐码生成，总计 96 步 | LIBERO 固定 `N_p=64,K_p=25000`，按选定 `N_a,K_a` 改 prompt、label mask、切换位置及最大长度；训练 teacher forcing 合理，闭环必须只用自由生成的码 |
| DiT 维度 concat | `QwenPI_v4.py` 的 `ConcatCrossAttentionFusion`，联合模型逐层调用 | 此处 **concat 的是三个 cross-attention 结果的特征维**：`[prefix_attn; action_attn; physics_attn]`，经 LayerNorm、瓶颈 MLP 映射回专家维度，作为残差；三组 memory 本身先沿**序列维**拼接、再拆开分别注意力。它不是把动作向量和物理向量直接拼到 DiT 输入动作维上 |

### 当前联合策略的数据流

```text
观测图像 + 指令 (+ 状态文本) ──┬──> 原 Qwen/π prefix ────────────────┐
                              └──> 双流 AR: a₁…a₃₂, p₁…p₆₄ ─┐  │
                                   码表向量 + 对齐隐状态       │  │
                                   ↓                         ↓  ↓
                              action / physics memory → DiT 每层三路 cross-attn
                                                         ↓
                                         特征维 concat → MLP 残差 → 连续动作 chunk
```

联合模型实现：`starVLA/model/framework/QwenPIv4DualAR32x64.py`。其中 `TokenStateMemory` 将每个码的专属 embedding 与预测该码的 VLM hidden state 沿特征维拼接，投影到 DiT 隐维并加位置 embedding。训练目标为

```text
L = L_flow + λ_a · CE_action + λ_p · CE_physics
```

现有 RoboTwin 联合配置的 `λ_a=1e-3`、`λ_p=1e-4`，是该实验的数值设定，不应视为 LIBERO 已调优值。两流 CE 各自在自己的码表上归一化。DiT 的输出始终是连续动作，动作 tokenizer 的 ID 用作条件及辅助监督，并不在这条联合模型中直接解码成执行动作。

## 2. 先锁定 LIBERO 的数据契约

仓库原生 LIBERO 基线在 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`：`libero_all` 包含 spatial/object/goal/10 四套 LeRobot 数据，动作类型 `delta_qpos`，`action_dim=7`，`future_action_window_size=7`（即 8 步）。本地四套数据各自 `meta/info.json` 均标注原生图像/动作 **20 FPS**；`examples/LIBERO/eval_files/eval_libero.py` 保存回放视频时的 `fps=10` 只是回放文件播放帧率，**不能**作为训练数据采样依据。数据准备参见 `examples/LIBERO/README.md` 和 `examples/LIBERO/train_files/modality.json`。评测接口在 `examples/LIBERO/eval_files/`，以 `normalized_actions [B, chunk, 7]` 接收策略输出、用该 checkpoint 的统计量反归一化，再转成环境 `[Δx,Δy,Δz,Δroll,Δpitch,Δyaw,gripper]`。评测端对 `agentview_image`、`eye_in_hand_image` 做 180° 翻转；训练图像视角和变换必须一致。

建议先固定以下契约，写入 LIBERO 专属配置/manifest；不要依赖 RoboTwin 的默认值：

| 字段 | 建议起点 | 需要核验 |
| --- | --- | --- |
| 样本键 | `(suite, task, episode_id, base_index)` | 四套数据中 episode ID 可重复；task 名及索引在每份缓存中稳定 |
| 动作目标 | 当前帧开始的 8×7 归一化相对动作 | 无跨 episode 填充；`action_valid_mask` 与实际有效步一致；夹爪方向和归一化统计匹配评测 |
| 物理目标 | 同一 `base_index` 起，20 FPS 原视频按 8 FPS 目标时间戳重采样的 33 帧 | 使用同一相机与图像变换；未来不足 4 秒时重复本 episode 的末帧补齐，并记录实际帧数/补帧数；不跨 episode 或拉伸演示 |
| 输入 | 只用当前可见图像、任务文本和可用本体状态 | 未来视频仅用于训练 tokenizer/目标 ID；评测时不能传给策略 |
| 数据划分 | 以 episode 为单位，在任何视频窗口/动作缓存生成前划分 | train/val/test 的源 episode 不相交；跨 suite 的重复演示也需去重核对 |

**已选帧采样协议**：保持 PhiZero 预训练时的 **8 FPS、33 帧**，所以名义首末帧时间差是 `32/8=4.0 秒`（不是把 33 除以 8 得到的播放时长 4.125 秒）。对原生 20 FPS 的每个 LIBERO episode，以当前决策帧 `base_index=s` 为 `t=0`，对 `j=0,…,32` 在时间 `t_j=j/8 秒` 取最近的原始观测帧；若时间戳严格均匀、该 episode 长 `N`，统一使用 `source_index_j=min(N−1, s+floor(20j/8+0.5))` 固定半格取整规则。超过 episode 末尾的输出帧都重复原始第 `N−1` 帧；首帧始终是 `s`。实际 exporter 应优先核验源视频时间戳及 LeRobot step index 一致性，并把源帧索引列表、`real_frame_count`、`padded_frame_count`、相机、裁剪/旋转/缩放规则写入 manifest。训练和导出 token 必须使用**同一**预处理；默认用基线的 primary image (`observation.images.image`)，不要混用 wrist 相机。LIBERO 源视频为 256×256，而现有 PhiZero loader 会直接插值到 512×896、拉伸宽高比；接入前需确定并统一宽高比处理（建议等比缩放加边框）及评测端图像旋转，否则可能恰好丢失或扭曲首帧机械臂。

**时长与覆盖率代价**：8 个 20 Hz 动作对应约 `8/20=0.4 秒` 控制时长，而物理码名义预测未来 **4 秒**，是长时物理上下文，不是与当前动作 chunk 等长的重建目标；补齐样本的**真实未来跨度更短**，其余视频由末帧保持静止。按本地四套原始 episode 长度（尚未做 train/val 划分）计算，仅有 **138,050 / 273,465** 个原始 step 满足 `s+80<N`、能形成完整 4 秒窗口。新方案让**每个 episode 至少进入物理 tokenizer 训练一次**，但不对每个尾部起点重复抽训练片段；后续稠密物理码缓存才覆盖所有原始决策起点。若 4 秒跨度令闭环效果不合理，再另设 20 FPS/33 帧（1.6 秒）等短时长**对照**，但这会改变 PhiZero 的时间语义，不能与 8 FPS 主方案混称同一配置。

**窗口起点 stride 与首尾覆盖**：这里的 stride 单位是原始 **20 Hz 控制帧**，不是输出视频的 8 FPS 帧。对长度 `N≥81` 的 episode，先取所有**完整**窗口的规则起点 `s=0,8,16,…≤L`，其中 `L=N−81`；若 `L` 尚未在集合中，再加入这个以末帧结束的完整窗口。然后只额外选**一个**含补帧窗口：取规则起点中**第一个** `s>L` 的位置，并把缺失视频帧重复为本 episode 末帧 `N−1`。由于 stride 8 小于 80 帧的窗口跨度，这个最早的短尾窗口只需少量补帧，不会选靠近结束、几乎全是静止画面的窗口。若 `N<81`，没有完整窗口，只取起点 `s=0` 的**一个**补帧窗口。这样每个 episode 的第一帧和最后一帧都进入训练片段，包括原本不足 4 秒的 14 个 episode；不跨 episode，也不重复保存相同起点。

按本地四套 LIBERO 原始 `meta/episodes.jsonl` 的 episode 长度、**完整窗口 stride 采样 + 每个 episode 最多一个末帧补齐窗口**计算，划分 train/val 前的窗口数为：

| 起点 stride（原始 20 Hz 帧） | 起点时间间隔 | 全部窗口 | 其中完整 4 秒 | 其中含末帧补齐 |
| --- | --- | ---: | ---: | ---: |
| 1 | 0.05 秒 | 139,743 | 138,050 | 1,693 |
| 4 | 0.20 秒 | 38,077 | 36,384 | 1,693 |
| **8：tokenizer 训练推荐** | **0.40 秒** | **21,151** | **19,458** | **1,693** |
| 16 | 0.80 秒 | 12,677 | 10,984 | 1,693 |

上述 stride 8 的 21,151 段分别来自 spatial 3,297、object 4,878、goal 3,204、LIBERO-10 9,772。此前 14 个短于 81 原始帧的 episode 现在也包含在补齐窗口里；没有任何 episode 因时长不足而完全无物理 tokenizer 样本。这些是**未划分数据的理论窗口数**，正式训练须先按 episode 划分，再分别计算 train/val 数量及导出文件数；不跨 split 滑窗。

**tokenizer 训练 stride 与 AR 缓存 stride 不同**：上述 stride 8 仅用于 10k-step 的物理 tokenizer 训练，减少高度重叠的视频片段。冻结 tokenizer 后，为 VLM/DiT 训练导出 `physics_token_ids` 时，默认对**每个原始决策帧**编码，即物理缓存起点 stride **1**，同样使用末帧重复补齐；四套数据划分前共约 273,465 个物理码窗口，并沿用完全相同的 20→8 FPS 帧索引及图像处理规则。否则缓存若也只取 stride 8，`discrete_token_cache_drop_missing: true` 会滤掉大量动作样本。完整联合样本仍需有可训练的 8 步动作目标：若动作 tokenizer 只接收完整 8 步，则每个 episode 最后 7 个物理码起点只能供物理侧使用，合并缓存前必须按键与完整动作缓存求交集，**不可重复末尾动作来伪造动作监督**。现有 `merge_multitask_dual_token_targets.py` 可选的 `--physics-feature-cache` 会按 `future_valid_mask` 过滤非完整未来，若沿用这个参数会把刚新增的补齐物理窗口再次删掉；需调整合并规则或不传该可选参数，并在 manifest 记录补帧比例。稠密物理编码的离线时间/存储要单独预算。

**补帧分布风险**：按当前 stride 8 规则，**1,693/21,151（约 8.0%）**物理 tokenizer 训练片段含重复末帧；当前 PhiZero flow loss 会让这些静止补帧也进入目标。训练日志、留出 flow loss、码利用率与 RGB 重建须分别报告“完整 4 秒”和“含补帧”两组，同时记录每段真实帧比例。注意 AR 物理码缓存采用 stride 1，**其补帧比例仍约为 135,415/273,465（49.5%）**，不能把 tokenizer 训练集的 8.0% 误写为下游监督的补帧比例；如自由预测偏向静止尾段，需再讨论 AR 样本权重或掩码。

## 3. 建议的实施顺序

### A. 先复现无 token 的 LIBERO 基线

按 `examples/LIBERO/README.md` 准备四套数据与 `modality.json`，运行现有 `run_libero_train.sh`/评测服务，确认一个 batch 的图像、状态、动作形状和 8 步闭环输出。保留基线成功率、模型版本、统计量与 episode 划分。RoboTwin 的 `abs_qpos`、14 维动作、统计文件和 100k 微调权重不能直接充当 LIBERO 基线。

### B. 训练 LIBERO 动作 tokenizer

参考 `scripts/physical_tokens/train_action_tokenizer.py`、`scripts/physical_tokens/export_robotwin_action_cache.py`。从 LIBERO 训练 episode 导出 `[N,8,7]` 动作 chunk 与有效掩码；按训练集统计归一化，并在固定留出 episode 上评估重建和码分布。建议先比 `N_a=8` 与 `N_a=32`、K525、EMA decay 0.95、dead-code 100；它们是候选而非 LIBERO 定论。报告连续动作 RMSE/MAE、夹爪正确率、有效码数/entropy、逐位置码重复率、逐任务误差。确认 32 码是否值得额外 AR 延迟；如果采用 8 码，要把联合模型中写死的 32 全部参数化。

### C. 训练或选择 LIBERO 物理 tokenizer

真正 PhiZero 路线参考 `scripts/physical_tokens/train_phizero_global_batch.py`；`prepare_phizero_libero_videos.py` 已按上面的 **20→8 FPS、33 帧、完整窗口训练起点 stride 8、每个 episode 最多一个短尾补帧窗口、另补不重复的完整末尾对齐窗口**协议制作 LIBERO train/val CSV，每行一个 clip，并在 manifest 记录来源 episode、`base_index`、全部 33 个源帧索引、真实/补齐帧数、相机与图像预处理。它将 256×256 源帧等比 letterbox 至 512×896，而不是拉伸。RoboTwin 的 `prepare_phizero_robotwin_videos.py` 硬编码了 RoboTwin `cam_high` 路径和任务列表，不用于 LIBERO。PhiZero 训练还依赖本地 PhiZero/DiffSynth、Wan2.2 VAE 与 DiT 权重。

本方案从**原始 PhiZero FSQ checkpoint** 初始化，量化器仍为 FSQ，levels `[8,5,5,5,5,5]`（`K_p=8×5^5=25,000`）。原 `act_embedding_num=32`；33 帧经两次时间下采样得到 9 个 latent 帧，即 8 个相邻帧 transition，原长 `8×32=256`。`--queries-per-transition 8` 先严格加载原 checkpoint，再等距选取原 query 行 `[2,6,10,14,18,22,26,30]` 初始化 8 个新位置；其余 tokenizer/conditioner/DiT LoRA 权重沿用原 checkpoint，并把保存的模型配置改成 64 码。它不等于直接截取原 256 码中的前 64 个 ID；后续仍须在 LIBERO 视频上续训。

先做 1-batch 的输出形状 `[B,64]`、ID 范围 `[0,24999]`、FSQ ID→码向量、保存/重载与反向传播 smoke；然后报告留出集 flow loss、码利用率及同一采样设定下的 RGB 重建图/视频。可视化须覆盖首帧无机械臂和首帧已有机械臂两类，重点检查**后续生成帧**中的机械臂、夹爪、物体接触及物体是否消失。若比较不同起点窗口，应对齐相同真实时间帧，同时注明预测跨度也会改变；首帧是给定条件，不是生成质量证据。只有 64 码本身的留出重建足够可用，才进入 AR/策略训练。

**训练预算口径**：上述 64 码 FSQ 的正式续训采用**全局 batch 8、10,000 次优化器更新**（不是 10,000 个 microbatch，也不是 10,000 个 epoch）。以当前 `train_phizero_global_batch.py` 的 4 卡实现为例，配置为 **4 GPU × 每卡 microbatch 1 × gradient accumulation 2 = global batch 8**；多次累积的前/反向只合计一次 optimizer step。它累计处理约 **80,000 个训练窗口呈现次数**，不等于 80,000 个互不重复的视频。按四套数据**尚未划分**的 21,151 段估算为 `80,000/21,151≈3.78` 个数据遍历；如果之后按 episode 划出约 10% validation、训练窗口约 19,000 段，则约 `80,000/19,000≈4.2` 个 train epoch。**精确 epoch 数须以实际 train split 的窗口数为准**：`epoch_equivalent≈10,000×8/N_train`；现有 sampler 每轮遍历所有样本、最后一个不足 8 的 batch 也保留，因此严格计算还应以 `ceil(N_train/8)` 个 optimizer steps/epoch 和最终部分 epoch 为准。相邻滑窗高度相关，4.2 遍不表示 4.2 倍独立数据量。

建议保留 5,000-step 中间 checkpoint，并以 10,000-step checkpoint 作为主交付权重，分别在相同留出集、相同随机种子和推理步数下评估。全局 batch 8 是**物理 tokenizer 的本轮设计配置**，并不表示 64-query LIBERO 训练已经实测无 OOM；正式开跑前须做 4 卡一个优化器步的显存/吞吐 smoke。batch size、累积步数、学习率及 LR schedule 要写入 run manifest。训练入口现支持 `--max-train-steps 10000` 精确截止并以此设定 scheduler；`--epochs` 仍须设到足够覆盖 10k（当前 19,037 train 窗口、batch 8 约需 5 epochs）。**动作 tokenizer 的训练步数仍需独立确定**；AR 与联合阶段各 10k 的预算见 E 节。

“基于 PhiZero 续训”必须注明初始化范围：本方案**不使用** `--fresh-phi-zero-components`，因为它会重新初始化 tokenizer/conditioner/LoRA，不能代表从 PhiZero FSQ 权重续训。即使沿用 FSQ，32→8 query 的那一部分也需要显式适配，因此不能声称所有权重原样载入。仓库另有 `starVLA/model/modules/tokenizers/spatiotemporal_physical_tokenizer.py`，输入是 Qwen 空间特征差分，虽然也是 PhiZero 风格时空结构，但与 Wan 视频 tokenizer 的 ID 没有互换性。

### D. 导出并合并双流目标缓存

冻结两套 tokenizer 后，按同一完整样本键、物理码**缓存起点 stride 1**导出 `action_token_ids [N,N_a]` 与 `physics_token_ids [N,64]`；物理 ID 必须在 `[0,24999]`，动作 ID 必须在其独立的 `[0,K_a-1]`。现有 `prepare_phizero_libero_videos.py --stride 1` 覆盖全部决策起点，`export_phizero_libero_token_ids.py` 从这些 clip 导出物理 NPZ；它可单独加载 64 码 tokenizer，不必同时加载 5B 视频 DiT。`merge_multitask_dual_token_targets.py` 默认严格按键合并；若动作 tokenizer 只接收完整 8 步，则显式加 `--drop-unmatched-physics` 丢弃没有动作目标的尾部物理行并记录数量，**不要**传会要求 `future_valid_mask` 全真的 `--physics-feature-cache`。LIBERO 动作缓存仍需另行导出，且必须使用物理缓存同样的 suite-qualified task 名和 task ID 编排；将两套 tokenizer checkpoint 路径与 SHA、动作归一化统计、视频采样规则、split 版本同时固化。训练配置使用 `discrete_token_cache_drop_missing: true`，记录过滤前后样本数、补齐窗口占比；不重复末尾动作来伪造动作监督。

缓存至少要满足现有 loader 的字段约定：`action_token_ids`、`physics_token_ids`、`task_ids`、`task_names`、`episode_ids`、`base_indices`。当前物理导出器按 `suite/<原 task 文本>` 排序生成 `task_names` 与 `task_ids`；动作导出器须完全复用这份映射，不能自行重新编号。`base_indices` 必须是原 LeRobot dataset 使用的 step 索引，视频 exporter 和动作 exporter 不能各自重新编号。合并后随机抽几条样本回看原视频帧、动作 chunk 和两个码的解码结果，检查起点一致。全量 stride 1 视频窗口约 273,465 段，磁盘和离线编码耗时明显高于 stride 8 tokenizer 训练集，正式导出前需预估容量。

当前代码入口的最小调用顺序（路径需按实际部署替换，正式训练前先以 `--max-episodes` / 少量窗口 smoke）：

```bash
python scripts/physical_tokens/prepare_phizero_libero_videos.py --data-root "$LIBERO_ROOT" --output-dir "$TRAIN_CLIPS" --stride 8
# 4 卡启动 train_phizero_global_batch.py，核心参数如下；原始 FSQ、Wan VAE/DiT 路径仍需传入。
# --csv "$TRAIN_CLIPS/train.csv" --queries-per-transition 8 --quantizer fsq
# --global-batch-size 8 --gradient-accumulation-steps 2 --epochs 5
# --max-train-steps 10000 --save-every 5000
python scripts/physical_tokens/prepare_phizero_libero_videos.py --data-root "$LIBERO_ROOT" --output-dir "$DENSE_CLIPS" --stride 1
python scripts/physical_tokens/export_phizero_libero_token_ids.py --manifest "$DENSE_CLIPS/manifest.jsonl" --checkpoint "$PHIZERO_64_CKPT" --output "$PHYSICS_NPZ" --split train
```

其中 `python` 必须是已装 PhiZero/DiffSynth、PyTorch 和 ffmpeg 的环境；`$PHIZERO_64_CKPT` 须指向真正续训得到的 64-query safetensors，而不是原版 256 码 checkpoint。视频训练和导出必须使用相同的尺寸与 motion resize 设置。联合模型现接受 `physics_vocab_size: 25000`、`physics_token_count: 64` 和动作码数 8/32，但**尚没有可直接运行的 LIBERO 双流 YAML/训练结果**，需按 E 节完成动作缓存与 checkpoint 接线。

### E. 改模型并分两阶段训练

1. 建立 LIBERO 专属 AR 配置。使用 Qwen3-VL 对应的基座（现有 LIBERO 示例是 Qwen2.5 的另一套架构），按 `N_a,64,K_a,25000` 扩词表、建专属输入/输出表，确保 action/physics token ID 连续且互不重叠。独立双流 VLM **LoRA + 新动作/物理表训练 10,000 个 optimizer steps**，不是只训练 LoRA 矩阵；**训练采用 teacher forcing**：输入历史真值 token，分别计算动作/物理 CE；这是选定的训练方式，不要求先加入 scheduled sampling 或整段 rollout 训练。验证仍同时测 teacher-forced CE/top-1 和真正自由生成的码准确率、动作码变化位置准确率、整段准确率、生成延迟。不能把不同词表大小的 CE 直接排序。
2. 建立 LIBERO 专属联合策略。`QwenPIv4DualAR32x64` 目前断言 `(32,64)` 和 `(525,2048)`，`TokenStateMemory` 长度也写死 32/64；将这些全部改成配置值，尤其物理词表改为 **25,000**、物理 memory 长度保持 **64**。扩展 25,000 个专属 token/输入输出表会增加 VLM 参数与显存，需实测。连续动作专家设 `action_dim=7`、`future_action_window_size=7`，输入状态/图像数量与 LIBERO 样本一致。`initialize_expanded_vocab_pretrained()` 严格核对底座参数键，`initialize_trainable_checkpoint()` 严格核对双流 AR 的 LoRA 和码表；若 LIBERO 基座不同，需从兼容的 LIBERO Qwen3/π 基座重新训，不可直接加载 RoboTwin trainable-only 权重。
3. 联合训练从 LIBERO 基线 warm start、加载**同一 tokenizer 版本**训练出的 10k-step AR 权重，再训练 VLM、桥接层和 DiT，**额外训练 10,000 个 optimizer steps**；若联合配置保留 `lora_adapters` 可训练，VLM LoRA 在两阶段都更新，不能把总训练写成“只有 10k”。现有 RoboTwin 联合 YAML `scripts/physical_tokens/qwenpi_v4_dualar_32x64_concat_20k.yaml` 只用作字段模板，**其中的 20k 不照搬**；绝对路径、RoboTwin 数据混合、14 维动作、32 步、checkpoint 路径也必须替换。先完成一个 batch 的前后向、保存/加载、无未来信息的自由生成及小规模 LIBERO 闭环烟测，再跑全套。

两个阶段的 10k 是**各自独立的 optimizer-step 上限**，不是把两阶段合计为 10k，也不改变物理 tokenizer 自己的 10k 预算。全局 batch 还没有为 LIBERO 的 K25,000 head/联合模型做显存验证，不能把 RoboTwin 的 global batch 128 直接当成已可运行配置。按当前“8 步动作目标必须完整”的口径，四套数据在划分前最多约 261,614 个可联合的决策起点；若假设 episode 级留出约 10%，训练样本约 235,000，则任一阶段在 global batch 128 时 `10,000×128/235,000≈5.4` 个 epoch，在 global batch 64 时约 **2.7** 个 epoch。正式记录须分别用 `10,000×实际 global batch/N_train` 重算，并在 5k/10k checkpoint 检查物理码自由生成、LIBERO 成功率和过拟合迹象；10k 是当前实验预算，不预设一定收敛。

配置修改点可按以下伪配置逐项审查；它**不是可直接运行的 YAML**，物理配置已锁定为 64×K25,000，`N_a/K_a` 由动作 tokenizer 结果确定：

```yaml
framework:
  name: QwenPIv4DualAR32x64  # 先参数化类中的 32/64、525/2048 断言与固定长度
  action_model: {action_dim: 7, future_action_window_size: 7}
  action_token_reasoner: {token_count: N_a, vocab_size: K_a}
  dual_autoregressive_reasoner:
    physics_token_count: 64
    physics_vocab_size: 25000
    separate_heads: true
    strict_modality_tying: true
  dual_ar_policy: {condition_source: predicted}
datasets:
  vla_data:
    data_mix: libero_all
    action_type: delta_qpos
    discrete_token_cache: <LIBERO 联合 NPZ>
    discrete_token_cache_drop_missing: true
trainer:
  max_train_steps: 10000  # 本伪配置对应联合阶段；独立 AR 配置也设 10000
  pretrained_checkpoint: <匹配 Qwen3/π 结构的 LIBERO 基线>
  dual_ar_trainable_checkpoint: <使用同一批 LIBERO token ID 训练的 AR 权重>
```

## 4. 迁移前需要修正或讨论的设计点

1. **物理码协议不一致。** 旧双流联合策略是 64×K2048 的 Qwen 特征差分码；本方案是 **64×K25,000 的 PhiZero FSQ 视频码**，仅长度相同，语义与词表都不同。K25000 的配置/词表/head 已放开，但仍须实际生成并对齐视频 ID、从头训练对应 AR/联合权重；不能把旧 64×K2048 checkpoint 当作这种新码的“续训”。PhiZero 原 256 码 checkpoint 也必须先经过 32→8 query 适配与续训。
2. **teacher forcing 是训练选择，不能误写成推理条件。** 联合模型的 `condition_source: predicted` 用 teacher-forced 序列算出的 logits 取 argmax，后续位置的 hidden state 仍可看到先前真值码；这仍是 teacher-forced 训练，不是完整自由生成。按本方案可以保留该训练方式，**不把 scheduled sampling/rollout 蒸馏列为上线前置条件**；但要在报告中明确标注，并用无未来信息的自由生成评测 LIBERO 闭环成功率。若离线 CE 良好但闭环退化，再讨论训练条件改造。
3. **AR 序列成本。** 采用 64 物理码后需 `N_a+64` 次自回归输出；若 `N_a=8/32`，分别为 72/96 码。256 物理码仅作对照，会额外增加 **192 次串行解码**，不作为默认部署方案。LIBERO 每 8 步重新规划一次，需记录完整 VLM+DiT 推理的 P50/P95 延迟、GPU 峰值和任务成功率，避免只看离线重建。
4. **concat 的收益未由结构保证。** 新 fuser 的输出投影零初始化，起点接近原 π 行为；这保证可作为残差训练，并不保证物理 token 有用。至少做 `π only`、`π + action`、`π + action + physics`，保持同基座、相同步数/随机种子/评测初态；物理码随机打乱或置零可检查依赖性。
5. **checkpoint 与服务端接线。** 旧 `examples/LIBERO/eval_files/run_policy_server.sh` 指向历史 checkpoint 和环境；新联合框架的 `baseframework.from_pretrained()`、训练保存的 config/统计、`predict_action()` 返回 `normalized_actions [B,8,7]`、评测端反归一化都要端到端验证。离线 token 指标不能替代 LIBERO rollout 成功率。

## 5. 交付/验收标准

最小可交付包括：LIBERO episode split 与缓存 manifest；LIBERO 动作 tokenizer 和 **64 码 PhiZero FSQ** checkpoint 及独立留出报告；逐样本键对齐的双流缓存；AR/联合模型配置与可加载 checkpoint；teacher-forced 训练指标、自由生成与服务端接口烟测；spatial/object/goal/10 四套各任务成功率、固定初态和同条件 π 基线。报告同时给出 64 码 RGB 留出重建（含首帧有/无机械臂的后续帧可视化）、动作/物理 token 预测指标、连续动作误差、完整 VLM+DiT 推理延迟与三组消融。若物理码只在训练集重建好、自由预测接近多数码基线或闭环无提升，应先解决可预测性/时间对齐问题，再增加模型规模。
