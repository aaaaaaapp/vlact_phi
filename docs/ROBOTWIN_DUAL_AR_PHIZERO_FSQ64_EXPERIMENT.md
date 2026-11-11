# RoboTwin：Clean+Randomized、PhiZero-FSQ64 + 动作32码 + 双流 AR + concat DiT 实验交接

本页是给后续 Codex 执行实验的**新方案说明**，不替代已有的 K2048/Qwen 特征差分物理码实验，也不表示新方案已训练成功。新方案的物理 tokenizer、策略训练数据均混合 RoboTwin 50 任务的 **Clean + Randomized**；分域验证仍须单独报告。任何新产物放在 `artifacts/robotwin_full50_study/phizero_fsq64_fps16/` 与独立的 run ID 下，**不覆盖**既有动作 tokenizer、K2048 物理缓存、40k AR checkpoint 或 20k 联合配置。

## 1. 当前已确定的部分

| 项目 | 本实验采用/来源 | 不能误用 |
| --- | --- | --- |
| 连续动作 | Clean + Randomized `robotwin_clean_randomized_wrap_32`、`abs_qpos`/`abs`、`action_dim=14`、32 步（`future_action_window_size=31`） | 不套 LIBERO 的 7 维、8 步或 `delta_qpos`；确认两个域的动作 schema/时间对齐一致 |
| 动作码 | 已完成的 EMA-VQ `32×K525`、decay 0.95、dead-code 100、20k；权重 `artifacts/robotwin_full50_study/action_tokenizers/ema_32x525_decay095_dead100_bs128_steps20000/step-00020000.pt` | 可复用已导出的 `action_32x525_ids_vlact_for_dual_ar.npz`，但必须复查末尾 32 步是否完整；不把动作 token 当成最终执行动作 |
| 物理码目标 | 从原始 PhiZero 视频 FSQ checkpoint 迁移，每个 latent transition 32→8 queries；Clean 50 FPS、Randomized 15 FPS 源视频各自按 **16 FPS × 33 帧**采样；常规窗口最多补 8 帧，短于 2 秒的 episode 强制保留两个窗口；产生 `8×8=64` ID，levels `[8,5,5,5,5,5]`，`K=25000` | 与旧 64×K2048 特征差分码只是长度相同，ID/embedding/checkpoint 完全不互换；原 256 码 checkpoint 不可直接当 64 码缓存编码器；16 FPS 与 PhiZero 原始 8 FPS 时间分布不同，迁移效果必须实测 |
| VLM | Qwen3-VL-4B，动作/物理各自输入表与绑权输出 head；teacher-forced CE 训练，推理严格自回归 `32+64=96` 码 | 推理不能传未来真值码；teacher-forced `condition_source: predicted` 也不等同真正自由生成 |
| 连续动作 expert | Qwen3-0.6B，hidden 1024；三路 cross-attention 输出沿特征维 `1024+1024+1024=3072`，经 `3072→256→1024` fuser 残差 | 32/64 是 memory **序列长度**；原 prefix 路径保留，不是直接把物理码拼进 14 维动作输入 |
| 数据划分 | 复用 `docs/robotwin_tokenizer_episode_split.json`：每任务留出 episode `[5,20,25,39,49]` | 先按 episode 划分再采窗口；Randomized 与 Clean 相同数字 ID 不一定同一轨迹，必须加来源域/任务键 |

现有代码已支持 `scripts/physical_tokens/train_phizero_global_batch.py --queries-per-transition 8 --max-train-steps 10000`、VLM/联合模型的 `physics_vocab_size: 25000`。`prepare_phizero_robotwin_fps16.py` 与 `prepare_phizero_robotwin_clean_randomized_fps16_node1.sh` 将两域分别按 50/15 FPS 源时钟采样到 16 FPS；新数据独立写入 `tokenizer_clips_clean_randomized_fps16/`，不覆盖旧 Clean-only 目录。按当前 metadata 和 pad-cap 规则预计 **96,561 train / 10,754 val** 窗口；最终以 exporter 报告为准。PhiZero64 稠密物理 ID 缓存、K25000 双流训练权重和闭环结果仍待后续阶段。旧 `prepare_phizero_robotwin_videos.py` 只用于历史 256 码对照，其 8 FPS/拉伸采样不符合本轮协议。

原始权重：`/mnt/jfzn/liujn/models/phizero_model/tokenizer/tokenizer.safetensors`（同目录 `model_config.json`）、`wan2.2-vae/Wan2.2_VAE.safetensors`，以及 `wan2.2-ti2v-5b/` 下的 `diffusion_pytorch_model-00001-of-00003.safetensors`、`diffusion_pytorch_model-00002-of-00003.safetensors`、`diffusion_pytorch_model-00003-of-00003.safetensors`。权重目录目前可读，但正式运行前仍要逐文件校验。训练依赖本地 `PhiZero-main/DiffSynth-Studio` 和 `vlact` 环境。旧 RoboTwin PhiZero FSQ 对比的 256 码、EMA-VQ 15k 图/指标不能当作新 64 码效果；后者必须重新生成 RGB 留出重建。

## 2. 数据与时间协议：16 FPS、补帧上限 8 帧，重建 gate 仍需验证

Clean 源数据为 **50 FPS**、Randomized 源数据为 **15 FPS**；策略 32 步动作跨度在两域分别约 `32/50=0.64` 秒和 `32/15=2.13` 秒。Clean 与 Randomized episode 独立套用相同的 episode ID split，不假定同 ID 是同一轨迹。物理 tokenizer 窗口数按 33 帧、16 FPS 采样网格及最多 8 帧末帧补齐规则统计；策略缓存则还须与各域完整 32 步动作精确 join，不能把 tokenizer 窗口数当成策略训练键数。这里“补帧”严格指目标时间越过 episode 最后一帧后重复末帧；15→16 FPS 采样中因最近邻而重复源帧不算末端补帧。

| 目标 FPS | 33 帧首末跨度 | 训练窗口有补帧 | 平均补帧/33 帧 | 补帧占所有目标帧 | 完整训练窗口 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8（PhiZero 原时域） | 4.00 s | 70.17% | 11.32 | 34.31% | 126,532 |
| 20 | 1.60 s | 25.95% | 2.70 | 8.19% | 314,151 |
| 25（对照） | 1.28 s | 17.50% | 1.53 | 4.64% | 349,990 |
| **16（主线，上限前）** | **2.00 s** | **36.08%** | **4.21** | **12.77%** | **271,187** |
| 32（短跨度对照） | 1.00 s | 10.08% | 0.66 | 2.01% | 381,490 |
| 50（动作同频对照） | 0.64 s | 0.53% | 0.01 | 0.02% | 421,990 |

上表及这一段的 **424,240/48,047** 是旧 Clean-only `action_all50_wrap32.npz` 的候选策略起点统计，不是本轮混合视频 tokenizer 的数据量，也不能直接代表 Randomized 覆盖率。旧起点套用 16 FPS/8 帧筛选后预计训练 **325,389**、留出 **37,063**；新主线需要分别重建两域动作目标缓存，再按完整 32 步动作有效性、物理窗口规则和 `source_domain` 精确 join。排除的终止阶段需核验接触/成功分布。16 FPS 保留 2 秒物理未来，比 32 FPS 更接近 PhiZero 原 8 FPS 时间间隔；**这不是 PhiZero 16 FPS 迁移已成功的证据**，需看分域留出 RGB 重建、运动/接触质量和自由生成。

每个样本键必须是 `(source_domain, task_name, episode_id, base_index)`；Clean/Randomized 共用固定的 50 任务 task-ID 顺序，但 domain 是键的一部分，不能把相同 episode 数字错误 join。`base_index` 保持 LeRobot 原 step，不允许视频 exporter 与动作 exporter 各自重编号。训练时输入当前 `cam_high`、指令、状态；未来帧只作 tokenizer/AR 真值，不能泄漏到 VLM prefix 或闭环推理。训练、缓存、可视化统一相机、旋转、颜色通道、尺寸/letterbox 与归一化；需回看两域首帧确认机械臂朝向。不要把 LIBERO 的 256×256 letterbox 规则原样套到 RoboTwin 640×480 视频上。

两域都以各自 metadata 声明的源 FPS 构造最近邻采样索引 `r_j=s+floor(source_fps·j/16+0.5)`，`j=0…32`；Clean 50 FPS 的末 offset 是 100，Randomized 15 FPS 的末 offset 是 30。目标 FPS 固定 16；Randomized 的 15→16 FPS 最近邻可能重复源帧，但这不计作 episode 末端 padding。定义 `padded_count=sum_j [r_j>=N]`，常规窗口满足 `padded_count<=8`，源索引截到 `N−1` 后只在越过 episode 末尾时重复最后一帧。窗口起点 stride 分别为 Clean 20 源帧、Randomized 6 源帧（均约 0.4 秒），并补入最大合法起点以覆盖尾部；合法起点上限用第 25 个目标 offset（允许最后 8 个目标帧补齐）计算。若整条 episode 短于 33 帧时间跨度，则强制保留 `s=0` 与一个 stride 起点，即使 padding 超过 8；Clean 当前最多补 15 帧，Randomized 当前没有此类 episode。若 episode 连一个 stride 都不够，预处理器报错而不伪造不存在的第二个源起点。记录 33 个源索引、真实/末端补帧数、`forced_short_episode`、`episode_is_short`、来源域/FPS、源视频 SHA 与 task/episode/base key。训练器的 `--num-frames 33` **不是** FPS 参数，采样语义以 exporter 和 manifest 为准。

按当前两域 metadata 与固定 split，Clean tokenizer 窗口预估 train **19,606** / val **2,216**，其中短 episode 强制窗口分别 **334/38**；Randomized train **76,955** / val **8,538**，当前无短 episode。合计 train **96,561** / val **10,754**；Clean 常规窗口最多补 8 帧，forced short 最多 15，Randomized 目前最多补 7。10k 步×全局 batch **16** 约等效 **1.66** 次混合训练集遍历。最终数值须由新 manifest 复核；可额外筛首帧含机械臂的窗口，但须记录采样权重，不能让静止尾段堆高样本数。

冻结动作 tokenizer 后，须为 Clean 与 Randomized 分别编码 32×K525 动作 ID；旧 `action_all50_wrap32.npz` 及旧缓存统计是 Clean-only，不能直接当成混合缓存，也不能用相同数字 episode ID 跨域 join。物理 ID 与动作 ID 按 `(source_domain,task_id,episode_id,base_index)` 精确 join，只保留 `action_valid_mask.all()` 且满足物理 pad-cap 或 forced-short 例外的起点。重新报告分域候选数、有效动作数、join 后保留率、终止阶段排除分布；不能把旧 K2048 缓存改名或沿用旧 Clean-only 行数。

### 时间跨度 gate（正式训练前决定）

在固定留出 episode 的**合格窗口**上按 Clean/Randomized、完整/末帧补齐、首帧有无机械臂分别报告：未来帧 RGB 重建视频/帧板、flow loss、FSQ 码熵/活跃率/重复、运动量及首/末帧差异；短例外和被 pad-cap 排除的终止阶段单独列统计。首帧是**给定条件**，只看首帧清晰不能证明重建好。两域都须核对时间戳、动作对齐、相机与外观；Randomized 15 FPS 对应 16 FPS 目标会有少量最近邻重复，需检查其运动质量。若分域留出重建或自由生成仍主要学静止、物体/机械臂消失，先看首帧机械臂覆盖与重采样索引，再决定单独训练 32/50 FPS 对照；改变 FPS 必须独立续训并重导缓存/AR 权重。

## 3. 逐阶段执行与必要修改

1. **盘点/冻结基线。** 记录 Clean π 基线成功率、100k 权重及 dataset statistics、旧 32×K525 动作 tokenizer 重建和旧 64×K2048 AR/联合指标。固定 `docs/robotwin_50task_list.txt`、episode split、相机与代码 commit；所有新文件独立路径。开始任何长训前先抽查 10 条同键动作/视频样本，确认 `[32,14]`、`action_valid_mask.all()`、图像时间戳和夹爪方向。
2. **新 RoboTwin 视频 exporter。** 运行 `prepare_phizero_robotwin_clean_randomized_fps16_node1.sh`；旧目录与旧训练结果保留。先对两个域各做至少 1 episode smoke，再检查分域 FPS、索引、pad cap、short 标志与首末帧覆盖；预期 96,561/10,754 窗口，最终以 preparation report 为准。
3. **PhiZero FSQ64 从头续训。** 原始 256 码 FSQ 权重严格载入，训练器按 `[2,6,10,14,18,22,26,30]` 从每 transition 的 32 query 初始化 8 query，保持 `33 frames`、`K25000`、`fsq_entropy_weight=0.05`，不传 `--fresh-phi-zero-components`。八卡全局 batch **16**、per-rank batch 2、gradient accumulation **1**、gradient checkpointing 开、10,000 optimizer steps、5k/10k 保存；运行 `run_phizero_robotwin_fsq64_fps16_node2.sh`。混合 train 预计 96,561 窗口，约 6,036 steps/epoch；10k steps 约 1.66 epochs。必须留新 checkpoint、严格重载/ID 范围检查、分域留出 RGB 重建和未来机械臂/接触可视化。新 run 自动接 offline W&B 镜像；旧 Clean-only run 已在 **162/10,000 steps** 停止、未产生 checkpoint，run `20261109_151915` 的日志/W&B 只保留作失败历史，不续训。
4. **冻结物理码，导出 Clean+Randomized 稠密缓存。** 输出 `physics_token_ids [N,64]`、domain-aware key、每条窗口真实/补帧 count、`forced_short_episode`、各域 source FPS、target FPS、pad cap、short starts 和 checkpoint SHA。可参考 `export_phizero_libero_token_ids.py` 的“只加载 tokenizer、不加载 5B DiT”方式，但不能直接用 LIBERO 的 manifest/camera。分别编码 Clean/Randomized 动作目标，与物理码按 `(source_domain,task_id,episode_id,base_index)` 精确 join；完整动作有效性必须验证，不能把 Clean-only 动作缓存误用于 Randomized，也不能混入旧 K2048 物理 ID。
5. **独立 AR 10k。** 从 RoboTwin-All Qwen3/π 100k 基座初始化，**不能**接着训旧 K2048 的 40k AR checkpoint；新建 25,000 行物理专属绑权表，动作表 K525，LoRA rank/alpha 32。保留 32 动作→64 物理目标顺序、teacher forcing 两流均值 CE、固定 96 次自由生成。用新缓存训练 10k optimizer steps，保存 5k/10k。先 1 batch、100 step 烟测 K25k logits/显存/保存加载，再看分域/补帧比例及真自由生成准确率，不能只报低 teacher-forced CE。旧 267 个预留词表行的初始化问题仍需检查；`uniform_physics_init: true` 可作起点，但必须报告真实初始化、不能加载旧 K2048 训练权重。
6. **concat 联合 10k。** 另从相同 RoboTwin-All 100k π 基座和**本轮 K25k AR 10k** trainable-only 权重 warm start；DiT continuous action expert 保持 14 维/32 步，fusion bottleneck 256。新配置改 `physics_vocab_size: 25000`、新缓存、新 checkpoint、`max_train_steps: 10000`；不复用旧 `qwenpi_v4_dualar_32x64_concat_20k.yaml` 的 20k run ID/源权重。VLM LoRA 若在联合阶段继续可训练，则它是两阶段连续更新，共计最多 20k 次更新，不是“LoRA 只训 10k”。先验证加载边界、teacher-forced 联合反向、无未来真值的自由生成、服务端 `[B,32,14]` 输出和统计反归一化，再开长训。
7. **闭环与消融。** 同初态/同任务报告 Clean、Randomized 及合计的 π-only、π+action、π+action+physics 成功率与逐任务变化；补充物理码置零/打乱、自由生成 vs oracle 上限。记录 VLM 96 码串行生成 + DiT 4-step 的端到端 P50/P95 延迟、峰值显存；离线重建好或 teacher-forced 准确率高都不能替代闭环收益。

## 4. 新实验配置：明确值与待验证值

已另建**两个新的**配置模板：`scripts/physical_tokens/qwen_dual_ar_32a64p_phizero_fsq64_robotwin_10k.yaml` 与 `scripts/physical_tokens/qwenpi_v4_dualar_32x64_phizero_fsq64_robotwin_10k.yaml`。它们基于旧 AR/concat 配置改成 K25000、**fps16/padcap8/short2 独立缓存/run ID**、每阶段 10k，旧 K2048 YAML 保留原样。FPS、补帧上限和短 episode 例外不由这两个训练 YAML 执行，而应由物理目标缓存 manifest 的 `source_fps=50,target_fps=16,num_frames=33,max_padded_frames=8,short_episode_starts=[0,20]` 与 SHA 锁定；**缓存与训练入口的断言尚待后续实现**，在完成前不能启动长训。新模板在新缓存/权重存在前**不得启动长训**；其中 batch、loss 权重和显存开关仍是候选，必须先过 smoke。配置要点如下：

| 字段 | AR 阶段 | 联合阶段 | 状态/理由 |
| --- | --- | --- | --- |
| `framework.name` | `QwenDualAutoregressiveReasoner` | `QwenPIv4DualAR32x64` | 已有实现 |
| `action_token_reasoner.{token_count,vocab_size,token_offset}` | `32,525,151669` | 同左 | 现有 RoboTwin 动作 checkpoint/词表 |
| `dual_autoregressive_reasoner.{physics_token_count,physics_vocab_size}` | `64,25000` | 同左 | PhiZero FSQ64 新协议 |
| `separate_heads / strict_modality_tying` | `true / true` | `true / true` | 专属输入表与输出 head 绑权 |
| `prompt_template` | `Predict 32 robot action tokens, then 64 future physics tokens` | 完全相同 | 训练/推理顺序一致 |
| `dual_ar_policy.{condition_source,fusion_bottleneck}` | 不适用 | `predicted,256` | `predicted` 训练仍有 teacher forcing 历史；必须另测自由生成 |
| `action_model.{action_dim,future_action_window_size,num_inference_timesteps}` | 不适用 | `14,31,4` | 现有连续动作 π 配置 |
| `datasets.vla_data.{data_mix,action_type,action_mode}` | `robotwin_clean_randomized_wrap_32,abs_qpos,abs` | 同左 | Clean/Randomized 等任务权重；确认域路径与动作时间戳，评测结果分域报告 |
| `datasets.vla_data.discrete_token_cache` | 新 `phizero_fsq64_fps16/...k25000_clean_randomized_padcap8_short2_targets.npz` | 同一 NPZ | 必须检查分域源 FPS、目标 FPS、补帧上限/短例外、tokenizer SHA、码域与键数；键包含 `source_domain` |
| `exclude_episode_ids` | `[5,20,25,39,49]` | 同左 | 与旧固定 split 一致 |
| `trainer.pretrained_checkpoint` | RoboTwin-All 100k 完整权重 | 同一 π 100k 完整权重 | 不继承 K2048 trained physics rows |
| `trainer.dual_ar_trainable_checkpoint` | **不设置** | 本轮 AR 10k 权重 | 旧 K2048 trainable-only checkpoint 不兼容 |
| `trainer.max_train_steps` | `10000` | `10000` | 两阶段各自预算，不合并计算 |
| `trainer.save_interval` | 建议 `5000` | 建议 `5000` | 5k/10k 比较；正式前检查存储 |
| `per_device_batch_size / gradient_accumulation_steps` | 起点 `8 / 4`（4 GPU 时 global 128） | 起点 `8 / 4`（global 128） | **未在 K25k 实测**；先显存/吞吐 smoke，OOM 则降低 microbatch、调整累积保持 global 128；再不行降低 global 并重算 epoch |
| `dual_ar_policy.{action_loss_weight,physics_loss_weight}` | 不适用 | 起点 `1e-3,1e-4` | 旧 K2048 数值不能视为 K25k 最优；看 flow/CE 梯度尺度再调或消融 |

按当前 metadata 预估，物理 tokenizer 混合 train 有 **96,561** 个窗口：Clean **19,606（20.3%）**、Randomized **76,955（79.7%）**；val 共 **10,754**（Clean 2,216、Randomized 8,538）。全量窗口直接拼接意味着 Randomized 在物理 tokenizer 的均匀行采样中约占 80%，不是 1:1 域平衡；先按这一明确默认方案从头训练，若 Clean 留出重建明显退化，再增设域平衡 sampler 对照，不要悄悄复制或丢弃样本。global batch 16 时约 `ceil(96561/16)=6,036` optimizer steps/epoch；10k steps 约 **1.66 个等效 epoch / 160,000 个窗口呈现**，最终按实际 manifest 复算。策略阶段使用命名 mixture `robotwin_clean_randomized_wrap_32` 的等域、等任务权重，不能拿物理 tokenizer 的 20/80 窗口比例推断策略 dataloader epoch。

### 待决策事项（先做小实验，不要暗自定案）

1. **2 秒跨度与训练覆盖的取舍。** 16 FPS 比原始 8 FPS 的帧间隔短 2 倍，且 2 秒比动作 chunk 长约 1.36 秒；8 帧上限加短 episode 两窗例外后仍丢弃约 23.30% 完整动作起点。先报告 16 FPS 留出 RGB 重建、码熵、运动覆盖、AR 自由生成、短 episode 例外与被丢弃起点的阶段分布；若不过 gate，用 32/50 FPS 独立续训对照，不可直接换采样却沿用 ID/权重。
2. **Clean/Randomized 的采样比例与对齐。** tokenizer 默认全量窗口均匀行采样（约 Clean 20% / Randomized 80%）；策略 mixture 等域、等任务权重。若 tokenizer 的 Clean 留出重建退化，再比较 1:1 域平衡 sampler；不能只看总指标，需分别核 FPS、时间戳、首帧机械臂比例、动作对齐与视觉风格，Randomized 改善不能替代 Clean 闭环改善。
3. **首帧机械臂覆盖与 RGB 可信度。** 15k EMA-VQ 图不是 64 FSQ 证据。新留出面板同时展示首帧有/无机械臂、源帧 vs 重建未来帧、几种训练步数；手和物体接触、消失、夹爪运动都要看。
4. **K25k 的工程成本。** 25k 物理专属表约 `25000×2560≈64M` 参数（bf16 权重约 128 MiB，优化器/梯度另计），并增加 64 次大词表分类与 AR 生成。先做 1/100 step、保存/重载、自由生成延迟与显存 smoke；若失败不可悄悄缩 K 到 2048，否则已不是同一 FSQ 标签协议。
5. **32 个动作码的实际价值。** 动作 ID 相邻重复率在旧缓存约 92.27%；新报告必须给码变化位置准确率、动作解码误差及 8 码对照，不能用总体 top-1 掩盖重复。若真机/仿真延迟不合要求，再考虑 8 动作 + 64 物理，但须从该 8 码 tokenizer 重新导缓存、训练 AR/联合权重。

## 5. 后续 Codex 的停机条件与交付

先做到 **混合数据/shape/权重 smoke → 物理 10k → 分域留出 RGB/padding gate → 两域物理/动作 ID 精确 join → AR 10k → 自由生成 gate → 联合 10k → 分域闭环消融**。前一 gate 不通过，不应自动启动后一轮昂贵训练。每阶段留独立 manifest：代码 commit、原始/输出 checkpoint SHA、source domain/FPS、数据 split、任务/episode/base key、采样 stride/首末索引、`max_padded_frames=8` 与 short exception、真实/补帧分布、图像预处理、全局 batch、microbatch、累积、实际优化步和评测种子。新缓存需断言各域 `source_fps` 正确、`target_fps=16`、常规 `padded_count<=8` 或有效 short exception、64×`[0,24999]`、动作 32×`[0,524]`，并与新 YAML 精确一致。报告须区分**已运行结果、配置候选、待实现代码**；Clean 与 Randomized 都完成闭环后再讨论总体结论。
