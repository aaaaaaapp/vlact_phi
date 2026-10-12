# VLAct 离散动作与物理语言条件研究计划

> 状态：D41 predicted-distribution head 已通过四项 gate，低成本 DiT 验证已授权  
> 日期：2026-09-28  
> 代码副本：`/mnt/jfzn/liujn/code/VLAct-PhysToken`  
> 上游代码：`/mnt/jfzn/liujn/code/VLAct-main`

> 当前主配置固定为 `4 action tokens + 256 physics tokens`、EMA-VQ K256。D36 8000-step dual VLM 的
> physics accuracy 为 11.331%；D41 使用同一 VLM 的 predicted action/physics 分布重训真实 residual head，
> 在完全排除 head 训练/选模的固定验证 episodes 上，相对 action-only 改善 3.284%，相对 episode-local shuffled
> 改善 8.290%，四项 gate 全过，`dit_authorized=true`。详见 21.49--21.55。

## 1. 目标与核心假设

目标是在 VLAct continued-pretraining checkpoint 之后、单个下游 benchmark 的训练阶段，增加一个分层的离散推理接口：

1. VLM 根据当前图像、语言指令和本体状态，自回归预测离散动作 token；
2. VLM 同时预测描述未来可见动力学的离散 Physical Language token；
3. 两类 token 的 embedding/hidden state 与原始 VLM 多模态 prefix 一起作为连续动作头的条件；
4. DiT/flow-matching action head 保留连续动作生成能力，离散 token 只提供粗粒度计划和动力学先验，而不替代连续控制。

核心假设是：

- `H1`：动作 token 提供低频、长时域动作结构，DiT 负责量化残差和细粒度控制；
- `H2`：Physical Language token 提供动作之外的未来视觉变化、接触和物体运动信息，改善仅由当前 VLM feature 条件化时不足的动力学建模；
- `H3`：联合条件比任一单独条件更有效，但前提是 token 在部署时可由 VLM 准确预测；
- `H4`：EMA-VQ 和 FSQ 的稳定性、利用率及可预测性不同，需要在相同 token budget 下比较。

这不是“增加 token 一定不降点”。新增条件可能因预测错误、teacher-forcing 暴露偏差、上下文稀释、码本塌缩或捷径学习而降低性能。因此模型必须具备关闭新增条件后退化为原 QwenPI 基线的路径，并把 oracle/predicted token 差距作为主要诊断量。

## 2. 与已有工作的关系

### 2.1 VLAct 可直接继承的部分

VLAct 已经提供：

- Qwen3-VL-4B continued-pretraining backbone；
- 下游 benchmark 的 LeRobot 数据读取、动作归一化和评测接口；
- 连续 QwenPI / QwenPI_v4 flow-matching 动作头；
- VLM hidden states 到 action head 的条件通路；
- checkpoint warm start、模块冻结、分层学习率和 Accelerate/DeepSpeed 训练框架；
- FAST action tokenizer 的早期适配代码，可借鉴接口，但不能直接视为已经接通的功能。

VLAct 的论文结果也给出明确警告：纯离散动作解码存在精度损失，且单一 decoder 会造成 decoder lock-in。因此本方案保留连续 DiT，并把离散动作预测设计为辅助监督和条件，而不是最终控制输出。

### 2.2 PI0.5 可借鉴的部分

- 使用 VLM prefix 表示当前视觉、语言和状态；
- action expert 通过 flow matching 预测连续动作；
- 推理时 prefix 只计算一次，并供多个去噪步复用；
- 离散高层表示与连续低层动作生成解耦。

当前 `QwenPI_v4` 已经是 Qwen-VL prefix + Qwen3-0.6B action expert 的 PI0.5 风格实现，可优先作为主干。但 MVP 也应保留现有轻量 `QwenPI` 作为低成本选项，避免 action expert 本身成为计算瓶颈。

### 2.3 PhiZero 可借鉴与不可照搬的部分

PhiZero 的 Physical Language 是未来视频的离散运动表示，不是普通自然语言：

- 视频 tokenizer 从 33 帧视频提取 motion feature；
- 使用相邻帧 QFormer 和 FSQ；
- 原配置对每段视频输出 256 个 token，词表大小 25,000；
- reasoner 只看第一帧和 caption，自回归预测未来 motion token；
- token 再作为视频 DiT 的条件。

机器人控制是否应直接使用 256-token/25k-vocab 配置必须由容量收益与在线延迟共同决定，不能先验排除。
PhiZero 的 256 个符号来自 33 帧经时间下采样后的 16 个相邻时间段，每段由 16 个 learned QFormer
queries 提取运动表示，即 `16 transitions x 16 queries`。我们借鉴“未来视频 -> 离散物理语言 ->
当前观测 reasoner -> 下游 DiT 条件”的分解方式，并同时评估短序列和 PhiZero 量级序列。

## 3. 建议架构

### 3.1 数据流

```text
当前多视角图像 o_t + 指令 l + 状态 s_t
                  |
                  v
             VLAct Qwen3-VL
                  |
       autoregressive token generation
                  |
      <boa> z_action <eoa>
      <bop> z_physics <eop>
                  |
       token embedding / hidden states
                  |
原始 VLM prefix + action condition + physics condition
                  |
       QwenPI(v4) flow-matching DiT
                  |
       连续动作 chunk a_t:t+H
```

建议第一版输出顺序固定为 action token 后接 physics token，以符合当前设想。顺序本身是后续消融项，因为 `physics -> action` 可能更符合先预测动力学再决定控制的因果分解。

### 3.2 条件融合

MVP 不直接重写 DiT block，而是在进入 action expert 前构造统一 prefix：

```text
C = concat(C_vlm, gate_a * P_a(E_a[z_a]), gate_p * P_p(E_p[z_p]))
```

- `E_a/E_p`：各自码本 embedding 或 VLM 对应 token 的 final hidden state；
- `P_a/P_p`：投影到 action expert cross-attention dimension；
- action/physics type embedding 用于区分语义；
- padding mask 必须随条件传入；
- `gate_a/gate_p` 使用很小初始化，并支持运行时显式 bypass；
- condition dropout 独立作用于两路，建议初始 `p=0.1~0.2`。

注意：追加零值 token 仍会改变 cross-attention softmax 分母，所以“gate=0”并不严格等价于
基线。严格回退必须完全不追加离散 token（显式 bypass），或使用独立且输出投影为零初始化
的 attention residual branch。MVP 使用显式 bypass，并通过单元测试验证原 prefix 张量不变。

### 3.3 动作 tokenizer

输入为已按 VLAct 当前统计量归一化、mask 后的动作 chunk `[H, D]`。建议：

- 时序 encoder：小型 1D Conv 或 2~4 层 Transformer；
- 下采样到 `N_a=4/8` 个 latent；
- 主方案：EMA-VQ，初始 `K_a=256` 或 `512`，embedding dim `64/128`；
- 对比方案：FSQ，在接近相同有效词表大小与 token 数下比较；
- decoder 重建完整连续动作 chunk；
- loss：有效动作维度上的 L1/Huber + delta/velocity loss + gripper BCE + 周期关节 wrap-aware loss；
- EMA-VQ 附加 commitment loss、dead-code 重启和 usage entropy 监控。

动作 tokenizer 的目标不是只追求重建最低误差。还应控制 token 序列足够短、可被 VLM 预测，并验证 token 是否跨场景表达相似的动作结构。

### 3.4 Physical Language tokenizer

定义：Physical Language token 表达执行 action chunk 后的可见未来运动和交互结果，不等同于 action token，也不编码不可观测的模拟器特权状态。

MVP 输入建议为与 action horizon 对齐的同一相机视频：`[o_t, o_{t+k1}, ..., o_{t+H}]`。为降低外观捷径，编码目标应偏向 motion：

- 相邻帧/首末帧视觉 feature difference；
- temporal QFormer 或小型 temporal transformer；
- 可选 optical-flow/feature-delta 辅助目标；
- pooled-feature 基线扫描 `N_p=4/8/16/32/64/128/256`；时空对齐版本固定检查 256 tokens；
- 主方案 EMA-VQ，初始 `K_p=512/1024`；
- 对比方案 FSQ；
- decoder 优先重建未来 frozen-vision feature 或 feature delta，而不是第一版直接重建 RGB 视频。

选择 feature reconstruction 而不是完整视频 DiT，是为了小成本验证 token 是否含有对控制有用的未来动力学。只有 token 有效后，才考虑更接近 PhiZero 的视频重建训练。

必须防止三种泄漏：

1. 训练/验证按 episode 和 task 划分，不能随机按 frame 划分；
2. tokenizer 可看未来帧来产生监督 token，但 policy/reasoner 输入绝不能看未来帧；
3. 不使用 benchmark 在部署时不可获得的 privileged state 作为 Physical Language reasoner 输入。

### 3.5 VLM token vocabulary

增加互不重叠的原子 token：

```text
<boa>, <eoa>, <act_0> ... <act_Ka-1>
<bop>, <eop>, <phy_0> ... <phy_Kp-1>
```

必须验证每个符号被 tokenizer 当作一个原子 token，并正确 resize input embedding 和 LM head。词表扩展后的新 embedding 可由对应 codebook projection 初始化，也可以随机初始化后单独 warm up；两种方式作为实现选择，不在首轮同时展开。

## 4. 多阶段训练设计

### Stage 0：冻结基线与数据契约

目的：得到可信的对照组。

- 选择一个 benchmark、一个 VLAct continued-pretraining checkpoint 和一个现有 QwenPI 配方；
- 固定 train/validation episode 划分、action horizon、相机、归一化统计和随机种子；
- 重跑短训练和 offline eval，确认 loss、吞吐、显存、checkpoint load 与部署接口；
- 保存首批固定样本，用于所有 tokenizer 和模型版本的回归测试。

产物：`baseline_manifest.json`、数据 fingerprint、固定 validation IDs 和 baseline 指标。

### Stage 1A：离线训练动作 tokenizer

- 只读取 action/state，不加载大 VLM；
- EMA-VQ 为主，FSQ 为对比；
- 先用 benchmark 的 5%~10% episode；
- 早停依据不是单一 reconstruction loss，而是重建误差、码本利用率和 token stability。

进入下一阶段的建议门槛：

- active code ratio > 20%，且无持续恶化；
- perplexity 不塌缩到极少数 code；
- 连续维度反归一化后的误差明显小于 action 标准差；
- gripper accuracy > 98%；
- token-only decoder 能复原可执行的粗动作趋势。

### Stage 1B：离线训练 Physical Language tokenizer

- 先用单相机、短 horizon、低分辨率 frozen feature；
- EMA-VQ 为主，FSQ 为对比；
- 对同一 clip 的轻微颜色扰动，token 应尽量稳定；
- 对时间打乱或未来帧替换，token 应明显改变；
- 做 nearest-neighbor 可视化，检查 code 是否对应运动/接触而非背景或机器人身份。

门槛：验证集 feature-delta reconstruction 优于无运动/均值基线，码本不塌缩，且 token 对未来时序敏感。

### Stage 2：冻结 tokenizer 并缓存监督 token

- tokenizer 权重完全冻结；
- 为每个训练样本缓存 `action_token_ids` 和 `physics_token_ids`；
- cache key 必须包含 tokenizer checkpoint hash、数据版本、horizon、相机、归一化统计和代码版本；
- 输出采用分片文件并原子 rename，避免中断产生半成品；
- cache 读取失败时 fail fast，不允许静默在线重算造成多卡不一致。

### Stage 3：只训练 VLM token reasoner

目标是先测“当前观测能不能预测这些 token”，不接 DiT。

输入：当前图像、指令、可用 state。目标：两段离散 token。使用 assistant-only autoregressive CE：

```text
L_reason = lambda_a * CE(z_a) + lambda_p * CE(z_p)
```

第一轮建议冻结 vision encoder 和 LLM 下半层，训练上半层、新 token embedding 与 LM head，沿用 VLAct 的 prior-preservation 经验。评估：

- teacher-forced token accuracy；
- free-running exact/token accuracy；
- 编辑距离；
- 预测 token 经 tokenizer decoder 后的动作误差；
- predicted physical token 的 feature reconstruction；
- token 生成延迟。

若 free-running 远差于 teacher-forced，不应直接进入联合训练。先缩短序列、减小词表或使用 scheduled sampling/token corruption。

### Stage 4：Oracle-token 条件 DiT

冻结 VLM 和 tokenizer，使用 ground-truth token 条件训练或微调 DiT。这一步回答最关键的表示问题：如果 token 完全正确，它们是否对连续动作预测有帮助？

对照必须包含：

- `B0`：原始 QwenPI，无新增 token；
- `B1`：action oracle token；
- `B2`：physics oracle token；
- `B3`：action + physics oracle token；
- `B4`：随机打乱 token，作为负对照。

若 `B3` 相对 `B0` 没有稳定收益，不应继续投入 VLM 联合训练；说明 token 表示或融合方式没有提供增量信息。

### Stage 5：Predicted-token 条件与分布对齐

把 oracle token 替换为冻结 reasoner 的 free-running 预测，比较：

- oracle token；
- teacher-forced token；
- predicted token；
- predicted token + condition dropout；
- predicted token + 训练时 token corruption。

推荐逐步增加预测 token 比例，例如 `0 -> 25% -> 50% -> 100%`。若 predicted 条件恶化，gate 应允许模型回退使用原始 VLM prefix。

### Stage 6：端到端联合微调

只有 Stage 4/5 通过后进行：

```text
L_total = L_flow + lambda_a * L_action_token_CE
                 + lambda_p * L_physics_token_CE
                 + lambda_vlm * L_VLM_preserve
```

首轮不让 flow loss 穿过离散采样；tokenizer 始终冻结。建议先冻结 vision encoder 和 LLM 下半层，action head 学习率高于 VLM 10 倍。稳定后再比较全量解冻。

## 5. 最小成本可行性实验

### 推荐 benchmark

优先使用 RoboTwin 的单任务或很小任务子集，原因是本机已有数据处理、checkpoint、offline/official eval 和 QwenPI/OFT 对照记录。为避免挑中异常容易或异常困难的任务，最终任务需根据现有 baseline success rate 选择：建议选成功率约 30%~70%、有明显提升空间且评测方差可控的任务。

当前 `press_stapler` 正在被原仓库评测进程使用，因此不要复用它的输出目录或端口。可以读取结果帮助选任务，但新实验写入 `VLAct-PhysToken/results`。

首个任务已固定为 `click_alarmclock`。本机数据包含 50 条 clean episodes（4252
frames，50 FPS）和 500 条 randomized episodes（42267 frames，15 FPS）。MVP-0
先只使用 clean：两套数据的 FPS 不同，不能直接把同为 32 steps 的 chunk 当作相同物理
时长混训。加入 randomized 前必须按时间重采样或分别设置 horizon。

### MVP-0：无需训练大 VLM

1. 从一个任务抽取 2k~10k action chunks；
2. 训练小型 EMA-VQ action tokenizer 和匹配规模的 FSQ tokenizer；
3. 对 1k 固定验证样本报告重建、码本和稳定性指标；
4. 用 oracle action token 条件微调轻量 QwenPI action head 1k~3k steps；
5. 与相同 step、seed、batch 的无 token baseline 比较 offline action error 和小规模 rollout。

停止条件：oracle action token 都无收益，则先修 tokenizer/融合，不做物理 token和大 VLM SFT。

### MVP-1：加入短 Physical Language

1. 使用当前帧加 3~7 个未来帧，训练 feature-level physical tokenizer；
2. 全局 pooled-feature tokenizer 做 `4/8/16/32/64/128/256` 容量扫描；
3. 做 oracle physics 和 action+physics 条件实验；
4. 做 shuffled-physics 负对照，验证模型不是仅靠增加参数或序列长度获益。

该 pooled-feature 扫描只用于确定当前简化表示的容量上限。另建 PhiZero-aligned tokenizer：在约
0.64 秒 action horizon 内采 33 帧，经时间下采样形成 16 个相邻时间段，每段使用 16 个 queries
cross-attend 保留空间结构的视觉特征，输出 256 个 physics tokens。它与简单地对四个 mean-pooled
frame features 使用 256 个 learned queries 分开报告。

### MVP-2：小规模 reasoner

1. 冻结 tokenizer；
2. 只训练新 token embedding、LM head 和最后 2~4 个 LLM block；
3. 使用 5%~10% benchmark 数据跑 1k~5k steps；
4. 先生成 action token，再生成 physics token；
5. 比较 oracle 与 predicted 条件下 DiT 性能及端到端延迟。

只有 predicted-token 模型在至少 3 个 seed 或足够 rollout 数上超过严格匹配基线，才进入全量训练。

## 6. 实验矩阵与判定

第一轮必要实验：

| ID | Action token | Physics token | DiT condition | 用途 |
|---|---|---|---|---|
| E0 | 无 | 无 | 原始 VLM prefix | 严格基线 |
| E1 | Oracle EMA-VQ | 无 | 是 | 动作表示上限 |
| E2 | Oracle FSQ | 无 | 是 | 量化器比较 |
| E3 | 无 | Oracle EMA-VQ | 是 | 物理表示上限 |
| E4 | Oracle EMA-VQ | Oracle EMA-VQ | 是 | 联合上限 |
| E5 | Shuffled | Shuffled | 是 | 负对照 |
| E6 | Predicted | 无 | 是 | 动作 reasoner 实际收益 |
| E7 | Predicted | Predicted | 是 | 完整方法 |
| E8 | Predicted | Predicted | 关闭 gate | 同 checkpoint 回退检查 |

主指标必须是 benchmark rollout success。辅助指标包括：

- normalized/denormalized action L1、L2；
- endpoint error、rotation geodesic error、gripper accuracy；
- flow matching validation loss；
- action/physics token accuracy、编辑距离、perplexity；
- code utilization、dead-code ratio、commitment loss；
- oracle-to-predicted performance gap；
- 首 token 延迟、完整 token 生成延迟、DiT 推理延迟；
- 失败类型：抓取前定位、接触、闭合、运输、放置、恢复。

“可行”的最低证据不是单次 loss 更低，而是：

1. oracle 条件显著优于 E0；
2. predicted 条件保留大部分 oracle 收益；
3. E7 优于 E6，证明 physics token 有独立增量；
4. E5 不优于 E0，排除仅增加参数/上下文的解释；
5. rollout 提升大于跨 seed 和评测随机性的置信区间。

## 7. EMA-VQ 与 FSQ 的公平比较

需要澄清：“都用 EMA，也可以加 FSQ 对比”不能原样作用于 FSQ。EMA-VQ 的 EMA 更新的是 learned codebook；FSQ 没有 learned embedding codebook，因此不存在同一种 codebook EMA。可以比较：

- EMA-VQ：EMA decay、commitment weight、dead-code reset；
- FSQ：levels 配置、entropy regularization；
- 两者都可额外保存 encoder/decoder 参数的 model-weight EMA，但这与 EMA codebook 是两件事。

公平性约束：相同训练数据、latent 数、近似 bits/token、encoder/decoder 容量、训练 step 和重建损失定义。除 reconstruction 外还要比较 VLM 可预测性，因为重建更好的 token 不一定更容易从当前观测预测。

## 8. 主要风险与规避

### 8.1 Teacher forcing 与部署分布不一致

训练时 GT token 条件很好，部署时一个早期 token 错误会污染后续 token 和 DiT。必须报告 free-running 指标，并使用 token corruption、scheduled sampling 和 condition dropout。

### 8.2 Action token 成为捷径

若 action tokenizer 几乎无损，DiT 只是在解码 action token，Physical Language 不会产生价值。应限制 action token bitrate，并比较 token-only decoder、DiT residual refinement 和移除视觉 prefix 的性能。

### 8.3 Physical token 学到外观而非物理

通过 feature delta、时间打乱对照、跨视角一致性和 nearest-neighbor 可视化诊断。必要时增加 motion mask/optical flow，但不作为 MVP 前置依赖。

### 8.4 多相机 token 不一致

MVP 先固定单个主相机。后续比较共享 token、每视角 token 和多视角融合后 token，不能在第一轮同时引入。

### 8.5 动作语义跨 embodiment 不统一

沿用 VLAct 的 20-D 部分统一 action layout 和有效维 mask。tokenizer loss 必须尊重 mask；否则 padding 维会主导码本。单 benchmark MVP 可先避免跨 embodiment，但实现不能写死 7-D。

### 8.6 生成延迟

256 个 physics token 的自回归生成可能成为在线 VLA 的延迟瓶颈，但在完成实测前不排除。必须同时
报告 `16/64/256` 的 VLM free-running 生成延迟和 action-server loop latency；若 256 的表示收益明确，
可尝试并行/分组预测、非自回归预测或每 N 个控制周期更新一次 physics token。

### 8.7 Checkpoint 兼容

新增 vocab 后 embedding 和 lm_head shape 改变，原 checkpoint 只能部分加载。保存 tokenizer 配置、special-token mapping、codebook checkpoint 和 model checkpoint 的联合 manifest；加载时对 hash 和 shape fail fast。

### 8.8 数据缓存污染

原仓库数据目录包含共享 cache。读取可以共享，但任何会重写 dataset metadata/statistics 的脚本必须先复制对应小型 metadata 到实验目录，不能原地修改共享数据。

## 9. 代码落地方案

建议新增而不直接改写已有 QwenPI：

```text
starVLA/
├── model/
│   ├── framework/QwenPIPhysicalTokens.py
│   └── modules/
│       ├── tokenizers/action_tokenizer.py
│       ├── tokenizers/physical_tokenizer.py
│       └── conditioning/discrete_token_conditioner.py
├── dataloader/
│   └── token_cache.py
└── training/
    ├── train_action_tokenizer.py
    ├── train_physical_tokenizer.py
    └── export_discrete_tokens.py
tests/
├── test_quantizers.py
├── test_token_cache.py
├── test_condition_fallback.py
└── test_checkpoint_roundtrip.py
```

实施顺序：

1. 数据审计与固定 split；
2. action tokenizer + 单元测试；
3. oracle action condition；
4. physical tokenizer + 表示诊断；
5. oracle joint condition；
6. vocab 扩展与 reasoner；
7. predicted-token condition；
8. 联合训练与正式 rollout。

## 10. 仓库与共享资源策略

已创建独立副本：`/mnt/jfzn/liujn/code/VLAct-PhysToken`。

当前共享关系：

```text
VLAct-PhysToken/playground/Datasets
  -> VLAct-main/playground/Datasets
VLAct-PhysToken/playground/Pretrained_models
  -> VLAct-main/playground/Pretrained_models
VLAct-PhysToken/shared_checkpoints
  -> VLAct-main/results/Checkpoints
```

新仓库的 `results/`、`logs/` 和 `wandb/` 是独立目录。新配置引用上游 checkpoint 时使用 `shared_checkpoints/...`，但所有新 checkpoint 写入本仓库的 `results/`。不要把 `results/Checkpoints` 本身软链接到上游，否则旧实验和新实验容易同名覆盖。

建议后续把共享资源迁移到中立目录（例如 `/mnt/jfzn/liujn/shared/vla/{datasets,models,checkpoints}`），再让两个仓库都指向它。目前的链接足够做原型，但语义上仍由 `VLAct-main` 持有资源。

## 11. 每阶段检查清单

每完成一个阶段更新本文档的实验记录，并至少检查：

- 数据 ID、split 和 cache hash 是否固定；
- tensor shape、action mask、归一化/反归一化是否通过测试；
- checkpoint 能否保存、加载和恢复训练；
- 单卡 20~100 step smoke test 是否无 NaN/OOM；
- code usage 和 token 样例是否合理；
- baseline 与新方法是否使用相同数据、step、seed、head 初始化和评测协议；
- 运行是否写入独立 output dir；
- 是否记录 git diff/代码快照、配置、环境和模型 hash。

## 12. 需要确认的决策

以下三项已经确认：

1. **首个 benchmark/任务**：RoboTwin `click_alarmclock`，先 clean、后 randomized。
2. **Physical Language 定义**：“未来视觉 feature dynamics token”，不使用 simulator privileged state。
3. **首轮主干**：先用轻量 QwenPI 做 oracle-token MVP，通过后接 QwenPI_v4。

已确认不能只测试 2/4/8 个 physics tokens。当前同时保留 pooled-feature 全范围容量扫描和
PhiZero-aligned 256-token 路线；后者借鉴 33 帧、16 个 encoded transitions 和每段 16 queries 的
协议，但仍使用与 VLAct continued-pretraining 一致的 Qwen3-VL feature 空间，不直接假设 Wan
feature 或 PhiZero checkpoint 能无缝迁移到机器人域。

## 13. 当前进度

- [x] 审阅 VLAct 模型、训练入口、数据字段和 QwenPI 条件路径；
- [x] 审阅本机 PhiZero 的 Physical Language 定义与训练顺序；
- [x] 创建隔离代码副本和共享资源链接；
- [x] 完成研究假设、架构、多阶段训练、MVP、消融和风险设计；
- [x] 确定首个 benchmark/task、Physical Language 定义和首轮 action head；
- [x] 审计 clean/randomized `click_alarmclock` 的 LeRobot v2.1 数据契约；
- [x] 实现 Stage 1A EMA-VQ/FSQ action tokenizer、分布式 EMA、mask 和 checkpoint 回归测试；
- [x] 实现不解码视频的真实 action cache 导出，并按 episode 固定验证 split；
- [x] 完成 clean 数据的 500-step EMA-VQ/FSQ GPU smoke；
- [x] 运行 5000-step action tokenizer 对照并导出带 SHA256 manifest 的 token cache；
- [x] 实现 oracle action-token conditioner、显式 baseline bypass 和数据集 cache join；
- [x] 增加轻量 8-layer QwenPI oracle-token MVP 配置；
- [x] 完成真实 VLM + oracle token 的单 batch forward/backward smoke；
- [x] 完成严格 bypass 的 E0 单 batch forward/backward smoke；
- [x] 完成 E0/E1 八卡 20-step 正式 dataloader/DeepSpeed/checkpoint smoke；
- [x] checkpoint 改为只保存 42.3M 可训练参数（约 81 MiB），不重复保存冻结 4B VLM；
- [x] 固定 5-episode held-out split，并实现可复现的离线连续动作 evaluator；
- [x] 实现 pooled feature-level Physical Language tokenizer，完成 EMA-VQ 2--256 token 容量扫描及 FSQ 对照；
- [x] 实现 PhiZero-aligned 时空 feature exporter、固定 256-token tokenizer、训练与 token 导出入口；
- [ ] 完成 PhiZero-aligned 真实 episode cache smoke 与小规模 tokenizer 训练。

## 14. Continued-pretraining 权重与执行方案

最终 continued-pretraining checkpoint 已就绪：

`/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt`

文件大小 11,736,465,942 bytes，`torch.load` 可完整读取。配置对应
`QwenHybrid_xrobot_padding` 三动作头联合预训练；checkpoint 含 714 个
`qwen_vl_interface.model.*` tensors。本地 Qwen3-VL backbone 的 713 个 tensors 全部存在，
shape mismatch 为 0，额外项仅为展开保存的 `lm_head.weight`。

| 工作 | 是否依赖最终权重 | 当前处理 |
|---|---|---|
| 数据审计、episode split、action cache | 否 | 立即完成 |
| EMA-VQ/FSQ action tokenizer | 否 | 立即训练和比较 |
| future visual-feature cache/physics tokenizer | 否，但依赖冻结视觉编码器选择 | 先用现有 frozen encoder 做表示验证 |
| oracle-token conditioner 单元测试 | 否 | 用随机/小模型张量打通 |
| QwenPI 加载、shape、显存 smoke | 否 | 可用现成官方下游 checkpoint，仅作工程检查 |
| 严格 E0 baseline | 是 | checkpoint 已就绪，先跑加载与单 batch smoke |
| VLM action/physics token reasoner | 是 | 可进入实现与小规模训练 |
| predicted-token 与联合训练结论 | 是 | 等 oracle E0/E1 通过后运行 |

本机已有
`/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`
（约 11 GB）和 QwenOFT checkpoint。它们是已经做过 RoboTwin 下游训练的权重，只允许用于
加载/接口/显存 smoke，不能替代正在复现的 pretraining checkpoint，也不能用其结果宣称
本方法相对严格 baseline 的收益。

当前目录已经包含训练 step、config、原始 config 和 dataset statistics。预训练统计来自四组
混合数据，不用于 RoboTwin 下游归一化；RoboTwin 使用目标数据自身统计。E0/E1 只加载
`qwen_vl_interface`，不加载 `oft_action_model`、`gr00t_action_model`、`pi_action_model`；
轻量 DiT 和 discrete conditioner 使用相同 seed 重新初始化。已经冻结且数据契约一致的
tokenizer 不需要重训。

## 15. 首轮 Action Tokenizer 记录

数据 cache：
`artifacts/action_caches/click_alarmclock_clean_wrap32.npz`，共 4252 个 `[32,14]`
chunks。模型输入 action layout 是 `[左6关节, 右6关节, 左夹爪, 右夹爪]`，因此关节维
是 `0..11`、夹爪维是 `12,13`；这与原始 Parquet 的交错 layout 不同。

固定 seed 42，按 episode 留出 5/50 条验证轨迹，500-step smoke 结果：

| Quantizer | Joint wrapped MAE | Gripper acc. | Active codes | Global perplexity |
|---|---:|---:|---:|---:|
| EMA-VQ, K=512 | 4.88 deg | 99.78% | 185/512 | 78.10 |
| FSQ, 8x8x8 | 4.59 deg | 99.52% | 43/512 | 10.93 |

这只证明数据、训练、保存加载和评估链路正常。EMA-VQ 的码本覆盖明显更好，FSQ 的短训
joint MAE 略低；两者都继续进入收敛级对照，不能根据 500 steps 选择最终量化器。

修正夹爪 decoder 语义后（decoder 输出为 logit，动作重建统一经过 sigmoid），相同 seed、
split 和 5000 steps 的最终结果：

| Quantizer | Joint wrapped MAE | Velocity MAE | Gripper acc. | Active codes | Global perplexity |
|---|---:|---:|---:|---:|---:|
| EMA-VQ, K=512 | **1.64 deg** | **0.00561 rad** | **100.00%** | **128/512** | **55.19** |
| FSQ, 8x8x8 | 2.39 deg | 0.00966 rad | 99.89% | 80/512 | 31.79 |

首轮主方案选择 EMA-VQ，FSQ 保留为 E2 对照。正式 EMA token cache：
`artifacts/token_caches/click_alarmclock_clean_ema_step5000.npz`，包含 4252 个样本、每样本
8 tokens，并已验证 `(episode_id, base_index)` 覆盖率 100%。

oracle-token MVP 配置位于
`examples/Robotwin/train_files/qwenpi_physical_tokens_click_alarmclock.yaml`。其中 action head
是 8 层、512 hidden 的轻量 DiT；`framework.discrete_conditioning.enabled=false` 时完全不
追加 token，用作同结构 E0。配置使用 `trainer.reload_modules: qwen_vl_interface`，防止加载
预训练 checkpoint 中的旧动作头。

## 16. 真实权重与八卡工程验证记录

真实单样本 smoke 均使用 continued-pretraining 的 `qwen_vl_interface` 和真实 RoboTwin
样本：E1 成功追加 8 个 action tokens，conditioner 收到有限梯度；E0 实际追加 0 个 token，
conditioner 无梯度。二者都完成 forward/backward，证明显式 bypass 严格生效。

八卡 DeepSpeed ZeRO-2、全局 batch 8、seed 42、相同 action-head 初始化的 20-step 结果：

| ID | 离散条件 | Step 1 loss | Step 20 loss | 稳定 step 时间 | 结论 |
|---|---|---:|---:|---:|---|
| E0 | 关闭 | 459.62 | 21.38 | 约 0.17 s | 正式基线链路通过 |
| E1 | 8 个 oracle EMA-VQ action tokens | 517.18 | 26.08 | 约 0.17--0.23 s | 条件链路与反向通过 |

这 20 steps 只用于验证工程链路，不能据此判断 E1 是否优于 E0。随机 flow-matching timestep、
短程过拟合和 cosine scheduler 都会让单点 loss 不具比较意义；方法判断需要 1k--3k steps、
固定 validation panel 和最终 RoboTwin rollout success。

本轮发现并修复了三项会污染实验的问题：

1. `LayerwiseFlowmatchingActionHead` 的 interleaved self-attention 层错误接收 2560-D VLM
   prefix，现改为仅偶数 cross-attention 层接收 prefix，奇数层走 512-D self-attention；
2. `LeRobotMixtureDataset` 直接调用 leaf dataset 的内部取样接口，绕过了 token-cache
   wrapper 的 `__getitem__`，现通过显式 `attach_discrete_tokens` protocol 按真实
   `(trajectory_id, step_index)` join；
3. 原训练器每次保存完整冻结 VLM，现支持 `trainer.save_trainable_only=true`。E0/E1
   checkpoint 均仅含 `action_model` 和 `discrete_conditioner`，142 tensors、42,325,007
   parameters、84,699,300 bytes。

启动脚本位于 `scripts/physical_tokens/run_qwenpi_oracle_ablation.sh`。八个 rank 同时装载
11 GB `.pt` 主干仍需数分钟；checkpoint 已使用 `mmap=True` 降低额外复制，但各 rank 的
模型参数 materialization 仍触发共享存储读页。该问题影响启动效率，不影响训练数值，后续
可评估 safetensors 或 rank-0 load + distributed broadcast。

## 17. 固定 Held-out Policy Evaluator

训练/验证现按 episode 严格拆分。seed 42 的验证 episode 是 `[5, 20, 25, 39, 49]`，共
439 个 chunks；训练仅保留其余 45 个 episode、3813 个 chunks。split manifest 位于
`artifacts/splits/click_alarmclock_seed42.json`。配置中的 `exclude_episode_ids` 保证长训不读取
验证 episode，evaluator 则把它反转为 `include_episode_ids`，因此评估不再消费训练 iterator。

离线 evaluator 位于 `scripts/physical_tokens/evaluate_qwenpi_oracle.py`，具备以下约束：

1. 从五条 held-out episode 均衡、无放回地选择固定 panel；
2. 每个样本使用 `seed + panel ordinal` 固定 flow-matching 初始噪声，E0/E1 可逐样本配对；
3. 先加载 continued-pretraining checkpoint 的 `qwen_vl_interface`，再严格检查并加载 81 MiB
   的 `action_model + discrete_conditioner` checkpoint；
4. JSON 保存所有 `(episode_id, base_index)`、checkpoint/config/statistics SHA256、推理步数和
   latency；
5. 报告 action MAE/RMSE、关节 wrap-aware MAE、关节速度 MAE、末端关节误差及夹爪准确率。

真实数据检查确认本配置的 `sample["action"]` 范围约为 `[-1.88, 2.80]`，12 个关节维已经是
弧度，最后两个维度是 0/1 夹爪值。`predict_action` 的返回 key 虽名为
`normalized_actions`，其数值实际与训练 target 一样处于 raw action space；evaluator 因此不做
`q01/q99` 二次反归一化。这一点必须保持，否则物理误差会被错误缩放。

20-step 旧 checkpoint 已各完成 1-sample 真实 smoke，结果保存在：

- `artifacts/evaluations/qwenpi_e0_steps20_smoke_1sample.json`
- `artifacts/evaluations/qwenpi_e1_steps20_smoke_1sample.json`

二者使用同一 held-out 样本 `(episode=49, base_index=32)` 和同一扩散噪声，完整模型加载及
E0 bypass/E1 oracle-token 推理均通过。误差仍很大，只反映 20-step 动作头尚未收敛，不能用于
判断方法优劣。当前相关回归测试 27/27 通过；环境没有安装 `pytest`，测试通过直接调用测试
函数执行。

便捷评估命令：

```bash
bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
  e0 results/Checkpoints/<E0_RUN>/checkpoints/steps_3000_pytorch_model.pt 32 0
bash scripts/physical_tokens/run_qwenpi_oracle_eval.sh \
  e1 results/Checkpoints/<E1_RUN>/checkpoints/steps_3000_pytorch_model.pt 32 0
```

长训 launcher 已修正两点：warmup 默认取总步数 10% 且上限 200，不再永久写死为 2；训练器
原有的 `eval_action_model` 会取下一批训练数据，既非 held-out 又会改变训练数据流，因此 launcher
默认把 `eval_interval` 设为 `max_train_steps + 1`，统一在 checkpoint 后执行上述离线 evaluator。

截至 2026-09-26 14:15，node4 八卡被其他用户的两组 PONI 训练及显存保留进程占用，每卡仅余
约 30--38 GiB，且多卡利用率为 100%。本轮只利用单卡余量完成短推理 smoke，没有终止或挤占
这些进程。待八卡释放后的执行顺序是：E0/E1 各 100-step split smoke -> 固定 32-sample panel
评估 -> E0/E1 各 3000 steps -> 相同 panel 评估 -> RoboTwin rollout。

## 18. Node4 长训与中间结果（2026-09-26）

实验在 `sh-xty-gpua100-node004` 的 8xA100 80GB 上运行，DeepSpeed ZeRO-2，每卡 batch 1，
global batch 8。E0/E1 共用 seed 42、45-episode 训练 split、5-episode held-out split、初始化、
scheduler 和训练步数，唯一主变量是是否向 DiT prefix 追加 8 个 oracle EMA-VQ action tokens。

100-step 训练均完成且无 NaN/OOM。固定 32-sample held-out panel 结果如下：

| Metric | E0: no token | E1: oracle action token |
|---|---:|---:|
| Action MAE | 0.92499 | **0.85682** |
| Action RMSE | 1.16779 | **1.07475** |
| Joint wrapped MAE | 53.03 deg | **48.94 deg** |
| Endpoint joint MAE | 51.88 deg | **48.07 deg** |
| Joint velocity MAE | 1.09456 | **1.08819** |
| Gripper accuracy | 54.44% | **58.35%** |

这是全指标同向的早期正信号，但 100 steps 的两个头都远未收敛，不作最终结论。
结果文件为 `artifacts/evaluations/qwenpi_e0_steps_100_pytorch_model_32samples.json` 和
`artifacts/evaluations/qwenpi_e1_steps_100_pytorch_model_32samples.json`。

100-step E0 首次启动时在 step 1 遇到 `Bad CRC-32 for file 'action_token_ids.npy'`。根因是
`DiscreteTokenCache` 长期保留压缩 `NpzFile`/ZIP handle，DataLoader fork 后多 worker 并发共享同一
handle。现在初始化时将 cache 数组完整拷入内存并立即关闭 archive。修复后 3 个 cache
回归测试通过，8 workers 对真实 cache 完成 17,008 次 lookup，无 CRC 错误。

E0 3000-step 已完成：step 1 loss 552.1695，step 3000 loss 0.07066，训练本体耗时
8 分 31 秒，稳定 step 约 0.15--0.18 s，无 CRC、NaN、OOM 或 traceback。checkpoint 为
`results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e0_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt`，
固定 32-sample held-out 评估为：

| Metric | E0, 3000 steps |
|---|---:|
| Action MAE / RMSE | 0.13331 / 0.21277 |
| Joint wrapped MAE | 7.13 deg |
| Endpoint joint MAE | 6.99 deg |
| Joint velocity MAE | 0.06453 rad |
| Gripper accuracy | 92.24% |

结果保存在 `artifacts/evaluations/qwenpi_e0_steps_3000_pytorch_model_32samples.json`。

E1 3000-step 也已完成：step 1 loss 551.5875，step 3000 loss 0.06019，训练本体耗时
8 分 47 秒，无 CRC、NaN、OOM 或 traceback。checkpoint 为
`results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e1_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt`。

32-sample panel 上的结果是混合的：E1 的 velocity MAE 改善 12.98%、gripper accuracy 提升
0.83 个百分点，但 action MAE/RMSE 轻微变差 0.46%/0.52%，joint/endpoint MAE 变差
2.30%/3.72%。这也否定了“只是多加 token，所以至少不会降点”的假设：附加条件会
改变注意力、有效序列长度和优化动力学，即使 oracle token 也可能降点。

为减少 32 样本 panel 方差，进一步执行了最大严格 episode-balanced panel。五条验证
episode 分别有 `[101, 87, 93, 82, 76]` 个 chunks，因此每条无放回抽取 76 个，共 380 个，
覆盖全部 held-out chunks 的 86.6%。E0/E1 的 sample IDs、config SHA256、statistics SHA256
已逐项确认一致：

| Metric | E0: no token | E1: oracle action token | E1 relative change |
|---|---:|---:|---:|
| Action MAE | 0.13047 | **0.12378** | **-5.13%** |
| Action RMSE | 0.20801 | **0.19527** | **-6.13%** |
| Joint wrapped MAE | 6.90 deg | **6.60 deg** | **-4.29%** |
| Endpoint joint MAE | 7.19 deg | **6.99 deg** | **-2.73%** |
| Joint velocity MAE | 0.06012 rad | **0.05183 rad** | **-13.80%** |
| Gripper accuracy | 92.57% | **93.08%** | **+0.51 pp** |
| Gripper MAE | 0.19104 | **0.17516** | **-8.31%** |

大 panel 上所有主指标同向改善，说明 oracle EMA-VQ action token 对连续动作头有小而稳定的
离线收益，其中速度平滑性和夹爪最明显。但这仍是 oracle-token upper-bound，不等于 VLM 预测 token
后仍能保持收益，也不等于 RoboTwin rollout success 提升。oracle token 在闭环中不可得，
因此下一个 go/no-go 点必须是 VLM action-token predictor，随后才能进行诚实的 clean rollout。
大 panel 结果保存在：

- `artifacts/evaluations/qwenpi_e0_steps_3000_pytorch_model_380samples.json`
- `artifacts/evaluations/qwenpi_e1_steps_3000_pytorch_model_380samples.json`

### 18.1 动作头支持边界

当前 MVP 只训练一个 `QwenPIPhysicalTokens` 动作头，即 8-layer、512-hidden 的轻量
QwenPI/DiT。continued-pretraining checkpoint 虽然来自可联合训练 OFT/GR00T/PI 三头的
`QwenHybrid_xrobot_padding`，本阶段只加载其共享 `qwen_vl_interface`，不加载三个旧动作头。

三头联合训练在架构上可扩展，但当前代码尚未把离散 token 条件分别接入 OFT query、
GR00T last-hidden 和 PI multilayer embedding 三条路径。为避免将 head 差异和 token 收益混在一起，
推进顺序定为：先用单 QwenPI 头完成 E0/E1 -> 训练 predicted-token reasoner 后做 rollout ->
分别移植到 OFT、GR00T、PI 做
单头配对消融 -> 最后才进行三头联合训练及 loss-weight 消融。

## 19. VLM Action-token Reasoner

为了保持与 PI 0.5 类似的“VLM 先生成离散动作 token”路径，本阶段没有使用并行
classifier probe，而是复用仓库 `QwenFast` 的 causal-LM special-token 机制。
`Qwen3-VL-4B-Instruct-Action` 已包含 2048 个 `<robot_action_i>` tokens，EMA-VQ 的 512 codes
映射到前 512 个 token，不改动 tokenizer vocabulary。每个样本自回归预测固定 8 tokens，推理时
logits grammar 限制在这 512 个 action tokens 内。

实现位于：

- `starVLA/model/framework/QwenActionTokenReasoner.py`；
- `examples/Robotwin/train_files/qwen_action_token_reasoner_click_alarmclock.yaml`；
- `scripts/physical_tokens/run_action_token_reasoner.sh`；
- `scripts/physical_tokens/evaluate_action_token_reasoner.py`。

首轮冻结 vision tower 和 LLM 前 34/36 层，只训绑定的 input-embedding/LM-head、final norm 和最后
2 个 language blocks，共 596.1M trainable parameters。真实单样本 forward/backward 完成，24 个有梯度
tensor 全部 finite；未训练时 8-token free-running accuracy 为 0。

500-step、八卡、global batch 8 训练本体耗时 3 分 08 秒，CE 从约 10.8 降到 1.77，但固定
32-sample held-out free-running 结果仍不足以接入 DiT：

| Metric | 500-step reasoner |
|---|---:|
| Token accuracy | 2.34% |
| Exact 8-token sequence accuracy | 0.00% |
| Mean correct tokens / sequence | 0.1875 / 8 |
| Predicted / target active codes | 25 / 76 |
| Predicted-token decoder action MAE | 0.25381 |
| Oracle-token decoder action MAE | 0.02232 |
| Predicted-token decoder joint MAE | 15.19 deg |
| Oracle-token decoder joint MAE | 1.49 deg |

虽然 token accuracy 约为 512-way 随机水平的 12 倍，但 predicted-token decoder MAE 已明显高于无 token
E0 policy 的 0.13331，现在接入 DiT 会消耗 oracle 收益。因此先延长 reasoner 至 3000 steps；
若 6 个 epoch 后 free-running 仍无法保留大部分 oracle decoder 质量，应降低 token bitrate、增加可预测性
正则或改用分层 code，而不是盲目进入 DiT 联训。500-step 结果位于
`artifacts/evaluations/action_reasoner_steps_500_pytorch_model_32samples.json`。

3000-step reasoner 也已完成。训练本体耗时 17 分 41 秒，step 3000 CE 为 1.255，末段最低
约 0.72；全程无 NaN、OOM 或 checkpoint CRC 错误。checkpoint 位于
`results/Checkpoints/qwen_action_token_reasoner_click_alarmclock_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt`。
在与 500-step 完全相同的固定 32-sample held-out panel 上：

| Metric | 500 steps | 3000 steps |
|---|---:|---:|
| Free-running token accuracy | 2.34% | **3.52%** |
| Exact 8-token sequence accuracy | 0.00% | **0.00%** |
| Mean correct tokens / sequence | 0.1875 / 8 | **0.28125 / 8** |
| Teacher-forced token accuracy | 未记录 | **27.73%** |
| Teacher-forced mean correct tokens / sequence | 未记录 | **2.21875 / 8** |
| Predicted / target active codes | 25 / 76 | **30 / 76** |
| Predicted-token decoder action MAE | 0.25381 | **0.27524** |
| Oracle-token decoder action MAE | 0.02232 | **0.02232** |
| Predicted-token decoder joint MAE | 15.19 deg | **16.06 deg** |
| Oracle-token decoder joint MAE | 1.49 deg | **1.49 deg** |

结果位于
`artifacts/evaluations/action_reasoner_steps_3000_pytorch_model_32samples.json`。teacher-forced
明显高于 free-running，说明 8-token AR 链存在严重 exposure error；但第一个位置在两种模式下
准确率都只有 6.25%，所以问题不只是误差传播，也包括单帧观测对未来动作 code 的可预测性不足。
继续增加 AR 训练步数不能直接解决这两个问题，而且 3000-step 的 decoded action MAE 反而比
500-step 更差。

因此当前判定为 **no-go：暂不把 predicted action tokens 接入 DiT，也不做该路径的在线
rollout**。oracle E1 的离线收益仍是有效 upper bound，但当前 reasoner 无法实现它。下一轮先做
并行 classifier probe，将 8 个 code 位置同时预测，以隔离 AR exposure error；随后按成本从低到高
比较 `token_count=2/4/8`、`codebook_size=64/128/512`，并评估 coarse-to-fine code。只有
predicted-token decoder action MAE 至少接近 E0 policy 的约 0.13，才恢复 DiT 条件接入。

## 20. 并行 Action-token 可预测性 Probe

为拆分“token 本身不可预测”和“AR exposure error”，新增
`QwenActionTokenProbe`：取多模态 prompt 最后一个有效 hidden state，经两层 MLP 一次输出
`[8,512]` logits，各位置独立做 cross-entropy。它与 AR reasoner 使用相同图像、语言、数据 split
和 EMA token target，但推理不消费前一位置的预测。视觉塔和前 34/36 个 language blocks 冻结，
只训练最后 2 个 blocks、final norm 和 probe，共 208.69M 参数；AR reasoner 为 596.06M。

实现与入口：

- `starVLA/model/framework/QwenActionTokenProbe.py`；
- `examples/Robotwin/train_files/qwen_action_token_probe_click_alarmclock.yaml`；
- `scripts/physical_tokens/run_action_token_probe.sh`；
- `scripts/physical_tokens/run_action_token_probe_eval.sh`。

真实单样本 BF16 forward/backward smoke 已通过，28 个有梯度 tensor 全部 finite。首次 smoke
暴露了 probe 输入被强制转 FP32、而 LayerNorm 已随模型转为 BF16 的 dtype mismatch；移除不必要的
强制转换后通过。500-step 八卡训练本体约 1 分 45 秒，稳定 step 约 0.14--0.18 秒，CE 从 6.28
降至最后一个 batch 的 2.78，无 NaN/OOM。

固定 32-sample held-out panel：

| Metric | AR, 500 steps | Parallel probe, 500 steps |
|---|---:|---:|
| Token accuracy | 2.34% | **2.73%** |
| Exact sequence accuracy | 0.00% | **0.00%** |
| Mean correct tokens / sequence | 0.1875 / 8 | **0.21875 / 8** |
| Predicted active codes | 25 | **38** |
| Predicted-token decoder action MAE | 0.25381 | **0.13555** |
| Predicted-token decoder joint MAE | 15.19 deg | **7.88 deg** |
| Predicted-token decoder velocity MAE | 未记录 | **0.02911 rad** |
| Predicted-token decoder gripper accuracy | 未记录 | **87.60%** |

probe 的 exact token 指标仍低，但 decoded action MAE 已接近相同 32-sample panel 上 E0 policy 的
0.13331，并远好于 AR。原因是 EMA-VQ code 的“完全相等”不等价于动作空间距离：预测到邻近或
功能相似 code 时，0/1 token accuracy 仍记错，但 decoder 可以输出接近的动作。因此后续选择模型
必须以 decoded action metric 和最终 policy/rollout 为主，token accuracy 只作诊断。当前结果是
恢复可行性的正信号，但仅 32 samples 且 probe 只训练约 1 epoch；继续进行 3000-step 与 380-sample
评估后再决定是否接入 DiT。结果文件为
`artifacts/evaluations/action_probe_steps_500_pytorch_model_32samples.json`。

3000-step 八卡训练也已完成，训练本体 8 分 17 秒，约 6.29 epochs，step 3000 CE 为
0.0747。虽然训练误差显著下降，但 held-out 动作质量没有同步改善：

| Panel / Metric | Probe 500 steps | Probe 3000 steps | E0 policy 3000 steps |
|---|---:|---:|---:|
| 32 / token accuracy | 2.73% | **4.30%** | N/A |
| 32 / decoded action MAE | **0.13555** | 0.13953 | **0.13331** |
| 380 / token accuracy | 4.51% | **5.56%** | N/A |
| 380 / decoded action MAE | **0.14078** | 0.15005 | **0.13047** |
| 380 / joint MAE | **8.25 deg** | 8.77 deg | **6.90 deg** |
| 380 / velocity MAE | 0.03050 rad | **0.02689 rad** | 0.06012 rad |
| 380 / gripper accuracy | **87.88%** | 86.72% | **92.57%** |

大 panel 上 500-step probe 比 E0 action MAE 差 7.90%，3000-step 差 15.01%。随着训练继续，
exact-code accuracy 提升但 decoded action MAE 变差，说明 45 条训练 episode 下存在过拟合，而且
512-way code 的 CE 距离不反映 code 在动作空间中的邻近程度。并行 probe 已证明 AR exposure
error 是原 reasoner 失败的重要来源，但 `8 tokens x 512 codes` 仍不满足接入 DiT 的门槛。

维持 **no-go**：不把当前 probe/AR token 接入 DiT。下一步优先重新训练低码率 EMA tokenizer：
`4x128` 与 `2x64`。先检查各自 oracle 重建 MAE；只有 oracle 上限合理的方案才生成 cache 并训练
parallel predictor。对应评估文件为：

- `artifacts/evaluations/action_probe_steps_500_pytorch_model_380samples.json`；
- `artifacts/evaluations/action_probe_steps_3000_pytorch_model_32samples.json`；
- `artifacts/evaluations/action_probe_steps_3000_pytorch_model_380samples.json`。

### 20.1 低码率表示筛选

为降低未来动作在单帧观测下的歧义，固定 EMA-VQ 和相同 action cache/split，训练了两个 5000-step
低码率 tokenizer：

| Tokenizer | Joint oracle MAE | Velocity MAE | Gripper acc. | Active codes | Perplexity |
|---|---:|---:|---:|---:|---:|
| 原 `8x512` | **1.64 deg** | **0.00561** | **100.00%** | 128/512 | 55.19 |
| `4x128` | 2.20 deg | 0.00647 | 99.63% | 65/128 | 31.20 |
| `2x64` | 3.09 deg | 0.00808 | 99.11% | 42/64 | 33.16 |

两套低码率 oracle 重建都仍明显优于 E0 policy，具备作为条件摘要的容量。随后以相同 global batch 8、
seed 42、45/5 episode split，各训练 500-step parallel probe，并直接评估 380-sample panel：

| Representation | Token acc. | Exact seq. | Decoded action MAE | Joint MAE | Velocity MAE | Gripper acc. |
|---|---:|---:|---:|---:|---:|---:|
| `8x512`, 500 steps | 4.51% | 0.00% | 0.14078 | 8.25 deg | 0.03050 | 87.88% |
| `4x128`, 500 steps | 13.36% | 1.58% | **0.11437** | **6.77 deg** | 0.01367 | 90.80% |
| `2x64`, 500 steps | **22.50%** | **4.47%** | 0.12261 | 7.31 deg | **0.01106** | **90.85%** |
| E0 continuous policy | N/A | N/A | 0.13047 | 6.90 deg | 0.06012 | 92.57% |

`4x128` predicted-token decoder action MAE 比 E0 低 12.34%，joint MAE 也低 1.89%；`2x64`
action MAE 低 6.02%，但 joint MAE 较差。主线因此选择 `4x128`，`2x64` 保留为码率消融。
这是第一个达到 predicted-token go 条件的结果，可以进入 E2 DiT 条件实验。注意 decoder 误差与
policy 误差不是同一模型的直接胜负，它只证明预测 token 已有足够动作信息；最终价值仍必须由
E0 与 E2 配对 policy 评估和在线 rollout 决定。

下一步先用冻结 probe 对完整 4252 个 observation 生成 keyed predicted-token cache，再用该 cache
训练 E2。离线 cache 关闭随机 augmentation，因此属于低成本验证；正式 rollout 必须实时运行 probe，
不能用未来动作生成的 oracle cache，也不能把离线 cache 当作部署结果。

### 20.2 E2：Predicted Action Token 条件实验

已用 500-step `4x128` parallel probe 对 clean `click_alarmclock` 的全部 4252 个 observation
导出 keyed cache：
`artifacts/token_caches/click_alarmclock_predicted_ema_4x128_probe_step500.npz`。cache 以
`(episode_id, base_index)` 对齐，每个样本固定 4 个 action token；导出时关闭随机图像增强，manifest
记录 probe、tokenizer、source cache 的路径和 SHA256。该 cache 只依赖当前 observation，不读取未来
action，因此可用于低成本 predicted-token 条件验证；但它仍是离线预计算，不能替代实时 reasoner
rollout。

E2 使用该 predicted cache 条件化同一个 QwenPI/DiT 动作头，其他设置与 E0/E1 保持一致，训练
3000 steps。固定 5 个 held-out episodes、380-sample episode-balanced panel 的结果如下：

| Metric | E0 no token | E1 oracle `8x512` | E2 predicted `4x128` |
|---|---:|---:|---:|
| Action MAE | 0.13047 | **0.12378** | 0.13032 |
| RMSE | 0.20801 | **0.19527** | 0.21317 |
| Joint MAE | 6.90 deg | **6.60 deg** | 6.79 deg |
| Endpoint MAE | 7.19 deg | 6.99 deg | **6.87 deg** |
| Velocity MAE | 0.06012 rad | **0.05183 rad** | 0.05657 rad |
| Gripper accuracy | 92.57% | **93.08%** | 90.45% |

E2 相对 E0 的 action MAE 只改善约 0.12%，joint MAE 改善约 1.55%，velocity MAE 改善约
5.90%；同时 RMSE 和 gripper accuracy 变差。因此结论是 **弱正向但不稳定**：predicted action
token 路径已做到不明显破坏主策略，并在部分连续控制指标上有收益，但尚不能宣称总体提升。
checkpoint 位于
`results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt`，
评估结果位于
`artifacts/evaluations/qwenpi_e2_steps_3000_pytorch_model_380samples.json`。

### 20.3 双 Token 共存约束与下一阶段

后续加入 Physical Language token 时，**必须保留动作离散 token**。两者表达不同信息：action
token 是未来动作轨迹的离散摘要，physics token 是未来可见状态变化/交互动力学的离散摘要；physics
token 不是 action token 的替代品。主模型的条件序列固定为：

```text
C_policy = [C_vlm ; E_action(z_action[1:4]) ; E_physics(z_physics[1:N])]
```

其中 action 路径主配置固定为已筛选出的 `4 tokens x 128 codes`。`N in {2,4,8}` 只作为已完成的
极低码率初筛；正式容量扫描改为 `N in {4,8,16,32,64,128,256}`。policy 端先比较 16/64/256，
再在收益拐点附近补 32/128。所有比较使用相同 tokenizer 训练预算、数据 split 和 policy 训练预算；
physics codebook 首轮比较 128/256，EMA-VQ
为主方案，FSQ 只作匹配码率对照。Conditioner 继续使用独立的 action/physics embedding、position、
type embedding 和 gate，最终 concat 后送入动作头，因此任一路可独立 dropout、shuffle 或关闭。

下一轮最小实验矩阵为：B0=`E0` 无 token；B1=predicted action only（即 E2）；B2=oracle
physics only；B3=predicted action + oracle physics；B4=predicted action + shuffled physics；在 physics
predictor 达到有效 feature reconstruction 后再做 B5=predicted action + predicted physics。B3 必须与
B1 配对，才能判断 physics token 是否在 action token 已存在时仍提供独立增量。

## 21. Physical Language MVP 实现与数据管线

### 21.1 Future visual-feature cache

RoboTwin clean `click_alarmclock` 是标准 LeRobot v2.1 数据集，共 50 episodes、4252 frames、
50 FPS，包含 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路 640x480 AV1 视频。底层
`LeRobotSingleDataset.get_video` 已支持 `base_index + delta_indices` 的 timestamp 对齐，并在 episode
边界 clamp，因此 MVP 没有另写视频时间同步逻辑。

新增 `scripts/physical_tokens/export_robotwin_future_features.py`。首轮只使用 `video.cam_high`，按
offset `[0, 8, 16, 31]` 采样约 0.62 秒范围内的四个时间锚点。每个 episode 只解码并编码全部帧
一次，再按 source index 重排，避免同一帧被多个 action chunk 重复编码。视觉编码器使用当前
Qwen3-VL-4B 的 frozen vision tower，并从 continued-pretraining checkpoint 精确加载 315 个
`qwen_vl_interface.model.model.visual.*` tensor；每帧 patch embedding 做 mean pooling。tokenizer target
不是绝对外观 feature，而是 `feature(t+k) - feature(t)`，以减少背景、光照和物体静态外观捷径。

完整 cache 位于
`artifacts/future_feature_caches/click_alarmclock_cam_high_qwen3vl_full.npz`：

| Property | Value |
|---|---:|
| Samples / unique keys | 4252 / 4252 |
| Episodes | 50 |
| Feature shape | `[4252, 4, 2560]` |
| Storage dtype / size | float16 / 51.2 MiB |
| Valid at offsets `[0,8,16,31]` | `[4252,3852,3452,2702]` |
| Mean / max absolute delta | 0.03662 / 0.69971 |
| Non-finite values | 0 |

cache 同时保存 `future_valid_mask` 和 `source_frame_indices`；clamp 后的尾帧不会被误当成真实静止
监督。checkpoint SHA256 为
`5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d`。当前只用 scene camera
是低成本 MVP 选择，左右 wrist view 是后续多视角消融，而不是已完成结论。

### 21.2 Physical Language tokenizer

新增 `PhysicalLanguageTokenizer`：标准化后的四帧 feature delta 先经过 temporal Transformer，随后
用固定数量 learned queries cross-attend 时间特征，再进入 EMA-VQ；decoder 使用四个 learned time
queries cross-attend quantized latents，重建 feature delta。learned query 结构允许 physics token 数量
独立取 2、4 或 8，不受四个时间锚点数量限制。loss 包含 masked Smooth-L1、cosine direction、
temporal-delta 和 EMA commitment；offset 0 的零 delta 权重降为 0.1，episode 尾部 invalid future
完全 mask。FSQ 使用相同 encoder/decoder，只替换 quantizer。

实现与入口：

- `starVLA/model/modules/tokenizers/physical_language_tokenizer.py`；
- `scripts/physical_tokens/train_physical_language_tokenizer.py`；
- `scripts/physical_tokens/run_physical_language_tokenizer.sh`；
- `scripts/physical_tokens/export_physics_token_ids.py`；
- `scripts/physical_tokens/merge_discrete_token_caches.py`。

主配置 `4 tokens x 128 EMA codes` 在相同 45/5 episode split 上八卡训练 1000 steps，约 41 秒。
验证结果：

| Metric | Value |
|---|---:|
| Reconstruction RMSE | **0.04639** |
| No-motion RMSE | 0.05505 |
| Train-time-mean RMSE | 0.05391 |
| Improvement vs no-motion / mean | 15.7% / 13.9% |
| Validation-batch active ratio | 31.6% |
| Shuffle-future token change | 86.9% |

800 steps 后 reconstruction 已基本平台，因此保留 step-1000，不盲目延长。全 4252 样本编码后的
token ID 范围为 `[0,127]`，128 codes 全部使用，全局 perplexity 102.77，各 token position 的
active codes 为 `[109,128,124,124]`。shuffle-future 高敏感度以及优于 no-motion/mean baseline
共同说明 token 在编码未来视觉变化；这仍不等同于它对 action policy 必然有效，必须由 B2/B3/B4
配对实验判断。

oracle physics cache 位于
`artifacts/token_caches/click_alarmclock_oracle_physics_ema_4x128_step1000.npz`。它读取未来帧，明确是
privileged upper-bound supervision，不能用于真实 rollout。与 observation-predicted action token
严格按 `(episode_id, base_index)` 合并后得到：

- B3：`click_alarmclock_pred_action4_oracle_physics4.npz`；
- B4：`click_alarmclock_pred_action4_shuffled_physics4_seed42.npz`。

B4 在每个 episode 内对 physics token 做非零 cyclic shuffle，既破坏当前 observation/future 的对应，
又不改变 task、episode 和 token 边际分布。训练/评估入口现支持 `b2/b3/b4`；B3 的 20-step 双条件
smoke 已通过，action 与 physics 两路 conditioner 均进入 optimizer，41.67M 参数可训练，无 NaN、
OOM 或 token 越界。

### 21.3 B3：Predicted action + oracle physics

B3 固定保留 4 个 observation-predicted action tokens，并追加 4 个读取未来视觉变化的 oracle
physics tokens。3000-step 八卡训练本体约 8 分 56 秒，final loss 0.05474，无 NaN/OOM。固定
380-sample panel 结果为：

| Metric | E0 no token | E2 predicted action | B3 action + oracle physics |
|---|---:|---:|---:|
| Action MAE | 0.13047 | **0.13032** | 0.13412 |
| RMSE | **0.20801** | 0.21317 | 0.22096 |
| Joint MAE | 6.90 deg | **6.79 deg** | 7.26 deg |
| Endpoint MAE | 7.19 deg | **6.87 deg** | 7.55 deg |
| Velocity MAE | 0.06012 rad | 0.05657 rad | **0.05173 rad** |
| Gripper accuracy | **92.57%** | 90.45% | 92.01% |

B3 的 velocity 明显优于 E0/E2，且接近 E1 oracle-action 的 0.05183 rad，说明 physics 表征包含
部分动态信息；但 action MAE、RMSE、joint 和 endpoint 均退化，因此总体结论是 **no-go：当前
oracle physics 注入没有在 action token 已存在时带来独立、稳定的 policy 增益**。它更不能支持
“多 token 至少不会降点”的假设。训练后 action gate 为 0.10596，physics gate 为 0.10107（均从
0.1 初始化），表明模型对 physics 路径的利用很弱。checkpoint 位于
`results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_b3_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt`，
评估位于 `artifacts/evaluations/qwenpi_b3_steps_3000_pytorch_model_380samples.json`。

下一步先完成 B4 episode-internal shuffled-physics negative control。若 B3 不优于 B4，当前注入的
正确物理语义没有可辨识价值，应先改表示/注入方式，而不是立即训练 physics reasoner；若 B3 显著
优于 B4 但仍低于 E2，则说明语义有信号、但 condition strength 或融合结构需要修正。

B4 已以完全相同预算完成。结果表明正确对齐有一定价值，但仍不足以超过 action-only：

| Metric | E2 action only | B3 aligned physics | B4 shuffled physics |
|---|---:|---:|---:|
| Action MAE | **0.13032** | 0.13412 | 0.13847 |
| RMSE | **0.21317** | 0.22096 | 0.21811 |
| Joint MAE | **6.79 deg** | 7.26 deg | 7.32 deg |
| Endpoint MAE | **6.87 deg** | 7.55 deg | 7.56 deg |
| Velocity MAE | 0.05657 rad | **0.05173 rad** | 0.05713 rad |
| Gripper accuracy | 90.45% | **92.01%** | 91.54% |

B3 相对 B4 的 action MAE 改善约 3.14%，velocity MAE 改善约 9.46%，gripper accuracy 提高
0.47 pp；但 RMSE 略差。说明 aligned physics token 不是纯噪声，尤其携带动态信息；然而 B3 相对
E2 的总体退化说明 concat 后仅靠独立 scalar gate 的融合不足，或者当前 feature-delta token 混合了
与控制无关的场景运动。当前决策是：保留双 token 架构和 physics tokenizer；已完成的 2/4/8
只视为极低码率基线，追加 16/32/64/128/256 容量扫描以及更强的 token
dropout/FiLM/cross-attention 融合小实验，不进入 predicted physics
reasoner，也不将 B3 当成正向结果。B4 checkpoint 与评估分别位于
`results/Checkpoints/qwenpi_physical_tokens_click_alarmclock_b4_seed42_steps3000/checkpoints/steps_3000_pytorch_model.pt`
和 `artifacts/evaluations/qwenpi_b4_steps_3000_pytorch_model_380samples.json`。

### 21.4 Physics token 数量与量化器对照

在完全相同的 future-feature cache、45/5 episode split、网络宽度和 1000-step 预算下，physics
token 数量的 EMA-VQ 对照为：

| Quantizer | Physics tokens x codes | Validation RMSE | Batch perplexity | Batch active ratio | Shuffle token change |
|---|---:|---:|---:|---:|---:|
| EMA-VQ | 2 x 128 | 0.04749 | 28.52 | 26.0% | 87.3% |
| EMA-VQ | 4 x 128 | 0.04639 | 30.87 | 31.6% | 86.9% |
| EMA-VQ | 8 x 128 | **0.04603** | 33.62 | 35.6% | 88.7% |
| FSQ | 4 x 128 | **0.04602** | 38.62 | 38.9% | 87.8% |

EMA 由 2 增至 4 tokens 的 RMSE 改善约 2.32%，由 4 增至 8 只再改善约 0.79%，但 policy
条件长度翻倍。因此主配置仍保留 `4 physics tokens`，8 tokens 只作为 rate 上限，不因 tokenizer
重建指标的微小提升直接增加策略成本。FSQ 4x128 的 batch RMSE 比 EMA 4x128 低约 0.82%，差距
很小；EMA 在全量 4252 样本上 128 codes 全覆盖、global perplexity 102.77，而 FSQ 当前 batch
active ratio 仅 38.9%，两种利用率口径也不能直接混为一谈。结论是 EMA-VQ 继续作为主线，FSQ
保留为必要对照，尚无证据仅凭重建 RMSE 切换主量化器。FSQ checkpoint 位于
`artifacts/physical_language_tokenizers/fsq_4x128_steps1000/step-00001000.pt`。

上述 2/4/8 是首轮 8-GPU DDP 结果。收到 token 数可能过小的反馈后，增加同一单卡
`global batch=256` 的完整 pooled-feature 容量扫描；4/8 也按新口径重跑，避免将不同 validation
sampler 的微小差异误当成容量收益：

| Physics tokens x codes | Validation RMSE | Batch perplexity | Batch active ratio | Shuffle token change | Train time |
|---:|---:|---:|---:|---:|---:|
| 4 x 128 | 0.04671 | 46.66 | 64.7% | 89.7% | 35.1 s |
| 8 x 128 | 0.04630 | 50.90 | 66.1% | 89.4% | 35.9 s |
| 16 x 128 | 0.04669 | 43.38 | 63.7% | 89.8% | 36.6 s |
| 32 x 128 | 0.04636 | 43.09 | 63.1% | 89.9% | 37.6 s |
| 64 x 128 | 0.04636 | 46.22 | 67.8% | 89.5% | 40.9 s |
| 128 x 128 | 0.04633 | 49.06 | **74.5%** | 89.8% | 49.4 s |
| 256 x 128 | **0.04594** | 45.21 | 70.9% | **90.1%** | 66.4 s |

256 tokens 的 reconstruction 最好，但相对 8 tokens 只改善约 0.78%，且曲线并不单调，说明四个
mean-pooled 时间特征上的额外 queries 存在明显冗余。不能据此排除 256，也不能据此选定 256：
已导出 16/64/256 的 aligned-oracle 与 episode-shuffled cache，policy 端固定同时保留 4 个 predicted
action tokens 做成对短训。另一方面，新增 PhiZero-aligned 数据/模型路线，按源码的
`16 temporal transitions x 16 QFormer queries = 256 symbols` 保留空间运动结构；其结果必须与本表
分开报告。

### 21.5 PhiZero-aligned 256-token 路线

本地 PhiZero 源码确认其长度公式为：

```text
physical_language_length = (encoded_frames - 1) * act_embedding_num
                         = 16 transitions * 16 queries
                         = 256 symbols
```

因此“对四个全局 feature 使用 256 个 learned queries”不是公平复现。新增独立实现，且不修改已完成
的 pooled-feature 实验：

- `export_robotwin_spatiotemporal_features.py`：在 32-frame action horizon 内取 offset
  `[0,2,...,32]` 共 17 个时间点；保留每帧 `4x4=16` 个 Qwen3-VL 空间 token，不再全局 mean
  pool；每个 episode 的 frame feature 只存一次，样本通过 `source_frame_indices` 引用，避免重叠
  clip 展开到约 5.6 GB；
- `SpatiotemporalPhysicalLanguageTokenizer`：对 16 个相邻时间段分别用 16 个 learned queries
  cross-attend 空间 feature delta，逐段量化后 flatten 为固定 256 IDs；EMA-VQ 为主，FSQ 仍可切换；
- `train_spatiotemporal_physical_tokenizer.py`：只用 future-valid 的完整 0.64 秒 clips 训练码本，避免
  episode 尾部 clamp 产生的假静止污染；
- `export_spatiotemporal_physics_token_ids.py`：训练后覆盖全部 policy 样本导出 256 IDs，尾部 validity
  保留在输入 cache 中；
- `run_spatiotemporal_physical_tokenizer.sh`：8-GPU 训练入口。

辅助索引/空间 pooling 断言，以及 EMA-VQ、FSQ 两种 quantizer 下的 shape、loss、decode 断言均已
通过；默认模型检查 `17 frames -> 16 transitions -> 16 queries -> 256 tokens`。下一检查点是用 1 个
真实 episode 导出 cache，核对 Qwen vision grid、文件体积与 finite 值，再决定完整导出。该路线的
256-token 结果将与 21.4 的 pooled-feature 256-token 基线分表报告。

token 长度与词表大小必须分开消融。21.4 固定 128-code 词表，只回答序列长度问题；PhiZero 原协议
同时使用 25,000-code FSQ 词表。时空路线先用 `256 tokens x 128 EMA codes` 验证数据、重建和
policy 信号，再保持 256 长度比较更大 EMA codebook 与乘积为 25,000 的 FSQ levels。不能在首个
实验中同时改变 feature、长度、词表和 quantizer，否则无法判断收益来源；最终 reasoner 还必须报告
25k vocabulary 对 token accuracy、首 token accuracy、free-running error 和生成延迟的影响。

下一步同时推进容量与融合诊断：动作条件固定为 4 个 predicted action tokens；pooled-feature
tokenizer 完成 `16/32/64/128/256` 扫描，policy 先比较 16/64/256 physics tokens；加入可配置的
整组 condition dropout，并允许 action/physics gate 独立初始化。另行实现保留时空结构、按
`16 transitions x 16 queries` 输出 256 tokens 的 PhiZero-aligned tokenizer。只有 aligned physics
稳定优于 shuffled 且超过 action-only，才投入完整 reasoner。

### 21.6 Token 数 policy 短训结果（500 steps）

在同一个 seed、45/5 episode split、global batch、训练步数和固定 380-sample evaluation panel 下，
动作条件始终为 `4 predicted action tokens`；physics 分支使用 pooled-feature EMA-VQ oracle tokens，
physics dropout 为 0.2，action/physics gate 初值分别为 0.1/0.5。结果为：

| Condition | Action MAE | RMSE | Joint MAE | Endpoint MAE | Velocity MAE | Gripper acc. | Learned gates (action/physics) |
|---|---:|---:|---:|---:|---:|---:|---:|
| action-only | **0.14313** | **0.21879** | **7.74 deg** | **7.76 deg** | **0.08799 rad** | **92.39%** | 0.11279 / - |
| action + 16 physics | 0.16866 | 0.24523 | 9.13 deg | 9.34 deg | 0.12110 rad | 90.95% | 0.11182 / 0.49414 |
| action + 64 physics | 0.17341 | 0.24920 | 9.63 deg | 10.04 deg | 0.11848 rad | 91.46% | 0.11035 / 0.48633 |
| action + 256 physics | **0.16702** | **0.24474** | **9.13 deg** | **9.47 deg** | **0.10330 rad** | **91.80%** | 0.10986 / 0.48438 |

表中 physics 配置内的粗体表示三种长度中的最好值，不表示超过 action-only。256 tokens 在 action
MAE、RMSE、joint MAE、velocity MAE 和 gripper accuracy 上均为 physics 三组最佳，说明将长度从
极小的 2/4/8 扩到 PhiZero 量级是合理方向；但 16/64/256 三组仍全部显著落后于 action-only。
例如 256 相对 action-only 的 action MAE 增加约 16.7%，RMSE 增加约 11.9%。同时 physics gate
只从 0.5 降到约 0.484，并未在短训中自动关闭有害条件，因此“多一些 token 至少不会降点”不成立。

当前结论是：**pooled-feature 路线中优先级为 256 > 16 > 64，但整条路线 no-go**。不能把该排序
解释为 256 已验证有效，也不值得立即在同一表示上做更长训练。下一主实验改为 PhiZero-aligned
`16 transitions x 16 spatial queries = 256 tokens`；只有该时空表征在 aligned-vs-shuffled
负对照中显示语义增益，并达到或超过 action-only，才继续训练 VLM physics-token reasoner。

四个可复核评估文件位于：

- `artifacts/evaluations/qwenpi_e2_count_sweep_steps_500_pytorch_model_380samples.json`
- `artifacts/evaluations/qwenpi_b3_p16_drop20_pg05_steps_500_pytorch_model_380samples.json`
- `artifacts/evaluations/qwenpi_b3_p64_drop20_pg05_steps_500_pytorch_model_380samples.json`
- `artifacts/evaluations/qwenpi_b3_p256_drop20_pg05_steps_500_pytorch_model_380samples.json`

### 21.7 PhiZero-aligned 真实数据 smoke

单个真实 `click_alarmclock` episode 的导出已通过：76 个 unique frames，每帧保留
`16 spatial tokens x 2560 dims`，17 个 offset 为 `[0,2,...,32]`，最终 cache 为 5.5 MB。
`frame_features` 全部 finite；`source_frame_indices` shape 为 `[76,17]`；`future_valid_mask` shape
为 `[76,17]`；尾部所有 invalid 引用均 clamp 到 episode 最后一帧且 mask 正确。smoke cache 位于
`artifacts/future_feature_caches/spatiotemporal_smoke_1ep.npz`。全量 50 episodes 导出已在 node4
GPU 4 启动，预计文件为 300--400 MB；完成后先训练 `256 tokens x 128 EMA codes` 短预算版本。

### 21.8 PhiZero-aligned 全量 tokenizer 与 cache

全量导出完成，cache 为 308 MB，包含 50 episodes、4252 个 unique frames 和 4252 个 policy
samples；`frame_features` shape 为 `[4252,16,2560]`，全部 finite，索引范围为 `[0,4251]`。
按完整 0.64 秒 future clip 过滤后有 2652 个样本，其中 train/validation 分别为 2373/279；其余
1600 个 episode 尾部样本不用于 tokenizer 训练，但仍会在 token cache 中编码并保留 validity mask。

`256 tokens x 128 EMA codes` tokenizer 使用 8 GPUs、global batch 32 训练 1000 steps。最终验证：

| Metric | Value |
|---|---:|
| Reconstruction RMSE | 0.08622 |
| No-motion RMSE | 0.08823 |
| Relative RMSE reduction | 2.29% |
| Batch perplexity | 24.80 |
| Batch active ratio | 22.42% |
| Shuffled-future token change | 96.18% |

checkpoint 位于
`artifacts/physical_language_tokenizers/spatiotemporal_ema_256x128_steps1000/step-00001000.pt`。
短预算下重建只略优于 no-motion，因此其 policy 价值必须由 aligned/shuffled 对照判断，不能仅凭
token change 较高就认定有效。

对全部 4252 样本编码后，全局统计为 128/128 codes 全覆盖、perplexity 78.62、每个 transition
分别覆盖全部 128 codes、4153/4252 条 token sequence 唯一。validation 的 22.4% active ratio 是
单次分布口径，不等同于全局 code collapse。episode-internal shuffled cache 与 aligned cache 的
逐 token 差异率为 94.73%。已生成：

- `artifacts/token_caches/click_alarmclock_oracle_physics_spatiotemporal_ema_256x128_step1000.npz`
- `artifacts/token_caches/click_alarmclock_pred_action4_oracle_spatiotemporal_physics256.npz`
- `artifacts/token_caches/click_alarmclock_pred_action4_shuffled_spatiotemporal_physics256_seed42.npz`

下一实验保持 pooled-256 的所有 policy 超参数不变，训练 500-step aligned 和 shuffled 两组；二者都
固定包含 4 个 predicted action tokens 和 256 个 physics tokens。该成对实验完成前，不开始 VLM
physics-token reasoner。

### 21.9 PhiZero-aligned policy 成对结果（500 steps）

固定同一 action-only checkpoint 和 380-sample panel，时空 256-token 的结果为：

| Condition | Action MAE | RMSE | Joint MAE | Endpoint MAE | Velocity MAE | Gripper acc. | Physics gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| action-only | **0.14313** | **0.21879** | **7.74 deg** | **7.76 deg** | **0.08799 rad** | 92.39% | - |
| pooled-256 aligned | 0.16702 | 0.24474 | 9.13 deg | 9.47 deg | **0.10330 rad** | 91.80% | 0.48438 |
| spatiotemporal-256 aligned | 0.16295 | 0.24084 | 9.02 deg | 9.33 deg | 0.10525 rad | **92.72%** | 0.49023 |
| spatiotemporal-256 shuffled | 0.18427 | 0.26826 | 10.05 deg | 10.24 deg | 0.14998 rad | 89.76% | 0.49805 |

时空 aligned 相对 shuffled 的 action MAE 改善 11.57%，RMSE 改善 10.22%，velocity MAE 改善
29.83%，gripper accuracy 提高 2.96 pp；正确逐时刻物理语义有明确价值。时空 aligned 也优于
pooled-256 的 action MAE/RMSE/joint/endpoint/gripper，说明保留空间和 transition 结构是正确的。
但时空 aligned 相对 action-only 的 action MAE仍高 13.85%、RMSE 高 10.07%，仅 gripper accuracy
超过 action-only 0.33 pp，因此当前整体仍是 **representation pass, policy fusion no-go**。

结构诊断：现有 concat conditioner 先把 physics embedding 乘 scalar gate，再将 256 个 token append
到每层 VLM prefix。即使 gate 接近 0，这 256 个近零 key/value 仍进入 DiT cross-attention softmax
分母，因此 gate=0 并不严格还原 action-only；token 数越大，attention dilution 越明显。这解释了
“aligned 明显优于 shuffled，但两者都低于 action-only”的主要风险，且否定“只是多 token 所以至少
不会降点”的假设。

下一最小修正是保留 256 个输入 physics tokens，但改用 bottleneck residual cross-attention 将其写回
原 action-conditioned prefix：`prefix_out = prefix_action + gate * PhysicsCrossAttn(prefix_action,
physics_tokens)`。该形式在 gate=0 时严格等于 action-only，并保留旧 concat 模式用于对照。先做
20-step smoke，再只训练 aligned 500 steps；若仍不接近 action-only，停止增加训练预算，转向按
16 transitions 分层聚合或 FiLM。

Residual cross-attention 已实现并通过直接断言与 20-step 八卡 smoke：旧 concat 默认行为不变；
新模式读取全部 256 physics tokens，但 DiT prefix 长度保持为原 prefix 加 4 action tokens；输出投影
零初始化，初始化或 `physics_gate=0` 时均逐元素严格等于 action-only prefix。500-step aligned 结果为
action MAE 0.16572、RMSE 0.24653、joint MAE 8.98 deg、endpoint MAE 9.22 deg、velocity MAE
0.12035 rad、gripper accuracy 91.29%，没有优于 concat 时空 aligned，也仍低于旧 action-only。

该结果尚不能单独否定 residual 方案：新增模块构造会改变全局 RNG 状态，继而改变短训中的 diffusion
noise 序列；500 steps 的随机初始化方差可能与方法差异同量级。下一步补同一个 residual 架构、相同
token/dropout/seed、但 `physics_gate=0` 的 paired action-only control。由于 residual 输出投影零初始化，
gate=0 时 physics 分支所有梯度均为零，该 control 在整个训练过程中严格不使用 physics，同时保持
参数布局和 RNG 消耗一致。后续只看 residual gate=0.5 相对这个 control 的 paired 增益，不再与不同
架构的旧 E2 单次短训做过强结论。

### 21.10 Exact paired control 与当前最终判断

审计发现“可训练 gate 初始化为 0”并非严格 control：该 run 训练后 gate 漂到 0.00050，physics
output projection 也变为非零，因此不用于最终结论。新增 `physics_condition_enabled` 硬开关；关闭时
仍构造完整 residual 模块、保持相同参数布局和初始化顺序，但 forward 明确跳过 physics，直接断言
输出逐元素等于 action-only prefix，且 physics output projection 无梯度。

同架构、同 seed、同 token cache、同 dropout、同 500 steps 的 exact paired 结果为：

| Residual condition | Action MAE | RMSE | Joint MAE | Endpoint MAE | Velocity MAE | Gripper acc. |
|---|---:|---:|---:|---:|---:|---:|
| physics hard-disabled | **0.13163** | **0.21023** | **7.17 deg** | **7.55 deg** | **0.07184 rad** | **93.24%** |
| spatiotemporal-256 enabled | 0.16572 | 0.24653 | 8.98 deg | 9.22 deg | 0.12035 rad | 91.29% |

启用 physics 相对 exact control 的 action MAE 退化 25.90%、RMSE 退化 17.27%、joint MAE 退化
25.18%、endpoint MAE 退化 22.15%、velocity MAE 退化 67.52%，gripper accuracy 降低 1.95 pp。
control checkpoint 中 physics gate 保持 0.5，但 output projection weight/bias 均严格全零，证明
physics 分支未被训练或使用。不同架构的旧 E2 action-only 为 MAE 0.14313，而本 paired control 为
0.13163，也证明 500-step 单次短训存在不可忽略的架构初始化/RNG 方差，最终方法判断必须依赖 paired
control。

因此当前结论分两层：

1. **token 数/表示层面：选择 256 physics tokens。** pooled-feature 扫描中 256 的 reconstruction
   最好；policy 的 physics 配置中 256 总体优于 16/64；PhiZero 的 256 又有明确的
   `16 transitions x 16 queries` 结构依据。2/4/8 只保留低码率基线，不能作为主配置。
2. **policy 收益层面：当前 no-go。** 时空 aligned 显著优于 shuffled，证明 256-token 表示含有真实
   物理语义；但 concat 与 exact-safe residual 两种融合都没有超过各自 action-only control。暂不做
   3000-step 长训，也暂不训练 VLM physics-token reasoner。

下一轮若继续，应优先回答“哪些 physics 信息对动作有因果帮助”，而不是继续增加 token 数。最低成本
候选是按 16 transitions 保留分组、每个 transition 聚合 16 code embeddings 后做 FiLM/分层注入，
并用 hard-disabled paired control；同时对 future-valid 与 episode-tail 样本分开报告。只有 500-step
paired 指标至少不低于 control，才进入多 seed 和 3000-step 验证。

最终 residual 评估文件：

- `artifacts/evaluations/qwenpi_b3_st256_resxattn_drop20_pg05_steps_500_pytorch_model_380samples.json`
- `artifacts/evaluations/qwenpi_b3_st256_resxattn_disabled_control_steps_500_pytorch_model_380samples.json`

### 21.11 `doc1.md`、PhiZero 训练规模与数据瓶颈判断

`/mnt/jfzn/liujn/code/doc1.md` 的核心启发与当前路线一致：动作 token 和物理 token 应使用独立词表及
明确边界；先训练 tokenizer、离线导出监督，再训练 VLM reasoner，最后由连续动作头恢复控制精度；
离散 token 应承担粗粒度计划而不是单独承担精确低层控制。最终协议应分别严格校验两段输出，而不是
只按 token ID 范围筛选：

```text
<physics_bos> + 256 x <physics_i> + <physics_eos>
<action_bos>  +   4 x <action_i>  + <action_eos>
```

这里必须修正一个容易混淆的结论：本项目保留 `256` 的是 **physics tokens**，其结构依据是
`16 transitions x 16 spatial queries`；动作条件仍固定为 `4 predicted action tokens`。因此
`doc1.md` 中“不直接照搬 256 个动作 token”的提醒成立，但不能据此把结构化 physics tokens 也缩回
2/4/8。EMA-VQ 仍为主线，FSQ 为同长度、同输入、同 decoder 容量下的量化器对照。

PhiZero 公开材料能确认的正式训练信息如下：

- tokenizer 输入为 4 秒、8 FPS、33 帧、896x512 视频；没有随机 crop、时间采样或颜色增强；
- `33 frames -> 17 encoded frames -> 16 adjacent transitions x 16 queries -> 256 tokens`，FSQ
  vocabulary 为 25,000；
- 发布的 tokenizer 训练入口从已有 checkpoint 开始，同时优化 tokenizer、inverse-FSQ/context
  conditioner 和 DiT LoRA；Wan base DiT/VAE 冻结；目标是 future-video latent flow matching；
- 正式默认预算为 100,000 optimizer steps、batch size 1/GPU、gradient accumulation 2、LR `2e-5`、
  1000-step warmup、cosine decay、motion dropout 0.1 和 bf16；若用 `G` 张卡，每步有效 global batch
  为 `2G`，但这只说明训练 sample presentations，不能反推出 unique videos 数；
- reasoner 先做 general motion-token stage，再做 physics-filtered SFT；最终 SFT 为 8 nodes x 8 GPUs、
  batch 8/GPU、global batch 512、3 epochs、ZeRO-2，并全量训练 language model、visual encoder 和
  aligner；
- 公开仓库没有披露 tokenizer 原始视频数、两阶段 reasoner JSONL 行数或训练 step 数。因此不能声称
  PhiZero 使用了具体多少万/多少百万视频；仓库中的 11 个 `toy_videos.csv` 样本仅用于 1000-step
  source-domain adaptation，不是原始 tokenizer/reasoner 训练集。

与之相比，当前时空 tokenizer 只有 1 个任务、50 episodes，其中 45 train episodes、2373 个
future-complete train clips。1000 steps x global batch 32 等于约 32,000 次样本呈现，即对这批高度
重叠 clips 重复约 13.5 遍，但独立轨迹仍只有 45 条。其 reconstruction RMSE 只比 no-motion baseline
改善 2.29%。因此当前数据 **明显不足以支持“通用物理语言 tokenizer”**；继续把同一批数据机械地从
1000 steps 延长到 100,000 steps，增加的是重复次数而不是物体、接触模式和动力学覆盖，优先风险是
过拟合。

数据不足也不是唯一解释。当前目标是 Qwen visual feature delta reconstruction，而 PhiZero 用未来
视频 VAE latent 的 flow-matching reconstruction；另外，已验证 physics aligned 优于 shuffled，
但 policy 仍低于 hard-disabled control。因此后续必须分别诊断三个环节：

1. **representation/objective**：连续 latent 本身能否显著优于 no-motion；若不能，先修视觉目标或
   时域设计，不能归罪于量化器；
2. **quantization**：若 continuous latent 有效、EMA/FSQ token 无效，才是码本容量或离散化损失问题；
3. **action relevance/fusion**：若 token probes 和 aligned-vs-shuffled 有效，但 paired policy 无收益，
   才继续 FiLM、分层注入或因果信息筛选。

本机可用数据足以做低成本规模曲线：RoboTwin 标准目录有 50 个数据集、2500 episodes、549,787
frames，均为 50 FPS；`Randomized` 另有 37 个数据集、18,500 episodes、3,689,832 frames，但为
15 FPS。第一轮只选标准 50 FPS 数据，避免把数据规模收益与时域采样差异混在一起。建议覆盖八类运动：
`click_alarmclock`（按压/接触）、`beat_block_hammer`（冲击）、`open_microwave`（铰链）、
`shake_bottle`（往复动态）、`move_playingcard_away`（平面滑动）、`pick_dual_bottles`（双臂抓取）、
`handover_block`（交接）和 `place_object_basket`（容器放置）。

最小数据消融固定模型、256-token EMA 配置、optimizer steps 和抽样次数，先隔离“多样性”和“数量”：

| Split | 训练组成 | 目的 |
|---|---|---|
| D0 | 1 task x 45 episodes（现状） | 基线 |
| D1 | 8 tasks x 6 episodes = 48 episodes | 近似固定轨迹数，只增加运动多样性 |
| D2 | 8 tasks x 25 episodes = 200 episodes | 中等规模曲线 |
| D3 | 8 tasks x 45 episodes = 360 episodes | 首轮最大规模，不先动 Randomized |

每个 split 必须按 episode 切 train/validation，另保留至少 1--2 个完整未见任务做 held-out-task
reconstruction/probe；不能随机拆高度重叠 clips。统一报告 no-motion-relative RMSE、continuous/quantized
gap、global/per-transition perplexity、token temporal stability、aligned-vs-shuffled，以及对 joint
velocity、endpoint displacement、gripper/contact 的 frozen-token linear probe。判定规则为：

- D1 在近似相同 episode 数下明显超过 D0：主要瓶颈是运动多样性；
- 只在 D2/D3 改善：主要瓶颈是数据量；
- continuous oracle 随数据改善而 quantized 不改善：先做 `FSQ(4,4,8)=128` 与 EMA-128 的 matched
  quantizer control；若两者都受容量限制，再同时扩大 EMA codebook 与 FSQ effective vocabulary 到约
  25k，不能只放大 FSQ 后把容量收益归因于量化器类型；
- continuous 和 quantized 都不改善：当前 Qwen feature-delta objective 不适合，应转向 video latent 或
  action-relevant auxiliary targets；
- token probe 有效且 aligned 稳定优于 shuffled，但 policy 仍低于 paired control：才执行
  `16 transitions -> 16 vectors` 的 FiLM/分层注入。

`Randomized` 数据留到第二轮。其 15 FPS 与当前标准集 50 FPS 不同；当前 `[0,2,...,32]` frame offsets
在 50 FPS 表示 0.64 秒，在 15 FPS 会变为约 2.13 秒。混合前必须按真实 timestamp 重采样到同一物理
horizon，并同步动作标签，不能直接复用 frame index offsets。

### 21.12 Beyond Text Condition / BiVidGen 对当前方案的启发

论文源文件位于 `/mnt/jfzn/liujn/code/fusion_reproduction/paper_source`。它与本项目的对应关系很直接：
EMA tokenizer 将 VLM visual features 离散化，MLLM 自回归预测显式 visual tokens，DiT 再以这些 tokens
为条件恢复连续视频。它不是机器人控制论文，但回答了“离散中间 token 如何训练、如何进入 DiT”这两个
关键问题。

其公开数据和训练规模远大于当前 smoke：

- EMA tokenizer 使用完整的 10M videos + 20M images；视频来自 Sekai、FineVideo、Mira、OpenVid、
  InternVid、Panda 和 Koala，图像来自 LAION；
- 从 10M 视频中过滤 2M high-quality subset，用于 MLLM 最后阶段 SFT、视频 DiT 训练和主要消融；
  image DiT 另用 5M LAION images；
- MLLM 为 Qwen3-VL-2B，经历 image pretraining、video pretraining、2M-video continued training 三阶段；
  image batch 256、video batch 64，视频输入 121 frames。论文给出学习率但没有给出各阶段 epoch/step，
  因而不能继续反推总 sample presentations；
- renderer 为 Wan-2.2-5B，使用 32 x A100-40G、ZeRO-2、LR `2e-5`、warmup ratio 0.03、weight decay
  0.03；text/visual condition dropout 分别为 0.1/0.2；
- codebook 为 `16,384 x 2,048`，EMA decay 0.99，连续 20 steps 未使用的 code 会被重置。

论文的量化消融支持当前“EMA 主线、FSQ 对照”选择。EMA 的 quantization error 为 0.130，明显低于普通
embedding codebook 的 0.551；entropy 0.812、active ratio 0.841、top-10 ratio 0.146。加入 decoder 或
MLLM auxiliary loss 的可学习 embedding codebook 没有整体超过 EMA。该结果不是 EMA 在机器人数据上的
直接证明，但说明首先扩大数据覆盖和做好 dead-code reset，比在 45 条轨迹上不断叠加复杂 tokenizer loss
更有依据。

MLLM 侧也支持显式离散输出：同预算下 discrete-token CE 的 VBench-Long total score 为 72.79/72.95，
continuous-token MSE 为 70.19/69.14；离散 token 使用 causal attention 时 semantic score 为 48.74，
高于 bidirectional 的 46.26。可迁移设计是 physics/action 两段都采用严格 BOS/EOS、teacher forcing 和
target-only CE，并保留原生 causal attention；不能直接迁移这些视频生成分数到动作成功率。

对当前失败结果最有价值的是 DiT 注入位置。BiVidGen 最终不是先把 visual tokens 融入 text/MLLM
prefix，而是在 **每个 DiT block** 中让正在去噪的 latent `X_t` 作为 query、离散 token embedding 作为
key/value，执行 residual cross-attention：

```text
X_b+1 = X_b + CrossAttn(X_b, Proj(discrete_tokens))
```

它冻结原 DiT backbone，只训练新 cross-attention modules 和 projection。其 VBench-Long 对照为：

| Interface | Total score |
|---|---:|
| DiT-only | 74.88 |
| prompt refinement | 76.36 |
| latent bridging + connector | 67.43 |
| explicit tokens + first-block cross-attn | 78.18 |
| explicit tokens + multi-layer cross-attn | 80.77 |

这与本项目现象一致：条件包含信息不代表 connector 一定能利用；distribution mismatch 会让 latent
bridging 显著降点。也揭示了当前 `residual_cross_attention` 实验与论文的本质差别：当前实现以 VLM
prefix 为 query，先把 physics 写回 prefix，然后 action DiT 仍只 cross-attend 该 prefix；BiVidGen
直接以每层 noisy action/video latent 为 query。所以下一步不应把当前 residual no-go 错误表述为已经
否定了 BiVidGen-style multi-layer adapter。

拟增加一个独立融合对照 `dit_multilayer_cross_attention`，设计约束如下：

1. `4 action tokens` 仍进入原 VLM prefix 路径；`256 physics tokens` 不 concat 到 prefix，而是作为
   独立 key/value 送入每个 action-DiT block，两者必须同时存在；
2. 每层在原 block 后执行 `h <- h + gate * Out(CrossAttn(Norm(h), physics))`，其中 `h` 包含正在去噪的
   action latents/future tokens；output projection 零初始化，hard-disabled 时完全跳过分支；
3. 第一轮从同一个 action-only checkpoint 初始化并冻结 VLM 与原 action-DiT，只训练 physics
   embedding、position/type embedding、projection 和 adapters。这样 step 0 严格复现 action-only，
   避免重新初始化整个模型带来的 500-step 方差；
4. physics dropout 使用 0.2；先比较 first-block、early-4-block 和 all-block 三种注入深度，不同时修改
   tokenizer、token 数或 codebook；
5. 每个 enabled run 都配 hard-disabled paired control，并报告新增参数量、吞吐、显存和推理延迟；
6. 先用 oracle physics 判断动作头上限；只有 oracle 至少不低于 control，才训练 VLM 预测 physics
   tokens，并比较 oracle、teacher-forced predicted 和 free-running predicted 三档误差传播。

论文还测试了 hierarchical tokens：多层 ViT codebooks 能改善 DAVIS reconstruction，但 VBench-Long
generation 反而低于单层 token，因为预测难度和层间误差传播增加。因此“分层注入”不能被默认视为收益；
本项目的 `16 transitions x 16 queries` 应先保留同一个 tokenizer 的时间分组，只改变 adapter routing，
不要马上增加多层 codebook 或多套 VLM generation heads。

执行顺序据此更新为：先完成 21.11 的 continuous-vs-quantized 诊断和 D0/D1 数据多样性对照；若
representation 随数据改善，再做上述 BiVidGen-style adapter 的 20-step exactness smoke 与 500-step
paired test；FiLM/多层表征仅作为后续对照。原因是 BiVidGen 的成功依赖千万级 tokenizer 数据与百万级
下游训练，不能用它的结构结果来掩盖当前仅 45 条独立训练轨迹的数据缺口。

### 21.13 Continuous/quantized 复核与 D1 多任务落地（2026-09-27）

为把 representation objective 与离散量化损失分开，时空 tokenizer 增加了绕过 quantizer 的
`encode_continuous` / `reconstruct_continuous` 路径，并用独立评估入口
`scripts/physical_tokens/evaluate_spatiotemporal_physical_tokenizer.py` 对现有 D0 checkpoint 做完整
validation 重评。metric schema v2 的结果保存于
`artifacts/evaluations/spatiotemporal_ema_256x128_steps1000_metrics_v2.json`：

| 路径 | Validation RMSE | 相对 no-motion 改善 |
|---|---:|---:|
| no-motion | 0.0882812 | 0.00% |
| continuous（绕过量化） | 0.0851112 | 3.59% |
| EMA quantized | 0.0862601 | 2.29% |

quantized 相比 continuous 的绝对 RMSE gap 为 `0.0011489`。这说明量化确实丢失了一部分信号，但不能把
当前弱效果主要归因于 codebook：即使使用 continuous oracle，改善也只有 3.59%。因此优先级仍是扩大独立
轨迹和运动类型覆盖，并检查 feature-delta reconstruction 是否学到了动作相关动力学；不能先靠增大码本、
改 FSQ levels 或延长同一小数据集训练来解决。

D1 数据多样性对照已经实现，入口为
`scripts/physical_tokens/run_d1_multitask_tokenizer.sh`。它固定主配置为 **4 action tokens + 256 physics
tokens**；本阶段只训练 256-token EMA physics tokenizer，不改变既有 4-token action tokenizer。选取八个
标准 50 FPS 任务，每任务前 8 episodes，其中末 2 episodes validation、前 6 episodes train：

```text
click_alarmclock, beat_block_hammer, open_microwave, shake_bottle,
move_playingcard_away, pick_dual_bottles, handover_block, place_object_basket
```

为避免 physics tokenizer 看到 policy held-out 轨迹，导出先对每个任务排除固定 policy validation IDs
`[5,20,25,39,49]`，再取前 8 条剩余轨迹；每任务实际选择 `[0,1,2,3,4,6,7,8]`，其中最后两条
`[7,8]` 仅用于 tokenizer representation validation，前六条用于训练。manifest 显式保存排除集合和每任务
episode IDs，后续 cache reuse 必须核对该 provenance。导出包含 64 条独立轨迹、14,077 个 unique frames，
cache 约 1.0--1.1 GB。未截断时有 8,785 个 future-complete train clips 和 3,244 个 validation clips；
训练时对每个任务 cap 300 clips，click-alarmclock 只有 294 个完整训练 clips，因此合计 2,394，
与 D0 的 2,373 clips 近似相同。这样 D0/D1 的主要自变量是运动多样性，而不是 optimizer steps 或每步看到
的 clip 数；八卡 global batch 32、1000 steps 下约为 D1 13.37 vs D0 13.48 个 effective epochs。
cache schema 升级为 v2，保存 `task_ids` / `task_names`；split 使用 `(task_id, episode_id)`
复合键，避免不同任务中同名 episode 串到同一个 split。旧单任务 cache 缺少 `task_ids` 时仍兼容为 task 0。

训练入口在导出或复用后都会运行 `validate_d1_feature_cache.py`，不解压 1 GB feature tensor，只读取 manifest
与小型 key/mask arrays，强制核对：八任务顺序、排除 IDs、每任务 8 个实际 episode、无 policy-held-out
交集、14,077 unique frames、8,785/3,244 uncapped train/validation clips 和 2,394 capped train clips。
这既阻止误复用旧的 episode-5 泄漏版 cache，也能在昂贵八卡训练前发现导出不完整或数据源变化。exporter
selection 与 validator 的定向回归合计 `8 passed`。

D1 完成后必须用相同 evaluator 同时报告 no-motion、continuous 和 quantized 三条结果，并按以下门槛决定
后续路线：

1. continuous 相对改善明显高于 D0 的 3.59%，且 quantized 跟随改善：进入 BiVidGen-style action-DiT
   adapter smoke；
2. continuous 改善、quantized 不改善：representation 已有收益，先诊断 EMA utilization/dead codes，
   再做同输入、同长度、同 decoder 容量且同 128-code 容量的 `FSQ(4,4,8)` 对照；只有 matched 对照仍
   指向容量瓶颈，才同时扩大 EMA 与 FSQ 到约 25k effective vocabulary；
3. continuous 仍接近 D0：停止堆叠融合结构，优先替换为 future-video latent objective，或加入 joint
   velocity、endpoint displacement、gripper/contact 等 action-relevant auxiliary targets；
4. 无论 reconstruction 是否改善，在 oracle physics 的 paired action test 不低于 hard-disabled control
   之前，都不启动昂贵的 VLM physics-token reasoner 训练。

上述 D0/D1 数值不能来自各自不同的 validation 分布。D0 原报告只含 `click_alarmclock`，D1 报告含八任务，
直接比较会把任务难度差异误当成数据多样性收益。D1 runner 现会把 **D0 与 D1 两个 checkpoint 都在同一个
D1 validation cache** 上重评，分别写出 `evaluation_d0_on_d1_per_task.json` 与
`evaluation_per_task.json`，再由 `compare_spatiotemporal_tokenizers.py` 生成
`comparison_d0_vs_d1.json`。输入路径、任务顺序、总 clips 和每任务 clips 任一不一致都会拒绝比较。
此外 no-motion baseline 完全不依赖 tokenizer checkpoint，comparator 现要求两份报告的 aggregate、task-macro
及每任务 `no_motion_rmse` 在数值容差内一致；任一不一致都视为 cache/split/evaluator provenance 已改变，
直接拒绝 gate。报告同时保存 checkpoint SHA256/size 和完整 tokenizer `model_config`；comparator 要求 D0/D1
架构完全一致，从而本轮只比较训练数据多样性与所得权重，不把 token 数、hidden dim 或 codebook size
变化混入。上述不变量均有专门负例测试。

为避免看结果后移动门槛，paired representation gate 预注册为同时满足：continuous task-macro RMSE 相对
D0 改善至少 2%，quantized macro RMSE 改善至少 1%；两者都至少在 75% 的任务上获胜（八任务时为 6/8）；
D1 自身相对 no-motion 的 task-macro continuous/quantized 改善至少为 5%/3%。门槛报告每个 check 的布尔值，
只有全部为真才允许进入 adapter action smoke。阈值不是统计显著性声明，而是本次低成本筛选的最小 effect
size；完整结论仍需后续多 seed policy success rate。

GPU 启动条件：运行前重新检查 8 卡占用；若存在其他用户训练或显存保留进程，不抢占、不终止。D1 导出和
训练仅在资源明确空闲后执行：

```bash
bash scripts/physical_tokens/run_d1_multitask_tokenizer.sh
```

D1 validation 的 clip 数并不均衡：八个任务依次为 100、164、1261、442、167、193、490、427，长轨迹
`open_microwave` 占接近一半。因此只看 sample-weighted aggregate 会掩盖其他运动类型。评估入口现同时输出
`metrics`（clip-weighted aggregate）、`task_metrics`（逐任务）和 `macro_metrics`（八任务等权平均）；D1
脚本训练结束后自动写入 candidate、D0-on-D1 和 paired comparison 三份 JSON。脚本默认复用已存在的非空 cache，
只有 cache 不存在或显式设置 `D1_REEXPORT=1` 时才重新导出。

为避免共享节点抢卡，D1 入口默认在特征导出前、八卡训练前各执行一次 GPU preflight：只要 `nvidia-smi`
报告任一 compute process 就以
exit code 2 拒绝启动，并打印占用 PID/进程/显存。只有人工确认资源可共享时才允许设置
`D1_REQUIRE_IDLE_GPUS=0`。2026-09-27 00:30 检查时，当前 `node001` 八卡均被其他用户占用；授权的
`node004` 也有其他用户任务（GPU 0--2 为小显存高利用率训练，GPU 3--7 为约 9 GB/卡的评估），因此本轮
没有启动 D1，也没有终止或干扰这些进程。

长跑重入也采用保守策略：目标 step checkpoint 已完整存在时跳过训练、直接重新评估；若输出目录只有部分
checkpoint/日志，则拒绝把新 run 追加到旧 `metrics.jsonl`。此时应先审计 partial run，再选择显式 resume
实现或新的 `D1_TOKENIZER_OUTPUT`，不能静默覆盖实验 provenance。

node004 使用 `scripts/physical_tokens/wait_and_run_d1_on_idle_gpus.sh` 做保守排队：文件锁阻止重复 watcher，
默认每 60 秒检查一次，连续 3 次完全没有 compute process 且所有 GPU used memory 均不超过 256 MiB 才
启动 D1。显存条件用于捕获 `--query-compute-apps` 不报告的 graphics/simulator contexts，阈值可由
`D1_IDLE_MEMORY_THRESHOLD_MIB` 调整；D1 导出前、训练前和 gated 后续阶段的 preflight 使用相同规则。
watcher 只解决无人值守等待，不会
绕开 D1 的两次 preflight；若确认空闲后出现竞争，D1 仍会退出。`D1_WATCH_MAX_CHECKS` 可用于有限次 smoke，
默认 0 表示持续等待。

2026-09-27 00:44 已在 `node004` 启动唯一 watcher（PID `130145`），日志为
`logs/d1_node004_watcher.log`。首轮检测仍有 8 个其他用户的 GPU compute processes，因此当前状态是
等待而非训练；它在满足连续三次空闲检查后才会进入 D1。

2026-09-27 01:20 复核：远端 watcher 仍存活，且工作目录与锁均指向本实验副本。此时 GPU 0--2 已空闲，
GPU 3--7 仍有 5 个外部 Python compute processes（每进程约 9.2 GiB，卡上总占用约 16 GiB），所以按
“八卡全部空闲并连续确认三次”的预注册条件继续等待，D1 cache、输出目录和训练均未提前创建。再次静态
审计确认 watcher 继承的默认 `vlact` Python 路径在 node004 存在；D1 runner 的导出前/训练前双 preflight、
完整 checkpoint 重入和 partial-output 拒绝覆盖逻辑保持有效。

2026-09-27 02:25 发现旧检查只看 compute-app table，会漏掉 GPU2 上约 2.3 GiB 的 graphics/simulator
contexts。已仅终止并替换本实验自己的旧 watcher PID `130144/130145`，新 watcher PID 为 `325179`；
历史日志采用 append 保留。新检查同时观察 compute apps 与 per-GPU used memory，启动时正确识别 GPU0--3
仍各有约 16 GiB 外部占用并继续等待；未终止或修改任何外部进程。

本轮定向回归为 `12 passed`，覆盖 EMA/FSQ shape、默认 256-token contract、continuous bypass、跨任务
frame index、复合 episode split、legacy cache、逐任务 grouping，以及 metric schema v2 的 batch-size
invariance；shell syntax 和相关 Python `py_compile` 也通过。全量 `tests/` 在 `starvla_qwen` 环境收集阶段
因缺少可选依赖 `snntorch` 停止，而包含 `snntorch` 的项目默认 `vlact` 环境没有安装 pytest。这是现有环境
组合问题，不影响上述定向覆盖；没有为了凑全量测试结果而修改框架自动导入或在线安装依赖。

### 21.14 BiVidGen-style action-DiT adapter 骨架（未越过 D1 gate）

代码审计确认，真正的 noisy action query 位于
`LayerwiseFlowmatchingActionHead` 的 `model_output`：它由 state token、future tokens 和当前噪声动作轨迹
embedding 组成，并逐个通过 `self.model.transformer_blocks`。现有 `DiscreteTokenConditioner` 的
`residual_cross_attention` 则以 VLM prefix 为 query、在进入动作头前修改 prefix；两者不是同一种接口。

新增 `starVLA/model/modules/conditioning/dit_physics_adapter.py`，提供独立的
`MultiLayerPhysicsTokenAdapter` 骨架：

1. 输入保持 `256 physics token IDs`，共享一套 physics code embedding、256-position embedding、type
   embedding 和 memory norm；4 个 action tokens 仍保留在原 VLM prefix 路径，adapter 不吞并动作 token；
2. 每个选定 DiT block 使用独立 `LayerNorm -> MultiheadAttention -> zero-init output projection`，调用位置
   设计为原 block 之后，query 是该层正在去噪的 action states，key/value 是 physics memory；
3. 支持 `first`、`early-4`、`all` 和显式逗号分隔层号；physics dropout 在每次 denoising pass 的 memory
   preparation 阶段只抽样一次，所有注入层共享同一决定；
4. `enabled=False`、dropout 命中或未选择的层完全不进入 attention，直接返回原 tensor；零初始化 enabled
   分支也逐元素严格复现原输出，随后 output projection 能先获得梯度，再逐步打开上游 attention/embedding；
5. 当前 512 hidden、8 heads、8 DiT blocks、128-code EMA vocab 下，first / early-4 / all 新增参数分别约
   `1.51M / 5.46M / 10.72M`，其中共享 token/position 部分约 `0.20M`。

`QwenPIPhysicalTokens -> LayerwiseFlowmatchingActionHead` 的 dormant 接线也已完成：新融合模式
`dit_multilayer_cross_attention` 会让 `DiscreteTokenConditioner` 只把 action IDs 追加到每层 VLM prefix，
另由 framework hook 把 physics IDs 直接传给 action head；训练阶段 repeated diffusion batch 会同步 repeat
physics IDs，推理阶段每个 Euler denoising step 都复用同一 physics memory。旧 YAML 仍默认 `concat`，因此
已有运行行为不变。

adapter-only 下所有旧模块均冻结，因此 whole-condition physics dropout 命中时若直接 bypass adapter，loss
将不依赖任何可训练参数，`backward()` 会报 no-grad；不同 rank 独立命中还会造成 DDP 参数使用集合不一致。
现保持 dropout batch 的输出逐元素不变，同时给所有 trainable adapter 参数附加零值 autograd anchor：
该 rank 得到显式零梯度，未命中的 rank 正常贡献梯度，DDP 可安全归并。adapter 单元与完整冻结 action-head
路由均新增强制 dropout 的 backward 回归，全部 adapter 参数 grad 非空且 dropped branch 的 grad 全零。

`dit_freeze_base_action_model=true` 时，action encoder、原 DiT blocks、decoder、state/future embeddings
全部冻结，仅 physics adapter 可训练；配合冻结 `qwen_vl_interface,discrete_conditioner` 和加载同一个
action-only checkpoint，才能形成 adapter-only paired test。`physics_condition_enabled=false` 不准备 memory、
不进入 attention，作为同权重 hard-disabled control。

训练器新增严格的 `trainer.action_init_checkpoint` 第二阶段加载：先按原流程从 100k VLA checkpoint 只恢复
`qwen_vl_interface`，随后从 action-only checkpoint 恢复 `action_model.*` 和
`discrete_conditioner.*`。loader 只允许新模型缺少 `action_model.physics_adapter.*` 与
`discrete_conditioner.physics_*`；任何原动作头/action-token 参数缺失、unexpected key 或 shape mismatch
都会报错，不能退化成宽松的全模型 `strict=False`。真实
`qwenpi_physical_tokens_click_alarmclock_e2_seed42_steps3000` checkpoint 已完成 CPU smoke：142 个旧键成功
恢复，原 action decoder 与 action embedding 逐元素匹配，新增 all-block adapter 为 `10,720,768` 参数且
只有该 adapter 在 action head 内可训练。

测试覆盖层选择、256-token shape、zero-init exactness、hard-disabled object identity、训练后分支确实改变
输出、dropout-once、action-head 层级 exact control、adapter-only requires-grad 边界、optimizer 中只有
physics adapter 参数，以及严格二阶段 checkpoint loader。2026-09-27 01:20 的完整定向回归为
`28 passed`。

同一轮还专项审计了八卡 tokenizer 的 EMA 语义。`EMAVectorQuantizer` 的 codebook、EMA count/sum 都是
buffer，不能依赖 DDP gradient synchronization；当前实现会对每步各 rank 的 assignment counts 和 latent
sums 做全局 `all_reduce(SUM)`，首次 codebook 初始化也由 rank 0 广播，因此各 rank 不会用局部 mini-batch
各自演化出不同 codebook。新增真实 Gloo 两进程回归：给两个 rank 不同 latent/ID 后，断言
`codebook`、`ema_count`、`ema_sum` 逐元素一致，并核对全局计数；`tests/test_action_tokenizer.py` 为
`6 passed`。该测试将分布式 EMA 同步作为 D1 八卡运行前的硬性不变量。

同时修正训练期 distributed validation 的精确性：PyTorch `DistributedSampler(shuffle=False)` 会为使
3,244 个 validation clips 能被 8 整除而补入 4 个重复样本。虽然 D1 gate 最终使用单卡 evaluator、不会
受此影响，但训练日志和 checkpoint metadata 会出现轻微 sample weighting 偏差。现改为 rank-strided、
无 padding 的 exact shard；评估前向本来就使用 raw model，只在循环后 `all_reduce` totals，因此各 rank
拥有不同 batch 数不会造成 DDP collective 错位。新增 11 samples / 4 ranks 覆盖恰好一次的回归；相关
tokenizer 测试合计 `10 passed`，Python `py_compile` 与 D1/watcher shell syntax 均通过。

node004 的 `/tmp/vlact_qwen3_pretrain_steps100000.pt` 本地副本已重新计算 SHA256，为
`5da4b96f9c923eabf35b7ac4c8928bef847266a044dbb8a239e3696aa305095d`，与导出器预注册的 100k 权重
完全一致。D1 runner 现与 policy 训练入口采用相同选择规则：本地副本存在时优先读取，缺失时回退共享
checkpoint；feature exporter 仍强制核验上述哈希。这样不改变模型 provenance，并减少共享文件系统读取
11 GB 权重的压力。

为支持无人值守推进，D1 runner 在写完三份 paired representation 报告后默认执行严格 gated continuation
（可用 `D1_AUTO_GATED_SMOKE=0` 禁用）：先调用同一 gate guard；只有全部六项门槛通过，才生成 click-alarmclock
的 `4 predicted action tokens + 256 D1 oracle physics tokens` cache，训练 first-block adapter-only 20 steps，
随后对同一 checkpoint、同一 sample panel 做 physics-enabled / hard-disabled 380-sample paired evaluation。
cache 导出前、adapter 训练前和 paired evaluation 前均再次要求八卡无 compute process。若 representation
gate 失败，只记录失败 checks 并停止，不训练 adapter。20-step smoke 同样支持安全重入：最终 checkpoint
完整时复用，只有 partial run directory 时拒绝覆盖。500-step first/early-4/all 对照仍不自动启动，必须先
审计 20-step 结果。

纳入 action tokenizer、D1 metric/gate、checkpoint loader、DiT adapter、routing 和 optimizer 边界后的完整
定向回归在加入 bootstrap、episode exclusion、cache-contract 和 no-motion paired invariant 测试后为
`52 passed`（2026-09-27 02:08）；相关 shell syntax 在 node001/node004 均通过。

paired policy evaluator 进一步增加逐 episode 的 metrics 与可合并充分统计量，comparator 用固定 seed、
10,000 次 **paired episode-cluster bootstrap** 报告 95% 区间、逐 episode wins 和 cluster 数。这里以整条
episode 为重采样单位，避免把同一轨迹的相邻 observation 当成独立样本；enabled/control 每个 bootstrap
replicate 使用完全相同的 episode multiset。原 0.5% aggregate no-regression gate 保持不变，bootstrap
只增加不确定性诊断。20-step 结果仍只能证明接线、梯度和无明显退化，不能声明方法有效；500-step 晋级时
除 aggregate gate 外，应要求核心 action MAE / wrapped joint MAE 的区间方向一致，并在多 seed rollout 前
明确报告仅 5 个 episode clusters 导致的宽区间。纯 comparator 新测试为 `4 passed`，`vlact` 环境中的
MetricAccumulator sufficient-statistics smoke 通过；完整 evaluator pytest 在 `starvla_qwen` 中仍受既有
可选依赖 `snntorch` 缺失限制。

adapter paired evaluator 还会显式传入训练时的 `physics_condition_dropout=0.2` 并写入报告，而不是沿用
基础 YAML 的 `0.0`。dropout 在 `model.eval()` 下不会改变预测，但保持该字段一致可避免把训练配置的
provenance 记录错；enabled 与 hard-disabled 报告除唯一开关外仍必须完全一致。

另在含完整框架依赖的 `vlact` 环境完成 routing smoke，确认 `4 action prefix tokens + 256 separate DiT
physics IDs`，关闭全局 discrete conditioning 后两条路径均严格 bypass。这里仍没有创建训练 run；必须先看
D1 metric schema v2 的 continuous/quantized 与 task-macro 结果。若 D1 不过 representation gate，代码
保留但不运行，以免用更强 connector 掩盖无效表征。

评估口径同步升级为 metric schema v2：RMSE 从全验证集累计 MSE 后统一开方，perplexity/active ratio 从
全局 code histogram 计算，不再平均每个 batch 的非线性统计。由此 D0 的全局码本使用情况被纠正为
active ratio `0.9297`（119/128 codes）和 perplexity `96.46`，说明当前 EMA codebook 没有明显 collapse；
旧报告中的 `0.2405/26.25` 是逐 batch 平均，不能用于判断全局利用率。原跨 batch 的
`shuffle_token_change` 也替换为 batch-size 无关的 `temporal_reverse_token_change`：固定反转每个样本内部
的 future 帧序列。v1/v2 的该诊断值不可直接横向比较。

若 D1 过 gate，下一项实现固定为 `dit_multilayer_cross_attention`：4 个 action tokens 保持原 prefix
路径，256 个 physics tokens 独立作为每层 action-DiT 的 key/value，noisy action latent 作为 query；新增
output projection 零初始化，并提供 hard-disabled exact control。先做 20-step exactness smoke，再比较
first-block / early-4 / all-block 的 500-step paired test。这里不再把“prefix residual cross-attention”误称
为 BiVidGen-style 注入。

### 21.15 D1 无人值守启动复核（2026-09-27 02:36）

node004 的新 watcher（PID `325179`）持续存活，但 D1 仍未启动。02:31--02:34 一度只剩 GPU3 上一个
外部 compute process；随后其他用户的 Habitat/OVON evaluation 又在 GPU0、GPU2、GPU3 建立约
16 GiB/卡的上下文并保持约 70% 利用率。02:36 检查为 3 个 compute-app rows、3 张卡超过 256 MiB，
因此连续三次全八卡空闲条件没有成立。只做了只读 PID/命令核验，没有终止、暂停或修改任何外部进程。
02:58 再次复核时只剩 GPU3 上同一个外部 OVON 全量 evaluation（约 15.3--16.1 GiB，进程仍活跃），
其余七卡空闲；D1 cache/output 仍未创建，watcher 继续等待连续三次全空闲。

启动前再次逐项审计了 D1 runner：特征 cache 与 manifest 使用临时文件加 `os.replace`，tokenizer
checkpoint 同样原子替换；cache 完成后必须通过 schema v2、固定八任务/episode IDs、policy-held-out
排除和精确 clip/frame 数验证；训练只复用完整的 1000-step checkpoint，存在 partial output 时明确拒绝
覆盖。D0/D1 评估固定读取同一 D1 validation cache，comparator 再检查 input、task order、clip counts、
model config 以及 aggregate/macro/per-task no-motion RMSE 一致。自动 continuation 仍受预注册 gate
保护，未通过时不会生成 D1 policy cache 或启动 adapter；通过后也会在 cache、训练、paired evaluation
三个边界重新执行 GPU idle preflight。上述复核没有改变 `4 action + 256 physics`、EMA-128、1000 steps
或任何 gate 阈值。

### 21.16 256 个名义 token 的结构冗余诊断（2026-09-27）

全局 active ratio/perplexity 只能说明整个 validation set 用过多少 code，不能证明单条 256-token 序列
包含 256 个有效离散变量。新增
`scripts/physical_tokens/analyze_spatiotemporal_token_structure.py`，在不修改 checkpoint 的前提下统计：
逐位置 entropy/active codes、transition 内唯一 code 比例与 query-pair change、相邻 transition 同 slot
变化、整条样本唯一 code 数，以及 token position 与 code 的 mutual information。D0 的 279 个严格
validation clips 已用 CPU 完成，报告为
`artifacts/evaluations/spatiotemporal_ema_256x128_steps1000_token_structure.json`：

| 指标 | D0 结果 | 解释 |
|---|---:|---|
| Global active codes / perplexity | 119 / 96.46 | 全数据集码本覆盖健康 |
| Mean per-position active codes / perplexity | 105.13 / 88.89 | 每个位置跨样本也会变化 |
| Within-transition unique ratio | **0.06426** | 每组 16 tokens 平均仅约 1.03 个不同 code |
| Within-transition pairwise token change | **0.00861** | 同组 query pair 有 99.14% 输出相同 code |
| Adjacent-transition same-slot change | 0.99731 | 相邻时间 transition 基本都会换 code |
| Per-sample unique code ratio | 0.11999 | 每条序列平均约 15.36/128 个不同 code，接近 16 transitions |
| Position/code normalized MI | 0.01800 | code 几乎不承载 query-position specialization |

进一步在同一脚本中加入 continuous latent 诊断，以区分 encoder collapse 与 quantizer collision。D0 的
transition 内连续 latent pairwise cosine 为 `0.999934`，相对组内离散度
`RMS(z - mean_query(z)) / RMS(z)` 仅 `0.007997`。中心化后的微小残差有效秩为 `10.79`，但残差能量只占
总 latent 能量约 `0.007997^2 = 0.0064%`。因此 16 个 query **在进入 EMA 之前已经近乎同向同值**；
量化只是将这些微小残差合并成同一个 code，并非本次重复的首要来源。

因此当前 tokenizer 虽然输出 shape 是 `16 transitions x 16 queries = 256 IDs`，信息结构实际上接近
**16 个随时间变化的 code，各自在组内复制 16 次**。这也解释了为什么“增加到 256 tokens”本身没有
保证得到 PhiZero 意义上的 256 个物理 token。根因更可能是 decoder 可以依靠 output/spatial queries 从
单个 transition latent 重建全部 16 个空间位置，训练目标没有迫使 16 个 motion queries 分工，而不是
EMA codebook collapse。

为避免 reconstruction gate 通过后把这种重复序列接入更强 DiT 并误判收益，新增独立的、在看到 D1
结果前固定的结构 gate：`within_transition_unique_ratio >= 0.25`（平均至少 4/16 个不同 code）且
`within_transition_pairwise_token_change >= 0.20`。门槛只表示“最低限度不是复制序列”，不是最佳 token
数的理论结论。D0 两项均失败，gate 报告为
`artifacts/evaluations/spatiotemporal_ema_256x128_steps1000_token_structure_gate.json`。

D1 runner 现会在相同 D1 validation cache 上分别生成 D1 与 D0 的 token-structure per-task 报告，并写出
`token_structure_gate.json`。D1 训练和 reconstruction 对照仍会完整执行，用于判断数据多样性的影响；
但自动 policy cache/adapter smoke 必须同时通过原六项 representation gate 和新增两项 structure gate。
`prepare_d1_policy_token_cache.sh`、adapter train 与 paired eval 三个独立入口也都会重新验证两种 gate，
不能通过绕开总 runner 误启动。新增 accumulator/gate 单元测试为 `4 passed`，相关 shell syntax 和 Python
编译检查通过。

若 D1 reconstruction 改善但结构 gate 仍失败，下一轮不应先扩大 EMA/FSQ codebook，因为重复来自组内
continuous latent 利用不足，FSQ 无法恢复 encoder 没有编码的信息。优先实验应是保持 256 输出长度和相同
视觉 cache，加入 query diversity/slot assignment 约束，或让每个 token 直接负责固定空间位置，再比较
continuous reconstruction 与结构多样性；只有 continuous latent 已明显分工、量化后却重新合并时，才进入
matched EMA-128 vs FSQ-128 的量化器诊断。

### 21.17 Spatial-aligned 结构对照骨架（2026-09-27）

本地 PhiZero tokenizer 代码进一步确认其 256 tokens 来自 adjacent-frame Q-Former：每对相邻视频 latent
使用多层 cross-attention/MLP 和 learned queries，并为 FSQ 提供 soft entropy loss。当前简化实现只有单层
cross-attention，同时 decoder 的每个空间 output query 可以读取同一组全局 transition memory，因而没有
迫使 16 个 motion queries 分别承担空间内容。这个差异与 21.16 的 continuous collapse 诊断一致。

在不影响旧 checkpoint 的前提下，`SpatiotemporalPhysicalLanguageTokenizer` 新增默认关闭的
`token_routing=spatial_aligned`：仍使用相同 encoder/decoder attention、EMA-VQ、hidden/latent 维度和
`16 transitions x 16 queries`，但 attention mask 只允许第 i 个 query 读取第 i 个 4x4 pooled spatial
位置，decoder 也只允许第 i 个 output 读取第 i 个 latent。这样不是额外加容量，而是施加明确的 slot
责任；要求 `queries_per_transition == spatial_tokens`。默认 `global_qformer` 的参数、state dict 和 D1
行为保持不变，旧 checkpoint 的 `model_config` 也不会因默认字段而改变。

真实 2048/2560 维 feature cache 上的 2-step CPU smoke 已跑通 forward/backward、EMA、validation 和原子
checkpoint 保存；结构指标如下。该 smoke 使用 hidden 64 / latent 32，只用于接线和机制检查，不能与正式
D0 reconstruction 数值作模型质量比较：

| 指标 | D0 global routing | Spatial-aligned 2-step smoke |
|---|---:|---:|
| Continuous query pair cosine | 0.99993 | **0.16987** |
| Continuous relative dispersion | 0.00800 | **0.87937** |
| Within-transition unique ratio | 0.06426 | **0.80330** |
| Within-transition pairwise token change | 0.00861 | **0.96534** |
| Per-sample unique code ratio | 0.11999 | **0.62273** |
| Adjacent-transition same-slot change | 0.99731 | 0.99332 |

这证明 spatial slot 约束能解除“每个 transition 复制同一个 token”的结构性退化；但 smoke step 2 的
continuous/quantized RMSE 仍比 no-motion 差约 1.70%/1.74%，所以尚不能认为表示有效。若 D1 仍未过结构
gate，D2 应在同一 D1 cache 上用完全相同的 1000 steps、global batch 32、EMA-128 和模型宽度训练
`spatial_aligned`，同时要求：原 representation effect-size 不退化、结构 gate 通过。只有这两类条件同时
成立，才进入 action-DiT；不能因为 token 看起来更丰富就跳过重建/策略信号验证。相关 routing/locality、
训练器和结构诊断测试当前合计 `18 passed`，2-step 端到端 CPU smoke 通过。

D2 的可重入入口已准备为
`scripts/physical_tokens/run_d2_spatial_aligned_tokenizer.sh`，但不会被 D1 watcher 自动触发。它要求 D1
cache、1000-step checkpoint 和 D1 per-task evaluation 均已完整存在，重新执行精确 cache validator 和 GPU
idle preflight，然后以相同 D1 split/cap、global batch、1000 steps、EMA-128 只改变
`token_routing=spatial_aligned`。完成后输出 `evaluation_per_task.json`、`token_structure_per_task.json`、
`comparison_d1_vs_d2.json` 和 `token_structure_gate.json`。通用 comparator 默认仍严格要求 model config
完全相同；只有 D2 入口显式允许 `token_routing` 这一项预注册差异，其他宽度、码本、token 数等任一变化仍
会拒绝比较。当前相关定向集合更新为 `19 passed`，shell syntax/Python compile 均通过。

D2 不直接复用 D0→D1 的“再改善 2%/1%”晋级语义。新增 joint gate 将结构修复视为主要自变量：相对
D1 的 task-macro continuous/quantized RMSE 最多退化 0.5%；至少 6/8 任务各自退化不超过 1%；D2 自身
相对 no-motion 的 continuous/quantized 改善仍须达到 5%/3%；同时满足 unique ratio 0.25 和 pairwise
change 0.20。这样允许为真实 256-slot 表示付出不超过预注册范围的微小 reconstruction tradeoff，但不允许
仅凭 token 多样性放行明显更差的表示。comparison 会标记为
`paired_same_cache_d1_vs_d2_spatial_routing`，并记录唯一允许变化的 config key；joint gate 输出
`joint_gate.json`。纳入 D1/D2、token structure、DiT adapter、checkpoint loader 和 paired evaluator 的统一
定向回归现为 `54 passed, 1 warning`；warning 仅来自 albumentations 在离线环境中查询版本失败。

### 21.18 正式 Spatial-aligned、matched FSQ 与 action-DiT 结果（2026-09-27）

在与原 D0 相同 split/cache、EMA-128、1000 steps 和 global batch 32 下，已完成正式
`spatial_aligned` 训练。为不等待被外部任务占用的整机，本次在 node004 GPU7 以单卡 batch 32 运行，样本
暴露与原八卡 global batch 32 完全一致，训练约 87 秒。checkpoint 为
`artifacts/physical_language_tokenizers/spatiotemporal_d0_spatial_aligned_ema_256x128_steps1000_b32_1gpu/step-00001000.pt`，
SHA256 为 `756e20d4a78db146b70371f4a581d7f877276ff460e1c5a9ace60cb9d28cade6`。

schema-v2 paired 结果如下：no-motion RMSE `0.08828116`，continuous RMSE `0.07756699`
（相对 no-motion 改善 `12.14%`），quantized RMSE `0.08032148`（改善 `9.02%`），全局 active codes
`128/128`、perplexity `124.16`。相对原 global-routing D0，continuous/quantized RMSE 分别改善
`8.86%/6.88%`。结构指标也从复制退化恢复：continuous cosine `0.12349`、relative dispersion
`0.90663`、within-transition unique ratio `0.89526`、pairwise token change `0.98441`、per-sample unique
ratio `0.81706`。因此正式 representation gate 与 structure gate 均通过，说明 256 个 token 在当前
spatial slot 定义下不再只是 16 个 transition code 的组内复制。

matched FSQ 对照只将量化器改为 `(4,4,8)=128`，其余保持一致。审计时发现旧 continuous evaluator 对
FSQ 直接绕过了 `project_in/project_out`，造成错误口径；现增加 `project_continuous()`，EMA 为 identity，
FSQ 保留 learned projection/tanh、只取消 rounding。修正后的 FSQ continuous/quantized RMSE 为
`0.08109758/0.08134974`，quantization gap `0.00025216`，相对 no-motion 的 quantized 改善 `7.85%`；
unique ratio/pairwise change 为 `0.90544/0.98645`。FSQ gap 比 EMA 小约 `90.8%`，但最终 quantized RMSE
仍比 EMA 差 `1.28%`，且 EMA continuous 表征更强。因此 EMA-128 保持主线，FSQ-128 作为 matched 对照，
不以更小 gap 代替最终重建质量判断。

基于通过双 gate 的 spatial EMA checkpoint，已生成 click-alarmclock 的严格 key-aligned cache：4252 个
训练样本，每样本 `4 predicted action IDs + 256 oracle physics IDs`；另有固定 seed=42 的 shuffled physics
control。第一层 BiVidGen-style action-DiT adapter-only 接线、梯度与 trainable-only checkpoint 保存均正常。
单卡 batch 8、20 steps 的 380-sample paired evaluation 中，启用 physics 相对同 checkpoint hard-disabled
control 的 action MAE/RMSE、endpoint joint MAE、wrapped joint MAE 分别变化
`-0.645%/-0.425%/-0.491%/-0.532%`，gripper accuracy 下降 `0.214` 个百分点，仅 velocity MAE 改善
`0.154%`；gate 失败。由于该 run 总样本暴露只有目标八卡方案的 1/8，没有据此放大结论。

随后在 GPU3 被外部任务占用时，使用 GPU0/1/2/4/5/6/7 做了 near-matched 重复：7 ranks、per-device
batch 9、global batch 63、20 steps，共 1260 个样本，接近目标八卡方案的 1280。checkpoint SHA256 为
`2c949010ab5ed1893f26dc6931be5fc0823abf28c37acc9b1e60351f5878f94c`。同一 380-sample panel 上，启用
physics 的 action MAE/RMSE、endpoint、wrapped joint 分别为
`-0.649%/-0.374%/-0.569%/-0.584%`，gripper accuracy 下降 `0.177` 个百分点，velocity MAE 变化
`-0.022%`。action MAE 在 5 个 validation episodes 上为 `0/5` 胜，paired episode-cluster bootstrap 95%
区间为 `[-0.946%, -0.351%]`；RMSE、endpoint、wrapped joint 和 gripper 的区间也均落在负方向。结果位于
`artifacts/evaluations/qwenpi_physical_tokens_click_alarmclock_b3_d0_spatial_st256_7gpu_dit_first_adapteronly_seed42_steps20_380samples/comparison.json`。

因此当前证据支持“256-token tokenizer 有真实结构和 future reconstruction 信号”，但不支持“当前随机
初始化的 first-block policy adapter 能在 20 steps 利用该信号”。按预注册规则停止 500-step 扩展，不运行
early-4/all。下一轮低成本诊断优先检查：随机 code embedding 是否丢失 EMA codebook 几何、zero-init output
branch 的有效学习幅度、physics dropout=0.2，以及 aligned-vs-shuffled 条件训练是否可区分；每次只改变一个
因素并继续使用 hard-disabled paired control。只有短程结果先消除稳定负向，才允许更长训练。

D1 watcher 在首次满足八卡空闲后暴露 runner 缺陷：`--exclude-episode-ids=5,20,25,39,49` 误用了要求
offsets 必须从 0 开始的 parser，导出在加载模型前即退出，没有产生 partial cache。现已拆分独立的非负、
唯一 episode-ID parser，真实参数 smoke 通过；没有安装额外测试依赖。修复后已重新启动唯一 watcher，仍
要求八卡显存均不超过 256 MiB、无 compute app 且连续三次满足才运行 D1。

### 21.19 D1 数据量诊断与 tokenizer-code geometry 接口（2026-09-27）

修复 runner 后，D1 八任务 cache 已完成并通过严格 manifest 校验：8 tasks、64 episodes、14,077 samples
与 unique frames、8,785 train-complete clips、3,244 validation-complete clips，逐任务最多 300 后训练集为
2,394 clips。D1 global-QFormer 使用八卡、per-device batch 4、global batch 32、EMA-128、1000 steps，约
45 秒完成；checkpoint SHA256 为
`081b35ae0b7f6f623175308d9c5ec66c45e6b9a47b0150a74347916ee676489d`。

在完全相同的 D1 validation cache 上，D1 自身 aggregate continuous/quantized 相对 no-motion 只改善
`2.343%/1.632%`，未过 `5%/3%` 绝对门槛。相对 D0 global 的 task-macro continuous RMSE 反而退化
`0.112%`，quantized RMSE 只改善 `0.209%`；continuous 仅 5/8 tasks 获胜，quantized 为 7/8。
representation gate 因此失败。D1 的量化 gap task-macro 改善约 `30.3%`，但 continuous 表征无增益，
不能将更小 gap 当成成功。

D1 也完整复现组内 collapse：global active codes `128/128`、perplexity `118.40`，但
within-transition unique ratio `0.06550`、pairwise token change `0.01463`；continuous pairwise cosine
`0.99985`、relative dispersion `0.01199`、per-sample unique ratio `0.12131`。这直接回答了“是否只是
tokenizer 数据太少”：从单任务增加到八任务并提高全局码本覆盖，仍不能使无 slot 约束的 queries 分工；
主要问题是 routing/训练目标的不可辨识性，不是单纯数据量。D2 spatial-aligned 已排队，保持同 cache、split、
batch、steps、EMA 和宽度，只改变 routing。

对 PhiZero decoder 源码的进一步核对发现，motion ID 不会直接进入随机 ID embedding：
`FiniteScalarCodeDecoder` 先按 mixed-radix basis 将 ID 还原为 FSQ scalar code values，再经过训练得到的
output projection、post-quant projection、context projection 和 context projector 后送入 DiT。也就是说，
下游显式保留 tokenizer 的 code geometry。当前 action-DiT adapter 的 physics `nn.Embedding(128,512)`
完全随机初始化，20-step policy smoke 必须从 ID 共现重新学习 code 之间的关系，与 PhiZero 的接口并不
matched；这成为 near-matched smoke 稳定负向后的首要单变量诊断。

直接量化该 mismatch：7-GPU 20-step checkpoint 中 policy physics embedding 与 EMA tokenizer codebook
的 8,128 个 pairwise distances，Pearson/Spearman 相关为 `-0.00026/-0.00191`，每个 code 的 top-5 最近
邻居平均重合仅 `5.47%`，接近随机。policy ID embedding 的平均 row norm 为 `1.00`，而独立 position
embedding 为 `0.45`，所以短程训练时 position 分量相对 code identity 也过强。这些结果说明当前接口没有
保留任何 tokenizer code geometry，不能把负向结果简单归因于 physics tokens 本身无信息。

现已增加可选的 EMA-codebook 初始化路径：从 tokenizer checkpoint 的 `model.quantizer.codebook`
读取 `128 x 128` code vectors，将每个 vector 重复四次并除以 `sqrt(4)` 扩展到 DiT 的 512 维。该映射严格
保持所有 code pair 的欧氏距离和内积，不增加参数或改变 adapter 架构，之后 embedding 仍可训练。入口要求
显式 tokenizer checkpoint SHA256，哈希、key、vocab rows、finite values 或维度整除任一不符均 fail closed；
旧随机初始化路径保持默认，已有 checkpoint 可继续加载。相关 adapter/严格 checkpoint 初始化定向测试为
`17 passed`，shell syntax、Python compile 和真实 checkpoint smoke 通过。下一次 policy 训练只改变这一项，
仍固定 first block、dropout 0.2、global batch 约 64、20 steps 和同一 paired evaluation。

### 21.20 D2 joint gate 与 codebook-init policy 反证（2026-09-27）

D2 已在同一八任务 D1 cache、同一 split/cap、global batch 32、EMA-128、1000 steps 下完成，唯一模型差异
是 `token_routing=spatial_aligned`。checkpoint SHA256 为
`24403f39af17178a9285f678883d1fae814c0003dcb1b9fc515ddcb9e63cef17`。正式 aggregate
continuous/quantized RMSE 为 `0.07865454/0.08034276`，相对 no-motion 改善 `9.880%/7.946%`；task-macro
改善为 `9.640%/7.793%`。相对 D1 global，task-macro continuous/quantized RMSE 改善 `7.271%/6.104%`，
两项均为 8/8 tasks 改善。

D2 结构指标为：128/128 active codes、perplexity `123.72`、within-transition unique ratio `0.89676`、
pairwise token change `0.98476`、continuous pairwise cosine `0.11065`、relative dispersion `0.91304`、
per-sample unique ratio `0.81066`。D2 joint gate 的 8 个 checks 全部通过。这一 paired 结果确认：八任务数据
本身不能修复 global query collapse，而明确 spatial slot routing 能同时恢复 256-token 结构并显著改善
future reconstruction；D2 spatial-aligned EMA-128 因此取代 D1 global 成为后续多任务 tokenizer 主线。

随后完成严格单变量的 D0 codebook-init policy smoke：仍使用同一 D0 spatial token cache、first block、
physics dropout 0.2、20 steps 和 seed 42，只将随机 ID embedding 改为 SHA-verified EMA codebook 初始化；
八卡 per-device batch 8，global batch 64，共 1280 个样本。初始化日志确认读取正确的 `128 x 128`
codebook，adapter trainable 参数仍为 1.513M。checkpoint SHA256 为
`18cc22ee3873b489bbdc25a0e36d91748031eb95622036a7cdb0068eb0d1e347`。

380-sample paired evaluation 中，physics-enabled 相对 hard-disabled 的 action MAE/RMSE、endpoint、wrapped
joint 分别为 `-1.321%/-0.866%/-0.784%/-1.054%`，gripper accuracy 下降 `0.502` 个百分点，velocity MAE
为 `-0.115%`；全部 gate checks 失败。action MAE bootstrap 95% CI 为
`[-2.207%, -0.316%]`。因此保留 code geometry 虽然修正了与 PhiZero decoder 接口的一处差异，但它不是
当前 policy adapter 获益的充分条件，不能据此进入 500 steps。下一项使用相同随机 embedding 与 global
batch 64，只把 aligned physics cache 换成固定 seed 的 shuffled cache；比较两者负向幅度以判断 adapter
是在利用 token 内容，还是只学到对冻结动作头的通用扰动。

### 21.21 Shuffled 因果对照与下一轮低学习率诊断（2026-09-27）

固定 seed 的 episode-internal shuffled-physics 对照已按预注册设置完成：D0 spatial EMA cache、随机 ID
embedding、first DiT block、physics dropout 0.2、20 steps、八卡 global batch 64，除 physics token 与样本的
对应关系外均与 aligned run 一致。checkpoint SHA256 为
`4eb24f13617773134886136dfa375c2c17fcf1040170c4b09f44fa88292f7a82`。

同一 380-sample panel、同 checkpoint hard-disabled control 下，shuffled physics 使 action MAE/RMSE、
endpoint joint MAE、wrapped joint MAE 分别变化 `-0.728%/-0.500%/-0.493%/-0.612%`，gripper accuracy
下降 `0.243` 个百分点，只有 velocity MAE 改善 `0.334%`；action MAE 为 `1/5` episode 获胜，paired
episode-cluster bootstrap 95% CI 为 `[-1.331%, +0.016%]`，joint gate 失败。报告位于
`artifacts/evaluations/qwenpi_physical_tokens_click_alarmclock_b3_d0_spatial_st256_shuffled_8gpu_dit_first_adapteronly_seed42_steps20_380samples/comparison.json`。

早期单卡 aligned run 的 action MAE 为 `-0.645%`，但其样本暴露量不匹配；使用 7-GPU、global-batch-63
near-matched aligned 重复作为主参照时为 `-0.649%`，而八卡 shuffled 为 `-0.728%`，aligned 只领先
`0.079` 个百分点。这个差值
远小于 episode-cluster 不确定性，且二者主要指标都为负，因此不能声称 adapter 在利用 token 语义。当前
最符合证据的解释是：20-step、`1e-4` LR 的 zero-init adapter 对冻结动作模型产生了内容无关的通用扰动，
真实 aligned 信号即便存在，也被该扰动淹没。codebook-init 的更差结果也排除了“只恢复 code geometry
即可解决”的假设。

下一项严格只改变 adapter 参数组 LR：从 `1e-4` 降到 `2.5e-5`，其余保持 aligned 八卡 20-step 设置。
入口为 `scripts/physical_tokens/run_dit_physics_adapter_lr_diagnostic.sh`；底层 runner 新增显式
`ACTION_MODEL_LEARNING_RATE` 覆盖，避免错误调整对 DiT adapter 无效的 prefix `physics_gate_init`。该实验
仅是扰动幅度诊断：即使落入 `0.5%` non-inferiority tolerance，也不能当成有效收益。只有 action MAE 点估计
非负、至少 3/5 episode 获胜，且 aligned 明显优于随后同设置的 shuffled control，才允许延长到 50 steps；
否则停止该融合形式，转向加入 token-usefulness/aligned-vs-shuffled 辅助目标。500-step 扩展继续禁止。

本地登录节点 node001 已被其他用户八卡任务占用，每卡约 42 GB；没有终止或抢占外部进程。随后通过
SSH 独立检查用户指定的 node004，确认八卡均为 1 MiB、0% utilization 且无 compute process，因此 LR
诊断可在 node004 安全启动。

低 LR 八卡 run 随后完成，checkpoint SHA256 为
`95460ede32fb29c109a076690918a31af07e8cf43333d181590128c7eabf49b5`。380-sample paired evaluation 的
action MAE/RMSE、endpoint、wrapped joint 分别为 `-0.063%/-0.046%/-0.040%/-0.044%`，gripper accuracy
下降 `0.033` 个百分点，velocity MAE 改善 `0.029%`。宽松的 0.5% non-inferiority gate 全部通过，但
action MAE 只有 `2/5` episode 获胜，bootstrap 95% CI 为 `[-0.117%, -0.007%]`，仍严格落在负方向。
因此降低 LR 只把内容无关扰动缩小到接近零，没有产生正收益；未达到预注册的非负点估计和 3/5 wins，
禁止 50-step 与 500-step 扩展。

准备 D2 policy cache 时审计出一个关键导出缺陷：
`export_spatiotemporal_physics_token_ids.py::build_model()` 重建 checkpoint 时漏传 `token_routing`，导致
`spatial_aligned` checkpoint 在 token export 阶段静默回落到默认 `global_qformer`。tokenizer 自身的 D2
evaluation/structure 脚本使用完整 `model_config`，所以 D2 tokenizer 的双 gate 结论不受影响；但此前 D0
所谓 spatial policy cache 及其 aligned/shuffled/codebook/LR policy 实验均基于错误 routing 的导出 token，
不能作为 spatial-aligned policy 价值的最终反证，只能保留为错误 global-export 路径的诊断。

现已修复导出器，显式从 checkpoint 恢复 `token_routing`，并把 routing 写入 cache manifest；新增回归测试
要求 rebuilt model 保留 `spatial_aligned` 及其 `4 x 4` alignment mask。下一步先用 D2 主线 checkpoint
重新导出 click-alarmclock aligned/shuffled cache，再从最小 20-step policy smoke 重启证据链。旧 cache
不得继续用于新结论。

修复后的真实 checkpoint smoke 确认 rebuilt tokenizer 为 `spatial_aligned`、alignment mask 为 `16 x 16`、
输出 256 tokens。由于当前 vlact 环境未安装 pytest，未改动环境依赖；Python compile、shell syntax 和等价
checkpoint load/assert smoke 均通过。

D2 checkpoint 随后重新导出 click-alarmclock 全量 4,252 样本，manifest 明确记录
`token_routing=spatial_aligned`。aligned cache SHA256 为
`eceb56e374a65f65acdd9f222195b4b967d76ed8cd6c8e4d6ff9ac89433ee43d`，shuffled cache SHA256 为
`95130732a14999a3f38ad75fe2ac0397f81722897849ced241b7d589f23115a0`。新旧 policy cache 有
`99.424%` token IDs 不同；每个 transition 的 16 个 token 平均 unique codes 从旧 cache 的 `1.222`
恢复到 `12.288`，per-sample unique ratio 从 `0.0558` 提升到 `0.3603`，定量确认旧导出确实重新引入了
近似组内 collapse。

用修复后的 D2 aligned cache、D2 EMA codebook 初始化、first-block adapter、dropout 0.2、八卡 global
batch 64 重新训练 20 steps。checkpoint SHA256 为
`2f18efa5f0c057447cac0944603a0f55e76b1f2387eee63591e36248a4bbca7f`。paired evaluation 的 action
MAE/RMSE、endpoint、wrapped joint 分别变化 `-0.549%/-0.396%/-0.304%/-0.416%`，gripper accuracy
下降 `0.222` 个百分点，velocity MAE 改善 `0.214%`；action MAE `2/5` wins，95% CI
`[-1.017%, -0.035%]`，gate 失败。

matched D2 episode-shuffled run 的 checkpoint SHA256 为
`0d0fc43da7b452e30bbb7c144c2e34e485957dafcb743219650aeac25a8fa944`。其 action MAE/RMSE、endpoint、
wrapped joint 分别为 `-0.690%/-0.485%/-0.396%/-0.565%`，gripper accuracy 下降 `0.259` 个百分点，
velocity MAE 改善 `0.516%`；action MAE `1/5` wins。aligned 比 shuffled 好 `0.140` 个百分点，方向正确
但远小于不确定性，且两者对 hard-disabled control 都为负。当前证据变为：正确 spatial token 内容有微弱
可利用方向性，但普通 flow-matching loss 更容易让 adapter 学到内容无关扰动。下一单变量应是 paired
aligned-vs-shuffled usefulness/ranking objective；在该目标完成梯度与成本 smoke 前，仍禁止 50/500 steps。

usefulness objective 已按最小侵入方式接入 action-DiT，并默认关闭以保证旧配置完全兼容。训练时对同一
noisy trajectory、同一 diffusion timestep、同一 VLM prefix 和同一个 physics-dropout 决策分别计算 aligned
与 batch-internal cyclic-shuffled physics 前向，目标为
`L = L_aligned + lambda * relu(margin + L_aligned - L_shuffled)`。shuffle 在每个原始 local batch 内进行，
不会跨越 repeated-diffusion group；因此不会用不同 diffusion noise 制造伪差异。当前只训练 adapter，负项
不能破坏冻结动作头。

CPU 小模型端到端 smoke 中，zero-init 时 aligned/shuffled loss 相等、hinge 精确约为 `0.01`；反向后只有
physics adapter 获得 finite non-zero gradient，冻结 action base 无梯度。代码编译、runner shell syntax、
grouped shuffle 与强制共享 dropout smoke 均通过。预注册首轮固定 `lambda=0.5`、`margin=0.01`、20 steps、
D2 corrected cache、D2 codebook-init、first block、八卡 global batch 64；入口为
`scripts/physical_tokens/run_d2_physics_usefulness_smoke.sh`。晋级要求仍是 paired action MAE 点估计非负、
至少 3/5 episode wins，并在相同训练目标下优于 shuffled-cache control；否则不延长训练。

首轮 usefulness smoke 已完成，checkpoint SHA256 为
`3eb4b4a0a3a8dec05f7f7914c6fee333c3d24aa451e3b41376ad496c123a2f14`。训练日志确认 paired 分支执行且
aligned/shuffled/usefulness 三项 loss 均被记录。380-sample paired evaluation 的 action MAE/RMSE、endpoint、
wrapped joint 分别为 `-0.287%/-0.183%/-0.199%/-0.240%`，gripper accuracy 下降 `0.095` 个百分点，
velocity MAE 改善 `0.176%`。宽松 non-inferiority gate 通过，但 action MAE 仅 `1/5` wins，95% CI
`[-0.528%, -0.066%]`，没有达到预注册晋级门槛。相对无 usefulness 的 D2 aligned run，负向幅度从
`-0.549%` 约减半，说明辅助目标方向有作用，但不足以产生收益；不运行 50-step 或 matched shuffled run。

审计该目标后发现当前 hinge 作用于 batch-reduced scalar flow loss，样本间的 aligned/shuffled 差可相互抵消，
弱化了每个样本必须依赖其自身 physics condition 的约束。下一项只把 usefulness 项改成 per-sample velocity
MSE ranking 后再对 batch 求均值，主 flow loss、lambda、margin、cache、初始化和训练预算均不变。若仍不能
使 action MAE 非负并达到 3/5 wins，则停止 ranking 调参，进入 256-token 分层聚合/FiLM 结构实验。

per-sample reduction 已实现并复用统一 mask 语义；异构 valid-step/action-dim mask 的数值 smoke、Python
compile、端到端 adapter-only gradient smoke 均通过。新 run tag 使用 `usefulpsw05`，避免错误复用上一版
batch-scalar checkpoint。

per-sample usefulness run 的 checkpoint SHA256 为
`ce71d5809083742468c1ca13a45313c639885c1d1af907dc2e6458db49abd800`。其 action MAE/RMSE、endpoint、
wrapped joint 为 `-0.339%/-0.214%/-0.225%/-0.286%`，gripper accuracy 下降 `0.099` 个百分点，velocity
MAE 改善 `0.152%`；action MAE `1/5` wins，95% CI `[-0.576%, -0.110%]`。它比 batch-scalar ranking
略差，也远未达到晋级门槛。因此 usefulness/ranking 路线正式停止，不再搜索 lambda、margin 或训练步数。

下一项先于 16-transition 聚合测试更保守的 query-scope 修复：保留全部 256 physics tokens 和相同
cross-attention，只把 adapter residual 直接施加到 DiT 序列末尾的 32 个 action queries，而不直接修改
state/future-token prefix。后续 self-attention 仍允许条件信息传播，因此这是限制直接扰动范围，而不是切断
交互。训练和 evaluator 均新增默认关闭的 `dit_action_tokens_only`，paired hard-disable 继续严格有效；测试在
last-block 注入下确认 prefix 逐元素不变、action suffix 改变。预注册入口为
`scripts/physical_tokens/run_d2_action_only_adapter_smoke.sh`，仍固定 D2 corrected cache、codebook-init、first
block、20 steps、global batch 64，且关闭 usefulness loss。只有点估计非负和至少 3/5 wins 才晋级。

### 21.22 Action-only 与 transition-level pooling 结果（2026-09-27）

action-only 八卡 smoke 已按上述预注册设置完成，checkpoint SHA256 为
`ea4d08ed14fb15277da3e402870e5ec0e1b31998d25c1cf65889510cdaca1ec6`。同一 380-sample paired panel
上，physics-enabled 相对 hard-disabled 的 action MAE/RMSE、endpoint、wrapped joint 分别变化
`-0.554%/-0.401%/-0.315%/-0.424%`，gripper accuracy 下降 `0.214` 个百分点，velocity MAE 改善
`0.203%`；action MAE 为 `2/5` episode wins，95% CI 为 `[-1.029%, -0.031%]`。这与全序列 D2
cross-attention 的 `-0.549%` 几乎相同，说明负向结果不是由 adapter 直接扰动 VLM/state prefix 导致；该分支
未达到晋级门槛，不运行 50/500 steps。

随后实现了保持 tokenizer/VLM 输出不变的分层读取：输入仍是完整 256 个 physics IDs，按原始布局 reshape
为 `16 transitions x 16 spatial slots`，在 action head 内沿 spatial 维求均值，使 first-block
cross-attention memory 从 256 slots 变为 16 temporal slots。配置为
`dit_physics_memory_pooling=transition_mean` 与 `physics_tokens_per_transition=16`，默认 `none` 保持旧路径；
256→16 shape、none→256 shape、hard-disabled identity 和非整 transition fail-closed smoke 均通过。固定实验
入口为 `scripts/physical_tokens/run_d2_transition_pooling_smoke.sh`，其余设置仍为 corrected D2 cache、D2 EMA
codebook-init、first block、dropout 0.2、20 steps、八卡 global batch 64、usefulness off。

transition-mean checkpoint SHA256 为
`3d148ecc147a897113ed081d4917397102342b8e90a3d579c9d1de8bce6032f0`。paired evaluation 中 action
MAE/RMSE、endpoint、wrapped joint 分别变化 `-0.930%/-0.645%/-0.467%/-0.728%`，gripper accuracy
下降 `0.370` 个百分点；只有 velocity MAE 改善 `0.266%`。action MAE 仅 `1/5` episode wins，95% CI
为 `[-1.618%, -0.144%]`，gate 失败，并且明显差于保留 256-slot memory 的 `-0.549%`。因此简单 spatial
mean 不但没有解决条件利用问题，还丢失了 D2 tokenizer 已验证有用的 spatial variation；该分支同样停止，
不延长训练。

这轮结果也澄清了“16 transition 分组聚合”的含义：它不是把物理 token 数从 256 改成 16，也不改变 VLM
应输出的 256 个离散 physics tokens，而只是 action head 对完整 token 的一种层次化读取方式。失败原因更可能
是 mean pooling 丢空间结构后，16 个 temporal slots 与 32 个 action queries 之间仍由无约束 cross-attention
自行学习对应关系。下一轮只测试显式 temporal alignment 的 FiLM：每个 transition 条件确定性对应连续两个
action queries，zero-init modulation 仅作用于 action suffix；若这一更强归纳偏置仍不能达到 action MAE 非负
和至少 `3/5` wins，则停止当前 post-hoc adapter 路线，转向在联合训练阶段让 physics token prediction 与
continuous action head 共同塑造 VLM hidden states。

### 21.23 显式 temporal-aligned FiLM 结果与 post-hoc adapter 停止条件（2026-09-27）

已实现 `dit_physics_adapter_type=temporal_film`：仍先把完整 256 IDs 编码并按 16 个 spatial slots 聚合为
16 个 transition memories，然后确定性 `repeat_interleave(2)` 对齐 32 个 action queries。每层由一个
zero-initialized linear 产生 scale/shift，并对归一化后的 action hidden states 做 residual FiLM；初始输出与
禁用分支逐元素相同。该模式 fail closed 地要求 `dit_action_tokens_only=true` 和
`dit_physics_memory_pooling=transition_mean`，且 action-query 数必须能被 transition 数整除。相关 Python
compile、shell syntax、32-query 对齐、zero-init identity、非对齐拒绝和 finite gradient smoke 全部通过；
预注册入口为 `scripts/physical_tokens/run_d2_temporal_film_smoke.sh`。

八卡 global batch 64、20-step run 仅有 0.724M trainable parameters，checkpoint SHA256 为
`aa9e10be6a2ba1c5d7b05ba9fc0969b84b4faab5367f2fc9171dcb2dce1fd8cc`。最终保存 config 已核验为
`temporal_film + transition_mean + action_only=true + usefulness=0`。380-sample paired evaluation 中，
action MAE/RMSE、endpoint、wrapped joint 分别变化 `-0.702%/-0.393%/-0.997%/-0.578%`，gripper
accuracy 下降 `0.234` 个百分点，velocity MAE 也变差 `-0.747%`。action MAE 为 `0/5` episode wins，
95% CI 为 `[-1.209%, -0.247%]`，gate 失败。

因此三种结构假设都已得到 matched 反证：全序列 256-slot cross-attention 为 `-0.549%`，仅 action-query
cross-attention 为 `-0.554%`，transition-mean cross-attention 为 `-0.930%`，显式 temporal FiLM 为
`-0.702%`；ranking objective 也未转正。继续搜索 adapter 层数、FiLM 位置、lambda、margin 或延长到
50/500 steps 已不符合当前证据，post-hoc frozen-base adapter 路线正式停止。

下一步先做 action relevance probe，而不是直接扩大训练：在 train/validation episode 严格分离下，用相同
轻量 probe 比较 `state/action-token` 与 `state/action-token + physics-token` 对连续 action trajectory 的预测，
并加入 episode-shuffled physics control。若 aligned physics 明显优于两者，说明 token 有条件增量信息，问题
主要在 policy 联合优化与 hidden-state 接口；届时进入 VLM 同时预测 action/physics tokens、action DiT 读取
对应 VLM hidden states 的联合训练阶段。若 aligned 不优于 shuffled，则需给 tokenizer 增加 action/contact/
dynamics-aligned objective，而不是继续要求 action head 从纯 future-visual reconstruction codes 中自行发现动作
语义。

### 21.24 Physics-token action relevance probe（2026-09-27）

新增 `scripts/physical_tokens/probe_physics_action_relevance.py`，直接从 RoboTwin parquet 按 cache 的
`(episode_id, base_index)` 重建当前 14-D state 与未来 32x14 actions；joint target wrap 到 `[-pi, pi)`，
gripper 按训练口径二值化，episode 尾部 padding 只训练但不计 validation metric。官方 5 个 validation episodes
完全隔离；剩余 45 个训练 episodes 再固定拆成 40 fit + 5 calibration，用 calibration 选择 ridge alpha，最后
在全部 45 个 episodes 重训并评估 439 个 held-out samples。

第一版直接把 2048-D physics 特征与 526-D state/action-token 特征联合 ridge，因额外维度改变 baseline 系数，
aligned/shuffled 都退化约 37%，不具备可解释性，未作为结论。正式口径改为 conditional residual ridge：先固定
state + 4 个 position-wise action-ID one-hot baseline，再只允许 physics features 预测 baseline residual；alpha
扩展到 `1e-1 ... 1e6`，因此强正则可以实际退化回同一个 baseline，不会被迫使用 physics。

transition histogram 表示下，baseline validation action MAE 为 `0.1023072`；aligned physics 为
`0.1020567`，相对改善 `+0.245%`；episode-shuffled physics 为 `0.1021064`，改善 `+0.196%`。aligned 相对
shuffled 的净优势只有 `+0.049%`。因为 histogram 会擦除 D2 的 spatial slot 身份，又补做同为 2048-D 的
position-aware CountSketch：它对每个 `(slot, code ID)` 独立 signed hash，保留 spatial/temporal 位置且不增加
模型容量。该口径下 aligned action MAE 为 `0.1028913`，相对 baseline 退化 `-0.571%`；shuffled 为
`0.1031399`，退化 `-0.814%`；aligned 相对 shuffled 好 `0.241%`，但仍没有对 baseline 的正增益。

两种表示一致显示 aligned 略优于 shuffled，说明 corrected D2 IDs 含有微弱的样本对应信号；但该差值仅
`0.05%-0.24%`，且 position-aware 表示不能在 held-out episodes 上提供增量 action 信息。结合 oracle-token
adapter 的多轮负结果，当前主要瓶颈不应再归因于 adapter 形状，而是 tokenizer 只优化 future visual
reconstruction，所得 code 对 action/control 的条件充分性不足。下一阶段保留 256 physics tokens 与 EMA-VQ
主线，但应给 tokenizer 增加 action-aligned auxiliary objective（优先预测 future joint delta/velocity 和
gripper transition，后续再考虑 contact proxy），并要求新 tokenizer 在相同 residual probe 上同时满足：
aligned 对 baseline action MAE 至少改善 `1%`，且 aligned 对 shuffled 至少改善 `1%`。未过这一 representation
gate 前，不再启动 action-DiT 训练。

### 21.25 D3 action-aligned tokenizer 与最终 representation gate（2026-09-27）

实现了保持 256 physics tokens、spatial-aligned routing 和 EMA-VQ 不变的 action auxiliary head：每个
transition 的 16 个量化 spatial latents 求均值，非线性 head 预测 12-D wrapped future joint delta 与 2-D
gripper，附加 joint Smooth-L1、temporal velocity 和 gripper BCE。配置新增 `action_aux_dim`、
`action_aux_weight`、`action_velocity_weight`、`action_gripper_weight`；默认值保持旧 checkpoint 严格兼容。
训练数据另生成与 8-task visual cache 按 task/episode/base/offset 严格对齐的 17x14 action sidecar，SHA256 为
`0aee32e15705cc890abc0e9e9886d7fd4cc90af8702cbf104be83fd2add98c29`。静态 compile、旧路径逐元素兼容、
sidecar key/shape fail-closed、辅助输出、编码层 finite gradient 和 strict checkpoint rebuild smoke 全部通过。

D3 固定 8 tasks、每任务最多 300 train clips、后 2 episodes/task validation、1000 steps、global batch 32，
loss weights 为 action `0.25`、velocity `0.25`、gripper `0.1`。八卡训练约 42 秒；最终 checkpoint SHA256
为 `fa92ac049f1625924d1548f46ef4e77086aaf3641f59b53df7090935a88209c2`。码本 128/128 全激活，
perplexity `124.44`。相对 D2，macro continuous RMSE 退化 `0.258%`，quantized RMSE 退化 `0.017%`，
quantization gap 改善 `11.98%`；8/8 tasks 均在预注册 `-1%` task floor 内。token structure 的
within-transition unique ratio 为 `0.8976`、pairwise change 为 `0.9850`，因此 visual/structure gate 通过。

但最终 action-relevance gate 失败。transition histogram 下 baseline/aligned/shuffled action MAE 分别为
`0.1023072/0.1038333/0.1041805`：aligned 对 baseline 为 `-1.492%`，对 shuffled 为 `+0.333%`。
position-aware CountSketch 下 aligned 为 `0.1042662`、shuffled 为 `0.1020172`：aligned 对 baseline
`-1.915%`，对 shuffled `-2.205%`。最终结果已写入 `joint_gate_final.json`，`pending=[]`、`passed=false`。
oracle/aligned/shuffled D3 cache SHA256 分别为
`65759265f482e10320397eb0d2bfd07b88bbce6c1703679f06cc2040e9b88a56`、
`9a46c7d1b9fc1f518057dc32403235d337ad3f0c7f713514fbcd48d1ced619d4`、
`826262eb96e1e7f7cc841047366a1411a6a30b884a2f52a1691bfd6257bf52a8`。按门禁不启动 policy 训练。

### 21.26 D4 数据覆盖审计、matched control 与 residual-target 决策（2026-09-27）

数据审计发现 D3 每任务 300 clips cap 并未截断 click_alarmclock：其 6 个训练 episodes 只有 294 个 complete
clips；真正限制是 episode 覆盖。现有单任务 spatial feature cache 已覆盖全部 50 episodes，并与官方 split
严格对应为 45 train + `5,20,25,39,49` validation。为区分数据量与辅助目标，新增
`run_d4_click_full_action_alignment.sh`，在同一全量 cache、相同初始化规则和 1000-step budget 下训练
reconstruction-only control 与 action weight `1.0` candidate。单任务 sidecar exporter 同时补齐了无
`task_ids/task_names` cache 的兼容路径，仍执行严格 key alignment。

D4 control/candidate checkpoint SHA256 分别为
`c8589def6744b8c7e81bfc6201482f3636a1d379752c9d95f83114364ee4894c` 与
`2ac004bf346e794a7fa7c08d93129b75950d495696f1f0367509d1664dc034fb`。candidate 相对 matched control
的 continuous/quantized RMSE 分别退化 `0.842%/0.987%`，quantization gap 退化 `4.87%`；held-out
action auxiliary loss 在约 step 100 最低，随后升高，显示绝对动作目标发生过拟合。action relevance 也未过：
control 的 histogram/positional baseline gain 为 `-1.509%/+0.082%`，candidate 为
`-0.972%/+0.167%`；四项 aligned-vs-shuffled 均为负，candidate 分别为 `-0.597%/-0.825%`。

为排除线性 probe 漏读，新增 frozen auxiliary decoder correspondence 诊断。5 个 held-out episodes 上，aligned
相对 episode-shuffled IDs 的 joint wrapped MAE 改善 `8.30%`、joint Smooth-L1 改善 `14.36%`、gripper
BCE 改善 `24.41%`，gripper accuracy 增加 `5.58 pp`，而 velocity 退化 `0.10%`。这证明离散 IDs 中确有
非线性可读的绝对动作信息；但把 frozen decoder 的 224-D 输出作为 residual-ridge physics feature 后，aligned
对 state+4 action-token baseline 仍退化 `1.638%`，且比 shuffled 差 `1.872%`。因此失败本质是新增信息与
现有 action tokens 高度冗余，而非单纯 probe 容量不足。

D5 预注册方向据此改为 complementary residual supervision：用现有 observation-predicted 4 action IDs 和固定
EMA action tokenizer 解码 32x14 trajectory，构造 `true action - decoded action-token trajectory` 的 wrapped
joint/gripper residual sidecar。physics tokenizer 只辅助预测 action tokens 未解释的连续残差，视觉重建、
256-token spatial EMA-VQ、45/5 split、1000 steps 和 matched control 均保持不变。D5 仍必须同时满足视觉
non-regression、aligned 对 baseline action MAE至少 `+1%`、aligned 对 shuffled 至少 `+1%`，否则不进入
action-DiT 训练；不再继续搜索 raw absolute-action loss weight。

### 21.27 D5 action-token residual supervision 结果与 shared-codebook 停止条件（2026-09-27）

新增 `export_action_token_residual_sidecar.py`，严格按 `(episode_id, base_index)` 对齐 observation-predicted
4-token cache，并用固定 action tokenizer
`artifacts/action_tokenizer_low_bitrate/ema_4x128_steps5000/step-00005000.pt` 解码 32x14 trajectory。
物理帧 offset 为 `2,4,...,32`，对应 decoder indices `2,4,...,30,31`；joint residual wrap 到
`[-pi,pi)`，gripper residual 为 binary target 减 decoded probability。sidecar shape 为 `[4252,17,14]`，
数值 finite，joint/gripper 范围分别为 `[-3.141482,3.141578]` 和 `[-1,1]`，SHA256 为
`d71a35ec4edda3f509dc73b328b70023634ee2553c8592cc6236dd7d0cdb19c8`。

tokenizer 新增默认 `absolute` 的 `action_aux_target_mode`；`action_token_residual` 模式直接回归 joint/gripper
residual 与 residual velocity。旧 checkpoint 重建路径通过 default 保持兼容；residual shape、finite loss、
编码器非零 gradient、Python compile 和 shell syntax smoke 全部通过。固定 D5 candidate 仍为 256 spatial
EMA tokens、codebook 128、45/5 split、1000 steps、action residual weight `1.0`，复用 D4 matched control。

D5 checkpoint SHA256 为
`3c16f4575ed73943b9a9a63d2d1ee7c8e3a0f6e04e602233b7616defe8501eff`。action residual validation loss
在 step 100 为 `0.1984`，之后上升，step 1000 为 `0.2226`；相对 control 的 continuous/quantized RMSE
分别退化 `1.883%/1.824%`，视觉 non-regression 失败。码本仍为 128/128 active，perplexity `121.45`，
故失败不是 code collapse。oracle/aligned/shuffled cache SHA256 分别为
`b1a6352c32351223757596a9c85fffb3f39b19a47d6edaa3761492940dbf3da8`、
`d04a5837292bce0513132e1b334ff76bced83964ce6ab328fe5cc2b1740f71f4`、
`42b67d3561e3e0d98a86fe1e24fad3e1b2f084bf7a7bef29595b44498d2f986f`。

action relevance 同样失败：transition histogram 的 aligned 对 baseline/aligned 对 shuffled 为
`-2.101%/-2.213%`；position-aware CountSketch 为 `-3.024%/-3.534%`；用冻结 residual auxiliary decoder
输出作为非线性 feature 仍为 `-0.692%/-0.542%`。因此 complementary target 的概念合理，但当前实现仍要求
同一组 256 visual codes 同时承担重建与 residual，梯度竞争只造成视觉退化，没有形成稳定的增量离散信息。

停止条件：不再搜索 absolute/residual loss weight、velocity/gripper weight、MLP 深度或 100/500/更多 steps；
这些属于同一 shared-codebook hypothesis，D3-D5 已有一致反证。下一候选必须做显式容量隔离：保持总 physics
token budget 优先为 256，在每个 16-token transition group 中预注册一部分 control-relevant slots 使用独立
EMA/FSQ residual quantizer，其余 slots 保持 visual reconstruction；同时与 256 个全 visual tokens 做 matched
control。需要先比较 `15 visual + 1 residual`（总数仍为 256）和 `16 visual + 1 residual`（总数 272）的信息
预算、VLM vocabulary/序列成本与 action-head routing，再选一个实现。无论哪种方案，4 个离散 action tokens
都保持独立存在；未过 representation gate 前仍不启动 action-DiT/policy 训练。

### 21.28 D6 独立 residual tokens：oracle gate 通过、VLM learnability gate 失败（2026-09-27）

为避免用减少 visual slots 引入混杂，先选择保留 256 visual physics tokens 并额外增加 16 residual tokens，
总 physics condition 为 272，序列成本增加 `6.25%`。新增 full-horizon residual cache：固定 observation-predicted
4 action IDs，经原 EMA action tokenizer 解码后，以真实 32x14 action 减 decoded trajectory，joint wrap、
gripper 连续 residual；cache SHA256 为
`4c9bf883af9851458f112ba0160e97ae6d21977a0b62febf88cb5dac8083ddba`。独立 residual tokenizer 为
`16 tokens x EMA-128`，官方 45/5 episode split、5000 steps；checkpoint SHA256 为
`c85279d52a84d4c4f36eb25df7d17e74a3e7c13ac36e870c9f28352e728a6f08`。验证 reconstruction loss 最低约
`0.0060`，最终 active ratio `0.625`、perplexity `29.98`。

拼接后的 `256 visual + 16 residual` oracle physics cache SHA256 为
`2572ba85c1e65563bcc19a55e9b87918f0f5e87f343be1f1ada05d60d18e2fd6`；与 predicted 4-action tokens
合并的 aligned/shuffled cache SHA256 分别为
`f645244cea348db1868aad77fadf7066a4d1cb2c9a0d1cbc94e3a3aaecd74242` 和
`932712dd93408e8a3f570937f425635d6c5e50a0f1295223c3ff9a07fe544505`。冻结 residual decoder 输出作为
448-D residual feature 时，同一 held-out residual-ridge probe 的 aligned action MAE 为 `0.0771668`，相对
state+4 action-token baseline `0.1023072` 改善 `24.573%`；episode-shuffled 为 `0.0999562`，aligned 相对
shuffled 改善 `22.799%`。因此显式独立容量的 representation gate 显著通过，证明此前 shared-codebook
失败确为容量/目标竞争，而不是 residual 概念无效。

随后复用 `QwenActionTokenProbe`，从当前图像与语言并行预测 16 residual IDs，只训练 VLM 后两层和 probe，
八卡 500 steps。checkpoint SHA256 为
`8d588673bf355f151d587ace62df928d404675acbccaa16905a928db174c2fef`。380-sample 官方 held-out panel
上 token accuracy 仅 `2.303%`、exact sequence `0%`、平均每样本命中 `0.368/16`；target 使用 107 codes、
perplexity `47.30`，prediction 只使用 39 codes、perplexity `11.48`。连续动作直接相加评估中，仅 predicted
4 action tokens 的 MAE 为 `0.114846`；加 oracle residual tokens 降至 `0.050634`，改善 `55.912%`；加
VLM-predicted residual tokens 升至 `0.117022`，退化 `1.895%`，gripper accuracy 从 `90.757%` 降至
`89.864%`。因此 learnability gate 失败，不进入 DiT。

D6 把问题明确分解为：专用 tokens 有足够且高度互补的信息容量，但 16x128 residual code 的条件熵太高，
当前观测下难以预测。训练集整体 residual-code perplexity 为 `79.62`，各位置约 `42.9-83.8`；相比之下，
原 4x128 action-token probe 在同样 500 steps 的 accuracy 为 `13.36%`。下一步 D7 固定 256 visual tokens
不动，只扫独立 residual branch 的低码率 `4x32 / 4x64 / 8x64`。先用 held-out quantization fidelity 与 oracle
action gain 筛选，再只对最优候选训练一次 VLM probe；不直接延长 D6 训练，因为目标熵/预测分布 collapse
已表明主要矛盾是码率，而不是 500 steps 尚未收敛。

### 21.29 D7 低码率 residual sweep 与 VLM learnability gate（2026-09-27）

已实现 `evaluate_oracle_residual_tokenizer.py` 与 `run_d7_residual_bitrate_sweep.sh`，在同一个 full-horizon
residual cache、EMA-VQ、官方 45/5 episode split、5000 steps 下并行训练 `4x32 / 4x64 / 8x64`。三个
checkpoint SHA256 分别为
`694ea0a189ef893d61b8dc091b4e177600de25552cf35bc59b33134f3cc0eecf`、
`b247577db6f2d7a65a24b4b0b50f3610f4a2632b25af33b65454dbc9f4bc944e`、
`f862ca6d7727fe3171a67b57fc042982b0729b4ca79c5a645a9e914d39c92dad`；对应 token cache SHA256 为
`7228fb726b31b569b2be987674ad97e2cd235fbb46a46bc6cb2e6d79bf85fb0f`、
`d0ab49fba35cdc8b73e2ddcce7e4e4b8bed39315621b285863e41efff38e3bb1`、
`09295f2d17294d538ad94546d46bb1372807e6a983daae9005d838e876684159`。

step 5000 validation reconstruction loss 依次为 `0.010169 / 0.008859 / 0.007000`，active ratio 为
`0.828 / 0.719 / 0.773`。在全部 439 个 held-out samples 上，固定 observation-predicted 4 action tokens
的 baseline action MAE 为 `0.113100`；加入 oracle residual 后三组 MAE 分别为
`0.065377 / 0.059797 / 0.054269`，相对改善 `42.195% / 47.129% / 52.017%`，因此三组 oracle
representation gate 均显著通过。训练集经验序列熵依次约为 `16.58 / 21.35 / 42.64 bits`；`4x64` 只需
预测 4 个 IDs，码率约为 `8x64` 的一半，却保留其大部分 oracle 收益，且比 `4x32` 多约 4.93 个百分点收益，
因此按预注册规则只选择 `4x64` 做一次 VLM probe，不运行其余两组 VLM 训练。

`evaluate_residual_token_probe.py` 已参数化 `token_count/vocab_size`，并新增固定、拒绝覆盖的
`run_d7_selected_vlm_probe.sh`。`4x64` probe 仍使用 100k pretrain、VLM 后两层、8 GPUs、global batch 8、
500 steps；checkpoint SHA256 为
`9e981986a66a72f69c761779eb3cb53e6575b8c14ccd2e0e84339bba9c600ce2`。380-sample evaluation
SHA256 为 `d5e49cbaf9f94a17b4979f3bc501d26e72ce5182e250d047cc32f0c8b9b36c37`。token accuracy 为
`6.908%`，exact sequence 为 `0%`，平均命中 `0.276/4`；target 使用 59 codes、perplexity `30.99`，
prediction 只使用 16 codes、perplexity `6.78`。这比 D6 的 `2.303%` accuracy 明显提高，但预测分布仍
collapse。

连续动作最终 gate 失败：仅 predicted 4 action tokens 的 MAE 为 `0.114846`；加 oracle `4x64` residual
tokens 为 `0.060453`，改善 `47.362%`；加 VLM-predicted residual tokens 为 `0.114979`，反而退化
`0.116%`。gripper accuracy 从 `90.757%` 降至 `89.934%`，velocity MAE 也变差。wrapped joint MAE
从 `0.118661` 降至 `0.116953`（约改善 `1.44%`），说明预测 IDs 可能保留少量 joint 信号，但不足以通过
预注册的整体 action-MAE gate。结论保持严格：不启动 DiT，不盲目延长 500-step probe，也不在看到结果后
补跑 `4x32/8x64` 来挑最好结果。下一步仅做 component-gating 诊断，以区分“token 完全不可用”和“joint
信息可用但 residual decoder 的 gripper/幅度需要由可学习 action head 抑制”；该诊断不改变 D7 fail 决策。

D7 post-hoc component diagnostic 使用同一 checkpoint、同一 380 samples、无重训练。直接只应用 predicted
joint residual 时 action MAE 为 `0.113382`，相对 baseline 改善 `1.275%`；只应用 gripper residual 时为
`0.116443`，退化 `1.391%`；全部 residual 统一缩放到 `0.5x` 时为 `0.114193`，改善 `0.568%`。诊断文件
SHA256 为 `18f902d9be35002994d4e9af3a7b09eb50acecbf59a0bec779a48efefa42106f`。这说明低码率预测 tokens
包含可用 joint 信号，D7 整体失败主要由无条件 decoder 的 gripper 与幅度校准造成，而不是 IDs 与动作完全无关。

D8 因此预注册为 frozen-VLM learned-gating probe：先为全部 4252 samples 导出 `4x64` predicted residual IDs，
再把 frozen residual decoder 的 32x14 输出作为 feature；严格使用 40 fit + 5 calibration train episodes 选择
ridge alpha，最终在全部 45 train episodes 重训，只评估官方 5 held-out episodes，并加入 episode-shuffled
predicted IDs control。门槛保持 representation gate 的双 `+1%`：aligned 相对 state+4 predicted-action-token
baseline action MAE 至少改善 `1%`，且 aligned 相对 shuffled 至少改善 `1%`。通过才允许启动最小 predicted-token
conditioned DiT smoke；失败则不再优化 action-head gating，而转向 observation-predictable tokenizer/joint
action+physics token training。

### 21.30 D8 predicted-token learned gating 结果与 D9 联合预测预注册（2026-09-27）

D8 已导出全部 4252 个 observation-predicted `4x64` residual IDs，cache SHA256 为
`81e23e984a9efdb91ce48a6c41542986cd8259cb2c72ed95132a0a339c5c1429`。与固定 predicted 4 action
tokens 合并后的 aligned/shuffled cache SHA256 分别为
`31ff155ad93cc8eb3062b4807bad0e591ca260cc28b2bfec87ff06d2cb2cf1c9` 和
`9e2fa1f3e3e81e3225c1e432b660f31daee73840eceeecc7860f93e47731d6b3`。评估文件 SHA256 为
`31315f13cff236317aacdd3c3f356e85fb86d94b472d5bcd62adec8fa7afdd8d`。

严格 held-out episode 结果未通过双 `1%` gate。state + predicted 4 action-token baseline MAE 为
`0.1023072`；加入 aligned predicted residual decoder feature 后为 `0.1021099`，只改善 `0.193%`；
episode-shuffled 为 `0.1025507`，aligned 相对 shuffled 只改善 `0.430%`。因此 380-sample 上手工
joint-only `+1.275%` 不能稳定转化为从训练 episodes 学得并泛化的 gating 规则；不启动 DiT，也不继续搜索
ridge/MLP/action-head 容量。

D9 转向最终设计本身的最小联合预测验证。训练 cache 使用固定旧 action probe 的 4 个 predicted action IDs
作为 action target，并使用基于同一 IDs 构造的 oracle `4x64` residual IDs 作为 physics target，保证两者语义
严格对应。VLM pooled state 后保留独立 `4x128` action head 与 `4x64` physics head；physics head 额外读取
action logits 的 soft expected embedding，physics loss 可通过该路径塑造 action 表征，且训练/推理均使用 soft
prediction，避免 teacher-forcing exposure mismatch。loss 为两个 CE 的等权和，仍只训练 VLM 后两层与双 heads，
首轮固定 500 steps、八卡 global batch 8。

D9 gate 在同一 380-sample held-out panel 上要求：（1）dual predicted action + dual predicted physics 的 action
MAE 优于 dual predicted action-only；（2）组合结果不差于现有固定 4-action-token baseline `0.114846`；（3）
physics prediction 不出现比 D7 更严重的分布 collapse。未同时满足则不进入 DiT；不因 joint CE 降低而放行。

D9 checkpoint SHA256 为
`0e96fe1d83746ab46db9f310b5a59ab47d521f66978812458c791cf18c07804c`。held-out action/physics token
accuracy 分别为 `54.342% / 7.961%`；physics prediction 仍为 16 active codes、perplexity `6.11`，未解决
collapse，但 accuracy 比独立 D7 的 `6.908%` 提高。dual action-only MAE 为 `0.131592`，加入 dual predicted
physics 后为 `0.124949`，相对改善 `5.048%`，说明显式 action-conditioned joint training 确实让 physics
分支提供了更强动作增量；但组合仍比固定成熟 action baseline `0.114846` 退化 `8.797%`，D9 总 gate 失败。

固定成熟 action tokens、只接入 D9 predicted physics 的 post-hoc diagnostic MAE 为 `0.112900`，相对 baseline
改善 `1.694%`；诊断文件 SHA256 为
`c5af997c3cae5289550ff86fc96708cd299a3a9e23b3ee9349b2c546123e1d08`。因此主要瓶颈是 D9 随机初始化
action head 在 500 steps 只复现约一半 IDs，而不是 physics 分支无效。

D10 预注册为 action-initialized matched joint run：从成熟 `4x128` probe checkpoint
`d9ef087d1cb94427f119b84bf011ff81c3fce3f8dd23141db747219b267f431a` 精确加载 VLM 后两层、norm 与
action head；physics head/action-condition embedding 保持随机。为保护 action 能力，VLM/action-head LR 固定为
`2e-6/1e-5`，physics modules 仍为 `1e-4`，其余数据、loss、500-step/八卡预算与 D9 相同。gate 要求 dual
action-only 不相对固定 baseline 明显退化，且 dual action + dual physics 的最终 MAE 严格优于固定 baseline；
否则仍不进入 DiT。

### 21.31 D10 结果与 D11 冻结动作分支预注册（2026-09-27）

D10 八卡 500 steps 正常完成，成熟 action checkpoint 的 SHA256 校验通过并精确加载 29 个 tensor；最终
checkpoint SHA256 为 `671cae2c4262aa1fc1ea5512bcc1b28d840d660277af2f6ddf9c2ac26ead52f4`，380-sample
evaluation SHA256 为 `6dd3f3389b60854ccc218e0d2974447f1161f1c0f9db93a305d6c64b1be9bfcc`。action/physics token
accuracy 分别为 `61.184% / 7.895%`，action exact-sequence accuracy 为 `30.526%`；physics prediction
只使用 14 个 codes、perplexity `5.93`，相对 59-code、perplexity `30.99` 的 target 仍明显 collapse。

尽管 action accuracy 高于 D9，联合微调仍严重破坏成熟动作输出：固定成熟 action-token baseline MAE 为
`0.114846`，D10 dual action-only 为 `0.150758`，退化 `31.270%`。加入 D10 predicted physics 后 MAE
为 `0.148807`，相对其自身 dual action-only 改善 `1.294%`，但相对固定成熟 baseline 仍退化 `29.571%`。
固定成熟 action + D10 predicted physics 的 MAE 为 `0.113576`，相对固定 baseline 改善 `1.106%`；oracle
physics 分别可将 dual/fixed action 降到 `0.111445 / 0.060453`。因此 physics 分支仍提供方向正确的增量，
D10 失败来自 action/VLM 联合更新导致的灾难性遗忘；按预注册 gate 不进入 DiT。

D11 只检验这一因果解释：从同一成熟 action checkpoint 初始化后，严格冻结整个 `qwen_vl_interface` 和
`action_token_probe`，只训练随机初始化的 `action_condition_embedding` 与 `physics_token_probe`。数据、4 个
action tokens、4x64 physics tokens、45/5 episode split、global batch 8、500 steps 和 physics LR `1e-4`
均保持不变。检查点只保存两个可训练 physics modules；评估必须先按 SHA256 恢复成熟动作分支，再加载 physics
modules，防止随机 action head 混入。D11 gate 要求：（1）dual action-only 与固定成熟 baseline 逐数值一致；
（2）dual action + predicted physics 严格优于固定 baseline；（3）改善至少 `1%` 才允许下一步最小 DiT smoke。
若仅满足严格改善但不足 `1%`，结论为信号存在但效应偏弱，仍不启动 DiT；若不改善，则停止当前 residual-token
条件形式，转向降低 observation-conditional entropy 的 tokenizer 目标，而不继续调 physics-head 容量。

### 21.32 D11 结果与 D12 action-logit 蒸馏预注册（2026-09-27）

D11 checkpoint 只包含 7 个 physics-module tensors、大小约 7.6 MB，SHA256 为
`1f262ba24dbdaadc6a49b00ddb20acd9c1f48c3273298fe4cd83e8b8e78c9de2`；evaluation SHA256 为
`f82dffbd858a560b8f12922082b11ad8d5e43766baa6bd7a10fd12aa7a0058b0`。评估器先校验并恢复成熟 action
checkpoint，再加载 physics-only checkpoint，参数集合严格匹配。action/physics token accuracy 为
`66.513% / 6.908%`；physics prediction 仅 13 active codes、perplexity `6.23`，collapse 仍存在。

冻结动作分支的直接 action MAE 为 `0.111924`，加入 predicted physics 后为 `0.112594`，相对自身基线退化
`0.599%`，因此核心增量 gate 失败。固定 cache 中 action IDs 的解码 MAE 为 `0.114846`，加入相同 physics
预测后为 `0.114626`，只改善 `0.192%`，同样不足 `1%`。组合结果表面上比固定 cache baseline 改善
`1.960%`，但这是当前成熟 action 模型重新推理所得 IDs 本身更好造成的，不能归功于 physics tokens；此前
“dual action-only 必须与 fixed baseline 逐数值一致”的假设忽略了 cache 生成与本次 materialization 的图像
增强/推理口径差异，正式判定改为组合相对同次 dual action-only，避免错误放行。D11 不进入 DiT。

D9/D10 中可训练 VLM 时 physics 对各自 action baseline 分别改善 `5.048% / 1.294%`，而 D11 完全冻结后
退化 `0.599%`，说明 physics 分支需要适配视觉语言表征；但 D10 又证明无约束适配会破坏 action。D12 因此
引入同初始化的冻结成熟 teacher，在每批完全相同的增强图像上计算 teacher action logits；student action hard-CE
权重设为 0，以免 cache IDs 与当前增强视图不一致，loss 固定为 physics CE + `10 * KL(teacher || student)`，
temperature `1.0`。student VLM/action LR 沿用 D10 的 `2e-6/1e-5`，physics modules 为 `1e-4`，其余数据、
500-step 和八卡预算不变。teacher 仅训练时存在、不保存且评估时不构建。

D12 gate 要求：（1）student action-only MAE 相对 D11 的同次成熟 action baseline `0.111924` 退化不超过
`0.5%`；（2）加入 predicted physics 后相对本次 student action-only MAE 改善至少 `1%`；（3）最终 MAE
优于 `0.111924`。三项同时满足才启动最小 predicted-token-conditioned DiT smoke；失败则不搜索 KL 权重，
而转向显式降低 physics target 的 observation-conditional entropy。

### 21.33 D12 结果与 D13 observation-predictable residual 上界预注册（2026-09-27）

D12 checkpoint SHA256 为 `07978fbdaaa6733a1a6ba21018d0af39f8eb9bd71caf45f5658c2553989201b6`，
evaluation SHA256 为 `519f78018fa6376ba820770f1b0704ba3e946c8a03d768a79173784a46aa7a87`。训练中 action KL
通常维持在约 `1e-3`，最终 action token accuracy `65.921%`；student action-only MAE 为 `0.111482`，相对
D11 成熟 action `0.111924` 改善 `0.396%`，所以动作保护 gate 通过。但 physics accuracy 仅 `7.303%`，
预测仍只有 13 active codes、perplexity `6.16`；加入 predicted physics 后 MAE 为 `0.112318`，相对自身
action baseline 退化 `0.750%`。固定 cache action + physics 也从 `0.114846` 退化到 `0.115000`。因此 D12
明确失败，不进入 DiT，也不搜索 KL weight、temperature、VLM LR 或 physics-head 容量。

D9-D12 的共同证据是 oracle residual 价值很高，但 observation-predicted code 分布持续 collapse；即便解决
动作遗忘，离散 physics 仍没有稳定增量。D13 先不训练 tokenizer/VLM，而测“当前观测可预测残差”的连续
上界：复用无增强的 4252-sample `4x4x2560` 当前帧 Qwen visual cache，对 16 spatial tokens 求均值后用固定
seed-42 Gaussian 投影到 512-D，再拼接 4x128 action-ID one-hot。严格使用 40 fit episodes、5 calibration
episodes 选 ridge alpha、全部 45 train episodes refit、5 官方 held-out episodes 只评估一次；target 是同一
predicted action IDs 未解释的 32x14 wrapped residual。另做 episode-shuffled current-visual control，保留 action
IDs 不变。

D13 gate 要求 aligned residual prediction 相对 zero-residual decoded-action baseline 至少改善 `1%`，且 aligned
相对 shuffled 至少改善 `1%`。通过后才把 ridge 的 observation-predictable residual 输出作为新 tokenizer 的
训练 target，并重新做 `4x32/4x64` oracle bitrate gate；失败则说明当前单帧视觉表征与 45 episodes 数据不足以
稳定预测 residual，应优先增加任务/episode 覆盖或引入短历史，而不是继续制造更多离散 token。

### 21.34 D13 结果与 D14 短历史动态上界预注册（2026-09-27）

D13 evaluation SHA256 为 `00484dc254cfaacd48006349009a2be11e21f6d73226126ec6fdb40dc6e67953`。
439 个 held-out samples 上 zero-residual decoded-action baseline MAE 为 `0.113100`；aligned current-image
residual ridge 为 `0.115791`，相对 baseline 退化 `2.379%`；episode-shuffled visual 为 `0.118415`，aligned
相对 shuffled 改善 `2.216%`。两项 `1%` gate 只通过 correspondence control、未通过实际 utility，因此不把
ridge 输出离散化。calibration 最优 alpha 已到搜索上界 `1e5`，且 calibration MAE `0.067665` 与 validation
差距很大，说明单帧映射存在明显 episode distribution shift，而不是 ridge 正则太弱。

D14 检验短历史能否补足速度/相位信息。复用同一无增强 frame cache，按 `(episode_id, base_index)` 只在同一
episode 内构造 `[current, current-past2, current-past4, current-past8]`，episode 起点因果回退到最早可用帧；
拼接后仍固定投影到 512-D，故与 D13 的视觉特征预算、action-ID features、split、alpha search 和模型容量
完全一致。gate 仍为相对 zero-residual baseline 与 episode-shuffled history 各改善至少 `1%`。通过才离散化；
失败则停止单任务 50-episode representation tuning，把下一资源投入放到多任务/更多 episode 数据覆盖。

### 21.35 D14 结果、单任务停止条件与 D15 数据扩展要求（2026-09-27）

D14 evaluation SHA256 为 `03e836d47bc091d4dce6bb186a4f5b039f86e5b4dea444a59ce14f121fe9ec14`。
zero-residual baseline 仍为 `0.113100`；aligned causal-history residual ridge MAE 为 `0.115412`，相对 baseline
退化 `2.044%`；episode-shuffled history 为 `0.117880`，aligned 相对 shuffled 改善 `2.093%`。相比 D13
单帧 `0.115791` 只有小幅改善，仍未改变 utility gate 失败。calibration 继续选择搜索上界 alpha `1e5`，
calibration/validation MAE 为 `0.066496 / 0.115412`，跨 episode gap 依然很大。按预注册停止所有基于
click_alarmclock 50 episodes 的 projection/history/ridge/tokenizer 调参，不离散化 D13/D14 输出，不进入 DiT。

现有多任务资产审计显示，`robotwin_d1_8tasks_8episodes_spatial4x4_17frames.npz` 覆盖 8 个代表任务：
click_alarmclock、beat_block_hammer、open_microwave、shake_bottle、move_playingcard_away、
pick_dual_bottles、handover_block、place_object_basket；共 64 个训练 episodes、14,077 samples，视觉 cache
为 16x2560 current-frame features。它比单任务 4,252 samples 大约 3.3 倍且任务更多，但每任务仅 8 个训练
episodes；更关键的是现有配套 action sidecar 是 17x14 offsets，而当前 action tokenizer、residual cache 和
最终 action metric 都是完整 32x14 horizon，不能直接拼接使用，否则 token 定义与 residual target 不一致。

D15 不应立即训练模型，先补齐 matched multi-task 数据层：对上述 8 tasks 至少导出相同 episode 集的
32x14 action cache、有效位 mask、task-local `(task_id, episode_id, base_index)` keys，并做 train/validation
episode 隔离；随后训练共享的多任务 4x128 EMA action tokenizer 与 action-token probe，再以其 observation-
predicted IDs 生成 matched residual。为真正检验“数据量不足”，优先从每任务 8 episodes 扩至每任务 40 train
+ 5 validation episodes，而不是只复用当前 64 episodes。完成 key/shape/hash 审计后，先重复 D13/D14 连续
上界 gate；只有跨任务 macro action-MAE 相对 baseline、aligned 相对 shuffled 都至少改善 `1%`，才训练新的
EMA physics tokenizer。FSQ 仍只作为通过 EMA gate 后的 matched control，不提前增加实验分支。

D15 数据层实现已从 full 32x14 action cache 开始：新增
`merge_multitask_action_caches.py` 与 `run_d15_export_multitask_action_cache.sh`。单任务 exporter 保持原逻辑，
合并器要求 8 个 part 的 action shape 完全一致，并把 key 提升为
`(task_id, episode_id, base_index)`；跨任务重复 episode ID 不再冲突，任何 task-name、manifest、row-count、
shape 或 qualified-key 不一致均 fail closed。合并产物记录每个 part 与 source manifest 的 SHA256；此阶段只补
数据，不训练 tokenizer 或 policy。

D15 action 导出已完成。8 tasks 均覆盖 50 episodes，总计 400 task-episodes、85,098 个 32x14 chunks；各任务
sample 数依次为 `4,252 / 5,682 / 24,333 / 12,436 / 5,884 / 6,129 / 14,084 / 12,298`。合并 cache
`artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz` SHA256 为
`f8dcbe47103a31c673bc504a1c50f380f55afe56edaf9e00329f61ff6e749630`，manifest SHA256 为
`91131abdbe2672899d4fca12fcbf47096386fde4016d2efea993efe5e74f355f`。最终 shape 为
`actions=[85098,32,14]`、`action_valid_mask=[85098,32]`、task/episode/base keys 均为 85,098 行且无重复。
新导出的 click_alarmclock part 与既有 `click_alarmclock_clean_wrap32.npz` 的 actions、valid mask、dim mask、
episode IDs 和 base indices 全部逐元素一致，确认扩展过程没有改变旧数据口径。下一数据依赖是为同一 400
task-episodes 导出 matched current/history visual features；在其完成前不训练多任务 tokenizer。

### 21.36 D15 多任务视觉 cache 导出预注册（2026-09-27）

D15 视觉侧固定为与 action cache 相同的 8 tasks x 50 episodes、stride 1 和无增强 `video.cam_high`；为控制
共享盘占用及先验证“数据覆盖”这一单变量，只保存 current frame (`offset=0`) 并把 Qwen3-VL patch feature
自适应池化到 `1x1x2560`。短历史不额外存储：后续 D14-style history 可根据 `(task_id, episode_id,
base_index)` 从 current-frame cache 因果索引 `[t,t-2,t-4,t-8]`，因此不会损失实验能力。

实现新增 `run_d15_export_multitask_visual_cache.sh`，在 node004 八卡上一任务一卡并行，单任务原子写 part，已完成
part 可恢复跳过；新增 `merge_multitask_visual_caches.py`，合并时累加重映射 deduplicated frame indices，要求
offset、feature shape/dtype、camera、VLM config/checkpoint hash 与 stride 全部一致，并将单任务 task ID 提升为
固定全局 ID。最终 merger 必须逐行验证 85,098 个 `(task_id, episode_id, base_index)` 与 D15 action cache
完全一致，否则 fail closed，不产出 merged cache。全量前先用 1 episode smoke 验证模型加载、schema 与 index
边界；全量 cache 完成且 SHA256 留档后，才运行多任务 D13/D14 continuous residual upper-bound gate。

多任务 probe 的严格口径已经补齐：所有 key alignment 在存在 `task_ids` 时都使用三元组；历史查询使用
`(task_id, episode_id, base_index)`；episode-shuffled control 只在同一 task 内置换；输入显式加入 task
one-hot，模拟真实 VLM 可读取任务指令；ridge alpha 由 8-task macro calibration MAE 选择，最终同时报告 micro、
每任务和 macro 指标。固定 split 沿用 episode IDs `5/20/25/39/49` 为 held-out，每任务其余 45 episodes 中
最后 5 个作 calibration、40 个作 fit，因而总计为 320 fit、40 calibration、40 held-out task-episodes。

首个 D15 上界不是最终 predicted-token 实验：`run_d15_multitask_residual_upper_bound.sh` 先用已有单任务成熟
`4x128` EMA tokenizer 对 8-task action chunks 作 oracle encode/decode，再预测其未解释 residual。这样在不先训练
新 tokenizer 的条件下，低成本回答“扩大到 400 episodes 后视觉 correspondence 是否能跨 episode 泛化”；
同时其跨任务 reconstruction baseline 会量化单任务 tokenizer 的 domain shift。只有 macro aligned 相对 zero-
residual 和 task-local shuffled 均改善至少 `1%` 才进入共享多任务 EMA tokenizer；此结果不能替代之后必须进行的
observation-predicted action-token gate，也不能直接放行 DiT。

### 21.37 D15 多任务连续上界结果与 D16 共享动作 tokenizer 预注册（2026-09-27）

D15 visual cache 完成：8 tasks x 50 episodes、85,098 samples，shape `[85098,1,2560]`，压缩大小
387,052,091 bytes；cache SHA256 为 `44d356027a31f374c5b2ff677ba6bebc5bde068e843767940cd1adc35d54aaae`，
manifest SHA256 为 `1721d2128fc486fbb7f56d8213bf1df8d6cd02284b195e48b4709107090525f6`。merger
验证全部 qualified keys 与 action cache 逐行一致。单任务 part hashes 已固化在 merged manifest 中。

冻结 click-only EMA `4x128` tokenizer 得到的 8-task oracle token cache SHA256 为
`a3aefea94ef7e4a56c576857acc9ca0de6ad97d2c0ac7d99fea6a5032c18bfa2`，wrapped residual cache SHA256
为 `ca4d784221ba4417ec08c520ce0e3887c08b03fd241ac8630a82263b21bfb312`。current probe evaluation SHA256
为 `8e6c3306e11d281bbf77354104fa42659c1f85f42a0ee7ffc73079e2f42bf604`：zero-residual macro MAE
`0.250578`，aligned current `0.122809`，改善 `50.990%`；task-local shuffled `0.155808`，aligned 相对其
改善 `21.180%`。history evaluation SHA256 为
`144d0caf9ea8f9ba569b4b392f690e9691cf038c00f55f9fd98f7137bd7afeb0`：aligned `0.136026`，相对
baseline 改善 `45.715%`，相对 shuffled `0.155403` 改善 `12.469%`。两者均通过双 `1%` gate，且 current
严格优于 history，因此下一阶段保持 current-only。

current 的逐任务 baseline -> aligned MAE 为：click `0.033604 -> 0.128046`、beat `0.080563 -> 0.079307`、
open `0.142374 -> 0.063361`、shake `0.192488 -> 0.104385`、move-card `0.140794 -> 0.172692`、pick-dual
`0.694020 -> 0.179440`、handover `0.486256 -> 0.133498`、basket `0.234525 -> 0.121741`。宏观大幅收益主要
来自旧 tokenizer 在新任务上的 domain shift；click/move-card 退化说明 ridge 不是最终动作头，只证明多任务数据
中存在可泛化 observation-residual correspondence。

D16 训练共享 8-task EMA action tokenizer，结构保持 `4 tokens x vocab 128`，不改 bitrate；5000 steps、batch
512、固定 45 train/5 held-out episodes/task，仍保留原 reconstruction/velocity/gripper/wrap loss 权重。训练固定
单 GPU：EMA codebook buffer 若用当前 DDP `broadcast_buffers=False` 会在各 rank 基于不同 batch 独立更新，不能
保证同一 codebook，故不为表面加速引入量化器语义偏差。trainer/evaluator 已增加 `--split-manifest`，确保使用与
D13-D15 相同的 held-out IDs。D16 通过条件为 held-out macro reconstruction 明显修复跨任务 domain shift、codebook
不 collapse；通过后才导出共享 oracle IDs，并训练新的多任务 observation->action-token probe。

D16 单卡 5000 steps 已完成，约 222 秒。checkpoint SHA256 为
`414a751119855119afa9604054e5413a845913e76de22c472bed5375973fdc86`，固定 held-out evaluation SHA256 为
`721098230809c6d473bc70418c8eccee1d2d49569dc146a0b86132f8b123521b`。8,989 held-out chunks 上 joint
wrapped MAE 为 `0.057927 rad / 3.319 deg`，joint velocity MAE `0.007896 rad`，gripper MAE/accuracy 为
`0.008539 / 99.381%`；全局 118/128 codes active，active ratio `92.188%`，perplexity `86.223`，明确没有
codebook collapse。训练期 per-batch perplexity 约 14 只是单 batch 覆盖统计，不能代替全 held-out usage。

D16b 在启动 VLM probe 前重新执行 shared-tokenizer continuous residual gate。原因是 D15 的 `50.99%` macro
改善部分来自 click-only tokenizer 在 pick/handover 等任务上的巨大 reconstruction domain shift；共享 tokenizer
已显著增强 baseline 后，旧结论不能直接外推。D16b 固定同一 visual cache、split、512-D projection、task context
及 task-local shuffle，只替换为共享 `4x128` oracle IDs 和对应 residual。仍要求 aligned 相对 zero-residual 与
shuffled 的 macro MAE 均改善至少 `1%`；失败则说明先前收益主要是修补 tokenizer，而不是值得离散化的额外 physics
条件，不启动多任务 physics-token/VLM 分支。

D16b 已通过。共享 oracle token cache SHA256 为
`1d8da1fb8c3768626bca531b21f27d20be558d2c51ea3410c934d12f3ef87f66`，residual cache SHA256 为
`f497d107943bd9444aca70784428a9c6659a3b6261b8b817dcaffb4df5b1cc27`，evaluation SHA256 为
`e3f714aefb927ba71e24b983b3909910a611b12e2ca885962123d71e624afbfa`。共享 tokenizer zero-residual
macro MAE 已从旧 tokenizer 的 `0.250578` 降至 `0.062550`；aligned current 为 `0.058971`，仍改善
`5.722%`；task-local shuffled 为 `0.062826`，aligned 相对改善 `6.135%`。逐任务 baseline -> aligned 为
click `0.064168 -> 0.066647`、beat `0.041751 -> 0.034538`、open `0.023311 -> 0.023148`、shake
`0.045908 -> 0.044349`、move-card `0.089767 -> 0.082573`、pick-dual `0.107882 -> 0.081358`、handover
`0.071051 -> 0.089433`、basket `0.056563 -> 0.049724`。macro 双 gate 通过，但 click/handover 退化必须在
后续报告保留，不能只报平均值。

### 21.38 D17 多任务 action-token VLM probe 预注册（2026-09-27）

多任务 token cache join 已从二元 `(episode,base)` 扩展为 task-qualified
`(task_id,episode_id,base_index)`；只有 cache 含 `task_ids/task_names` 时启用，新旧单任务 cache 保持兼容。
新增固定 mixture `robotwin_phystoken_d15_8tasks_wrap32`。真实数据 smoke 已逐任务构造 dataset 并核对首行 token，
8 个任务均与 merged cache 对应三元 key 逐元素一致，排除跨任务 episode ID 冲突和静默串标签。

D17 使用 Qwen3-VL 预训练权重、共享 D16 oracle `4x128` targets、每任务排除固定 5 held-out episodes；VLM 仅训练
最后 2 层、final norm 和 parallel token head，8 卡 global batch 8。首轮固定 1000 steps（约每任务 125 个
optimizer batches 的低成本 gate），不直接扩到长训练。通过条件：（1）held-out macro token accuracy 明显高于
chance `0.78125%` 且 predicted code usage 不严重 collapse；（2）decoded predicted-token macro action MAE 相对
简单 task/modal-code baseline 有实质收益；（3）生成 predicted IDs 后，基于其 residual 的 continuous visual upper
bound 仍通过双 `1%` gate。三项满足才训练 EMA physics tokenizer；否则先处理 action-token predictability。

D17 1000-step checkpoint SHA256 为 `d12a26c26fbff2efb32a37ecc2774ce12387e95566107aa81811092b7b7442a4`。
每任务 64 个、共 512 个 held-out samples 的 evaluation SHA256 为
`08f707b65b0ec6ca70998e62aca2e1f3619cdd7a87a19ab6dd9e793fd08c151c`。macro token accuracy
`36.865%`、exact sequence accuracy `10.547%`，显著高于单 token chance `0.781%`；全 panel prediction 使用
83 codes、perplexity `55.58`，没有严重 collapse。decoded predicted action macro MAE 为 `0.124509`，oracle
token 下限 `0.060777`，task modal-token baseline `0.473449`，预测相对 modal 改善 `73.702%`。逐任务 token
accuracy / predicted action MAE：click `27.34% / 0.12391`、beat `64.06% / 0.06886`、open `31.25% /
0.05668`、shake `28.13% / 0.10022`、move-card `14.84% / 0.19661`、pick-dual `50.39% / 0.15219`、
handover `36.33% / 0.13416`、basket `42.58% / 0.16343`。D17 action-predictability gate 通过，但相对 oracle
仍有约 2x MAE gap，后续 physics residual 必须基于这些自由运行预测而非 oracle IDs。

下一步 D17 export 对全部 8 tasks x 50 episodes 生成 observation-predicted IDs。一任务一卡、eval batch 16，
每个 part 原子落盘且可恢复；merger 将任务 ID 提升为全局固定顺序，并逐行要求 85,098 个 qualified keys 与 D16
oracle cache 完全一致。只有 merged predicted cache 完成后，才构造真实 predicted-action residual 并重复 continuous
upper-bound gate；不允许用当前 512-sample panel 直接训练 physics tokenizer。

D17 full-cache exporter 的初版逐样本 PyAV seek 在长任务上过慢，因此新增严格等价的 episode cache 路径：每个
episode 的三路相机各完整解码一次，临时替换 dataset `get_video`，其余 `get_step_data -> transforms ->
pack_sample -> Qwen processor` 链路完全不变。不能直接把 640x480 原帧送给模型，因为正式 config 的
`image_size=[224,224]` 会经过 VideoToTensor/VideoResize/VideoToNumpy；两次失败的输入等价 smoke 正是因此
fail closed，均未留下 part。最终实现把输入比较放在 processor 前，并限制每进程 OMP/MKL/OpenBLAS 为 4、
OpenCV 为 1，防止八进程嵌套线程过载。

优化后的 click full part 已完成，4,252 samples，首 batch fast/slow 三路变换后图像与语言完全相等，重复前向
64/64 token agreement 为 `1.0`。part SHA256 为
`5b3458c1e48cdaf84933101e3434e214bc8d1aeccd8d85d8a2d532c2297c7f2d`，manifest SHA256 为
`9bf7f180a9d2c081c95b890e034215ceb7fba998f04e59522cc46f298dc9f830`。其余七任务尚未完成：node004 后续出现
另一用户的 8 个 Habitat/OVON eval（每卡约 1.27 GB 且 CPU load > 200），为避免资源干扰已停止本项目七个
未落盘进程。`run_d17_export_multitask_predicted_action_tokens.sh` 会自动跳过 click part，资源空闲后可恢复其余
任务并严格合并。

D18 runner 已预置但不会在 merged D17 cache 前启动。它用自由运行 predicted 4 action IDs 解码，再从真实
32x14 action chunk 构造 wrapped residual，重复 current-only macro continuous gate。D18 仍要求 aligned 相对
predicted-action zero-residual baseline 与 task-local shuffled 各改善至少 `1%`；这才是允许训练多任务 EMA physics
tokenizer 的最终 representation gate。oracle-token D16b 通过不能替代 D18。

### 21.39 D17 全量 predicted action-token cache 与 D18 最终 representation gate（2026-09-27）

D17 已在 node004 与现有任务共享 GPU 的条件下完成。八个 task part 均通过首 batch 快/慢数据路径逐元素一致检查，
且重复前向 token agreement 全为 `1.0`；日志无 OOM、traceback 或进程被杀。合并 cache 为
`artifacts/token_caches/robotwin_d17_8tasks_predicted_shared_ema4x128_steps1000.npz`，包含 85,098 个样本，
token shape `[85098,4]`，预测覆盖 119/128 codes；与 D16 oracle cache 的
`(task_id,episode_id,base_index)` 逐行完全一致。cache SHA256 为
`ff8d845667778686586f6fb30c2d6987eed5bcde6e23648975b3033b90435bc5`，manifest SHA256 为
`54b6a97ac13671873854c3d4cbd1e50db9b3a2eff8ce0bbea9dbf01f2c24ad48`。

D18 由自由运行 predicted IDs 解码动作，再构造 `[85098,32,14]` wrapped residual；residual cache SHA256 为
`2d3de727d0d25ac53d714886f5e0ab3460f3f71fe421719b4c4a2cf923f77541`。新增机器可执行的
`require_continuous_residual_gate.py`，只有 macro aligned 相对 zero-residual 和 task-local shuffled 均改善至少
`1%` 才返回成功。D18 evaluation SHA256 为
`4645f0c06cd131b5d44e09f8a76210e2be5a9997815a045fe1f8ede2dc1efaa2`：predicted-action baseline macro
MAE `0.12521269`，aligned current `0.12146602`，改善 `2.992%`；task-local shuffled `0.12518732`，aligned
相对改善 `2.973%`。gate SHA256 为
`23de53f231cc07f80f9ee46ee144c44729e0d754261c71370ebcbd4d6ef31237`，双 gate 正式通过。该结果低于
oracle-action D16b 的约 `5.7/6.1%`，符合 action-token 预测误差增加 residual 不确定性的预期，但仍证明真实
推理条件下存在 observation-predictable 的增量连续信号。

### 21.40 D19 episode-cross-fitted predictable-residual teacher（2026-09-27）

不能直接把 ridge 在自身训练样本上的拟合输出当 physics supervision，否则会把 episode-specific 拟合噪声离散化。
新增 `export_observation_predictable_residual_teacher.py`：45 个 policy train episodes 固定 seed 后分为 5 folds，
每 fold 用另外 36 episodes x 8 tasks 拟合 ridge，只预测未参与拟合的 9 episodes；5 个官方 validation episodes
只由全部 45 train episodes 拟合后预测。所有 85,098 行恰好得到一次 finite prediction，且 action IDs、task
context、视觉投影、alpha=`100000` 与通过 gate 的 D18 aligned 模型一致。

D19 teacher cache shape 为 `[85098,32,14]`，SHA256 为
`98a39523f6e9a0edeae0c0d7d6c81df6e8a2ba886764136d9b59b31e54417805`；manifest 明确记录
`no_episode_self_fit=true`。teacher 相对原始 predicted-action residual 的全数组 MAE 为 `0.107781`，这不是最终
动作指标，而是说明 teacher 只保留当前视觉/任务/action-token 条件下能由 ridge 稳定抽取的低熵部分。后续 VLM
学习的是该可预测物理残差信号，不能重新使用原始高条件熵 residual IDs。

### 21.41 D20 物理 token 数量 matched sweep：16/64/256（2026-09-27）

旧 `ActionChunkTokenizer` 结构要求 `num_tokens <= horizon=32`，因此把 residual codec 硬扩到 64/256 会产生
重复 temporal latent，不是与 PhiZero 256 physics tokens 的公平对比。D20 保持真正的 16-transition
spatiotemporal physical tokenizer：每 transition 分别用 `1/4/16` 个 spatial groups，总计 `16/64/256`
physics tokens。三组固定同一 8-task x 8-episode future-feature cache、EMA-128、spatial-aligned routing、1000
steps、batch 32 和 D19 predictable-residual auxiliary target，只改变空间分组数。

为此将 `spatial_aligned` 从只允许 4x4 identity 扩展为确定性的 nested grid grouping：1 query 覆盖完整 4x4，
4 queries 各覆盖一个 2x2 quadrant，16 queries 各覆盖一个空间位置；encoder mask 为 `[Q,16]`，decoder 使用
严格转置 `[16,Q]`。16/64/256 的 shape、forward 和 backward smoke 均通过；已有 256 checkpoint 语义保持
identity 不变。`vlact` 环境已安装 `pytest 9.1.1`；tokenizer、structure analyzer、trainer、comparison 和 D2
gate 的定向回归共 `24 passed`，并另用直接 tensor smoke 覆盖 16/64/256 的 forward/backward。

D20 sidecar 从 D19 teacher 按 qualified key 精确抽取 14,077 clips，并按 decoder indices
`[2,4,...,30,31]` 对齐 16 个 future transitions；SHA256 为
`1a00629a913aaaf94d4f3229bc481cc8633add12eefa93024ffb9392f868ab6d`。固定 per-task validation 结果：

| physics tokens | macro quantized RMSE | macro continuous RMSE | macro action-aux loss | macro active ratio | macro perplexity |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.092089 | 0.091668 | 0.004179 | 0.8672 | 91.14 |
| 64 | 0.091095 | 0.090520 | 0.004255 | 0.9961 | 114.36 |
| 256 | **0.086395** | **0.084805** | 0.004192 | 1.0000 | 120.03 |

checkpoint SHA256 依次为 `145f454630d6dc78d0be8ef14458124942341648539f7c823e0c759c23cdb31e`、
`e14529aac37b6a3fb86c2fd75263d55709364d37446986bdd91322ae8512bc58`、
`06553cddfc91938b351b277654164bde66beaa470a41b03b0d61c56648d61978`。256 相对 64 的 macro quantized
RMSE 改善约 `5.16%`，64 相对 16 仅约 `1.08%`；action auxiliary 三者近似持平，所以 256 的优势来自更完整的
时空重建，而不是 residual target 泄漏。当前选择 **256 physics tokens** 进入下一步 token-structure gate 和
VLM predictability probe；16/64 保留为序列成本消融。仍不启动 DiT：必须先证明 VLM 能在保住 4 个 action
tokens 的前提下预测这 256 个 physics IDs，并且 predicted physics 相对 action-only 连续头有增益。

三组全 held-out token-structure 分析随后完成。16-token 单 query 没有“组内 pairwise/effective rank”的数学定义，
旧分析器因此发生除零；现将这两项明确输出为 JSON `null`，而不是伪造数值。16/64/256 的 structure report
SHA256 分别为 `65b181da79ca1cffac9753808e3ee786308b39a31c9fa90aa1c4f3a6d099cdfc`、
`bf9325fb864d2c0bf6ec6d422b2200b27d82935e1da4e79dfd3bea6bc39ebcfd`、
`46f22c07d097fce9203fde8f9c09ff1e91892e6101b1843f2cb4018895fc8347`。三组均为 128/128 global
active codes；global perplexity 为 `118.52/124.61/124.41`。64/256 的组内 unique ratio 为
`0.9509/0.9006`，pairwise token change 为 `0.9657/0.9853`；256 continuous latent 的组内相对离散度
`0.9176`、pairwise cosine `0.1020`、effective rank `3.546`，明确没有先前 global-QFormer 的 query 同质化。
因此 256 同时通过 reconstruction、code utilization 和 token-structure 三项表示门控；下一风险已经从 tokenizer
容量转移到 256-token VLM 可预测性与序列成本。

### 21.42 D21/D22 physics codebook 容量与数据丰富性交互（2026-09-27）

D20 的 256-token EMA-128 在 held-out 上 128/128 codes active、perplexity `120.03`，接近词表上限，因此值得
检查更大 codebook；但这不等于应直接升级，因为 256 个 VLM 输出位置的 vocabulary 从 128 增到 256 会额外
增加约 1 bit/token，并把随机准确率从 `0.781%` 降到 `0.391%`。D21 固定 D20 的 256 physics tokens、
spatial-aligned routing、cap 300/task、batch 32、1000 steps、EMA 和 predictable-residual auxiliary，只把
codebook 从 128 增到 256。K256 checkpoint/evaluation SHA256 分别为
`34ade8bf98a8c205082af9e52dea0f31f980c217a86372788d813bc074f8f064` 和
`da0db0b4b70aedd2bae320b6b9c431458a877e8f8e43408320c8525b4f606991`。

固定 per-task held-out macro 结果为：K128 quantized/continuous RMSE `0.086395/0.084805`，K256 为
`0.085930/0.084231`；最终 reconstruction 只改善 `0.539%`，低于预注册的 `1%` 实用阈值。更关键的是
quantization gap 从 `0.001591` 增至 `0.001699`，action auxiliary loss 从 `0.004192` 退化到 `0.004307`。
K256 batch-averaged active ratio 为 `1.0`、perplexity `235.23`，所以失败不是 code collapse，而是更大词表
没有把额外容量转化为足够的 held-out 收益。

为检验数据丰富性交互，D22 将每任务 train-clip cap 从 300 提到 1000，仍固定相同 8 tasks x 8 episodes、
1000 optimizer steps，其余单变量完全一致，并并行训练 K128/K256。结果为：

| train cap/task | codebook | macro quantized RMSE | continuous RMSE | quantization gap | action-aux loss | perplexity |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 128 | 0.086395 | 0.084805 | 0.001591 | 0.004192 | 120.03 |
| 300 | 256 | 0.085930 | 0.084231 | 0.001699 | 0.004307 | 235.23 |
| 1000 | 128 | 0.086378 | 0.084731 | 0.001648 | 0.004173 | 119.89 |
| 1000 | 256 | 0.086039 | 0.084543 | **0.001496** | 0.004209 | 235.62 |

cap1000 下 K256 相对 K128 的 quantized RMSE 只改善 `0.393%`，比 cap300 的 `0.539%` 更小；但其
quantization gap 相对 K128 缩小约 `9.21%`，说明数据丰富性确实会改善大码本的量化效率，只是当前规模与固定
训练预算尚不足以转化为最终收益。K128 cap1000 相对 cap300 几乎不变，K256 cap1000 反而略差于 cap300；原因
之一是固定 1000 steps 时更大的样本池降低了每个 clip 的重复训练次数。该实验增加的是同一 64 episodes 内的
clip 多样性，不是新的 episode/场景多样性，不能外推为“更多真实轨迹无效”。

D22 K128/K256 checkpoint SHA256 分别为
`0714f8089794bf1b9aa9bad9fdb9e7a26dadfb2af950e12a8a6452cb7687554e`、
`2ccfe06ea9e5bbc9f7ff741fa47fdd0f690b766f66459bb50b78609360accafa`；evaluation SHA256 分别为
`c99acb76ffab4c510386c0954358df950aed175fd2bd30298825b5f6b2ad9b7c`、
`4988069dc7d111e8099ca5731bd76218d73d24811af0dfbda55f17b22f5d548a`。当前主线保持 **256 physics
tokens x EMA-128 codes**。只有未来导出 40 train + 5 validation episodes/task 的 full spatiotemporal cache，
并按数据量同比增加 optimizer steps 后，才重新比较 K128/K256；在此前不启动 K512/16k/25k vocabulary。

### 21.43 D23 `4 action + 256 physics` 联合 VLM 低成本 gate（2026-09-28）

数据丰富性不仅影响 codebook 容量，也直接决定 physics-token predictor 能否学习跨场景规律。D20/D22 的
14,077 clips 虽比 cap300 更丰富，但仍只来自每任务固定 8 条轨迹（episode `0,1,2,3,4,6,7,8`），不能把
同一 episode 内更多时间点误写成更多场景。D23 因而明确定位为 idea viability probe：每任务用前 6 条
`0,1,2,3,4,6` 训练，用最后 2 条 `7,8` 做 physics held-out；若通过才导出正式 40 train + 5 validation
episodes/task cache。D17 action probe 曾见过这些 8 条轨迹，但从未见过 physics labels；所以 D23 能检查新增
physics supervision 的学习性和动作遗忘，不能作为最终跨场景泛化结果。

修复 `export_spatiotemporal_physics_token_ids.py`：多任务输入现在保留 `task_ids/task_names`，不再把不同任务的
相同 `(episode,base)` 静默碰撞。新增 `merge_multitask_dual_token_targets.py`，只按
`(task_id,episode_id,base_index)` 精确 join D16 oracle action IDs 与 D20 physics IDs，并拒绝重复、缺失或任务顺序
不一致。输出 `artifacts/token_caches/robotwin_d23_8tasks_dual_action4_physics256.npz` 的 action/physics shape 为
`[14077,4] / [14077,256]`；physics 范围 `0..127`、128/128 codes active、global perplexity `117.008`。
physics-only cache SHA256 为 `647b8223db694f44dd3f820867c699ccc498b244e90ba3e72294fb8c80061801`，dual cache
SHA256 为 `1cd7a6d6e71d0c70916e90278f5601240af01bba1244ecd271d10e794eef70e6`，manifest SHA256 为
`187e92ac44439541614c4d7519cb9550853df8ed57200b8fbd1704c548b6df02`。

D23 复用 D12 action-teacher distillation，但修正一个容易被忽略的初始化偏差：dual probe prompt 与 D17 原 prompt
保持完全相同 `Predict the next robot action tokens.`，这样加载 D17 的最后两层与 action head 后，step-zero action
logits 才严格一致。训练 loss 固定为 physics CE + `10 x` action-logit KL，action target CE 仅记录不反传；physics
head 从同一个 pooled VLM state 和 soft action-token context 预测 256x128 logits。500-step 低成本训练固定 8 卡、
global batch 8、最后两层 LR `2e-6`、physics head LR `1e-4`，不在本轮引入 DiT。

新增 `evaluate_multitask_dual_token_probe.py`，在每任务 episode 7/8 的等量 panel 同时报告：（1）dual action 与
D17 action 的 token agreement/decoded MAE；（2）physics token accuracy、active codes、perplexity；（3）D20
action-aux decoder 将 predicted physics IDs 解码为 16 个 transition residual 后，相对同 run action-only 和
task-local shuffled predicted physics 的连续动作收益。正式 gate 要求 action MAE 相对 D17 退化不超过 1%、physics
accuracy 高于 `1/128` chance、predicted physics 相对 action-only 与 shuffled 均至少改善 1%。四项必须同时通过，
否则不启动 DiT。

2-step 端到端 smoke 已通过数据解码、8 卡 ZeRO-2 前后向、checkpoint 重载和 evaluator。联合模型含冻结 teacher
共 8.930B 参数，可训练 242.315M；叠加 node004 原有约 16-18 GB/卡任务后，总占用约 40 GB/卡，仍余约 40 GB，
没有 OOM。首步 physics CE `4.8732`（随机初始化参考 `ln(128)=4.8520`），action distill loss 约 `0`；2-step
评估 action prediction 与 D17 agreement 为 `1.0`。该 smoke 的每任务 1 sample 和随机 physics head 不用于效果判断。
定向语法、cache、tokenizer 回归为 `15 passed`。500-step 正式 probe 已启动，结果待补。

D23 正式训练已完成，checkpoint 为
`results/Checkpoints/qwen_dual_token_probe_d23_action4_physics256_8tasks_seed42_steps500/checkpoints/steps_500_pytorch_model.pt`，
SHA256 `9f6871d6af32b5d9f863537deb92528eaf0b79446fedd198db09bc60ae5cba02`。8 卡 global batch 8 全程无
OOM。修正 evaluator 使 oracle/predicted physics 均通过同一 residual decoder 后，正式 256-sample panel 报告为
`artifacts/evaluations/d23_multitask_dual_action4_physics256_steps500_v2_oracle.json`，SHA256
`3627dfccd7649cd2055cc93572ae3afa12a65572cb1782d12ddb2887c5e93068`。physics token accuracy
`7.6797%`，约为 chance `0.78125%` 的 9.83 倍；prediction 使用 128/128 codes，perplexity `69.86`。
dual action 与 D17 agreement `98.34%`，D17/dual action MAE 为 `0.103570/0.104200`，退化 `0.607%`，
动作保护 gate 通过。

但连续动作 utility gate 失败：同一 dual action base 上，action-only sampled MAE `0.104179`；加 predicted physics
后为 `0.107632`（退化 `3.315%`），加 oracle physics 后仍为 `0.107547`（退化 `3.233%`）；predicted 相对
task-local shuffled 只改善 `0.071%`。这说明当前首要瓶颈不是 VLM 是否能把 physics accuracy 从 7.7% 继续提高，
而是 D20 auxiliary decoder 连 oracle IDs 都不能转化为动作增益。D23 总 gate 失败，明确不启动 DiT。

### 21.44 D24 冻结 physics tokenizer 的 residual decoder 重训（2026-09-28）

先修复 `diagnose_action_aux_token_correspondence.py` 的两个口径问题：`action_token_residual` target 不再被误当作
absolute action；shuffle 改为 task-and-episode-local，并增加 zero-residual baseline。D20 原 decoder 在 3,244 个
episode 7/8 held-out clips 上 oracle aligned joint MAE `0.054491`，zero 为 `0.053501`，即 oracle 反而差
`1.85%`；velocity MAE 比 zero 差 `50.9%`，aligned 相对 shuffled 只改善 `0.115%`。报告为
`artifacts/evaluations/d23_d20_aux_decoder_ep7_8_diagnostic.json`。

D24 冻结 encoder、EMA codebook 与全部 token IDs，仅用 8,785 个完整 train clips、task-balanced batch 512 重训
`action_aux_head` 3000 steps，并按 held-out macro loss 保存最佳 step 1300。checkpoint SHA256 为
`8eff385cbc8175dc5e7df8fd03b8008a96634a9518e94fe8844377e1e561f46f`，diagnostic SHA256 为
`2255c947ed738451fab450c9ca9cceff90c1753c4f472c6eef566760a72a49a2`。aligned joint MAE 降到
`0.053758`，但仍比 zero `0.053501` 差 `0.482%`；相对 shuffled `0.053813` 只改善 `0.102%`，velocity
仍比 zero 差 `21.35%`。虽然 joint SmoothL1 相对 zero 改善 `2.88%`，主 MAE 与对应性 gate 仍失败，不能通过
改 loss 口径宣布成功。

### 21.45 D25 position-aware decoder 与 D26 action-conditioned gate（2026-09-28）

D20/D24 decoder 会对每个 transition 的 16 个 spatial latents 直接求均值，可能丢失位置关系。新增向后兼容的
`action_aux_pooling={mean,flatten}`：旧 checkpoint 缺字段时仍严格按 `mean` 加载；`flatten` 按固定 4x4 空间顺序
输入 MLP。训练、diagnostic 和 D23 evaluator 统一调用 `predict_action_aux`，防止训练/评测 pooling 语义分叉。
定向测试 `21 passed`，D25 checkpoint SHA256 为
`41aede7ba021fa790c8452294edf1960d3045474469abf0628241803ca355e15`。其最佳状态很早出现且明显过拟合：正式
diagnostic aligned/shuffled/zero joint MAE 为 `0.057467/0.057979/0.053501`；aligned 相对 shuffled 改善
`0.883%`，但相对 zero 退化 `7.41%`，position flatten gate 失败。因此不继续增加 decoder 宽度或训练步数。

上述失败还暴露出条件不一致：通过的 D18 continuous upper bound 使用 visual + task + D17 predicted action 条件，
而 D20-D25 auxiliary decoder 只看 physics IDs；目标 DiT 则会同时收到 action/physics tokens。D26 因而做更公平的
oracle 增量 gate：精确 join D17 predicted action IDs，解码为 16 个对应 transition 的 base action；同数据、同 split
并行训练 `action-only` 与 `action + mean-pooled oracle physics latent` 两个 head。正式判据是 aligned physics 同时优于
各自最佳 action-only 与 task-and-episode-local shuffled physics，而不只是优于 zero residual。D26 已在 node004
GPU 5 启动 3000-step run；若仍失败，停止 auxiliary MLP 搜索，回到 tokenizer supervision/representation 设计，
不启动 DiT。

D26 已完成，best step 2000；checkpoint/report SHA256 分别为
`b1d3f1780cee48865e225ac402417d9c7eebf91a74f0d8a4bd8c96aa03a38e17` 与
`8f3d9956e9d07f07936f5caa0e74391885f7976eb407a1ec9338ddbd93fd219b`。在全部 3,244 个 held-out clips
上，aligned/action-only/episode-local shuffled/zero joint MAE 为
`0.047827/0.048754/0.048163/0.056979`：aligned 相对 action-only 改善 `1.902%`，相对 shuffled 改善
`0.698%`；macro loss 对二者分别改善 `3.455%/1.426%`。逐任务 aligned 相对 action-only 为 6/8 改善，
但 shake 与 pick-dual 分别退化 `0.29%/1.79%`。因此 action condition 解决了“oracle IDs 完全无效”的主要口径
问题，但 aligned-vs-shuffled 的 MAE 分离仍低于 1%，只能判为弱正信号，不能据此启动 DiT。

随后在原 D23 固定 256-sample panel 上接入同一 D26 head。报告
`artifacts/evaluations/d26_d23_conditioned_dual_probe_steps500.json` SHA256 为
`cf005f8177aa8c39f9340d6817004c92cd75180e40e2b2319025e42cda817039`。conditioned action-only/oracle/
predicted/shuffled-predicted MAE 为 `0.110737/0.109370/0.110055/0.110389`：oracle 相对 action-only 改善
`1.234%`，确认表示有可用上限；predicted 只改善 `0.616%`，相对 shuffled 只改善 `0.303%`，双 1% gate
失败。当前瓶颈因此转回 VLM physics prediction，而不是继续搜索 auxiliary decoder。

D27 固定 D23 数据、256x128 physics targets、模型结构、action distillation 和所有 optimizer 配置，只把训练从
500 延长到 1000 steps，检查低成本 under-training 假设；8 卡 run 已在 node004 启动。若 token accuracy 和
conditioned predicted utility 没有同步提升，则不继续在 8-episode cache 上堆步数，下一步应导出更多 episode/场景
的正式 cache 后再训练 predictor。

D27 已完成。1000-step checkpoint 为
`results/Checkpoints/qwen_dual_token_probe_d23_action4_physics256_8tasks_seed42_steps1000/checkpoints/steps_1000_pytorch_model.pt`，
SHA256 `d85f50110983766fa7c4890495d7478eac59d778007fc8bf01d91445bc82203c`。标准评估与 conditioned 评估
SHA256 分别为 `1e2c3abd2ea6bacc8af87e1d7d790464564cf46602cfbe86b86ecaed5d89c455`、
`61aa774679b5563426bf29d3c57a468f0686b46f4340fdda5215f5a011dc49ac`。相对 500 steps，physics token
accuracy 从 `7.680%` 降至 `7.118%`，prediction perplexity 从 `69.86` 升至 `91.83`；action MAE 退化从
`0.607%` 缩小至 `0.184%`。conditioned oracle 相对 action-only 改善 `1.203%`，predicted 改善
`0.662%`，predicted 相对 shuffled 只改善 `0.269%`。因此延长训练没有提高 physics accuracy 或连续动作
utility，predicted 双 1% gate 仍失败。结论是停止在同一 8 episodes 上继续堆训练步数；瓶颈是 episode/场景
多样性，而非单纯 under-training。

### 21.46 D28 全 episode 时空 cache 与正式多样性实验（2026-09-28）

D28 将每任务数据从 8 episodes 扩展到全部 50 episodes，固定 8 个任务、Qwen3-VL 预训练视觉编码器、4x4
空间网格、17 个未来帧 offsets `0,2,...,32` 和 fp16 cache；不在导出阶段改变 tokenizer 或 VLM 结构。8 个任务
各占一张 GPU 独立导出 part，part 与 manifest 同时存在才视为完成并可恢复跳过；全部 part 成功后才做严格
多任务合并。预期输出为
`artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz`，并与
`artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz` 的 qualified keys 精确核对。

node004 启动前 GPU 0/1/2 分别已有约 `16.6/16.8/15.3 GiB` 占用，仍各余约 `64 GiB`；GPU 3--7 空闲。
runner 已通过 `bash -n`，默认 batch 32。共享盘约余 560 GiB，足够容纳预计 6--7 GB 的 merged cache；节点
`/tmp/vlact_qwen3_pretrain_steps100000.pt` 已存在，因此导出不重复从共享盘加载 11 GB checkpoint。导出后必须
验证 8 tasks x 50 episodes、数组 shape/finite、key 完整性和 SHA256，验证未通过不得进入 tokenizer 训练。

D28 后续严格顺序为：（1）用 D19 all-sample predictable-residual teacher 生成 full-data sidecar；（2）固定
episode-level train/validation/test split，禁止同 episode 泄漏；（3）按有效样本量而不是沿用 1000 steps 训练
256x128 EMA tokenizer，并在同预算下重新判断 K256 是否值得；（4）导出 full oracle physics IDs 并训练扩大数据的
`4 action + 256 physics` VLM predictor；（5）复用 D26 action-conditioned evaluator。只有 predicted physics
相对 action-only 和 task-and-episode-local shuffled 的 held-out MAE 都至少改善 1%，且 action protection 通过，
才启动 DiT。

D28 8-task runner 已在 node004 启动，进程号与逐任务日志记录于
`artifacts/logs/d28_full_export_runner.log` 和 `artifacts/logs/d28_spatiotemporal_export/`。8 个视觉编码器加载后
每卡新增约 5.9--12 GiB，GPU 0--2 即使与已有任务共存仍有充足余量，未发生 OOM；首轮检查已有 7 个任务进入
episode 编码，`open_microwave` 仍在处理首条长视频，所有 8 个进程均存活。

新增 `validate_d28_full_feature_cache.py`，独立检查 finite fp16 features、`[F,16,D]` shape、17 offsets、无重复
qualified keys、每任务精确 episodes `0..49`、与 D15 action cache 逐行相等，以及固定 validation episodes
`5,20,25,39,49` 的无泄漏 split；报告将写入
`artifacts/evaluations/d28_full_feature_cache_validation.json`。同时修复 predictable-residual sidecar 导出器：过去会
把 template 中所有数组（包括数 GB 的 `frame_features`）再次复制到 sidecar；现在只保留 task/episode/base keys、
task names 和 offsets，再附加 action targets。相关 validator、sidecar 与 tokenizer tests 为 `8 passed`。独立 watcher
只在 D28 merged cache 成功生成后运行 validator 和 full sidecar，失败时不会继续训练。

### 21.47 D29 full-diversity K128/K256 预注册（2026-09-28）

D29 用完整 D28 cache 回答“更大 codebook 是否只有在更多 episode/场景下才有收益”。K128/K256 固定相同
256 physics tokens、EMA、spatial-aligned routing、D19 predictable-residual auxiliary、全部训练 clips 和固定
episode validation split，仅改变 codebook size。每组单 GPU batch 32、8000 optimizer steps、每 500 steps
评估、每 2000 steps保存；两组并行使用 GPU 3/4。选择单 GPU 是为了避免每个 DDP rank 都解压并常驻一份
full feature cache；8000 steps 约对应完整训练集数个 epoch，明显高于 D20/D22 的固定 1000-step 欠覆盖风险。

K256 只有在 held-out quantized RMSE 相对 K128 至少改善 1%，且 action auxiliary 不退化、code utilization 健康时
才晋升；否则主线继续使用 K128，因为 VLM 对 256 个位置预测 K256 会承担更低 chance accuracy 与额外 1 bit/token
的成本。D29 仍只是 tokenizer 容量选择，不改变最终 predicted-physics 双 1% gate，也不提前启动 DiT。

D29 增加机器可执行 `require_d29_codebook_gate.py`：除 quantized RMSE 至少改善 1% 外，还要求 continuous RMSE
不退化超过 0.5%、action auxiliary loss 不退化、K256 active ratio 至少 0.9、perplexity 至少 128；任一失败就
选择 K128。训练完成后两个 checkpoint 会在相同 held-out episodes 上独立做逐任务评估，再生成
`artifacts/evaluations/d29_full_diversity_codebook_gate.json`，其中记录 selected checkpoint 及 SHA256。训练、评估、
选型均已挂成依赖前一阶段成功产物的可恢复 watcher；相关新增 gate tests 通过。

### 21.48 D30 新 tokenizer 的 action-conditioned oracle 上限（2026-09-28）

D26 conditioned head 只适用于旧 D20 tokenizer latent，不能复用来评估 D29 新码本。D30 读取 D29 自动选择结果，
用 selected checkpoint 对 D28 全量 cache 导出 256 oracle physics IDs，再与 D16 全量 4-token oracle action cache
按 `(task_id,episode_id,base_index)` 精确合并。随后在新 tokenizer latent 上重训 action-conditioned physics residual
head 与 action-only head，固定 validation episodes `5,20,25,39,49`、3000 steps、batch 512，并保留
task-and-episode-local shuffled control。

D30 oracle gate 以 held-out macro joint MAE 为唯一决策指标，要求 aligned oracle physics 相对 action-only 和
episode-local shuffled 都至少改善 1%。只有双条件通过才启动 full-data VLM predictor；若失败，说明即使 oracle IDs
在目标 action+physics 条件下也没有足够增量上限，应回到 supervision/representation，而不是用 4B VLM 训练掩盖
问题。D30 target export、conditioned probe 和 gate 已挂在 D29 evaluation 后自动运行；新增测试当前通过。

### 21.49 D31 full-diversity VLM staged probe 预注册（2026-09-28）

D31 只在 D30 oracle 双 gate 通过时启动。运行时配置由 D29 selection report 结构化生成：physics vocabulary 自动取
K128/K256 赢家，token cache 指向 D30 full dual targets，训练排除 validation episodes `5,20,25,39,49`，其余
45 episodes/task 可用于训练；不再沿用 D23 的 6-train/2-validation 小 cache split。action token 仍为 4x128，
physics token 仍为 256 个，D17 action initialization、action-logit distillation weight 10、最后两层 VLM 与 physics
head 的学习率均保持 D23 不变。

第一阶段固定 8 GPU、global batch 8、2000 steps，定位为低成本 diversity probe，而不是直接跑完约 8500 steps 的
一整个 full-data epoch。正式评估使用每任务 64 个 held-out samples、新 D30 conditioned head、D17 action baseline，
并单独生成机器 gate：action MAE 退化不超过 1%、physics accuracy 高于对应 vocabulary chance、conditioned
predicted physics 相对 action-only 与 episode-local shuffled 都至少改善 1%。四项全部通过才令
`dit_authorized=true`；否则仍不启动 DiT。若 2000-step 只显示 accuracy/utility 明确上升但未过线，再预注册延长
步数；不能仅凭训练 loss 决定续训。D31 watcher 已排在 D30 后，oracle gate 失败会以明确状态退出。

### 21.50 D28 完成与恢复性验证（2026-09-28）

D28 首轮 8-task 并行导出中，`open_microwave` 单 episode 为约 400--700 frames，成为两小时量级的串行尾部；其余
7 个任务已经完整落盘后，将该任务改为 6 个互斥 episode shards（`9+9+8+8+8+8=50`），使用 GPU 2--7
并行导出。新增 exact `--include-episode-ids`、单任务 shard merger、无重复 key 和 contract 校验；其定向 exporter
tests 为 `8 passed`。6 shards 合并为标准单任务 part 后，可恢复主 runner 跳过其余 7 个完整 parts，再执行原有
严格多任务 merge。该优化只改变导出调度，不改变 image encoder、offsets、spatial grid、dtype 或样本顺序。

D28 merged cache
`artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz` 已完成，约 6.1 GB，
cache SHA256 `386b2154dd7c1bbdf6be253bcbfd7a66ab2894aee62598f75325f240f3f3d519`，manifest SHA256
`c1ad30fa076931280271873d5c627e87c6438ea861c9336d4f2688693a811ef6`。独立 validator 报告 SHA256
`da429278c5df80b24ed3255901127890592a6012567d844c775d34b050ce9b61`，确认：

- 8 tasks，每任务精确 50 episodes，合计 85,098 samples；
- `frame_features` shape `[85098,16,2560]`，fp16 且全部 finite；
- 72,298 complete clips，其中固定 episodes `5,20,25,39,49` 给出 7,709 validation clips，其余 64,589
  train complete clips；
- qualified keys 无重复，并与 D15 all-episode action cache 逐行完全一致。

full predictable-residual sidecar shape `[85098,17,14]`，仅约 66 MB；SHA256
`e7d972d4506d6adc7544e5f32f112fdaee2c148567e8ec2030245853da5240b9`，manifest SHA256
`3baca08e488330e6506bd623f7113647b0ca66d7ed813f955f65f82ec6031665`。D29 K128/K256 已在 GPU 3/4
并行启动；每组 8000x32 约看 256k clips，相当于约 3.96 个完整 train-set epochs，不再是 D22 固定 1000 steps
导致的数据覆盖不足。

D29 正式训练与逐任务评估已完成，codebook gate **全部通过**，主线从 EMA-K128 升级为
**256 physics tokens x EMA-K256**。K128/K256 checkpoint SHA256 分别为
`6e2b0e39ecb7d6ba9422a7bcff7dcf7c71e78f39459e8f278c53a38ec8c9b9f3`、
`8625b3d7580625b1a4a33bce874537a2863cd18ddfab0e595fa20537a8f76b5c`；evaluation SHA256 分别为
`688896958bf1e045990113c500a9e965650f78eb4dfd57a2ae65edd16defabb1`、
`a58c34d937b8bf28a4bd6ef639a5170de30c0aa42df976e83f5315886d8a5e6d`，gate report SHA256 为
`a9f133e8b385cfba3c31484a83904a6f8c49229b69ab0f8cdf92b9a882da00cf`。

固定 held-out per-task macro 结果：K128/K256 quantized RMSE 为 `0.084234/0.083225`，K256 改善
`1.198%`；continuous RMSE 为 `0.077526/0.075751`，改善 `2.290%`；action auxiliary loss 为
`0.004794/0.004709`，改善 `1.778%`。K256 active ratio `0.97998`、perplexity `206.77`，容量利用健康。
这与 8-episode D21/D22 的 K256 改善仅 `0.39--0.54%` 形成明确对照：扩大同 episode 内 clip 数量不足以释放
大码本收益，而从 8 扩到 50 episodes/task 并同步增加训练覆盖后，额外容量才转化为超过 1% 的 held-out 重建收益。
因此“codebook 大小与训练数据丰富性有关”得到当前数据支持；后续 VLM physics vocabulary 相应使用 256，chance
accuracy 为 `1/256=0.390625%`，不再沿用 D23 的 K128。

D30 full oracle physics/dual cache SHA256 分别为
`fd00a355c591b866002e81b2625e40a519484f72e927a477688a10e50915f6f7`、
`b5f114936e0c97db9eb95b6b22af31d578fd6ff523bcfbf1f95a75dad4d91212`。在新 K256 latent 上重训的
conditioned checkpoint SHA256 为 `75437cbc846a4b6cebbea38cb737c1d3c2800018c26a1c17ecb1bdefd7dd273b`，
report SHA256 `3a7794afc85267d1b1723dccfea407d89e03074d8b3b32821d98caf2179296c9`，gate report SHA256
`3132a2e3167e007af4e280d85e28031b495ca436321d331c6e7a0586d110779c`。最佳 held-out aligned
oracle physics 相对 action-only macro joint MAE 改善 `1.350%`，相对 task-and-episode-local shuffled 改善
`1.979%`，D30 双 gate 正式通过。相较旧 8-episode D26 的 aligned-vs-shuffled `0.698%`，full diversity +
K256 显著加强了 ID 对动作残差信号的对应性，因此有充分 oracle 上限进入 predicted-token 阶段。

D31 已据此自动启动，rendered config 明确使用 physics vocabulary 256、physics token count 256、action tokens 4x128，
并排除固定五个 validation episodes；OmegaConf 重载检查确认 prompt 与 D17/D23 的原字符串逐字符一致，避免破坏
step-zero action logits。训练使用 node004 本地预训练 checkpoint 副本，8 卡、2000 steps，结果待补。

### 21.51 D31 full-diversity VLM probe 结果与 D31b 评估扩样（2026-09-28）

D31 2000-step 训练完成且全程无 OOM，checkpoint SHA256 为
`b866b0324ae00ec8d79250f3db97de92854ae52f4e52d9b13fdd3f7fd585c15a`。在正式评估启动前修复了两个只影响
评估契约、不影响训练的缺口：（1）shuffled physics 从 task-local 全局置换改为预注册的
task-and-episode-local 非恒等 cyclic shuffle；（2）evaluation JSON 补齐 action/physics vocabulary 与 token count，
gate 同时拒绝错误 shuffle scope。相关新增测试与 gate/DiT 条件注入回归均通过。

固定 64 samples/task、共 512 个 held-out samples 的 evaluation SHA256 为
`bb487b7a3be0203e578052cc4689fc782c804661ccf4844655dffe4e4b0532ba`，gate SHA256 为
`8c32cc5fd12e81dff7d95150dde37e5d710efbc9f521c33754a20d0440ea5840`。D31 action MAE 相对 D17 改善
`0.858%`，动作保护通过；physics token accuracy 为 `10.622%`，是 K256 chance `0.390625%` 的 27.19 倍，预测
使用 256/256 codes、perplexity `79.99`。但 D30 conditioned head 上 predicted physics 相对 action-only
**退化 `0.540%`**，相对 episode-local shuffled 仅改善 `0.333%`，预测 utility 双 1% gate 均失败，故
`dit_authorized=false`，不启动 DiT。

该 512-sample panel 同时暴露出评估方差问题：conditioned oracle physics 相对 action-only 也退化 `0.030%`，而
D30 在全部 7,709 个 held-out clips 上的同口径 oracle 增益为 `+1.350%`。当正控制在小 panel 上都无法复现时，不能
把 predicted utility 的约 0.3--0.5% 差异解释为可靠效应，也不能据此直接延长训练。D31b 因而预注册为**只扩评估、
不改 checkpoint**：固定同一 seed、五个 validation episodes、episode-local cyclic shuffle 和所有 decoder，扩大到
256 samples/task（共 2,048）。先检查 oracle 正控制是否恢复为正且接近 D30 方向，再检查 predicted 相对 action-only/
shuffled 的方向与置信稳定性。D31b 仍使用原 1% 双门槛；若 oracle 仍不恢复，优先改为全 validation 的 cached/batched
评估或报告 bootstrap 区间，而不是训练更多 steps。只有 oracle 恢复且 predicted utility 相对 D27 显示清晰上升趋势，
才预注册下一段训练；physics accuracy 单独上升不足以授权续训或 DiT。

D31b 按预注册完成 256 samples/task、共 2,048 samples 的扩样评估。evaluation/gate SHA256 分别为
`00675c8a57af13959371e388f70b29d07ac92464eabd6eb91fe04cb092fa2e17` 与
`febd7802c675af47a586994208f8bd71bc1a31e93b9db59d6c4553099d012f77`。physics accuracy 稳定为
`10.832%`，action MAE 相对 D17 仅退化 `0.056%`，两项基础 gate 通过；但 conditioned predicted physics
相对 action-only 退化 `0.642%`，相对 episode-local shuffled 只改善 `0.451%`，双 utility gate 仍失败。
oracle 相对 action-only 仍为 `-0.021%`，未恢复 D30 全 validation 的正控制，因此不延长 VLM steps、不启动 DiT。

进一步检查发现一个明确的条件分布偏移：D30 conditioned head 的 action 输入来自冻结 D17 predicted action cache，
而 D31/D31b utility evaluator 使用 dual VLM 自己的 action prediction；两者虽有约 `96.6%` token agreement，少量 ID
变化仍可能使只在 D17 action 分布上训练的 residual head 失准。D31c 固定 D31b 的 checkpoint、2,048-sample panel、
seed、shuffle 与所有权重，仅新增并行的 `D17 action + predicted/oracle/shuffled physics` 指标；原
`D31 action + physics` 指标和机器 gate 保持不变。若 D17-reference oracle 恢复而 D31 oracle 不恢复，下一步应先用
D31 joint predicted action cache 重训/校准 conditioned head；若两者都不恢复，则进入 full-validation cached evaluation
与 bootstrap，不用更多 VLM steps 掩盖评估口径问题。

D31c 同 panel 诊断完成，evaluation/gate SHA256 分别为
`57f9eb493c58cf086654ffc9f1e1811f4e62dd91f89af2789946eb9d416286d5` 与
`a8c9d4e8471a499b7bbe045dcf2f2b8fee454af54c1da9e03d7f85efe26ac485`。D17-reference 与 D31-action
conditioned oracle 增益分别为 `-0.020%/-0.021%`，predicted-vs-action-only 为 `-0.667%/-0.642%`，
predicted-vs-shuffled 为 `+0.438%/+0.451%`；两条路径几乎相同，排除了 D31 相对 D17 的少量 action-token
变化是 oracle 正控制消失的主因。

继续审计 D30 runner 后发现真正的条件分布错误：`run_d30_selected_oracle_conditioned_gate.sh` 将
`robotwin_d16_8tasks_oracle_shared_ema4x128.npz` 同时用于 dual VLM 的监督 target 和 conditioned-head 的 action
输入。前者正确，但后者与文档预注册的 D17 predicted action condition 不符。D16 oracle 与 D17 predicted cache 的
qualified keys 完全一致，但逐 token agreement 仅 `40.420%`、4-token sequence agreement 仅 `11.187%`；因此 D30
head 实际在严重不同的 action 分布上训练，不能作为 D31 utility 的可靠 decoder。D30 原结果保留用于审计，但其
“D17 predicted action conditioned”解释撤回。

D32 是该问题的严格单变量修复：physics cache/tokenizer、predictable-residual targets、完整 episode split、head
结构、batch 512、3000 steps 与 D30 全部固定，只将 conditioned-head action cache 改为真正的
`robotwin_d17_8tasks_predicted_shared_ema4x128_steps1000.npz`；dual VLM 监督 cache 不改。先复用 D30 的 oracle
joint-MAE 双 1% gate。若通过，再在相同 D31b panel 上替换为 D32 head，要求 oracle 正控制先恢复，然后才解释
predicted physics；若失败，则不再把 D30 gate 当作支持 predicted-token DiT 的上限证据。

D32 完成，best step 为 2700；checkpoint/report/gate SHA256 分别为
`1dd7d2353bf0e6a19bc21e8deeac7be27f9a5948b0e7f228704bc67ab158d5ba`、
`814ce30c4a95ea28021db8a3c8444d4c875963e939b85a210e40aff649308978`、
`e5fda03685b0cb28b0a1fcd98e53006f04886494bfefb52cfa2d78e685f50616`。在全部 7,709 个 held-out
clips 上，aligned/action-only/episode-local shuffled macro joint MAE 为
`0.050466/0.051174/0.051704`：oracle physics 相对真正的 D17-predicted-action-only 改善 `1.385%`，相对
shuffled 改善 `2.394%`，修正后的双 gate 通过。该结果说明 K256 physics 表示仍有可靠增量上限，D31b/c 的 oracle
正控制失败主要来自 D30 conditioned head 训练时用错 action-cache 分布。

D33 固定 D31b/c 的 VLM checkpoint、2,048-sample panel、seed、episode-local shuffle、tokenizers 与所有评估口径，
唯一替换为 D32 corrected conditioned head。正式四项 gate 仍要求 action 退化不超过 1%、physics accuracy 高于
chance、predicted physics 相对 action-only 与 shuffled 都至少改善 1%。D32 oracle 正控制也必须在该 panel 上恢复；
若 predicted 双 gate 仍失败，则不启动 DiT，并根据相对 D27 的 utility 趋势决定是先导出全 validation prediction cache
还是停止 VLM step 扩展。训练 loss 或 token accuracy 单独不构成续训依据。

D33 使用 corrected D32 head 完成同一 2,048-sample panel，evaluation/gate SHA256 分别为
`25f5c8ac537f0a731934b19c7d820372dd893844a73c61c536ecd0fa424fd80c` 与
`66c143244a10e0d74d87a531da11a6b3b15ccd184e6b47c1cb800752773146b3`。oracle physics 在真实 action MAE
上相对 action-only 恢复为 `+0.139%`，但远低于 D32 在 predictable-residual teacher target 上的 `+1.385%`；
predicted physics 相对 action-only 退化 `1.279%`，相对 shuffled 改善 `0.612%`，双 gate 失败，仍不启动 DiT。

这揭示第二个目标错配：D29 tokenizer 和 D32 conditioned head 的辅助监督是 D19 observation-predictable residual
teacher，而正式 utility 是真实机器人动作。teacher target 可用于约束 physics token 可预测性，但在 teacher residual
上的 head gate 不能替代真实动作 gate。D34 固定 D29 K256 tokenizer、D17 predicted action condition、head 结构与
训练预算，只把 head target 改为 held-out-safe 的真实 residual：RoboTwin absolute action 减去 D17 action-tokenizer
decoded action，12 个关节维做角度 wrap、2 个 gripper 维做直接差，使用相同 16 个 decoder transitions 和 episode
split。D34 首先要求 oracle physics 在真实 residual joint MAE 上同时优于 action-only 与 episode-local shuffled 1%；
若不通过，停止 residual-head/DiT 注入；若通过，再用同一 D31 checkpoint 做 D35 predicted-token gate。

D34 完成。真实 residual sidecar/manifest SHA256 分别为
`7b40369ea54d382d1b5d146d0b50bb87c1222a83516c52d45445c4600ac785dc`、
`81f0ee0c0342aee9ff6c3b38a052a9df7119445436e6379d988e8f3a5ce0d922`；conditioned checkpoint/report/gate
SHA256 分别为 `bd1df2261e22259b38d153a6146174baadf7b7940f5b4acea49bb6497b275f1b`、
`2919042102072531b3ea55d718d8c8034aa7d8e5dcd3dbd086bde8520f3aff56`、
`5af2037e80478a69f103b2df96d21ddad6e0c7b0cc522d42ed3cc457f12bfdfa`。best step 3000；在 7,709 个
held-out clips 上 aligned/action-only/episode-local shuffled macro joint MAE 为
`0.131522/0.134374/0.139633`，oracle physics 相对 action-only 改善 `2.123%`，相对 shuffled 改善
`5.809%`，真实动作双 gate 明确通过。这证明当前 K256 physics tokens 对真实连续动作确有增量上限。

D35 固定 D31 的 2000-step VLM checkpoint、D31b 的 2,048-sample panel、D29 tokenizer 与全部 shuffle/metric
口径，唯一替换为 D34 actual-residual conditioned head。正式四项 gate 不变；只有 action protection、physics
accuracy、predicted-vs-action-only 和 predicted-vs-shuffled 全部通过才授权 DiT。若只通过 shuffle 而未通过
action-only，则说明 predicted IDs 携带对应性但误差仍过大，下一步可预注册 VLM step extension；若两项 utility
均无改善，则停止增加 VLM steps。

D35 完成，evaluation/gate SHA256 分别为
`9635c27f11471f3de652d51a5602a42b387e83d45f293d84398de5f3e95c96c7` 与
`991d20d4c7e022fd2f7338ac3d84f605ebeea8c3de3594461dfb2da4aa1e82e3`。D34 head 在该 panel 上的 oracle
physics 相对 action-only action MAE 改善 `0.847%`；predicted physics 相对 action-only 退化 `3.356%`，但相对
episode-local shuffled 改善 `3.093%`，后者明确超过 1% 且远高于 D27 的 `0.269%`。因此 2000-step predictor
已经学到与动作相关的 token 对应关系，但错误 token 的残差幅度仍使其差于不用 physics；四项 gate 为 3/4，
`dit_authorized=false`。

D36 按预注册的“utility 明确上升才扩 steps”分支执行。D31 的 global batch 为 8，2000 steps 只看约 16k samples，
约为 64,589 个 train complete clips 的四分之一；D36 固定同一模型、D17 initialization、distillation weight 10、
K256 targets、45 train episodes/task 与全部 optimizer 参数，只把训练预算设为 8000 steps，使采样量约覆盖一个完整
train-set epoch。D36 从相同原始初始化重新训练，不续接 D31 已衰减到 `4e-8` 的 cosine schedule。评估固定 D34
actual-residual head、validation episodes `5,20,25,39,49`、256 samples/task 和 episode-local cyclic shuffle。
仍要求四项 gate 全过才启动 DiT；若 predicted-vs-action-only 仍未转正，停止单纯 step scaling，转向 token-confidence
gating/soft latent 或降低 residual scale，并始终保留 action-only/shuffled 对照。

D36 已在 node004 八卡启动，run ID 为
`qwen_dual_token_probe_d31_full_diversity_k256_steps8000`，日志为
`artifacts/logs/d36_full_epoch_dual_token_probe.log`。初始化确认 360 个 train episodes、76,109 dataset steps、D17
action initialization 和本地预训练 checkpoint 均正确载入；主循环约 `2.45 step/s`，八卡无 OOM。runner 将在
8000-step checkpoint 保存后自动执行 D34 actual-residual head 的 256 samples/task 评估与四项 gate，不会在 gate
前启动 DiT。另将 conditioned-head trainer 改为只读取 feature cache 的 keys/valid mask，不再为一个 mask 解压并常驻
6.1 GB `frame_features`；actual-residual/exporter 与 selective-loader 定向测试为 `4 passed`，DiT 注入与 D31 gate
相关回归此前为 `40 passed`。

D36 训练、自动评估与 gate 均已完成，八卡释放。checkpoint/evaluation/gate SHA256 分别为
`a6f18b13e42f31546cc4bd0cff6add860fb5be1f48a066723acb1dda99d27368`、
`9574c100382ffd535d2bc7bda94a56bb916ae4fbd4f502053140becb6f0a73ab`、
`cc511398de476a7a118e7467051b0181f6c69d18c28e7f70fc3b0b5b812d324c`。physics accuracy 从 D35 的
`10.832%` 升至 `11.331%`，prediction perplexity 从 `75.73` 升至 `110.64`，256/256 codes active；action MAE
相对 D17 仅退化 `0.065%`。predicted physics 相对 episode-local shuffled 的增益从 `3.093%` 增至 `5.409%`，
但相对 action-only 仍退化 `2.781%`，四项 gate 仍为 3/4，`dit_authorized=false`。因此按预注册停止单纯增加
VLM steps。

D37 是不训练模型的 residual calibration diagnostic。D34 physics head 输出 `r_phys`，action-only head 输出
`r_action`；测试固定 `r = r_action + alpha * (r_phys - r_action)`，其中
`alpha={0,0.125,0.25,0.5,0.75,1}`，alpha=0 精确等于 action-only，alpha=1 精确复现 D36。oracle、predicted、
episode-local shuffled 使用同一 alpha；不从 test panel 连续拟合参数，也不改变原 alpha=1 gate。若某个预设非零
alpha 同时优于 action-only 与 shuffled，说明方向正确但 hard-ID residual 幅度需校准；该结果只授权在独立 split 上
验证固定 alpha/confidence gating，不直接授权 DiT。相关 scale/shuffle/gate 回归为 `6 passed`。

### 21.52 D38 disjoint residual-scale confirmation 预注册（2026-09-28）

D37 使用全部五个固定 validation episodes，因此只能作为全 panel 探索性诊断，不能在同一批样本上选择 alpha 后再把
最优值当作独立结论。D38 在查看 D37 数值前锁定 episode 级拆分与选择规则：episodes `5,20` 为 calibration，
episodes `25,39,49` 为一次性 confirmation；两边固定相同 samples/task、seed 42、D36 8000-step VLM、D34
actual-residual head、同一 K256 tokenizer 和 episode-local cyclic shuffle。

候选仍只限预注册网格 `{0.125,0.25,0.5,0.75,1}`。calibration 上对每个 alpha 计算 predicted-vs-action-only 与
predicted-vs-shuffled 两项相对增益，选择两者较小值最大的 alpha，平分时取更小 alpha；该规则兼顾“有用性”和
“token 对应性”，并避免只优化其中一个对照。confirmation 只判定这个唯一选定 alpha：动作相对 D17 退化不超过
1%、physics accuracy 高于 K256 chance、预测 physics 相对 action-only 和 shuffled 均至少改善 1%，且 calibration
阶段两项方向都为正。自动报告即使通过也固定 `dit_authorized=false`：它只把固定 residual scale 升格为候选；后续仍需
决定是在更完整的 held-out cache 上确认，还是把同样的校准原则实现为可学习的 confidence gate 后再进入 DiT。

D38 首次启动在构造第一个任务 panel 时即停止，尚未执行模型推理或产生 calibration/confirmation JSON：两 episode
split 的最短任务只有 251 个样本，三 episode split 的最短任务只有 188 个样本，无法满足原定 256 samples/task。
在读取 D37 scale sweep 前作技术性修订，将两边统一改为 128 samples/task；episode split、seed、固定 alpha 网格、
maximin 选择规则和 1% confirmation thresholds 全部不变。输出文件名显式加入 `128samples`，避免与失败尝试混淆。

D37 已完成。evaluation/gate SHA256 分别为
`8a5e9ed44e86881d7c39e0f42453002db7271efd1b636eefdcba57c6820878fa` 与
`de8eb6ba2b42941273a4c1342e50050a3233b48bb6af8ce14cf7f68ac85c1b05`；alpha=1 精确复现 D36：physics
accuracy `11.331%`、predicted-vs-action-only `-2.781%`、predicted-vs-shuffled `+5.409%`。缩放后最优固定候选为
alpha=0.25，predicted MAE 从 action-only 的 `0.132315` 降至 `0.131586`，相对 action-only 改善 `0.550%`、
相对同 alpha shuffled 改善 `1.465%`。alpha=0.125/0.5 的 action-only 增益为 `0.430%/0.103%`；到
alpha=0.75/1 则退化 `1.060%/2.781%`。oracle 在 alpha=0.5 最佳、改善 `1.730%`，共同说明 residual 方向有效，
但 hard predicted IDs 所对应的 residual 幅度明显过强，合适全局幅度约为原来的四分之一。

alpha=0.25 的逐任务 action-only 增益为：pick_dual_bottles `+3.127%`、handover_block `+1.133%`、
place_object_basket `+0.614%`、move_playingcard_away `+0.381%`，而 shake_bottle `-1.668%`、
beat_block_hammer `-0.939%`、open_microwave `-0.821%`、click_alarmclock `-0.330%`。因此全局缩放虽消除了
大部分伤害，任务/样本间预测可靠性仍显著不同；D38 独立确认若不过 1%，后续应优先验证 soft codebook expectation
或基于 VLM physics logits 的置信度门控，而不是继续增加相同目标的 VLM steps。

D38 以 128 samples/task 完成。calibration/confirmation/selection SHA256 分别为
`2ad2a6eb7a469de02f6959b88b528bbb91d4c7e11f904427cda6f1e766c0a3f0`、
`0adb95e1ccaee5fbbacf9e9408d92c07a67266765c8e7cdf49ea68847dbc1d35`、
`932c57370c913efef5d441c23c3554376f60c19aeefce6c55acbaac6f7fee474`。calibration 按 maximin 规则选择
alpha=0.25，两项增益为 action-only `+0.398%`、shuffled `+0.980%`；在未参与选择的 confirmation episodes 上，
两项增益分别升至 `+0.734%` 与 `+1.364%`。physics accuracy `11.026%`，action 相对 D17 改善 `0.637%`。
这说明四分之一残差的正收益能跨 episode split 复现，不是 D37 同 panel 选参假象；但 action-only 增益仍低于 1%，
独立 gate 为 4/5，`dit_authorized=false`。

### 21.53 D39 soft codebook conditioning 预注册（2026-09-28）

D39 不训练任何权重，针对 hard argmax ID 在 token accuracy 约 11% 时放大错误 residual 的问题。对每个 physics token
位置保留 VLM 的 256-way logits，计算 `softmax(logits / temperature) @ EMA_codebook`，将 256 个 expected latents
按原 16 transitions x 16 queries reshape/聚合后送入完全冻结的 D34 physics head。它保持离散 tokenizer/codebook
语义，但让动作头利用 VLM 对候选 code 的不确定性；hard action tokens、D17 baseline、数据、shuffle 和所有 checkpoint
均不变。

候选在运行前固定为 temperature `{0.5,1,2}` x residual alpha `{0.125,0.25,0.5,0.75,1}`。沿用 D38 的
128 samples/task 和严格 episode split：`5,20` calibration，`25,39,49` confirmation。calibration 上最大化
predicted-vs-action-only 与 predicted-vs-shuffled 两项增益的较小值；平分先取较小 alpha，再取最接近 1 的
temperature。confirmation 只检查选中的唯一组合，仍要求动作保护、accuracy 高于 chance、两项 utility 各至少 1%。
D39 自动报告仍固定 `dit_authorized=false`；若通过，先审计 soft conditioning 与最终 DiT 接口如何对齐，再决定是否
启动连续动作头训练。若失败，则进入基于 mean max probability/entropy 的 sample-level confidence gate。

D39 已完成。calibration/confirmation/selection SHA256 分别为
`23de53671833db85cabc063d2059b17af77e3cce1584e1fea8c946e352306295`、
`a60abbdca4e3ea8bccaf206ca49da9804e5e9042cc042d9748e74218f0fe6833`、
`f7edbc592fbb55da0ecd051f28604e08e21a71156ecf052a38354d27c37b34df`。calibration/confirmation 的 raw
softmax mean max probability 仅 `9.828%/9.376%`，normalized entropy 为 `0.887/0.891`，说明 256-way physics
分布较宽。maximin 选择 T=0.5、alpha=0.25：calibration 两项增益为 `+0.423%/+0.900%`，但 confirmation 为
`-0.028%/+1.113%`，独立 gate 失败。T=1/2 随温度提高进一步伤害 action-only，表明 EMA codebook latent 的概率
均值会稀释有用 mode，不能直接替代 hard ID。

### 21.54 D40 hard-ID confidence gate 预注册（2026-09-28）

D40 保留 hard argmax code 及 D34 head，仅按每个样本 256 个 physics 位置的 `mean(max softmax probability)` 决定是否
启用 physics residual；低于阈值时严格回退 action-only。依据 D39 已观测的约 9.4--9.8% 均值范围，候选绝对阈值固定
为 `{0.06,0.08,0.10,0.12,0.15}`，residual alpha 固定为 `{0.125,0.25,0.5,0.75,1}`。阈值为可在线部署的绝对值，
不使用 confirmation 样本的标签或分位数。shuffle 对照把 hard IDs 与其 confidence 作为同一 package 在 episode 内
cyclic shift，保证 aligned/shuffled 总体 active coverage 相同。

仍使用 episodes `5,20` calibration、`25,39,49` confirmation、128 samples/task。选择规则继续最大化两项增益的
较小值，平分先取较小 alpha，再取更高阈值；confirmation 只检验唯一组合和原四项阈值。D40 报告继续固定
`dit_authorized=false`，通过后还需确认最终 DiT 能接收同一 confidence/gate 语义；失败则回到 supervision，考虑把
真实 residual utility 或 token confidence 显式纳入 predictor/head 训练，而不继续做推理期启发式搜索。

D40 已完成。calibration/confirmation/selection SHA256 分别为
`a29c382c0e6a194b881a9ebe0de2fc5eb86bb8264f22cec907df373fae4dd78d`、
`375d1ba97fae5dbb8b49ce40a37011f523dc249ae54ee12b660ffe6f38a7640a`、
`e48c02b1d7e9a744f3d35b90857bc5b8e645477b3d7247392c163f177e1b7a37`。calibration 选择 threshold=0.15、
alpha=0.75，active fraction `12.99%`，两项增益 `+0.649%/+0.679%`；confirmation active fraction `14.26%`，
两项仅 `+0.059%/+0.705%`。因此 max-softmax confidence 与真实 residual utility 的关系不能跨 episode 稳定复现，
独立 gate 失败；停止继续搜索推理期 scale/temperature/threshold。

### 21.55 D41 predicted-distribution residual head 预注册（2026-09-28）

D34 head 的 physics 输入在训练时始终为 oracle K256 IDs，而部署时为 token accuracy 约 11% 的 D36 predicted IDs。
D37--D40 表明预测包含稳定对应性，但 oracle-trained head 会把错误 IDs 解码成过强 residual，推理期校准最多得到小于
1% 的稳定 action-only 增益。D41 直接修复该训练分布错配：冻结 D36 dual VLM、D29 tokenizer 和所有已有模型，先在
8 GPU 上按任务导出全部 50 episodes 的 dual predicted action4 + physics256 IDs；用 dual predicted action IDs 重新
计算 angle-wrapped 真实 residual target，再以 predicted physics latent + dual decoded action 为输入训练同结构 head。

正式 validation episodes `5,20,25,39,49` 不参与 D41 head 的训练或 checkpoint selection。内部选模只用 D36 VLM
训练范围内的 episodes `0,10`，并从 head training 中排除；其余 43 episodes/task 用于训练。head 仍为 batch 512、
3000 steps、相同 loss/optimizer/hidden size。最终固定 D36 VLM、D41 head 和原 256 samples/task panel，使用未缩放
hard predicted IDs 做原四项 gate；不因 D37--D40 的 alpha/temperature/confidence 结果改变门槛。只有 action protection、
physics accuracy、predicted-vs-action-only 和 episode-local shuffled 均过 1% 才重新考虑 DiT。

D41 已完成并成为首个通过正式 predicted-token 四项 gate 的配置。predicted dual cache/actual-residual target/head/head
report/evaluation/gate SHA256 分别为
`5b7c4d2a55fd8065464c0e8df8b54db9263565964b81df0b5cfbf75f5aee3c8f`、
`948ccc8acdfde347ef0049e90bce37ef0298e7e884f4ec6805d88a7526bf60b7`、
`b6199af0c3eed119d36138c4d2387a77db666463ddd4715bb27c21558ed48cd0`、
`04f5818a085ec1b4733bb80233b7afd479605b9c7ee083d596023044a34c6be5`、
`a694f1fc49ad469227da42643f1a3623874b1c13bfc14c8c0d81d194c9fd9b0d`、
`2224834550c0e7a24a0370fe3c332ad8c659ca132b24850cd97dbc6b32c84d90`。

导出得到精确 key-aligned 的 85,098 x action4 与 85,098 x physics256 predicted IDs。head 用 61,851 clips 训练，
episodes `0,10` 的 2,738 clips 只用于内部选模，正式 episodes `5,20,25,39,49` 完全排除；best step 为 2900。
固定 2,048-sample 正式 panel 上，action MAE 相对 D17 仅退化 `0.065%`，physics accuracy `11.331%`；未缩放
hard predicted physics 相对 action-only 改善 `3.284%`，相对 episode-local shuffled 改善 `8.290%`，四项均通过，
`dit_authorized=true`。这比 D36 在 oracle-trained D34 head 上的 `-2.781%/+5.409%` 有决定性改善，确认核心问题是
动作头的 oracle/predicted 条件分布错配，而不是 physics token 没有动作信息。

逐任务 action-only 增益为 beat_block_hammer `+2.051%`、click_alarmclock `+1.441%`、handover_block `+4.199%`、
move_playingcard_away `+0.081%`、open_microwave `-3.489%`、pick_dual_bottles `+14.893%`、
place_object_basket `+0.868%`、shake_bottle `-2.318%`；所有任务的 aligned-vs-shuffled 均为正（`+2.49%` 至
`+22.14%`）。因此 macro gate 已授权低成本 DiT，但正式 benchmark 还必须报告逐任务结果，并保留 action-only、
predicted physics、episode-local shuffled physics 三路；不能用 pick_dual_bottles 的大收益掩盖两个负收益任务。

### 21.56 D42 predicted-token DiT 低成本验证预注册（2026-09-28）

D41 已满足此前约定的 DiT 启动条件，但首轮不直接做八任务 full training。D42 先选择已有成熟 3000-step continuous
action-only checkpoint 的 click_alarmclock，使用 D41 对应 single-task cache（4 dual-predicted action IDs + 256
dual-predicted physics IDs）做 20-step adapter-only smoke。冻结 Qwen VLM、discrete conditioner 与原 DiT base，只训练
zero-init first-block physics cross-attention adapter；physics embedding 从 D29 EMA-K256 codebook 初始化，vocabulary
明确为 256，condition dropout 0.2，并设置 `dit_action_tokens_only=true`，只修改 32 个 action queries。这样 step 0 严格复现 action-only，20 steps 只验证 predicted token 能否给冻结
continuous head 提供有方向的增量。

评估固定同一 checkpoint、样本 panel 与 diffusion noise，比较三路：（1）predicted physics enabled；（2）hard
disabled physics，即 action-only control；（3）同 episode 内把 physics IDs 与其 package 整体 cyclic shuffle。
必须同时报告 action MAE、joint MAE、gripper、逐 episode bootstrap/胜率；enabled 至少优于 hard-disabled 和 shuffled
才可把训练延长到 100/500 steps。20-step 单点即使改善也不直接授权八任务训练；若 enabled 与 shuffled 同向等幅，
视为内容无关 adapter 扰动并停止。后续逐任务优先覆盖 D41 正收益最大的 pick_dual_bottles，以及负收益的
open_microwave/shake_bottle，避免只验证容易任务。

启动前资源检查发现 node004 GPU7 有约 8.3 GB 的其他进程，D42 因而固定使用 GPU0--6、7 processes、
per-device batch 8（global batch 56），不占用 GPU7；这只改变 smoke 的 batch 规模，不改变三路 paired panel、
checkpoint 或判定规则。

D42 runner、aligned-vs-shuffled paired comparator 及其定向测试已实现；shell syntax、Python compile 与 comparator
测试均通过（`1 passed`）。2026-09-28 已在 node004 启动 20-step 作业，launcher PID `391482`，日志为
`artifacts/logs/d42_predicted_dit_smoke.log`。实际命令参数已复核为 GPU0--6、7 processes、global batch 56、
action4 + physics256、first-block cross-attention、`dit_action_tokens_only=true`、physics dropout 0.2，且冻结
Qwen VLM、discrete conditioner 与 base DiT；GPU7 未被本实验占用。runner 已成功生成 episode-local shuffled cache，
当前处于大权重加载阶段；本段只记录启动状态，不提前宣称 D42 gate 通过。

D42 20-step 训练与三路 380-sample paired evaluation 已完成，无 OOM。adapter checkpoint SHA256 为
`e1f7cf9f2136e8abefa5eef9006737e28427fbab581129809237054276824156`；aligned/hard-disabled/shuffled
evaluation SHA256 分别为 `89d8e5893b2527a0b54d3a79e0a50dec29914f3082be7c86b840e9710ec35473`、
`ce6b4c750b0c0f3b5f2b7ef660518cfbc02076c9cbdf96b0ed668462fde03ecc`、
`185175bb675ef2271fc9a0a6e7a48740383a1dae34d5aa45ed28a220163d4e4c`。三路 action MAE 为
`0.138493/0.136851/0.138214`：aligned 相对 hard-disabled 退化 `1.200%`，相对 episode-local shuffled
退化 `0.202%`；前者 action-MAE episode wins 为 `1/5`、bootstrap 95% CI `[-2.253%,-0.033%]`，后者为
`1/5`、95% CI `[-0.409%,-0.040%]`。action RMSE 的 aligned-vs-shuffled 为微弱 `+0.057%`，但 primary
action MAE、endpoint、wrapped joint、gripper 与 velocity 均未显示 aligned 优势。故 D42 gate 失败，不按原
规则直接扩到 100/500 steps。

这项失败只否定“当前 first-block frozen-base adapter 在 20 steps 已有效”，不能直接否定更长训练或真正的
pi0.5 action expert。当前 `QwenPIPhysicalTokens` 实际调用 `LayerwiseFlowmatchingActionHead`；该 head 的来源与
结构是 GR00T N1.5-style cross-attention DiT，和仓库中的 `QwenPI_v4`（Qwen3-0.6B suffix action expert、
AdaRMSNorm、pi0.5-style prefix/suffix）不是同一连续头。另有明确优化不对称：action-only base 已训练 3000
steps，physics adapter 只有 20 steps且 output projection 从零初始化；20-step checkpoint 中 code embedding、
physics type、output projection 已变化，但三个 LayerNorm scale 在 BF16 保存值上仍逐元素等于初始化。因而应先
区分优化不足、训练集过拟合和结构不匹配，再决定更长预算。

### 21.57 D43 train-vs-heldout 与连续头路线判别（2026-09-28）

D43 首先不训练新参数，在 D42 同一 checkpoint 上选择训练 episodes `0,1,2,3,4` 的固定 190-sample panel，
重复 aligned、hard-disabled、episode-local shuffled 三路评估。该 panel 只用于诊断，不能替代 episodes
`5,20,25,39,49` 的正式 held-out 结果。若训练 panel 上 aligned 同时优于 disabled 与 shuffled，而 held-out
为负，则支持“小数据 adapter 过拟合/泛化不足”，下一步做正则化或多任务训练；若训练 panel 上 aligned 优于
shuffled但仍差于 disabled，则支持“已学对应关系但 residual 幅度/收敛不足”，允许预注册从相同 action-only
初始化独立训练的 `20/100/500` learning curve；若训练 panel 上 aligned 仍不优于 shuffled，则停止给当前
GR00T-style post-hoc adapter 单纯增加 steps，优先给真正 `QwenPI_v4` 接入 action4 + physics256 条件。

若进入 longer-step 分支，必须增加 matched action-only fine-tune control：从同一个 3000-step checkpoint 出发，
相同 train samples、steps、seed、batch 与 LR，一路解冻同范围 base action parameters但禁用 physics，一路使用
physics；同时保留 frozen-base adapter-only 曲线。这样才能检验“action-only 更快过拟合、physics 随难度/步数
后期反超”，而不是把 3000-step base 与 20-step新分支直接比较。正式判定看 held-out learning curve 和
aligned-vs-shuffled，不以 train loss 或单一最终点选优。

三头联合训练作为后续独立阶段：action-token CE、physics-token CE、continuous flow loss 可同训，但当前代码路径
尚未把 dual-token VLM probe 与 continuous policy 组成一个完整联合 forward；D42 本身仅训练一种连续 physics
adapter，action/physics token 均来自离线 cache。联合阶段必须从 D10 的灾难性遗忘吸取约束：保留成熟 action
teacher 的 logit/continuous distillation，先冻结大部分 VLM、只开放最后少量层或 LoRA，并做三路 loss/gradient
norm 平衡。不能把“代码中分别存在三个 head”误写成“当前 runner 已在同时训练三个 head”。

D43 已完成。aligned/hard-disabled/shuffled 的 train-panel action MAE 为
`0.106547/0.103900/0.106349`；aligned 相对 disabled 退化 `2.548%`，action-MAE `0/5` episode wins，
bootstrap 95% CI `[-3.888%,-1.576%]`；相对 shuffled 也退化 `0.186%`，`1/5` wins，95% CI
`[-0.379%,-0.019%]`。两项负向在训练 episodes 上比正式 held-out 更明显，因此排除“20-step adapter 已拟合
训练对应关系、只是 held-out 泛化失败”。五份 D43 JSON 的 SHA256 依次为 aligned
`d0b78e48562b51ab57c15c87000c1dde6a269ce8a94a83b6587f7779d8ee481d`、disabled
`b8455896917bf7a23b9d0755240e147d1de240ece190f5b8bcaebab68d4b519f`、shuffled
`1ce871996e3e890bb62259cd1b63e632bb20b10f313cfb1e15431fa607683314`、aligned-vs-disabled
`b0e384cf8b142e98ea0032cbd7acc4c584212a774a717beb28e4077b249c5343`、aligned-vs-shuffled
`9339f63906f4c1d9a7e373c700f6e8e7b272f64a9dbc6cef1b22911edef8a4e2`。

按预注册分支，停止给当前 GR00T-style first-block post-hoc adapter 单纯增加 steps。下一最小结构对照改为真正
`QwenPI_v4` action expert：先使用离线 predicted action4 + physics256 cache，建立同 checkpoint 的
enabled/disabled/shuffled exact control，只验证连续头与 token 条件接口；这一步通过后才接 dual-token VLM 做
action-token CE + physics-token CE + continuous flow 三头联合训练。若缺少 Qwen3-0.6B expert 或成熟 pi baseline，
先完成代码/梯度/小 batch smoke，不把随机未收敛 pi head 与 3000-step旧 head 的绝对 MAE作结论。

D44 的真实 pi 初始化骨架已完成。新增 `QwenPIv4PhysicalTokens` 只通过默认恒等 hook 扩展原
`QwenPI_v4`，把 action/physics ID embeddings 追加到投影后的 1024-D expert prefix；关闭离散条件时返回原
prefix 对象本身，旧 pi 路径不改变。轻量 prefix routing、hard-disable identity、EMA code geometry 与既有
conditioner/DiT tests 合计 `31 passed`。本机已有成熟 Robotwin-all pi checkpoint：
`/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`，
不是随机 action expert；其 SHA256 为
`646881ca046c2de3babb6370fed4b5358532869f3570064376593131e9aaea12`。

真实构建审计确认 Qwen-VL hidden 2560、Qwen3-0.6B expert hidden 1024、28 expert layers、horizon 32；checkpoint
的 1,346 个旧 tensors 无 unexpected key，missing keys 精确等于新增 conditioner 的 10 个 tensors。总参数
5.244B；冻结成熟 pi 主干后 conditioner 为 663,554 trainable parameters。审计报告为
`artifacts/evaluations/d44_qwenpi_v4_physical_init_audit.json`。正式 smoke 前还需同时从 D16 action EMA 与 D29
physics EMA 初始化两套 ID embeddings；只初始化 physics 而让 action4 随机，会把 action-condition 接口误差
混入 physics 对照。

### 21.58 D45 true-pi matched conditioner curve（2026-09-28）

D45 已把离线 predicted action4 + physics256 条件真正接到 `QwenPI_v4` 的 28 层 Qwen3-0.6B action expert，而非
D42/D43 的 GR00T-style DiT。两套 embedding 分别从 action EMA `4x128` 与 physics EMA `256x256` codebook
初始化；physics256 每连续 16 个 ID 聚合为一个 expert-memory token，因此保留全部 256 个离散预测但只向成熟 pi
prefix 追加 16 个 physics positions，加上 4 个 action positions 共 20 个。成熟 100k pi/VLM 全部冻结，只训练
conditioner 的 663,554 个参数。严格初始化审计确认总参数 5,244,442,128、旧 checkpoint 1,346 个 tensors 全部
加载、无 unexpected key，missing keys 精确等于 10 个新 conditioner tensors；审计文件为
`artifacts/evaluations/d45_qwenpi_v4_grouped_physical_init_audit.json`。

首次 7-GPU smoke 在 step 1 出现所有 rank 100% utilization 但无进度。根因不是 pi 计算量或 OOM，而是整组
physics dropout 在各 rank 独立采样，导致各 rank 使用不同参数子集并令 ZeRO/DDP collective 死锁。现已在
`DiscreteTokenConditioner._drop_group` 中从 rank 0 broadcast dropout draw；修复后恢复约 0.25 sec/step。该约束也
适用于后续三头联合训练：任何会改变参与反传参数集合的 routing/dropout 必须跨 rank 同步。

20-step matched curve 已完成。action-only 与 action+physics 均从同一个成熟 100k pi checkpoint 出发，使用相同
八任务数据、held-out episodes、seed、LR、batch、warmup 和总步数；checkpoint SHA256 分别为
`52f7398dd252c793e17e5f6c3a6e8ff17e5b1339332c69ba18e94e094dddeae` 与
`93e481544762da064c05408627f898ba7f0c4b4d6c85ddccce17386bdf4930f9`。click_alarmclock 固定 held-out
episodes `5,20,25,39,49`、64 samples、4 inference steps 的 action MAE 为：pure pi `0.01259646`、
action-only `0.01343037`、同 dual checkpoint hard-disabled physics `0.01343076`、aligned physics
`0.01657500`、episode-local shuffled physics `0.01658255`。action-only 相对 pure pi 退化 `6.620%`，95% CI
`[-11.401%,-2.874%]`；aligned 相对 dual hard-disabled 退化 `23.411%`，5/5 episodes 均未获胜，95% CI
`[-35.811%,-13.095%]`。aligned 相对 shuffled 仅改善 `0.0455%`，95% CI
`[-0.0671%,+0.1404%]`，不能证明 20 steps 已使用 token 对应关系。完整结果位于
`artifacts/evaluations/d45_grouped_concat_steps20_click_alarmclock_64samples/`。

这一结果不能单独否定“action-only 更快、physics 随训练步数和任务难度后期反超”，但暴露了 concat 接口的强混杂：
新增 memory positions 从 step 0 即进入成熟 pi 每层 cross-attention 的 softmax 分母，gate=0.1 只缩放 token
表示，并不能使新增位置成为严格零扰动。20-step dual checkpoint 的 physics gate 仅从 BF16 `0.10010` 变为
`0.10059`，因此早期优化很可能主要在补偿 attention 分布变化。D45 仍按预注册扩到 100 steps，以直接观察同一接口
是否只是收敛较慢；但不因增加 steps 而取消 shuffled control。

### 21.59 D46 pi zero-impact multilayer residual 预注册（2026-09-28）

D46 与 D45 使用同一成熟 pi、同一 predicted cache、同一 EMA 初始化、同一八任务 split 和 matched
action-only/dual 训练预算，只改变 discrete-to-pi 接口。base visual-language prefix 继续单独送入原 28 层
cross-attention；action/physics memory 使用同一层已经训练成熟的 cross-attention K/V/O 映射形成额外 residual，
再乘每层独立的可学习 scalar gate。28 个 gate 全部零初始化，因此启用离散分支的 step 0 必须在固定输入和噪声下
逐元素复现成熟 pi，不改变原 prefix attention 的 softmax 分母。训练仍只更新 conditioner 与 layer gates，先不解冻
pi expert，以便和 D45 隔离比较。

D46 的最低实现 gate 为：（1）hard-disabled 与 pure pi 数值等价；（2）zero-gate enabled 与 pure pi 数值等价；
（3）非零 gate 对 token ID permutation 敏感；（4）多 rank dropout 同步测试继续通过。20-step 首轮仍比较
pure pi、action-only、dual hard-disabled、aligned、episode-local shuffled 五路；只有 aligned 明显优于 shuffled
且不再出现 D45 的大幅 hard-disabled 退化，才扩到 100/500 steps 和 pick_dual_bottles/open_microwave 困难度对照。

三头联合训练（action-token CE + physics-token CE + continuous flow）是合理的下一阶段，但不能用来替代 D45/D46
结构诊断。它可能通过在线 token 表示共同适配解决离线 cache 与 action expert 的表征错位，也可能因 CE 梯度主导、
VLM 遗忘或错误 token 的早期反馈放大而更差。进入三头阶段时先冻结大部分 VLM，只开放 dual probe/最后少量层或
LoRA，并加入成熟 action-logit teacher、continuous-policy teacher、显式 loss/gradient norm 监控；同时保留 matched
action-only、physics shuffled 和 frozen-token controls。这样才能检验“联合训练是否让 physics 后期反超”，而不是
把解冻参数量增加误认为物理推理收益。

D45 的 100-step matched curve 已完成，未支持“concat 只是在 20 steps 收敛较慢”的解释。click_alarmclock 同一
64-sample held-out panel 上，pure pi/action-only/dual hard-disabled/aligned/shuffled 的 action MAE 分别为
`0.01259646/0.01339699/0.01340035/0.01637248/0.01638239`。action-only 相对 pure pi 退化
`6.355%`，95% CI `[-10.939%,-2.763%]`；aligned 相对 dual hard-disabled 退化 `22.179%`，0/5 episode
wins，95% CI `[-33.767%,-12.588%]`；aligned 相对 shuffled 仅改善 `0.0605%`，95% CI
`[-0.0889%,+0.1874%]`。相较 20-step 的 `-6.620%/-23.411%/+0.0455%` 只有极小恢复，增加 80 steps
并未产生 token correspondence。action-only/dual checkpoint SHA256 分别为
`67d5bacaf0fd76e56b3eccc580eedb5d499c9ab0f95838e3d64bd9dea4395b06` 与
`96b2a02d3a94b0020f3e0ff85a1d2a25d16ff7805b17cdbced450e78c4a6af1c`；完整评估位于
`artifacts/evaluations/d45_grouped_concat_steps100_click_alarmclock_64samples/`。因此停止给 D45 concat 单纯
增加 steps；后续长程预算转向零扰动 D46。

D46 实现与初始化 gate 已完成。`Qwen3ExpertLayer` 保持原 visual-language prefix cross-attention 不变，并可选对
独立 discrete memory 再调用同一套已训练 K/V/O 权重；额外 residual 乘逐层 scalar gate 后加入。默认 base
`QwenPI_v4` hook 返回无 residual，旧框架行为不变。`QwenPIv4PhysicalTokens` 仅在
`pi_multilayer_residual` 模式把 conditioner 产生的 4 action + 16 grouped physics memory 与 base prefix 拆开，
28 个 gate 均零初始化。扩展 runner/config/evaluator 已实现，相关 compile、shell syntax 和扩大定向测试为
`37 passed`。

真实 5.244B 初始化审计报告为 `artifacts/evaluations/d46_qwenpi_v4_residual_init_audit.json`：成熟 checkpoint
1,346 个 tensors 全部加载，无 unexpected key；missing 精确为 10 个 conditioner tensors + 28 个 layer gates；
总/trainable 参数为 `5,244,442,156/663,582`，260 个离散 IDs 形成 20 个 residual-memory positions，追加到
原 prefix 的 positions 为 0。初始化扩展 checkpoint 为
`artifacts/checkpoints/d46_qwenpi_v4_residual_initialized.pt`，SHA256
`34fd910e19f55f0846c25755008b9e7b4eda195601260c7e84bbe7c40618be1d`。在真实模型、固定 seed 和 8 个
click_alarmclock held-out samples 上，pure pi 与 enabled action4+physics256 zero-gate 的 aggregate metrics 和
逐 episode sufficient statistics 完全相等，action MAE 均为 `0.011784871687788316`，确认 step-0 数值零扰动。

D46 20-step action-only/dual matched 训练已完成，无 OOM，约 `0.27--0.30 sec/step`。checkpoint 各含 10 个
conditioner tensors + 28 个 gates；action-only gates 范围约 `[-8.62e-4,+4.83e-4]`，dual 为
`[-9.57e-4,+7.29e-4]`，说明零初始化使早期更新保守，20 steps 主要先学习 residual 路由，embedding 有效梯度
启动更晚。五路 held-out 评估正在进行，结果未完成前不判断是否扩到 100 steps或提高 gate 专属 LR。

D46 20-step 五路评估已完成。pure pi/action-only/dual hard-disabled/aligned/shuffled action MAE 为
`0.0125964616/0.0125966177/0.0125963291/0.0125964536/0.0125964556`。action-only 相对 pure pi 为
`-0.00124%`，95% CI 跨 0；aligned 相对 dual hard-disabled 为 `-0.000989%`，绝对 delta 仅
`1.25e-7`；aligned 相对 shuffled 为 `+0.0000164%`，95% CI 跨 0。action+physics 相对独立 action-only
checkpoint 为 `+0.00130%`，同样跨 0。完整结果位于
`artifacts/evaluations/d46_pi_multilayer_residual_steps20_click_alarmclock_64samples/`。因此 D46 已把 D45 的
step-0/短程 `6%--23%` 结构性退化降至数值噪声量级，但 20 steps 尚无 token correspondence 证据，不能宣称
physics 有效。

按单变量原则，下一步先保持 gate/conditioner LR `1e-4`、数据和其余配置不变，将 matched action-only/dual 延长到
100 steps；只有仍无信号时才另立“提高 layer-gate LR”实验，不能同时改变步数与 LR。100-step 作业已按用户要求
在 node3 (`10.120.154.141`) GPU0--6 启动，ports `29847/29848`。启动前每卡约 62GB 可用，而 D46 在 node1
实测每卡增量约 1.5GB，预期不会 OOM；仍需监控 node3 上已有计算负载造成的吞吐影响。

资源安排更新：当前交互节点实际为 node1；用户授权后续使用 node3。node3 地址 `10.120.154.141`，检查时每卡
约 18GB 已用、约 62GB 空闲。D46 冻结主干训练在当前节点的每卡额外显存约 1.5GB，因此后续 100-step、困难任务
及三头 smoke 优先通过共享路径在 node3 启动；每次仍先检查实时显存和进程，不能把“可使用”解释为可造成 OOM。

D46 100-step matched curve 已在 node3 完成。action-only/dual checkpoint SHA256 为
`761d3362224a544954f53edc947c691e47bfb1bdf736a81b03b1888b8a59677b` 与
`01e9adb1b74203e666278f1b60e26b464d8bac55b87a7a5e8f0eb384d5faba40`。pure pi/action-only/
dual hard-disabled/aligned/shuffled action MAE 为
`0.0125964616/0.0125964886/0.0125956355/0.0125959571/0.0125959577`。action-only 相对 pure pi
`-0.000215%`；aligned 相对 hard-disabled `-0.00255%`；aligned 相对 shuffled 仅
`+0.00000458%`；action+physics 相对独立 action-only 为 `+0.00422%`，所有实质 utility/correspondence 仍为零。
完整结果位于 `artifacts/evaluations/d46_pi_multilayer_residual_steps100_click_alarmclock_64samples/`。这排除“保持
相同优化设置只延长到 100 steps 即会出现 physics 收益”，但仍不能否定 gate 学习率过低。

100-step action-only/dual layer-gate mean absolute value 仅为 `1.92e-3/1.31e-3`，dual gate 范围
`[-3.89e-3,+1.84e-3]`。与初始化比较，dual physics type mean absolute delta 为 `1.09e-3`，但 physics
code/position 的观测变化大多接近 BF16 保存量化尺度；零 gate 造成的梯度瓶颈仍然明显。因此 D47 固定 D46 的
20 steps、conditioner LR `1e-4`、数据、seed、batch、dropout 和五路评估，只把 28 个 layer-gate LR 提高到
`1e-2`。选择 100x 而非继续微调步数，是为了在短程内把 gate 推到可测 residual 幅度；若出现明显 action damage
但 aligned 不优于 shuffled，则回退到 `1e-3` 没有意义，应改分支分离或联合监督。runner 已支持独立
`D46_RUN_TAG/D46_GATE_LR`，D47 run tag 为 `d47_gate_lr1e2`，作业已在 node3 GPU0--6 启动。

D47 20-step 训练与五路评估已完成。提高 gate LR 后 action-only/dual gate mean absolute value 达到
`0.0419/0.0462`，dual 范围 `[-0.0957,+0.0732]`，确认 residual 通道已实际打开。action-only/dual
checkpoint SHA256 分别为 `8050e83bd27c17a60378348e14ed146ac7fe0503066e062c90b6488f8e316c0e` 与
`a90057bf60e41c7a686eeabbf9b42650cbb7f7a91dc586a6a908b87ee9435e18`。pure pi/action-only/
dual hard-disabled/aligned/shuffled action MAE 为
`0.0125964616/0.0126145417/0.0125831447/0.0125958856/0.0125958933`。action+physics 相对独立
action-only 改善 `0.1479%`，95% CI `[+0.0017%,+0.2650%]`；但 aligned 相对同 dual hard-disabled 退化
`0.1013%`，0/5 episode wins，95% CI `[-0.1498%,-0.0548%]`。aligned 相对 shuffled 仅改善
`0.0000611%`，95% CI `[-0.0221%,+0.0223%]`，没有 token correspondence。完整结果位于
`artifacts/evaluations/d47_gate_lr1e2_pi_multilayer_residual_steps20_click_alarmclock_64samples/`。

D47 说明 dual 分支相对独立 action-only 的小收益来自训练后 action residual 路径，而不是 physics 内容；真正启用
physics 反而伤害同 checkpoint control。结构原因是 D46 虽已把 discrete memory 从成熟 visual prefix 中分离，但
action4 与 grouped physics16 仍在同一次 residual cross-attention 的 softmax 中竞争。加入 physics positions 会改变
action residual 的分母和幅度，所以它仍有一个较小版本的 D45 稀释问题；aligned/shuffled 相同正是该解释的证据。
停止继续搜索同接口 gate LR 或 steps。

### 21.60 D48 action/physics residual 分离与三头进入条件（2026-09-28）

D48 应把 action 与 physics memory 完全拆成两次独立 attention：每层分别计算 `A_l(action4)` 与
`P_l(grouped_physics16)`，使用两套零初始化逐层 gates `g_action[l]`、`g_physics[l]`，更新为
`x <- x + g_action*A_l + g_physics*P_l`。原 visual prefix attention 不变，physics 的启用/禁用不再改变 action
attention 的 softmax 分母。必须新增三项精确 gate：（1）两套 gate 为 0 时等于 pure pi；（2）physics gate 为 0
时 aligned/shuffled physics 都与 action-only residual 逐值相等；（3）仅 physics gate 非零时 permutation 才改变
输出。D48 先做 20-step matched action-only/dual，physics gate 可用 D47 已验证能打开通路的专属 LR，但 action 与
physics gate 的 LR/统计必须分别记录。

只有 D48 出现 aligned 明显优于 shuffled，才进入三头联合训练。三头很可能比当前离线 cache 更有潜力，因为
action-token CE、physics-token CE 能让 VLM 表示与 continuous expert 的 residual memory 在线共适配；但它不会自动
修复 shared-softmax 稀释，且会引入 CE/flow 梯度冲突与遗忘。因此正确顺序是先完成 D48 分离接口，再增加 dual VLM
probe forward、action CE、physics CE、flow loss，并用 teacher distillation 和 gradient-norm/loss-weight 监控保护
成熟 pi。三头实验仍需 action-only CE+flow、physics shuffled、frozen-token 三个 controls，不能只与纯 pi 比较。

D48 独立双路接口现已实现并完成训练前审计。新增
`pi_separate_multilayer_residual` 模式保留成熟 visual-language prefix attention，随后对 action memory 与 physics
memory 分别调用一次 cross-attention；两次调用不共享 softmax 分母，并使用各自 28 个零初始化逐层 scalar gates。
这里复用成熟 pi 已训练的 Q/K/V/O 投影，只拆 attention 调用与 gate，不新增两套随机投影参数。旧
`pi_multilayer_residual` 路径保持不变，可继续复现 D46/D47。

模块级 gate 已覆盖并通过：两套 gate 为零严格等于 pure pi；physics gate 为零时，加入任意 physics memory 都严格
等于 action-only residual；physics gate 非零时改变 physics memory 会改变输出。相关 conditioner、D46 向后兼容和
D48 定向测试合计 `26 passed`，Python compile 与两份 shell runner syntax 均通过。真实 5.244B 初始化审计显示成熟
checkpoint 的 1,346 个 tensors 全部加载、无 unexpected key；missing 精确为 conditioner 10 个 tensors、action
gates 28 个与 physics gates 28 个。总/trainable 参数为 `5,244,442,184/663,610`。初始化扩展保存为
`artifacts/checkpoints/d48_qwenpi_v4_separate_residual_initialized.pt`，SHA256 为
`32b0d22f1f5971ac7205646b1d61ad7d9559d608f660baadfd8fc5b7003be0a9`；审计报告为
`artifacts/evaluations/d48_qwenpi_v4_separate_residual_init_audit.json`。

关于“action128 / physics128”必须区分 sequence positions 与 vocabulary/codebook size。当前 action 是 **4 个位置、
128 类码本**，physics 是 **256 个原始位置、256 类码本**；动作头通过 16-group mean pooling 实际接收独立的
physics16 memory。因此当前竞争是 action4 vs physics16，而不是 action128 vs physics256。PhiZero 本地代码与数据
协议核实为固定 256 个 motion positions、25,000 类词表；它的 256 对应 33-frame 视频未来运动，不能直接作为
32-step robot action chunk 的最佳位置数证据。当前 `ActionChunkTokenizer` 又限制 `num_tokens <= horizon=32`，
所以 action positions=128 不是配置修改，而需要更换 tokenizer 结构，且每个位置平均不到一个 action step，现阶段
缺少合理性。

已有 D20 的 physics 原始位置消融也不支持立即把 256 降到 128：16/64/256 的 held-out macro quantized RMSE 分别
为 `0.092089/0.091095/0.086395`，continuous RMSE 为 `0.091668/0.090520/0.084805`，256 明显最好。下一轮 count
实验因此分两层进行：（1）保留 raw physics256，只在 D48 成立后比较动作头 physics memory `8/16/32`；（2）另训
matched action tokenizer，比较 action positions `4/8/16/32`。优先检验 action16 + physics16 的平衡接口，不在
没有 32-position 饱和证据前跳到 action128。若确实要回答 raw physics128，必须另训 128-position EMA/FSQ
tokenizer并做与 D20 同协议的重建、code usage、held-out 对照，不能用动作头 memory pooling 结果替代。

D48 首轮固定 action4、raw physics256→memory16，conditioner LR `1e-4`，action/physics layer-gate LR 均为
`1e-2`，先跑 20-step matched action-only/dual。训练已按用户授权在 node3 GPU0--6 启动；启动时每卡约
62GB 可用，历史实测该冻结主干任务额外显存约 1.5--1.7GB/卡。完成后仍做 pure pi、action-only、dual
hard-disabled、aligned、episode-local shuffled 五路固定 panel；只有 aligned 明显优于 shuffled 才进入
physics memory `8/16/32` sweep 与三头联合训练。

D48 20-step matched 训练与 click_alarmclock 五路 held-out 评估已完成。训练未 OOM，峰值约 20.2GB/卡；action-only
checkpoint 中 action gate mean absolute 为 `0.041920`，未使用的 physics gates 精确保持 0。dual checkpoint
中 action/physics gate mean absolute 分别为 `0.041928/0.046868`，范围分别为
`[-0.08447,+0.04883]` 与 `[-0.08154,+0.08203]`，说明两条独立通路均已打开。action-only/dual checkpoint
SHA256 分别为 `d014c155239a7f289ed04aefe4d20dd2207c6c50b1bd47238524c9c71890c1e3` 与
`8e2c341b6e30324349b09634a965973bde9a5aac84327c97818176076f4995c0`。

pure pi/action-only/dual hard-disabled/aligned/shuffled 的 action MAE 分别为
`0.0125964616/0.0126145408/0.0126144731/0.0126015129/0.0126025502`。aligned 相对同 dual
hard-disabled 改善 `0.10274%`，4/5 episode wins，但 paired episode bootstrap 95% CI
`[-0.02575%,+0.22397%]` 跨 0；相对独立 action-only checkpoint 改善 `0.10328%`，95% CI
`[-0.02638%,+0.22612%]` 也跨 0。更关键的 aligned 相对 episode-local shuffled 仅改善 `0.008231%`，
2/5 action-MAE episode wins，95% CI `[-0.02022%,+0.05298%]`，仍无 token correspondence 证据。完整结果位于
`artifacts/evaluations/d48_separate_attention_steps20_click_alarmclock_64samples/`。

这与 D47 的区别很重要：D47 shared residual attention 中启用 physics 相对 hard-disabled 退化 `0.1013%`；D48
分离 attention 后转为改善 `0.1027%`，说明 4-vs-16 softmax 竞争确实被移除。但 aligned/shuffled 仍近乎相等，
所以融合结构修复只是必要条件，不足以证明 physics representation 有用。评估期间还发现并修复了 D48 `e0` 在
两个 residual count 均为 0 时错误拒绝纯 prefix 的边界条件；新增 hard-disable regression 后定向测试仍为
`26 passed`，训练 checkpoint 不受该推理路由 bug 影响。

按预注册条件，当前不立即启动 token-count sweep：没有 correspondence 时比较 memory 8/16/32 只能测 attention
扰动，不能选出“物理信息量最优”的长度。下一最低成本判别是复用同一 D48 checkpoint，在较难的
`pick_dual_bottles` 与 `open_microwave` 固定 held-out panel 上只跑 aligned/shuffled；若困难任务出现稳定差异，再扩
100-step matched curve 并做 count sweep。若困难任务仍无差异，则优先检查 predicted physics token 本身相对 oracle
token 的任务互信息，以及进入 action head 前的 mean grouping 是否破坏 code identity；届时先比较 learned resampler
而不是盲目增大 action positions。

D48 20-step 困难任务 correspondence 复核已完成。`pick_dual_bottles` 上 aligned/shuffled action MAE 为
`0.0208448150/0.0208552084`，aligned 改善 `0.04984%`，4/5 episode wins，paired bootstrap 95% CI
`[+0.01606%,+0.08203%]`；action RMSE、endpoint joint MAE、joint velocity MAE、joint wrapped MAE 的 CI 也均
严格为正。这是当前首个同 checkpoint、仅打乱 physics ID 对应关系后显著变差的结果，说明分离接口下 physics 内容
并非完全未被使用，但效应仍非常小。`open_microwave` 上 aligned/shuffled action MAE 为
`0.0093665375/0.0093649292`，aligned 反而退化 `0.01717%`，2/5 wins，95% CI
`[-0.05003%,+0.02815%]`，不显著。结果分别位于
`artifacts/evaluations/d48_separate_attention_steps20_pick_dual_bottles_64samples/` 与
`artifacts/evaluations/d48_separate_attention_steps20_open_microwave_64samples/`。

因此 D48 达到“至少一个困难任务 aligned 显著优于 shuffled”的最低进入条件，但还不足以立刻选 token count：效应
可能是真实的任务依赖物理信息，也可能是单任务/短程小效应。下一步固定接口、数据与 gate LR，先扩 matched
action-only/dual 到 100 steps，并在 click_alarmclock、pick_dual_bottles、open_microwave 三任务保留 shuffled
control。若 pick_dual_bottles correspondence 随步数增强或跨任务出现，再比较 raw physics256→memory
`8/16/32`；若 100 steps 消失，则先复核多 seed，不以单次 20-step CI 选择长度。

D48 100-step matched 训练与三个任务的完整五路评估已完成，node3 GPU0--6 未 OOM。当前 token 定义再次固定为：
action tokenizer 每个 action chunk 输出 **4 个 token positions、128 类 codebook**；physics tokenizer 输出
**256 个 raw token positions、256 类 codebook**，进入策略前按 16 个一组聚合为 **16 个 physics memory
positions**。本文中的 `action-only` 是“启用 4 个离散 action token、关闭 physics token”，而 `pure pi/e0`
才是“离散 action 与 physics 条件均关闭”的成熟连续动作 baseline。

100-step action-only/dual checkpoint SHA256 分别为
`77688e1cb00cff69409a11a580f20c72b2cd41d8aa86dac3189334e02e167623` 与
`abc0df1284e400e72c8ea7a9ef1cd90b2c8987ec0128e25fc4560e684114db6f`。action-only 的 action gate mean
absolute 为 `0.161298`，未使用 physics gates 精确为 0；dual 的 action/physics gate mean absolute 分别为
`0.163564/0.129977`，范围分别为 `[-0.330078,+0.140625]` 与 `[-0.300781,+0.275391]`。相比 20-step 的
约 `0.042/0.047`，两路条件残差均已充分打开。

三任务 action MAE（pure pi/action-only/dual hard-disabled/aligned/shuffled）如下：

- `click_alarmclock`: `0.01259646/0.01258035/0.01258344/0.01249765/0.01250002`。aligned 相对
  hard-disabled 改善 `0.6818%`，4/5 wins，95% CI `[+0.0474%,+1.3189%]`；相对 shuffled 改善仅
  `0.01892%`，3/5 wins，95% CI `[-0.07803%,+0.14528%]`。
- `pick_dual_bottles`: `0.02084975/0.02086214/0.02086146/0.02085323/0.02088276`。aligned 相对
  hard-disabled 改善 `0.03943%`，CI 跨 0；关键的 aligned 相对 shuffled 改善 **`0.14139%`**，4/5 wins，
  95% CI **`[+0.04442%,+0.23310%]`**。该 correspondence 效应相较 20-step 的 `0.04984%` 增强。
- `open_microwave`: `0.00936032/0.00934607/0.00934645/0.00931926/0.00931575`。aligned 相对
  hard-disabled 改善 `0.29087%`，5/5 wins，95% CI `[+0.14603%,+0.50797%]`；但 aligned 相对 shuffled
  为 `-0.03776%`，2/5 wins，95% CI `[-0.12310%,+0.08502%]`，没有 correspondence。

完整结果位于 `artifacts/evaluations/d48_separate_attention_steps100_{click_alarmclock,pick_dual_bottles,
open_microwave}_64samples/`。解释上必须分开两件事：click/open 的 enabled-vs-disabled 正收益证明 physics 分支
作为可学习 residual 有效，但不能证明其使用了正确 physics 内容；pick_dual 的 aligned-vs-shuffled 严格正 CI
才是 token correspondence 证据。该信号随 20→100 steps 增强，达到预注册的 count-sweep 进入条件。

下一步 D49 保持 raw physics positions=256、action positions=4、codebooks、cache、数据、seed、D48 分离
attention、conditioner LR `1e-4` 与两套 gate LR `1e-2` 不变，仅比较 physics memory positions `8/16/32`
（对应固定 mean group sizes `32/16/8`）。主选择指标为 `pick_dual_bottles` aligned-vs-shuffled paired CI，
click/open 用于检查跨任务副作用；不能用 enabled-vs-disabled 单独选择长度。先做 20-step 低成本筛选，胜者与
memory16 control 再扩 100-step；在此之前不改 raw tokenizer 到 128，也不启动 action positions sweep。

D49 的 20-step physics-memory sweep 已完成。memory8 与 memory32 分别新训 dual checkpoint，memory16 复用
完全同协议的 D48 20-step dual checkpoint；三者均保持 raw physics256、action4 和各自独立 attention/gate。
memory8/memory32 checkpoint SHA256 分别为
`cac7510a95709e84572c3340dda1ae5f83cfbab8bb58dae6abc2d4edd9ebff43` 与
`fe0bd26db325f81aac82c426eebb8454fdf991b6e70fb6697c1f4e27e62ae358`。训练和三路并行评估均未 OOM。

`pick_dual_bottles` 上 memory8/16/32 的 aligned action MAE 为
`0.0208448801/0.0208448150/0.0208448030`，shuffled MAE 为
`0.0208554372/0.0208552084/0.0208551670`。对应 aligned-vs-shuffled 相对改善分别为
`0.05062%/0.04984%/0.04970%`，均为 4/5 episode wins；95% CI 分别为
`[+0.01636%,+0.08333%]`、`[+0.01606%,+0.08203%]`、`[+0.01592%,+0.08192%]`。三种长度的
hard-disabled MAE 也仅相差约 `8e-9`，aligned MAE 相差小于 `8e-8`。

因此 20-step 下 memory `8/16/32` 在当前 fixed mean grouping 中实质等效，不能把 memory8 的极小数值领先解释
为表示能力更强；它只能在统计等效时凭计算量更低成为优先候选。这一结果还提示：当前有效信号可能主要来自
physics token 的低频/全局统计，简单分组数不是瓶颈；若后续需要更细粒度信息，应比较 learned resampler，而不是
继续堆 memory positions。按预注册方案，下一步只把 memory8 扩到 100 steps，与已有 memory16-100 在同一
`pick_dual_bottles` panel 上比较 aligned-vs-shuffled；memory32 不再消耗 100-step 预算。若 memory8 维持等效则
选 8 作为低成本默认，若 memory16 的长程 correspondence 明显更强则保留 16。

memory8 的 100-step 扩展与三路评估已完成。checkpoint SHA256 为
`67dca228c97186ee6ed70808467f670b78488b926822c91283dd5929c8ba840a`；action/physics gate mean absolute
为 `0.163549/0.130071`，与 memory16-100 的 `0.163564/0.129977` 匹配，排除了 gate 强度差异。
`pick_dual_bottles` 上 memory8 的 hard-disabled/aligned/shuffled MAE 为
`0.0208614732/0.0208533846/0.0208830091`，aligned-vs-shuffled 改善 `0.14186%`，4/5 wins，95% CI
`[+0.04452%,+0.23368%]`。memory16 对应为
`0.0208614552/0.0208532295/0.0208827561`，改善 `0.14139%`，4/5 wins，95% CI
`[+0.04442%,+0.23310%]`。两者长程 correspondence 与绝对 MAE 仍实质相同。

D49 结论：当前默认改为 **raw physics256 -> policy memory8（group size 32）**，不是把物理 tokenizer 改成
128 或 8 个 token。memory16 保留为复现实验 control，memory32 停止扩展。该选择依据是效果统计等效时降低
cross-attention K/V 长度和后续三头实验成本，而不是宣称 memory8 信息更多。下一阶段可进入三头联合训练最小
smoke：continuous flow/DiT loss + action-token CE + physics-token CE，并保留 frozen-cache dual、action-only
CE+flow、physics shuffled 三个 controls；正式实现前先审计现有 VLM token-logit 接口、loss 路由和可训练参数，
避免错误地把“当前两个离线离散条件 + 连续头”称为三头联合训练。

### 21.61 D50 三头联合接口与 2-step smoke（2026-09-28）

代码审计确认此前并不是真正的三头联合训练：`QwenDualTokenProbe` 能输出 action/physics logits 和两个 CE，
`QwenPIv4PhysicalTokens` 能让连续 Pi/DiT 消费缓存 IDs，但两者是独立 framework。现新增
`QwenPIv4JointTokens`，一次 VLM forward 的 hidden state 同时进入：（1）4x128 action token predictor；
（2）以 action soft context 为条件的 256x256 physics token predictor；（3）成熟 Pi continuous flow head。
连续头仍使用 D48 的 action/physics 两次独立 residual attention，physics raw256 聚合为 memory8。

joint framework 支持两种明确区分的条件源：`cached` 用 cache IDs teacher-force 连续头；`predicted` 用当前 VLM
两个 predictor 的 argmax IDs 在线条件连续头。`predicted` 已实现真实前向闭环，但 argmax 是非可导的，所以 flow
loss 不会穿过 ID 选择器反传到 token logits；token predictor 当前由 action/physics CE 更新。后续若 hard-token
curve 成立，再单独比较 straight-through/soft embedding，不能把当前实现描述成 flow 对 tokenizer 的端到端梯度。

初始化严格分三段并已从日志核验：成熟 100k Pi checkpoint 加载全部原参数，仅允许 joint extension missing；
D31 full-diversity 8000-step checkpoint 只加载 token predictor 的 13 tensors，不覆盖成熟策略 VLM；EMA action/
physics codebook 先初始化 conditioner，最后由 D49 memory8-100 checkpoint 精确覆盖 conditioner + 两套 gates 共
66 tensors。主干 VLM、prefix projection 和 Pi expert 均冻结。loss 权重初设 action CE `1e-3`、physics CE
`1e-4`，使约 4--6 的 CE 不会压过约 `1e-2` 的 flow loss。

node3 七卡的 cached 与 predicted 2-step smoke 均成功、无 OOM，实际每卡总显存约 19.7--20.2GB。cached 的
step1→2 为 flow `0.0098367→0.0059559`、action CE `4.7578→4.6641`、physics CE
`5.4761→5.3958`；predicted 为 flow `0.0098139→0.0059479`，两个 CE 相同。predicted 与 cached flow 已出现
小而有限的差异，证明在线预测 IDs 确实改变了连续条件，而不是旁路未接通。两份 checkpoint 都精确含 79 个
trainable tensors、大小约 143MB；SHA256 分别为
`b7780fe7b95e5d421e533961ff4f0badffce53c7c652e7013ddba5986d7d38ec` 与
`ed32f163af9747396fa9e3fcd5d6cf15c81bb372b01b8b8a32c9a9e91eacfc59`。

下一步先固定所有初始化、loss 权重、数据与 seed，跑 cached/predicted 各 20 steps，检查三项 loss 曲线和
predicted 条件是否造成连续头明显退化；随后增加 token accuracy/per-position usage 评估。只有在线 hard-token
分支稳定后才解冻 VLM 最后层或改 soft/ST 条件，避免一次引入条件源、梯度路径和主干遗忘三个变量。

D50 cached/predicted matched 20-step 已完成。两个分支的 action/physics CE 逐 step 完全相同，20-step 均值为
`4.4541/4.7267`；这是 hard argmax 阻断 flow→logits 梯度后的预期结果。predicted/cached flow loss 均值为
`0.017619/0.017555`，predicted 仅高约 `0.36%`，训练未崩坏。但允许已成熟 policy extension 再以 conditioner
LR `1e-4`、gate LR `1e-2` 更新会造成 gates 漂移：cached action/physics gate mean absolute 达到
`0.19268/0.18041`，而 D49 初始化为约 `0.16355/0.13007`。其 held-out MAE 因此不能用于隔离在线 token 质量。

D51 将 conditioner 和两套 gate 的 LR 全设为 0，只训练 action/physics predictor；保存后核验 66 个 policy
extension tensors 与 D49 memory8-100 checkpoint 逐位完全相等。D51 checkpoint SHA256 为
`5a205953376e2144417ea72c3b76a60e92850b83a607658a6919816b5b62fe37`。同一 checkpoint、同一 panel 仅切换
condition source 后，`pick_dual_bottles` cached/predicted MAE 为 `0.02232615/0.02234349`，predicted 相对
cached 为 `-0.07763%`，1/5 episode wins，95% CI `[-0.27469%,+0.11526%]`，没有显著退化。当前 hard-token
闭环因此通过“可训练、可推理、短程不显著伤害”的最低检查。

复现审计发现历史 D49 artifact 的 MAE `0.02085338` 在当前代码/运行环境不能逐值复现；用原 D49 physical class
和原 D49 checkpoint 重新跑相同 64-sample panel 得到 `0.02232615`，与 D51 joint cached 逐项完全相同。说明
joint cached 路径已恢复原 physical 行为，但跨代码时刻的绝对 MAE 存在推理可复现性漂移。后续结论只采用同一
代码时刻、同 checkpoint 或严格 matched checkpoint 的配对比较，并保留旧 artifact 但不混用绝对数值。

D51 新增 held-out token evaluator。`pick_dual_bottles` 64 samples 上，action token accuracy 为 `26.5625%`，
四个 positions 分别为 `29.6875/10.9375/46.8750/18.7500%`，sequence exact 为 0；预测/目标使用
`17/9` 个 action codes。physics token accuracy 为 `18.8843%`，256-position sequence exact 为 0，单位置
accuracy 范围 `0--60.9375%`；预测/目标使用 `169/249` 个 physics codes。两者显著高于随机 chance
`1/128=0.78125%` 与 `1/256=0.390625%`，证明 predictor 有真实对应能力，但 physics code coverage 收缩且
action position 1（第二个 token）是明显短板。下一步保持 policy LR=0，把 predictor-only 训练扩到 100 steps，
复测 token accuracy；只有 accuracy 提升后才重新评估在线 continuous MAE，避免用动作头指标盲调 predictor。

### 21.62 D51 100-step predictor curve、hybrid attribution 与 PhiZero 初始化审计（2026-09-28）

当前 token 数量口径再次固定，避免把 sequence positions 与 codebook size 混淆：action tokenizer 每个 32-step
action chunk 输出 **4 个 token positions**，每个位置从 **128 类 codebook** 中取值；physics tokenizer 输出
**256 个 raw token positions**，每个位置从 **256 类 codebook** 中取值；进入连续 Pi 动作头前，physics256
按 group size 32 聚合为 **8 个 policy memory positions**。`action-only` 指启用 4 个离散 action tokens、禁用
physics tokens；`pure pi/e0` 才指两类离散条件都关闭。当前 action tokenizer 是在 RoboTwin 多任务 action chunks
上训练的 EMA-VQ，不是 pi0/pi0.5 的公开 tokenizer。

D51 predictor-only 100-step 训练在 node3 GPU0--6 正常完成且未 OOM。checkpoint 为
`results/Checkpoints/qwenpi_v4_d50_frozen_policy_predicted_seed42_steps100/checkpoints/steps_100_pytorch_model.pt`，
SHA256 为 `12fb3810ad6ac7b623fe137e8ddd0f78f3c844d4c8ecb1e3f028c7ceeeda9246`。它与 D49 memory8-100
checkpoint 共有的 conditioner、28 个 action gates 和 28 个 physics gates 共 66 tensors 全部 bitwise equal，确认
连续策略没有发生训练漂移；只有 13 个 predictor tensors 更新。

同一 `pick_dual_bottles` 64-sample held-out panel 上，20→100 steps 的 action token accuracy 从
`26.5625%` 提升到 **`32.4219%`**，四个位置从
`29.6875/10.9375/46.8750/18.7500%` 提升到
`32.8125/15.6250/50.0000/31.2500%`；sequence exact 从 `0/64` 提升到 `1/64`，预测/目标 action
code coverage 为 `16/9`。physics token accuracy 从 `18.8843%` 提升到 **`25.7080%`**，单位置最高 accuracy
从 `60.9375%` 提升到 `68.7500%`，但预测/目标 physics code coverage 从 `169/249` 进一步收缩为
`151/249`。因此更长 predictor 训练提高对应准确率，但 physics mode contraction 同时加重。结果位于
`artifacts/evaluations/d51_frozen_policy_steps100_pick_dual_bottles_64samples/token_accuracy.json`。

满足预注册的 accuracy 提升条件后，用同一 checkpoint、同一 panel、相同 diffusion seeds 做 cached/predicted
连续动作评估。cached/predicted action MAE 为 `0.0223261537/0.0223506062`，predicted 相对 cached 为
`-0.10952%`，1/5 episode wins，paired episode bootstrap 95% CI
`[-0.32014%,+0.11502%]`；结果仍是统计不显著的小幅退化。相比 20-step 的 `-0.07763%`，token accuracy
提高尚未转化为连续动作收益，不能据此宣称三头方法优于 teacher-forced 条件。

为定位两路误差，joint framework 新增两个只改变推理 ID 来源的 hybrid modes，并保持原 `cached/predicted`
行为不变：`predicted_action_cached_physics` 与 `cached_action_predicted_physics`。四种 ID 路由和 physics-disable
边界测试均通过，相关 physical/joint/evaluator 定向测试合计 `12 passed`。同一 panel 上，仅预测 action 的 MAE
为 `0.0223387309`，相对全 cached 为 `-0.05633%`，95% CI `[-0.21317%,+0.10110%]`；仅预测 physics
的 MAE 为 `0.0223357935`，相对全 cached 为 `-0.04318%`，95% CI
`[-0.12906%,+0.03303%]`。两路 action-MAE 影响都很小且不显著，全 predicted 退化近似为二者叠加。
但仅预测 physics 使 joint-velocity MAE 显著退化 `0.18194%`，95% CI
`[-0.29656%,-0.07839%]`，而仅预测 action 对该指标改善 `0.03108%`，说明 physics code contraction/时序表达
是更明确的局部瓶颈。完整结果位于
`artifacts/evaluations/d51_frozen_policy_steps100_pick_dual_bottles_64samples/`。

PhiZero 本地仓库审计说明为何当前没有直接复用其 tokenizer。PhiZero 的 Physical Language Tokenizer 针对
4 秒、8 FPS、33-frame、896x512 视频，以未来视频重建训练，固定输出 256 个 FSQ symbols、codebook size
25,000；其 checkpoint 同时包含 tokenizer、inverse-FSQ/context conditioner 和用于 Wan2.2 video DiT 的 LoRA。
这与本项目的 32-step robot action/state 表征、256 类本域 codebook 和 Pi action expert 不同，code ID 没有可直接
对齐的语义。PhiZero 官方 README 也明确声明通用 `tokenizer.safetensors` 不能直接用于 robot-domain 或
sim-to-real motion transfer，必须先用同域 source videos 做 reconstruction adaptation。

因此将 PhiZero 纳入实验的正确 control 是：下载官方 `tokenizer.safetensors`、`model_config.json` 和 Wan2.2
基座后，把 RoboTwin episode 转成严格的 33-frame/8-FPS clips；比较（A）PhiZero 通用初始化后同域 reconstruction
fine-tune 与（B）相同 robot-video 数据从头训练/本域 EMA，并分别报告 reconstruction、FSQ usage/perplexity、
held-out temporal correspondence，最后再决定是否蒸馏/映射进 policy memory。不能直接把 PhiZero 的 25k IDs
替换当前 256 IDs。当前本地 PhiZero repo 只有代码、没有上述 checkpoint 或 Wan 权重，这条消融暂不阻塞 D51
主线。

下一步 D52 保持 D49 连续策略完全冻结，从相同 D31 predictor 初始化把 predictor-only curve 扩到 500 steps，
而不是从 100-step resume，以保证 20/100/500 曲线可比。训练已在 node3 GPU0--6 启动，run ID 为
`qwenpi_v4_d50_frozen_policy_predicted_seed42_steps500`。完成后先核验 66 个 policy tensors bitwise equal，再复测
token accuracy、四个 action positions 和 physics code coverage。晋级条件不是 accuracy 单独提高：还要求 physics
coverage 不继续严重收缩，且同 checkpoint predicted-vs-cached continuous gap 至少不比 100-step 更差。若 500-step
accuracy 上升但 coverage/continuous gap 恶化，则判定 hard-CE 延长训练无效，下一步优先做 physics usage-balanced
CE 或 entropy regularization；若 action position 1 仍显著落后，再做 per-position weighted CE。此时仍不解冻完整
VLM/策略，也不直接进入 soft/ST conditioning。

补充严格 baseline 审计：仅与均匀随机 chance 比较不足以证明 predictor 学到了 sample-conditional 信息，因为 token
target 明显不均衡。对 `pick_dual_bottles` 训练 episodes 按每个 token position 统计最常见 code，并把这个不看图像、
不看状态的 train-derived position-majority predictor 应用到同一 64-sample held-out panel，action/physics accuracy
分别为 **`41.0156%/28.6255%`**；均高于当前 100-step predictor 的 `32.4219%/25.7080%`。action 四个位置的
majority baseline 为 `25.0000/31.2500/54.6875/53.1250%`，当前 predictor 只有第 0 个位置超过 baseline，
第 1/2/3 个位置仍低。physics 训练目标本身也高度偏斜：全八任务训练 split 虽使用全部 256 codes，但 aggregate
perplexity 仅 `133.999`，top-1 code 占 `18.341%`；这解释了高于 `1/256` 随机率并不是充分证据。

因此修正此前解释：20/100-step predictor 的 raw accuracy 证明它没有随机崩坏，但尚不能证明超过简单先验；真正的
sample-conditional 预测门槛应是 train-derived per-task/per-position majority。D52 500-step 只有 action 与 physics
都超过各自 majority baseline，或至少连续动作 hybrid 指标出现明确改善，才允许进入 soft/ST。若仍低于 baseline，
优先检查 CE 优化、target imbalance 与 probe capacity，而不是延长 hard-token policy 联训。oracle/cached physics
相对 shuffled 的 correspondence 结论不受此审计影响，因为那是同 checkpoint 下正确配对关系的独立检验。

最终 policy 对照不能只在 action-only 与 dual 之间进行。`pure pi/e0` 虽关闭 action/physics 两类离散 memory，仍
保留成熟主路径的视觉语言条件：当前样本的图像、任务文本和注入 instruction 的 robot state 一起通过 Qwen-VL，
最后层 per-token hidden states 经 `prefix_proj` 映射后，作为 Pi expert 各层 cross-attention 的 prefix；noisy action
trajectory 和 diffusion time 则走 action expert自身输入。离散 action/physics tokens 是附加在该 prefix 之外的两条
独立 residual cross-attention，而不是替代 prompt hidden state。D48 已包含 pure pi/action-only/dual 五路对照；
D52 500-step 在线预测正式评估仍须在同一代码时刻、checkpoint、panel 和 diffusion seeds 下至少报告：（1）pure
pi/e0；（2）predicted action-only，即 physics disabled；（3）dual cached teacher-force；（4）dual predicted。
physics-only 可作为 gate attribution 诊断，但不是主方法或必要主 baseline。最终是否有帮助必须看 dual predicted
相对 pure pi 和 predicted action-only，而不能只看 predicted 相对 cached 的误差。

### 21.63 D52 500-step token gate 与完整 policy baseline 矩阵（2026-09-28）

D52 predictor-only 500-step 在 node3 GPU0--6 完成，用时约 7 分 24 秒且未 OOM。checkpoint SHA256 为
`f1591e1877f78798488f5bea17202ff5cee815c6a185e05d9849fcad4a385c29`；与 D49 memory8-100 共有的 66 个
conditioner/gate tensors 全部 bitwise equal，再次确认连续策略完全冻结。标准 token evaluator 已加入
train-derived per-task/per-position majority baseline，并新增对应单测；joint hybrid route、physical route 和 evaluator
本轮定向测试通过（最近两组分别为 `12 passed` 与 `6 passed`）。

同一 `pick_dual_bottles` 64-sample panel 上，500-step action accuracy 为 **`48.0469%`**，超过 train-derived
position-majority 的 `41.0156%` 共 `7.0313` 个百分点；四个位置 accuracy 为
`34.3750/39.0625/62.5000/56.2500%`，均超过各自 majority
`25.0000/31.2500/54.6875/53.1250%`。action sequence exact 达到 `17/64=26.5625%`，预测/目标 code
coverage 为 `13/9`。physics accuracy 为 **`38.5437%`**，超过 majority `28.6255%` 共 `9.9182` 个百分点；
预测/目标 code coverage 为 `160/249`，相比 100-step 的 `151/249` 略有恢复，但仍存在 coverage gap。结果位于
`artifacts/evaluations/d52_frozen_policy_steps500_pick_dual_bottles_64samples/token_accuracy.json`。

因此 500-step 首次证明两路 predictor 在该 held-out panel 上都获得了超过静态任务/位置先验的 sample-conditional
信息，修复了 20/100-step 只高于均匀随机但低于 majority 的问题。不过 token accuracy 仍不能替代 policy 指标。
当前已在同一 checkpoint/panel/seeds 上并行启动四路正式评估：pure pi/e0、predicted action-only、dual cached、
dual predicted。主问题是 dual predicted 是否同时优于 pure pi 与 predicted action-only；dual cached 只提供
teacher-forcing upper/reference condition，不作为部署方法的唯一 baseline。

四路 `pick_dual_bottles` 连续动作评估已完成。pure pi/e0、predicted action-only、dual cached、dual predicted
的 action MAE 分别为 `0.0223840065/0.0223893530/0.0223261537/0.0223601704`。predicted action-only 相对
pure pi 为 `-0.02389%`，2/5 wins，95% CI `[-0.10335%,+0.07372%]`，说明 action token 单独没有收益。
dual predicted 相对 pure pi 改善 **`0.10649%`**，5/5 wins，95% CI
**`[+0.07748%,+0.13109%]`**；相对 predicted action-only 改善 **`0.13034%`**，4/5 wins，95% CI
**`[+0.00376%,+0.23212%]`**。这是当前首次在保留成熟 visual-language prefix、两类 token 均由 VLM 在线
预测的部署路径上，同时显著优于无离散 token 的 pure Pi 与 predicted action-only，支持 physics token 提供了 action
token 之外的增量信息。

dual predicted 仍比 dual cached 差 `0.15236%`，1/5 wins，95% CI
`[-0.27599%,-0.03123%]`，说明 predictor 与 teacher-forced upper/reference condition 之间仍有空间。当前正结果的
效应量只有约千分之一，且来自此前用于 correspondence/count 选择的 `pick_dual_bottles` 单任务离线 MAE，不能直接
外推为 RoboTwin rollout success 提升。为控制任务选择偏差，下一步保持同 checkpoint/panel/seeds，在
`click_alarmclock` 与 `open_microwave` 复测 pure pi、predicted action-only、dual predicted 三路；跨任务结果出来
前不宣称方法普遍提升。

两项跨任务复核已完成，结果不支持“当前版本普遍提升”。`open_microwave` 的 pure pi/action-only/dual predicted
MAE 为 `0.0078786069/0.0078839523/0.0078606500`。dual predicted 相对 pure pi 改善 `0.22792%`，3/5
wins，但 95% CI `[-0.07656%,+0.48763%]` 跨 0；相对 action-only 改善 **`0.29557%`**，5/5 wins，95% CI
**`[+0.13053%,+0.57090%]`**。该任务 action/physics accuracy 为 `34.3750/31.9580%`，明显超过各自
train-derived majority `16.0156/18.3472%`，分别高 `18.3594/13.6108` 个百分点。

`click_alarmclock` 的 pure pi/action-only/dual predicted MAE 为
`0.0110419686/0.0110426597/0.0111059821`。dual predicted 相对 pure pi 为 `-0.57973%`，1/5 wins，95% CI
`[-1.65530%,+0.17902%]`，方向为退化但不显著；相对 action-only 为 `-0.57343%`，CI 同样跨 0。该任务
action accuracy `35.1563%` 低于 majority `38.2813%`，physics accuracy `41.0461%` 与 majority
`41.0706%` 几乎相同，说明 predictor 尚未获得可靠的 sample-conditional 增益。

当前三任务呈现一致的诊断相关性：pick-dual 两路 predictor 超过 majority，dual predicted 显著优于 pure pi 与
action-only；open 两路明显超过 majority，dual predicted 显著优于 action-only、相对 pure pi 正向但不显著；click
未超过 majority，dual predicted 无收益。这个相关性尚不是因果证明，但把下一瓶颈指向 token prediction 的任务覆盖，
而不是优先继续改 Pi/DiT 融合。已启动剩余五个训练任务的 500-step predictor-vs-majority 评估；在看到八任务全貌
前不解冻连续策略，也不启动 soft/ST。

八任务 500-step predictor-vs-majority 评估已完成（数值为 model accuracy / train-derived position-majority，括号
为百分点差）：

| Task | Action | Physics | Predicted/target physics codes |
|---|---:|---:|---:|
| click_alarmclock | 35.156/38.281 (-3.125) | 41.046/41.071 (-0.024) | 188/251 |
| beat_block_hammer | 15.625/16.406 (-0.781) | 32.062/26.208 (+5.853) | 215/252 |
| open_microwave | 34.375/16.016 (+18.359) | 31.958/18.347 (+13.611) | 202/247 |
| shake_bottle | 36.719/23.438 (+13.281) | 33.032/32.208 (+0.824) | 205/256 |
| move_playingcard_away | 20.703/21.875 (-1.172) | 34.290/34.088 (+0.201) | 204/256 |
| pick_dual_bottles | 48.047/41.016 (+7.031) | 38.544/28.625 (+9.918) | 160/249 |
| handover_block | 17.578/20.703 (-3.125) | 26.331/15.302 (+11.029) | 226/253 |
| place_object_basket | 21.875/12.891 (+8.984) | 33.478/23.688 (+9.790) | 188/245 |

action predictor 超过 majority 的任务为 4/8，physics 为 7/8，但 physics 在 shake/move 上仅高约 0.8/0.2 个
百分点，不能视为强预测。两路都明显超过先验的是 open、pick-dual、place-object；click 两路均失败；beat/handover
是 physics 强而 action 弱；shake 是 action 强而 physics 接近先验。该异质性说明 500-step probe 的任务覆盖仍是
主要问题。为避免只选择正向任务，已开始补齐其余五任务的 pure pi/action-only/dual predicted 连续评估，最终报告
八任务逐任务与 macro，而不是以单任务决定方法成立。

八任务 pure pi/action-only/dual predicted 矩阵现已补齐。下表报告 action MAE，以及括号内 candidate 相对 pure pi
的 improvement；正数为改善：

| Task | Pure Pi/e0 | Predicted action-only | Dual predicted |
|---|---:|---:|---:|
| click_alarmclock | 0.01104197 | 0.01104266 (-0.0063%) | 0.01110598 (-0.5797%) |
| beat_block_hammer | 0.01210974 | 0.01211727 (-0.0622%) | 0.01211612 (-0.0527%) |
| open_microwave | 0.00787861 | 0.00788395 (-0.0678%) | 0.00786065 (+0.2279%) |
| shake_bottle | 0.01564041 | 0.01560361 (+0.2353%) | 0.01562315 (+0.1103%) |
| move_playingcard_away | 0.01466356 | 0.01468629 (-0.1550%) | 0.01468184 (-0.1246%) |
| pick_dual_bottles | 0.02238401 | 0.02238935 (-0.0239%) | 0.02236017 (+0.1065%) |
| handover_block | 0.01117177 | 0.01113052 (+0.3693%) | 0.01109469 (+0.6899%) |
| place_object_basket | 0.01388337 | 0.01389259 (-0.0664%) | 0.01390786 (-0.1764%) |

dual predicted 相对 pure pi 为 4/8 tasks 正向：pick-dual 与 handover 的 paired episode bootstrap CI 显著为正，
place-object 显著为负，其余跨 0。相对 predicted action-only 为 5/8 正向：open、pick-dual、handover 显著为正，
place-object 显著为负，其余跨 0。八任务 mean relative improvement 中 action-only vs pure pi 为 `+0.02789%`，
dual predicted vs pure pi 为 `+0.02516%`，dual predicted vs action-only 为 `-0.00264%`；equal-task pooled MAE 的
对应改善为 `+0.02501%/+0.02112%`。因此当前 dual 方法在 aggregate 上与 action-only 实质持平，不能宣称总体
胜出，但在部分任务存在可重复的 physics 增量。

predictor accuracy 与 policy usefulness 相关但不是充分条件：click 两路未超过 majority 且 dual 退化；open/pick
两路超过且 physics 相对 action-only 显著增益；shake 的 physics 接近 majority，加入后削弱 action-only 收益。
然而 place-object 两路均明显超过 majority，dual 仍显著退化；handover 的 action 低于 majority、physics 很强，
dual 却显著改善。这说明还需区分 predictor error 与 cached/oracle physics 本身的任务适配。已为除 pick-dual 外
七个任务启动 dual cached 评估；若 cached 也负向，优先解决 representation/global gate；若 cached 正向而 predicted
负向，优先解决 predictor/confidence。该诊断比立即增加 soft/ST 或统一 confidence threshold 更有因果识别力。

八任务 dual cached 诊断已完成。cached 相对 pure pi 在 7/8 tasks 正向，mean relative improvement 为
**`+0.25126%`**；beat-block、pick-dual、handover 的 paired CI 显著为正，没有任务显著为负。逐任务
cached-vs-pure improvement 为：click `-0.3227%`、beat `+0.5247%`、open `+0.0729%`、shake `+0.5097%`、
move-card `+0.0246%`、pick-dual `+0.2585%`、handover `+0.8356%`、place-object `+0.1067%`。这说明在
teacher-forced 正确 ID 下，当前 physical representation、memory8 分离 attention 与成熟 gates 在总体上有用，主问题
不是 Pi/DiT 没接收到 prompt hidden 或 physics memory 完全无效。

相反，dual predicted 相对 dual cached 的八任务 mean relative improvement 为 **`-0.22687%`**，仅 open 一个任务
数值正向且不显著；beat、shake、pick-dual、place-object 显著退化，click/move/handover 也为负但 CI 跨 0。尤其
place-object cached 相对 pure pi 为 `+0.1067%`，predicted 相对 cached 为 `-0.2833%`，解释了即使 raw token
accuracy 超过 majority，错误 token 的动作相关代价仍可能很高。结论是 hard argmax predictor/condition transfer
目前是主要瓶颈，raw top-1 accuracy 不是足够的 policy usefulness 指标。

下一低成本实验应在保持 checkpoint 和连续策略完全冻结时比较 hard argmax 与 soft expected embedding。soft 路径
用 `softmax(logits / T) @ conditioner_embedding` 保留不确定性，再加 position/type、LayerNorm/gate，并继续沿 D48
action/physics 独立 residual attention 注入；先只做 inference，不立即让 flow loss 反传。温度必须预注册或在与最终
panel 隔离的 calibration episodes 上选择，不能在八任务结果上逐任务挑最优 T。若统一 soft 路径不能改善 hard
argmax 的八任务 macro，则停止该方向，再考虑 cost-sensitive CE/最终 VLM layer；若改善，再讨论 ST 和端到端训练。

soft expected-embedding inference 路径现已实现。`DiscreteTokenConditioner` 新增 action/physics categorical
probability 输入：分别计算 `probability @ embedding.weight`，再加入原 position/type embedding，复用原
LayerNorm、conditioner gate、physics memory8 grouping 和 D48 两路独立 residual attention。joint framework 新增
`condition_source=soft` 与正数 `soft_temperature`；评估 JSON 现在显式保存完整 `joint_token_prediction` 配置，修复
过去 cached/predicted CLI override 未直接出现在报告字段中的复现缺口。one-hot soft distribution 与 hard ID 的
action/physics embedding 逐值一致，conditioner/joint/evaluator 定向测试 `23 passed`，Python compile 通过。

首轮预注册统一 `T=1.0`，不在八任务结果上逐任务挑温度，也不做任何参数更新。八任务相同 64-sample panels 的
`dual_soft_t1` 已在 node3 八卡并行启动。判断顺序为：（1）soft vs hard predicted 八任务 macro；（2）soft vs pure
pi 与 action-only；（3）逐任务是否减少 hard 的显著负向。只有统一 T 明确改善，才另设 calibration split 选择温度；
否则不在当前 validation panels 上事后扫温度。

统一 `T=1.0` 八任务结果支持 soft expected embedding。soft 相对 hard predicted 的 mean relative improvement 为
**`+0.32495%`**，5/8 tasks 正向；beat、pick-dual、handover、place-object 的 paired CI 显著为正，没有任务
显著为负。soft 相对 pure pi 的 mean improvement 为 **`+0.34964%`**，6/8 正向；beat、pick-dual、handover
显著为正，没有显著负向。soft 相对 predicted action-only 的 mean improvement 为 **`+0.32224%`**，5/8
正向；beat、pick-dual、handover、place-object 显著为正，没有显著负向。

逐任务 soft-vs-hard improvement 为：click `+0.4589%`（CI 跨 0）、beat `+0.6534%`（显著）、open
`-0.1281%`（跨 0）、shake `-0.0554%`（跨 0）、move-card `-0.1243%`（跨 0）、pick-dual `+0.4062%`
（显著）、handover `+1.0259%`（显著）、place-object `+0.3629%`（显著）。此前 hard dual 在 place-object
相对 pure/action-only 均显著退化；soft 后相对 action-only 改善 `+0.2534%`，4/5 wins，95% CI
`[+0.07920%,+0.36802%]`，成功修复该失败任务。

soft 相对 dual cached 的八任务 mean improvement 仍为 `+0.09907%`，6/8 数值正向；pick-dual 与 handover
显著优于 cached，shake 显著低于 cached，其余 CI 跨 0。soft 超过单一 cached ID 不矛盾：EMA token target 本身
是有损量化，概率混合可表达 code embedding 的局部期望，也可能保留未来多模态不确定性；但该结果仍只是离线
action reconstruction，不能把 soft 称为更准确的物理语言。当前最强部署候选从 hard argmax 更新为统一 T=1 的
soft expected embedding，且未使用逐任务温度选择。

### 21.64 D52 soft-T1 真实 RoboTwin rollout 接口审计（2026-09-28）

离线 panel 通过后，已把 joint extension 接入标准 websocket policy server。D52 checkpoint 只保存 79 个 extension
tensors，部署入口因此先加载 D49 成熟 Pi base，再严格要求 D52 checkpoint 的 key 集合与全部 joint extension keys
完全一致；服务器支持 `soft/predicted/cached` condition source、统一 soft temperature、关闭全部 discrete memory 和
只关闭 physics memory。soft `T=1.0` server 已在 node3 GPU7 成功加载并监听，模型端首个请求能正常生成
`[1,32,14]` 动作 chunk，证明 joint soft conditioner、两路 residual attention 和 Pi flow sampling 的在线链路可执行。

第一次 simulator smoke 在执行动作前暴露出部署协议不一致，而不是模型推理错误：RobotWin wrapper 已负责把原始
robot state 归一化并把模型输出反归一化为 `actions`，但 PhysToken client 仍按旧协议先归一化 state、再只读取
`normalized_actions`。这不仅触发 `KeyError`，也意味着若只改响应字段会留下隐蔽的 state 二次归一化。现已按 server
metadata 明确职责：`env=starvla_policy_server` 时，client 发送未经归一化但已从 runtime order 换到 model order 的
state，并传递 `unnorm_key`；server 唯一负责 state normalization/action unnormalization；client 读取 raw `actions`
后只做 model-order 到 RoboTwin runtime-order 的转换。legacy server 路径仍保留原本的 client-side normalization。
同时保留 aspect-ratio image buckets、control-frequency instruction 和 action postprocess 行为。

新增 client/server contract 回归测试，检查发送 state 没有被 client 预归一化、`unnorm_key` 被传递、返回 raw action
只发生一次正确的左右臂/gripper 次序转换；与 joint soft 定向测试合计 `4 passed`。修复后的
`pick_dual_bottles/demo_clean/seed0` 三回合 soft-T1 rollout 已在 node3 GPU7 完成，峰值总显存约 35 GB、未 OOM，
三个 episodes 均完整执行 400 steps，success 为 **`0/3`**。结果文件为
`RoboTwin_eval/eval_result/pick_dual_bottles/model2robotwin_interface/demo_clean/d52_soft_t1_smoke_v2/2026-09-28 19:31:21/_result.txt`。
该结果确认 websocket、预处理、soft joint inference、动作次序转换和 simulator 执行链路能够端到端运行，但 3 episodes
只属于接口 smoke，不能用于方法优劣判断，也不能把离线 MAE 的千分之几改善解释成 rollout success 改善。

为区分“soft condition 伤害”与“当前成熟 Pi/D49 policy 本身尚无闭环成功率”，已在完全空闲的 node4 GPU0/GPU1
并行启动相同 task/config/seed/3 episodes 的 pure Pi（两类 discrete condition 均关闭）和 predicted action-only
（physics condition 关闭）对照；单路模型加 simulator 约 16.7 GB，未 OOM。soft rollout 完成后已停止仅属于本实验的
node3 server 并释放其额外显存。只有匹配对照完成后，才能判断下一步应优先修连续 policy/deployment baseline，还是
继续优化 token condition；即便三路出现 0/3 与非零差异，3 episodes 也只用于选择下一步而不作正式统计声明。

三路 matched smoke 已完成：pure Pi、predicted action-only、soft dual 分别为 **`0/3`、`1/3`、`0/3`**。
action-only 的唯一成功发生在第一个 matched seed `100000`，seed `100001/100002` 失败；pure 与 soft 三个 seed 均
失败。action-only 与另两路只差一个 episode，样本量远不足以声称 action token 提升或 physics token 伤害。三份
结果文件分别位于 setting `d52_pure_pi_smoke`、`d52_action_only_smoke`、`d52_soft_t1_smoke_v2`；完成的三个专用
server 均已停止，node4 GPU0/GPU1 和 node3 soft server 占用已释放。

基座复核确认 D52 的 `trainer.pretrained_checkpoint` 是官方 RoboTwin-all 下游 finetune 的
`VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`，不是只有预训练、未做下游动作学习的
模型；D52 只额外加载 D49/D52 extension，连续 Pi 权重冻结且与该基座一致。另一方面，现有 RoboTwin 历史结果中，
另一个 100k policy 在 `pick_dual_bottles` 上两次均为 `0.0`，说明该任务在小样本 rollout 中存在明显 floor effect。
因此不应立即把 3 回合扩成同任务 10/20 回合浪费预算。下一轮改用历史上更容易获得非零成功率、且本项目八任务
离线 panel 已覆盖的 `click_alarmclock`，对 pure/action-only/soft 继续做同 seed 三路 smoke。若 pure baseline 在该
任务仍为 0/3，再优先做官方 base checkpoint 与 joint-disabled 路径的逐动作等价/标准部署校准；若 baseline 非零，
再扩 matched episodes 判断 soft physics 是否有真实闭环增益。

`click_alarmclock/demo_clean/seed0` 三路 matched smoke 已完成，pure Pi、predicted action-only、soft dual 的 success
分别为 **`1/3`、`1/3`、`2/3`**。RoboTwin expert precheck 会跳过不稳定 scene seed，因此三路实际评测 seeds 均为
`100000/100002/100003`，仍然严格配对：`100000` 三路均失败；`100002` 只有 soft dual 成功；`100003` 三路均
成功。也就是说，soft physics condition 在这组三个 episode 上相对 pure 与 action-only 各增加一个成功，action token
单独没有改变 episode outcome。这与该任务离线 panel 中 soft 相对 hard 改善 `+0.4589%`、但 CI 跨 0 的方向一致，
但 `n=3` 不能形成统计结论，尤其不能据此报告成功率提升 `33.3` 个百分点为可泛化效应。

该结果比 `pick_dual_bottles` 的 floor-effect smoke 更有信息量：同一部署链路下 pure baseline 已出现非零成功，说明
joint-disabled 路径和官方 100k RoboTwin Pi base 至少能够完成任务；soft 的额外成功也说明物理条件没有在所有闭环
episode 上破坏控制。下一步使用 `seed=1` 生成新的 scene-seed 区间（从 `200000` 起），重复三路各 3 episodes，
不重复 seed0 的三个场景。只有独立 seed 组方向仍支持 soft，才扩到更大 matched rollout；若不复现，则按小样本高
方差处理，仍以离线八任务结果为候选筛选证据而不宣称 benchmark 改善。

独立 `seed=1`（实际 scene seeds `200000/200001/200002`）复现完成：pure Pi、predicted action-only、soft dual
分别为 **`1/3`、`3/3`、`1/3`**。逐 seed 上，`200000` 三路均成功，`200001/200002` 只有 action-only
成功。与 seed0 合并为 6 个严格配对 episodes 后，pure/action-only/soft success 为 **`2/6`、`4/6`、`3/6`**。
soft 相对 pure 有 1 个 discordant win、0 个 discordant loss，其余 5 个相同；soft 相对 action-only 有 1 win、2 losses、
3 ties；action-only 相对 pure 有 2 wins、0 losses。如此小的 discordant counts 无法提供显著结论，但排序上 action-only
暂时最好，soft 只表现为相对 pure 的弱正向迹象，不能再用 seed0 的 `2/3 vs 1/3` 单独宣称 physics rollout 增益。

该结果也与离线诊断并不矛盾：`click_alarmclock` 的 action/physics predictor 都没有超过 majority baseline，hard dual
曾明显退化；soft 虽修复大部分 hard 伤害，但离线相对 action-only 并没有建立强优势。因此下一步不继续在同一任务
追加大量 episode，而选择 `open_microwave` 做跨任务 matched smoke：该任务两路 predictor 明显超过 majority，hard
dual 离线相对 action-only 显著正向，历史同类 100k policy 的 rollout success 约 `0.4--0.55`，同时避免
pick-dual/handover/place-object 的 floor effect。仍先做三路各 3 episodes；若 soft 相对 action-only 在这一更有利条件
下仍无增益，后续重点应转向 action-only + cost-aware/confidence physics routing，而不是无条件注入 physics memory。

`open_microwave/demo_clean/seed0` 三路 smoke 最终均为 **`3/3`**。这提供两个边界结论：（1）在非 floor 任务上，
pure Pi base 可以稳定完成任务；（2）action-only 与 soft dual 均未在这三个回合造成明显闭环破坏。但三路都满分产生
ceiling effect，无法检验 physics 增益。结果不能与 `click_alarmclock` 直接合并，因为任务难度和 episode 上限不同。

本轮同时发现正式 rollout 协议必须修正：虽然三路都传入 CLI `seed=0`，RoboTwin 会先运行 expert trajectory 检查并
跳过不稳定 scene seed；并发仿真下三路实际接受的 seeds 不一致。pure 为 `100000/100003/100008`，action-only 为
`100000/100001/100003`，soft 为 `100000/100002/100003`。因此只有 seed `100000/100003` 可作部分配对，原始
`3/3` 只能作为兼容性/无明显伤害 smoke，不能当作严格 matched 统计。`click_alarmclock` 两轮恰好得到相同 seed list，
其 6-episode 配对仍有效，但正式扩展不能依赖这种偶然一致。

下一工程步骤是在 PhysToken 仓库内增加固定 stable-scene-seed 的评测入口：先独立运行一次 expert precheck 生成并保存
task/config 对应 seed manifest，再让所有 policy modes 读取同一 manifest，禁止各自在线跳 seed；结果 JSON/文本同时
记录 requested/accepted seeds。不要直接修改共享 RoboTwin checkout 的默认行为，以免影响其他代码库。完成后优先在
中等成功率且 episode 较短的任务上做 paired rollout；`open_microwave` 可保留作 no-harm control，但 3/3 ceiling
不值得立即追加预算。

固定 seed-list 评测入口已在 PhysToken 仓库内实现，不修改共享 RoboTwin checkout。设置
`ROBOTWIN_FIXED_TEST_SEEDS='[100000,100002,100003]'` 后，`examples/Robotwin/eval_files/eval.sh` 自动调用
`eval_policy_fixed_seeds.py`：严格检查列表非空、全为整数、无重复且数量等于 `test_num`；每个 seed 仍做 expert
precheck，但任何失败都会立即报错，绝不替换为下一个 seed；完成后在结果目录写 `_fixed_seeds.json`，保存 seed 列表、
success 数和 episode 数。默认不设置该变量时仍使用原 RoboTwin evaluator，保持现有工作流不变。

新增 seed parser 和 client/server protocol 测试，与 joint soft 测试合计 **`5 passed`**；固定-seed evaluator 与 client
Python compile 通过，`eval.sh` 的 `bash -n` 通过。下一次扩大 rollout 时先用单独 seed-discovery pass 形成 manifest，
然后三种 mode 顺序或并行消费完全相同 manifest；若某个 manifest seed 在复跑 expert precheck 中不稳定，则丢弃整组
三路结果并重新生成 manifest，而不是只给失败的一路换 seed。

固定-seed 入口的首次端到端 smoke 还发现并修复了一个启动路径问题：从 PhysToken 文件路径启动时，RoboTwin 的
`test_render` 必须显式从 `script.test_render` 导入，不能依赖原 `script/` 入口自动加入 `sys.path`。修复后使用
`click_alarmclock`、pure Pi、固定 seed `[100003]`、1 episode 完整跑通；该 episode 在 80 steps 成功，日志明确
打印 `fixed seed: 100003`，结果目录生成 `_fixed_seeds.json`：`seeds=[100003]`、`successes=1`、`episodes=1`。
这证明固定 manifest 已覆盖真实 expert precheck、policy server、simulator 和结果落盘，而不只是 parser 单测。所有本轮
专用 servers/evaluators 已停止，node4 GPU0--2 显存恢复为空闲。

后续所有正式闭环对比必须同时保存逐 episode MP4，不能只保留 success rate。固定 seed evaluator 已增加视频完整性
校验：ffmpeg 非零退出、视频缺失或小于 1 KiB 都会使该次评测失败；`_fixed_seeds.json` 同时记录每个 episode 的
scene seed、success、视频文件名和字节数，便于 pure Pi/action-only/soft dual 按相同 seed 逐视频对照。此前
`click_alarmclock` 的两组三路 6-episode smoke 实际也已生成全部 18 个 MP4，分别保存在共享 RoboTwin_eval 的
`eval_result/click_alarmclock/model2robotwin_interface/demo_clean/d52_click_{pure_pi,action_only,soft_t1}[_seed1]_smoke/`
目录。连续 Pi 基座的实际权重为
`/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`
（约 11 GiB）；这是 RoboTwin-all 下游 finetune 100k checkpoint，不是 Qwen3 预训练权重。
固定 seed evaluator 进一步把完成的视频按 episode 结果重命名为 `episodeN_success.mp4` 或
`episodeN_failure.mp4`，manifest 中的 `video.file` 保存重命名后的文件名，可直接从目录列表区分成功和失败。
历史 `click_alarmclock` 两组三路对比的 18 个视频也已根据逐 seed 结果完成 success/failure 重命名，并生成
`docs/CLICK_ALARMCLOCK_VIDEO_COMPARISON.html`：每个 scene seed 一行，pure Pi、action-only、soft dual 三列并排，
支持逐行同步播放、暂停、复位和倍速切换；结果存在分歧的 seed 行使用浅黄色标题栏标识。

### 21.65 固定 seed 视频协议下的 `move_playingcard_away` 闭环对照（2026-09-28）

为避开 `pick_dual_bottles` 的 floor 与 `open_microwave` 的已知 ceiling，选择历史同类 100k policy 成功率约
`0.27--0.30`、step limit 400 的 `move_playingcard_away`，使用 D52 checkpoint 和已审计部署链路比较 pure Pi、
predicted action-only、soft dual T=1。两组 seed 在运行前分别预注册为 `[100000,100001,100002]` 和
`[200000,200001,200002]`；所有 policy 使用完全相同的 scene seeds，expert precheck 不允许替换。

结果出现新的 ceiling：两组共 6 个 episode 中，pure/action-only/soft 均为 **`6/6`**，没有任何 discordant
outcome。因此这组结果只能说明两类 token condition 没有破坏这些场景，不能支持 action token 或 physics token
提升 success rate。逐 seed 终止 control steps 为：pure `90/105/93/95/88/95`，action-only
`91/95/157/93/92/110`，soft dual `91/51/92/89/89/93`。soft 在 seed 100001 明显更早成功，但 step count
受轨迹与 success detector 共同影响，且 n=6，不作为性能结论。

六份 `_fixed_seeds.json` 均记录 3/3 success 和对应视频；18 个 MP4 全部通过 ffmpeg、文件大小校验，并命名为
`episodeN_success.mp4`。三列同步对比页为 `docs/MOVE_PLAYINGCARD_AWAY_VIDEO_COMPARISON.html`。实验服务器使用
node4 GPU0--2，单路 server+sim 峰值远低于 80 GiB、无 OOM；完成后全部停止，三张卡恢复为 1 MiB。

本结果与历史 27%--30% 的差异提示当前 checkpoint/seed 区间或部署版本更容易，不能继续把连续小 seed block 当作
“中等难度”保证。下一步不追加该任务 episode，资源转向跨任务 predictor 的 cost-sensitive/usage-balanced 训练或
预注册的 held-out confidence routing；闭环验证仍保留 pure/action-only/soft 三路和固定 seed 视频协议。

### 21.65 固定 seed 视频协议下的 `move_playingcard_away` 闭环对照（2026-09-28）

为避开 `pick_dual_bottles` 的 floor 与 `open_microwave` 的已知 ceiling，选择历史同类 100k policy 成功率约
`0.27--0.30`、step limit 400 的 `move_playingcard_away`，使用 D52 checkpoint 和已审计部署链路比较 pure Pi、
predicted action-only、soft dual T=1。两组 seed 在运行前分别预注册为 `[100000,100001,100002]` 和
`[200000,200001,200002]`；所有 policy 使用完全相同的 scene seeds，expert precheck 不允许替换。

结果出现新的 ceiling：两组共 6 个 episode 中，pure/action-only/soft 均为 **`6/6`**，没有任何 discordant
outcome。因此这组结果只能说明两类 token condition 没有破坏这些场景，不能支持 action token 或 physics token
提升 success rate。逐 seed 终止 control steps 为：pure `90/105/93/95/88/95`，action-only
`91/95/157/93/92/110`，soft dual `91/51/92/89/89/93`。soft 在 seed 100001 明显更早成功，但 step count
受轨迹与 success detector 共同影响，且 n=6，不作为性能结论。

六份 `_fixed_seeds.json` 均记录 3/3 success 和对应视频；18 个 MP4 全部通过 ffmpeg、文件大小校验，并命名为
`episodeN_success.mp4`。三列同步对比页为 `docs/MOVE_PLAYINGCARD_AWAY_VIDEO_COMPARISON.html`。实验服务器使用
node4 GPU0--2，单路 server+sim 峰值远低于 80 GiB、无 OOM；完成后全部停止，三张卡恢复为 1 MiB。

本结果与历史 27%--30% 的差异提示当前 checkpoint/seed 区间或部署版本更容易，不能继续把连续小 seed block 当作
“中等难度”保证。下一步不追加该任务 episode，资源转向跨任务 predictor 的 cost-sensitive/usage-balanced 训练或
预注册的 held-out confidence routing；闭环验证仍保留 pure/action-only/soft 三路和固定 seed 视频协议。

### 21.66 D53 soft 三头联合训练预注册（2026-09-28）

闭环 smoke 表明当前首要问题不是接口或普遍性崩坏，而是 soft physics 相对 action-only 的增益不稳定。下一实验不再
增加相邻 seed，而测试 D50 尚未覆盖的可微 soft 三头路径：action-token CE、physics-token CE 与 continuous flow
loss 同一 forward；`softmax(logits/T=1) @ codebook` 作为连续头条件，使 flow gradient 能通过 soft embedding
回传至两路 predictor。成熟 100k Pi/VLM 主干仍冻结，避免把结果混入主干遗忘。

D53 从 D52 500-step checkpoint 同时精确初始化 13 个 predictor tensors 与 66 个 conditioner/gate tensors，保持
action `4x128`、raw physics `256x256`、physics policy memory8 和 D49 两套独立 attention 不变。首轮只训练 20
steps，global batch 7。为控制 D50 的 gate drift，conditioner LR 设 `1e-5`、两套 gate LR `1e-4`，均低于旧 D50；
action probe/action embedding/physics probe 均为 `1e-5`。CE loss 权重保持 `1e-3/1e-4`，唯一新增机制是 flow 对
soft token logits 的可微反馈。

20-step gate：训练必须无 OOM/NaN，三项 loss 有限，checkpoint 只含预期 79 个 extension tensors；验证 soft 条件
确实给 action/physics logits 非零梯度，并核对 D52 初始化。通过后扩到 100 steps，再在原八任务固定 panel 比较
D52 soft、D53 soft、pure Pi 和 action-only；若 macro 或显著负向任务未改善，不继续扩步。launcher 为
`scripts/physical_tokens/run_d53_soft_joint.sh`。

D53 20-step smoke 已在 node4 GPU0--6 完成，global batch 7，用时约 3 分钟（训练 loop 约 22 秒，其余为七路
模型/数据初始化与 checkpoint 保存），每卡峰值约 `19.8--20.1 GiB`，无 OOM/NaN。20-step 日志中的 flow、action
CE、physics CE 均有限；checkpoint SHA256 为
`4367acc086470787c0e26ab3db31de18f37080537349799c2759bf406c2dff12`，严格保持 D52 的 79-key extension 集。
相对 D52 有 51/79 tensors bitwise 改变；action/physics gate 分别 13/28、20/28 改变。未变化项均为 BF16，最大
gate delta 为 `9.765625e-4`，说明部分 `1e-4` LR 更新低于当前权值量级的 BF16 可表示步进，而非 optimizer 漏参。
在决定调高 LR 或扩到 100 steps 前，先用 `evaluate_d53_soft_joint_panel.sh` 跑原八任务固定 64-sample panel；若
20-step 方向已负，则不靠增加 steps 掩盖该问题。

D53-20 的八任务固定 panel 已完成。下表均为相同 sample/seed/token cache 下，D52 soft-T1 到 D53-20 soft-T1 的
episode-cluster paired comparison；`improvement` 为 action MAE 的相对下降，CI 使用 10,000 次、按 episode 聚类的
bootstrap：

| task | D52 action MAE | D53-20 action MAE | improvement | episode wins | 95% paired CI |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | 0.012036952209 | 0.012034654492 | +0.01908886% | 4/5 | [+0.00664138%, +0.02901230%] |
| click_alarmclock | 0.011055016163 | 0.011054123807 | +0.00807195% | 3/5 | [-0.01306611%, +0.02164496%] |
| handover_block | 0.010980870613 | 0.010979790303 | +0.00983811% | 5/5 | [+0.00671030%, +0.01595853%] |
| move_playingcard_away | 0.014700087780 | 0.014700008206 | +0.00054131% | 2/5 | [-0.00896632%, +0.00921126%] |
| open_microwave | 0.007870718285 | 0.007867232278 | +0.04429084% | 5/5 | [+0.02722633%, +0.07703256%] |
| pick_dual_bottles | 0.022269338503 | 0.022268011409 | +0.00595929% | 5/5 | [+0.00434002%, +0.00806877%] |
| place_object_basket | 0.013857390762 | 0.013855923574 | +0.01058777% | 5/5 | [+0.00689345%, +0.01339374%] |
| shake_bottle | 0.015631809301 | 0.015629476609 | +0.01492272% | 4/5 | [-0.00262861%, +0.02504319%] |

宏平均相对改善为 **`+0.01416261%`**：8/8 tasks 数值正向，5/8 的 paired CI 严格为正，0/8 显著负向。
这说明 flow loss 经过 soft token distribution 联合更新 predictor/conditioner 的方向目前一致，但效应只有万分之一量级，
尚不能称为实际有意义的策略改善。它通过了预注册的“20 steps 方向不负才扩训”门槛，因此下一步保持初始化、LR、
batch、数据和 `T=1` 全部不变，从 D52 独立训练 D53-100；不从 D53-20 resume，以免 20-step cosine schedule 与
100-step schedule 混淆。D53-100 完成后在同一八任务 panel 同时比较 D52、D53-20、D53-100。只有改善幅度随步数
稳定扩大且不引入显著负向任务，才进入闭环；若仍停留在约 `0.014%` 的尺度或回落，则停止单纯增加训练步数，转向
BF16 更新分辨率/优化器状态或 cost-aware physics routing 的独立消融。

### 21.82 昨夜四任务 seed discovery 结果（2026-10-09）

昨夜完成了四组用于后续固定 seed 配对的 seed-discovery/基线评测。结果文件均已生成，视频按成功/失败保存，
没有发现 server、协议或显存错误：

| task | checkpoint/条件 | episodes | success |
|---|---|---:|---:|
| `click_alarmclock` | P64 fixed pure Pi | 3（seed `100000,100002,100003`） | **3/3 = 100%** |
| `hanging_mug` | 原始 pure Pi seed discovery | 10（`200000`--`200009`） | **8/10 = 80%** |
| `open_microwave` | 原始 pure Pi seed discovery | 10（`200000`--`200009`） | **10/10 = 100%** |
| `place_object_stand` | 原始 pure Pi seed discovery | 10（`200000`--`200009`） | **10/10 = 100%** |

其中 `click_alarmclock` 是此前 P64 fixed-pure 协议的 3-seed 复核；另外三项是新发现的标准 seed
序列，不是 action-only/joint 的 matched comparison。合计 33 个 episode、31 次成功（宏观合并
93.9%），但不能把这个合并数字当作跨任务 benchmark 成绩：三个任务中有两个达到 ceiling，
`hanging_mug` 也只剩 2 个失败 seed，适合做稳定性/回归检查，不适合作为主要增益任务。

下一步：优先从 `hanging_mug` 的失败 seed 和现有 expert-stable failure panel 中扩充同 seed 的
pure/action-only/joint 三路重复；`open_microwave` 与 `place_object_stand` 暂作为 ceiling sanity check，
只有在 token 条件出现失败而 pure 成功时才纳入回归分析。所有 rollout 继续以 `_fixed_seeds.json`、
MP4 解码和同一 scene seed manifest 为纳入条件。

### 21.83 改用 `eval_seeds.json` 的四任务对照（2026-10-09）

此前四任务中的 `click_alarmclock`、`hanging_mug`、`open_microwave`、`place_object_stand` 并不都在
`examples/Robotwin/eval_files/eval_seeds.json` 中，因此不能跨任务套用 seed。为避免把非 expert-stable
场景混入统计，改用该文件中已经验证为 `success_seeds` 的四个 joint 训练任务：
`pick_dual_bottles`、`handover_block`、`shake_bottle`、`beat_block_hammer`。每个任务预注册前 10 个
success seed，pure Pi、P64 action-only、P64 joint hard 三路共享完全相同的 seed list。

本轮 seed list 为：

| task | fixed seeds（前 10 个） | 是否在 P64 joint 训练集 |
|---|---|---|
| `pick_dual_bottles` | `100100001–100100010` | 是 |
| `handover_block` | `100100000, 005, 006, 009, 015, 023, 027, 028, 036, 039` | 是 |
| `shake_bottle` | `100100002, 003, 004, 006, 010, 011, 012, 015, 017, 022` | 是 |
| `beat_block_hammer` | `100150000, 001, 004, 005, 006, 008, 009, 010, 011, 013` | 是 |

截至本节更新时，12 路 rollout 仍在 node3 八卡并行队列中，尚未生成这四个任务的完整三路
`_fixed_seeds.json`，因此不提前报告成功率。已有的完整闭环结果仍为：`click_alarmclock` pure/action-only/joint
为 `3/3 / 0/3 / 1/3`；`place_object_stand` 为 `10/10 / 9/10 / 8/10`，但后者不属于上述
`eval_seeds.json` 任务集，不能与本轮四任务宏平均混合。

本轮实验回答两个问题： (1) 在已经见过的任务上，physics token 是否改变闭环成功率；(2) 成功率高低
是否只是任务难度/seed 选择造成。最终表必须同时给出 pure、action-only、joint 三路成功率、逐 seed
paired win/loss、有效 episode 数和 seed 文件来源，并标注这些任务都使用过 P64 physics tokenizer、
64 raw physics IDs、32 policy memory tokens。

> **更正：** 21.82 中 `hanging_mug/open_microwave/place_object_stand` 的 10-episode 数字来自普通
> seed-discovery 入口，不经过 fixed evaluator 的 expert-stable precheck；它们只能作为历史 discovery
> 记录，不能与本节的 `eval_seeds.json` fixed-seed 结果混合，也不能据此声称是严格 10-seed benchmark。
此前四任务中的 `click_alarmclock`、`hanging_mug`、`open_microwave`、`place_object_stand` 并不都在
`examples/Robotwin/eval_files/eval_seeds.json` 中，因此不能跨任务套用 seed。为同时检验训练内效果和
跨任务泛化，改用该文件中已经验证为 `success_seeds` 的两项 joint 训练任务和两项未训练任务：
`handover_block`、`beat_block_hammer`（训练内），`click_bell`、`place_phone_stand`（训练外）。每个任务
预注册前 10 个 success seed，pure Pi、P64 action-only、P64 joint hard 三路共享完全相同的 seed list。
| `click_bell` | `100100001–100100010` | 否 |
| `place_phone_stand` | `100100002, 003, 005, 006, 007, 009, 010, 013, 015, 017` | 否 |
本轮实验回答三个问题： (1) 在已经见过的任务上，physics token 是否改变闭环成功率；(2) 在未见任务上，
physics token 是否提供跨任务迁移；(3) 成功率高低是否只是任务难度/seed 选择造成。最终表必须同时给出 pure、action-only、joint 三路成功率、逐 seed

另外，HTML 简报补充了此前 8 个 pure Pi 困难 episode 的三次重复结果：
`click_alarmclock/{100005,100017,100018,100022}`、`click_bell/100008`、
`hanging_mug/{100002,100003}`、`press_stapler/100053`。pure repeat1 总体为 2/8；其中只有
`click_alarmclock` 属于 P64 joint 训练任务，其余均为未训练任务，不能与本轮四任务平均混合。

八个困难 episode 的三路对照（成功次数/有效重复次数）为：

| task/seed | pure | action-only | joint |
|---|---:|---:|---:|
| click_alarmclock/100005 | 2/3 | 0/3 | 0/3 |
| click_alarmclock/100017 | 0/3 | 2/3 | 3/3 |
| click_alarmclock/100018 | 1/3 | 0/3 | 3/3 |
| click_alarmclock/100022 | 0/3 | 2/3 | 3/3 |
| click_bell/100008 | 0/3 | 0/3 | 0/3 |
| hanging_mug/100002 | 0/3 | 0/3 | 0/2 |
| hanging_mug/100003 | 0/3 | 1/3 | 0/1 |
| press_stapler/100053 | 0/3 | 1/3 | 2/3 |

其中 hanging_mug 的 joint 有效重复数不足 3，不能按 3 次重复解释。训练内的 click_alarmclock 困难
seed 上 joint 对 100017/100018/100022 有恢复，但 100005 没有；训练外任务没有显示稳定 joint 增益。

### 21.88 Action token 数量与 Pi/FAST 的可比性

本轮讨论明确了一个容易混淆的问题：Pi 系列中的 action chunk 长度、FAST 的离散 action token 数量，以及
本项目 action condition 的 token 数量不是同一个概念。

Pi0/Pi0.5 的主动作路径通常是连续 action expert（flow matching 或 diffusion-style action prediction）。例如
一个 32-step action chunk 中的 32 是时间步长度，不等于 32 个离散 token。本项目的 `ema_4x128` 则把一个
action chunk 编码为 4 个离散 action latent，作为连续 action expert 的辅助条件，而不是替代 Pi 的连续动作
预测路径。

FAST 更接近“连续 action chunk → 量化/序列化 → 离散 token 序列 → FAST decoder → 连续 action chunk”。
FAST 的 token 数量由 tokenizer 的压缩率、量化方式和序列合并决定，可能是固定长度或近似变长。FAST token
通常承担主要的动作生成任务，不能和本项目的 4 个辅助 action condition token 直接按数量比较。

因此，直接把 action token 从 4 改成 32 并不能构成合理消融。现有 action predictor、tokenizer 和 cache 都
固定为 4 个 token；强行改成 32 会产生 target shape 和 predictor 维度不匹配，复制或 padding token 也不代表
真正增加了 action 信息容量。若要研究 `action=32, physics=32`，必须先训练或准备真正输出 32 个 latent 的
action tokenizer，重新生成 action token cache，调整 predictor 输出头，再进行匹配的 20k 训练。

当前先保持 action=4，只改变 physics residual 数量：P64/M32 是原设置，新增 P64/M8 和 P64/M4。这样可以
先回答 physics 的收益是否依赖 memory token 数量；只有该实验显示明确且稳定的数量效应后，才值得投入 action-32
tokenizer。比较时应同时记录 action reconstruction error、endpoint error、token 序列长度、codebook 使用率、
action/physics residual norm 和 paired rollout success，而不能只比较 token 个数。

### 21.89 Gate 数值诊断与边界方案

对当前 20k joint checkpoint 的 gate 检查发现，28 层的
`pi_action_layer_gates` 范围为 `[-9.94, 4.81]`，平均值 `-2.70`、平均绝对值 `4.06`；
`pi_physics_layer_gates` 范围为 `[-9.94, 15.50]`，平均值 `2.59`、平均绝对值 `4.36`。
action-only 的 action gate 也已经达到约 `4.20` 的平均绝对值，而不是停留在零附近。

当前 gate 是直接乘到 residual cross-attention 输出上的无界标量，因此训练后可能把 residual 放大数倍，
甚至通过负号反向修改原始 Pure PI hidden state。这与 joint 在 Pure PI 已成功 seed 上退化的现象一致，
说明 residual 注入强度是当前优先检查项。

下一版加入可配置的 gate 边界，并比较两种形式：

```text
bounded_tanh:   alpha = alpha_max * tanh(raw_gate)
bounded_sigmoid: alpha = alpha_max * sigmoid(raw_gate)
```

建议先使用 `alpha_max=0.1` 或 `0.2`。`tanh` 保留正负残差方向，适合当前 gate 已经出现负值且需要修正
Pure PI 的情况；`sigmoid` 只能产生正向注入，稳定性更强，但不能表达“从 base action 中减去 residual”
的修正。第一轮优先比较 `tanh`，再用相同初始化和数据跑 `sigmoid`，同时记录 gate、residual norm、
Pure PI 成功 seed 的保持率和困难 seed 的救回率。

边界实验必须与当前 P64/M32 joint 使用相同 backbone、数据、冻结边界和 20k steps；不把 gate 边界效果与
physics token 数量效果混在一起。停止标准是：Pure PI 已成功的共同 seed 不再明显下降，同时困难 seed 仍有
净收益；若 `sigmoid` 稳定但救回能力下降，则保留 bounded tanh 并加入 observation-dependent confidence gate。
### 21.89 Gate 数值诊断与边界方案

当前 20k joint checkpoint 的 28 层 gate 检查结果：`pi_action_layer_gates` 范围为 `[-9.94, 4.81]`，平均绝对值 `4.06`；`pi_physics_layer_gates` 范围为 `[-9.94, 15.50]`，平均绝对值 `4.36`。action-only 的 action gate 平均绝对值也约为 `4.20`。gate 已经远离零，并且是直接乘到 residual cross-attention 输出上的无界标量，因此可能把 residual 放大数倍，甚至通过负号反向修改 Pure PI hidden state。这与 joint 在 Pure PI 已成功 seed 上退化的现象一致。

下一版应加入可配置的 gate 边界并比较：

```text
bounded_tanh:    alpha = alpha_max * tanh(raw_gate)
bounded_sigmoid: alpha = alpha_max * sigmoid(raw_gate)
```

先取 `alpha_max=0.1` 或 `0.2`。tanh 保留正负残差方向，能够修正 base action；sigmoid 只能正向注入，通常更稳定但表达能力更弱。两种方式必须使用相同 backbone、数据、冻结边界和 20k steps，同时记录 gate、residual norm、Pure PI 成功 seed 保持率和困难 seed 救回率。第一轮优先 bounded tanh，再比较 sigmoid。

轻量 checkpoint 映射诊断已经完成：在不重新训练的情况下把现有 joint gate 映射到 `alpha_max=0.2`，click_bell 同一 10-seed 结果为原始 joint `6/10`、bounded tanh `6/10`、bounded sigmoid `5/10`。这支持保留有符号修正的 tanh 作为正式重训方案；该结果仍属于推理时诊断，不能替代带边界的 20k 重训。

### 21.90 P64/M8 与 gate sweep 补测（2026-10-11）

在同一组 `click_bell` 十个固定 seed 上，node2 直接仿真完成了以下对比：

| checkpoint | success |
|---|---:|
| action-only | 5/10 |
| physics-only | 5/10 |
| no-token v4 | 6/10 |
| joint P64/M32 | 6/10 |
| joint P64/M8 | 5/10 |
| derived bounded tanh α=.05 | 7/10 |
| derived bounded tanh α=.10 | 6/10 |
| derived bounded tanh α=.20 | 7/10 |
| derived bounded tanh α=.50 | 7/10 |
| derived bounded sigmoid α=.20 | 6/10 |

P64/M8 没有在该 panel 上超过 P64/M32 或 action-only；因此当前没有证据支持仅减少 physics memory token 数量就能改善 rollout。bounded tanh 的 `.05/.20/.50` 显示了同样的 7/10，但这些是由已有 checkpoint 派生的推理时 gate 映射，不是重新训练结果，暂时只能视为方向性信号。no-token 与 joint 都为 6/10，说明这组十 seed 上 joint 的收益尚未能和 v4 架构本身、随机 seed 方差区分开。后续应优先做共同 seed 的 paired wins/losses 和重复 panel，不继续扩大 token 容量。

### 21.91 Pure PI 困难 episode 四路复测（2026-10-11）

针对 Pure PI 之前表现困难的 8 个固定 episode，在 node2 八卡上使用同一组 scene seed，依次评测 Pure PI、
action-only、joint P64/M32 和 joint P64/M8。四路均完整结束，无 OOM 或中途退出。结果如下（每格为成功/总数）：

| task / seed | Pure PI | action-only | joint P64/M32 | joint P64/M8 |
|---|---:|---:|---:|---:|
| click_alarmclock / 100005 | 1/1 | 0/1 | 0/1 | 0/1 |
| click_alarmclock / 100017 | 0/1 | 0/1 | 1/1 | 1/1 |
| click_alarmclock / 100018 | 1/1 | 0/1 | 1/1 | 1/1 |
| click_alarmclock / 100022 | 0/1 | 1/1 | 1/1 | 1/1 |
| click_bell / 100008 | 0/1 | 0/1 | 0/1 | 1/1 |
| hanging_mug / 100002 | 0/1 | 1/1 | 0/1 | 0/1 |
| hanging_mug / 100003 | 0/1 | 0/1 | 0/1 | 0/1 |
| press_stapler / 100053 | 0/1 | 1/1 | 0/1 | 0/1 |
| **总计** | **2/8** | **3/8** | **3/8** | **4/8** |

相对 Pure PI，joint P64/M32 救回了 click_alarmclock 的 100017 和 100022，但丢失了 Pure PI 已成功的
100005；P64/M8 额外救回 click_bell 100008，同时保留了 100017/100018/100022，但仍未解决两个
hanging_mug 和 press_stapler。由于样本只有 8 个，不能据此宣称 M8 普遍优于 M32；它更像是困难集上的
方向性信号。完整逐 seed 日志、服务日志和视频位于：
`logs/hard_episode_panel_node2_20261011_010239/` 及对应的 `logs/matched_*hard_{pure,actiononly,joint_m32,m8}_*/`。

### 21.91 Pure PI 困难 episode 四路复测（2026-10-11）

针对 Pure PI 之前表现困难的 8 个固定 episode，在 node2 八卡上使用同一组 scene seed，依次评测 Pure PI、action-only、joint P64/M32 和 joint P64/M8。四路均完整结束，无 OOM 或中途退出。结果如下（每格为成功/总数）：

| task / seed | Pure PI | action-only | joint P64/M32 | joint P64/M8 |
|---|---:|---:|---:|---:|
| click_alarmclock / 100005 | 1/1 | 0/1 | 0/1 | 0/1 |
| click_alarmclock / 100017 | 0/1 | 0/1 | 1/1 | 1/1 |
| click_alarmclock / 100018 | 1/1 | 0/1 | 1/1 | 1/1 |
| click_alarmclock / 100022 | 0/1 | 1/1 | 1/1 | 1/1 |
| click_bell / 100008 | 0/1 | 0/1 | 0/1 | 1/1 |
| hanging_mug / 100002 | 0/1 | 1/1 | 0/1 | 0/1 |
| hanging_mug / 100003 | 0/1 | 0/1 | 0/1 | 0/1 |
| press_stapler / 100053 | 0/1 | 1/1 | 0/1 | 0/1 |
| **总计** | **2/8** | **3/8** | **3/8** | **4/8** |

相对 Pure PI，joint P64/M32 救回了 click_alarmclock 的 100017 和 100022，但丢失了 Pure PI 已成功的 100005；P64/M8 额外救回 click_bell 100008，同时保留了 100017/100018/100022，但仍未解决两个 hanging_mug 和 press_stapler。由于样本只有 8 个，不能据此宣称 M8 普遍优于 M32；它更像是困难集上的方向性信号。完整逐 seed 日志、服务日志和视频位于：`logs/hard_episode_panel_node2_20261011_010239/` 及对应的 `logs/matched_*hard_{pure,actiononly,joint_m32,m8}_*/`。

### 21.92 Bounded-tanh αmax=0.2 20k 重训闭环结果（2026-10-11）

node1 上 `qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000` 已完成 20k 训练并保存 checkpoint。随后直接在 node1 八卡评测新 checkpoint，没有重复运行任何 baseline。困难集 8/8 有效，结果为 **5/8**：`click_alarmclock` 3/4、`click_bell/100008` 成功、`hanging_mug` 0/2、`press_stapler/100053` 成功。与同 seed 已完成结果相比，Pure PI 为 2/8、P64/M32 为 3/8、P64/M8 为 4/8；有界 tanh 在这组困难 seed 上出现方向性提升，但仍有 `click_alarmclock/100005` 回归，不能据此判定普遍优于 Pure PI。

同一新 checkpoint 还评测了四任务 panel。由于 fixed evaluator 的 expert precheck 拒绝了 11 个预注册 seed（不是模型失败，也没有执行策略 rollout），有效结果为：

| task | bounded-tanh αmax=0.2 | Pure PI 历史结果 |
|---|---:|---:|
| `handover_block`（训练内） | 6/8 | 8/8 |
| `beat_block_hammer`（训练内） | 6/7 | 7/8 |
| `click_bell`（训练外） | 8/10 | 9/10 |
| `place_phone_stand`（训练外） | 4/5 | 7/7 |

四任务结果暂未缓解 joint 相对 Pure PI 的差距；但这些分母按新模型的 expert-precheck 有效 seed 计算，Pure PI 历史分母并不完全相同，表中不能直接作为严格 paired 差值。下一步应先限定三路共同有效 seed 做 paired wins/losses，而不是直接根据宏平均调 gate。11 个被 precheck 拒绝的 seed 为 `handover_block/{100100000,100100039}`、`beat_block_hammer/{100150001,100150004,100150005}`、`click_bell/100100009`、`place_phone_stand/{100100002,100100005,100100009,100100013,100100015}`。日志和视频位于 `logs/posttrain_eval_qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000_20261011_025605/` 及相应 `logs/matched_*posttrain_qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000_*/`。

### 21.93 四路共同有效 seed 的严格配对重算（2026-10-11）

上一节的表使用了各 checkpoint 自己的有效 episode 数，不能直接比较。现根据历史 Pure/action-only/joint 的
`matrix_* / _fixed_seeds.json` 与新 tanh checkpoint 的逐 seed结果，取四路都实际执行且有记录的交集：

| task | 共同 seed | Pure | action-only | joint P64/M32 | bounded tanh |
|---|---|---:|---:|---:|---:|
| `handover_block` | `009,015,023,027,028,036` | 6/6 | 6/6 | 5/6 | 5/6 |
| `beat_block_hammer` | `000,008,009,010,011,013` | 5/6 | 4/6 | 3/6 | 5/6 |
| `click_bell` | `001,002,003,004,005,006,007,008,010` | 8/9 | 1/9 | 7/9 | 7/9 |
| `place_phone_stand` | `003,006,017` | 3/3 | 3/3 | 3/3 | 3/3 |

在相同 seed 上，bounded tanh 相对 Pure 的 paired `win/loss/tie` 为：`handover_block 0/1/5`、
`beat_block_hammer 1/1/4`、`click_bell 1/2/6`、`place_phone_stand 0/0/3`；相对原始 joint P64/M32
为：`handover_block 1/1/4`、`beat_block_hammer 2/0/4`、`click_bell 1/2/6`、
`place_phone_stand 0/0/3`。因此严格配对后，tanh 在训练内 `beat_block_hammer` 相对原始 joint 有
改善信号，但相对成熟 Pure PI 仍未稳定占优；`click_bell` 也没有超过 Pure。该交集表才是当前可用于
判断“新 gate 是否缓解 joint 退化”的有效比较，不能使用上一节不一致的分母。

### 21.94 新 tanh 的 demo_clean 完整四任务十 seed 补测（2026-10-12）

上一轮新 tanh 四任务 rollout 使用了 `demo_randomized`，而历史 baseline 使用 `demo_clean`，因此上一节结果
不能作为严格同配置结论。已在 node4 八卡上用与 baseline 完全一致的 `demo_clean` 补跑全部 40 个预注册 seed。
其中 expert-precheck 拒绝的 seed 与历史 evaluator 的不稳定 scene 相同，不计作模型失败。

在四路（Pure/action-only/joint/new tanh）共同实际执行的 seed 上，结果为：

| task | 共同有效 seed 数 | Pure | action-only | joint P64/M32 | new tanh |
|---|---:|---:|---:|---:|---:|
| `handover_block` | 8 | 8/8 | 8/8 | 7/8 | 7/8 |
| `beat_block_hammer` | 7 | 7/7 | 5/7 | 4/7 | 5/7 |
| `click_bell` | 10 | 9/10 | 1/10 | 8/10 | 9/10 |
| `place_phone_stand` | 7 | 7/7 | 6/7 | 7/7 | 6/7 |

严格 paired 结果（new tanh 相对 Pure / joint，格式为 win/loss/tie）：

| task | vs Pure | vs joint P64/M32 |
|---|---:|---:|
| `handover_block` | 0/0/8 | 0/0/8 |
| `beat_block_hammer` | 0/2/5 | 2/1/4 |
| `click_bell` | 1/0/9 | 2/0/8 |
| `place_phone_stand` | 0/1/6 | 0/1/6 |

这次同配置结果比之前更可靠：bounded tanh 在 `click_bell` 上达到 Pure 的 9/10，并相对原始 joint
额外赢 2 个 seed；在 `beat_block_hammer` 上相对 joint 赢 2/1，但仍低于 Pure；`handover_block` 与
joint 完全持平；`place_phone_stand` 反而损失 1 个 Pure/joint 已成功 seed。因此 tanh 能缓解部分 joint
退化，但还不是跨任务稳定增益。完整 node4 队列和视频位于：
`logs/tanh02_four_tasks_demo_clean_node4_20261011_225911/` 及对应的 `logs/matched_*tanh02_demo_clean_*/`。

### 21.95 Gate 学习率 `1e-2` 的历史依据与当前判断（2026-10-12）

当前 `qwenpi_v4_p64_m32_distill_medium_seed42_steps20000` 使用
`pi_action_layer_gates` 和 `pi_physics_layer_gates` 的学习率 `1e-2`。这不是误填，而是沿用 D47/D48/D50
的单变量实验结论：D46 使用 gate LR `1e-4` 时，100 steps 后 gate 平均绝对值仅约 `1.9e-3`，residual
通道基本没有打开，未观察到 token correspondence；D47 将 gate LR 提高到 `1e-2` 后，20 steps 的
action-only/dual gate 平均绝对值达到约 `0.0419/0.0462`，确认 residual 路径实际开始工作。D47 的
action+physics 相对独立 action-only 有 `0.1479%` 的改善信号，因此后续分离 residual 和 joint token
训练沿用了 `1e-2`。

该值仍属于偏激进的 gate 专用学习率，不能简单视为普适最优。gate 是每个 expert layer 一个 scalar，且从
零初始化；较大的 LR 用来克服零 gate 早期的梯度瓶颈，但也可能使 residual 过快放大、破坏成熟 Pure PI。
当前 tanh 版本通过

```text
alpha = 0.2 * tanh(raw_gate)
```

把实际注入强度限制在 `[-0.2, 0.2]`，因此比 D46/D47 的无界 gate 安全，但 raw gate 仍可能过快进入 tanh
饱和区。distill-medium 的 Pure teacher loss 能抑制控制退化，但不能证明 `1e-2` 是最优值。

当前 node3 的 20k distill run 不在中途修改 gate LR，以保持实验可比性。后续应在完成 checkpoint 后记录
raw gate、bounded alpha、residual norm 及 Pure-success 保持率，并以独立消融比较 `1e-3/3e-3/1e-2`；
若 `1e-2` 出现 gate 饱和或 Pure PI 回归，优先采用较小 gate LR，而不是继续增加训练步数。
