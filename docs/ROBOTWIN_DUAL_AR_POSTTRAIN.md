# RoboTwin：8 动作 + 64 物理 token 自回归后训练

> 当前正式配置（2026-10-27 更新）采用**冻结视觉编码器、严格分模态输入表与各自输出头绑权、语言层 LoRA rank=32/alpha=32**，全 50 任务长训起点为已有的 RoboTwin-All 微调权重 `models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`。旧共享头、仅独立输出头和 rank 16 烟测均保留为历史对照，不覆盖其 checkpoint；新长训状态见文末。

## 设计与状态

目标是将之前的“一次前向并行输出所有 ID”的双 token probe 改为 Qwen3-VL 因果语言模型的逐 token 预测。新框架 `QwenDualAutoregressiveReasoner` 沿用现有 8-token 动作 Reasoner 的训练接口，输入当前图像和任务语言，训练目标固定为 8 个 `<robot_action_i>`（EMA-VQ K525）后接 64 个新建的 `<robot_physics_i>`（空间对齐 EMA-VQ K2048）。两个命名空间独立；推理时逐位置强制动作域、物理域，贪心生成固定 72 个 ID，不访问未来帧或目标缓存。训练只对这 72 个符号计算 teacher-forced causal LM CE，提示词、padding 和 assistant 结束符都不计入 loss。因两流长度为 8:64，默认目标取两流各自的平均 CE 相加（各权重 1），避免动作在逐 token 等权平均中只占 1/9；日志同时记录两流 CE。

这模仿 PhiZero 的“先用冻结 tokenizer 导出真值，再让 VLM 自回归预测离散 ID”的流程，**并没有**把本项目的 64/K2048 物理 tokenizer 变成 PhiZero 的 256/K25000 视频 tokenizer，也没有引入 Wan 视频重建 loss。当前物理 ID 来自 17 帧 Qwen 视觉特征差分，动作 ID 对应 wrap32 动作块；两者时间窗口和目标不同，后训练监督必须按同一个 `(task_id, episode_id, base_index)` 键对齐。

全 50 任务数据使用固定 episode 留出：训练排除 `[5,20,25,39,49]`，最终验证应仅用这些 episode。物理缓存只在 421,990 个训练及 47,797 个留出完整 clip 上有公平的 tokenizer 评估，因此双流 SFT 的目标缓存也必须要求 `future_valid_mask.all(axis=1)`；其余动作缓存行不作为双流监督。`TokenConditionedDataset` 新增显式 `drop_missing` 选项，并让混合数据集看到相同的过滤后 `all_steps`，防止末端不完整 clip 被反复抽中。

已实现：模型模块 `starVLA/model/framework/QwenDualAutoregressiveReasoner.py`，训练配置 `examples/Robotwin/train_files/qwen_dual_ar_8action_64physics_full50.yaml`，可恢复入口 `scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh`。**统一使用 `/mnt/jfzn/liujn/miniconda3/envs/vlact`** 运行导出、测试和后训练。动作 ID 已在该环境从选定的 `ema_8x525_decay095_dead100_steps20000` 导出到 `artifacts/robotwin_full50_study/action_8x525_ids_vlact_for_dual_ar.npz`，覆盖 549,787 个样本；物理 ID 也已从 `ema_t64_k2048_steps20000` 导出并严格合并。

自回归训练/推理专页：`docs/ROBOTWIN_AR_GENERATION_TRAINING.html`，逐步展示输入、teacher forcing、标签移位、两流 CE、反向更新、checkpoint 复原、受限逐 token 推理及评估口径；node001 浏览器地址为 <http://10.120.154.139:8767/docs/ROBOTWIN_AR_GENERATION_TRAINING.html>。此前的概览页 `docs/ROBOTWIN_DUAL_AR_POSTTRAIN_STATUS.html` 仍保留在 8766；原有 8765 报告服务也保持不动。HTML 是固定时间快照，不会自动刷新训练进度。

运行状态（2026-10-27 14:21）：物理 ID 导出和双流缓存合并已完成，缓存含 50 任务、469,787 个完整 clip，其中训练 episode 对应 421,990 个、留出 episode 对应 47,797 个。node001 GPU 0–3 上使用 `vlact`/DeepSpeed ZeRO-2 的正式 500-step 后训练已完成，最终可训练参数 checkpoint 为 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500/final_model/pytorch_model.pt`。严格 warm-start 加载 714 个 Qwen 张量，2048 个新物理符号保留新初始化。第 10 步总 CE 26.704（动作 13.019、物理 13.685），第 500 步总 CE 11.740（动作 4.275、物理 7.465）；这些是单个训练 batch loss，不能代表留出效果。数据集初始化日志的 `steps kept 493990/549787` 是 episode 过滤后的底层步数，`TokenConditionedDataset(drop_missing=True)` 再从中筛出有完整双流目标的步，混合数据集抽样使用过滤后的 `len(dataset)`。此作业只管理自身进程，不停止或清理其他用户任务。

500-step 后的小留出诊断只覆盖 `adjust_bottle`、`click_alarmclock` 共 4 个样本：teacher-forced 动作 CE 5.750、物理 CE 7.662，两个流的 token top-1 accuracy 均为 0；另 2 个样本的 free-running top-1 accuracy 也均为 0。物理 CE 与 `ln(2048)≈7.625` 接近；因此此阶段只证明代码与训练流程跑通，**不能**宣称策略或物理预测有效，更不能推出全 50 任务的表现。详细原始结果在 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500/heldout_diagnostic.json`。

为进行较长后训练，已添加严格恢复可训练参数的 `trainer.dual_ar_trainable_checkpoint` 选项：先载入原 100k-step Qwen 基座，再恢复 500-step 的可训练参数，键必须完全匹配；不碰其他人的进程。新的 5,000-step 续训作业 `qwen_dual_ar_8a64p_full50_seed42_steps5000` 已在 `vlact`、node001 GPU 0–3 上进入优化，确认恢复了 24 个可训练张量；第 10 步动作 CE 4.316、物理 CE 7.492。按约 1.1 秒/步，训练本体预计约 1.5 小时，之后还需保存和留出诊断。尚无该作业的最终 checkpoint，不能把它当成已完成结果。

## 267 个预留词表行的统一初始化对照

旧 tokenizer 实际长度为 153,717，旧 Qwen embedding/LM head 有 153,984 行，差出的 267 行在增加物理词表后对应物理 ID 0–266。旧方案虽没有载入 100k-step checkpoint 的这些预留行，但扩词表时保留了基座模型的预留行；物理 ID 267–2047 则是真正新增的行。首轮 500-step checkpoint 中两组平均 L2 范数约 0.182 和 0.00894，存在明显起始分布差异，但不能仅凭范数证明它造成了留出准确率低。

新对照使用 `DUAL_AR_UNIFORM_PHYSICS_INIT=1` 从相同 100k-step 基座重新启动：保持全部旧有效 token 和物理 ID 267–2047 的向量原样，只用后者逐维均值/标准差形成的分布重新采样物理 ID 0–266；输入 embedding 与绑权输出头使用同一组值。这样只改变原 267 个预留位置的起点，避免把整个词表随机重置。新实验 run ID 为 `qwen_dual_ar_8a64p_full50_seed42_steps500_uniform_physics_init`，在 node001 GPU 4–7、独立通信端口 29673 完成 500 步，与 GPU 0–3 上已有的 5,000-step 续训并存，没有停止其他任务。运行日志确认 `uniform_physics_init=True`、`reinitialized_old_padding_rows=267`；9 项相关测试通过。第 500 步单 batch 动作 CE 4.235、物理 CE 7.480、总 CE 11.715，不能据此排名。

两组各训练 500 步后，在同样的 `adjust_bottle`/`click_alarmclock` 留出 32 样本 panel 上重评，原初始化 vs 统一初始化：动作 CE 5.638 vs 5.772，物理 CE 7.694 vs 7.670；动作 top-1 为 27/256 vs 25/256，物理 top-1 为 3/2048 vs 0/2048。两边在 4 个真正 free-running 样本的逐位/整段准确率都为 0。原始结果分别是 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500/heldout_panel32.json` 与 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500_uniform_physics_init/heldout_panel32.json`。统一初始化物理 CE 略低，但 top-1 没有改善；仅两个任务、32 样本，不能认定下游更好。当前 5,000-step 原初始化方案尚在运行，长训最终效果待评估。

启动入口：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DUAL_AR_NUM_PROCESSES=4 \
DUAL_AR_MAX_STEPS=500 \
bash scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh
```

默认先从本地 `Qwen3-VL-4B-Instruct-Action` 构造模型，再严格 warm-start `/mnt/jfzn/liujn/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt` 的 Qwen 参数：只有共享 embedding/lm_head 的新增词表行允许扩展，其余键与形状必须完全匹配；不加载旧的并行预测头。后训练冻结视觉前端和较早的语言层，优化最后两层、norm 和共享输入/输出 embedding。新增 2048 个物理符号后必须保存并复用相同词表构造顺序，不能把新增 ID 混成已有动作 ID。初始 500 步只做可运行性与短训诊断；需要报告留出 token CE/accuracy、逐流准确率、全序列准确率和最终动作解码误差，才可判定是否胜过并行 probe 或原 action-only 策略。这里的物理目标含未来视觉信息，评估时只能用于监督和指标，绝不能输入预测器。

短训后的部分留出诊断入口为 `scripts/physical_tokens/evaluate_dual_autoregressive_reasoner.py`；训练 runner 在最终 checkpoint 落盘后也会自动对 `adjust_bottle`、`click_alarmclock` 各采 2 个留出样本，报告 teacher-forced CE/逐流 token accuracy 与少量真正 free-running 的 72-token accuracy，写入 run 目录 `heldout_diagnostic.json`。这只是两个任务的小 panel，不能冒充全 50 任务结果或机器人 rollout 成功率。

## 独立预测头方案（新增；500 步短训已完成）

应用户要求，`QwenDualAutoregressiveReasoner` 支持 `separate_heads=true`，并已将 RoboTwin 正式训练配置与 runner 默认切到独立头。文本仍使用原 Qwen `lm_head`；动作使用独立的 `Linear(2560, 525, bias=False)`；物理使用独立的 `Linear(2560, 2048, bias=False)`。两套离散码仍占用 Qwen 统一的**输入** token ID/embedding，因而 8 个动作预测会作为后续 64 个物理码的自回归上下文；分开的只是**输出分类头**。原文本头因 Qwen 绑权/词表兼容，物理上仍有动作和物理 ID 对应的行，但在这 72 个输出位置不会调用；两个新头从 warm-start 后对应的原 `lm_head` 行复制初始化，随后不与原 embedding、原文本头或彼此绑权。训练通过 Qwen multimodal backbone 提取位置隐藏状态，分别在 525 和 2048 个候选码上计算 CE，再按既有权重相加；推理逐步调用相应头，预测 ID 回填进 Qwen 输入，并复用 KV cache。历史共享头代码和已有 checkpoint 不受影响。

默认启动独立头版本，runner 使用 `_separate_heads` run ID 后缀并让留出评估加载相同架构。要重跑共享头对照，显式设置 `DUAL_AR_SEPARATE_HEADS=0`，runner 会写入新的 `_shared_heads` run ID；手工评估历史共享头 checkpoint 时必须传 `--shared-heads`，避免按新配置误建独立头。**旧共享头的可训练参数 checkpoint 不能直接当作新架构 checkpoint 加载**：新头是新增参数；应从同一基座 Qwen warm-start 后重新后训练。两种方案的 CE 候选空间不同（共享头对完整 Qwen 词表归一化，独立头分别对 525/2048 个码归一化），原始 CE 数值不可直接作为优劣排名；应在相同留出集比较 token accuracy、free-running 准确率、动作解码误差和下游 rollout。

2026-10-27 16:11：在 node001 GPU 4–7 上完成 500 步独立头短训，使用 `DUAL_AR_SEPARATE_HEADS=1 DUAL_AR_UNIFORM_PHYSICS_INIT=1`；独立 run ID 为 `qwen_dual_ar_8a64p_full50_seed42_steps500_separate_heads_uniform_physics_init`，最终可训练参数文件为 `results/Checkpoints/<run_id>/final_model/pytorch_model.pt`。warm-start 日志确认 714 个 Qwen 张量严格载入、2048 个物理码扩表、267 个旧预留行重新初始化且两个独立输出头已启用。第 500 步单 batch 动作 CE 3.985、物理 CE 7.229；这不是验证集结果。新 checkpoint 严格复原、自由运行 72-token 推理均已通过。没有停止或覆盖其他作业。

同样两个任务（`adjust_bottle`、`click_alarmclock`）、同样 32 个留出样本、同样 500 步且同样统一物理初始化的对照：共享头动作 top-1 为 **25/256**，物理为 **0/2048**；独立头动作 **21/256**，物理 **1/2048**。独立头留出动作 CE 5.342、物理 CE 7.638；共享头分别为 5.772、7.670，但分母词表不同，**CE 不能直接比较**。独立头的 2 个 free-running 样本两流准确率均为 0；此面板太小，不能认为独立头更好，也不能据此确定最终架构。原始结果：`results/Checkpoints/<run_id>/heldout_panel32.json`；共享头对照：`results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500_uniform_physics_init/heldout_panel32.json`。后续若决定正式采用独立头，应做更长后训练、全 50 任务留出集和机器人 rollout；这里没有宣称完成这些实验。

## 严格分模态绑权（当前正式默认）

这是对前述“仅独立输出头”的进一步修改，不是原方案的同义词。Qwen3-VL-4B 隐维为 2560，保留原文字输入 embedding/文字 `lm_head`（扩表后为 `[155765, 2560]`，两者沿用 Qwen 的原绑权关系）。新建动作表 `E_action=[525,2560]` 和物理表 `E_physics=[2048,2560]`；两张表分别兼任本模态**输入向量表与输出分类矩阵**，即 `logits_action = h @ E_action.T`、`logits_physics = h @ E_physics.T`。没有额外复制一份动作/物理输出参数，也没有让它们和 Qwen 文字表共享参数。已有动作/物理 token ID 仍留在 Qwen tokenizer 中以保持数据与序列格式兼容，但其在原文字 `lm_head` 中的对应行不会用于动作/物理输出位置。

训练时先由原始 `input_ids` 计算 Qwen 图像感知的 M-RoPE `position_ids/rope_deltas`；再用 Qwen 原 embedding 处理文字/视觉占位符，把动作、物理位置替换为 `E_action/E_physics` 的查表向量，作为 `inputs_embeds` 送进 Qwen multimodal backbone。这样图像标记检查和图像特征注入保持原实现，ID 只是位置计算和监督标签，不再让动作/物理从 Qwen 文字输入表取向量。目标前一位置的隐状态按模态送往本模态绑权分类矩阵；仍是 8 动作接 64 物理的 teacher-forced CE。推理时前 8 步用动作表预测、后 64 步用物理表预测，生成 ID 回填后由对应模态表嵌入；KV cache 保留，增量 M-RoPE 使用预填充得到的 `rope_deltas` 加 `cache_position`。未改动动作/物理 tokenizer 本身。

优化范围：冻结视觉塔、较早语言层以及**原 Qwen 文字输入 embedding/绑权 `lm_head`**；训练最后两层语言层、最终 norm、`E_action`、`E_physics`。冻结文字表能避免码任务直接改写文字输入/输出行，但最后两层和 norm 仍会更新，**不能据此保证文字生成能力完全不退化**；需要另做同一文本评测的训前/训后对照。两张新表从相同的 100k-step Qwen warm-start 权重中对应码 ID 的行复制初始化，再作为新参数独立优化；此前“仅独立输出头”的 trainable-only checkpoint 与严格模式的参数键不同，不能直接加载或假装是续训。

兼容性边界：为了保持原 Qwen tokenizer、图像 chat template 和旧离散目标 ID 可复用，原 Qwen 文字表在**物理存储**上仍有动作/物理 ID 的冗余行，但严格模式输入路由和动作/物理输出均不读这些行；三模态的**实际使用参数**互不绑权。若未来单独调用原 Qwen 的文字 `generate()`，仍应屏蔽动作/物理 ID，避免把码符号当作自然语言输出；当前双流 Reasoner 没有提供文字生成接口。这不是“完整词表物理裁剪”，而是兼容原 Qwen 的严格模态参数路由。

实现入口为 `starVLA/model/framework/QwenDualAutoregressiveReasoner.py`，正式配置 `examples/Robotwin/train_files/qwen_dual_ar_8action_64physics_full50.yaml` 中 `separate_heads=true`、`strict_modality_tying=true`。runner `scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh` 默认两项均开启，run ID 添加 `_strict_tied_heads`；历史“仅独立输出头”可用 `DUAL_AR_STRICT_MODALITY_TYING=0` 显式重跑，历史共享头可用 `DUAL_AR_SEPARATE_HEADS=0` 重跑，分别使用独立 `_output_only_heads` 和 `_shared_heads` run ID，避免覆盖旧结果。手工评估严格模式用默认配置或 `--strict-modality-tying`，评估旧仅输出头 checkpoint 必须显式 `--output-only-heads`，评估旧共享头 checkpoint 必须显式 `--shared-heads`。请勿将这三类 trainable-only checkpoint 混装。

验证状态（2026-10-27）：`vlact` 环境下 9 项相关测试通过，包括输入/输出同表、文字表冻结梯度、原 ID 的 M-RoPE 与增量位置计算；严格模式在 node001 GPU 5 运行 10 步短训，完成反向、保存、严格复原、72-token 自由生成及两个任务 4 样本诊断。烟测 checkpoint 的 25 个可训练张量键里包含 `action_embeddings.weight [525,2560]`、`physics_embeddings.weight [2048,2560]`，**不包含** Qwen 文字 `embed_tokens.weight` 或 `lm_head.weight`，核实了冻结边界。烟测动作 CE 6.371、物理 CE 7.624，两个流的 top-1 均为 0，**仅是端到端烟测，不是性能对比**。烟测 checkpoint/结果在 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps10_strict_tied_heads_uniform_physics_init/`。之后同初始化的 500 步严格模式实验已独立完成，没有停止、删除或覆盖其他任务。

500 步正式作业的 warm-start 日志再次确认 `strict_modality_tying=True`、`uniform_physics_init=True`、267 个旧预留行重新初始化；模型共有约 4,454M 参数，其中约 **208.451M** 可训练，相比此前仅独立输出头方案的约 **607.209M** 少了约 398.758M，主要来自冻结原 Qwen 文字输入/输出绑权表。此数字只说明训练范围改变，不代表模型输出质量提高。

500 步同条件小面板结果（两个任务 `adjust_bottle`、`click_alarmclock`，各 16 个留出样本，合计 32；均使用同一 100k-step 基座、统一物理初始化和相同训练步数）：

| 架构 | 动作 teacher-forced top-1 | 物理 teacher-forced top-1 | 动作 CE | 物理 CE |
| --- | ---: | ---: | ---: | ---: |
| 共享头 | 25/256 | 0/2048 | 5.772 | 7.670 |
| 仅独立输出头 | 21/256 | 1/2048 | 5.342 | 7.638 |
| 严格分模态输入/输出绑权 | **19/256** | **1/2048** | 5.412 | 7.633 |

严格模式的 4 个 free-running 样本上动作/物理逐码准确率均为 0，整段准确率亦为 0。共享头 CE 在完整 Qwen 词表归一化，另两种在各自码表归一化，**不能跨方案仅凭 CE 数字排名**；三组 top-1 都很低且仅测两个任务，无法证明哪种架构有更好的机器人动作或文本生成能力。严格方案是按用户指定的参数隔离原则选作当前默认，**不是因该小面板取得性能优势**。严格结果位于 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps500_strict_tied_heads_uniform_physics_init/heldout_panel32.json`，checkpoint 为同目录 `final_model/pytorch_model.pt`；历史两版结果见本节上方路径。若要下最终性能结论，仍需全 50 任务留出、文本能力训前/训后测试与闭环 rollout。

## 全 50 任务严格模式 LoRA 长训（2026-10-27 启动）

用户允许最多约 20,000 步，并建议全参数或 LoRA。当前 node001 的 8 张 GPU 均在运行其他作业，所以先选择语言层 LoRA：冻结 Qwen 视觉塔、文字输入 embedding/绑权文字 `lm_head` 和语言基座；在全部 36 个语言层的 `q/k/v/o` 投影及 MLP `gate/up/down` 上加 rank=16、alpha=32、dropout=0.05 的低秩残差。严格动作表 `[525,2560]` 与物理表 `[2048,2560]` 继续分别输入/输出绑权并训练。语言层残差和两张表均由 AdamW 更新，各自学习率为 `1e-4`、`1e-4`、`2e-4`，每流 CE 等权。冻结文字表降低直接覆盖文字词向量的风险，但 **LoRA 仍会改变语言网络，文字能力需要单独测，不能推断完全保留**。

LoRA 通过注册在原 `Linear` 上的 forward hook 加入；低秩参数登记在 `lora_adapters`，因此原 Qwen state-dict 键不变，仍须严格 warm-start 相同 100k-step 基座。初始 up 矩阵为零，起始前向等于冻结基座；训练器先建优化器再 warm-start，所以 LoRA 必须在模型构造时创建。此实现不需要新装 `peft`，仍使用 `vlact` 环境；若真实模型烟测出现兼容问题，再使用用户提供的 HTTP 代理安装最小必要依赖，不升级 Torch/Transformers。相关单元测试 10/10 通过，**但单元测试不等于真实多卡训练成功**。

配置：`scripts/physical_tokens/qwen_strict_lora_full50.yaml`。数据为完整 50 任务、469,787 个有两流目标的 clip；训练排除 episode `[5,20,25,39,49]`，这些留出约 47,797 个 clip 不参与梯度。主训练计划最多 20,000 优化步，warmup 500，cosine 学习率，每 1,000 步保存可训练参数到独立 run ID。先做独立 10 步真实 Qwen/DeepSpeed 烟测，检查 714 个基座键、参数组、反向、保存、严格恢复和 72 码自由推理，再启动长训。旧 checkpoint 不会覆盖，也不会停止其他人的进程。验证至少在 1k/5k/10k/20k 对固定的全 50 任务留出 panel 检查动作/物理 teacher-forced CE、逐码 top-1、free-running 误差；最终再做更充分的评估。物理 CE 的均匀参考值为 `ln(2048)≈7.625`，若仍接近此值，不能把训练 loss 下降说成物理语言学会了。

运行器支持 `DUAL_AR_CONFIG`、`DUAL_AR_LORA=1`、`DUAL_AR_SAVE_INTERVAL`、`DUAL_AR_RUN_TAG`、`DUAL_AR_EVAL_ALL_TASKS=1` 和各任务采样数；发现同名 run 目录则拒绝覆盖。使用示例（按当时空闲显存选择 `CUDA_VISIBLE_DEVICES`，不要清理其他进程）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 DUAL_AR_CONFIG=scripts/physical_tokens/qwen_strict_lora_r32_full50_robotwin_finetune.yaml \
DUAL_AR_LORA=1 DUAL_AR_LORA_RANK=32 DUAL_AR_LORA_ALPHA=32 \
DUAL_AR_UNIFORM_PHYSICS_INIT=1 DUAL_AR_MAX_STEPS=20000 \
DUAL_AR_NUM_PROCESSES=4 DUAL_AR_MICRO_BATCH_SIZE=8 DUAL_AR_GRAD_ACCUM_STEPS=4 DUAL_AR_GLOBAL_BATCH_SIZE=128 \
DUAL_AR_SAVE_INTERVAL=1000 DUAL_AR_EVAL_ALL_TASKS=1 DUAL_AR_EVAL_SAMPLES_PER_TASK=16 \
DUAL_AR_EVAL_FREE_SAMPLES=1 DUAL_AR_RUN_TAG=robotwin_ft_manual_bs128_r32_v1 \
bash scripts/physical_tokens/run_full50_dual_autoregressive_reasoner.sh
```

本节只记录已启动的工作和预定验证口径；在真实烟测、长训和留出评估完成前，不宣称有 20k 结果。

用户随后明确指定新的正式起点：`/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3PI_Robotwin_all_Finetune/checkpoints/steps_100000_pytorch_model.pt`。这是已在 RoboTwin-All 上训练 100k 步的 `QwenPI_v4` checkpoint，**并非**上文原 `VLAct_Qwen3_Pretrain` checkpoint。两者的 `qwen_vl_interface` 均有 714 个相同名称、相同形状的张量；新文件另外有 632 个 PI 相关张量，当前 AR 架构不用这些旧 PI 头参数。新配置单列为 `scripts/physical_tokens/qwen_strict_lora_full50_robotwin_finetune.yaml`，不改写原配置和原烟测 checkpoint，也无需下载任何权重。由于预训练基座和 RoboTwin 微调基座权值不同，两者的 LoRA/码表可训练参数 checkpoint 不得交叉恢复；新基座先独立跑 10 步烟测，再独立开启长训。

原预训练基座 LoRA 烟测已完成 10 步：严格 warm-start 714 张量、39.617M 可训练参数、506 个可训练 checkpoint 键（其中 LoRA 504、动作/物理表各 1），完成保存、严格恢复与 72 码自由推理。两任务各 2 样本留出诊断动作 CE 5.974、物理 CE 7.689、两流 teacher-forced top-1 均为 0；样本极少，**仅验证流程，不能说明学习效果**。独立目录为 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps10_strict_tied_heads_uniform_physics_init_lora_r16_smoke_v1/`。用户指定的 RoboTwin 微调基座烟测使用另一个 run ID，结果以实际完成为准。

RoboTwin-All 微调基座的独立 10 步烟测也已完成：严格载入 714 个 Qwen 张量、39.617M 可训练参数，在 node001 4 卡上完成反向、保存、严格恢复与自由推理。两任务各 2 样本留出动作 CE 6.635、物理 CE 7.608、teacher-forced top-1 均为 0；自由生成的四个样本也未得到准确序列。它仅检验技术流程，**不能**据此比较两个基座的性能，更不能当作全 50 任务结果。独立目录为 `results/Checkpoints/qwen_dual_ar_8a64p_full50_seed42_steps10_strict_tied_heads_uniform_physics_init_lora_r16_robotwin_ft_smoke_v1/`，具体数值在 `heldout_diagnostic.json`。

用户新增正式训练约束：**全局 batch size 必须为 128**。烟测先于该约束，使用 4 卡 × 每卡 1 × 梯度累积 1 = 4，仅用于工程检查。最初的 ACP 配置是 4 卡 × 每卡 1 × 梯度累积 32；根据烟测约 1.5 秒/微步估算，20k 优化步会很慢，因此在它还处于 `STARTING`、没有 worker/训练日志时，**只停止了本任务自己的** `pt-te0apusx`，没有清理任何其他人的任务。重新提交的正式方案改为 4 张 A100-80GB × 每卡 8 × 梯度累积 4 = **128**；这提高吞吐但可能更占显存，需看真实 worker 是否稳定。运行器显式校验三者乘积，并将每卡 batch/累积步数传给训练器，Accelerate/DeepSpeed 的累积设置一致。

此前 rank 16 的 v2 启动脚本使用 `vlact` 环境、共享 AFS 路径，预定输出到独立 run ID `qwen_dual_ar_8a64p_full50_seed42_steps20000_strict_tied_heads_uniform_physics_init_lora_r16_robotwin_ft_acp_bs128_v2`，日志预定在 `artifacts/robotwin_full50_study/acp_logs/dual_ar_robotwin_ft_20k_bs128_v2.log`。ACP 4 卡任务资源名 `pt-48myty1f`（显示名 `ljn-robotwin-dualar-lora20k-bs128-v2`），2026-10-27 21:47 提交成功，但尚未优化即因用户要求 rank 64 而停止。**提交/STARTING 不等于已进入优化**，以持久日志中实际加载权重、显示 global batch 128 和优化步为准。此任务使用用户指定的 RoboTwin 微调基座，不需下载权重。

### 用户指定改为 LoRA rank 64（2026-10-27 22:00）

用户决定不等 rank 16 长训结果，直接把正式 LoRA rank 调到 **64**。旧 rank 16 任务 `pt-48myty1f` 在 ACP `STARTING`、没有 worker/日志/训练 checkpoint 时由我们停止，随后确认状态为 `SUSPENDED`；没有停止其他人的作业。rank 16 的烟测配置和权重保留为历史技术对照，**不能**把其 LoRA checkpoint 加载进 rank 64 模型。

新配置单列 `scripts/physical_tokens/qwen_strict_lora_r64_full50_robotwin_finetune.yaml`，只相对 rank 16 版本修改 `rank=64`、`alpha=128`（维持 `alpha/rank=2`），其余同一 RoboTwin-All 100k 权重、冻结范围、两张严格绑权码表、学习率、50 任务目标和 episode 留出不变。运行器的 run ID 后缀现在按实际 rank 生成，不再硬编码 `_lora_r16`。ACP 启动脚本仍保证 **4 卡 × 每卡 8 × 累积 4 = 全局 batch 128**，最多 20k 步，每 1k 步保存。新独立 run ID 为 `qwen_dual_ar_8a64p_full50_seed42_steps20000_strict_tied_heads_uniform_physics_init_lora_r64_robotwin_ft_acp_bs128_r64_v3`，持久日志为 `artifacts/robotwin_full50_study/acp_logs/dual_ar_robotwin_ft_20k_bs128_r64_v3.log`。新 ACP 资源名 **`pt-9chf8ksr`**（显示名 `ljn-robotwin-dualar-lora-r64-bs128`），已提交；当前是否真正启动仍需按日志核实。rank 64 理论上会提高 LoRA 参数量和显存需求，故必须检查 8 样本/卡是否稳定，如 OOM 则调低每卡 batch 并等比例调高累积步数，保持全局 128。

### 最新调整：冻结视觉，语言 LoRA rank/alpha 均为 32（2026-10-27 22:03）

按用户最新要求，正式方案改为**视觉编码器冻结**，36 层语言模型的注意力 `q/k/v/o` 和 MLP `gate/up/down` LoRA 均设 `rank=32, alpha=32, dropout=0.05`，故残差缩放 `alpha/rank=1`。原文字 embedding/文字 `lm_head` 和其他 Qwen 基座参数继续冻结；动作 `[525,2560]` 与物理 `[2048,2560]` 两张输入/输出绑权码表继续直接训练，不对码表使用 LoRA。这保持了先前严格模态隔离；rank 32 模型预计约 72.65M 可训练参数，须由真实运行日志核实。若“其他部分”特指还要解冻或加 LoRA 到视觉以外的额外模块，需要另外指定；本次沿用已经运行验证过的语言层目标范围。

rank 64 任务 `pt-9chf8ksr` 当时仍为 `STARTING`，无 worker/日志/训练 checkpoint，已**仅停止此本方任务**，随后提交独立 rank 32 任务 **`pt-0b3hrb0l`**（显示名 `ljn-robotwin-dualar-lora-r32a32-bs128`）。配置单列 `scripts/physical_tokens/qwen_strict_lora_r32_full50_robotwin_finetune.yaml`；4 卡 × 每卡 8 × 梯度累积 4 = 全局 batch 128 不变。新 run ID 为 `qwen_dual_ar_8a64p_full50_seed42_steps20000_strict_tied_heads_uniform_physics_init_lora_r32_robotwin_ft_acp_bs128_r32_v4`，持久日志为 `artifacts/robotwin_full50_study/acp_logs/dual_ar_robotwin_ft_20k_bs128_r32_v4.log`。提交成功不等于已开跑；后续必须检查真实 worker 日志和验证结果。

### 2026-10-29：单卡 40k 趋势、学习率调度修正与四卡迁移

单卡正式任务 `pt-ctakm6co` 采用 1×16×8=全局 batch 128，W&B offline，5k 保存一次可训练权重及完整 DeepSpeed 状态。10 月 29 日晚约 9.5k/40k 步时仍在运行；5k 的可训练权重和完整训练状态均已落盘。每 100 步日志按 1k 步窗口平均，物理 CE 从首个 1k 的约 5.98 降到 8–9k 的约 2.45；动作 CE 从约 0.304 降到约 0.11。没有观察到 NaN/OOM，但这些是 teacher-forced 训练 CE，不是留出集自由生成准确率。旧的重复单卡任务 `pt-my24ff1g` 已核实归属并停止；其已有检查点保留。

发现 `train_starvla.py` 的外部 LR scheduler 只按 40k **优化步**构造，却在梯度累积的每个**微批**都调用 `step()`；单卡累积 8 时学习率约在优化步 5k 到低点 1e-5，10k 又返回约 1e-4，不是预期的 40k 单程衰减。已改为仅在 `accelerator.sync_gradients` 时推进调度器。当前已启动的单卡进程不会热加载此代码修正；它只对新启动任务生效。

四卡候选为 4×32×1=全局 batch 128。历史单卡每卡微批 32、关闭梯度检查点的烟测已在第一个反向 OOM（约 79.28 GiB 占用），所以四卡同样每卡 32 时开启梯度检查点，**仍不用梯度累积**。最初未开检查点的本方四卡烟测 `pt-k9wo6uci` 在 STARTING、尚未开始训练时停止；替代烟测为 `pt-5b8mm5eg`（单卡 5k 权重，微批 32，累积 1，检查点开启）。四卡换 world size 后不能声称原样恢复单卡的 ZeRO 优化器状态：新脚本 `run_acp_dual_ar_32action_4gpu_40k_offline.sh` 从单卡可训练权重 warm-start，保留对应已完成优化步数并把**修正后的** LR scheduler 快进到该步，但 AdamW 动量重置、数据遍历顺序不保证逐位连续。新四卡保存的完整状态后续可在相同四卡配置下真正 resume。替代烟测是否成功、显存和速度须以实际 worker 日志与 ACP 状态为准。正式四卡长训应使用单卡 10k 权重，在其落盘且烟测通过后提交独立 run ID；新四卡稳定优化后再停止旧单卡，避免无可用训练作业。

动作 CE 约 0.1 不足以单独证明学习率过高。严格独立头下这是 525 类动作码的 teacher-forced 下一码损失，后续动作码可以依赖前面的真值码，32 码序列内部也可能高度相关；因此暂不单独降低动作码表学习率（仍为 1e-4），以全 50 任务留出集、自由生成与动作解码指标检查是否过拟合或忽略图像。

用户随后将资源方案改为 **2 卡 × 每卡微批 32 × 梯度累积 2 = 全局 batch 128**。原四卡烟测 `pt-5b8mm5eg` 的 ACP worker 长时间 `Pending`、无宿主机 IP/训练日志，核实仍未真正开始后停止了该本方任务；不把排队推断为代码或显存问题。新脚本 `scripts/physical_tokens/run_acp_dual_ar_32action_2gpu_40k_offline.sh` 保持梯度检查点开启，采用相同修正后的 LR scheduler、W&B offline、5k 保存和单卡权重跨卡数 warm-start 方式。先提交两卡一步烟测 **`pt-d1pfjyxu`**（源为单卡 5k 可训练权重，max 5001），验证两卡前后向、保存和评估；正式 40k 任务待烟测通过及单卡 10k 权重落盘后再提交。迁移重置 AdamW 动量，不宣称跨卡数完整优化器恢复。

两卡烟测 v1 `pt-d1pfjyxu` 在初始化阶段失败，尚未执行训练：YAML 默认 `uniform_physics_init: true`，而 warm-start 已训练码表时必须为 false；原 runner 只在值为 1 时覆盖 YAML，环境值 0 未生效。已让 runner 在显式值 0 时向训练器传 `--framework.dual_autoregressive_reasoner.uniform_physics_init false`，并以新目录重提 `pt-e30l1qx4`。v2 已在 ACP 双卡真实完成从单卡 5k 权重接续的第 5001 步，确认 `2×32×2=128`、检查点开启、无 OOM，进度条显示该首步约 11 秒；同时成功保存可训练权重及两卡完整 DeepSpeed 状态。首步时长不代表稳定训练吞吐，后续评估和 ACP 终态仍需核对。正式迁移仍以单卡 10k 检查点落盘为条件。

v2 烟测 ACP 最终状态为 `SUCCEEDED`，两任务各 1 样本诊断的动作 teacher-forced/自由生成准确率均为 31/32，但物理码接近 0；**样本极少，不是正式泛化结论**。进一步直接检查正式训练目标缓存 `dual_ar_32action_64physics_targets.npz`（469,787 条）：动作码全局覆盖 525 个 ID，各位置覆盖 512–525 个 ID，并非单一码塌缩；但相邻动作码有 **92.27% 相同**，抽样 1000 条序列的平均行内不同 ID 数约 **4.26/32**。所以低动作 CE 很大程度可由时间相邻码重复和 teacher forcing 解释；31/32 的小面板准确率也必须与简单的“沿用上一动作码”基线比较。暂不以此为由降低动作 embedding 学习率；后续报告应单列动作码变更位置准确率、自由生成准确率及解码动作误差，避免总体 token accuracy 被重复码夸大。

兼容性核对：本次 AR 训练器及运行脚本修改不要求恢复旧的非自回归推理代码。`QwenPI_v4`/`QwenPIv4PhysicalTokens` 模型实现、RoboTwin 客户端与服务端加载入口未随两卡迁移改动；旧完整 checkpoint 仍由自身配置选择原框架，不能把新 AR 的 trainable-only checkpoint 直接传给旧推理入口。当前代码上实际严格加载并编码/解码了旧 EMA 8-action、旧 FSQ 8-action、EMA 64-physics 三份 tokenizer checkpoint，形状分别为 `[1,8]→[1,32,14]`、`[1,8]→[1,32,14]`、`[1,64]→[1,16,16,2560]`；相关 33 个 tokenizer/cache 单元测试通过。尚未重跑旧非自回归模型的完整 RoboTwin rollout，不能把代码路径兼容等同于端到端实测。

按用户提供的 `docs/公司服务器项目上传 GitHub.docx` 检查旧版 GitHub 对照：当前本地 `HEAD` 与本地跟踪的 `backup/main` 均为 `d4eed39`（2026-10-25 的 Backup project），当前实验代码是其上的未提交工作树增量；没有从远端 pull、checkout 或覆盖任何当前文件。旧 `QwenPI_v4.py`、`QwenPIv4PhysicalTokens.py`、`QwenPIv4JointTokens.py`、RoboTwin 推理客户端/服务端文件在当前工作树未修改，旧与新 AR 框架名均能注册。当前环境下 HTTPS 实时 `ls-remote` 未成功，因此这里的“旧版”严格指本地 `backup/main` 追踪提交，不能声称已验证 GitHub 此刻远端最新 SHA。若未来发现具体旧推理回归，可用 `git show backup/main:<路径>` 只读对照并局部修复，不应整体恢复仓库以免抹去当前实验。

2026-10-30 梯度检查点显存烟测：保持正式双卡任务 `pt-bj7d48cl` 不变，修改双卡启动脚本使微批、累积、梯度检查点开关及完整状态保存标志可由环境变量覆盖，默认配置仍是 `2×32×2=128`、检查点开启。另提交独立 ACP 短测 `pt-o4sofhor`，从单卡 10k 可训练权重加载，配置 `2×32×2=128`、梯度检查点关闭、目标仅第 10001 优化步；独立 run tag/log，不覆盖正式检查点。截至 2026-10-30 00:03（Asia/Shanghai）ACP 显示 STARTING、无训练日志，**尚不能判断 OOM**。若这组 OOM，下一组应测试 `2×16×4=128`、梯度检查点关闭；只有实测不 OOM 且速度有利时才考虑迁移正式训练。没有停止任何其他任务。

用户要求为保证排队同时提交累积 4 的候选烟测。已提交 ACP 任务 `pt-xgxiyfse`：梯度检查点关闭，`2×16×4=128`，同样从单卡 10k 权重仅测试至第 10001 步；输出 run tag `robotwin_ft_acp_2gpu_gc0_m16ga4_smoke_v1`，日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_2gpu_gc0_m16ga4_smoke_v1.log`。它与 `pt-o4sofhor` 的输出完全分离，两者均不影响正式双卡任务。是否 OOM、实际速度需等待 ACP 运行日志，提交成功不等于烟测通过。

2026-10-30 00:14（Asia/Shanghai）：应用户明确要求，核对归属为 `liujn`、状态 `RUNNING` 后，仅停止旧双卡正式任务 `pt-bj7d48cl`；ACP 返回 successfully stop，随后状态确认为 `SUSPENDED`。两组梯度检查点关闭的烟测 `pt-o4sofhor` 与 `pt-xgxiyfse` 当时仍在 STARTING，没有被停止或修改。旧双卡任务约运行到 10.5k，尚未达到 15k 保存点，因此 10k 之后的未保存训练进度不可恢复；已有单卡 10k 权重及历史完整状态保留。未执行 delete，没有删除任何权重或目录。

2026-10-30 结果与后续：`pt-o4sofhor`（关闭梯度检查点、`2×32×2=128`）在第 10001 步第一个前向/反向阶段 CUDA OOM，两个 rank 的单卡进程均占用约 79.1 GiB，ACP 为 FAILED；**不能用于正式长训**。`pt-xgxiyfse`（关闭梯度检查点、`2×16×4=128`）完成第 10001 优化步、写出约 145 MB 的可训练权重并完成短评估，ACP 为 SUCCEEDED。它只验证了 1 步，尚不能宣称长期显存/吞吐稳定；日志中评估阶段再次出现的“Gradient checkpointing enabled”来自评估器按 YAML 加载模型，不是训练阶段开启。依据用户此前“若 OOM 则梯度累积加倍”要求，提交新的正式 40k 双卡任务 `pt-7xxpatjy`，配置 `2×16×4=128`、训练阶段关闭梯度检查点，仍从单卡 10k 可训练权重 warm-start、W&B offline、每 5k 保存、同配置可恢复；独立 run tag `robotwin_ft_acp_2gpu_bs128_r32_32action_m16_ga4_gc0_fixedlr_v1`，日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_2gpu_m16_ga4_gc0_bs128_r32_40k_fixedlr_v1.log`。提交时 ACP 状态 INIT，尚不能称已开始优化。旧双卡仍 SUSPENDED，未覆盖其结果。

新正式任务随后进入 RUNNING，确认从 10k 权重继续优化且在第 10019 步前未见 OOM。按相同全局 batch 128 比较，旧双卡开检查点 `2×32×2` 在 10.2k–10.5k 区间约 6.2–6.3 秒/优化步；新双卡关检查点 `2×16×4` 在第 10010–10019 步约 5.2 秒/优化步，初步约快 17%（吞吐约 1.2 倍）。新任务仅几十步，此数值是早期估计，不代表长期稳定吞吐或最终完成时间。

2026-10-30 W&B 离线上传：使用服务器已有 W&B 凭据及 HTTP 代理，把三个正式 run 上传至 `jianingliu/vlact-physical-tokens`：旧单卡 `urutb436`、已停止的旧双卡 `5i88lvnq`、新双卡 `44wu476h`。ACP 生成的原始离线目录归 root，普通账号不能更新同步状态；因此仅复制到 `/mnt/jfzn/liujn/wandb_sync_BGkdYn88/` 的临时副本，再按 W&B record 分片边界裁剪最后未完成的分片并执行 `wandb sync`。**原始 .wandb 文件和检查点未改动。**网页 API 验收时三条 run 的历史记录分别到优化步 10500、10300、10100。正在运行的 run 需要以后重新截取新快照同步，网页不会自动实时更新；离线片段缺少正常完成标记时云端可能显示 `failed`，不能据此判断 ACP 训练状态。

旧单卡 `urutb436` 的学习率曲线不是预期的 40k 单程余弦：其 runner 将 warmup 设为 500 **调度器调用**，旧训练器又在每个微批调用 scheduler.step，而该任务累积 8。日志中 LoRA LR 在优化步 100 约 `9.9987e-5`，5000 为 `1e-5`，10000 又回到 `9.9964e-5`；余弦在约 5000 优化步已到最低点，超过计划的 scheduler 总调用数后因余弦公式周期性回升，形成先降再升的 V 形。新双卡 `44wu476h` 已改为仅每个真实优化步调用一次 scheduler.step，runner 未覆盖 warmup，故使用 4000 优化步 warmup、40k 总优化步、10k 权重接续时定位在学习率约 `9.397e-5`（LoRA/动作组）和 `1.879e-4`（物理组）。跨单卡迁移重置 AdamW 动量，不是严格 full-state resume。

2026-10-30 01:01（Asia/Shanghai）：用户认为旧单卡 10k 的异常学习率轨迹污染了权重，明确要求双卡从头训练。新增独立脚本 `scripts/physical_tokens/run_acp_dual_ar_32action_2gpu_fresh_40k_offline.sh`：仍以用户指定的 100k-step RoboTwin VLA checkpoint 作为**预训练基座**，但不加载旧 10k 或任何其他双流 AR 可训练权重，也不恢复旧优化器/调度器；动作/物理码表按统一初始策略初始化，LoRA 重新初始化，AR 优化步从 0 开始。配置为 2×A100-80GB、每卡微批 16、累积 4、全局 batch 128、训练梯度检查点关闭、warmup 4000 个真实优化步、余弦计划总共 40000 步、W&B offline、每 5k 保存可训练权重与同配置可恢复的完整状态、结束后评估全 50 任务。脚本拒绝继承 `DUAL_AR_FULL_STATE_RESUME_DIR`、`DUAL_AR_INIT_TRAINABLE_CHECKPOINT` 或 `DUAL_AR_INITIAL_COMPLETED_STEPS`，run tag/log 与旧任务隔离。ACP 新任务 `pt-9fdc8294` 已提交，初始状态 QUEUEING；旧接续双卡 `pt-7xxpatjy` 核对归属后按用户意图 stop，ACP 返回成功并进入 SUSPENDING。旧检查点未删除；其他人的任务和旧单卡未动。是否真正从 0 开始优化，仍须以新 worker 日志确认。

用户随后要求先加入梯度范数的 W&B 记录并验证代码，再跑正式长训。`pt-9fdc8294` 在日志确认从基座初始化、完成第 1 步后按要求停止，ACP 最终 SUSPENDED；该短片段没有覆盖后续 run。训练器新加可选 `trainer.log_grad_norm`：在梯度累积边界读取 DeepSpeed ZeRO-2 engine 的 `get_global_grad_norm()`，记录 **unscaled、全局、裁剪前 L2 norm** 为 `grad_norm`，并在缺失或非有限值时立即报错；不使用 Accelerate 的 `clip_grad_norm_()` 返回值，因为该函数在 DeepSpeed 模式返回 `None`。非边界微批不读范数，也不单独记录。fresh runner 默认开启 grad norm，按原 `logging_frequency=100` 与 loss 一起写入 W&B offline；提供烟测用的步数/日志频率/输出 tag 覆盖及跳过评估开关，正式默认保持 40k、warmup 4k、每 5k 保存、全 50 任务评估。

验证：相关单元测试 15/15 通过，含 DeepSpeed 风格返回值、累积边界、缺失范数报错和 W&B 指标透传；脚本 `bash -n`、`git diff --check` 通过。真实 ACP 双卡 4 步 fresh 烟测 `pt-z1e8bm6v` 状态 SUCCEEDED，日志每步 `grad_norm` 分别约 8.9714、6.9951、19.1798、12.2268，W&B offline 摘要显示 grad_norm 曲线，最终可训练权重成功保存；值大于裁剪阈值 1 是因为记录的是裁剪前范数。随后提交新的正式 fresh 40k 任务 `pt-3md971eq`，独立 run tag `robotwin_ft_acp_2gpu_fresh_m16_ga4_gc0_correctlr_gradnorm_v2` 与日志 `artifacts/robotwin_full50_study/acp_logs/dual_ar_32action_2gpu_fresh_m16_ga4_gc0_correctlr_gradnorm_40k_v2.log`。提交成功不等于已稳定训练，后续须核对 ACP 状态、第 0 步 warmup 与每 100 步 grad_norm。`train_starvla.py` 原只读权限已恢复。

`pt-3md971eq` 后续已进入 ACP RUNNING；worker 日志显示仅严格加载原 100k-step VLA 基座（`Strictly warm-started`），没有 `Restored dual-AR trainable checkpoint` 或调度器快进，进度从 `0/40000` 增加到至少第 6 步，`2×16×4=128`。正式配置每 100 个优化步写 W&B offline 一条 `grad_norm`；首次正式梯度范数值应在第 100 步观察，不能拿 4 步烟测值当作正式长训指标。旧无 grad norm 版本 `pt-9fdc8294` 为 SUSPENDED，其权重与日志保留。用户未要求停止旧单卡任务，因此未动旧单卡或其他人的任务。
