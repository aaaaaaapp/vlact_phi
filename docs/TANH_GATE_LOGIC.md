# Tanh 版物理 token policy：完整逻辑与代码索引

本文说明当前 `qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000` 的 tanh gate 版本，方便接手训练、修改和评测。

## 1. 一句话概括

模型先用 Qwen-VL 得到视觉/语言 prefix，再把 action token 和 physics token 转成额外的条件 memory，注入 Pi action expert。physics memory 作为 residual 分支逐层加入 DIT/flow-matching action expert。每一层的 residual 强度由可训练标量 gate 控制：

```text
alpha_l = alpha_max * tanh(raw_gate_l)
expert_l = PurePI_l + alpha_l * residual_l
```

当前 tanh 配置为 `alpha_max=0.2`，所以每层 residual 的缩放范围为 `[-0.2, 0.2]`。tanh 允许正负修正，gate raw value 从 0 初始化，因此初始 residual 为 0。

## 2. 数据和 token 流程

训练样本包含图像、语言指令、state、action target、action token ids 和 physics token ids。P64/M32 表示 physics tokenizer 产生 64 个 raw physics ids，再按 `physics_group_size=2` 组成 32 个 Pi memory tokens；action stream 当前是 4 个 token。

主要代码：

- `starVLA/model/framework/QwenPIv4JointTokens.py`：joint action/physics token predictor、token CE loss、condition source；
- `starVLA/model/framework/QwenPIv4PhysicalTokens.py`：离散 token conditioner、residual memory 拆分和 gate；
- `starVLA/model/modules/conditioning/discrete_token_conditioner.py`：token embedding、grouping、condition memory 构造；
- `examples/Robotwin/train_files/qwenpi_v4_joint_tokens_d50_smoke.yaml`：基础模型和数据配置；
- `artifacts/token_caches/robotwin_p64_predicted_action4_physics64_full_step20000.npz`：训练使用的 token cache。

## 3. Prefix 到 action expert 的路径

```text
image + language + state
        ↓
Qwen-VL hidden states
        ↓ prefix_proj
Pure PI prefix
        ↓ action/physics token conditioner
conditioned prefix = Pure prefix + token memories
        ↓
Pi action expert / flow matching DIT
        ↓
continuous action chunk
```

`QwenPIv4JointTokens._encode_qwen_hidden()` 负责调用 Qwen-VL、预测 action/physics token logits、读取 cached 或 predicted token ids、构造 conditioned prefix，并保存两个 token CE loss。

对应代码：

- `starVLA/model/framework/QwenPIv4JointTokens.py:238`：`_encode_qwen_hidden()`；
- `starVLA/model/framework/QwenPIv4JointTokens.py:149`：`_condition_prefix_hidden_from_ids()`；
- `starVLA/model/framework/QwenPIv4JointTokens.py:206`：`_condition_prefix_hidden_soft()`；
- `starVLA/model/modules/conditioning/discrete_token_conditioner.py`：实际 memory 构造。

## 4. 两种 residual gate 结构

当 `physics_fusion_mode=pi_multilayer_residual` 时，每个 expert layer 有一个 `pi_discrete_layer_gates[layer]`，action 和 physics residual 共用它。

当前 tanh 实验使用 `pi_separate_multilayer_residual`，每层有两个独立 gate：

```text
pi_action_layer_gates[layer]
pi_physics_layer_gates[layer]
```

这样 action residual 和 physics residual 可以有不同强度，但目前仍然是“每层一个标量”，不是每个样本一个 gate。

对应代码：

- `starVLA/model/framework/QwenPIv4PhysicalTokens.py:75-92`：创建 gate 参数；
- `starVLA/model/framework/QwenPIv4PhysicalTokens.py:222-237`：按 layer 读取 action/physics gate；
- `starVLA/model/framework/QwenPIv4PhysicalTokens.py:240-246`：tanh/sigmoid/无边界映射。

## 5. tanh、sigmoid 和无边界 gate

```python
if mode == "tanh":
    alpha = alpha_max * torch.tanh(raw_gate)
elif mode == "sigmoid":
    alpha = alpha_max * torch.sigmoid(raw_gate)
else:
    alpha = raw_gate
```

- `tanh`：范围 `[-alpha_max, +alpha_max]`，可以抑制或反向修正 residual；
- `sigmoid`：范围 `[0, alpha_max]`，只能增加 residual，不能反向抵消；
- `none`：没有幅度边界，可能逐渐偏离成熟 Pure PI。

当前 tanh 版本配置为：

```yaml
framework:
  discrete_conditioning:
    pi_gate_bound_mode: tanh
    pi_gate_alpha_max: 0.2
    physics_fusion_mode: pi_separate_multilayer_residual
```

## 6. 训练 loss

基础 joint loss 是：

```text
L = L_flow
  + action_token_loss_weight * L_action_CE
  + physics_token_loss_weight * L_physics_CE
```

其中 `L_flow` 是 continuous action expert 的 flow-matching loss。

对应代码：

- `starVLA/model/framework/QwenPI_v4.py:500-570`：action flow-matching forward；
- `starVLA/model/framework/QwenPIv4JointTokens.py:302-318`：加 action/physics token CE。

## 7. 新增 distill-medium 版本

为了让 joint/tanh 不破坏 Pure PI 已经能完成的轨迹，新增了同一 backbone 的 Pure teacher distillation：同一 batch 使用同一个 diffusion time 和 noise，保存不带离散条件的 Pure prefix，用相同 action expert 得到 Pure teacher velocity，再用 MSE 约束 tanh/student velocity 接近 teacher；teacher 前向不回传梯度。

```text
L = L_flow + lambda(t) * L_distill
  + action_token_CE
  + physics_token_CE
```

本轮 `distill-medium` 使用 `lambda: 0.20 → 0.02`，训练 20000 steps。

对应代码：

- `starVLA/model/framework/QwenPIv4JointTokens.py:47-55`：蒸馏参数；
- `starVLA/model/framework/QwenPIv4JointTokens.py:260` 附近：保存 Pure prefix；
- `starVLA/model/framework/QwenPI_v4.py:581-615`：teacher velocity、蒸馏 loss 和 schedule；
- `scripts/physical_tokens/run_p64_m32_distill_medium_train.sh`：训练启动脚本。

训练命令：

```bash
P64_POLICY_RUN_ID=qwenpi_v4_p64_m32_distill_medium_seed42_steps20000 \
P64_ACTION_DISTILL_WEIGHT_START=0.2 \
P64_ACTION_DISTILL_WEIGHT_END=0.02 \
bash scripts/physical_tokens/run_p64_m32_distill_medium_train.sh
```

## 8. 推理时的两类使用方式

标准 tanh policy 由 `deployment/model_server/server_policy_starvla_compat.py` 加载 checkpoint，再由 `deployment/model_server/starvla_robotwin_serve_wrapper.py` 提供 Robotwin 接口。action expert 逐层按 gate 注入 residual。

另有实验性 Pure anchor selector：

- `deployment/model_server/conservative_episode_selector.py`；
- `scripts/physical_tokens/run_episode_selector_node4.py`。

它在同一随机噪声下分别请求 Pure 和 tanh，再按 episode 固定是否启用 bounded residual。这是诊断用推理 wrapper，不是 tanh 模型训练逻辑。

## 9. checkpoint 和评测

标准 tanh checkpoint：

```text
results/Checkpoints/qwenpi_v4_p64_m32_gate_tanh02_seed42_steps20000/
```

distill-medium checkpoint：

```text
results/Checkpoints/qwenpi_v4_p64_m32_distill_medium_seed42_steps20000/
```

评测必须保持相同 task、`task_config`、fixed seed、checkpoint config、action horizon 和 diffusion sampling 参数。常用入口：

- `scripts/physical_tokens/run_policy_single_seed.sh`；
- `scripts/physical_tokens/run_tanh02_four_tasks_demo_clean_node4.sh`；
- `scripts/physical_tokens/run_hard_episode_panel_node2.sh`；
- `examples/Robotwin/eval_files/eval_policy_fixed_seeds.py`。

## 10. 结果判断标准

distill-medium 的目标是同时满足：

1. Pure 已经成功的 episode 不明显下降；
2. Pure 失败而 tanh 能修正的 hard episode 仍保留收益。

重点统计 Pure 成功且 distill 失败的数量、tanh 成功且 distill 成功的 hard episode 数量，以及 click_bell、click_alarmclock、press_stapler、hanging_mug 和 beat_block_hammer 的逐 seed 结果。

如果 distill-medium 退化到 Pure 水平，应降低蒸馏权重；如果仍损失 Pure 成功样本，再考虑 episode-conditioned gate。
