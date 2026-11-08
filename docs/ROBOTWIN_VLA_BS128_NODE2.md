# Robotwin VLA：node2 四卡 global batch 128

入口：[run_robotwin_vla_bs128_node2.sh](../scripts/physical_tokens/run_robotwin_vla_bs128_node2.sh)

默认使用 QwenOFT + `robotwin_clean_randomized_50`（50 个任务的 Clean + Randomized LeRobot 数据），申请 node2 的 GPU 0、1、2、3。Clean 任务目录直接位于 `RoboTwin/<task>`，Randomized 任务位于 `RoboTwin/Randomized/<task>`；两者都注册为 AgileX-compatible `robotwin50`：

```text
4 GPUs × 16 samples/GPU × 2 accumulation = global batch 128
```

## 环境

使用现有 conda 环境 `vlact`。首次准备可在仓库根目录执行：

```bash
cd /mnt/jfzn/liujn/code/VLAct-PhysToken
conda activate vlact
bash scripts/setup/install_vlact_deps.sh
```

该环境至少需要 PyTorch CUDA、Transformers、Accelerate、DeepSpeed、OmegaConf、decord/av、flash-attn 以及本仓库 editable package。已有可用环境时不要重复覆盖 PyTorch。

## node2 路径

脚本变量都可覆盖。node2 当前已确认的数据路径是：

```bash
export REPO_ROOT=/mnt/jfzn/liujn/code/VLAct-PhysToken
export DATA_ROOT=/mnt/jfzn/liujn/code/Starvla/Starvla/playground/Datasets/RoboTwin
export BASE_VLM=/mnt/jfzn/liujn/code/VLAct-main/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
```

`BASE_VLM` 必须是本地 HuggingFace 模型目录；缺失时脚本会在启动前退出。若要从已有预训练权重 warm-start，可设置：

```bash
export PRETRAINED_CHECKPOINT=/mnt/jfzn/liujn/models/StarVLA/VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt
```

脚本默认使用 `robotwin_clean_randomized_50`，会在启动前同时检查 `${DATA_ROOT}/adjust_bottle` 和 `${DATA_ROOT}/Randomized/adjust_bottle`，并在数据加载日志中分别出现两类路径。若只想训练 Randomized，可设置 `DATA_MIX=robotwin_randomized_50`。

## 烟测与正式训练

先验证数据读取、模型初始化、DeepSpeed/NCCL 和首个前反向（4 个优化 step）：

```bash
cd /mnt/jfzn/liujn/code/VLAct-PhysToken
bash scripts/physical_tokens/run_robotwin_vla_bs128_node2.sh --smoke
```

烟测成功且显存稳定后启动正式 150k steps：

```bash
bash scripts/physical_tokens/run_robotwin_vla_bs128_node2.sh
```

正式长训默认 150,000 steps，输出位于 `results/Checkpoints/robotwin_randomized_qwen3oft_bs128_150k_node2_nogc_m16_ga2/`，每 5000 步保存、每 1000 步评估。梯度检查点已关闭，以换取更高显存占用和更快计算；`WANDB_MODE` 默认 disabled。

## 注意事项

脚本只使用 GPU 0--3，不会杀掉或触碰 GPU 4--7 上的进程；node2 当前卡上已有其他显存占用，正式启动前应确认余量。若 OOM，优先设置 `GPU_IDS` 为四张空闲卡后重试。

`PER_DEVICE_BATCH_SIZE` 和 `GRADIENT_ACCUMULATION_STEPS` 可以覆盖，但脚本强制检查 `4 × per_device × accumulation = 128`，避免误训成其他 global batch。
