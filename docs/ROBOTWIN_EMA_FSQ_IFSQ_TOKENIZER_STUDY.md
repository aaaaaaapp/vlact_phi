# RoboTwin 全训练集动作与物理 tokenizer：EMA-VQ / FSQ / IFSQ 对比

状态：进行中。本文只把实际完成并可复现的运行标为结果；待跑实验和历史基线分开记录。

## 1. 范围与数据

- 当前仓库已有的完整缓存覆盖 8 个 RoboTwin 任务，每任务 50 条 episode，共 85,098 个对齐样本。动作缓存：`artifacts/action_caches/robotwin_d15_8tasks_all_episodes_wrap32.npz`，每个样本是 `[32,14]` 的动作 chunk。
- 物理缓存：`artifacts/future_feature_caches/robotwin_d28_8tasks_all_episodes_spatial4x4_17frames.npz`；完整 future clips 共 72,298 个，按 episode `5,20,25,39,49` 留出验证，训练 64,589 个、验证 7,709 个。可选的动作相关辅助目标在 `robotwin_d28_8tasks_all_episodes_predictable_residuals17x14.npz`。
- D28 缓存通过形状、finite、任务/episode 数及与动作缓存 key 对齐校验；详见 `artifacts/evaluations/d28_full_feature_cache_validation.json`。当前仓库明确可用的是这 8 个任务的完整缓存。更大 RoboTwin 任务全集的路径/定义待确认。

## 2. 比较协议

1. 三种量化器分别用于**动作**与**物理** tokenizer。每一模态内部固定同一输入、episode 划分、encoder/decoder、token 位置数、近似离散容量（bits/token）、训练步数与评估样本；仅替换量化器及其必要的专有超参数。动作与物理的损失量纲不同，不做跨模态 loss 排名。
2. 先复现共同预算，再对每种方法分别调优（EMA decay/commitment/dead-code；FSQ levels/投影/分布正则；IFSQ 按原论文机制和建议范围）。调优只看验证集，最终选择应另用固定测试集或明确注明属于验证集选择结果；不从训练 loss 单点选优。
3. 每次完整评估保存 checkpoint hash、配置、数据 hash、seed、训练时长、训练/验证损失分项、重建误差、按任务宏平均和总体微平均。动作包括关节 wrapped MAE、速度 MAE、夹爪准确率；物理包括 quantized/continuous/no-motion RMSE、action auxiliary loss（若启用）。
4. 码分布在**相同的完整验证集**累计 ID 直方图，再计算 active codes/K、Shannon entropy `H=-Σ p log p`（nats）、`exp(H)` perplexity、top-10 ID 总频率、归一化 entropy `H/log K`。另外保存逐任务与逐位置统计。不能把 batch active ratio 当成全局码本利用率；FSQ/IFSQ 的 ID 是离散组合，EMA-VQ 的 ID 是 learned codebook entry。
5. 选优顺序：先保证无数值问题和可接受的 held-out 重建，再比较码本分布；最后通过相同 predictor/下游策略预算比较可预测性和动作效果。若指标冲突，分别报告 Pareto 关系，不用单个 perplexity 宣称赢家。

## 3. 当前证据与待解决问题

| 项目 | 已有证据 | 本轮状态 |
|---|---|---|
| 动作 EMA-VQ | 历史单任务 `click_alarmclock` 的 8×512、5000 step 验证：关节 MAE 1.64°、active 128/512、perplexity 55.19。不是这次 8 任务全量重训结果。 | 待重训 |
| 动作 FSQ | 同一历史单任务 8×8×8：关节 MAE 2.39°、active 80/512、perplexity 31.79。 | 待重训 |
| 物理 EMA-VQ | D29 历史全量 256 tokens × K256、8000 step，held-out 宏平均 quantized RMSE 0.083225、active ratio 0.97998、perplexity 206.77。 | 可作参考基线，待同协议重跑 |
| 物理 FSQ | 仓库已有量化器及训练入口，尚无与 D29 全量同协议的成套结果。 | 待训练 |
| 动作/物理 IFSQ | 当前仓库尚无实现；指定论文 `https://arxiv.org/pdf/2601.17124` 在本机无法访问。 | 须先取得论文并核对算法，再实现及训练 |

## 4. 运行记录

2026-10-25：检查了仓库、D15/D28 缓存 manifest、既有训练/评估脚本和本机 GPU 占用。8 张 A100 均有其他任务，占用约 46–75 GiB/卡；不能把剩余显存视为独占资源。访问 arXiv PDF、摘要页及其他镜像均连接超时。已向用户询问论文 PDF 与“RoboTwin 全部训练集”的任务范围。当前尚未训练新的三方法 checkpoint，因此**无本轮赢家结论**。

## 5. 待完成

- 取得 IFSQ 论文，记录准确公式、可学习参数、梯度估计、量化范围和论文推荐设置；实现并用梯度/编码解码测试验证。
- 固定两类 tokenizer 的共同训练和 held-out 评估协议，补齐全量累计码分布统计。
- 为各方法做共同预算与方法内调优，完成动作、物理各三组正式训练和逐任务评估。
- 视下游成本完成相同预算的 token predictor/策略对照；若只比较 tokenizer，结论明确限定在 tokenizer 指标。
