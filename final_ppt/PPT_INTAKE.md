# 最终答辩 PPT 收件与整合清单（成员 1）

本目录只接收成员 3~8 已完成的 PPTX。成员 1 的职责是合并、排序、统一模板/字号/术语/数据口径和删重；不代做各成员内容页。

| 来源 | 应收内容 | 当前状态 | 合并前检查 |
|---|---|---|---|
| 成员 3 | W12/A12 位宽与资源依据 | 待接收 | Accuracy、资源、Latency、Clock、结论 |
| 成员 4 | Conv / MAC / Memory Banking | 待接收 | Conv1/Conv2 前后周期、II、示意图 |
| 成员 5 | ReLU/Pool/Flatten/FC | 待接收 | 64809→64551、FC 并行对比 |
| 成员 6 | Buffer/Controller/真实 latency | 待接收 | 64815 基线口径，禁止使用 474826 作为单图真实 latency |
| 成员 7 | Level 1 验证 + Level 2 批测 | 待接收 | 10000 MNIST、Level 2 Prediction/Accuracy、raw logits 一致性 |
| 成员 8 | Level 2 全流程和结果分析 | 待接收 | 数据集、Simple/Full、每类 Accuracy、混淆矩阵、错误样本 |

推荐总顺序：项目与 Level 1基础 → 原版本问题 → optimized 高精度优化 → 位宽资源依据 → aggressive5b 权衡 → 完整验证 → Level 2 → 最终结论。

收齐后，最终文件固定为本目录下的 `LeNet_Final_Defense.pptx`。
