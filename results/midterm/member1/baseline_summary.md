# 成员1/2 Float基准接收汇总

整理日期：2026-09-08。成员1与成员2由同一人承担，本包按公共网络定义及Float基准职责接收。以接收日期和文件哈希标识本次材料，未提供独立交付版本号，不将其等同于全组中期冻结版本。

## 1. 结果与来源

| 内容 | 结果 | 依据 |
|---|---|---|
| Float测试准确率 | 98.97%，9897/10000 | [准确率JSON](../../../provided_by_member1/results/baseline_accuracy.json)，与TXT及成员3转交记录一致 |
| 测试集 | MNIST test，10000张 | [评测代码](../../../provided_by_member1/python/inference.py)，train=False、ToTensor |
| 训练记录 | 10个epoch，最佳为第10个，test_accuracy=0.9897 | [训练记录](../../../provided_by_member1/results/baseline_training.json) |
| 设备 | cuda，原记录申报 | 准确率及训练JSON |
| checkpoint | 与成员6包内副本SHA-256一致 | [核查记录](verification.json) |
| 参数 | 5组weight，共44,190个参数，无Bias | [权重清单](../../../provided_by_member1/weights/exported/manifest.json)及[模型](../../../provided_by_member1/python/model.py) |
| 参考样本 | MNIST测试索引0～4；每个13组输入/逐层输出 | [原始清单](../../../provided_by_member1/reference/float/manifest.json)、[核查数据](reference_samples.json) |

本次没有重新训练、评测或导出，也未加载执行checkpoint。98.97%是已有实验结果的直接交付，与此前经成员3转交的是同一记录，不算第二次独立实验。代码按每轮test_accuracy选最佳checkpoint，因此该测试集也参与模型选择，不表述为完全独立的最终留出评测。优化器等代码默认参数不冒充已核实的实际运行参数。

## 2. 权重及附带文件

5组正式weight的NPY形状与清单一致，TXT与NPY仅有9位有效数字文本导出的精度差；其TXT数值与成员6使用的对应文件全部一致。checkpoint哈希相同、网络配置文件相同，支持关联现有整网结果，但不构成全组版本冻结确认。

目录另有5组bias的TXT/NPY，共10个文件；当前model.py明确bias=False，manifest也不列这些bias。本次不采用它们，不删除，也不推测其生成历史。正式参数清单以manifest和当前模型一致的5组weight为准。

## 3. 参考样本与成员6日志关联

| 测试索引 | 原始标签 | 本包Float预测 | 成员6 Float日志预测 | 成员6 Fixed日志预测 |
|---|---:|---:|---:|---:|
| 0 | 7 | 7 | 7 | 7 |
| 1 | 2 | 2 | 2 | 2 |
| 2 | 1 | 1 | 1 | 1 |
| 3 | 0 | 0 | 0 | 0 |
| 4 | 4 | 4 | 4 | 4 |

本包输入逐值对应MNIST原始图像除以255后的float32数组，标签对应原始IDX标签。成员6 Float输入与之相同；Fixed输入等于乘32并作ties-even舍入的整数编码。因此可以补充确认：成员6两种模式日志中的这5个样本预测与真实标签一致。这里5/5仅是参考样本检查，不是整网测试集准确率；没有重跑HLS。

65组参考张量的形状、float32类型、有限值及TXT/NPY一致性已核查。新参考从input到flatten与成员6包内对应TXT数值一致；FC后半段存在小幅差异，最终logits两套参考的最大绝对差约1.2e-6。成员6后半段参考由成员5按自己的计算流程生成，见[其来源说明](../../../provided_by_member6/m5/data/PROVENANCE.md)。本次只报告参考间实际差异，不单凭精度描述归因。

成员6历史误差仍是相对其原参考的误差，原仿真表和原始日志不修改；未保存HLS侧完整logits，不能重算相对新参考的精确误差。此前“未核验真实标签”是接收当时的状态，本节补充标签证据，不改变原日志含义。逐层参考文件也不是HLS逐层验证结果，成员7完整对齐仍待交付。

## 4. 网络与输入规范

见[网络结构与输入核对](network_structure.md)及[网络图](../../figures/lenet_network_structure.png)。本次已完成本包Baseline与现有预处理结果的格式和数值约定核对；未运行自采识别，不据此宣布模型效果通过。

公共接口说明及types.h为初版，顶层建议lenet_accelerator(input, logits)，类型为Float；成员6实际顶层显式传入权重。分别注明阶段，不用本包初版类型覆盖成员3定点配置或成员6实际接口。全组总体架构、完成度和最终冻结版本仍待后续提供，不阻断本次已交付范围整理。

## 5. 文件用途

- model.py：定义网络及中间输出边界。
- train.py：训练、按测试准确率保存最佳checkpoint和训练记录。
- inference.py：读取checkpoint评测MNIST测试集并保存准确率。
- export_weights.py：导出模型state_dict的weight为NPY/TXT，并生成参数清单。
- export_reference.py：对指定MNIST测试样本导出13组输入/逐层结果及标签、预测清单。

以上是阅读代码所得功能说明，不表示本次运行这些脚本。原始目录不改动，复现不属于当前范围。
