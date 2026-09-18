# LeNet网络结构与输入核对

日期：2026-09-08。依据成员1/2本次直接交付的[模型](../../../provided_by_member1/python/model.py)、[接口说明](../../../provided_by_member1/docs/interface.md)和[网络配置](../../../provided_by_member1/config/network_config.h)。网络元数据与两份导出清单一致，网络配置与成员6副本哈希相同。这里只描述本包网络，不是全组硬件总体架构或最终冻结声明。

## 1. 网络结构

| 层 | 输入形状 | 输出形状 | 运算与参数 |
|---|---|---|---|
| input | 1×28×28 | 1×28×28 | 单张输入，不含批次维度 |
| conv1 | 1×28×28 | 6×24×24 | 5×5，stride=1，padding=0，无Bias |
| relu1 | 6×24×24 | 6×24×24 | ReLU |
| pool1 | 6×24×24 | 6×12×12 | MaxPool 2×2，stride=2，padding=0 |
| conv2 | 6×12×12 | 16×8×8 | 5×5，stride=1，padding=0，无Bias |
| relu2 | 16×8×8 | 16×8×8 | ReLU |
| pool2 | 16×8×8 | 16×4×4 | MaxPool 2×2，stride=2，padding=0 |
| flatten | 16×4×4 | 256 | 按CHW顺序展平 |
| fc1 | 256 | 120 | 全连接，无Bias |
| relu3 | 120 | 120 | ReLU |
| fc2 | 120 | 84 | 全连接，无Bias |
| relu4 | 84 | 84 | ReLU |
| fc3 | 84 | 10 | 全连接，无Bias；输出logits，无ReLU/Softmax |

Conv权重为OIHW，FC权重为[OUT][IN]。Flatten索引为c×H×W+h×W+w。共44,190个weight参数，不计目录中未被manifest采用的bias文件。图中FC3输出10个logits，argmax作为后处理，不算额外训练层。

![LeNet网络结构](../../figures/lenet_network_structure.png)

图可直接用于网络结构展示；总体硬件架构和Float→Fixed→HLS验证流程另待整理。

## 2. 与现有自采预处理的核对

| 项目 | 本包Baseline依据 | 当前预处理 | 本次结论 |
|---|---|---|---|
| 尺寸/通道 | 单张1×28×28 | 1×28×28 | 一致 |
| 布局 | CHW，展平按C连续顺序 | CHW | 一致 |
| 类型/范围 | 导出输入float32，范围0～1 | float32，范围0～1 | 一致 |
| 归一化 | ToTensor，uint8像素除以255；无Normalize | PNG像素除以255一次 | 一致 |
| 极性 | MNIST原始灰度，背景0、笔画较亮；输入样例对应IDX | 黑底白字 | 约定一致，实际自采质量仍依图像判断 |
| 批次维度 | 导出脚本image.unsqueeze(0) | NPY不含batch维度 | 接入PyTorch时需增加N维，不需重做NPY |
| 几何处理 | Baseline直接读取已为28×28的MNIST；无额外裁剪/居中 | ROI阈值144，最长边20，居中补边 | 自采策略保留；不是已被Baseline证明的最佳参数 |
| 定点入口 | 本包初版types.h只定义Float | 当前NPY也是Float | 定点编码继续采用成员3规则，尚未联调 |

本次重新读取run_002的10个NPY及对应PNG，确认形状、类型、有限值、0～1范围及逐值等于PNG/255，未重跑预处理。除数不能重复应用。未运行模型，未验证识别准确率或新的裁剪参数；无需为本次材料核对更改默认144或重新生成图片。
