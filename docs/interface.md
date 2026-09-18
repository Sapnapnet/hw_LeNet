# LeNet HLS 公共接口说明（初版）

> 项目：任务二——卷积神经网络（LeNet）推理加速器  
> 适用阶段：中期开发初版  
> 当前软件基准：`LeNet5-MNIST`  
> 说明：本文件是成员 1 负责维护的公共接口约定。所有 HLS 模块、Testbench、权重导入和整网集成都应遵守本文件；接口一旦冻结，不应由个人模块私自修改。

---

## 1. 网络结构冻结

当前网络采用面向 MNIST 28×28 单通道输入的 LeNet 变体，结构与当前 Python Baseline 保持一致。

| 层 | 输入 Shape | 参数 | 输出 Shape |
|---|---|---|---|
| Input | `[1, 28, 28]` | — | `[1, 28, 28]` |
| Conv1 | `[1, 28, 28]` | `Cin=1, Cout=6, K=5, S=1, P=0, Bias=False` | `[6, 24, 24]` |
| ReLU1 | `[6, 24, 24]` | ReLU | `[6, 24, 24]` |
| Pool1 | `[6, 24, 24]` | MaxPool `2×2, S=2` | `[6, 12, 12]` |
| Conv2 | `[6, 12, 12]` | `Cin=6, Cout=16, K=5, S=1, P=0, Bias=False` | `[16, 8, 8]` |
| ReLU2 | `[16, 8, 8]` | ReLU | `[16, 8, 8]` |
| Pool2 | `[16, 8, 8]` | MaxPool `2×2, S=2` | `[16, 4, 4]` |
| Flatten | `[16, 4, 4]` | CHW 顺序 | `[256]` |
| FC1 | `[256]` | `256 → 120, Bias=False` | `[120]` |
| ReLU3 | `[120]` | ReLU | `[120]` |
| FC2 | `[120]` | `120 → 84, Bias=False` | `[84]` |
| ReLU4 | `[84]` | ReLU | `[84]` |
| FC3 | `[84]` | `84 → 10, Bias=False` | `[10]` logits |

最终网络输出为：

```text
logits[10]
```

预测类别定义为：

```text
pred = argmax(logits)
```

FC3 后不再接 ReLU 或 Softmax。

---

## 2. 输入数据约定

### 2.1 输入尺寸

```text
C = 1
H = 28
W = 28
```

输入为 MNIST 单通道灰度图。

### 2.2 输入数值范围

当前 Python Baseline 使用 `torchvision.transforms.ToTensor()`，因此输入像素范围为：

```text
[0.0, 1.0]
```

当前版本没有额外的 Normalize 操作。

后续如果 Level 2 自采数据进入网络，也必须经过预处理后变换为与此处一致的：

```text
28 × 28
单通道
同一数值范围
同一前景/背景约定
```

---

## 3. Feature Map 数据布局

所有 Feature Map 统一采用：

```text
CHW
```

也就是：

```text
[channel][height][width]
```

当 Feature Map 使用一维连续数组存储时，统一使用以下索引：

```text
index = c * H * W + h * W + w
```

其中：

```text
c = channel
h = row
w = column
```

禁止各模块自行改变为 HWC 或其他排列方式。

---

## 4. Flatten 顺序

Pool2 输出为：

```text
[16, 4, 4]
```

Flatten 后长度：

```text
16 × 4 × 4 = 256
```

Flatten 必须保持 Python Baseline 的 C-contiguous CHW 顺序：

```text
flatten_index = c * 4 * 4 + h * 4 + w
```

也就是先完整展开 channel 0，再展开 channel 1，依次到 channel 15。

---

## 5. 权重布局

### 5.1 Conv 权重

统一采用：

```text
[OUT_CHANNEL][IN_CHANNEL][KH][KW]
```

对应当前 PyTorch `Conv2d.weight` 的原始布局，因此导出后不需要转置。

Conv1：

```text
[6][1][5][5]
```

Conv2：

```text
[16][6][5][5]
```

### 5.2 FC 权重

统一采用：

```text
[OUT][IN]
```

FC1：

```text
[120][256]
```

FC2：

```text
[84][120]
```

FC3：

```text
[10][84]
```

### 5.3 Bias

当前冻结网络：

```text
Bias = False
```

Conv1、Conv2、FC1、FC2、FC3 均不存在 Bias 参数。

任何 HLS 模块不得自行加入 Bias。

### 5.4 `.txt` 导出文件

当前 Python 导出脚本将权重以 NumPy row-major 顺序拉平成一维 `.txt` 文件。

因此 HLS 读取 `.txt` 权重时必须按照本节定义的原始 Shape 还原，不得自行转置。

---

## 6. 激活与池化约定

### ReLU

定义：

```text
ReLU(x) = max(0, x)
```

使用位置：

```text
Conv1 → ReLU1
Conv2 → ReLU2
FC1   → ReLU3
FC2   → ReLU4
```

FC3 后无 ReLU。

### Max Pool

统一采用：

```text
Kernel = 2 × 2
Stride = 2
Padding = 0
```

为非重叠最大池化。

---

## 7. 公共数据类型

所有 HLS 模块必须包含：

```cpp
#include "types.h"
```

公共类型统一为：

```cpp
data_t
weight_t
acc_t
output_t
```

当前 `types.h` 初版先使用 `float`，用于基础功能验证和模块联调。

成员 3 已交付 `member3-fixed-v1`：`quant_config.json` 和 `docs/member3_quantization.md`。公共 `types.h` 默认仍为 Float；全组切换定点时，统一设置编译宏 `LENET_USE_FIXED`，启用 `quant_params.h` 中的 W12/A12、32 位累加配置。各成员不得在自己的 `.cpp/.h` 中另起一套数据类型。Python / 独立整数 C++ 已验证，HLS 工具链验证仍由集成阶段完成。

---

## 8. 推荐的顶层 Accelerator 接口

顶层函数统一命名为：

```cpp
lenet_accelerator
```

初版接口：

```cpp
void lenet_accelerator(
    const data_t input[INPUT_SIZE],
    output_t logits[NUM_CLASSES]
);
```

其中输入为按照 CHW 规则展平后的 784 个元素：

```text
INPUT_SIZE = 1 × 28 × 28 = 784
```

输出为：

```text
10 个 logits
```

预测类别通过：

```text
argmax(logits)
```

得到。

> 顶层接口当前只输出 logits，便于成员 7 与 Python Reference 逐项对齐。Argmax 可以作为独立模块或 Testbench 中的后处理，不应替代 logits 输出。

---

## 9. 模块命名建议

为避免集成阶段出现同名或接口混乱，初版统一使用以下命名：

```text
conv2d
relu
maxpool2d
fc
argmax10
lenet_accelerator
```

成员 4、5、6 在提交模块时，如确有必要修改函数签名，应先与成员 1 同步，不能自行改变公共布局、Shape、权重顺序或数据类型。

---

## 10. 顶层执行顺序

整网逻辑顺序固定为：

```text
Input
  ↓
Conv1
  ↓
ReLU1
  ↓
Pool1
  ↓
Conv2
  ↓
ReLU2
  ↓
Pool2
  ↓
Flatten
  ↓
FC1
  ↓
ReLU3
  ↓
FC2
  ↓
ReLU4
  ↓
FC3
  ↓
logits[10]
```

顶层 Accelerator 后续负责把成员 4、5、6 提供的计算模块、Buffer 和 Controller 按上述顺序组织到同一个 HLS 工程中。

---

## 11. Python / HLS 对齐要求

成员 2 当前 Python Baseline 可导出以下中间结果：

```text
input
conv1
relu1
pool1
conv2
relu2
pool2
flatten
fc1
relu3
fc2
relu4
logits
```

HLS 调试阶段应尽量保留对应层级的可观测结果，以便成员 7 做逐层比较。

至少需要保证：

```text
Python Fixed / Float Reference
          vs
HLS C Simulation
```

在以下内容上可以逐项核对：

```text
Shape
数据顺序
中间输出
最终 logits
predicted class
```

---

## 12. 当前冻结项与待冻结项

### 当前已冻结

```text
网络结构
输入尺寸
各层 Shape
Conv 参数
Pool 参数
FC 尺寸
Bias=False
CHW 数据布局
Flatten 顺序
Conv 权重布局
FC 权重布局
网络最终输出 logits[10]
```

### 当前尚未最终冻结

以下内容成员 3 已提供 v1 实验结果，等待成员 1 统一启用和 HLS 集成验证：

```text
data_t 位宽
weight_t 位宽
acc_t 位宽
output_t 位宽
Rounding 规则
Overflow / Saturation 规则
```

在定点规则冻结前，各模块先使用公共 `float` 类型完成基础 C Simulation 与接口联调。
