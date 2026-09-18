# 成员三交接：定点量化与 Fixed Golden Model v1

负责人：成员三；版本：`member3-fixed-v1`；日期：2026-09-04。

## 1. 结论与完成范围

- 原始 Float 基准已复现：MNIST 10,000 张，9897 正确，98.97%。
- 中期采用 W12/A12：9894 正确，98.94%，精度下降 **0.03 个百分点**。
- 全测试集与 Float 预测不同的样本有 7 张，净少正确 3 张；不要把两者混为一谈。
- 权重及各量化边界在该测试集的饱和次数均为 0。
- 已完成 8～16 bit 共 9 档 PTQ 精度实验；没有重训练，没有改变无 Bias 网络。
- 纯 NumPy int64 与整数精确 FP64 后端在选定配置、全测试集、13 组中间输出中比较了 115,380,000 个整数值，差异为 0。
- 独立标量 C++ int64 实现核对 5 个 Reference、57,690 个整数值，差异为 0。
- **以上不是 HLS C Simulation / RTL / 综合结果**；资源和延迟都标记为 `NOT_MEASURED`。

## 2. 定点格式（整数位包含符号位）

| 类型 | 总位宽 W | 整数位 I（含符号） | 小数位 F | 实数 = raw × scale | 范围 |
|---|---:|---:|---:|---:|---|
| `data_t` | 12 | 7 | 5 | 1/32 | [-64, 63.96875] |
| `weight_t` | 12 | 1 | 11 | 1/2048 | [-1, 0.99951171875] |
| `product_t` | 24 | 8 | 16 | 1/65536 | 完整乘积 |
| `acc_t` | 32 | 16 | 16 | 1/65536 | [-32768, 32768-1/65536] |
| `output_t` | 12 | 7 | 5 | 1/32 | 同 `data_t` |

采用二进制 scale、零点 0；不是不加说明地把各层独立除以 max 后直接串联。权重已在 [-1,1] 内，不需要改变网络中的权重幅值。激活的量化映射由 `real = raw / 32` 表示，保证层间数值语义一致。等效归一化激活可写成 `u = real / 64`，`raw = round(u * 2048)`，但硬件解释该 raw 时必须恢复上述 scale。

范围来源：从 MNIST **训练集**用固定随机种子 `20260904` 选出 2048 张，记录所有激活的 min/max，并加 25% 裕量。观察到最大绝对激活约 26.64383，加裕量后约 33.30478，故统一选择有符号 7 个整数位。具体样本索引、权重范围及每层范围在 `results/quant/calibration.json`。不使用这 5 张 Reference 代替校准集。

统一类型优先降低中期集成复杂度；未按层单独优化格式，也没有利用 ReLU 非负性去掉符号位。后续可以在保持对齐的条件下研究混合精度。

## 3. 必须一致的计算顺序

1. 输入、权重：`q = clip(round_to_nearest_ties_even(real / scale), -2048, 2047)`。
2. Conv/FC：计算全部 `q_input * q_weight`，保留完整乘积小数位，使用宽累加器。
3. 点积结束：对原始累加整数 **右移 11 位并作最近偶数舍入**，再饱和到 [-2048,2047]。
4. 然后执行 ReLU / MaxPool；二者不引入新的 scale。Conv/FC 的饱和在 ReLU 前。
5. Flatten 按 CHW 的 C-contiguous 顺序；最终 FC 后也量化，但无 ReLU、无 Softmax。
6. Argmax 相等时取最小索引，与 NumPy / PyTorch 一致。

舍入例子：-2.5→-2，-1.5→-2，-0.5→0，0.5→0，1.5→2，2.5→2。不能用普通截断代替，也不能把负数简单当正数右移处理。

最大点积长度是 FC1 的 256。每个 12×12 位有符号乘积的绝对值不超过 2^22，256 项绝对和不超过 2^30；32 位有符号累加器足够。乘积有 16 个小数位，累加器也保留 16 位，因此中间加法不需要舍入或溢出处理。**不要在每次 MAC 后把结果转回 `data_t`。**

参考 HLS 写法：

```cpp
#include "types.h"  // 全组统一 -DLENET_USE_FIXED
acc_t sum = 0;
for (int i = 0; i < N; ++i) {
    product_t product = input[i] * weights[i];
    sum += product;
}
data_t output = sum;  // 唯一缩窄点，AP_RND_CONV + AP_SAT
```

AMD 定义中整数位包含符号位，量化/溢出策略在初始化或赋值到目标类型时生效，参见 [Fixed-Point Identifier Summary](https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Fixed-Point-Identifier-Summary) 和 [Quantization Modes](https://docs.amd.com/r/2023.2-English/ug1399-vitis-hls/Quantization-Modes)。

## 4. 文件编码：尤其注意 raw 不是实数

- `weights/fixed/*.npy`：int64 容器里的 12 位有符号原始整数；不是 64 位硬件权重。
- `weights/fixed/*.txt`：同样的原始整数，十进制，row-major 展平；只有五组权重，无 Bias。
- `fixed_reference/sample_XXXXX/*.npy` / `.txt`：各层原始整数，无 batch 维。
- `fixed_reference/sample_XXXXX/dequantized/*.npy`：为画图/与 Float 比较提供的实数，不用于逐位比较。
- 形状、格式、预测类别、checkpoint SHA256 均在对应 `manifest.json`。

例如激活文件里的 `32` 表示实数 `1.0`，不是实数 `32.0`。HLS 可以按位装载：

```cpp
data_t value;
ap_int<LENET_DATA_W> raw = 32;
value.range(LENET_DATA_W - 1, 0) = raw;
```

读取权重应使用 `LENET_WEIGHT_W` 和 `weight_t`；或者先按 scale 转成实数再赋给目标定点类型。**不可直接执行 `data_t value = raw;`**，否则量级错误。

成员 7 对齐时应比较 Python raw 与 HLS 输出的有符号 raw 整数，预期 mismatch=0；不要把 Fixed-vs-Float 的量化误差当成 HLS 可容忍误差。

## 5. 实验结果

全部使用相同 10,000 张 MNIST 测试图、相同预处理 `[0,1]`、相同权重与统一整数位（A 的 I=7，W 的 I=1）。每档的累加位宽同步取 `2*bits+8`，所以这不是固定累加位宽的独立单变量实验。

| W/A | 准确率 | 相对 Float 下降（百分点） | 累加位宽 |
|---|---:|---:|---:|
| 8/8 | 98.79% | 0.18 | 24 |
| 9/9 | 98.86% | 0.11 | 26 |
| 10/10 | 98.98% | -0.01 | 28 |
| 11/11 | 98.92% | 0.05 | 30 |
| **12/12** | **98.94%** | **0.03** | **32** |
| 13/13 | 98.95% | 0.02 | 34 |
| 14/14 | 98.97% | 0.00 | 36 |
| 15/15 | 98.97% | 0.00 | 38 |
| 16/16 | 98.97% | 0.00 | 40 |

中期按预先采用的工程策略选择：从至少 12 bit 开始，精度下降不超过 0.10 个百分点且无观察到的饱和，取第一个通过的配置。这是稳定基线选择，**不是最小位宽证明或 PPA 最优结论**。10 bit 本次净多正确一张，不意味着量化一定提高泛化能力；低位宽改变了决策边界，结果可以非单调。14～16 bit 在本测试集上预测类别已与 Float 一致，但 logits 仍有量化误差。

W12/A12 的全测试集 logits 最大绝对误差 0.32865334，平均绝对误差 0.04977027。逐样本逐层 Float/Fixed 误差见 `results/quant/reference_float_vs_fixed.csv`。

完整结果：`quant_accuracy.csv`；可用于 PPT 的图：`results/quant/bitwidth_accuracy.png`。准确率字段是 0～1 比例，`accuracy_drop_pp` 是百分点。

## 6. 复现命令（主目录 PowerShell）

现有 `event` 环境已经足够，无需 pytest，无新增依赖。不要重新运行训练来覆盖成员二 checkpoint。

```powershell
# 完整校准、9 档精度扫描、全测试集整数复核、导出和图表
& 'D:\anaconda\envs\event\python.exe' -B -u project/python/quant/run.py

# 无 CUDA 时可运行 CPU 版本（更慢）
& 'D:\anaconda\envs\event\python.exe' -B -u project/python/quant/run.py --device cpu

# Python 回归测试
& 'D:\anaconda\envs\event\python.exe' -B -m unittest discover -s project/tests/quant -v

# 独立 C++ int64 核对，无需安装 HLS
g++ -std=c++11 -O2 project/tests/quant/check_reference.cpp -o project/results/quant/check_reference.exe
& '.\project\results\quant\check_reference.exe' project
```

运行脚本会覆盖成员三的同名生成结果，不写成员二的 `weights/exported`、`reference/float`、`baseline_accuracy.*`。请先备份希望保留的旧定点实验版本。固定输入得到可重复的配置和结果；`run_summary.json` 中记录环境和源码 SHA256。

`fixed.py` 的 `IntegerLeNet` 是权威整数实现，独立于 HLS。快速后端用 FP64 存放整数编码，乘积及任意部分和的绝对值上界小于 2^53，因而不是仅对权重 fake-quant 后做 Float 推理。选定配置已用全测试集逐层验证；其他位宽各用 8 个样本逐层核对。

## 7. HLS 接入与剩余工作

- `config/types.h` 默认 Float，定义 `LENET_USE_FIXED` 后启用统一定点类型。
- `config/quant_params.h` 从 `quant_config.json` 同次生成，不要单独改其中一个。
- `tests/quant/check_reference.cpp` 支持 `LENET_USE_FIXED`：配置 HLS 的 `ap_fixed.h` include 路径后，可将同一测试切到 ap_fixed 运算。尚未运行该模式，亦未使用 Vitis 工程运行 C Simulation。
- 需要成员 1 确认全组切换编译宏，成员 4、5 按 MAC 和缩窄规则实现，成员 7 进行真正的 HLS 逐层 raw 对齐。
- 需要成员 4/6/8 提供同配置的 LUT/FF/DSP/BRAM/Latency 报告；本交付没有估算资源或冒充完成 Level 3 综合实验。
- 本次位宽选择查看了测试集准确率，沿用课程实验方式，不把它宣称为未经选择的独立最终 holdout。后续泛化结论需额外数据。
- 校准裕量不保证未来自采数据永不饱和，接入 Level 2 时仍需统计范围和饱和率。
- 原始 `weights/exported` 的 Bias 文件属于当前 manifest 之外的遗留文件，未删除，严禁误加载。

## 8. 交付对象

| 对象 | 交付内容 |
|---|---|
| 成员 1 | `quant_config.json`、公共类型开关、生成参数头、本说明 |
| 成员 4、5 | 定点权重、完整乘积/累加规则、每层输出缩窄规则 |
| 成员 7 | `fixed_reference`、manifest、回归测试与独立 C++ 校验器 |
| 成员 8 | `quant_accuracy.csv`、准确率图、Float/Fixed 逐层误差表 |

成员四可直接接收自包含目录 `handoff/member4_conv_mac/`。其中聚合了成员一、二、三与 Conv/MAC 相关的内容，另含完整性校验、人工/随机/单位脉冲/真实样本测试向量、portable/HLS 双模式 C++ 校验器和回交模板。请先读包内 `README.md`，不要从源工程自行挑选散落文件。

本项目目录未包含 Git 仓库，因此以文件路径、版本字符串和 SHA256 标识交付，不虚构 Commit。
