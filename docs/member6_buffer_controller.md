# 成员六 Buffer / Address / Controller 交付

- Vivado HLS 2018.3  
- xc7z020clg400-1
- Clock: 10 ns

## 1. 交付结论



做了什么：

1. 层间数据线性 Buffer. 统一使用 CHW、OIHW 和 FC `[OUT][IN]` 地址规则；
2. 顺序 Controller 管理层间执行；
3. 将 Conv、ReLU、Pool、Flatten、FC 接成一次完整推理。

没做什么：

- 内部 Weight ROM；
- 权重二次复制和复杂搬运；
- Ping-Pong Buffer；
- DATAFLOW 和层间重叠；
- AXI 接口和最终 IP 封装。



## 2. 冻结网络

```text
1x28x28
 -> Conv1(6)
 -> ReLU1
 -> MaxPool1
 -> Conv2(16)
 -> ReLU2
 -> MaxPool2
 -> Flatten(256)
 -> FC1(120)
 -> ReLU3
 -> FC2(84)
 -> ReLU4
 -> FC3(10)
 -> logits[10]
```

数据布局：

```text
Feature Map：CHW
Conv 权重：OIHW
FC 权重：[OUT][IN]
Bias：无
FC3 后：不接 ReLU 或 Softmax
```

## 3. 顶层接口

```cpp
void lenet_accelerator_with_weights(
    const data_t input[INPUT_SIZE],
    const weight_t conv1_w[CONV1_WEIGHT_COUNT],
    const weight_t conv2_w[CONV2_WEIGHT_COUNT],
    const weight_t fc1_w[FC1_OUT / 8][8 * FC1_IN],
    const weight_t fc2_w[FC2_OUT / 12][12 * FC2_IN],
    const weight_t fc3_w[FC3_OUT / 5][5 * FC3_IN],
    output_t logits[NUM_CLASSES]);
```

## 4. 文件组织

```text
m6/
├── README.md
├── docs/
│   └── member6_buffer_controller.md
├── hls/
│   ├── address_gen.h
│   ├── buffer.h
│   ├── controller.h
│   ├── lenet_accelerator.h
│   └── lenet_accelerator.cpp
├── tests/
│   ├── tb_address_buffer.cpp
│   └── tb_lenet_accelerator.cpp
└── results/
    ├── reports/
    │   ├── float/
    │   └── fixed/
    ├── csim/
    └── rtl/
        ├── float/
        └── fixed/
```



## 5. 源代码职责

| 文件 | 内容 |
|---|---|
| `hls/address_gen.h` | CHW、Conv OIHW、FC 地址函数封装 |
| `hls/buffer.h` | 线性 Buffer 读、写、Copy 和 CHW 坐标访问 |
| `hls/controller.h` | 顺序层间 Controller |
| `hls/lenet_accelerator.h` | 顶层函数声明 |
| `hls/lenet_accelerator.cpp` | Buffer 分配和 Controller 调用 |
| `tests/tb_address_buffer.cpp` | 地址和 Buffer 单元测试 |
| `tests/tb_lenet_accelerator.cpp` | 五个样本的完整网络测试 |

## 6. 地址规则和 Buffer

统一地址规则为：

```text
CHW：
addr(c,h,w) = c*H*W + h*W + w

OIHW：
addr(oc,ic,kh,kw) = (((oc*Cin)+ic)*KH+kh)*KW+kw

FC：
addr(out,in) = out*IN + in
```

层间 Buffer 使用独立数组：

| Buffer | 元素数 | 用途 |
|---|---:|---|
| `input_buffer` | 784 | 输入图像 |
| `conv1_buffer` | 3456 | Conv1 输出 |
| `relu1_buffer` | 3456 | ReLU1 输出 |
| `pool1_buffer` | 864 | Pool1 输出 |
| `conv2_buffer` | 1024 | Conv2 输出 |
| `relu2_buffer` | 1024 | ReLU2 输出 |
| `pool2_buffer` | 256 | Pool2 输出 |
| `flatten_buffer` | 256 | Flatten 输出 |
| `fc1_buffer` | 120 | FC1 输出 |
| `relu3_buffer` | 120 | ReLU3 输出 |
| `fc2_buffer` | 84 | FC2 输出 |
| `relu4_buffer` | 84 | ReLU4 输出 |
| `output_buffer` | 10 | FC3 输出 |

`buffer.h` 中的模板函数会在整网 HLS 综合时被内联。综合报告的 Memory 部分包含这些 Buffer 的资源映射。

## 7. Controller 顺序

```text
M6_LOAD_INPUT
 -> M6_CONV1
 -> M6_RELU1
 -> M6_POOL1
 -> M6_CONV2
 -> M6_RELU2
 -> M6_POOL2
 -> M6_FLATTEN
 -> M6_FC1
 -> M6_RELU3
 -> M6_FC2
 -> M6_RELU4
 -> M6_FC3
 -> M6_WRITE_OUTPUT
 -> M6_DONE
```

上一状态调用的函数完全结束后，Controller 才进入下一状态。FC 使用成员五的 tiled 接口：

```text
FC1：N_TILE = 8
FC2：N_TILE = 12
FC3：N_TILE = 5
```

顶层不执行 argmax，只输出 `logits[10]`；预测类别由 Testbench 中的 `argmax10` 计算。

## 8. 工程配置

- Top Function：lenet_accelerator_with_weights
- Part：xc7z020clg400-1
- Clock：10 ns
- source files：
  - ./m4/hls/conv2d_systolic.cpp
  - ./m6/hls/lenet_accelerator.cpp
- testbench files
  - ./m6/tests/tb_lenet_accelerator.cpp
  - ./m6/tests/tb_address_buffer.cpp
- CFLAGS：
  - float:`-std=c++11`
  - fixed:`-std=c++11 -DLENET_USE_FIXED -DLENET_ACC_INT`


## 9. 验证结果

### Address / Buffer

Float 和 Fixed 均输出：

```text
MEMBER6 ADDRESS/BUFFER PASS
CSim done with 0 errors
```

测试覆盖：

```text
CHW 地址
OIHW 地址
FC 地址
CHW Buffer 写入/读回
Buffer copy
```

### Float Accelerator

```text
sample 0: max_err=2.55442e-006 argmax=7 ref=7
sample 1: max_err=3.20675e-006 argmax=2 ref=2
sample 2: max_err=2.39389e-006 argmax=1 ref=1
sample 3: max_err=3.5786e-006 argmax=0 ref=0
sample 4: max_err=3.19344e-006 argmax=4 ref=4
[Member6 Float] max abs error = 3.5786e-006
MEMBER6 FLOAT PASS
CSim done with 0 errors
```

### Fixed Accelerator

```text
sample 0: raw_mismatch=0 argmax=7 ref=7
sample 1: raw_mismatch=0 argmax=2 ref=2
sample 2: raw_mismatch=0 argmax=1 ref=1
sample 3: raw_mismatch=0 argmax=0 ref=0
sample 4: raw_mismatch=0 argmax=4 ref=4
[Member6 Fixed] raw mismatch = 0
MEMBER6 FIXED PASS
CSim done with 0 errors
```

五个样本的预测类别均为：

```text
7, 2, 1, 0, 4
```

## 11. 综合结果

两份报告均为顶层 `lenet_accelerator_with_weights` 的 Vivado HLS 2018.3 报告，包含：

```text
工程和器件信息
Performance Estimates
Timing
Latency
Utilization Estimates
BRAM / DSP / FF / LUT
顶层 Interface
```

| 模式 | Estimated clock | Latency | BRAM_18K | DSP48E | FF | LUT |
|---|---:|---:|---:|---:|---:|---:|
| Float | 9.325 ns | 459706 cycles | 30 | 139 | 71471 | 45272 |
| Fixed | 8.702 ns | 474826 cycles | 14 | 146 | 33932 | 17279 |

相对于 10 ns 目标时钟，两种模式的综合估计时钟均小于目标周期。

## 12. 结果文件

### 综合报告

```text
results/reports/float/lenet_accelerator_with_weights_csynth.rpt
results/reports/fixed/lenet_accelerator_with_weights_csynth.rpt
```

### C Simulation 日志

```text
results/csim/accelerator_float_csim.log
results/csim/accelerator_fixed_csim.log
results/csim/address_buffer_float_csim.log
results/csim/address_buffer_fixed_csim.log
```

### 顶层 RTL

```text
results/rtl/float/lenet_accelerator_with_weights.v
results/rtl/float/lenet_accelerator_with_weights.vhd
results/rtl/fixed/lenet_accelerator_with_weights.v
results/rtl/fixed/lenet_accelerator_with_weights.vhd
```

## 13. 验收状态

- [x] 已读取公共网络尺寸、数据类型和接口约定。
- [x] 已调用成员四 Conv1 / Conv2。
- [x] 已调用成员五 ReLU、MaxPool、Flatten 和 FC。
- [x] 已实现层间 Buffer。
- [x] 已验证 CHW、OIHW、FC 地址规则。
- [x] Float Address/Buffer 测试通过。
- [x] Fixed Address/Buffer 测试通过。
- [x] Float Accelerator C Simulation 通过。
- [x] Fixed Accelerator C Simulation 通过。
- [x] Float C Synthesis 完成。
- [x] Fixed C Synthesis 完成。
- [x] Float Verilog/VHDL RTL 已生成。
- [x] Fixed Verilog/VHDL RTL 已生成。

## 14. 当前限制

- 权重通过顶层端口显式输入；
- 尚未固化为内部 Weight ROM；
- 尚未连接 AXI 或外部存储器；
- 中间 Feature Map 使用独立 Buffer，没有做存储复用；
- 没有 DATAFLOW、Ping-Pong Buffer 和层间重叠；
- `address_gen.h` 是公共地址规则的 M6 封装和单元测试入口，成员四和成员五内部继续使用各自已验证的访问实现；
- 尚未完成主 Vivado 的 RTL/IP、时钟复位和 Zynq Block Design 集成。

详细的设计解释见 [`docs/member6_buffer_controller.md`](docs/member6_buffer_controller.md)。
