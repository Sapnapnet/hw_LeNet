# 成员四 Conv / MAC 交付说明

版本：`member4-conv-mac-final-v1`；日期：2026-09-06。

## 交付内容

本目录是成员四关于 Conv / MAC 的完整交付：

- `hls/`：Vitis HLS 源码（C++，输出驻留 GEMM 型脉动阵列）
- `tests/`：本地 g++ 测试脚本与测试程序
- `results/reports/baseline/`：优化前（legacy ap_fixed）综合报告
- `results/reports/final/`：优化后（`LENET_ACC_INT` 整数累加）综合报告

## 快速使用

在 WSL 中执行：

```bash
cd tests
./run_tests.sh
```

预期输出包含：

```text
SYSTOLIC FLOAT PASS
SYSTOLIC FIXED PASS
```

## 顶层接口

```cpp
void conv1_systolic(
    const data_t in[INPUT_SIZE],          // [1][28][28] CHW
    const weight_t w[CONV1_WEIGHT_COUNT], // [6][1][5][5] OIHW
    data_t out[CONV1_OUT_SIZE]);          // [6][24][24] CHW

void conv2_systolic(
    const data_t in[POOL1_OUT_SIZE],      // [6][12][12] CHW
    const weight_t w[CONV2_WEIGHT_COUNT], // [16][6][5][5] OIHW
    data_t out[CONV2_OUT_SIZE]);          // [16][8][8] CHW
```

## 对 README 约定的偏离

### 1. 顶层函数命名

原始 README 建议沿用成员一的 `conv2d` 命名。本交付使用 `conv1_systolic` / `conv2_systolic`。

- **价值**：两个卷积形状不同，独立命名可分别作为 HLS top 综合，避免一个 `conv2d` 函数因运行时参数导致无法 `ARRAY_PARTITION`；成员6集成时也可直接按层调用。
- 如果成员1仍要求统一 `conv2d`，只需在 `conv2d_systolic.cpp` 中做一层薄改名，内部实现不变。

### 2. 数据类型

原始 README 要求“不要自建另一套类型”。本交付在 `LENET_ACC_INT` 优化路径中引入了成员4内部类型：

```cpp
typedef ap_int<33> systolic_acc_t;  // 仅存在于 gemm_systolic.h / conv2d_gemm.h
```

- **价值**：共享 `types.h` 保持原样；默认路径仍使用公共 `acc_t`。只有成员4在 `LENET_ACC_INT` 下使用纯整数累加，消除每拍 `ap_fixed` 舍入/饱和逻辑，LUT 降低约 55%。
- 公共接口的 `data_t / weight_t / input / output` 全部未变。

### 3. 实现架构

原始 README 建议先做“通用 MAC/Conv 正确版”。本交付直接实现为 **输出驻留（Output-Stationary）GEMM / PE 阵列**，而不是标量 MAC。

- **价值**：Conv 与 FC 共用同一个 `gemm_systolic` 模板；DSP 利用率和吞吐显著高于标量 MAC；Conv1/Conv2 合计延迟约 0.30 ms。

## 综合结果摘要

| 模块 | 配置 | Latency | DSP | LUT | FF |
|---|---:|---:|---:|---:|---:|
| Conv1 | 8×6 PE | 17,426 cyc / 0.174 ms | 49 | final 5,340 | 9,221 |
| Conv2 | 8×8 PE | 12,226 cyc / 0.122 ms | 72 | final 8,822 | 13,556 |

详细报告见 `results/reports/baseline/` 与 `results/reports/final/`。

## 提供给谁

- 成员1：源码、接口、综合报告
- 成员6：可调用的稳定 Conv 模块
- 成员7：Testbench 与逐层对齐
- 成员8：基线/最终资源与延迟数据
- 成员5：可复用 `gemm_systolic.h` 模板实现 FC（见 `m5/README.md`）
