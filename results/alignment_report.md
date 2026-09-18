# 成员七：Python vs HLS 逐层对齐报告

负责人：成员七  
日期：2026-09-08  
状态：**Float + Fixed 逐层对齐均完成（Windows 实测，raw mismatch = 0）**

---

## 1. 对齐对象与方法

```text
Python Float Reference (成员2/5)
        ↓ 逐层比较
HLS Float 整网逐层输出（成员4/5 模块 + 成员6 调用顺序）

Python Fixed Raw Reference (成员3/5)
        ↓ 逐位比较（raw mismatch 必须为 0）
HLS Fixed 整网逐层输出
```

- 观测方法：`member7/tests/tb_layerwise.cpp` 按成员 6 `controller.h` 的
  状态顺序（LOAD_INPUT → Conv1 → ReLU1 → Pool1 → … → FC3）逐个调用同一批
  HLS 模块，每层输出 Dump 到 `member7/results/layerwise/`，并与参考
  数据当场比较。
- 参考来源：
  - `input / conv1 / pool1 / conv2`：成员 2 浮点参考（`m2/`）、成员 3
    定点 raw 参考（`m3/`）；
  - `relu1 / relu2 / pool2 / flatten / fc1 / relu3 / fc2 / relu4 /
    logits`：成员 5 `m5/data/`（与成员 2/3 同一条导出流水线，
    Fixed raw 位精确，见 `m5/data/PROVENANCE.md`）。
- 判据：
  - Float：mismatch（|diff| > 1e-5）为 0，且全层最大绝对误差 ≤ 1e-3；
  - Fixed：raw 整数逐位相等，**mismatch 必须为 0**（成员 3 规则，
    不把量化误差当作 HLS 可容忍误差）。
- 样本：冻结的 5 个 Reference Sample（7, 2, 1, 0, 4），全 13 层。

## 2. 结论（先说重点）

1. **HLS 与 Python 参考在全部 13 层、5 个样本上逐层对齐通过。**
2. Float 模式全层 mismatch = 0，最坏层 `fc1` 最大绝对误差
   `5.11e-6`、全层平均误差 ≤ `1.4e-6`，与参考模型浮点累加顺序差异一致，
   无功能错误。
3. Fixed 逐层（Windows 实测，Vivado HLS 2018.3 include）：全部 13 层 ×
   5 个样本 **raw mismatch = 0**（逐位相等），argmax 全部正确，三条
   独立路径复核一致（见第 5 节）；Fixed 整网与成员 6 CSim 日志结论相同。
4. 未发现需要成员 4/5/6 返工的功能问题；**建议成员 1 可以按计划冻结
   中期版本**（冻结范围按接口文档执行，定点配置 `LENET_USE_FIXED +
   LENET_ACC_INT`）。

## 3. 整网结果（成员 6 CSim 日志，复验通过）

| 模式 | 样本 | 结果 |
|---|---|---|
| Float | 5 | max abs err = 3.58e-6（m6 日志）/ 3.58e-6（本机复跑），argmax 7,2,1,0,4 全对 |
| Fixed | 5 | raw mismatch = 0，argmax 7,2,1,0,4 全对 |

本机（Windows 11，MSYS2 ucrt64 g++ + Vivado HLS 2018.3 include）已复跑
Float/Fixed 全部测试并复现通过。

## 4. Float 逐层对齐结果（实测）

工具：`member7/tests/tb_layerwise.cpp` + `member7/scripts/compare.py`，
5 样本聚合（完整逐样本数据：
`member7/results/tables/float_layerwise.csv`）。

| layer | mismatch | max abs err | mean abs err |
|---|---:|---:|---:|
| input | 0 | 4.99794e-10 | 4.18824e-11 |
| conv1 | 0 | 4.81455e-07 | 1.40007e-08 |
| relu1 | 0 | 4.81455e-07 | 1.02722e-08 |
| pool1 | 0 | 4.78059e-07 | 1.44663e-08 |
| conv2 | 0 | 3.84763e-06 | 3.06257e-07 |
| relu2 | 0 | 2.86451e-06 | 1.47344e-07 |
| pool2 | 0 | 2.86451e-06 | 2.37306e-07 |
| flatten | 0 | 2.86451e-06 | 2.37306e-07 |
| fc1 | 0 | 5.10608e-06 | 7.10144e-07 |
| relu3 | 0 | 5.10608e-06 | 3.71981e-07 |
| fc2 | 0 | 3.82373e-06 | 7.84035e-07 |
| relu4 | 0 | 3.82373e-06 | 4.89022e-07 |
| logits | 0 | 3.5786e-06 | 1.37542e-06 |

误差来源分析：Float 模式下误差量级（~1e-6）与 Python 参考自身的浮点
计算顺序差异相符——Conv 在 HLS 侧为脉动阵列累加、Python 侧为常规
逐元素累加，二者舍入次序不同；该量级远小于 1 LSB 定点量化步长
（1/32 ≈ 0.031），不影响分类。

## 5. Fixed 逐层对齐结果（Windows 实测）

环境：Windows 11 + Git Bash（MSYS2 ucrt64 g++ 15.1），Vivado HLS 2018.3
include（`F:\Vivado\Vivado\2018.3`，脚本自动探测）。执行
`member7/tests/run_tests.sh`。

### 5.1 raw 逐位对齐（判据：mismatch = 0）

全部 13 层 × 5 样本 **mismatch = 0、max raw diff = 0**，三条独立路径
一致：

1. `tb_layerwise.cpp` 内置逐层比较：`MEMBER7 LAYERWISE PASS`；
2. `compare.py --mode raw` 汇总（下表）；
3. `cmp` 对 65 个 dump 与 m3/m5 参考逐文件二进制比对，完全一致。

| layer | mismatch | max raw diff | mean raw diff |
|---|---:|---:|---:|
| input | 0 | 0 | 0 |
| conv1 | 0 | 0 | 0 |
| relu1 | 0 | 0 | 0 |
| pool1 | 0 | 0 | 0 |
| conv2 | 0 | 0 | 0 |
| relu2 | 0 | 0 | 0 |
| pool2 | 0 | 0 | 0 |
| flatten | 0 | 0 | 0 |
| fc1 | 0 | 0 | 0 |
| relu3 | 0 | 0 | 0 |
| fc2 | 0 | 0 | 0 |
| relu4 | 0 | 0 | 0 |
| logits | 0 | 0 | 0 |

argmax：5 样本 hls 与参考一致（7, 2, 1, 0, 4）。逐样本数据：
`results/tables/fixed_layerwise_raw.csv`。

### 5.2 反量化 vs Float 参考（量化误差，供成员 8 图表）

Fixed raw × 1/32 与 Float 参考逐层比较；mismatch 列为 |err| > 1e-5 的
元素数（量化误差本身即预期，不参与判据）：

| layer | mismatch | max abs err | mean abs err |
|---|---:|---:|---:|
| input | 650 | 0.0155637 | 0.00145052 |
| conv1 | 9338 | 0.0582417 | 0.00512339 |
| relu1 | 6111 | 0.049603 | 0.0033713 |
| pool1 | 2061 | 0.0416167 | 0.00454527 |
| conv2 | 5118 | 0.0959098 | 0.0159098 |
| relu2 | 2477 | 0.0878606 | 0.00753265 |
| pool2 | 919 | 0.0775452 | 0.0119339 |
| flatten | 919 | 0.0775452 | 0.0119339 |
| fc1 | 600 | 0.101067 | 0.0233726 |
| relu3 | 313 | 0.101067 | 0.0127927 |
| fc2 | 419 | 0.114189 | 0.0292226 |
| relu4 | 238 | 0.114189 | 0.017768 |
| logits | 50 | 0.130909 | 0.0501123 |

首层 max err 0.0156 ≈ 1/64，符合 Q5 量化步长 1/32 的舍入界；误差随层
累积，logits 最大 0.131、平均 0.050，量级与成员 3 量化分析一致。

### 5.3 首次运行问题记录（已修复）

首次在 Windows 上运行本脚本时 fixed 相关检查全部失败，根因有二：

1. **运行时段错误（主因）**：测试二进制由 MSYS2 ucrt64 g++ 编译，运行
   时却从 PATH 加载了 Git Bash 自带 mingw64 的 `libstdc++-6.dll`，ABI
   不匹配，`valid_root()` 内 std::string 操作段错误（exit 139），测试
   台未产出任何 dump。
2. **结果误导（次因）**：`compare.py` 在 dump 全部缺失时仍写出全 0 的
   汇总表（看起来像 PASS）——提交 `fixed results` 中的两张表即此产物，
   CSV 仅表头无数据行，属无效结果，已重做。

修复：

- `run_tests.sh`：MSYS/MinGW 下所有 g++ 编译加
  `-static-libstdc++ -static-libgcc`（仅 Windows 生效，不影响
  macOS/Linux/WSL）；
- `compare.py`：dump 缺失时拒绝写出表格；quant 模式改为信息性
  （量化误差不再判失败，位精确性已由 raw 模式保证）。

另：m4 fixed 测试的 generated-vector 阶段缺失 `test_vectors/` 资产
（从未入库，任何机器上 m4 fixed 都会在此失败），已由成员七用独立
Python 参考重建（`member6-handoff-v1/test_vectors/`，含 `generate.py`
与 `PROVENANCE.md`；三组向量在两种 fixed 构建下 mism=0）。成员四可
复核或替换为原始向量。

## 6. 回归测试入口

```bash
member7/tests/run_tests.sh   # 全部模块 Float+Fixed（Fixed 自动检测 ap_fixed.h）
```

覆盖：m4 Conv（float/fixed/legacy）、m5 算子（float/fixed/legacy）、
m6 地址与整网（float/fixed）、m7 逐层（float/fixed）、compare.py
三种模式交叉核对。任一失败退出码非 0。

本机 Windows 全量回归 15/15 通过（2026-09-08）。

## 7. 已知限制

1. 目前只有 5 个冻结样本的逐层参考；Level 1 的全测试集准确率验证
   需要成员 2/3 批量导出参考或直接喂 MNIST 原始数据（中期后工作）。
2. 逐层观测通过成员 7 测试平台镜像 Controller 调用顺序实现，顶层
   `lenet_accelerator_with_weights` 本身不导出中间层（接口文档允许）。
3. Fixed 逐层复验已于 Windows 完成（第 5 节）。m4 的 generated 测试
   向量由成员七重建（见 `member6-handoff-v1/test_vectors/PROVENANCE.md`），
   成员四需复核或替换为原始向量后再定稿。
