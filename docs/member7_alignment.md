# 成员七 Testbench / Python-HLS 对齐交付

版本：`member7-alignment-v1`；日期：2026-09-08

## 交付内容

| 路径 | 说明 |
|---|---|
| `tests/tb_layerwise.cpp` | 逐层对齐测试平台（Float + Fixed 双模式），按成员 6 Controller 顺序逐层调用成员 4/5 模块，每层 Dump 并与 Python 参考当场比较 |
| `tests/run_tests.sh` | 全项目回归入口：m4/m5/m6/m7 全部测试平台 + compare.py 交叉核对 |
| `scripts/compare.py` | 逐层 txt 对比工具：`float`（误差统计）/ `raw`（位精确比对）/ `quant`（反量化 vs Float）三种模式，输出表格与 CSV |
| `results/layerwise/` | 逐层 Dump（float + fixed 均已生成，各 5 样本 × 13 层） |
| `results/tables/` | 逐层误差表格（md + csv） |
| `alignment_report.md` | 对齐结论与完整数据（本报告） |

## 编译与运行

```bash
# 单跑逐层对齐（Float，任何平台）
cd member7/tests
g++ -std=c++11 -O2 tb_layerwise.cpp \
    ../../member6-handoff-v1/m4/hls/conv2d_systolic.cpp \
    -o /tmp/tb_layerwise_float
M7_ROOT=<repo root> /tmp/tb_layerwise_float

# Fixed（需要 Vivado HLS 的 ap_fixed.h）
g++ -std=c++11 -O2 -DLENET_USE_FIXED -DLENET_ACC_INT -I<HLS_INC> \
    tb_layerwise.cpp \
    ../../member6-handoff-v1/m4/hls/conv2d_systolic.cpp \
    -o /tmp/tb_layerwise_fixed
M7_ROOT=<repo root> /tmp/tb_layerwise_fixed

# 全量回归（自动跳过 Fixed 当 ap_fixed.h 不存在）
cd member7/tests
HLS_INC=/d/Xilinx/Vivado/2018.3/include ./run_tests.sh
```

### Windows 说明

- 在 **Git Bash**（或 WSL）中运行 `run_tests.sh`，需要 g++ 在 PATH
  中（MinGW 或 WSL 内 g++）。
- **克隆路径必须是纯 ASCII**（例如 `D:\FPGA_LeNet`，不要含中文
  目录名）——Vivado HLS 工具和 MinGW 都不支持非 ASCII 路径。
- Fixed 模式只需要 Vivado 安装目录里的 `include/ap_fixed.h`，
  不需要启动 Vivado 本身。脚本自动探测常见安装路径（含本机
  `F:\Vivado\Vivado\2018.3\include`，Git Bash 里写作
  `/f/Vivado/Vivado/2018.3/include`）；若安装在其他位置，用
  `HLS_INC=<你的 include 路径> ./run_tests.sh` 显式指定。
- **Git Bash / MSYS2 下必须静态链接 C++ 运行时**：`run_tests.sh` 已
  自动给所有编译加 `-static-libstdc++ -static-libgcc`。若不这样做，
  MSYS2 ucrt64 g++ 编译的二进制会在运行时加载 Git 自带 mingw64 的
  `libstdc++-6.dll`（PATH 顺序靠前），ABI 不匹配直接段错误——本机
  m7 fixed 测试台首跑即因此死在 `valid_root()`。手动编译时请自行
  加上这两个 flag。

## 判据

- Float：逐层 mismatch（|diff| > 1e-5）= 0，全层最大绝对误差 ≤ 1e-3；
- Fixed：raw 整数逐位相等（mismatch = 0，成员 3 规则）；quant
  模式（反量化 vs Float）为信息性输出，量化误差不判失败；
- 每层参考来源：`input/conv1/pool1/conv2` 用 m2/m3 官方导出，其余
  `m5/data/`（位精确，见 PROVENANCE.md）。

## 当前状态

- [x] Float 逐层对齐 13 层 × 5 样本：mismatch = 0，最坏层 conv2
      max err 5.75e-6（本机实测，见 `alignment_report.md`）
- [x] 整网 Float/Fixed 复验：与成员 6 CSim 日志一致（Fixed 整网
      raw mismatch = 0）
- [x] 回归脚本：Windows 全量回归 15/15 通过（含 Fixed 全部 9 项）
- [x] Fixed 逐层 raw 对齐：13 层 × 5 样本 mismatch = 0，三条独立
      路径复核一致，已填入 `alignment_report.md` 第 5 节
- [ ] 全测试集准确率（Level 1 完整指标）：中期后配合成员 2/3 批量
      参考完成

## 提供给谁

- 成员 1：是否可以冻结中期版本的验证结论；
- 成员 4/5/6：错误定位手段（逐层 Dump 定位到具体层）；
- 成员 8：逐层误差表、Float/Fixed 量化误差数据（PPT 用）。
