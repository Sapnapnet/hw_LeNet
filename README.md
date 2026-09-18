> 2026-09-14更新：成员8已接手统一预处理，当前190张及增量入口见[现行工作流](docs/member8_level2_workflow.md)。以下原接收说明保留历史范围。

# LeNet HLS 集成工程（成员 1）

这是从 `FPGA_LeNet` 整理出的独立成员 1 总工程。原始目录保持不变；本目录是唯一建议继续开发、运行和提交的工程副本。

## 目录

```text
config/                 冻结网络、公共类型与 W12/A12 参数
docs/                   接口、成员交接与验证说明
python/                 Float Baseline、权重/参考导出、定点核心
weights/                Float/Fixed HLS 权重
reference/              Float/Fixed 五个参考样本
hls/conv/               Conv/GEMM systolic 核
hls/operators/          ReLU、Pool、Flatten、FC、Argmax
hls/buffer/             地址生成与线性 Buffer
hls/top/                Controller 与整网顶层
tests/                  地址/Buffer 与 MNIST 整网 Testbench
scripts/                便携 CSim、Vivado HLS CSim 与结果比较脚本
data/                   MNIST 及 Level 2 自采数据
results/                已有准确率、逐层对齐、CSim 与综合历史结果
```

## 冻结接口

顶层函数为 `lenet_accelerator_with_weights`，输出 `logits[10]`；预测类别由 Testbench 的 `argmax10` 计算。Feature Map 使用 CHW，Conv 权重使用 OIHW，FC 权重使用 `[OUT][IN]`，所有层无 Bias。

Float 编译使用默认类型；Fixed 编译启用：

```text
LENET_USE_FIXED LENET_ACC_INT
```

目标器件为 `xc7z020clg400-1`，目标时钟 10 ns。

## 运行入口（本次未执行）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_full_mnist.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hls_csim.ps1 -Mode both
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hls_synth.ps1 -Mode both
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_unit_tests.ps1
```

本次任务只完成目录整理和代码集成，按要求没有替用户运行仿真或综合。

## 来源与限制

- HLS 主线来自 `FPGA_LeNet/src`，并统一改为本工程的 `config/` 与 `hls/` 相对路径。
- 成员 2 的 Python Baseline 与完整 Float reference 来自工作区既有成员 1/2 交付；权重、MNIST、准确率和 HLS 日志来自 `FPGA_LeNet`。
- 成员 3 的 `fixed.py`、量化配置和 Fixed reference 来自现有 handoff；原始交付缺少完整 PTQ `run.py`，因此这里保留核心定点实现和已生成资产，不虚构新的实验结果。
- `results/synthesis/` 和 `results/layerwise/` 保存已有历史报告，不能替代用户后续在本工程中的重新运行。
