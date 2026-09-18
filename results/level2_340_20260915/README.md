# Member 7 Level 2：340 条自采数据测试交付

## 1. 数据集与测试范围

本目录对应的数据版本为 `level2_340_20260915`。

- 总样本数：340 条
- 每个数字类别：34 条
- 每组测试的准确率分母：340
- 每类准确率分母：34
- 测试组数：6 组
- 合并逐样本记录数：2040 条

所有输入均按照唯一主清单
`data/self_collected/level2_manifest.csv` 读取，不扫描历史批次来决定样本数量。

Simple/Full 的 Float 测试读取清单中指定的 PGM 图像，并使用
`pixel / 255.0` 作为浮点输入。Python Fixed 和 optimized HLS 测试直接读取清单中指定的 Fixed 文本，不重新量化 PGM；Fixed 输入是 A12 signed I7/F5 原始整数编码，scale 为 `1/32`。

本目录中的 `optimized_hls` 表示使用
`optimization/lenet_fast.cpp`、`LENET_USE_FIXED` 和 `LENET_ACC_INT`
执行的 g++ C simulation，不是 Vivado HLS CSim，也不是 RTL 联合仿真。

## 2. 逐样本结果文件

每个逐样本结果文件均应包含 340 条数据记录，且顺序与
`level2_manifest.csv` 完全一致。

| 文件 | 对应交付产物 | 说明 |
|---|---|---|
| `results_simple_float.csv` | Simple + Python Float 逐样本结果 | 使用 Simple PGM，Float 网络推理；包含 `logit0` 至 `logit9`。 |
| `results_simple_fixed.csv` | Simple + Python Fixed 逐样本结果 | 使用 Simple Fixed raw 输入，Python 整数定点网络推理；包含 `raw_logit0` 至 `raw_logit9`。 |
| `results_simple_optimized_hls.csv` | Simple + optimized HLS 逐样本结果 | 使用 Simple Fixed raw 输入，optimized g++ C simulation；包含 HLS raw logits。 |
| `results_full_float.csv` | Full + Python Float 逐样本结果 | 使用 Full PGM，Float 网络推理；包含 `logit0` 至 `logit9`。 |
| `results_full_fixed.csv` | Full + Python Fixed 逐样本结果 | 使用 Full Fixed raw 输入，Python 整数定点网络推理；包含 `raw_logit0` 至 `raw_logit9`。 |
| `results_full_optimized_hls.csv` | Full + optimized HLS 逐样本结果 | 使用 Full Fixed raw 输入，optimized g++ C simulation；包含 HLS raw logits。 |
| `results_all_2040.csv` | 六组结果合并交付文件 | 将上述 6 组结果合并为 2040 条记录，便于统一筛选、统计和后续整理。 |

逐样本结果中的主要字段含义：

- `sample_id`：自采样本唯一编号。
- `label`：真实数字类别。
- `preprocess`：预处理方式，`simple` 或 `full`。
- `inference_mode`：推理方式，`float`、`fixed` 或 `optimized_hls`。
- `prediction`：模型预测类别。
- `correct`：预测是否正确，`1` 表示正确，`0` 表示错误。
- `logit0` 至 `logit9`：Float logits。
- `raw_logit0` 至 `raw_logit9`：Fixed/HLS 原始定点 logits，scale 为 `1/32`。

## 3. 统计与混淆矩阵

| 文件 | 对应交付产物 | 说明 |
|---|---|---|
| `summary.csv` | 六组总体及逐类统计表 | 每组包含总样本数、正确数、总体准确率，以及 0 至 9 每类的样本数、正确数和准确率。 |
| `class_stats.csv` | 六组逐类别统计明细 | 每组 10 行，共 60 行；每行对应一个真实数字类别，分母固定为 34。 |
| `confusion_simple_float.csv` | Simple Float 混淆矩阵 | 行表示真实类别，列表示预测类别。 |
| `confusion_simple_fixed.csv` | Simple Fixed 混淆矩阵 | 行表示真实类别，列表示预测类别。 |
| `confusion_simple_optimized_hls.csv` | Simple optimized HLS 混淆矩阵 | 行表示真实类别，列表示预测类别。 |
| `confusion_full_float.csv` | Full Float 混淆矩阵 | 行表示真实类别，列表示预测类别。 |
| `confusion_full_fixed.csv` | Full Fixed 混淆矩阵 | 行表示真实类别，列表示预测类别。 |
| `confusion_full_optimized_hls.csv` | Full optimized HLS 混淆矩阵 | 行表示真实类别，列表示预测类别。 |

每份混淆矩阵均为 10×10，矩阵元素总和应为 340。

## 4. Fixed 与 optimized HLS 一致性

| 文件 | 对应交付产物 | 说明 |
|---|---|---|
| `fixed_vs_optimized_hls_consistency.csv` | Fixed 与 optimized HLS 一致性汇总 | 分别比较 Simple 和 Full 的 340×10 raw logits，记录不一致样本数、不一致元素数和最大绝对差。当前两种预处理均为零差异。 |
| `fixed_vs_optimized_hls_mismatches.csv` | Fixed 与 optimized HLS 不一致样本明细 | 仅在发现差异时列出样本 ID、logit 下标及两侧具体 raw 值；当前文件只有表头，没有不一致记录。 |

当前一致性结果：

| 预处理方式 | 比较样本数 | 比较 raw 值数 | 不一致样本数 | 不一致值数 | 最大绝对差 |
|---|---:|---:|---:|---:|---:|
| Simple | 340 | 3400 | 0 | 0 | 0 |
| Full | 340 | 3400 | 0 | 0 | 0 |

## 5. optimized HLS 原始输出

| 文件 | 对应交付产物 | 说明 |
|---|---|---|
| `optimized_hls_raw_simple.csv` | Simple optimized HLS 原始输出留档 | optimized HLS testbench 直接生成的 340 条 raw logits，字段使用 `sample_id`、`label`、`pred` 和 `raw_logit0` 至 `raw_logit9`。 |
| `optimized_hls_raw_full.csv` | Full optimized HLS 原始输出留档 | optimized HLS testbench 直接生成的 340 条 raw logits，字段格式同上。 |

这两个文件用于保留 HLS 驱动程序的原始输出；整理后的正式逐样本交付结果分别见
`results_simple_optimized_hls.csv` 和 `results_full_optimized_hls.csv`。

## 6. 运行说明、日志与 provenance

| 文件 | 对应交付产物 | 说明 |
|---|---|---|
| `run.log` | 实际运行日志 | 保存输入校验命令、g++ 编译命令、Simple/Full optimized HLS 执行命令及进度输出；失败时应保留失败信息。 |
| `run_manifest.json` | 运行 provenance 和文件哈希清单 | 记录数据版本、主清单 SHA256、Python/NumPy/操作系统、编译器、HLS include 路径、权重 SHA256、定点配置、源文件 SHA256，以及本目录各交付文件的字节数和 SHA256。 |
| `README.md` | 本目录交付说明 | 说明测试范围、输入规则、推理模式，以及目录内每个文件对应的交付产物。 |

原始 PGM、Fixed 文本和原图不复制到本结果目录，统一保存在
`data/self_collected`，由 `level2_manifest.csv` 按相对路径引用。重复图像、困难样本和质量问题样本均保留，不因识别结果删除或筛选。
