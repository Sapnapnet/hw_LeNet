# Level 2 HLS 预处理验证记录

运行日期：2026-09-14

## 完成项

- HLS C++ 单元测试：Vivado HLS 2018.3 CSim 通过（0 errors）。
- HLS Full 核：在 `xc7z020clg400-1`、10 ns 时钟约束下完成 C synthesis 并生成 RTL。
- 批处理：55/55 样本已生成 Simple 与 Full 输出各 55 张，以及 A12 raw 输入各 55 份。
- 交付校验：110 个 PGM 均为 P5、28x28、单通道（797 bytes）；110 份 Fixed 文本均为 784 行，所有值在 `[0,32]`。

## HLS 报告口径

`level2_full_preprocess_rgb_csynth.rpt` 是 HLS Full 核的原始 C synthesis 报告。当前估计时钟为 9.562 ns；由于该解决方案仍保留 HLS 默认的 1.25 ns 时钟不确定度，日志会给出其超过 8.75 ns effective budget 的 warning。该预处理核用于 Level 2 离线批处理与实现可综合性验证，不纳入已冻结的 LeNet `optimized` accelerator latency/资源结果，不能混写入其性能表。

三次 RGB 外存扫描（统计阈值、ROI、缩放采样）替代整帧片上缓存，因此不需要输入图像 BRAM；报告中的少量 DSP 来自动态 ROI 地址/缩放计算，而非灰度转换。

## 证据文件

- `hls_csim_and_synth.log`：CSim 成功与 C synthesis 日志。
- `level2_full_preprocess_rgb_csynth.rpt`：Full 核 C synthesis 原始报告。
