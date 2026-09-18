# 成员4卷积实现与综合结果汇总

整理日期：2026-09-07。成员4申报版本：`member4-conv-mac-final-v1`（交付日期2026-09-06，无Git Commit）。范围：Conv1、Conv2模块，供成员8中期整理与展示。

## 1. 采用口径与来源

**本次比较初始朴素卷积与最终版，两者结构不同。**“基线来自最开始实现的朴素卷积，与目前最终结构不同”是成员4补充、由用户在本次会话转述的信息，记录日期2026-09-07；不是从现有代码反推得到的结论。

按用户指定，基线仅使用下列R1、R2；最终版使用R3、R4。成员4说明文档中的旧基线数值及依赖旧基线的下降幅度不采用，变化全部重新计算。其余无明确冲突的交付信息沿用并标明来源。原始报告、文档和代码均未修改。

| ID | 模块与版本 | 原始报告 |
|---|---|---|
| R1 | Conv1，初始朴素卷积 | [baseline/conv1_baseline_csynth.rpt](../../../provided_by_member4/member4_conv_mac/成员4/results/reports/baseline/conv1_baseline_csynth.rpt) |
| R2 | Conv2，初始朴素卷积 | [baseline/conv2_baseline_csynth.rpt](../../../provided_by_member4/member4_conv_mac/成员4/results/reports/baseline/conv2_baseline_csynth.rpt) |
| R3 | Conv1，最终版 | [final/conv1_systolic_csynth.rpt](../../../provided_by_member4/member4_conv_mac/成员4/results/reports/final/conv1_systolic_csynth.rpt) |
| R4 | Conv2，最终版 | [final/conv2_systolic_csynth.rpt](../../../provided_by_member4/member4_conv_mac/成员4/results/reports/final/conv2_systolic_csynth.rpt) |

其他证据：[成员4交付说明](../../../provided_by_member4/member4_conv_mac/成员4/README.md)、[接收与回交记录](../../../provided_by_member4/member4_conv_mac/成员4/ACCEPTANCE.md)、[设计说明](../../../provided_by_member4/member4_conv_mac/成员4/docs/member4_conv_mac.md)。仅使用包内“成员4”目录对应说明，包根目录的成员5说明不作为成员4结果来源。

## 2. 综合指标

四份报告均为Vitis HLS 2020.2，Build 3064766，Vivado IP Flow Target，目标器件`xc7z020-clg400-1`。R1、R2报告日期为2026-09-05，R3、R4为2026-09-06。完整元数据及来源路径见[synthesis_summary.csv](synthesis_summary.csv)。

资源使用顶层Utilization Estimates的Summary中Total行，未误用子实例的Total。

| 来源 | 顶层函数 | LUT | FF | DSP | BRAM_18K |
|---|---|---:|---:|---:|---:|
| R1 | conv1_2d | 6,987 | 1,845 | 25 | 0 |
| R2 | conv2_2d | 1,161 | 291 | 1 | 0 |
| R3 | conv1_systolic | 5,340 | 9,221 | 49 | 0 |
| R4 | conv2_systolic | 8,822 | 13,556 | 72 | 0 |

| 来源 | Latency min/max（cycles） | 报告Latency min/max（ms） | 顶层Interval min/max（cycles） | 目标时钟（ns） | 估计时钟（ns） |
|---|---:|---:|---:|---:|---:|
| R1 | 44,945 / 44,945 | 0.449 / 0.449 | 44,946 / 44,946 | 10.00 | 7.244 |
| R2 | 162,818 / 162,818 | 1.628 / 1.628 | 162,819 / 162,819 | 10.00 | 7.247 |
| R3 | 17,426 / 17,426 | 0.174 / 0.174 | 17,427 / 17,427 | 10.00 | 7.027 |
| R4 | 12,226 / 12,226 | 0.122 / 0.122 | 12,227 / 12,227 | 10.00 | 7.131 |

四组时钟Uncertainty均为2.70 ns，顶层Pipeline Type均为`none`。表中的ms沿用报告舍入值，变化比例以精确cycles计算；不从舍入后的ms反算比例。估计时钟是综合估计，不是上板实测。

成员4回交记录另报告最终版GEMM k-loop的II为Conv1 3、Conv2 4；按其说明保留为**内部循环II**，不填入上述顶层Interval列。所给顶层报告的Loop Detail为N/A，本次未独立复核内部循环II。

## 3. 按指定基线重算的变化

计算口径：

- 增减量 = 最终版 − 基线。
- 变化率 =（最终版 − 基线）÷ 基线 × 100%；负值表示下降，正值表示增加。
- 延迟下降率 =（基线cycles − 最终版cycles）÷ 基线cycles × 100%。
- 延迟比值 = 基线cycles ÷ 最终版cycles；是模块延迟之比，不是整网吞吐率提升倍数。

| 模块 | 指标 | 基线 → 最终版 | 增减量 | 变化 |
|---|---|---:|---:|---:|
| Conv1 | LUT | 6,987 → 5,340 | −1,647 | 下降23.57% |
| Conv1 | FF | 1,845 → 9,221 | +7,376 | 增加399.78% |
| Conv1 | DSP | 25 → 49 | +24 | 增加96.00% |
| Conv1 | BRAM_18K | 0 → 0 | 0 | 数量不变；百分比不适用 |
| Conv1 | Latency（cycles） | 44,945 → 17,426 | −27,519 | 下降61.23% |
| Conv2 | LUT | 1,161 → 8,822 | +7,661 | 增加659.86% |
| Conv2 | FF | 291 → 13,556 | +13,265 | 增加4558.42% |
| Conv2 | DSP | 1 → 72 | +71 | 增加7100.00% |
| Conv2 | BRAM_18K | 0 → 0 | 0 | 数量不变；百分比不适用 |
| Conv2 | Latency（cycles） | 162,818 → 12,226 | −150,592 | 下降92.49% |

Conv1基线/最终版延迟比为 **2.58倍**，Conv2为 **13.32倍**。两层延迟均缩短；Conv1的LUT减少，Conv2的LUT增加，两层FF和DSP均增加。这是当前指定两种实现之间的报告对比，不能沿用旧文案的“LUT降低约55%”。

完整计算值见[comparison_summary.csv](comparison_summary.csv)，CSV保留计算精度，展示文字统一保留两位小数。基线为0的变化百分比留空，并由`percent_status=NOT_APPLICABLE_BASELINE_ZERO`说明；非延迟行的延迟下降率及比值留空，表示不适用。

`source_file`相对Project目录；比较表通过`baseline_source_id`、`final_source_id`关联综合表R1～R4。不得将两层独立综合的资源或延迟简单相加后称为整网综合结果。

## 4. 现有代码支持的实现说明

以下描述的是交付代码中可直接查看的结构和计算步骤，不据此重建各报告的准确源码、编译宏或类型版本。

| 可说明的内容 | 代码依据 |
|---|---|
| Conv1处理8个空间输出位置、6个输出通道；Conv2处理8个空间位置、8个输出通道，16个通道分两组 | [conv2d_systolic.cpp](../../../provided_by_member4/member4_conv_mac/成员4/hls/conv2d_systolic.cpp)中的两个模板实例；[conv2d_gemm.h](../../../provided_by_member4/member4_conv_mac/成员4/hls/conv2d_gemm.h)中的oc0循环 |
| 每个输出位置和通道对应局部累加元素，完成点积后写回；乘加的m、j循环展开，局部累加数组完全分割 | [conv2d_gemm.h](../../../provided_by_member4/member4_conv_mac/成员4/hls/conv2d_gemm.h)中的acc数组、UNROLL及ARRAY_PARTITION |
| 卷积输入按ic、kh、kw循环直接读取，没有显式构造完整im2col矩阵 | 同一文件中的av_loc读取和索引计算 |
| 在LENET_USE_FIXED与LENET_ACC_INT同时启用时，使用本地ap_int<33>累加；点积完成后调用quantize_acc | [gemm_systolic.h](../../../provided_by_member4/member4_conv_mac/成员4/hls/gemm_systolic.h)中的条件类型、quantize_acc，以及conv2d_gemm.h中的乘加与写回分支 |
| quantize_acc按右移、最近偶数舍入、饱和及raw位装载的顺序写回 | 同一文件中NARROW_SHIFT与quantize_acc的函数体 |

成员4申报公共数值配置为激活`ap_fixed<12,7>`、权重`ap_fixed<12,1>`，共享累加器`ap_fixed<32,16>`，舍入`AP_RND_CONV`、溢出`AP_SAT`；本地整数路径另用33位累加器。上述数值规则见[设计说明](../../../provided_by_member4/member4_conv_mac/成员4/docs/member4_conv_mac.md)及[公共配置](../../../provided_by_member4/member4_conv_mac/成员1_成员3/config/quant_params.h)。它们记录为交付申报/代码配置，不将四份报告重新归入推定的统一代码版本。

成员4申报的接口为Conv1 `[1,28,28] → [6,24,24]`，Conv2 `[6,12,12] → [16,8,8]`，采用CHW/OIHW、无Bias、stride=1、padding=0，见[回交记录](../../../provided_by_member4/member4_conv_mac/成员4/ACCEPTANCE.md)。本次没有进行整网集成或接口联调。

## 5. 沿用的验证记录

以下来自成员4[接收与回交记录](../../../provided_by_member4/member4_conv_mac/成员4/ACCEPTANCE.md)，作为“成员4报告的测试结果”采用；本次未运行测试程序、C Simulation或综合。

| 原记录项目 | Conv1 | Conv2 |
|---|---:|---:|
| Float最大绝对误差 | 4.81455e-07 | 4.80131e-06 |
| Float平均绝对误差 | 1.40007e-08 | 2.97237e-07 |
| Fixed raw Mismatch Count（原表分项） | 0 / 13,824 | 0 / 8,576 |
| C Simulation | 成员4报告PASS | 成员4报告PASS |

同份记录还说明：WSL g++ Float的5个真实样本通过；Fixed整数路径的5个真实样本与3个生成向量无不匹配；legacy路径无不匹配；HLS CSim统计29,316个输出、0 mismatch。

原表分项计数与CSim总数按各自原文分别保留，不合并或补算为同一范围。当前材料没有提供这些统计范围之间的逐项映射，本次不推定其对应关系。[测试程序](../../../provided_by_member4/member4_conv_mac/成员4/tests/tb_conv_systolic.cpp)与[测试脚本](../../../provided_by_member4/member4_conv_mac/成员4/tests/run_tests.sh)用于定位交付测试方法，不替代本次未执行的复测。

这些是Conv模块测试记录，不是10,000张MNIST的整网HLS准确率，也不能替代成员7的整网对齐交付。成员3材料中记录的HLS尚未运行，仍保留其原有包内范围；成员4的模块CSim申报另行记录，二者不合并为统一冻结版本。

## 6. 图表与可直接引用的文案

- [延迟对比图](../../figures/member4_latency_comparison.png)：两层分别展示指定基线与最终版cycles，附下降率和延迟比。
- [资源对比图](../../figures/member4_resource_comparison.png)：LUT、FF、DSP分面展示，各纵轴从0开始；BRAM_18K四组均为0，以文字说明。

图中`Naive baseline`为初始朴素卷积，`Final`为成员4交付最终版；来源编号对应本文件R1～R4。

**建议正文：** 相对于初始朴素卷积实现，成员4最终版Conv1的综合延迟由44,945降至17,426 cycles，下降61.23%；Conv2由162,818降至12,226 cycles，下降92.49%。Conv1的LUT下降23.57%，Conv2的LUT增加659.86%，两层FF和DSP均增加。两种实现结构不同，以上为整体实现对比；现有材料不支持把变化归因于单一优化。

**延迟图注：** Conv1/Conv2指定朴素基线与最终版的综合延迟。四份报告目标时钟均为10 ns，按cycles计算变化；对应两层的基线/最终版延迟比分别为2.58和13.32。结果限于独立卷积模块。

**资源图注：** Conv1/Conv2两种实现的顶层综合资源。LUT、FF、DSP分别按实际数量展示，BRAM_18K均为0；两种实现结构不同，图中不分解各项改动的独立贡献。

## 7. 核查记录与完成边界

已核对四份报告的顶层资源、Latency、Interval、时钟及元数据，完成两层共10项指标比较；全部143个成员4包内原始文件均记录SHA256，执行前后保持不变。源报告数值行号、原始文件哈希和成果核查见[verification.json](verification.json)。未修改旧MANIFEST，也未据旧清单宣称包版本已统一。

本次成员4获批范围完成。后续仍需其他成员的整网集成、综合、验证及冻结版本材料；本次不补造缺失数值、不解释无依据的变化原因、不重跑或修订混杂代码版本。
