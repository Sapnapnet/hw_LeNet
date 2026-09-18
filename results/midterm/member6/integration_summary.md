# 成员6整网集成与综合结果汇总

整理日期：2026-09-08。来源版本：`member6-handoff-v1`。本次仅核查和汇总已有材料，未运行模型、C仿真、综合或RTL验证，也不修复交付代码。

## 1. 来源与采用口径

- 原始说明：[接收与回交记录](../../../provided_by_member6/ACCEPTANCE.md)、[交付README](../../../provided_by_member6/README.md)。
- Float原始报告：[综合报告](../../../provided_by_member6/m6/results/reports/float/lenet_accelerator_with_weights_csynth.rpt)。
- Fixed原始报告：[综合报告](../../../provided_by_member6/m6/results/reports/fixed/lenet_accelerator_with_weights_csynth.rpt)。
- 工具：Vivado HLS 2018.3，Build 2405991；器件：`xc7z020clg400-1`；目标周期10 ns，时钟不确定性1.25 ns。
- 两份报告均针对整网顶层`lenet_accelerator_with_weights`，不是成员4的独立卷积模块报告。报告日期分别为2026-09-08 01:23:04、02:11:26。
- 本次按包内申报版本关联说明、代码和结果，不重建其编译历史。原始250个文件哈希见[核查记录](verification.json)。

## 2. 整网综合结果

| 指标 | Float | Fixed |
|---|---:|---:|
| LUT | 45,272 | 17,279 |
| FF | 71,471 | 33,932 |
| DSP48E | 139 | 146 |
| BRAM_18K | 30 | 14 |
| Latency min，cycles | 30 | 30 |
| Latency max，cycles | 459,706 | 474,826 |
| Interval min，cycles | 30 | 30 |
| Interval max，cycles | 459,706 | 474,826 |
| 顶层Pipeline类型 | none | none |
| 目标周期，ns | 10 | 10 |
| 综合估计周期，ns | 9.325 | 8.702 |

完整字段与来源见[综合表](synthesis_summary.csv)。展示延迟采用报告最大值，保留最小值但不把30 cycles当作完整推理的实测延迟。当前没有RTL协同仿真或上板计时证据，最大值同样只是报告估计，未独立测量实际一次推理周期数。

估计周期均小于10 ns目标周期。这仅描述HLS报告数值，不代表布局布线后的时序收敛或上板频率。报告还包含1.25 ns不确定性，不用简单的周期大小比较宣布最终时序验收通过。

### Float与Fixed比较

变化量=Fixed−Float；变化百分比=(Fixed−Float)/Float×100%。CSV保留计算精度，正文取两位小数。

| 指标 | 变化量 | 变化百分比 |
|---|---:|---:|
| LUT | -27,993 | -61.83% |
| FF | -37,539 | -52.52% |
| DSP48E | +7 | +5.04% |
| BRAM_18K | -16 | -53.33% |
| latency_max_cycles | +15,120 | +3.29% |
| estimated_clock_ns | -0.623 | -6.68% |

详细比较见[比较表](comparison_summary.csv)。Fixed减少LUT、FF和BRAM，DSP和报告最大延迟增加，不能概括为定点版全面加速。此处仅比较本包两种模式，不将其与成员3九档位宽准确率拼接成完整DSE，也不与成员4不同工具版本的模块报告相减来推断集成开销。

## 3. C仿真记录

原始日志：[Float整网](../../../provided_by_member6/m6/results/csim/accelerator_float_csim.log)、[Fixed整网](../../../provided_by_member6/m6/results/csim/accelerator_fixed_csim.log)、[Float地址与Buffer](../../../provided_by_member6/m6/results/csim/address_buffer_float_csim.log)、[Fixed地址与Buffer](../../../provided_by_member6/m6/results/csim/address_buffer_fixed_csim.log)。四份日志均记录`CSim done with 0 errors`。

| 样本 | Float最终输出最大绝对误差 | Fixed最终输出raw不匹配数 | 两种模式预测类别 | 参考argmax |
|---|---:|---:|---:|---:|
| sample_00000 | 2.55442e-06 | 0/10 | 7 | 7 |
| sample_00001 | 3.20675e-06 | 0/10 | 2 | 2 |
| sample_00002 | 2.39389e-06 | 0/10 | 1 | 1 |
| sample_00003 | 3.5786e-06 | 0/10 | 0 | 0 |
| sample_00004 | 3.19344e-06 | 0/10 | 4 | 4 |

统计范围：每种模式5个样本×10个最终logits。Float全局最大绝对误差3.5786e-6；Fixed共50个raw输出比较，无不匹配。逐行数据见[仿真表](csim_summary.csv)。不适用的误差字段和未提供的真实标签留空，不能把空值当成0。

[测试平台](../../../provided_by_member6/m6/tests/tb_lenet_accelerator.cpp)的Float通过标准为最大绝对误差≤1e-3且类别与参考argmax相同；Fixed要求raw逐值一致且类别相同。地址与Buffer日志另记录CHW、OIHW、FC寻址、CHW读写和copy测试通过。

参考数据来源：输入及Conv参考来自成员2/3副本；后半段logits来自成员5按这些数据生成的参考，见[来源说明](../../../provided_by_member6/m5/data/PROVENANCE.md)。本次核对了每份参考logits恰有10个值，argmax与日志一致，没有重新生成参考。

验证边界：

- 日志`ref`是参考输出的argmax，测试没有读取真实标签，因此只确认类别与参考一致。
- 测试只比较最终logits，未逐层比较，也未保存HLS侧完整逐层输出；包内已有参考层文件不等于已有逐层验证结果。
- Float均值误差等未报告的统计不补造。少量参考样本通过不等于MNIST测试集准确率。
- 这补充了整网C仿真证据，不能替代成员7的完整逐层对齐交付。

## 4. 代码支持的实现说明

[顶层代码](../../../provided_by_member6/m6/hls/lenet_accelerator.cpp)为各层分配独立线性Buffer，并调用[Controller](../../../provided_by_member6/m6/hls/controller.h)依次执行输入复制、Conv1/ReLU1/Pool1、Conv2/ReLU2/Pool2、Flatten、FC1/ReLU3、FC2/ReLU4、FC3及输出复制。顶层只输出10个logits，argmax在测试平台执行。

特征图采用CHW，卷积权重OIHW，FC权重按[OUT][IN]线性顺序传入分块接口。FC1/2/3输出通道分块分别为8、12、5。输入为1×28×28，Flatten为256，FC为256→120→84→10，无Bias，FC3之后无ReLU或Softmax。

公共配置为W12/A12，权重小数11位、激活小数5位，公共acc_t为32位。注意[实际GEMM整数分支](../../../provided_by_member6/m4/hls/gemm_systolic.h)使用局部33位整数累加器，FC也复用该GEMM。因此不把公共32位类型直接等同于所有模块实际MAC位宽。当前不进行版本修复或重建。

Float/Fixed综合报告的Memory表分别列出32位/12位层间存储及30/14个BRAM_18K，可据此说明本报告中的存储位宽和BRAM映射差异。DSP增加或延迟变化的具体原因未进一步验证，不作因果归因。

权重通过顶层端口输入，未固化内部ROM；无AXI、Buffer复用、Ping-Pong或DATAFLOW，尚未完成最终Zynq系统集成。旧目录include路径、缺少被引用的设计说明、仅提供顶层RTL等作为原始交付现状记录；用户已明确本次不要求复现，这些不阻断汇总，也不要求补齐后才验收本次工作。

## 5. 展示图与可引用结论

- [整网资源对比](../../figures/member6_resource_comparison.png)：分别展示四类资源，坐标轴不混用不同资源的计数。
- [整网最大延迟对比](../../figures/member6_latency_comparison.png)：展示报告最大cycles，两模式目标周期均10 ns。

可引用：成员6交付的Vivado HLS 2018.3整网报告显示，Fixed相对Float的LUT、FF、BRAM_18K分别减少61.83%、52.52%、53.33%，DSP48E增加5.04%，报告最大延迟增加3.29%。5个参考样本的Fixed最终50个logits raw比较无差异，Float最大绝对误差为3.5786e-6，预测类别与包内参考argmax一致。上述验证属于交付日志记录，本次未复测；全组逐层验收及冻结版本仍待确认。
