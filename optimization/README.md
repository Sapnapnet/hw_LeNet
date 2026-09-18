# LeNet latency 优化与功耗验证

本目录基于指定的 `FPGA_LeNet_final.zip` 开发，原始 `hls/`、`config/`、权重和参考数据保留。
ZIP 的位置和 SHA256 见 [source_archive.json](../../source_archive.json)。
本次实验统一使用 Vivado/Vivado HLS 2018.3、`xc7z020clg400-1`、10 ns 时钟、W12/A12 整数累加。

## 实现入口

- `lenet_fast.cpp`：优化整网顶层，仍使用 `lenet_accelerator_with_weights` 和原参数接口。
- `fast_kernels.h`：显式分银行卷积、融合 ReLU/Pool。
- 原来的 FC 运算与量化规则复用 `../hls/operators/fc.h`、`../hls/conv/gemm_systolic.h`。
- `experiments/cyclic_v1/`：第一版实验源码和报告，只加 cyclic partition 未解决 II=4，用于解释原因，不是推荐版本。
- `experiments/banked_v2/`：显式输入银行、Conv2 仍分两组通道的中间实验；其 CSim 在切换最终结构时主动中止，部分样本不能称为全量验证。

最终结构 Conv1 使用 8 个空间位置 × 6 通道，Conv2 使用 8 个空间位置 × 16 通道。C/C++ 参数布局仍为原 OIHW / `[OUT][IN]`。生成的 RTL 将 Conv2 权重划为 16 个通道存储块，每块 150 个数；接入 FPGA 系统时需按新生成的 RTL memory 端口连接，不能直接套用旧顶层的端口接线。

低延迟版 `aggressive5b` 在同一顶层上通过编译宏切换（`LENET_AGGRESSIVE5 -DLENET_FC1_TILE24`）：Conv1 输出裁为 5 通道、Conv2 输出裁为 12 通道、FC1 改为 24 路并行。保留哪些通道由 `../tests/tb_mnist_10k.cpp` 里的 `kConv1Keep`/`kConv2Keep` 决定，权重在 testbench 加载时按该表收集；选择依据是 `eval_prune_variants.py` 的 10000 张定点离线穷举（`results/channel_ablation.json`）。

## 七组对照

| Variant | 作用 |
|---|---|
| baseline | ZIP 原始顶层和算子，复现旧报告，并通过 RTL 仿真取得真实周期 |
| straight | 原算子和复制步骤不变，改成固定顺序调用，辨别状态机报告上界的影响 |
| noflatten | 在 straight 上只移除 Pool2 → Flatten 的复制，单独量化其收益 |
| optimized | 高精度交付版：显式分银行并行读取、简化卷积位置计算、ReLU/Pool 融合、消除 flatten 复制 |
| fc24b | 在 optimized 上把 FC1 并行输出从 8 提到 24（15,798 cycles） |
| aggressive12b | 再把 Conv2 输出通道裁到 12（13,936 cycles，只做了综合） |
| aggressive5b | 低延迟交付版：再把 Conv1 输出通道裁到 5（11,025 cycles，全部验证通过） |

## 运行

在 `FPGA_LeNet_final` 下打开 PowerShell：

```powershell
.\optimization\run.ps1 -Variant baseline -Stage synth
.\optimization\run.ps1 -Variant baseline -Stage cosim -Samples 3
.\optimization\run.ps1 -Variant straight -Stage synth
.\optimization\run.ps1 -Variant noflatten -Stage synth
.\optimization\run.ps1 -Variant optimized -Stage synth
.\optimization\run.ps1 -Variant optimized -Stage csim -Samples 10000
.\optimization\run.ps1 -Variant optimized -Stage cosim -Samples 3
.\optimization\run.ps1 -Variant fc24b -Stage synth
.\optimization\run.ps1 -Variant aggressive12b -Stage synth
.\optimization\run.ps1 -Variant aggressive5b -Stage synth
.\optimization\run.ps1 -Variant aggressive5b -Stage csim -Samples 10000
.\optimization\run.ps1 -Variant aggressive5b -Stage cosim -Samples 3
python optimization/check_layout.py
python optimization/eval_prune_variants.py
python optimization/collect_results.py
```

默认工具目录为 `D:\Xilinx\Vivado\2018.3`，可用 `-VivadoRoot` 改写。
各组 HLS 工程在 `build/<variant>/solution1`；C simulation 使用独立的 `<variant>_csim` 工程。
输出在 `results/<variant>/`，汇总脚本区分完整/未完成的样本结果，原始报告在各 HLS 工程内。
`cosim_3.csv` 在 C 测试向量生成阶段已会出现；是否通过 RTL 验证必须看 `*_cosim.rpt` 的 Pass，不能只看 CSV。

## 功耗测量口径

HLS C Simulation 执行 C/C++ 功能测试，没有已映射的 FPGA 电路、布线电容和信号翻转时序，不能直接提供 FPGA 功耗（W）。本工程使用相同 C testbench 驱动生成的 RTL，再由 XSim 记录 SAIF 活动，导入 Vivado 的布局布线后电路执行 `report_power`。

```powershell
python optimization/capture_activity.py baseline
python optimization/capture_activity.py optimized
& 'D:\Xilinx\Vivado\2018.3\bin\vivado.bat' -mode batch -source optimization/power_impl.tcl -tclargs baseline
& 'D:\Xilinx\Vivado\2018.3\bin\vivado.bat' -mode batch -source optimization/power_impl.tcl -tclargs optimized
```

低延迟版把上面两条命令的变体名换成 `aggressive5b` 即可（脚本已支持）。注意该版本的输入是784个全分区寄存器，`capture_activity.py` 里 XSim 在 `--debug all` 下枚举 SAIF 对象会比前两版慢很多。

如果布局布线已完成、后来才生成 SAIF，可用 `power_report.tcl` 读回 `routed.dcp` 更新功耗报告，无须重新实现。

功耗重放使用`power_memory_models`：对逻辑容量之外的无效预读保持上次有效读数，避免原HLS检查模型在空闲时输出X。原功能cosim检查文件和DUT RTL不改，重放后的30个raw logits会再与原C测试向量比较。真实系统应匹配这个外部存储使能约定；该wrapper不计入核心功耗。

动态功耗范围为 OOC（独立模块）的加速器核心，不含外部权重存储、PS工作负载、DDR和板级电源损耗；静态功耗使用工具的全器件估算。输入/输出存储器由 testbench 提供，不是把本工程顶层的全部 memory 端口连接到物理引脚。统一环境温度 25℃、100 MHz，其余环境采用报告默认值，未做板级校准。`activity_power.rpt` 是仿真活动驱动的 FPGA 功耗估算，不能称为 CSim 功耗或板上实测功耗；`vectorless_power.rpt` 仅供对照。

官方依据：[UG902 2018.3 HLS](https://docs.amd.com/v/u/2018.3-English/ug902-vivado-high-level-synthesis)、[UG907 功耗分析](https://docs.amd.com/api/khub/documents/Gf3ajG3H3eT7C1YKAuUjYg/content)、[read_saif 命令](https://docs.amd.com/r/2022.2-English/ug835-vivado-tcl-commands/read_saif)。本机 2018.3 `doc/eng/man/read_saif` 和 `common/man/EN/man.cosim_design.xml` 也已核对。

## Flatten 为什么可消除

Pool2 逻辑形状为 `[16][4][4]`，线性地址 `16*c + 4*h + w`；FC1 的第 i 个输入也按此顺序读取。原 flatten 完全是 256 个数的恒等复制，无转置和计算。优化版直接把 Pool2 存储传给 FC1，不需重排 FC 权重，也无需强制指针类型转换。局部指针只是给同一个数组首行起别名，不产生新的硬件存储。

内部 Conv/Pool1 使用银行布局服务并行访问；Pool2 的生产端仍写回连续 CHW，保留 FC1 所需的顺序。若日后把 Pool2 改成 HWC、带 padding 或其他分块布局，必须重新检查地址映射，不能直接套用本结论。

`check_layout.py` 检查 ZIP 中 5 组 Fixed + 5 组 Float Pool2/Flatten 参考数据，并穷举两层卷积的 24,000 次输入地址映射，结果在 `results/layout_check.json`。

仅删除 flatten 的对照综合结果：`straight` 64,809 cycles → `noflatten` 64,551 cycles，节省 258 cycles（0.398%）和 1 BRAM_18K。5 个 MNIST 样本的 noflatten HLS CSim 已通过；其50个最终raw logits已由 `collect_results.py` 与原始 Python Fixed 参考逐位比对，全部一致。
