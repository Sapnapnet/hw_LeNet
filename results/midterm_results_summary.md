# 中期结果汇总

课程：智能芯片与系统设计综合实践。路线B，成员8。

更新日期：2026-09-08。状态：成员8预处理框架、成员1/2 Float及网络、成员3准确率、成员4卷积、成员6整网及成员7逐层结果整理已完成；总体架构、整组状态和冻结版本仍待补齐。

## 1. 当前工作状态

已建立成员8工作区、数据规范、临时输入约定及实验结果模板。已实现单张/批量预处理，初始版本24项受控检查通过，见[检查报告](results/preprocess/check_report.md)。初始默认128处理0～9共10张时，8张ROI基本合理、3和6受杂点影响，原结果保存在run_001，见[初始实拍记录](docs/第2步_实拍预处理实验与结果.md#2-初始默认128发现3和6的定位问题)。2026-09-08已核对成员1/2本包Baseline的输入约定，现有输出格式一致；模型联调与自采识别尚未进行。

2026-09-07经过[128/136/144/152四档实验](docs/第2步_实拍预处理实验与结果.md#3-四档实验选择144)，按用户批准将当前默认阈值设为144，脚本版本更新为0.2.1。全部10张重新处理至run_002，格式与ROI人工检查通过，3、6问题已解决，其余8张未见明显新增裁剪问题，见[整批更新](docs/第2步_实拍预处理实验与结果.md#4-默认144整批重跑与前后对比)。128原结果及四档实验均保留，未运行识别。

已核查成员3交付包，整理Float及9档Fixed准确率、定点格式与选择依据，并复用已有位宽—准确率图。此次为材料及数据一致性核查，未重跑模型或HLS，不据计划日期推定全组工作已完成。

已完成成员4指定朴素基线与最终版的综合结果对比，形成四组综合记录、重算变化和两张图。成员4经用户补充说明：朴素基线与最终版结构不同；因此采用整体实现对比，不将变化归因于单一优化。详见[成员4完成说明](docs/第3步_成员4结果接收与整理.md)。

工作进度与验收见[总方案](docs/总方案与验收进度.md)，具体工作见[第1步完成说明](docs/第1步_完成说明.md)、[第2步完成说明](docs/第2步_完成说明.md)及[第3步成员3完成说明](docs/第3步_成员3结果接收与整理.md)。

成员1/2由同一人承担。已直接接收Float模型、权重、准确率及完整参考输出；本次与既有副本的关联、真实标签补证和限制见[Float基准汇总](results/midterm/member1/baseline_summary.md)。总体架构等未交付部分另待补充。

## 2. 实验结果接收状态

| 内容 | 来源 | 当前状态 | 经核查的记录/文件 |
|---|---|---|---|
| Float Accuracy | 成员1/2直接提供；此前经成员3转交 | 已核对两份记录一致 | [直接交付记录](provided_by_member1/results/baseline_accuracy.json)，98.97%（9897/10000），不重复计为实验 |
| 网络结构、Float模型、权重及输入约定 | 成员1/2 | 本次整理完成 | [网络与输入](results/midterm/member1/network_structure.md)，全组最终冻结仍待确认 |
| 定点配置、Fixed Accuracy与精度变化 | 成员3 | 本次整理完成 | [成员3汇总](results/midterm/member3/quantization_summary.md)，交付包member8-fixed-accuracy-v1 |
| Conv优化数据 | 成员4 | 本次整理完成 | [成员4汇总](results/midterm/member4/conv_optimization_summary.md)，指定朴素基线与final报告；申报版本member4-conv-mac-final-v1 |
| 整网集成、C仿真、资源及延迟/时钟 | 成员6 | 本次整理完成 | [整网汇总](results/midterm/member6/integration_summary.md)，member6-handoff-v1；未复现 |
| Float/Fixed逐层误差、logits与类别 | 成员7 | 本次整理完成 | [逐层汇总](results/midterm/member7/alignment_summary.md)，5样本C++模块链；非RTL或全测试集验证 |
| 总体架构、完成度、中期冻结版本 | 成员1 | 仍待整组确认 | 成员6提供其整网实现依据，不等于组长最终冻结 |

接收字段与核查规则见[结果模板](results/midterm/results_template.md)。数据到齐后填入经核查的记录链接，不在本文件另建不同口径的原始记录。

## 3. 答辩图表

| 图表 | 当前状态 | 文件 |
|---|---|---|
| 总体硬件架构图 | 总体架构尚未交付 | 待生成 |
| 网络结构图 | 按成员1/2本包模型完成 | [结构图](results/figures/lenet_network_structure.png)、[层表及说明](results/midterm/member1/network_structure.md) |
| Python Float → Fixed → HLS验证流程图 | 待核对实际流程 | 待生成 |
| Float/Fixed准确率表 | 成员3部分已完成 | [表格及说明](results/midterm/member3/quantization_summary.md)、[CSV](results/midterm/member3/accuracy_summary.csv) |
| 综合资源及可用性能结果表 | 成员4卷积及成员6整网部分已完成 | [综合表](results/midterm/member4/synthesis_summary.csv)、[比较表](results/midterm/member4/comparison_summary.csv)、[资源图](results/figures/member4_resource_comparison.png) |
| 成员6整网资源与最大延迟图 | 已完成 | [综合表](results/midterm/member6/synthesis_summary.csv)、[资源图](results/figures/member6_resource_comparison.png)、[延迟图](results/figures/member6_latency_comparison.png) |
| Float逐层与量化差异图表 | 已完成 | [逐层表](results/midterm/member7/layerwise_summary.md)、[Float误差图](results/figures/member7_float_layerwise_error.png)、[量化差异图](results/figures/member7_quantization_error.png) |
| 完成情况、问题和后续计划 | 待核对团队进度 | 待整理 |
| 位宽—准确率（补充项） | 已复用成员3原图，未修改 | [展示图](results/figures/member3_bitwidth_accuracy.png)、[图注及文案](results/midterm/member3/quantization_summary.md#4-图表和可直接使用的文案) |
| 朴素基线与最终版延迟（补充项） | 成员4卷积部分已完成 | [延迟图](results/figures/member4_latency_comparison.png)、[说明与图注](results/midterm/member4/conv_optimization_summary.md) |

## 4. 可引用结论

成员8预处理框架输出28×28灰度图及CHW浮点输入，2026-09-08已按本包Baseline核对输入约定。初始0.2.0版本24项受控检查通过；现0.2.1版本默认144，当前10张实拍格式及ROI人工检查通过。这里10/10是预处理检查结果，不是识别准确率；没有宣称0.2.1重跑了全部受控检查。见[初始框架说明](docs/第2步_完成说明.md)与[最新整批记录](docs/第2步_实拍预处理实验与结果.md#4-默认144整批重跑与前后对比)。

成员3交付结果显示，同一10,000张MNIST测试集上，Float为98.97%，W12/A12、32位累加器为98.94%，下降0.03个百分点；预测不同7张，净少正确3张。W12按“至少12位、下降≤0.10个百分点、未观察到权重及激活饱和”的规则选择，不是全局最优配置。见[成员3汇总](results/midterm/member3/quantization_summary.md)。

成员3整数验证记录报告115,380,000个逐层整数值比较无差异，属于Python数值实现之间的验证；HLS C Simulation及综合记录为NOT_RUN，资源与延迟为NOT_MEASURED。不能据此宣称Python-HLS对齐成功、资源节省或性能提升。见[证据与边界](results/midterm/member3/quantization_summary.md#5-验证证据与结论边界)。

成员4指定朴素基线与最终版的报告对比显示：Conv1延迟由44,945降至17,426 cycles，下降61.23%；Conv2由162,818降至12,226 cycles，下降92.49%。Conv1 LUT下降23.57%，Conv2 LUT增加659.86%，两层FF、DSP均增加。四份报告目标时钟均为10 ns；结果属于不同结构的独立Conv模块对比，不是单项优化贡献或整网性能，见[成员4汇总](results/midterm/member4/conv_optimization_summary.md)。

成员4另报告Conv模块CSim通过、Fixed raw无不匹配，本次保留其交付申报，未重跑验证。该模块记录与成员3包内HLS状态分别保留，不据此推定整网已对齐或版本已统一。

成员6整网报告显示，Fixed相对Float的LUT、FF、BRAM分别减少61.83%、52.52%、53.33%，DSP增加5.04%，报告最大延迟由459,706增至474,826 cycles，增加3.29%。两模式目标周期均10 ns，属于HLS综合估计。5个样本Fixed最终50个logits raw比较无差异，Float最大绝对误差3.5786e-6，类别与参考argmax一致；参考后半段由成员5生成，当时未核对真实标签，不等于逐层验收或测试集准确率。本次已通过成员1/2包内MNIST输入及标签补充确认这5个样本类别正确，原误差仍保持原参考口径。见[成员6汇总](results/midterm/member6/integration_summary.md)。

成员7交付的5个MNIST样本在13个观测点（含输入）上，Float共57,690个值超1e-5差异数为0，最大绝对差5.10608e-6；Fixed共57,690个raw值逐值一致。两种模式五个样本分类均正确。观测由C++测试平台镜像Controller顺序调用模块完成，未表示RTL协同仿真或全测试集通过。逐层参考保持成员7实际采用的成员2/3和成员5来源，见[验证范围](results/midterm/member7/alignment_summary.md)。

后续每条结论应附配置、比较口径及支撑记录链接，明确适用范围。

## 5. 当前限制与后续工作

- 已按成员1/2本包Baseline完成输入格式、布局、归一化约定核对；成员3定点规范已取得，模型与定点入口尚未联调，见[数据规范](data/self_collected/README.md)。全组冻结版本仍待确认。
- 当前默认144的10张整批预处理检查已通过，最新结果为run_002；初始128结果及四档实验保留。更广泛的新输入适配和模型效果仍待验证，见[最新整批记录](docs/第2步_实拍预处理实验与结果.md#4-默认144整批重跑与前后对比)。
- 成员1/2、3、4、6、7本次整理范围均已完成，暂无需用户补充材料；其他成员材料到齐后，按后续获批方案继续。成员4基线按用户指定采用，原始报告、代码和说明保持不变。
- 原计划9月7日22:00前收集首批，9月8日22:00前完成图表和交付；目前确认成员1/2、3、4、6、7获批范围整理完成，整体进度仍按实际证据记录。
- Float训练按测试准确率选择最佳checkpoint，新直接材料确认该事实；本次未重跑训练。位宽选择也使用了测试集，不能描述为完全独立的最终留出评测。9档实验同时改变W、A和累加器，尚无配套资源及延迟，不能视为完整Level 3 DSE。
- 成员4两种结构的综合结果不对应成员3九档位宽实验，不将其拼接成位宽—资源或位宽—延迟曲线。代码与报告的准确版本映射未重建。
- 成员6历史报告和日志保持不变；成员7附带版本的路径修正按最新采用，见[文件比对](results/midterm/member7/source_comparison.json)。两份整网综合报告及原日志完全相同，不重复制表。复现不在范围内，顶层RTL也不视作完整RTL工程。
- 成员7README过时误差不采用，以dump/CSV支持的fc1及relu3最大差5.10608e-6为准。15/15回归是其报告申报，未附完整控制台日志；现有逐层输出已核查。组长冻结建议不是实际冻结确认。
- 中期后在满足前置条件时开展Level 2和Level 3，具体日期未定。

## 6. 最终交付记录

中期冻结版本：待确认。

材料交付日期、接收对象、交付方式：待实际交接后填写。

本次未对外发送材料。
