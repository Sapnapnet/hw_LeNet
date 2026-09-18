# 成员7逐层验证与更新汇总

日期：2026-09-08。来源版本：`member7-alignment-v1`，见[原始报告](../../../provided_by_member7/member7/alignment_report.md)。本次只读取已有输出、参考、CSV和代码，重算统计并整理；未编译、运行测试平台、重跑推理、综合或RTL验证。

## 1. 与成员6重复及更新的内容

按相对路径和SHA-256比对成员6的250个文件与成员7附带handoff的261个文件。213个字节完全一致，25个仅换行不同，12个有文本变化，新增11个，无文件缺失。逐文件哈希及采用位置见[source_comparison.json](source_comparison.json)。完全一致的文件不复制到整理目录，不重复计作实验。

两份成员6整网综合报告、四份C仿真日志及顶层RTL完全一致，权重与参考数据也完全一致。继续使用[成员6综合表](../member6/synthesis_summary.csv)和[原图表与结果](../member6/integration_summary.md)，不重新计算资源或性能，不将新代码路径修正说成新综合实验。

有差异的文件按用户指示以成员7为最新，原件都保留。25个格式变化也登记最新路径，但不视作语义变化。12个文本差异涉及m4/README、m5/README及来源说明、5个算子头文件、综合/测试脚本及测试入口，主要修正旧目录名为m1/m1m3/m4等ASCII路径，未发现计算算法改动。旧包“路径未修正”保留为当时记录，当前采用成员7所附版本说明该问题已在其文件中修正，不由本次代为修复或复现。

新增test_vectors由成员7重建：3组向量各含input/weight/expected，加generate.py和PROVENANCE.md，共11个文件，见[向量来源](../../../provided_by_member7/member6-handoff-v1/test_vectors/PROVENANCE.md)。只登记来源和成员7申报的验证，不冒充成员4原向量；原报告所述成员4后续复核仍是向量交接事项，不阻断本次五个真实样本逐层汇总。此前用户指定的成员4朴素基线不受影响，继续采用原指定报告。

## 2. 逐层结果与判据

详表见[layerwise_summary.md](layerwise_summary.md)，完整精度及来源见[layerwise_results.json](layerwise_results.json)。

- Float：5样本×13观测点，57,690个值，|差值|>1e-5数量为0；全局最大绝对差5.10607544e-6，位于样本3的fc1，relu3也保留同一最大差。logits最大绝对差3.57860107e-6。
- Fixed：同样57,690个raw值，差异数量、最大及平均差均为0；65个dump文件与实际参考字节完全一致。
- 类别：Float与Fixed输出均预测7、2、1、0、4；对应输入与成员1本包MNIST参考一致，Fixed编码为ties-even乘32，标签关联原始MNIST IDX，五个样本均正确。
- 量化差异：Fixed raw/32与Float参考比较，logits最大绝对差0.1309085、均值约0.0501123。这里超1e-5的差异不是对齐失败；Fixed实现正确性由raw与Fixed参考的一致性判断。

Float C++测试要求mismatch(|diff|>1e-5)=0且全局max≤1e-3；compare.py默认直接按1e-5判断。当前输出同时满足两者。Fixed按整数raw精确相等。均值为每层5个等长样本的元素均值，不把所有层的均值上界称作全网总体均值。

原README写最坏层conv2、5.75e-6，与现存输出和详细报告不一致。本次采用重算后与CSV、详细报告一致的fc1/relu3、5.10608e-6；README原件不修改。也不直接引用报告中未经本次验证的浮点误差因果解释。

## 3. 验证范围

[测试平台](../../../provided_by_member7/member7/tests/tb_layerwise.cpp)按成员6 Controller的层顺序逐个调用同批HLS C++模块，保留并导出各层Buffer。它不是从成员6顶层直接导出中间信号，也不是综合RTL波形或RTL协同仿真。应表述为“五个参考样本的HLS C++模块链逐层对齐通过”，结合已有成员6顶层C仿真记录说明验证覆盖。

参考来源：input/conv1/pool1/conv2来自m2 Float或m3 Fixed，其余来自m5/data。后半段为成员5生成的数据，不宣称全部由成员2/3的官方Python模型直接导出。成员1新提供的Float后半段参考未替换本次比较对象，前次记录的参考差异仍有效。

成员7报告Windows/MSYS2 g++及Vivado HLS 2018.3 include环境全量回归15/15通过，当前没有附完整回归控制台日志，因此仅注明“成员7报告”，不说本次复跑通过。已交付的两模式130个dump和三张非空CSV足以核查现有逐层指标，不要求补日志才能完成整理。报告中的首次失败及修复属于其历史申报，不能把当时无数据的全零表当作本次结果。

成员7建议组长可冻结中期版本，这是建议，不是组长实际冻结确认。当前未完成全MNIST HLS准确率统计；少量样本逐层通过也不自动宣布完整Level 1或最终课程验收完成。

## 4. 图表与可引用结论

- [Float逐层最大误差图](../../figures/member7_float_layerwise_error.png)：保留观测顺序，线性坐标从0开始，标注1e-5判据。
- [反量化与Float差异图](../../figures/member7_quantization_error.png)：展示每层最大/平均绝对差；标明为信息性量化比较。Fixed raw全零结果在表中给出，不另做无信息的全零图。

可引用：成员7交付的5个MNIST参考样本在13个观测点上完成HLS C++模块链逐层验证。Float共57,690个值的超1e-5差异数为0，最大绝对差5.10608e-6；Fixed共57,690个raw值逐值一致。两种模式的五个样本分类均与原始标签一致。本次核查已有输出和参考，不是重新执行HLS测试，结果不代表全测试集准确率。
