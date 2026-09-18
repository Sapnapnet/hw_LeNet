# 成员8统一预处理流程

2026-09-14职责更新：成员8负责接收图片、统一Simple/Full预处理及Fixed导出，随后把完整工程交成员7推理。成员1仍负责全组PPT合并。

## 数据与代码

正式数据唯一维护位置是data/self_collected。label.csv是成员8维护的数字索引，source_manifest.json保存来源、文件哈希和裁剪信息；脚本生成original/label.csv作为成员1解码接口需要的filename、label、writer_id、folder四列表，不手工维护第二套标签。

prepare_member7_level2.py不实现图像算法，负责核对编号与输入哈希、选择新或变更样本、调用原RGB解码适配器、编译并执行HLS C++主机驱动、检查PGM与Fixed对应、发布累计清单。对export_level2_rgb.ps1只增加可选LabelCsvPath参数，以便处理某批子集；默认调用行为不变。

HLS预处理源码和头文件、C++驱动保持接收版本不变。Simple灰度并直接最近邻缩放28×28；Full按亮度范围中点确定阈值，边界亮度判断极性、求ROI、长边20、居中28×28并输出二值图。Fixed逐像素采用(pixel×32+127)//255，A12 I7/F5、scale=1/32。

## 增量与版本

每次使用新的ASCII批次名。batches/<批次>/中保留该批输入适配表、RGB、Simple/Full、Fixed、驱动日志和核验报告；根level2_manifest.csv是当前全部数字唯一交接清单，其输出路径已带批次前缀。processing_state.json记录每张输入哈希和处理代码哈希。代码变化会使旧样本重新处理，以保证当前清单算法版本一致。

数据根中的dataset_version.json提供当前版本和总量。若出现处理失败，不发布新的累计清单；修复后用新批次名重试。新增样本需先完成来源核查再运行。原有批次文件不覆盖。

## 检查范围

脚本检查唯一编号、标签、包内路径、来源哈希、PGM尺寸和通道、Full二值像素、Fixed每像素编码。人工检查原图/Simple/Full对照，观察ROI、极性和笔画；困难样本保留，不按识别结果删图。实际观察记录及总览存于results/member8_level2。

这是预处理的主机C++执行，不是重新进行Vivado HLS CSim、RTL联合仿真或综合。已有预处理核综合报告和Level 1网络报告保持历史范围；本次不生成自采准确率。

## 历史与交接

工程原55张成果已在个人工作区归档，其中50张数字与当前成员1样本重复，不重复计数。5张背景及旧两路输入单独原样保存在background_reference，不进入当前340条数字测试集。

完整交接方式及成员7输出要求见上层README_MEMBER7.md。代码、原图和两路输出全部包内可访问；个人总方案和过程文档不属于运行依赖。
