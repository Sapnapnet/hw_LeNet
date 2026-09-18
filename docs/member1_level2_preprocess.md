> 2026-09-14更新：成员8已接手统一预处理，当前190张及增量入口见[现行工作流](member8_level2_workflow.md)。以下原接收说明保留历史范围。

# 成员 1：Level 2 HLS C++ 预处理与交接规范

## 边界与实现归属

`hls/preprocess/level2_preprocess.cpp` 是唯一的图像预处理实现：灰度、阈值/前景极性、ROI、保持比例缩放、居中填充和 A12 导出均在该 HLS C++ 文件中。`scripts/export_level2_rgb.ps1` 仅使用系统 JPEG/PNG 解码器把原始文件适配为 P6 RGB；它不改变像素内容或执行预处理。

最大输入尺寸为 640x480，覆盖当前 55 张图片中的最大尺寸 600x410。更大原图应先在数据采集阶段裁为单个数字/背景区域，不能绕过这个上限。

## 两条预处理链

| 名称 | HLS C++ 数据通路 | 输出 |
|---|---|---|
| Simple | RGB -> 1:2:1 整数灰度 -> 像素中心最近邻 Resize(28x28) | 8-bit 28x28 PGM |
| Full | RGB -> 灰度 -> 由亮度范围确定阈值 -> 用边界亮度判断前景极性 -> ROI -> 长边缩放到 20 px -> 居中填充(28x28) | 二值、MNIST 极性 PGM |

Full 路径的无目标背景若不存在可分离前景，会输出全零图，同时在 `level2_manifest.csv` 中记录 `full_roi_valid=0`。该信息是抗干扰样本的可审计证据；十分类 LeNet 本身仍会强制输出某个 0~9 类，不能把该输出计为“background 识别正确”。

## 定点交接

每个 Simple/Full PGM 均由 HLS 函数 `level2_export_a12_raw` 导出 784 行 raw signed integer 文本。规则与 `tests/tb_mnist_10k.cpp` 一致：

```text
raw = round_ties_to_even((pixel / 255) * 32)
```

对应 A12：12-bit signed、I7/F5、scale 1/32。0 与 255 分别导出 0 与 32。

## 文件布局与成员 7 交接

```text
data/self_collected/
  original/{0..9,background}/
  raw_rgb/{0..9,background}/                 # 解码适配输入
  processed_simple/{0..9,background}/        # 28x28 PGM
  processed_full/{0..9,background}/          # 28x28 PGM
  fixed_input/simple/{0..9,background}/      # 784 raw values
  fixed_input/full/{0..9,background}/        # 784 raw values
  level2_manifest.csv                         # 全部路径、标签、Full ROI 审计字段
```

成员 7 用 `label` 为 `0`~`9` 的行计算十分类 Accuracy；`background` 行单独输出预测、logits、max-logit/置信度和 `full_roi_valid`，不得混入 Accuracy 分母。

## 复现

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_level2_preprocess.ps1 -PrepareRgb
```

脚本先编译并调用 HLS C++ 核的批处理驱动，随后生成全部 PGM、Fixed raw 输入与总 manifest。独立 CSim 还可执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_level2_preprocess.ps1 -RunHlsCsim -VivadoHls <vivado_hls.bat 的完整路径>
```

对 Full HLS 核执行 C synthesis：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_level2_preprocess.ps1 -RunHlsSynth -VivadoHls <vivado_hls.bat 的完整路径>
```

为兼容 Vivado HLS 2018.3 对非 ASCII 工作路径的限制，CSim 默认会将本工程的预处理源码和测试平台复制到 `D:\lenet_level2_hls_csim` 后执行。可用 `-HlsStageRoot <ASCII 可写目录>` 改写；该目录仅为可删除的构建副本，工程内源码仍是唯一权威版本。
