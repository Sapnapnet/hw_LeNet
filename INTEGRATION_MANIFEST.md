# 成员 1 集成清单

生成日期：2026-09-09

## 已纳入的成员交付

| 成员 | 集成内容 | 位置 |
|---|---|---|
| 1 | 网络尺寸、接口、公共配置 | `config/`, `docs/interface.md` |
| 2 | Float 模型、导出脚本、权重、五个 Float reference | `python/`, `weights/float/`, `reference/float/` |
| 3 | W12/A12 参数、公共 Fixed 类型、核心整数定点模型、Fixed 权重/reference | `config/`, `python/quant/`, `config/quant_config.json`, `weights/fixed/`, `reference/fixed/` |
| 4 | Conv/GEMM systolic、单元测试与报告说明 | `hls/conv/`, `docs/member4_conv_mac.md` |
| 5 | ReLU、MaxPool、Flatten、FC、Argmax | `hls/operators/` |
| 6 | Buffer、地址生成、Controller、整网顶层 | `hls/buffer/`, `hls/top/` |
| 7 | 整网 Testbench、逐层结果和对齐说明 | `tests/`, `results/layerwise/`, `docs/member7_alignment.md` |
| 8 | 自采数据目录、预处理结果和中期汇总材料 | `data/self_collected/`, `docs/midterm/`, `results/midterm/` |

## 静态完整性判断

- HLS 主链完整：Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → ReLU → FC → ReLU → FC → logits。
- 所有 HLS include 已改为本工程内的 `config/`、`hls/conv/`、`hls/operators/`、`hls/buffer/` 和 `hls/top/` 路径。
- Float/Fixed 权重五层均存在，数量分别为 150、2400、30720、10080、840。
- 五个样本的 Float/Fixed layerwise reference 均已合并；Fixed 后半段来自成员 5 交接，保留原始 provenance。
- 可执行入口和 Vivado HLS Tcl 已改为本工程路径；本次没有运行它们。

## 尚需后续处理的事项

成员 3 原始交付中没有完整的 PTQ 运行入口（如从 checkpoint 自动生成全部定点资产的 `run.py`）。本集成保留了可复用的 `fixed.py` 与既有生成资产；如果需要从头重做量化扫描，应另行补齐该流水线。
