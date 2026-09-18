# 验证已完成

- 高精度版 RTL latency：64815 → 18428 cycles，下降71.57%；低延迟版 aggressive5b：11025 cycles，下降82.99%。两组3张C/RTL cosim均Pass。
- 高精度版10000张HLS CSim：9894/10000，98.94%；100000个raw logits与原参考完全一致。
- 低延迟版10000张HLS CSim：9655/10000，96.55%；通道选择经全量离线穷举（首轮93.44%，改删通道3后96.55%）。
- Flatten删除：单独节省258 cycles和1 BRAM_18K；5张Csim、地址映射与参考布局验证通过。
- SAIF功耗估算已覆盖 baseline、optimized，外部存储模型无效预读保持旧值；各自30个重放logits与原C向量匹配。

完整结果见[REPORT_CN.md](REPORT_CN.md)，复现见[README.md](README.md)。中间实验保存在experiments，不作为最终结果。
