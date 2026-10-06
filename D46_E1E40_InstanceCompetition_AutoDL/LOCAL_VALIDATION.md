# D46 本地交付验证

2026-10-06，本地Windows/Python3.9/Torch2.6，CUDA设备RTX4060 Laptop 8GiB。

已实际执行：

- CPU单元检查：512个3x3二分图与独立穷举oracle完全一致；空GT/空候选/错误类别；重复GT独立覆盖与一对一覆盖差异。
- CUDA单元检查：候选ID携带不改变原NMS；同类抑制、跨类保留、平分数不确定性、30005候选的max_nms=30000预算及top500前缀；原raw张量不变。
- CPU真实checkpoint合成集成：E1 seed1、E40 seed1，各四张不同宽高合成图（包含空GT图）；标准Predictor对齐、坐标转换、权重SHA/无梯度、几何输出、原官方评分与报告生成通过。
- CUDA真实checkpoint合成集成：同上，在本地GPU和4GiB allocator cap下通过。

这些是工程测试，不是真实VisDrone诊断结果，不证明拥挤是根因，不替代AutoDL/Torch2.5.1环境门禁。
服务器需执行check与四模型各8张真实图smoke；全量548图原始预测/官方指标复现后才能解释审计。

未执行：远程服务器命令、真实val全量推理、任何训练或远程修改。
