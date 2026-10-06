# 已执行的本地工程验证

2026-10-05；本机 Python3.9.25 / torch2.6.0+cu124 / Ultralytics8.3.0。
服务器原环境不同，代码/协议签名与真实CUDA门禁仍需在那里复核。

- E50、E51各5个CPU单元测试通过：独立循环/展开oracle、边界、梯度、
  归一化、常量保持、零起点、loss-only两步真实更新、BF16值对照；
  公共E40 state与RNG对齐，19组参数完整归属，4种THOP hook政策匹配。
- 两项完整CPU check通过：原始YOLO损失及公共梯度、19组真实参数更新、
  活跃实际图路径、E40非平凡checkpoint旁路对齐、重载/fuse/新进程、
  标准Predictor、空GT、TAL等价、官方及几何单元检查。
- 两项2轮合成数据标准Trainer集成通过：真实优化、EMA、
  best/last保存重载、最终验证、4图矩形预测、官方评分fixture。
  首轮本地受限依赖读取失败后，已在可访问既有pandas的本地环境重跑通过；
  未替换原Trainer的CSV读取或服务器依赖。
- RTX4060 Laptop GPU上的两项小CUDA单元检查通过：
  FP16权重/值保真、AMP反向、独立索引/梯度、zero-start和实际更新。
- 并行控制器4个模拟测试通过：跨平台manifest路径、两侧屏障、命令/目录隔离、
  一项正式任务失败后等待另一项、不强杀另一项。
- 全部Python AST与manifest检查通过；shell脚本按LF打包。

参数：完整模型9,847,202；相对E40各增加8,778。
本地THOP2.0.20：E50 40.215296 / E51 40.2218496 GFLOPs640。
不同THOP的GN/Upsample计数有差异，检查按原/新同政策动态核对，
不硬编码本地总数；功能性邻域混合另列估计，不用FLOPs代替速度。

没有执行服务器32GiB、真实batch16/640、双开完整batch32/548图smoke。
没有执行远程200轮。没有在本地发CUDA服务器smoke或并行通过凭据。
本地没有Linux/bash运行时；控制器进程/信号行为已模拟验证，
实际Linux入口、真实数据与双开峰值由服务器check/smoke核实。
工程通过不是AP提升、因果根因或论文录用保证。
