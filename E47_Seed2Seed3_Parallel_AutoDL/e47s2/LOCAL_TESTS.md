# 实际本地验证记录（2026-10-02）

环境：Windows，Python3.9.25，torch2.6.0+cu124，Ultralytics8.3.0固定源码。
CPU完整工程测试；RTX4060 8GiB仅运行小尺寸CUDA数学/梯度/AMP/FP16单元。
未在AutoDL torch2.5.1上运行，没有执行真实6471图训练或batch16/640显存smoke。

已通过：

- 对比基正交且零和；独立“先逐深度投影再求和”oracle对照生产“先对比再投影”。
- 同时激活原E40和新增读出，在13×19、8×8、1×1特征上核对数学输出。
- 非零原E40输出上的零读出恒等；原输入和旧参数梯度数值一致。
- 分类/回归独立输出与直接梯度；无输入原地修改，无detach。
- 每个新增读出有非零有限损失梯度及真实SGD更新（无weight decay/momentum伪更新）。
- CPU bfloat16、CUDA float16和autocast值/梯度；参数dtype无原地变化。
- 原E1/E40参数逐张量初始化一致，全局RNG保留；初始矩形/方形raw/decoded严格一致。
- 原YOLO损失/分项严格一致；基础梯度数值一致；26组参数分阶段梯度与实际更新；空标签。
- cv2回归/cv3分类实际入口；P4/P5旁路；保存、重载、fuse、独立新进程和标准Predictor。
- 原TAL等价测试及真实模型损失/梯度/两步更新；官方评分器及坐标/NMS单元。
- 原生DetectionTrainer在4张合成图上2轮集成：完整optimizer覆盖、EMA、best/last、半精度存储后float重载、矩形预测和官方评分接口。
- 同Conv/GN统计参数9,928,136、42.285824GFLOPs；新增73,728参数和2.359296卷积GFLOPs。
- Bash入口语法验证，不带并行/队列/自动关机逻辑。

证据在本地主工程comparison_reports/E47_CHECK_*/check.json、
E47_LOCAL-INTEGRATION_20261002_183803_018389/integration.json，以及tmp/e47_unit_*.json。
打包前对最终代码重新运行CPU check与单元；服务器不复用这些CPU凭据。

开发中曾被完整检查捕获并修复：FP32 stateless调用干扰THOP卷积统计。
已改为FP32参数直接调用注册Conv，半精度参数才用stateless FP32投影；重新核对独立数学oracle与实测Conv统计。
最初合成集成遇到本地sandbox无法读取pandas依赖；授权读取已有依赖后原Trainer通过。
失败记录保留，不作为通过凭据。

边界：不宣称CUDA真实显存门禁通过、完整200轮/多seed有效或模块必然涨点。
