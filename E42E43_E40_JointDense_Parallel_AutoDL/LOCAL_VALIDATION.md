# 发布前验证：2026-10-01

环境：Windows CPU，Python3.9.25，torch2.6.0+cu124，工作区Ultralytics8.3.0。
AutoDL目标环境是torch2.5.1/CUDA；二者不能视为同一执行环境。
未连接/修改服务器，没有启动正式训练。

## 已执行

- E42完整CPU check：`E42_CHECK_20261001_025514_452713`。
- E43完整CPU check：`E43_CHECK_20261001_025526_092888`。
- 独立公式/维度/任务oracle、联合路由完整与分块前向/反向、极端相反logits、C=0退化E40。
- 原始E40两个block公共参数严格保持；新增模块在单独CPU RNG范围构造；训练CPU/CUDA RNG检查。
- E1方形/矩形raw和decode预测、实际v8DetectionLoss、原始参数初始梯度严格一致。
- 新分支18组非零有限检测损失梯度与SGD参数变化分别检查；前层延迟梯度不能判死路径。
  工程check固定原网络、更新新头来保持小合成batch的TAL正样本稳定；真实Trainer使用全参数优化器。
  更新包含SGD momentum/weight decay，但不能只靠weight decay充当非零loss梯度证据。
- 实际cv2/cv3输入、P4/P5头入口旁路、空GT、保存/重载/fuse、新进程重载、标准Predictor通过。
- D27官方评分单元检查、D38原图几何检查、TAL FP32/FP16及P2真实loss/gradient/update等价检查通过。
- E42/E43两套各四张合成图、160/batch2/2epoch标准DetectionTrainer集成通过，
  覆盖完整优化器参数注册、EMA、验证、best/last、矩形预测、原图坐标和官方评分入口。
  结果分别在 `E42_LOCAL-INTEGRATION_20261001_025332_876503` 和
  `E43_LOCAL-INTEGRATION_20261001_025134_611499`。这是工程fixture，不是性能probe，不用于淘汰。
  Trainer测试获准在沙箱外使用现有pandas依赖；未安装服务器依赖。
- 五项release测试通过，包括两套纯FP16 block与FP32参考、固定正式协议、源文件保持、控制器不杀peer、Python语法。
- E40参考报告/权重指纹本地核验通过，Official AP 26.2224523431。

## v1发布时同工具完整模型复杂度

|模型|参数量（未fuse）|GFLOPs640（卷积统计工具）|
|---|---:|---:|
|E1|9,678,968|34.337536|
|E40原始|9,854,408|39.926528|
|E42a|9,892,040|39.9686144|
|E43a|9,877,832|40.663808|

该FLOPs工具未完整统计softmax、加权、内存访问，不能拿E42几乎不增加GFLOPs断言没有速度/显存开销。
E43相对E40新增卷积0.73728G，与独立手算一致。

## 未执行

服务器CUDA完整check、AMP真实batch16八批优化、双方548图batch32验证、并行峰值、Linux真实flock/进程调度，
新模型完整200轮结果及多seed效果均未执行，必须依README在服务器完成。
每进程14GiB上限保护分配器，不涵盖全部外部CUDA内存，也不保证未知密集batch不OOM。
不能承诺代码无任何bug、模型一定涨点、创新性充分或论文录用。

评分失败保留训练结果，使用eval独立补评，不重新训练。两项正式作业隔离，一项失败不杀另一项。

## v2 smoke计数修复回归

2026-10-01新增执行10项本地release测试，全部通过：包括用户日志6次跳步/2次更新的重放、
32次持续溢出必须失败、真实CPU GradScaler/FP16跳步后成功更新、有限更新证据检查。
两项模型/梯度检查/评测/验证代码与v1原ZIP逐字节一致；正式config/build_model/trainer_class/train函数AST严格一致。
新增工程smoke最多32真实批次，要求至少8次成功更新，并保持全部18组梯度/参数更新及完整验证门禁。
正式训练不降低init_scale，不改变AMP/优化器/学习率/batch/seed。
v2真实服务器CUDA smoke尚未执行；不能把CPU模拟和之前2次真实CUDA更新当成完整双开通过。

## v3复审与本地CUDA补充

用户服务器v2 smoke失败见 V3_AUDIT.md：E42有24次有效更新，但P2通道MLP前层无loss梯度；
E43是同步barrier连带失败。v2计数修复不能解决核心AMP数值隐患。

v3把新通道MLP置于FP32精度区间，不改全局AMP、正式训练超参数/初始化，也不删除梯度门禁。
本地RTX4060/torch2.6上旧故障重现；修复后weak-gradient/value oracle通过，全部11项release回归通过。
两套完整本地CUDA审核及640/batch2全模型YOLO loss/AMP/SGD分阶段更新通过，
E42/E43分别17/18次有限更新，18组非零loss梯度和真实参数更新齐全。
保存/重载/fuse、FP16保真、Predictor、TAL真实FP32/AMP与评分入口检查通过。
目录：`tmp/e42e43_v3_e42_CUDA_20261001_035809`、`tmp/e42e43_v3_e43_CUDA_20261001_040013`。
两套标准Trainer/EMA/checkpoint/官方入口的四图合成集成也已在v3上重新通过：
`E42_LOCAL-INTEGRATION_20261001_041226_254714`、`E43_LOCAL-INTEGRATION_20261001_041236_050065`。

FLOPs补充边界：v3新通道MLP以functional conv调用，现有模块hook计数工具不再捕获它，
因此E42的get_flops静态值39.926528不是完整算子总量，也不能声称与E40完全无新增计算。
通道MLP双尺度两层1×1实际另有73,728卷积FLOPs/图，池化/softmax/加权/访存仍未完整计数。
参数量不变。论文复杂度应统一统计工具并明确算子边界，不能用这个不完整hook值虚构效率改善。

上述本地CUDA补充不能替代AutoDL/torch2.5.1的batch16真实双开、548图batch32验证smoke。
