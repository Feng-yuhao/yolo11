# 本地验证记录 / 2026-09-29

Windows CPU，Python3.9.25 / torch2.6.0+cu124 / 固定Ultralytics8.3.0。
本地CPU没有执行服务器CUDA、真实batch16/640 smoke或正式200轮训练。

CPU check已通过：初始方形/矩形raw及decoded输出和E1严格相等；
主检测loss和原参数梯度严格相等；14组新增参数三次主loss SGD获得实际更新。
独立手工oracle按不同任务和通道选择不同深度，验证softmax轴、展平顺序与残差塔输入。
输入不被原地修改、不detach；P4/P5检测头输入和输出旁路验证。
空GT、torch保存重载、Conv/BN fuse、新进程重载、标准Predictor验证通过。
原官方评估/坐标单测和TAL FP32/FP16案例及真实模型loss/gradient/update对照通过。

实测未融合参数9,878,648，新增199,680；同工具640 FLOPs39.5405824G，
E1为34.337536G，约增加15.15%。参数与FLOPs不等于实际训练显存或速度。
CPU报告：E41_CHECK_20260929_214734_035420。

标准Trainer集成已通过：4张合成图，160px/batch2/2epoch，完整优化器参数覆盖、
EMA、验证、best/last保存重载，以及4图矩形raw/最终预测与官方评分调用。
此2epoch是工程fixture，不是用户实验，不复用权重、不生成CUDA门禁。
沙箱现有pandas依赖阻断read_csv后，已获准沙箱外重跑通过，无服务器依赖修改。
报告：E41_LOCAL-INTEGRATION_20260929_215058_362276。

队列测试使用模拟/proc、模拟nvidia-smi和模拟子进程，不启动任何真实GPU任务。
覆盖E40训练失败、评分失败、没有旧锁、fork workers、多个根进程、僵尸/PID复用、
两次空闲后单次启动、旧锁仍占用、启动前任务重新出现继续等待、过期门禁、
等待期间代码改变、已有运行拒绝覆盖、GPU状态未知、超时、E41自身失败记录、
smoke分配上限和异常后的恢复。

仍须在服务器分别执行CUDA check和真实smoke。通过本地CPU不保证CUDA通过或AP提升。
E40失败允许接续是用户本次明确授权的策略，错误仍保留，不伪造成功或强杀E40。

发布修正：第一版ZIP的手动打包名单漏掉p2_tal_chunked.py，虽然本地文件存在，服务器导入失败。
v2改为从包目录枚举全部发布文件；增加独立必需文件清单，并在torch/CUDA导入前强制校验。
test_release.py从实际ZIP解压到临时目录，在新进程导入依赖并检查来源；
另在一次性临时副本移除p2_tal_chunked.py，检查能否提前明确报错。
