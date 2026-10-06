# 本地验证及不能声称的事项

2026-09-29，Windows CPU，Python3.9 / torch2.6.0+cu124 / 本仓库固定Ultralytics8.3.0。
现有测试依赖来自 tmp/e37_test_deps；Trainer集成测试因沙箱阻挡现有pandas依赖，获准在沙箱外重跑通过。
未安装或修改服务器依赖；未连接服务器、未停止E39、未执行CUDA smoke或正式训练。

已执行：

1. `run_e40.py --mode check --device cpu`。
   - 独立手算四深度softmax加权、分类/回归任务索引、输入不被原地修改、输入梯度未detach。
   - 初始方形/矩形完整raw/decoded预测与E1严格相等；实际v8DetectionLoss及原参数梯度严格相等。
   - 三次真实主检测损失SGD：两个尺度14组分支参数获得有限非零梯度和真实更新。
   - hooks捕获真实cv2/cv3输入，验证reg/cls路由；P4/P5头部旁路输出严格不变。
   - 空GT、保存重载、fuse、新进程重载、标准Predictor全部通过。
   - 原D27官方评分与D38几何单元测试通过；TAL FP32/FP16匹配案例和实际P2-s模型损失/梯度/更新对照通过。
   - 参数9,854,408，其中新增175,440；640同工具FLOPs39.926528 vs E1 34.337536，+16.28%。
2. `run_e40.py --mode local-integration --device cpu`。
   - 四张合成图、160px/batch2/2epochs是工程fixture，不是性能探针。
   - 真正标准DetectionTrainer、优化器全参数覆盖、EMA、验证、best/last保存与重载通过。
   - 四图矩形Predictor、原图坐标、raw P2/P3及final IoU范围和冻结官方评分路径通过。
   - 正式入口依旧200epoch/640/batch16/3轮warmup，不复用集成测试权重。
3. `test_safety.py`：14项通过。
   - 24个fork worker不被误认成24个训练；独立多任务保留；PID复用与僵尸检测。
   - E39失败、早停120轮、非完整548图评估、非有限指标、权重哈希变化、缺报告均阻止启动。
   - 没有活进程却未完成时拒绝；多报告要求显式选择；GPU未空闲不启动。
   - 模拟控制器只有连续两次GPU空闲才启动一次，并在启动前重新验门禁。
   - smoke显存14GiB上限/20.5GiB启动线/6GiB余量及异常恢复通过模拟CUDA接口检查。

未执行、不能声称已通过：服务器torch2.5/CUDA/AMP真实批训练，真实显存峰值，Linux真实flock与GPU队列接续，
完整548图新模型评测、多seed性能、正式200轮可用性、涨点与论文录用。
这些必须依照README的独立check/smoke/queue步骤实际验证。

重要边界：原梯度在零输出初始化时相同，不保证原参数的更新在加入新增梯度后的全局梯度裁剪中仍完全相同。
正常联合训练就是本实验干预，不声称完整因果隔离。CPU检查不能冒充服务器CUDA回执。
