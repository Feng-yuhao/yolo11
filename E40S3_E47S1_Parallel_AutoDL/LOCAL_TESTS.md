# 本地验证与执行边界

本包只交付 E40 seed3 与 E47 seed1，并不自动启动服务器训练。
两项均从原始预训练迁移开始；不复用 E40 best 微调，不重跑已有 E1 seed3。

已实际执行的验证：

- E47 独立深度读出公式对照、零初始化 E40 等价、原参数梯度对照、任务专属梯度、六组新读出参数的真实更新：CPU 通过。
- E47 完整 CPU check：E1 初始预测/主损失/梯度、26 组新增路径更新、空标注、训练过的 E1 序列化夹具、新进程 reload/fuse、标准预测器、TAL 等价与更新、官方评测与坐标单元通过。
- E47 小型 CUDA 单元：FP16/AMP 数值对照、读出梯度与参数更新通过。
- E40 seed3 完整 CPU check：初始预测、原损失与基础梯度对照、14 组参数更新、新进程序列化/fuse、TAL、评分/坐标单元通过；小型 CUDA 路由单元通过。
- 两项均通过原生 DetectionTrainer 的 4 图/2 轮合成集成：optimizer 覆盖、EMA、best/last、重载、矩形预测与官方评分接口；此结果不是正式 VisDrone 训练。
- 根控制器 17 项标准库测试通过：同次回执、陈旧/混用回执拒绝、三个 barrier 的真实双线程等待、失败退出、另一任务继续、控制器中断状态、资源边界。
- 三个 Bash 入口经 Git Bash 的 `bash -n` 实际语法检查通过。

这些检查不能替代服务器上的完整 CUDA check、真实 VisDrone batch16/640 更新和 batch32/548 图并行验证 smoke。
本地使用 Python 3.9.25 / PyTorch 2.6.0，服务器使用既有 yolo11 环境；只有服务器当前代码、数据、环境与 GPU 绑定的并行 smoke 才能签发正式运行凭据。

资源策略是每进程 13 GiB PyTorch allocator 上限，smoke 和 formal 相同。
上限不是显存足够或不会 OOM 的保证；CUDA context 等显存不属于 PyTorch allocator。
任一 smoke 失败都不能启动正式双开，也不自动降低 batch 或改变模型。

验证证据在本地主工程的 `E40S3_CHECK_20261002_222246_350467/check.json`、
`E40S3_LOCAL-INTEGRATION_20261002_222420_913098/integration.json`、
`E47P_LOCAL-INTEGRATION_20261002_222245_018828/integration.json`、
最终 `E47P_CHECK_20261002_222649_581446/check.json` 等报告中。
服务器不复用这些 CPU 检查凭据。

本地未保存完整 548 图原始 VisDrone annotations，未重算完整 E1 seed3 官方评分。
已实际核验其既有 predictions、GT mapping、best 权重的 SHA256；官方四项精确基准来自已归档报告，服务器正式评测将再次复现并严格核对。

本地合成集成第一次受到依赖目录 sandbox 读取限制，读取授权后两个原生 Trainer 均通过；失败记录保留，不能当作通过凭据。
交付 ZIP 不含检查凭据、权重、数据、缓存或本地生成的合成训练结果；打包时检查 manifest、路径、LF 行尾与 ZIP CRC。
