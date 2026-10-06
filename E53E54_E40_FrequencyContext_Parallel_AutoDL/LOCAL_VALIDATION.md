# 本地验证记录（2026-10-05）

环境：Windows / Python3.9.25 / PyTorch2.6.0+cu124 / Ultralytics8.3.0。服务器PyTorch2.5.1环境另需check，不冒充同一环境。

两组均已执行并通过：

- 独立Haar正变换、伴随逆变换、重建及输入梯度；不同halo边界输出对照；所有频带保留全网格。
- 上下文near/far选择独立对照、softmax求和、纯旁路仍保留local、23x23名义支持范围。
- 分支初始严格恒等、仅末输出投影首步获得梯度、第二步内部参数非零有限梯度和更新；无原地修改输入。
- 源码/原预训练映射、E40公共40001初始化及训练RNG不变；真图路由、初始化预测/原loss/公共梯度对照。
- 小尺寸真实YOLO loss/空GT、两组本地CUDA/AMP更新、EMA-half、fuse、fresh-process和标准Predictor。
- 真实640形状的全模型THOP差额与独立torch.profiler卷积计数吻合：E53 41.04324096G，E54 40.1385728G，同口径E40 39.926528G。Haar额外算术和memory不当作免费。
- 两组各2轮四图合成数据的标准Trainer训练→EMA→best/last保存→reload→预测→官方评分集成通过。不是20轮性能probe，不给性能结论。
- 4项控制器mock测试：失败一组不杀正常另一组、子进程独立、双端barrier、签名路径可迁移。

完整CPU报告：E53_CHECK_20261005_231804_719026、E54_CHECK_20261005_231902_182597。
小CUDA结果：tmp/e53e54_local_cuda_e53/result.json、tmp/e53e54_local_cuda_e54/result.json。
Trainer集成：E53_LOCAL-INTEGRATION_20261005_232742_389402、E54_LOCAL-INTEGRATION_20261005_233206_950561。

第一次本地E54单元检测到参数算术常量多写100，已修正为16665并重跑通过；第一次本地合成Trainer因缓存pandas目录的沙箱读取权限失败，获准读取后两组完整集成通过。未修改服务器环境。

**尚未执行：服务器真实6471图训练数据上的双进程batch16/640、至少8次成功更新、EMA-half batch32全548图验证smoke。**
README保留必需门禁；仅这些本地结果不能承诺服务器显存、时间或精度提升。不能跳过check/smoke直接开训练。
