# 本地工程验证记录

日期：2026-10-04。CPU / 小CUDA环境：Python 3.9.25、PyTorch 2.6.0+cu124、
工作区固定 Ultralytics 8.3.0；小CUDA为 RTX 4060 Laptop 8GiB。
服务器 PyTorch/GPU 不同，以下结果不能替代服务器 CUDA check / 真实 smoke。

已执行并通过：

- 独立索引 oracle：3×3 九偏移、四细网格位置、复制边界、脉冲、常数、矩形、最小网格。
- 独立 coarse/kernel 梯度比较；零 gain 恒等；浅层描述确实生成位置相关核。
- FP32、stored FP16/BF16 值保真与小CUDA AMP；原始参数不被原地修改。
- 实际模型路由 model16/model2→model23→Concat24；公共E40初始化及训练RNG保持。
- 初始主损失/公共梯度；新增路径的检测损失梯度及实际优化更新。
- 激活原E40权重的恒等夹具；保存、新进程重载、fuse、标准Predictor、半精度EMA。
- 原TAL逐图分块等价、官方评分单元和矩形坐标/NMS单元。
- 标准训练器四张合成图的两轮训练、EMA、best/last保存、重载与官方预测管线。
- 小CUDA从真实零 gain 初始化，4×320合成批次，8次成功AMP更新；全部19组获得检测损失梯度。
- 官方报告文件写入、E1/E40对照连接的mock测试；mock数字不是实验精度。

工作区原始记录（不打入服务器门禁凭据）：

```
yolo11_project/comparison_reports/E49_CHECK_20261004_150732_644422/check.json
tmp/e49_small_local_cuda_20261004_150744/local_cuda.json
yolo11_project/comparison_reports/E49_LOCAL-INTEGRATION_20261004_151343_004202/integration.json
tmp/e49_synthetic_zero_start_cuda.json
tmp/e49_mock_eval_wiring_20261004_151338/MOCK_NOT_EXPERIMENT.json
```

合成check先证明零起点gain的loss更新，再以可丢弃gain=0.1夹具检查完整核路径；
该夹具不进入正式初始化。上述小CUDA零起点测试未使用这个激活夹具。
server smoke仍须从gain=0开始，以真实VisDrone数据观察loss梯度、更新、显存和完整验证。

参数核对：新增重建8778；包含完整E40共184218新增参数；未融合总参数9863186。
本地 THOP 2.0.20 的 GFLOPs@640 为40.2116096；它不是跨统计器版本固定常数。
与E40的比较必须使用同一安装版本、同一统计口径。
重组/softmax/复制操作未全部计入THOP，另列算术估计，不宣称无额外耗时。

未执行：服务器32GiB上的真实batch16/640 smoke、548图完整CUDA验证、200轮训练。
没有本实验的精度结果，也不能保证涨点、完全无bug或论文录用。

## V2 统计器兼容修复

服务器检查差值为0.004096G；本地通过公共THOP custom_ops切换两层
GroupNorm计数，精确复现旧公式的这个差值。四种GroupNorm/nearest计数
组合均已回归验证：新公式与全模型统计一致，卷积独立形状算术仍为0.2916352G。
不修改THOP安装、不改变模型算子、不放宽全模型1e-5G容差。
服务器实际叶子计数记录由下一次check写入check.json；本地测试不能冒充服务器通过。

V2重新执行通过的完整CPU与小CUDA证据：

```
yolo11_project/comparison_reports/E49_CHECK_20261004_153624_124593/check.json
tmp/e49_small_local_cuda_20261004_153635/local_cuda.json
```

7项原目录单元通过；FixV2 ZIP干净解压测试通过（6项执行通过，
1项原目录源码复制对照因解压位置没有历史父包而跳过，该项在原目录已通过）。
另外逐个比较原ZIP中的Python文件，仅checks.py/test_package.py改变，
其他原Python文件内容全部保持；新增profiler_accounting.py与修复说明。
