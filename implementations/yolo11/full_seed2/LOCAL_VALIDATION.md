# 本地交付验证（2026-10-11）

本机 Python3.9.25 / PyTorch2.6.0+cu124 / RTX4060 Laptop；服务器仍须运行自己的 CUDA check 和真实 smoke。
未连接服务器、未启动远程训练；关机测试全部模拟，未执行真实关机。

## 已通过

1. 15 项回归测试全部通过：冻结源码、初始化/RNG、父参数映射、52 组、参数数量、协议、
   分组运算 oracle、FP16 极小梯度保护、EMA 保存、SGD 舍入、旧/错误 smoke 拒绝、显存字节单位、
   官方对照复现、配对 seed1/2 汇总及关机的忙碌/取消/无 shebang/失败分支。
2. 读取实际 E60 seed1 的 actual_source.zip，14 个冻结实现文件逐字相同（统一 LF 换行）。
   当前本地 E1/E40/E58 seed2 与 E60 seed1 报告、权重 SHA 和训练协议全部通过检查。
3. 本地完整 CUDA 工程 check：方形/矩形 raw/decoded 初始预测精确等于对应 E1；
   原损失/公共梯度检查、52 组更新、空 GT、AMP、标准 Predictor、新进程保存重载/fuse、
   官方评分单测及 TAL FP32/FP16、真实检测损失两步更新等价全部通过。
   冷启动 AMP 共 16 次尝试达到 8 次成功更新，52/52 有梯度且 52/52 有实际参数变化；
   无需用 sub-ULP 分类替代实际变化。GPU峰值 reserved 2.174GiB（小输入，不是正式显存估计）。
4. 2 轮 CPU 合成集标准 Trainer 测试通过：真实 forward/backward/optimizer、EMA、
   best/last 保存重载，新增极小 FP32 权重未在 EMA-half checkpoint 丢失；
   独立四图矩形官方预测/评分入口可用，checkpoint train_args.seed=2 检查通过。
5. 配对汇总保留负收益 seed，mock 报告测试验证 candidate=seed2、E60 两 seed 均值和输出文件。
   mock 指标只测试流程，不是性能实验。

## 结构及精度

11,173,112 参数，新增检测头支路 1,494,144 参数；52 组学习检查。
P2/P3 每层 3 个分组读出残差编码，P4/P5 无此编码。
同工具本地640 GFLOPs=59.5993344，E58=54.8438784，E1=34.337536；与冻结 E60 对齐。
原父路径 AMP 不变；新增六个编码器保持 FP32 参数/内部计算/EMA，输出加回前恢复原表示 dtype。

## 验证边界

本地测试不能证明服务器 640/b16 完整数据无 OOM，不能证明性能提升或统计显著。
启动 all 必须重新进行服务器 check、至少 8 次真实 AMP 成功更新及全548图 EMA-half b32 验证。
Linux shell/controller 交由服务器执行，本地未声称执行 Linux 自动关机或正式200轮训练。

本地证据目录（未打包权重/回执）：

```text
tmp/e60_seed2_local_cuda0_20261011_113133_350054/local_check.json
tmp/e60_seed2_local_integration_20261011_113431_920481/integration.json
```
