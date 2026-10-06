# E44/E45 v2 整包复核（2026-10-01）

## 已定位并修改

1. `routing_audit`将不同计算图的数学等价梯度要求为bitwise相等，CUDA报告最大绝对差1.46e-11/2.91e-11便中断。这是错误的检查要求，不能据此宣称训练模块有误。
   - raw/decoded输出、隐层初值、原参数初值、输入不变检查仍严格相等。
   - FP32梯度检查采用逐张量参考幅度：atol=64×FP32_eps×max(abs(reference))，rtol=8×FP32_eps；没有固定绝对误差地板。全零参考必须仍为零。
   - 每张量的实际差值/预算/bitwise标志写入check.json。shape/device/dtype、None和非有限梯度仍拒绝。
   - 主模型CPU公共梯度仍逐位对照；CUDA使用上述数值比较。
2. 将禁用TF32的严格数学上下文覆盖至模块路由和弱权重检查，原版这两项在上下文外。此修改仅作用检查，不改变正式训练的计算配置。
3. 启动锁从包目录移至项目目录，防止同一个实验从不同解压副本重复启动；支持--project及--project=参数。

## 增强的检查，不修改训练机制

- 激活新分支后，独立公式对照全部注册参数的loss梯度，不再仅比较输入梯度。
- 故意加入梯度偏移/尺度错误/NaN/None及全零污染，验证新误差检查会拒绝真实错误。
- 原交付ZIP与v2逐项AST/字节对照：linear_fp32、局部校准、两种模块forward/decompose、install、branch_groups、独立oracle不变；正式runner、Detect、原core、TAL、评分文件不变。
- 校准器仍在任务混合后；RDR仍在F2/F3前添加零初始化残差；P4/P5与原预测塔、loss、DFL、assignment及NMS不变。
- 保留零输出起点、分阶段梯度检查；不要求第一步所有上游参数非零更新，不把weight decay独自造成的更新算作loss可学习性证据。

## 审查覆盖及证据

| 范围 | 核查内容 |
|---|---|
| 核心模块 | 注册参数、任务顺序、depth softmax轴、校准/复用公式、非原地运算、dtype/device、无训练detach |
| 初始化 | 固定原E1/E40公共参数、独立新参数随机流、全局CPU/CUDA RNG保护 |
| 反向及AMP | 独立公式全部参数梯度、loss-only更新、2e-8弱权重、half输出、AMP overflow不计成功 |
| Trainer | optimizer参数覆盖、原loss/TAL、200轮日程、640/batch16/seed1、EMA/best/last及保存重载 |
| 预测评估 | fresh-process reload/fuse、标准矩形Predictor、原ignore评分与几何测试、E1基准复现条件 |
| 计算量 | 同一工具统计新Conv增量，E44+0.21504G/E45+0.73728G；不能据此预测实际速度 |
| 并行控制 | 独立日志/退出码、对方失败不杀同伴、项目级锁、同一次并行smoke凭据、stale签名拒绝、禁止覆盖正式run |

最终本地验证：

- `test_release.py` 10项全部通过；Bash入口`bash -n`通过。
- E44完整CPU check：`E44_CHECK_20261001_231729_119082`。
- E45完整CPU check：`E45_CHECK_20261001_231744_791972`。
- E44合成Trainer：`E44_LOCAL-INTEGRATION_20261001_231837_906041`。
- E45合成Trainer：`E45_LOCAL-INTEGRATION_20261001_231847_965684`。

环境为本地Windows/torch2.6.0/固定Ultralytics8.3.0/CPU。Trainer测试各4图、160px、batch2、2轮，只验证工程链路，没有精度结论。不连接服务器，不修改远程源码，不签发CUDA或真实smoke凭据。

## 必须保留的边界

目前未发现需要改变核心训练前向的错误，但不能承诺所有GPU/数据条件下没有其他问题，更不能保证提升。服务器torch2.5.1 CUDA、AMP、真实batch16和并行验证峰值仍需实测。新版代码签名已改变，旧门禁不可复用。

v2 ZIP使用新的`E44E45_E40_LCC_RDR_Parallel_AutoDL_v2`根目录，不覆盖旧包。先check两项通过，再同一次并行smoke两项exit=0；任何一项失败均不启动正式训练。
