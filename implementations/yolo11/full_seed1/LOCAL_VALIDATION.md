# 本地工程验证记录（2026-10-09，v2精度修复 + v3合成SGD审计）

Windows / Python3.9.25 / torch2.6.0+cu124 / RTX4060 Laptop。
服务器目标仍为 Python3.10 / torch2.5.1+cu121 / Ultralytics8.3.0 / vGPU32GB。

本地已执行：

- 12 项回归测试：父代码逐文件 SHA 对齐、公共 E58 参数/RNG/通道/参数量、
  seed1 的 E1/E40/E58 协议与权重 SHA、独立编码值/梯度 oracle、显存 bytes/GiB、
  smoke 凭据拒绝、官方参考复现拒绝、关机脚本分派和安全拒绝；新增极小投影的 AMP 梯度/实际更新
  对照，以及 half/bfloat16/保存重载的 FP32 权重保真。
  v3新增独立SGD舍入/动量检查：CPU与CUDA均拒绝漏优化器、未执行可表示更新和零梯度冒充sub-ULP。
- 完整 CPU 工程检查：方形/矩形初始化 raw/decoded 与 E1 逐值相等，
  原主损失及公共梯度逐值相等；全部 52 参数组的真实梯度和更新，在合成实际检测 loss 下
  3 次 optimizer step 内出现（不是只检查 backward）。
- CUDA 小样本：同上，加半精度输入/输出及混合 FP32 新侧支数值对照、实际 AMP 溢出跳步后有限更新、
  CUDA TAL 原始/分块一致性；峰值 reserved 2.174GiB，仅此小样本，不代表真实 batch16/640。
- 保存/重载/融合及新进程导入、标准 Predictor、原官方 scorer 单元、矩形坐标/NMS 非原地共享修改检查。
- 原标准 Trainer 两轮四张合成图训练、原 SGD 参数覆盖、EMA/best/last 保存重载、
  自定义检测头保留，以及四图矩形预测/原官方 ignore 评分路径集成。
- Bash 启动脚本语法检查；全部 Python 源文件及最终 zip/manifest 校验。

v2 新增 AMP 冷启动：不经 FP32 预热，直接从原始嵌套零初始化在 CUDA 上运行原检测 loss。
合成覆盖样本下 18 次尝试、8 次有限更新，全部52组有梯度和真实参数变化。
GT 尺寸用于覆盖初始 DFL/TAL 的多个尺度，不声称等同 VisDrone 分布；没有修改 loss/assigner。

精度修复对照：FP16 输入、投影权重 1e-9 时，v1 同权重路径内层梯度全为零，
v2 三组内层梯度与独立 FP32 对照相符并发生实际 SGD 更新。
标准 Trainer 四步合成集成中，14个原/新投影在 live model 和 best/last EMA 中均非零，
新增 P2 投影的极小幅值不再因 EMA.half() 丢失。父模型参数仍按原 half 保存，仅149,472个新参数保留FP32。
v1 的半精度下溢审查不足已在 FIX_V2.md 明确记录，不以原本“记录即可”的处理继续运行。

服务器v2冷启动合成检查遇到有梯度但P2 router未可见变化；v3不因此改结构/提高学习率。
对每步实际SGD与独立FP32参考逐值核对，只有非零且所有更新严格小于0.5 ULP的组才可分类。
这是合成检查的数值解释，不计作实际更新，也不改变真实服务器smoke必须52/52实际变化的要求。

实测参数/同工具预算：

```text
参数（未 fuse）      11,173,112
增强头参数          1,494,144
相对 E58 新增       149,472
E1 profile@640      34.337536 GFLOPs
E58 profile@640     54.8438784 GFLOPs
E60 profile@640     59.5993344 GFLOPs
差值                4.755456 GFLOPs（同工具，2MAC）
```

报告路径（不作为服务器凭据随包伪造）：

```text
tmp/e60_local_cpu_20261009_154217_308979/local_check.json
tmp/e60_local_cuda0_20261009_154427_731228/local_check.json
tmp/e60_local_cuda0_20261009_155432_940036/local_check.json（最终版，含非零新投影的独立半精度检查）
tmp/e60_local_integration_20261009_155118_598707/integration.json
tmp/e60_local_cpu_20261009_161023_970641/local_check.json（v2）
tmp/e60_local_cuda0_20261009_161108_899647/local_check.json（v2，未含追加冷启动）
tmp/e60_local_integration_20261009_161215_254630/integration.json（v2，EMA保真）
tmp/e60_local_cuda0_20261009_162046_807277/local_check.json（v2，含冷启动AMP及52组实际更新）
tmp/e60_local_cuda0_20261009_165001_423850/local_check.json（v3，所有SGD step独立参考一致；本机52组实际变化）
```

未执行：6471 张 VisDrone 的真实训练 batch16、548 图 batch32 完整真实服务器显存验证、
200 轮训练、完整官方评分、远程连接或真实关机。关机所有测试的 runner 均为 mock。
本地临时 pandas 仅补足验证依赖，不修改服务器、原 Ultralytics 源码或训练算法。
