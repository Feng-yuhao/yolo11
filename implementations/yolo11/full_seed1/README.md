# E60a：E58 内部的分组残差编码（只做这一项）

母体固定为 **E58 四尺度 W96**。P2/P3 的 Z1、Z2、Z3 各增加一个分组残差编码器；
Z0、原始四深度生成链、逐位置空间路由、P4/P5、原 Detect tower / TAL / loss 全部保留。
本实验不同时扩大通道、不加新 channel gate、不加载 E58 best 微调、不用 test-dev 筛结构。

**v2 精度修复**：新增的六个分组残差编码器采用 FP32 参数/内部计算，避免嵌套零初始化
投影在 AMP 下舍入为零、阻断内层梯度。残差加回前恢复原特征 dtype；原 E58 编码链、
空间 softmax、检测塔与全网 AMP 不变。EMA.half() 时只保护这六个新编码器的 FP32 权重。
这是工程精度策略修复，不是更换结构/初始化/学习率，也不能据旧 smoke 判定性能失败。
详见 `FIX_V2.md`；真实显存和每轮速度必须重新测量，不复用 v1 的耗时估计。

**v3 合成检查修复**：只读核对每个真实SGD step的参数/动量参考及FP32舍入。
合成冷启动中的非零但低于0.5 ULP、理论就不能改变FP32值的更新会明确分类，不冒充真实变化。
模型和正式训练与v2相同；真实数据smoke依旧必须52/52实际更新，不能用合成分类代替。
详见 `FIX_V3.md`；check现在额外保存 `cold_start_amp_audit.json`，失败也保留最终组审计。

## 服务器运行

上传此 zip 到 `/root/autodl-tmp/`，解压一次：

```bash
cd /root/autodl-tmp
unzip -o E60a_E58_GroupResidualEncoding_AutoDL_v3.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E60a_E58_GroupResidualEncoding_AutoDL/run.sh check
bash /root/autodl-tmp/E60a_E58_GroupResidualEncoding_AutoDL/run.sh smoke
bash /root/autodl-tmp/E60a_E58_GroupResidualEncoding_AutoDL/run.sh verify-gates
```

等 **CUDA CHECK PASSED、CUDA SMOKE PASSED、RECEIPTS VALID** 后正式启动：

```bash
nohup bash /root/autodl-tmp/E60a_E58_GroupResidualEncoding_AutoDL/run.sh run --auto-shutdown > /root/autodl-tmp/yolo11_project/launcher_e60.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e60.log
```

也可以不分别手动做门禁，直接后台串联 check → smoke → verify → 正式训练 → 官方评分：

```bash
nohup bash /root/autodl-tmp/E60a_E58_GroupResidualEncoding_AutoDL/run.sh all --auto-shutdown > /root/autodl-tmp/yolo11_project/launcher_e60.log 2>&1 &
```

只选一种启动方式，不要同时启动 all 与 run。`run` 不重复 smoke，但必须验证本轮凭据。
尾随日志的 Ctrl+C 只退出 tail；不要对前台 check/smoke 用 Ctrl+C。
默认不关机；上面的 `--auto-shutdown` 显式开启关机。成功或失败先保存状态，确认无其他
训练/评估任务且 GPU 已释放后倒计时 60 秒。随时取消关机：

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_E60
```

存在这个文件时不会自动关机。没有 shebang 的 AutoDL `/usr/bin/shutdown` 经 `/bin/bash`
调用，修复之前 Errno 8；命令返回成功也不证明平台计费已停止，仍应检查 AutoDL 控制台。

## 训练协议与门禁

- seed1，640，batch16，workers8，SGD，200 轮 cosine，close_mosaic10，原 E1 预训练迁移。
- E58 公共新分支保持 40001 / 57001 初始化；E60 新编码隔离初始化 60001。
- **单开**，smoke 与 formal 同 28GiB allocator 上限；不自动降 batch 或跳过门禁。
- smoke 至少 8 次真实成功 AMP 更新（overflow skip 不计成功），逐项确认 52 个参数组更新。
- 训练模型和 optimizer 驻留时进行 EMA-half batch32 全 548 图验证，检查 loss / NMS / 峰值显存。
- CPU 或本机小 CUDA 验证不生成服务器凭据，不能替代上述真实 smoke。
- 200 轮跑完自动官方评分，FP32 / imgsz640 / batch4 / conf0.001 / NMS0.7 / max_det500。

零输出初始化会延迟上游梯度：第一步只应要求原 E58 输出投影有非零梯度，
新增编码投影及其内层不要求第一步更新；随后必须出现实际梯度和参数变化。
不能把正常 AMP 动态缩放的 overflow skip 当成训练成功，也不能遇到 skip 就认定结构错误。
v2 每次 smoke 打印梯度/更新覆盖组数，并在 `E60_SMOKE_时间/branch_gradient_audit.json`
保存 52 组覆盖、成功更新的缩放梯度最大值和新投影 FP32/FP16 非零数量。
失败时也保留该文件，不把它当成成功凭据。代码更新后必须重新 check、smoke；不手工修改 gate。

## 必须保留的现有文件

代码使用已有报告核对协议与同 seed 对照，不重新训练 E1/E40/E58。
这些目录不要在清理时删除；缺失时恢复本地备份，不绕过检查：

```text
yolo11_project/weights/yolo11s.pt
yolo11_project/comparison_reports/e1_yolo11s_p2add_img640_seed1_20260926_102750_718906/
yolo11_project/comparison_reports/E40_RUN_20260929_205658_507620/
yolo11_project/comparison_reports/E58_RUN_20261008_001519_559728/
yolo11_project/runs/e1_yolo11s_p2add_img640_seed1/weights/best.pt
yolo11_project/runs/e40a_spatial_task_head_img640_seed1_200e/weights/best.pt
yolo11_project/runs/e58a_e57_allscale_w96_img640_seed1_200e/weights/best.pt
/root/autodl-tmp/yolo11_src/
/root/autodl-tmp/project/VisDrone2019/
```

报告至少包含各自 training.json / comparison.json（E1 为 metrics.json），以及
area_predictions.json；E1 还需要 area_ground_truth.json。本包不包含预训练权重或数据。

## 结果及补评分

```text
yolo11_project/runs/e60a_e58_group_residual_encoding_img640_seed1_200e/
yolo11_project/comparison_reports/E60_RUN_时间/
  training.json / transfer.json / actual_source.zip
  training_completion.json
  official_metrics.json / comparison.json / REPORT.md
  area_predictions.json / gt_rows.json / completion.json
yolo11_project/logs/latest_e60_controller.json
yolo11_project/logs/latest_e60_run.json
```

实时日志包含每轮时间、累计时间、剩余 ETA、8 个原任务投影与 6 个新残差投影的权重幅值。
训练完成与评分完成分开记录；评分失败不重训，可独立补评分：

```bash
bash /root/autodl-tmp/E60a_E58_GroupResidualEncoding_AutoDL/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e60a_e58_group_residual_encoding_img640_seed1_200e/weights/best.pt
```

主比较是 E60 vs E58 seed1；E1/E40 是背景对照。三组缓存预测会用同一原始标注/官方评分器
重新评分验证复现。资源筛选门槛：对 E58 的 Official AP ≥ +0.20pp 且 AP75 不降。
这不是显著性门槛或涨点保证；达标再讨论配对多 seed。raw coverage 不是 AP-small。

## 核心结构

原 Z96 划为四组 U24：V1=U1，V2=K2(U2)，V3=K3(U3+V2)，V4=K4(U4+V3)。
三个 K 不共享，各为 Conv3×3 + GN6 + SiLU。
Z'=Z+P1×1([V1,V2,V3,V4])，只有 P 输出投影零初始化。
每个尺度的原始 Z1/Z2/Z3 都先生成，再独立修饰读出副本；不会递归改写原 E58 链。
原空间路由对增强库 `[Z0,Z1',Z2',Z3']` 做逐位置深度 softmax，然后进入原分类/回归塔。

新增 149,472 参数，卷积额外 4.755456 GFLOPs@640（2MAC；不含 norm/激活/加法）。
模型总参数 11,173,112。实际同工具 profile 由 check 报告给出，训练速度以服务器实测为准。

组内层级连接借鉴 [Res2Net 原论文](https://arxiv.org/abs/1904.01169)。这不是首创分组卷积，
也不是复现整个 Res2Net；研究假设是保持原库和空间路由的内部残差编码能否带来额外收益。
E58 加宽有益不能证明压缩是根因，也不能保证继续增强编码有效。
