# E40a Spatial Task Head — E39 后接续正式训练

本包独立于 E39，不修改共享 Ultralytics 源码，也不读取 E39 权重。
父模型是原始预训练迁移得到的 E1-640 seed1，正式训练 **200 epochs / batch16 / workers8 / AMP**。
原 E1 的训练、Mosaic 关闭时点、TAL、CIoU/DFL/BCE、预测塔、框解码和 NMS 保留。
这是一项待验证的结构假设，不承诺涨点、论文录用或无 OOM。

## 改动是什么

`model.26` Detect 的 P2/P3 入口，各加入独立的多深度表征与空间任务路由：

```
原 P2/P3 → reduce 128→48 → 连续三个普通 3×3 卷积
              Z0             Z1 / Z2 / Z3
                       ↓ 保留四种深度
                每位置、每任务 softmax(depth=4)
                    ↓                 ↓
               分类加权表征       回归加权表征
               48→128 投影        48→128 投影
                    +原X              +原X
                  原cv3              原cv2
```

两个输出投影零初始化，初始 raw/decoded 预测、原检测损失及原参数梯度与 E1 一致。
只有首步输出投影获得非零梯度；输出离开零后，上游编码和路由获得梯度。
P4/P5 的直接检测路径不变，但共同训练可以间接改变其特征和预测。
不是 P1 注入、Neck 重写、后处理修框或额外监督。借鉴 TOOD 的任务分解思想；空间路由不是原创性证明，详见 PLAN.md。

同一工具、640 输入本地统计：9,854,408 参数、39.926528 GFLOPs；E1 34.337536 GFLOPs，增幅16.28%。
理论 profile 不等同于实际延迟，正式时长以服务器日志为准。

## 1. 上传压缩包，然后解压

把 `E40a_SpatialTaskHead_AutoDL.zip` 上传到 `/root/autodl-tmp/`。

```bash
cd /root/autodl-tmp
unzip -o E40a_SpatialTaskHead_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
```

不要覆盖正在训练的 E39 包，不要复制本包文件到 yolo11_src。
启动器使用 `/root/miniconda3/envs/yolo11/bin/python`，不依赖终端显示 base/yolo11。

## 2. 独立 check

```bash
bash /root/autodl-tmp/E40a_SpatialTaskHead_AutoDL/run_e40.sh check
```

看到 `E40 CUDA CHECK PASSED` 和 `E40 check FINISHED` 才继续。
检查实际通道路由、softmax 维度、原始初始化/损失/梯度、真实参数更新、空GT、CUDA AMP、
FP16推理、保存重载/fuse/新进程重载、标准Predictor、TAL等价性、官方评分单元测试、计算预算。
CPU check 不能替代这一步。输出实时显示，不是后台隐藏任务。

## 3. 独立 real CUDA smoke

```bash
bash /root/autodl-tmp/E40a_SpatialTaskHead_AutoDL/run_e40.sh smoke
```

使用真实 train split、batch16/640/AMP 跑 **8 个 batch**，实时打印 loss、显存与用时。
看到 `E40 CUDA SMOKE PASSED` 和 `E40 smoke FINISHED` 才继续。
它不是2轮、20轮探针，更不是正式200轮训练，不用 smoke AP 判断效果。

为了检验零初始化后的整条分支学习，smoke 专用优化器固定 SGD lr=.01、每 batch 更新；
**正式训练重新从原始初始化开始，仍使用 E1 的3轮 warmup、梯度累积和200轮余弦日程**。
此优化器测试不会改变正式权重/训练协议。14组参数都必须真实更新。

smoke 支持 E39 同时运行，但启动时至少需要 **20.5 GiB 当前空闲显存**：
本进程 PyTorch allocator 上限14GiB + 其他任务余量6GiB + 0.5GiB外部分配余量。
若空闲不足或有OOM，直接失败且不发通行回执；不自动减batch、不动E39。
这个上限只作用于smoke进程，不限制正式训练，不能保证其他任务任意突发峰值也安全。

check/smoke 回执绑定代码、基线协议、数据YAML原始字节哈希、预训练权重、GPU/torch环境。
改包/改训练参数后旧回执不能复用；不要求因排队时间长而重新smoke。

## 4. 后台等待 E39 → 自动 E40

两项通过后：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E40a_SpatialTaskHead_AutoDL/run_e40.sh queue > launcher_e40_after_e39.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e40_after_e39.log
```

队列会立即验证已完成的check/smoke，不自动补测。日志出现 `E40 QUEUE watching ...`、`E40 WAIT ...` 才表示成功挂队列。
如果看到 traceback/Exit 1，就是没挂上，不要当作等待成功。
每分钟打印 E39轮次/状态；训练开始后原始逐batch/验证/每轮ETA日志直接进入同一launcher日志。
**这里 Ctrl+C 只停止 tail，nohup队列/训练继续；独立check/smoke中的Ctrl+C则会中止检查。**

触发必须同时满足：

- 唯一E39主训练进程（按PID/启动时间绑定，排除其DataLoader后代）；
- E39正常完成200轮，官方548图评估与REPORT完整，best权重哈希对应；
- E39主进程和控制器退出/锁释放；GPU连续两次无计算进程、空闲≥24GiB；
- E40安全回执仍有效，没有同名已存在训练。

E39失败/提前退出/缺评估/多主任务时**不启动E40**。不杀进程、不自动关机。
排队只判断工程正常完成，不根据E39 AP高低决定是否运行E40，两项是独立假设。

只读检查主任务识别：

```bash
bash /root/autodl-tmp/E40a_SpatialTaskHead_AutoDL/run_e40.sh inspect
```

如果有多个E39正式报告，显式指定这轮正在跑的目录：

```bash
nohup bash /root/autodl-tmp/E40a_SpatialTaskHead_AutoDL/run_e40.sh queue --e39-report /root/autodl-tmp/yolo11_project/comparison_reports/E39_RUN_实际时间 > launcher_e40_after_e39.log 2>&1 &
```

不要对已有同名launcher再次重定向重启队列，先确认前一个队列已退出。
已存在正式run时不允许覆盖；本包不提供自动断点续训，意外中断应先保留并审查checkpoint。

## 结果

- `yolo11_project/runs/e40a_spatial_task_head_img640_seed1_200e/`：results.csv、args.yaml、best.pt、last.pt。
- `yolo11_project/comparison_reports/E40_RUN_时间/`：training.json、transfer.json、official_metrics.json、comparison.json、REPORT.md、area_predictions.json、gt_rows.json。
- `yolo11_project/logs/queue_e40_时间/state.json`：接续状态、绑定PID、开始/结束时间。

训练后自动用冻结D27官方评分计算AP/AP50/AP75/AR500，复现E1官方分数后再比较。
同时输出P2/P3原始几何、类别正确候选和最终框覆盖的尺寸组诊断；覆盖率不等于AP或一对一召回。
单seed筛选线：AP≥E1+0.5pp、AP75≥E1+0.3pp、AP50不降；0.8pp是目标而非预测。

## 本地与服务器验证边界

见 LOCAL_VALIDATION.md。本地验证为CPU及合成数据，只证明已执行的工程路径；
真实服务器CUDA/AMP/batch16容量依赖上述独立check/smoke。未宣称正式训练或性能已验证。
复用的Ultralytics相关代码遵循AGPL-3.0，见 LICENSE-Ultralytics.txt。
