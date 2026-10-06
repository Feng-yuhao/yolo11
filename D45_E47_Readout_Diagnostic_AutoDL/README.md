# D45：E47 新增读出作用与剩余缺点诊断

只分析已完成的 E47 与同 seed E40；不训练、不更新权重、不改 Ultralytics 源码、不关机。
默认 seed1，配对已有 E47P/E40 正式 200 轮报告。seed2 只有正式训练和评分全部完成后才允许选择。
无法从一次推理诊断证明训练根因、种子波动原因，或保证下一版涨点。

## 诊断回答什么

1. E47 相对 E40 在哪些类别、尺寸、同一 GT 上改善或退化？净提升背后有多少新增覆盖和损失？
2. P2/P3 分类/回归的四处新增深度读出，在**已训练权重**下分别有怎样的作用？
3. 剩余失败是缺少 raw 高 IoU 候选、缺少 raw argmax 同类别候选，还是最终候选缺失？这只是位置描述。
4. 新增读出是否实际输出非零残差？它与原 E40 残差的局部幅度/方向关系如何？统计不能单独证明收益。

六种状态：normal、p2_cls_off、p2_reg_off、p3_cls_off、p3_reg_off、all_readout_off。
**只旁路新增 DRF 读出**；原 E40 编码库、空间路由、分类/回归输出投影、原 Detect 塔都保留。
all_readout_off 的 E40 部分已经随 E47 联合学习，不等于单独训练的 E40。
不会逐个删除三个对比基；基方向是参数化选择，不能把每个基当成独立语义机制。

## 解压后：先跑不占 GPU 的缓存分析

上传 zip 到 `/root/autodl-tmp`：

```bash
cd /root/autodl-tmp
unzip -n D45_E47_Readout_Diagnostic_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh offline
```

这一步用 stdlib，不导入 Torch，不读取 checkpoint。可在 seed2 训练期间运行。
通常很短，会实时打印报告路径。严格核对 dataset SHA、GT 键、类别、尺寸、同 seed 与官方指标记录。
旧 GT 缓存没有坐标/密度，明确报告未逐坐标检查、不编造密度结果。

## GPU 检查与 8 图功能 smoke

建议等 seed2 退出并释放 GPU，再运行：

```bash
bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh check
bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh smoke
```

check 是非零读出下独立公式、四尺度真实头、未选任务一致、异常恢复、权重不变、
真实 best.pt 的合成图 Predictor/NMS/缓存集成；**不做训练**。
smoke 是真实验证集前 8 图，在全部六状态下推理，并复现正常状态保存预测与几何缓存。
8 图不计算性能排名、不作模块淘汰，也不是两个 epoch 的训练。
CPU check 只能验证本地工程，不代替服务器 CUDA 8 图 smoke。

默认要求没有其他 CUDA 计算进程；检测到正在训练会明确退出，绝不杀训练。
若你明确要并行诊断，可在 check/smoke/run 都加 `--allow-busy-gpu`，但会争用算力、拖慢训练；不推荐。
固定 FP32、imgsz640、batch4、conf0.001、iou0.7、max_det500。
默认分配器上限 6 GiB + 至少 1 GiB空闲余量；OOM 就失败，不静默降 batch。
每批 Backbone/Neck 只算一次，重放六个头状态，统计只读取真实前向捕获值，不再算一遍编码器。

## 后台全量诊断和实时进度

check/smoke 成功后：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh run > launcher_d45_e47.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_d45_e47.log
```

Ctrl+C 只退出这里的 tail，后台继续。不需要 conda activate，因为入口显式使用 yolo11 环境 Python。
会显示图像进度、推理 ETA、显存，以及每状态官方 class1/10..10/10 的评分进度。
`all` 会依次执行 check→smoke→run，任一步失败即停；也不会开始任何训练。
normal 必须逐图复现原坐标/分数（atol1e-5），raw/final 几何缓存（atol1e-5），
全量官方 AP/AP50/AP75/AR500 与原记录差值 <=0.002 pp；不通过就禁止解释其他干预。
源实现、代码、环境、权重、dataset、batch/显存上限改变，smoke 凭证会失效。

## seed2 完成后的复核

```bash
bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh offline --seed 2
bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh check --seed 2
bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh smoke --seed 2
nohup bash /root/autodl-tmp/D45_E47_Readout_Diagnostic_AutoDL/run_d45.sh run --seed 2 > launcher_d45_e47_seed2.log 2>&1 &
```

发现多个有效正式报告会要求显式 `--e47-report /绝对路径 --e40-report /绝对路径`，不猜最新文件。
不会读取未完成 seed2 的半成品来判定波动。

## 输出

`yolo11_project/comparison_reports/DIAG_D45_模式_S种子_时间/`：

- `offline/REPORT_OFFLINE.md`：官方逐类差异、尺寸得失、剩余失败描述。
- `offline/paired_slices.json`：按类别、尺寸×类别，同 GT 改善/损失/净变化、状态转移。
- 全量顶层 `REPORT.md`、`decision.json`：六状态 AP 与解释边界，不自动批准下一轮训练。
- `run/metrics.json`、各状态 `*_official.json`、`*_predictions.json`、`*_gt_rows.json`、`*_geometry.json`。
- `run/readout_rows.json`、`readout_summary.json`：P2/P3 的 GT中心/3×3路由、残差 RMS/比值、cosine，按官方预测 ignore 过滤前的最终同类别覆盖成功/失败分组。
- `controller.json`、独立 check/smoke/run 实时日志、失败时 `failure.json`。

旧 final coverage 在官方预测 ignore 过滤之前。新推理还额外给出 `official_final_*` 过滤后几何覆盖，
仍不是官方尺度 AP/一对一 Recall；输入 sqrt(area) 的像素分组不能写成 COCO AP-small。
四项单点变化不能相加，旁路更好不证明训练有害，旁路更差也不证明该机制贡献了全部训练增益。
不要把验证集上挑出的最佳旁路状态作为最终论文模型，不搜索阈值、不使用 test-dev。

本地实际验证范围见 `LOCAL_VALIDATION.md`；服务器实际 CUDA smoke/全量评分须由用户执行。
