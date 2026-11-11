# VisDrone → Songdo：冻结权重直接跨数据集评估

只评估，不训练，不微调，不更新 BN，不自动关机；旧 Songdo seed1/2 训练包和结果不动。

## 固定实验

- 使用已完成的 VisDrone E1 和 ST-MDRF/E60，seed1、2、3 共六份源域验证集选出的 best.pt。不是 Songdo 训练权重，更不是原始 COCO80 预训练。
- 保留原 YOLO11 的10类输出头；没有构造/替换为新的4类分类器。只在输出端做固定语义映射。
- Songdo car 包括 vans/light-duty vehicles；因此 `car=max(source car, source van)`，bus←bus，truck←truck，motorcycle←motor。其余源类不计入本次目标四类评分。依据随包 `references/songdo_author_README.md`，不是看完测试结果才选择映射。
- 在每个原始候选位置合并分数，再执行最佳类别选择、class-aware NMS、top500。独立 sigmoid 不能直接当互斥概率相加；这里预先固定最大值，不进行 max/sum 比较选优。
- 固定 FP32、640、batch4、conf=.001、NMS IoU=.7。COCO bbox AP@[.50:.05:.95] / maxDets100 为主，附 AP50/AP75、每类 AP、原图面积分层 AP。不是 VisDrone ignore 评分，也不是训练日志的原生 mAP。
- 输入预处理沿用原8.3.0 Predictor：最大边640；同尺寸横图批次实际为384×640，混合横竖图批次为640×640。旧版本的 predict 不依据 rect=False 强制方形。每模型固定相同ID顺序/batch划分并记录实际形状，不能把“imgsz640”写成所有推理都640方形。
- 全部1084张官方 test、55528个 GT；包括一张竖图，按真实尺寸还原框。没有读取目标 train/val 来学习或选权重。
- 三组源模型描述源域训练重复下的直接迁移结果；不等于三个新数据集，也不是开放词汇/新类别零样本检测。之前在 Songdo 重训的实验是另一种设定，不可混成同一张性能增量表。

## 服务器使用

需要保留：`/root/miniconda3/envs/yolo11`、`/root/autodl-tmp/yolo11_src`、原项目 runs 下六份 VisDrone best.pt、已核验 Songdo 数据。无需再次上传数据或权重。缺失/哈希不一致会停，不会自动下载或换成别的权重。

把 ZIP 上传到 `/root/autodl-tmp/yolo11_project`，核验后解压到 `experiment_code`。不能覆盖已有同名包。

```bash
bash experiment_code/VisDrone_to_Songdo_SourceOnly_AutoDL/run.sh prepare
nohup bash experiment_code/VisDrone_to_Songdo_SourceOnly_AutoDL/start_all.sh \
  > launcher_visdrone_to_songdo_source_only.log 2>&1 < /dev/null &
```

`prepare` 不导入 torch，不查询 GPU；核验代码、六份源权重和目标 test 字节。`all` 自动执行：完整 preflight → 类别映射/NMS/COCO单元检查 → 六模型各8张真实目标图 smoke（含竖图、不计算用于挑选的 smoke AP）→ 六次完整 test 预测与评分 → 三种子配对汇总。

先等正在运行的 Songdo seed2 等任务自然结束再启动。发现其他 GPU 任务或未知 Python 任务/内核时会安全停止，查看 launcher 日志；不杀进程、不抢任务、不降低 batch、不静默切 CPU。

```bash
/root/miniconda3/envs/yolo11/bin/python -u \
  experiment_code/VisDrone_to_Songdo_SourceOnly_AutoDL/watch_progress.py --follow
```

仅变化时更新。Ctrl+C 仅退出监看，不会停止后台评估。

## 结果只需下载一个目录

`comparison_reports/VISDRONE_TO_SONGDO_SOURCE_ONLY_时间_PID/` 整个目录。

含 `REPORT.md`、`summary.json`、六模型逐图预测/标准COCO评分/类别AP、工程检查及smoke、独立日志、实时状态、实际源码、原权重与数据/代码哈希、推理前后参数及BN不变凭据、最终结果文件校验清单。权重本身不重复复制，原始六份已在主项目中。

用 `logs/latest_visdrone_to_songdo_source_only.json` 找这个目录。只有 `state.json status=completed`、六份评分成功且 `ARTIFACT_SHA256.json` 生成才是完整完成；失败保留已有结果和真实错误，服务器不关机。重新运行会建新会话，不覆盖旧结果。

本包没有训练、优化器、目标域校准、阈值搜索或关机入口。只运行推理和评分，耗时以实际图像解码/推理/COCO匹配速度为准，不能套用200轮训练时间。
