# D16a：跨视图局部候选可靠度校准（分组OOF探针）

## 为什么是这一步

D15b 的可部署动态1.5块结果为 `mAP50-95=33.574%`、`AP-small=26.056%`。标签有效性上界达到 `36.063%` 与 `29.298%`，在置信度0.25处减少5151个FP、只损失198个TP，说明主要缺陷是局部候选误检与重复，而不是继续增强P2/P3特征。

D15b 的选择效用Oracle不是COCO AP的精确全局Oracle，因此只能否定该效用函数，不能证明所有选区方法都没有空间。本轮保持D15a选区不变，只验证融合端的可学习性。

## 方法

每个局部候选使用以下推理期信息：

- 局部置信度、尺度、长宽比、裁剪边界距离；
- 全图中同类框与任意类别框的一致性；
- 全图与局部框的IoU、置信度差和尺度差；
- 局部候选之间的重复程度；
- 类别、裁剪位置和D15a区域分数。

轻量残差MLP只重标定局部框分数，不改框坐标，不改全图框，不改E1、D15a或Ultralytics。目标为同类、IoU不低于0.5的一对一局部候选，软标签使用IoU。

正式诊断采用按航拍序列分组的5折OOF和内部组留出选择。每张图都由没有见过该序列标签的校准器处理；但它仍是验证集交叉验证探针，最终论文结果必须使用未参与开发的测试集。

## AutoDL执行

上传压缩包到 `/root/autodl-tmp/`：

```bash
cd /root/autodl-tmp
unzip -o D16a_CVRC_OOF_Probe_AutoDL.zip

conda activate yolo11
python /root/autodl-tmp/D16a_CVRC_OOF_Probe_AutoDL/install_d16a_cvrc_oof.py

cd /root/autodl-tmp/yolo11_project
python -u diagnose_d16a_cvrc_oof.py --check-only
python -u diagnose_d16a_cvrc_oof.py --smoke2
```

两项通过后执行完整OOF诊断：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash run_d16a_cvrc_oof.sh > launcher_d16a_cvrc_oof.log 2>&1 &
tail -f launcher_d16a_cvrc_oof.log
```

结束标志：

```text
D16a completed. Report: .../metrics.txt
Automatic shutdown is disabled; the instance remains running.
```

同步整个 `DIAG_D16A_e1_cvrc_oof_*` 目录。正式门槛为相对D15a同时满足：mAP50-95至少提升0.2个百分点、AP-small至少提升0.3个百分点、AP-medium下降不超过0.5个百分点。

所有阶段都不会自动关机。
