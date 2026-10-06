# E1 / E40，test-dev，三个配对随机种子

此包只评估现有权重，不训练、不改权重、不安装依赖、不修改共享 yolo11_src，也不自动关机。
固定六个 **val 选出的 best.pt**，不挑 test-dev 最佳 seed，不做集成，不加入 E47/E48。

## 数据和路径

默认使用原始 `VisDrone2019-DET-test-dev/images` 与 `annotations`，必须完整 **1610 对**。
不是 DET-val 的548张，不是无公开标注的 test-challenge。无需转换 YOLO labels。

默认路径：

- Python：`/root/miniconda3/envs/yolo11/bin/python`
- 项目/权重：`/root/autodl-tmp/yolo11_project`
- 原源码：`/root/autodl-tmp/yolo11_src`
- 数据：`/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev`

数据另放时，在 **check、smoke、run三个命令都加入同一** `--dataset /实际路径`。
权重名称/SHA固定在 `models.json`；缺数据、缺权重或不符时明确报错，不下载或猜路径。

## 先核验，再后台运行

上传 ZIP 到 `/root/autodl-tmp` 后：

```bash
cd /root/autodl-tmp
unzip E1_E40_TestDev_3Seed_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E1_E40_TestDev_3Seed_AutoDL/run_testdev.sh check
bash /root/autodl-tmp/E1_E40_TestDev_3Seed_AutoDL/run_testdev.sh smoke --allow-busy-gpu
```

check不做GPU推理；它检查数据、图片内容SHA、六个权重、当前源码/环境和评分单元测试。
smoke依次用六个真实权重各推理8张，每批4图；只测工程功能/显存，不用于结论，不评分。
看到 `TEST-DEV SIX-WEIGHT CUDA SMOKE PASSED` 后再运行：

```bash
nohup bash /root/autodl-tmp/E1_E40_TestDev_3Seed_AutoDL/run_testdev.sh run --allow-busy-gpu > launcher_testdev_e1_e40.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_testdev_e1_e40.log
```

`Ctrl+C` 只退出此后台任务的 tail 查看；不要在前台 smoke/check 尚未完成时按 Ctrl+C。
nohup后台不依赖VSCode开着。这个评估任务不会停止或接管当前训练。
也可以用 `verify-gates` 单独验证凭据；run本身会验证，不能跳过smoke。

## 并行和协议边界

这是 **一个评估进程队列与现有训练并行**，不是同时开六个评估任务。
六个权重按 seed 配对顺序逐个推理，CUDA子进程退出后彻底释放自己的显存。
全部预测保存后才用 CPU/NumPy 评分，CPU阶段不加载GPU模型，线程限制为2。

协议固定：640、FP32、每批4图、conf=0.001、IoU-NMS=0.7、max_det=500、无TTA、类内NMS。
沿用Ultralytics8.3.0原预测器的letterbox/最后不满批处理；不改batch为1或改半精度。
沿用项目deterministic/warn-only设置，关闭cuDNN benchmark，记录TF32等实际后端状态。
保留原始坐标和全部后NMS预测，不用converted-label COCO评分替代官方风格AP。

每个推理进程默认4GiB PyTorch allocator上限，入口另要求至少3GiB自由余量。
**4GiB不是整个CUDA进程总占用的硬上限**（context/NMS库等不全归allocator）；训练也可能有峰值。
smoke通过不保证后续全数据绝不OOM；不会静默减batch、切FP16或杀训练。
如发生OOM或NMS超时，会停止并保留前面已完成的预测。并行会争用算力，可能减慢训练。
如果不希望影响训练墙钟时间，可等训练完成后使用相同命令。

## 结果、恢复和纯CPU评分

主日志一开始打印 `REPORT=/.../TESTDEV_E1_E40_RUN_时间`，完成目录包括：

```text
input_manifest.json       六个SHA、1610图映射、原标注/图片内容/代码/环境签名
e1_seed1/                 其余五个对应目录相同
  predictions.json        原图坐标，官方category_id=1..10
  prediction_manifest.json 完整性、SHA、实际输入shape、推理耗时和显存
  official_metrics.json   AP/AP50/AP75/AR、10类指标、ignore过滤计数
summary.json              3个配对差值与均值/样本标准差，不挑最佳seed
REPORT.md                 汇总表
COMPLETED.json            六个完整评分均成功后才生成
```

单有预测文件不表示整个评估结束。不要用smoke目录写论文。
若后处理失败或用户中断，已经完成的权重不必重推：

```bash
nohup bash /root/autodl-tmp/E1_E40_TestDev_3Seed_AutoDL/run_testdev.sh run --allow-busy-gpu --report /实际/TESTDEV_E1_E40_RUN_时间 > launcher_testdev_resume.log 2>&1 &
```

如果六份预测已全保存，只需要补CPU评分（不需要GPU或六份pt）：

```bash
bash /root/autodl-tmp/E1_E40_TestDev_3Seed_AutoDL/run_testdev.sh score --report /实际/TESTDEV_E1_E40_RUN_时间
```

也可下载完整报告和原始test-dev数据到本地，用有NumPy/Pillow的Python执行
`python testdev.py --mode score --report 本地报告绝对路径 --dataset 本地test-dev绝对路径`。
它仍会核验原图/标注和预测SHA；不需要Ultralytics、PyTorch或GPU。

## 评分来源与科研边界

`d27_official.py` 与 `spatial_task_head.py` 从已完成E40 seed1包逐字复制；SHA固定且check核验。
官方风格评分使用已有D27的公开MATLAB v1.0.4语义Python移植，保留class-0区域过滤、
score-zero GT忽略匹配、全图maxDet后类别过滤和class-availability聚合；**不声称官方在线服务器逐位一致**。
新入口仅把独立进程里的dataset count设为1610；不编辑旧的验证集评分器。
轻量ignore过滤与原D27过滤做数组等价测试；省略与本次无关的val FP/raw geometry诊断。

相关原始工具：<https://github.com/VisDrone/VisDrone2018-DET-toolkit>。
总体AP不能直接证明小目标AP提升；此包不伪造官方AP-small。
test-dev用于锁定方法的留出验证，不能根据这份分数反复选择新模块、epoch或seed。
尚在搜索的E48仍只用val决策；后续若改方法并复用test-dev，要披露已有test使用，避免当作未见数据。

本地测试：评分/映射/失败保护单元测试，加真实六权重加载和E1/E40 seed1拼图fixture上的CPU/小CUDA预测等价测试。
另以合成1610图验证六模型纯CPU评分、配对汇总与续评入口；不是正式test-dev分数。
这些**不能代替服务器CUDA smoke**；正式test-dev数据未在本地进行完整评估。
