# VisDrone -> DIOR / TinyPerson：三种子冻结权重直接评估

只评估，不训练。固定六份已有 VisDrone E1/ST-MDRF（E60）源域 val-best，两个目标测试集各评估六次，共12次，完全串行。
不改 YOLO11/26 环境，不更新模型、BN，不选目标域 checkpoint，不调参或裁片；不是在第二数据集重新训练。

## 固定任务

1. 检查两套完整数据、六份权重 SHA、实际代码和环境。
2. 工程单测，再对12个组合分别运行8图真实 CUDA smoke，含推理与相应评分。
3. DIOR：E1/E60 seed1，seed2，seed3；然后 TinyPerson 同样六次，全量串行。
4. 全部保存原始预测、评分、每种子的正负差值、均值及样本标准差；两个数据集不混成一个 AP。
5. 全文件 SHA 校验、完成凭据通过后，独立关机程序等待控制器退出、锁释放、其他任务及 GPU 空闲，60秒倒计时后请求平台关机。

DIOR 固定11738张官方test，不丢弃8430张无vehicle图片；vehicle=max(car,van,truck,bus)，这是预先声明的语义映射假设，不是完整DIOR20类指标。
使用已审计的连续HBB，四角减1、宽高不加1；一条原始零面积条目保留为ignore，不伪造框。主分数为标准COCO AP/AP50/AP75、maxDets100。

TinyPerson 固定786张原常规test，不择优改成dense扩展；person=max(pedestrian,people)，sea/earth合并。
使用作者MIT TinyBenchmark旧分支原始COCOeval匹配实现，保留ignore/uncertain/logo及IOD匹配；不是普通YOLO.val或vanilla COCOeval。
版本固定为包内`references/author_evaluator.json`的原始代码SHA，IOD阈值沿用v1的IOD>=IoU，不与后来huicv变换阈值版本混称同一协议。
仅为NumPy兼容将np.float转builtin float、linspace样本数转int；未改匹配算法。
主表另报ignore-aware COCO-style AP@[.50:.05:.95]/max100；补充原生tiny尺寸AP25/AP50/AP75/max200（tiny面积1²–20²）。不把两种AP混列为同一个数。
所有原标注字段保留；仅在报告内生成评分GT时将Tiny标注ID统一+1，避免COCO以0作为未匹配哨兵。图片ID、框、面积和原数据不变。

推理固定FP32/原生8.3.0 LetterBox最大边640/batch4/conf.001/NMS.7/top500；保留原十类头，先映射再NMS。
两数据集尺寸不同，记录真实输入形状，不宣称均为640方形。不复制作者裁片/高分辨率训练协议，不与其论文表格直接排名。

## 明天运行

先开GPU模式，不要在当前无卡模式运行all。两套数据已安装，无需再上传数据。
只上传本代码ZIP和`.sha256`到`/root/autodl-tmp/yolo11_project`，依次执行：

```bash
cd /root/autodl-tmp/yolo11_project
if sha256sum -c VisDrone_to_DIOR_TinyPerson_ThreeSeeds_SourceOnly_AutoDL.zip.sha256 &&
mkdir -p experiment_code &&
test ! -e experiment_code/VisDrone_to_DIOR_TinyPerson_ThreeSeeds_SourceOnly_AutoDL &&
unzip -q VisDrone_to_DIOR_TinyPerson_ThreeSeeds_SourceOnly_AutoDL.zip -d experiment_code; then
  nohup bash experiment_code/VisDrone_to_DIOR_TinyPerson_ThreeSeeds_SourceOnly_AutoDL/start_all_auto_shutdown.sh \
    > launcher_dior_tiny_s123.log 2>&1 < /dev/null &
  printf '控制器PID：%s\n' "$!"
fi
```

只检查数据/权重（不运行推理，不关机）：

```bash
bash experiment_code/VisDrone_to_DIOR_TinyPerson_ThreeSeeds_SourceOnly_AutoDL/run.sh prepare
```

## 不刷屏监控

```bash
/root/miniconda3/envs/yolo11/bin/python -u \
  experiment_code/VisDrone_to_DIOR_TinyPerson_ThreeSeeds_SourceOnly_AutoDL/watch_progress.py --follow
```

Ctrl+C只结束查看；不会结束后台评估。实时细日志：`tail -n 80 -F launcher_dior_tiny_s123.log`，其中打印每项独立日志位置。
监控同时显示推理进度、评分的image×area进度、12项smoke和12项全量完成数。

## 保存空间与关机安全

原始预测直接写无损`predictions.json.gz`，不另存一份巨大的明文JSON，不裁框、不降低精度、不删以前的文件。
每份完成后读取gzip并解析校验，保存SHA及全部已见图片ID证明。
启动要求数据盘至少4.5GiB；每批保留1GiB最低空间，总报告上限3.5GiB、单份压缩预测512MiB；触线保留已有内容并停止，不关机。
数据盘近期剩约5.9GiB，但实际启动仍重新核验，不承诺磁盘永不被其他进程占用。

check/smoke/评分/保存任一步失败均不关机。不杀其他进程、不通用poweroff回退；只调用预先核验SHA的AutoDL文本关机helper。
不要同时运行另一训练/评估。关机返回0不等于确认平台停止计费，最终看控制台实例状态。
取消本批的最后关机：

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_DIOR_TINY_S123
```

这个文件存在时不会关机；需要关机的新运行前，自行确认后移走该取消文件。
不自动清理现有取消文件。低层`start_all.sh`不启用关机，本次应使用`start_all_auto_shutdown.sh`。

## 结果下载

仅新增一整个报告目录：`comparison_reports/VISDRONE_TO_DIOR_TINY_S123_SOURCE_ONLY_<时间_PID>/`。
索引：`logs/latest_visdrone_to_dior_tiny_three_seeds_source_only.json`。
目录包含两个目标GT、12份完整gzip预测及评分、各自三种子汇总、源码、数据/权重绑定、日志、文件SHA及关机记录。
不产生训练runs目录。下载整个新报告目录及launcher日志即可；不要只下载REPORT.md后说原始预测已归档。

## 评分来源

- https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark
- https://github.com/ucas-vg/PointTinyBenchmark/blob/master/dataset/TinyPerson_annotation.md
- https://gcheng-nwpu.github.io/#Datasets
- https://github.com/cocodataset/cocoapi

已有Songdo/UAVDT/State-Air结果均保留；本批也保留全部正负种子，不择优展示或据目标测试AP调整规则。
