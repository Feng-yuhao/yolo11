# E47 seed2：原模块不变，单开正式200轮

只运行E47 seed2；不运行seed3，不重训E1/E40，不启动其他模型，不默认关机。
本包不修改服务器已安装的Ultralytics源码，不从E40/E47 best微调。

## 与seed1保持一致

- 原E40空间任务深度库、路由与输出投影全部保留；新增逐深度对比读出完全不变。
- `depth_readout.py`、`spatial_task_head.py`、`parent_spatial_task_head.py`、`extension.py`、TAL和原模型YAML与seed1包逐字节一致。
- 原始`weights/yolo11s.pt`迁移，独立40001初始化E40块，独立47001初始化新读出；只有全局训练种子改为2。
- 640、batch16、workers8、200轮、完整余弦LR、SGD、AMP、patience0、close_mosaic10及原YOLO损失保持不变。
- 不降低batch，不用smoke权重开始正式训练。单进程28GiB allocator上限，smoke与formal一致；启动要求GPU空闲至少30GiB。
- 官方AP/AP50/AP75分别配对已有E1 seed2、E40 seed2；不拿seed1数字作对照，不调用seed1的D38几何差值。
- 原seed1的+0.25pp资源优先门槛没有被修改；本次是用户决定复核低于该门槛的单seed正信号，不是宣称此前已通过筛选。

## 需要已有文件

项目：`/root/autodl-tmp/yolo11_project`；固定源码：`/root/autodl-tmp/yolo11_src`。
原预训练：`yolo11_project/weights/yolo11s.pt`。

复用原报告和预测，不重训对照：

```text
comparison_reports/e1_yolo11s_p2add_img640_seed2_20260926_172907_911324/
comparison_reports/E40S2_RUN_20260930_130808_314014/
runs/e1_yolo11s_p2add_img640_seed2/weights/best.pt
runs/e40a_spatial_task_head_img640_seed2_200e/weights/best.pt
```

E1报告需要metrics.json、area_ground_truth.json、area_predictions.json；E40报告需要comparison.json、training.json。
check会核对两者种子、协议、完整200轮、权重SHA以及官方标注/原图映射；缺文件就停，不伪造或跳过。

## 上传并解压

把`E47_Seed2_AutoDL.zip`上传到`/root/autodl-tmp`，首次解压：

```bash
cd /root/autodl-tmp
unzip E47_Seed2_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
```

不要覆盖正在运行的同名包。入口固定使用`/root/miniconda3/envs/yolo11/bin/python`，不依赖当前conda提示符。

## 推荐：一次后台命令，门禁通过才自动训练

GPU空闲后执行：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E47_Seed2_AutoDL/run_e47.sh all > launcher_e47_seed2.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e47_seed2.log
```

顺序固定为`check → smoke → verify-gates → run200 → official evaluation`。
任一步失败就退出，不启动后续步骤。check/smoke分开进程，不污染正式初始化。
真实smoke包括至少8次batch16/640成功更新和batch32完整548图EMA半精度验证，不是短训练性能probe。
日志实时输出每个阶段、每轮epoch_seconds、elapsed、ETA、输出读出L1和官方评估进度。
`tail -f`中Ctrl+C只退出日志查看；本地关机不停止nohup后台任务。不对训练PID执行Ctrl+C/kill。
每次同名启动由独立`.e47_seed2_execution.lock`防重入；已有正式run拒绝覆盖。

## 也可分开执行

```bash
bash /root/autodl-tmp/E47_Seed2_AutoDL/run_e47.sh check
bash /root/autodl-tmp/E47_Seed2_AutoDL/run_e47.sh smoke
bash /root/autodl-tmp/E47_Seed2_AutoDL/run_e47.sh verify-gates
nohup bash /root/autodl-tmp/E47_Seed2_AutoDL/run_e47.sh run > launcher_e47_seed2.log 2>&1 &
```

只使用本包seed2门禁；代码、配置、环境或数据YAML改变后必须重新check/smoke，不能沿用seed1凭据。

## 结果与评估恢复

权重/曲线：`runs/e47a_p2_depth_readout_img640_seed2_200e/`。
报告：`comparison_reports/E47S2_RUN_时间/`，包括：

```text
transfer.json
training.json
area_predictions.json
gt_rows.json
official_metrics.json
comparison.json
REPORT.md
completion.json
```

只有正式200轮和官方评估全部完成才生成completion.json。评分包含相对E1 seed2和E40 seed2的差值。
若训练结束而评分失败，不重训，补评分：

```bash
bash /root/autodl-tmp/E47_Seed2_AutoDL/run_e47.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e47a_p2_depth_readout_img640_seed2_200e/weights/best.pt
```

补评分产生独立E47S2_EVAL报告，不把原失败RUN改成成功。
raw@.75是候选存在率诊断，不是AP-small或一对一recall。两seed只提供重复性线索，不等于显著性或录用保证。
这次完成后先判断E47相对E40的额外收益是否重复，再决定是否补seed3，不自动开展新变体。

## 验证边界

见LOCAL_TESTS.md。本地测试不发服务器CUDA smoke通过凭据；服务器真实check/smoke仍由上述命令执行。
官方评分器与原seed1完全相同，改变的只有同seed参考选择与输出路径。保留Ultralytics AGPL许可证。
