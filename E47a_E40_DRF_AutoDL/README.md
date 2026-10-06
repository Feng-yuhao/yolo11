# E47a：E40-DRF（独立 seed1 / 640 / batch16 / 200轮）

本包仅包含第一项固定实验。没有第二模型、并行控制器、监测队列或默认关机。
不修改已安装的 Ultralytics 源码，不读取 E46 权重，不从 E40 best 微调。

## 固定研究变量

原 E40 的 P2/P3 各保留：128→48 编码、三层串行3×3、四深度库、两任务逐位置深度 softmax，以及原有零初始化输出投影 O。

原读出：`X_t = X + O_t(sum_k w_tk Z_k)`。

新增读出：`X_t = X + O_t(sum_k w_tk Z_k) + sum_j R_tj(sum_k B_kj w_tk Z_k)`。

固定的4×3对比基 B：

```text
 0.5   0.5   0.5
 0.5  -0.5  -0.5
-0.5   0.5  -0.5
-0.5  -0.5   0.5
```

每个 R 是无 bias 的48→128线性1×1卷积，全部零初始化。分类与回归各有3个 R，P2/P3分别独立。
不添加归一化、激活、门控、gain、辅助监督或 detach。P4/P5、原 cv2/cv3 塔、DFL、TAL、NMS 和损失不改。
新增小读出采用 FP32 计算，输出恢复原 E40 dtype；原 E40 精度路径不改。半精度 EMA 的参数不会被原地转换。

这等价于 `A_tk = O_t + sum_j B_kj R_tj` 的逐深度变换，是对读出容量的受控增强，不是新注意力算子。
它测试“混合前后是否需要不同深度的线性变换”。D44的冻结干预不构成这个限制的因果证明。
相关方法应与 TOOD 等任务分解结构比较，不能宣称首次任务解耦或保证涨点/录用。

## 初始化、协议和资源

- 从原 `yolo11s.pt` 迁移，原 E1 和 E40 所有共有参数逐张量核对。
- 先完整创建两个 E40 块（独立CPU种子40001），再安装读出（独立CPU种子47001）；不改变全局 CPU/CUDA RNG。
- seed1、200轮完整余弦日程、640、batch16、workers8、SGD、AMP、patience0、close_mosaic10，其余与 E1 原报告一致。
- 新增73,728参数；未融合模型9,928,136参数。E40为9,854,408参数。
- 同Conv/GN统计约42.286 GFLOPs，E40约39.927。新增卷积2.359 GFLOPs。
  元素乘、固定对比组合和归约没有计入该Conv统计口径；实际延迟需重新测量，不能只用GFLOPs推算。
- 当前默认单开。真实 smoke 和正式训练都使用28GiB分配上限，不静默降batch。启动前要求32GiB卡有至少30GiB空闲。
  不要暂停正在跑的E46；待其结束释放GPU后再执行本包的check/smoke。第二项和并行门禁后面另写。
- 原输出零初始化和新增输出零初始化保证初始 E1 一致性；额外参数的梯度会影响全局梯度裁剪，不能声称训练轨迹完全等于 E40。

## 上传、解压

上传本ZIP到 `/root/autodl-tmp`。首次安装执行：

```bash
cd /root/autodl-tmp
unzip E47a_E40_DRF_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
```

ZIP包含顶层 `E47a_E40_DRF_AutoDL` 文件夹，不解压进 `yolo11_project`。已存在同名包时不要覆盖运行中的目录。
Python使用 `/root/miniconda3/envs/yolo11/bin/python`，不依赖命令行是否显示 `(yolo11)`。

## 分开执行 check 与 smoke

GPU空闲后，先运行并等待 `E47 CUDA CHECK PASSED` 和 `check FINISHED`：

```bash
bash /root/autodl-tmp/E47a_E40_DRF_AutoDL/run_e47.sh check
```

然后运行真实batch16训练（至少8次成功更新）及完整548图、batch32的半精度验证：

```bash
bash /root/autodl-tmp/E47a_E40_DRF_AutoDL/run_e47.sh smoke
```

等待 `E47 CUDA SMOKE PASSED` 和 `smoke FINISHED`。这不是2轮性能probe，不用于淘汰方法。
check/smoke均实时打印；修改代码、环境、数据、训练参数后门禁失效，必须重新执行。
本地CPU和小CUDA单元验证不发服务器smoke凭据，不可代替真实batch16显存测试。

## 正式后台训练

两个门禁都通过后：

```bash
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E47a_E40_DRF_AutoDL/run_e47.sh verify-gates
nohup bash /root/autodl-tmp/E47a_E40_DRF_AutoDL/run_e47.sh run > launcher_e47_drf_seed1.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e47_drf_seed1.log
```

`tail -f`里Ctrl+C仅退出日志查看，不终止nohup后台训练；关闭本地电脑也不终止它。
不要在前台训练进程中Ctrl+C，也不要对训练PID发信号。本包不会自动关机或覆盖已有同名run。
每轮输出时间、elapsed、ETA和16个输出投影的L1值；初始化阶段和评估阶段也有明确日志。

## 结果文件与补评估

权重：

```text
yolo11_project/runs/e47a_p2_depth_readout_img640_seed1_200e/weights/best.pt
yolo11_project/runs/e47a_p2_depth_readout_img640_seed1_200e/weights/last.pt
```

报告：`comparison_reports/E47_RUN_时间/`，至少包括：

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

`completion.json`只在正式200轮和官方评分全部成功后生成。只有 best/last 不等于评分已完成。
官方 VisDrone 的 AP/AP50/AP75 与 E1 seed1、E40 seed1比较；评分器、ignore处理和坐标转换沿用D27/D38。
check前核对原图标注、548张映射、缓存预测和 E40 best SHA，评分时复现 E1 官方数字。
raw `.75`覆盖仅作补充诊断，不是AP-small，也不是一对一Recall。

若训练完成而评估异常，不要重训，可独立补评估：

```bash
bash /root/autodl-tmp/E47a_E40_DRF_AutoDL/run_e47.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e47a_p2_depth_readout_img640_seed1_200e/weights/best.pt
```

单seed的资源筛选规则保持：相对E40 AP至少+0.25pp且AP50/AP75不退才优先配对seed2；不是显著性阈值或论文保证。
即使暂时不够，也不能因此否定所有读出方法，更不能自动追加新变体训练。

## 核查入口

`depth_readout.py`为核心读出，`spatial_task_head.py`为装配；`extension.py`独立数学oracle；
`checks.py`完整工程验证；`run_e47.py`原生Trainer和门禁；`evaluate.py`官方评分；
`parent_spatial_task_head.py`原E40保持不变。见 `LOCAL_TESTS.md` 区分已执行与未执行验证。
保留原Ultralytics AGPL许可证，未混入Swin代码。
