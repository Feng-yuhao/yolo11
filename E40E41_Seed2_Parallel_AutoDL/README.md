# E40 + E41 seed2，640 像素并行正式训练

两个实验都从原始 `yolo11s.pt` 和 E1-P2 结构重新初始化，分别使用 seed=2。
训练协议与已完成的 E1 seed2 一致：200 epochs、imgsz=640、batch=16、workers=8、AMP、SGD 和 close_mosaic=10。
E40、E41 的多深度编码与任务路由保持各自 seed1 正式版的设计。
官方 VisDrone 评分使用已有 E1 seed2 的预测文件作为配对基线，训练和推理都不读取 E40/E41 seed1 权重。

文件包只解压到新的 `/root/autodl-tmp/E40E41_Seed2_Parallel_AutoDL/`，不覆盖 E40/E41 seed1 包、模型或报告。
从本地上传 `E40E41_Seed2_Parallel_AutoDL.zip` 后：

```bash
cd /root/autodl-tmp
unzip -o E40E41_Seed2_Parallel_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E40E41_Seed2_Parallel_AutoDL/run_pair.sh check
```

`check` 顺序执行两套独立 CUDA 检查，必须看到 E40、E41 均打印 `CUDA CHECK PASSED`，且控制器退出码为 0。
接着在 GPU 空闲时运行两个真实训练 batch16/640 的并行 smoke：

```bash
bash /root/autodl-tmp/E40E41_Seed2_Parallel_AutoDL/run_pair.sh smoke
```

`smoke` 同时启动两个子进程；要求启动时无其他计算进程、空闲显存至少 27 GiB。
E40 smoke 的 PyTorch 显存上限为 14 GiB，E41 为 12 GiB。
必须在控制器日志中看到两个 `exit=0`，并可在打印的两个子日志中看到各自的 `CUDA SMOKE PASSED`。
smoke 只验证工程路径，不复用其参数作为正式权重。

两项检查通过后，后台启动完整 200 轮并行训练及两套独立官方评分：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E40E41_Seed2_Parallel_AutoDL/run_pair.sh run > launcher_e40e41_seed2.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e40e41_seed2.log
```

`run` 在启动前再次校验两个新包的 CUDA check/smoke 回执、E1 seed2 基线和训练目录未占用。
GPU 启动时需无其他计算进程且空闲至少 28 GiB；两份正式训练保持 batch16，不自动改 batch。
并行正式进程分别限定 PyTorch 分配上限为 E40 14 GiB、E41 12 GiB，留出系统和验证余量；
如果实际密集批超过各自上限，会记录该任务失败并保留已经产生的文件，不会偷改训练参数。
控制器每分钟报告两份训练轮次、退出码及 GPU 空闲显存，并打印两个子日志的绝对路径。
这里 `Ctrl+C` 只退出 `tail`，后台控制器和训练继续运行。
一份失败时，另一份继续完成；控制器最后以非零退出码报告失败，不自动关机。
正式训练按原版各自的包内 `run_e40.sh` / `run_e41.sh` 执行，不修改共享 Ultralytics 源码。

训练权重位于：

- `yolo11_project/runs/e40a_spatial_task_head_img640_seed2_200e/weights/`
- `yolo11_project/runs/e41a_channel_depth_task_head_img640_seed2_200e/weights/`

官方报告分别位于 `comparison_reports/E40S2_RUN_时间/` 和 `comparison_reports/E41S2_RUN_时间/`。
每份包含 `training.json`、`official_metrics.json`、`comparison.json`、`REPORT.md`、`gt_rows.json` 和预测文件。
控制器状态与每份独立日志位于 `logs/pair_e40_e41_seed2_run_时间/`。

如果 200 轮训练已完成而评分失败，保留 `best.pt`，不要重新训练；可单独补评：

```bash
bash /root/autodl-tmp/E40E41_Seed2_Parallel_AutoDL/e40/run_e40.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e40a_spatial_task_head_img640_seed2_200e/weights/best.pt
bash /root/autodl-tmp/E40E41_Seed2_Parallel_AutoDL/e41/run_e41.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e41a_channel_depth_task_head_img640_seed2_200e/weights/best.pt
```

评分重新读取 VisDrone 原始标注和 E1 seed2 的已有预测，遵循同一官方 ignore 规则。
raw 高 IoU 覆盖率只输出候选模型本身；原 D38 使用 E1 seed1，不能作为 seed2 的几何差值基线。
最终决定依据 seed1、seed2 各自与相同 seed 的 E1 所形成的配对差值。

本地验证只能证明代码、模型和模拟训练路径在本机环境可运行。
服务器 CUDA、实际双开显存和长训练效果必须以运行后的 check/smoke/正式日志为准。
