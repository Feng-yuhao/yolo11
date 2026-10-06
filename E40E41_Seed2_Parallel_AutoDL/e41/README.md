# E41a ChannelDepthTaskHead / AutoDL v2

解压到 /root/autodl-tmp，保留本包独立目录。不要覆盖E40文件。

```bash
cd /root/autodl-tmp
unzip -o E41a_ChannelDepthTaskHead_AutoDL_v2.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E41a_ChannelDepthTaskHead_AutoDL/run_e41.sh check
bash /root/autodl-tmp/E41a_ChannelDepthTaskHead_AutoDL/run_e41.sh smoke
```

必须分别看到 E41 CUDA CHECK PASSED 和 E41 CUDA SMOKE PASSED。
v2补全第一版漏打包的p2_tal_chunked.py，替换E41包后需重新check/smoke，旧回执不会复用。
ZIP只包含E41文件，不会覆盖E40源码或任何训练权重。
check/smoke失败不会开启队列；队列只验证回执，不自动重跑或跳过测试。
check使用合成小图验证；smoke使用真实640/batch16/8批训练和真实更新后的推理。
smoke要求当前空闲至少16.5GiB，自身PyTorch分配上限12GiB，保护4GiB余量。
这允许在E40运行时尝试smoke，但不能保证任何密集批都不会OOM；失败不发回执。

通过后推荐串行接续（E40无论成功或报错，只要退出并释放GPU就继续）：

```bash
nohup bash /root/autodl-tmp/E41a_ChannelDepthTaskHead_AutoDL/run_e41.sh queue > launcher_e41_after_e40.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e41_after_e40.log
```

看到 E41 QUEUE ACTIVE 即表示监听已建立。E41 WAIT 每分钟显示状态；
看到 STARTING E41 FORMAL 200 EPOCHS 即开始训练。同一日志实时显示训练、每轮时间、ETA和评分。
此处Ctrl+C只退出tail，不停止nohup队列/训练。不要反复重复启动。
队列不终止任何进程、不覆盖运行结果、不自动关机。
E40的报错会保留在 queue_e41_*/state.json 的 e40 字段，不伪造E40成功。
GPU有其他计算进程时继续等；需连续两次空闲且空闲>=24GiB。
如果E40意外卡死且始终占用GPU，队列不会强杀它，36小时超时后记录失败。

可选直接并行启动：完成相同check/smoke后，把上面的queue命令换成：

```bash
nohup bash /root/autodl-tmp/E41a_ChannelDepthTaskHead_AutoDL/run_e41.sh run > launcher_e41_parallel.log 2>&1 &
tail -f launcher_e41_parallel.log
```

两种入口只选一种。显存当前值不是峰值，不建议仅凭E40现在<12GiB就保证双开安全。
正式训练固定batch16且不限制PyTorch显存；不会自动降batch改变协议。

结果：runs/e41a_channel_depth_task_head_img640_seed1_200e 下 best.pt/last.pt/results.csv。
完整报告：comparison_reports/E41_RUN_时间 下 training.json、transfer.json、
official_metrics.json、comparison.json、area_predictions.json、gt_rows.json、REPORT.md。
先记录200轮完成，再执行本包自带官方评分；若仅评分失败可用相同best补评：

```bash
bash /root/autodl-tmp/E41a_ChannelDepthTaskHead_AutoDL/run_e41.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e41a_channel_depth_task_head_img640_seed1_200e/weights/best.pt
```

该命令只补评，不重训。不要因评分失败删掉已有训练结果。
