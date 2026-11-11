# E0 seed2/3 并行补训 + E0/E60 seed2/3 串行官方 test-dev

只补 E0 两个种子，**不重新训练 E60，不增加新模块**。
先保留 val-best，再测试；不用测试集选择 epoch/阈值，不删除负收益种子。

## 1. 上传、解压、实体安装

```bash
cd /root/autodl-tmp
unzip E0_Seeds23_Parallel_E60_Testdev_AutoDL.zip
bash /root/autodl-tmp/E0_Seeds23_Parallel_E60_Testdev_AutoDL/install.sh
```

真正的运行代码在项目的 `experiment_code/E0_Seeds23_Parallel_E60_Testdev`，不是指向 tmp 的软链接。
不依赖已删除的 E40/E58/E60 解压目录。已有 yolo11_src、数据、原预训练、E60 seed2/3 best.pt 必须存在。
包内固定了 E60 原训练/评分来源和只读推理类；不会把 E60 类挂到 E0 上。

## 2. 独立 check / 并行真实 smoke

```bash
cd /root/autodl-tmp/yolo11_project
bash experiment_code/E0_Seeds23_Parallel_E60_Testdev/run.sh check
bash experiment_code/E0_Seeds23_Parallel_E60_Testdev/run.sh smoke
```

check 显示实时输出：原始协议/预训练/标签检查、E0 三尺度架构和新进程保存重载/fuse、
E60 seed2/3 原 val8 预测复现和 test-dev8 官方坐标/ignore 检查。
smoke 启动两路 E0 同时 batch16 实际 AMP 更新（各至少8次成功更新），随后双方训练模型、优化器和 EMA
保持驻留，同时做 EMA-half batch32 方形压力及全部548验证图的 loss/NMS。
它只检查工程和显存，不选择好坏模型。每进程固定13GiB上限，正式和smoke一致；不自动降batch。
正式使用原 E0 的未经重写 train()，同 SGD、warmup、累积、cosine200、close_mosaic10、patience0。
双开改变争用/墙钟时间，不保证GPU非确定算子在不同机器上逐位一致。

只有两路真实并行smoke均 exit0，且当前代码/环境/数据签名一致，正式阶段才获准。
不足时明确失败，不跳门禁、不训练短轮数替代正式结果、不自动重新训练。

## 3. 正式后台运行（已通过门禁）+ 最后自动关机

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash experiment_code/E0_Seeds23_Parallel_E60_Testdev/run.sh run --auto-shutdown > launcher_e0_s23_e60_testdev.log 2>&1 &
tail -f launcher_e0_s23_e60_testdev.log
```

没有预跑门禁时可用 `all --auto-shutdown` 自动串起 check→并行smoke→并行formal→串行评估。
`check` / `smoke` 无论有无参数都不关机；默认任何模式都不关机，正式命令显式启用。

输出新训练目录：

```text
runs/baseline_yolo11s_img640_seed2
runs/baseline_yolo11s_img640_seed3
```

拒绝覆盖/自动加后缀/从其他seed的best微调。两者均完整200轮才进入评分；一方失败不会强杀另一方。
缺失/未完成的模型不会被当成正式结果评分。修复评分问题后可用 `run.sh eval --auto-shutdown` 重做串行评分，
无需再训练；中断训练不自动恢复或重跑，保留权重/日志等待明确处理。

测试阶段严格串行：E0 seed2 → E0 seed3 → E60 seed2 → E60 seed3。
每个先评全量548 val（E60复现原val结果），再评1610 test-dev，FP32/640/batch4/conf0.001/IoU0.7/max_det500。
沿用官方原标注和ignore规则，与旧测试协议一致。既有 E0/E60 seed1 和 E1三个seed 分数核验后复用。
最后报告三seed AP/AP50/AP75/AR500、均值/样本标准差、E60−E0总收益和 E60−E1新增贡献。

## 4. 分别看训练（不刷 tqdm）

控制器每15秒报两个任务退出码、已记录epoch和显存，并打印两个训练日志的准确路径。
另开终端，在输出的文件上运行 `tail -f 路径`。
只看每轮摘要：

```bash
session=$(/root/miniconda3/envs/yolo11/bin/python -c 'import json;print(json.load(open("/root/autodl-tmp/yolo11_project/logs/latest_e0_s23_e60.json"))["session"])')
tail -n 15 -F "$session/e0_seed2_run.log" | grep --line-buffered -E 'E0 seed2 epoch|COMPLETED|Traceback|RuntimeError'
```

另一个终端将 `seed2` 两处改为 `seed3`。
Ctrl+C 退出 tail 不影响 nohup 后台。停止实例/服务器关机当然会中断尚未完成的训练。

```bash
bash /root/autodl-tmp/yolo11_project/experiment_code/E0_Seeds23_Parallel_E60_Testdev/run.sh status
```

结果：`comparison_reports/TESTDEV_E0_E60_S23_.../REPORT.md`、`summary.json`及每模型val/testdev预测/官方评分。
状态索引：`logs/latest_e0_s23_e60.json`。session内有`state.json`、独立训练日志、评分日志和`shutdown_status.json`。

## 5. 自动关机的安全边界

双方训练、最后串行评估、报告写盘后才关机；正式已启动但中途失败也先记录退出码，等所有任务释放GPU再尝试。
关机helper等控制器退出，确认无其他GPU/Python训练评估任务，60秒倒计时后再次确认。
AutoDL无shebang的ASCII `/usr/bin/shutdown` 用 `/bin/bash /usr/bin/shutdown`，不直接exec，不自建破坏性替代命令。

取消关机：

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_E0_S23_E60
```

关机请求返回零不证明平台停止计费；请在AutoDL控制台确认实例已关闭。
不检查成功、不跳门禁，不对仍有其他任务的实例执行关机。关机测试在本地全部mock。

## 验证范围

见 LOCAL_VALIDATION.md。本地检查不替代服务器真实双开batch16/32显存门禁，不保证训练涨点。
