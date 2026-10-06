# E40 seed3 + E47 seed1：640 像素并行实验包

E40 保留空间任务分解检测头，使用 seed3，与同 seed 的 E1-640 seed3 对照。
E47 保留已交付的 E40-DRF 深度对比读出，使用 seed1，与 E1-640 seed1 和已完成 E40 seed1 对照。
两项均从原始 `yolo11s.pt` 迁移重新初始化，不读取 smoke 权重或另一项正式权重。
保持原 200 epochs、imgsz640、batch16、workers8、AMP、SGD、close_mosaic10 协议。
这是工程双开入口；当前原型的 smoke 不能说明最终 AP，也不能代替配对多 seed 结果。

包只部署到新的 `/root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/`。
本交付未启动远程训练、未暂停现有任务。根入口使用已有 yolo11 Python，不下载依赖。
以下命令由用户在服务器上执行；每个步骤成功后才能继续。

```bash
cd /root/autodl-tmp
unzip E40S3_E47S1_Parallel_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/run_pair.sh check
```

`check` 顺序执行 E40 和 E47 的 CUDA 检查，输出实时显示在终端。
一项失败仍打印另一项检查结果，根退出码非零；须修复失败项后重新检查。
CPU、本机小 CUDA 单元和合成集成结果均不能替代服务器 CUDA check 回执。

服务器 GPU 空闲后执行真正并行 smoke：

```bash
bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/run_pair.sh smoke
bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/run_pair.sh verify
```

`smoke` 和 `run` 启动前要求单张服务器卡、空闲显存至少 30 GiB、无其他 compute PID。
已知全卡 32229 MiB；每个子进程 PyTorch allocator 上限固定 13 GiB，两者共 26 GiB，
smoke 和正式训练使用同一上限，并要求保留至少 2 GiB 空闲余量。不自动降低 batch。
allocator 上限不等于整个进程全部 CUDA 开销，真实服务器双开是否通过以 smoke 结果为准。

两个 smoke 进程在训练开始前、完整验证开始前、验证完成后三次同步；每项须得到至少
8 次成功的真实 batch16/640 AMP/SGD 参数更新（最多 32 次尝试），并保留训练模型和 optimizer
进行 EMA 类 FP16 副本、batch32、548 张完整验证及 loss/NMS 有限性检查。
回执必须记录两项训练窗口和验证窗口均实际重叠。
共享 labels.cache 的 dataset/dataloader 构建受项目级文件锁保护，训练主体继续并行。

根控制器只有在两项真实 smoke 都成功、且属于同一个控制器 PID/start_ticks、token 和 invocation
时才签发 `comparison_reports/e40s3_e47s1_pair_smoke_receipt.json`。
两个单独通过的 gate、不同次 smoke、旧包 smoke、CPU gate 均不足以启动正式双开。
context 和 pair receipt 使用 SHA256 绑定 GPU UUID、路径、协议、源码、两套实际 gate 和证据文件。
子入口还核验 Linux 进程祖先关系和仍存活的控制器，单独手设 `PAIR=1` 无法跳过入口门禁。
这些是防混用/陈旧证据的工程检查，不是针对主动改包用户的安全隔离。

数据绑定包含 train/val 标签原始字节 SHA256、官方验证原始 annotation SHA256 和数据 YAML 原始字节 SHA256。
训练/验证图像绑定相对文件名、字节大小和 mtime_ns；没有读取所有图像像素求 SHA256，
因此保留大小和时间戳的图像内容改写无法被该 inventory 检出。数据缓存文件不进入签名。
源代码还绑定真实 trainer、validator、检测 train/val 和 dataset/build/utils 路径；改代码、标签、
图像 inventory、数据 YAML、GPU 或训练协议后必须重新 check/smoke。
本地缺少真实数据的 CPU 检查明确标为不可用于 CUDA gate。

根 `verify` 先由各子入口重新计算当前签名，再复核同次并行回执。
确认通过后可正式后台启动：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/run_pair.sh run > launcher_e40s3_e47s1.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e40s3_e47s1.log
```

根 `run` 再次执行 `verify` 的全部检查，并在启动两子进程前拒绝占用的正式目录：

- `runs/e40a_spatial_task_head_img640_seed3_200e/`
- `runs/e47a_p2_depth_readout_img640_seed1_200e/`

后一目录与已交付的独立 E47 使用同名正式 run。若独立 E47 已运行或已有该目录，
本包拒绝覆盖；须先保留并检查已有任务结果，不能靠改名或删除来绕过实验身份检查。
项目级根锁和两个唯一子锁拒绝重复控制器/训练入口。一个正式任务失败后另一项继续，
最终根退出码非零；不杀现有训练、不自动关机。

控制器每 30 秒输出两项 epoch、ETA、退出码和显存；根及两个子日志路径会在启动时打印。
状态 JSON 和日志保存在 `logs/pair_e40s3_e47s1_run_时间/`，smoke 使用对应 smoke 目录。
可查看最近一次状态：

```bash
bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/run_pair.sh monitor
```

也可分别跟随启动时打印的 `e40_run.log` 和 `e47_run.log`：

```bash
tail -f /绝对路径/logs/pair_e40s3_e47s1_run_时间/e40_run.log
tail -f /绝对路径/logs/pair_e40s3_e47s1_run_时间/e47_run.log
```

`Ctrl+C` 退出 tail 不会停止后台训练。若控制器被中断，其 state 会写出 `controller_interrupted`、
PID 和日志；子进程可能继续，检查这些 PID/日志后再决定如何处理，不直接重启另一份。
训练回报位于 `comparison_reports/E40S3_RUN_时间/`、`comparison_reports/E47P_RUN_时间/`；
每项有独立官方评分与完成/失败记录。已有 200 轮权重而评分失败时，可保留权重单独补评：

```bash
bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/e40/run_e40.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e40a_spatial_task_head_img640_seed3_200e/weights/best.pt
bash /root/autodl-tmp/E40S3_E47S1_Parallel_AutoDL/e47/run_e47.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e47a_p2_depth_readout_img640_seed1_200e/weights/best.pt
```

本地纯标准库控制器测试可运行 `python test_pair_controller.py`。
它验证证据混用拒绝、context 身份、显存边界、启动/失败/中断状态、另一任务继续等逻辑，
不执行真实 CUDA smoke、远程双开或 200 轮训练。交付的本地验证范围另见 `LOCAL_TESTS.md`。
