# E44-LCC / E45-RDR：E40 上的两个独立并行正式实验

本次为v2，整包复核和本地验证见`AUDIT_V2.md`。保留原包，不覆盖；新目录重新check和smoke，旧凭据不能复用。

固定 seed1 / imgsz640 / batch16 / 200 epochs，close_mosaic=10，patience=0。
从同一原始yolo11s.pt做精确E1迁移，不从E40 best.pt微调；原E40公共参数初值和RNG保持一致。
本包不安装依赖、不写共享Ultralytics源码、不复用E42/E43门禁、不默认关机。

## 模块与科学边界

E44：原E40空间深度选择后，分类/回归各自的48通道表示做局部逐通道校准。
1×1(48→16)→SiLU→DW3×3→SiLU→1×1(16→48)，2sigmoid，末层零初始化，初始gate=1。
P2/P3四个校准器，共增加6912参数。只在任务残差表示上操作，不缩放整个原输入。

E45：原F2/F3串行输入保留，增加浅编码复用的线性零初始化1×1输入残差。
Z2=F2(Z1+J2[Z0,Z1])，Z3=F3(Z2+J3[Z0,Z1,Z2])，P2/P3同时采用，共增加23040参数。
Z0..Z3是同分辨率编码深度，不是P2..P5。两项不同维度不代表已证明互补，不预先组合。

仅新分支采用FP32运算以保护小的零起点权重/梯度；输出恢复输入特征dtype。
原E40编码、router、预测塔和正式AMP协议不变，无detach/新loss/新assignment算法。
CPU合成测试不证明CUDA安全或涨点；两项都是待验证结构假设，不是D44确诊根因后的修复。

## 1. 上传zip并解压

把 E44E45_E40_LCC_RDR_Parallel_AutoDL_v2.zip 上传到 /root/autodl-tmp/ 后：

```bash
cd /root/autodl-tmp
unzip E44E45_E40_LCC_RDR_Parallel_AutoDL_v2.zip
cd /root/autodl-tmp/yolo11_project
```

不要把文件挪进yolo11_project或更改包内代码。解压保留完整包目录。
固定解释器 /root/miniconda3/envs/yolo11/bin/python，因此base/yolo11提示符不影响运行。

## 2. 先检查，再并行真实smoke

先确保之前的训练已经结束，GPU空闲；脚本不杀其他进程。

```bash
bash /root/autodl-tmp/E44E45_E40_LCC_RDR_Parallel_AutoDL_v2/run_pair.sh check
bash /root/autodl-tmp/E44E45_E40_LCC_RDR_Parallel_AutoDL_v2/run_pair.sh smoke
```

check依次执行两项，直接实时输出，不要Ctrl+C中断。CPU检查不能代替服务器CUDA检查。
smoke同时测试两项真实batch16训练更新，保留训练模型/optimizer时同步测试EMA式half batch32验证：548图及最坏square batch32。
最多32个真实batch，必须至少8次成功AMP更新，并检查全部新路径有成功步loss梯度与实际参数变化。
AMP overflow skip不算成功更新；持续失败不给凭据，不自动降batch。
两任务分别14 GiB allocator上限，与正式运行一致；GPU空闲至少28GiB才启动并行。
不保证200轮不会OOM。若并行smoke失败，停止，不绕门禁、不直接formal。
终端必须显示 `FINISHED smoke: {'e44': 0, 'e45': 0}`，且无failure.json。

## 3. 正式并行200轮

只在两项check和同一次并行smoke成功后执行一次：

```bash
nohup bash /root/autodl-tmp/E44E45_E40_LCC_RDR_Parallel_AutoDL_v2/run_pair.sh run > launcher_e44_e45_v2.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e44_e45_v2.log
```

nohup启动后，此tail的Ctrl+C只退出日志查看；关闭本地电脑不终止远程任务。
前台check/smoke与nohup启动前的操作不同，不要混淆。
一项正式训练/评分失败时控制器展示其末尾traceback，另一项继续。每项独立lock，拒绝重复启动/覆盖已有run。

两个终端分别看实际epoch/loss/速度/ETA：

```bash
bash /root/autodl-tmp/E44E45_E40_LCC_RDR_Parallel_AutoDL_v2/watch_logs.sh e44
bash /root/autodl-tmp/E44E45_E40_LCC_RDR_Parallel_AutoDL_v2/watch_logs.sh e45
```

进程/显存只读：`nvidia-smi`。上一对并行正式实验约14～15小时，不保证睡醒完成。

## 4. 结果及官方评分

各自目录：

```text
yolo11_project/runs/e44a_local_channel_calibration_img640_seed1_200e/
yolo11_project/runs/e45a_residual_depth_reuse_img640_seed1_200e/
yolo11_project/comparison_reports/E44_RUN_时间/
yolo11_project/comparison_reports/E45_RUN_时间/
```

每项自动原best验证、548图FP32官方预测/ignore评分、原E1评分复现、与既有E40比较。
包含transfer.json、training.json、completion.json、area_predictions.json、gt_rows.json、comparison.json、official_metrics.json、REPORT.md。
只有completion.json.status=completed且official_evaluation_completed=true才是整项完成；200轮训练完成不等于评分成功。
评分失败保留best/last，修复后可单独eval，不需重训：

```bash
bash /root/autodl-tmp/E44E45_E40_LCC_RDR_Parallel_AutoDL_v2/e44/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e44a_local_channel_calibration_img640_seed1_200e/weights/best.pt
```

E45同理使用e45路径。单独eval生成E44_EVAL_/E45_EVAL_报告，不伪造原训练完成凭据。
依赖既有E1 seed1正式报告和E40 seed1正式报告/权重；缺失时报明确错误，不换对照。
资源筛选目标AP>=26.44%、AP50/AP75不低于E40；主要看delta_vs_E40_pp，而不只看vs E1。
这是单seed算力筛选，不是统计显著或发表保证。input-size raw coverage不是小目标AP。

## 本地测试边界

test_release.py：共同源/协议、独立公式与梯度、弱权重数值、初始化、half值、AMP跳步、并行凭据、门禁和源码哈希等。
服务器仍需执行上面的实际CUDA check与并行真实smoke；本地CPU结果不可替代。
完整设计在工作区 E44_E45_REVISED_POST_D44_PLAN_20261001.md；不宣称首次发明残差、Dense、注意力或任务分解。
