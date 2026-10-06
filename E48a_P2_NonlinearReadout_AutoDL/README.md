# E48a: P2 非线性联合深度读出（E47 → E48）

固定正式实验：640 / batch16 / workers8 / seed1 / 200 epoch / SGD / AMP / cosine / close_mosaic10 / patience0。
原始 yolo11s.pt + 确定性 E1 迁移重新训练；不读取 E40/E47 best 作为初始化，不从正在训练的 seed2 微调。

## 本轮只改变什么

保留完整 E47（包括 E40 的 P2/P3 编码库和路由及 E47 所有线性读出）。
只在 P2 分别增加分类、回归联合残差读出：

```
原 E47 三个任务加权、有正有负的深度对比 C1/C2/C3（每个48通道）
  → concat(144)
  → 1×1 Conv(144→32, 无bias)
  → SiLU
  → 1×1 Conv(32→128, 无bias，权重零初始化)
  → 加到原 E47 对应任务输出
  → 原 Detect 对应 cv3 / cv2 塔
```

P3 严格保留 E47；P4/P5、Backbone、Neck、TAL、DFL、原 loss、decode、NMS 不变。
不新增归一化、门控、采样、detach、新特征库、loss或原图读取。
新增 17,408 参数，卷积计算增加约 0.8913 GFLOPs@640（与 E47 同 profiler 口径，未计元素运算）。
40001/47001 公共初始化不变；新增支路用隔离 CPU RNG 48001，不改训练 CPU/CUDA RNG。
新增读出在 FP32 计算，保留实际注册 Conv、梯度、hooks 和原输出dtype；半精度EMA用无状态权重视图，不原地改参数dtype。

这检验的是“已有 P2 深度对比的联合非线性读出是否优于只做线性读出”。
D45 的同权重旁路只支持 P2 读出依赖，不能证明非线性不足是根因，也不能保证上涨。
零末层使初始预测和公共参数裁剪前梯度等价；新增梯度参与全局裁剪后，不能声称整个训练轨迹等价。
第一步首层loss梯度为0、末层非零是正确的；末层更新后首层必须出现非零loss梯度。
不把权重衰减造成的变化当作路径起效证据。未来论文的同容量线性对照仍可单独补充，不在本轮同时搜索。

## 1. 解压

将 ZIP 上传到 `/root/autodl-tmp`，只解压到新的 E48 包目录，不覆盖共享源码或 E47。

```bash
cd /root/autodl-tmp
unzip E48a_P2_NonlinearReadout_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
```

## 2. 现在独立做 CUDA check 与真实 smoke

```bash
bash /root/autodl-tmp/E48a_P2_NonlinearReadout_AutoDL/run_e48.sh check --allow-busy-gpu
bash /root/autodl-tmp/E48a_P2_NonlinearReadout_AutoDL/run_e48.sh smoke --allow-busy-gpu
```

必须分别看到 `E48 CUDA CHECK PASSED` 和 `E48 CUDA SMOKE PASSED`，退出码0才继续。
它们不是训练收益探针，不消耗20/90轮，也不会修改 E47。
smoke 至少8次成功真实 batch16/640 AMP 更新，最多32次尝试；30组新增分支必须有非零loss梯度和实际更新。
保留训练model/optimizer同时验证 FP16 EMA-like 副本：square batch32 前向 + 全548张验证、AMP loss、NMS和更新后推理。
真实运行进度直接输出，不藏在子日志里。

所有服务器 CUDA check/smoke/formal 固定 **13GiB allocator 上限**，不自动降 batch。
忙卡门禁需显式 `--allow-busy-gpu` 且开始时至少16.5GiB空闲；运行中至少留2GiB物理余量。
这是可控共享显存检查，不保证 E47 未来显存峰值；余量不足会拒绝，不杀任何进程。
CPU check 或本地8GiB CUDA单元测试不生成服务器门禁。禁止改凭据或跳过门禁。
若 smoke 超过固定预算，不能排队；先提交日志重新审查，不随意换batch/上限。

## 3. 两个门禁通过后挂后台接续

```bash
nohup bash /root/autodl-tmp/E48a_P2_NonlinearReadout_AutoDL/run_e48.sh queue > launcher_e48_after_e47_seed2.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e48_after_e47_seed2.log
```

看到 `E48 QUEUE ACTIVE` 才是队列成功挂上。仅显示 `[1] PID` 不表示队列有效。
Ctrl+C 只退出 tail，不停止 nohup 后台队列/训练；关本地电脑也不会主动终止后台任务。不自动关服务器。

队列规则：

- E47 seed2 训练、保存、官方评分失败都记录，不作为启动障碍。不要求 completion.json，也不伪造 E47 成功。
- 识别 `E47_Seed2_AutoDL/run_e47.py`、`all_steps.py` 及子树；fork workers不是独立训练，但活着仍等待。
- 旧执行锁若存在必须释放，缺失则使用进程+GPU检测，不报“Missing lock”。
- 所有旧进程/控制器退出，无GPU计算进程且至少30GiB空闲；两次15秒轮询稳定后，再复核门禁并启动E48。
- GPU查询异常记为未知并继续等，绝不当成GPU空闲。默认等36小时，不主动杀训练。
- E48 自身代码/数据/环境变化会使凭据失效并停止；原有E48目录存在则拒绝重复/覆盖。
- 队列及训练独立锁，防止挂两份。不会在队列中重跑smoke或下载修复依赖。

只读查看当前识别状态：

```bash
bash /root/autodl-tmp/E48a_P2_NonlinearReadout_AutoDL/run_e48.sh inspect-queue
```

若你的 E47 seed2 包并非默认路径，给 queue/inspect-queue 加 `--e47-package /实际/E47_Seed2_AutoDL`。
路径/env 可通过 `--project --source --data --dataset` 指定；check/smoke与队列必须一致。
不需要激活conda：shell固定使用 `/root/miniconda3/envs/yolo11/bin/python`。

## 结果与比较

训练权重/曲线：`yolo11_project/runs/e48a_p2_nonlinear_readout_img640_seed1_200e/`。
正式报告：`yolo11_project/comparison_reports/E48_RUN_时间/`，包含：

- `training.json`、`completion.json`（后者仅训练和评分全部成功才写）
- `official_metrics.json`、`comparison.json`、`REPORT.md`
- `area_predictions.json`、`gt_rows.json`
- 出错时 `failure.json`，不删除权重，不把训练成功与评分成功混成一件事。

同 D27/D38 的原图坐标、四图padding chunk、conf=.001 / iou=.7 / max_det500 / FP32预测 / 官方ignore规则。
同 seed1 的 E1/E40/E47 都列出，尤其看相对 E47 的 Official AP/AP50/AP75；不是拿 seed2 波动当 E48 增益。
E47 官方 AP 26.4020%，E40 26.2225%，E1 25.6312%；先复核父报告、权重SHA和数据映射。
内部筛选标记为相对E47 AP至少+.20pp且AP50/AP75不降，仅是单seed筛选参考，不影响完整200轮训练或论文显著性结论。
尺寸覆盖是输入尺度下几何候选存在诊断，不是官方尺度AP或一对一Recall。

评分单独恢复（已有训练目录不会重训）：

```bash
bash /root/autodl-tmp/E48a_P2_NonlinearReadout_AutoDL/run_e48.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e48a_p2_nonlinear_readout_img640_seed1_200e/weights/best.pt
```

本地验证记录见 `LOCAL_TESTS.md`。必须以服务器本轮门禁为准，没有GPU峰值/收益保证。
