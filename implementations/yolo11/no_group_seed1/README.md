# E58 / E59：E57母体的两个独立200轮实验

- E58：P2–P5增强分支 hidden48→96；原检测塔通道不变，router瓶颈32不变。
- E59：完整保留E57 hidden48，在P2/P3增加独立通道—深度选择及独立输出投影。
  `Y_task = X + P_spatial(U_spatial) + P_channel(U_channel)`。
  两路分别沿depth归一化，不使用E42的softmax(S+C)。不保证两路因果独立或收益相加。
- 两项都从同一原始yolo11s.pt重新初始化训练，不微调E40/E57 best。
- 固定 seed1、640、batch16、workers8、SGD、cosine200、close_mosaic10及原loss/TAL/NMS。
- 不修改共享Ultralytics，不覆盖已有报告，不自动关机，不静默改batch。

## 上传与解压

将zip上传到 `/root/autodl-tmp/`，执行：

```bash
cd /root/autodl-tmp
unzip -n E58E59_E57_W96_DualReadout_Parallel_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
```

这是新目录；若目录已经存在，不混合旧版与新版，先确认来源。

## 先check，再并行smoke

```bash
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/run_pair.sh check
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/run_pair.sh smoke
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/run_pair.sh verify-gates
```

check逐项实时输出；smoke同时运行两项，日志路径立即打印。每项至少8次真实batch16成功AMP更新，
全部新增参数组有检测loss梯度并实际更新；随后在双方训练模型/optimizer仍驻留时同步进行
EMA-half batch32方形压力测试与完整548图验证。初期AMP溢出跳步不算成功更新。

显存cap固定：E58=15GiB、E59=13GiB，smoke与formal相同。要求GPU空闲且至少29GiB可用。
这不是“32GB一定够”的保证。任何OOM/非有限值/更新或验证失败均不签发双开凭据；不要跳过。

smoke实时分别看：

```bash
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/watch_logs.sh e58 smoke
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/watch_logs.sh e59 smoke
```

## 门禁全通过后启动正式双开

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/run_pair.sh run > launcher_e58_e59.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e58_e59.log
```

另开两个终端分别查看训练：

```bash
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/watch_logs.sh e58
bash /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/watch_logs.sh e59
```

Ctrl+C退出tail不会停止nohup后台任务。不要在前台check/smoke控制器上按Ctrl+C；中断控制器不会
强杀子任务，但smoke无法签发完整双开凭据，须先确认旧进程退出再重试。

也提供 `run_pair.sh all` 自动依次check→并行smoke→复核→并行200轮，任一门禁失败即停止。
默认不自动恢复/覆盖已有训练目录。一个正式任务失败不杀另一个。

## 结果与补评

正式目录分别为：

```text
yolo11_project/runs/e58a_e57_allscale_w96_img640_seed1_200e/
yolo11_project/runs/e59a_e57_dual_readout_img640_seed1_200e/
yolo11_project/comparison_reports/E58_RUN_时间/
yolo11_project/comparison_reports/E59_RUN_时间/
```

正式结束自动评同一官方VisDrone val协议，报告同时比较E1/E40/E57。
完成标记区分训练200轮完成与官方评分完成。评分失败不重训，可单独补评：

```bash
/root/miniconda3/envs/yolo11/bin/python -u /root/autodl-tmp/E58E59_E57_W96_DualReadout_Parallel_AutoDL/run_experiment.py --variant e58 --mode eval --weights /root/autodl-tmp/yolo11_project/runs/e58a_e57_allscale_w96_img640_seed1_200e/weights/best.pt
```

E59对应替换variant和权重路径。服务器必须保留已有E1/E40/E57报告和best.pt，否则预检明确停止。
single-seed筛选门槛为超过E57官方AP至少0.25pp且AP50/AP75不下降，不等于统计显著或论文录用。

## 历史与创新边界

独立读出与早期未训练的E44差分预案属于同一函数族；实际训练的E44是局部通道校准，不是本配置。
E47测试静态深度对比读出，本项是图像条件通道—深度选择；不声称此前从未探索过读出。
公开空间/通道注意力和多分支选择需引用，例如CBAM、SKNet、Dynamic Head。
本包不证明E42失败根因为路由竞争、48维饱和、必涨0.5pp或两项可以直接组合。
