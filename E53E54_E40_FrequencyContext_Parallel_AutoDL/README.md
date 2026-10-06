# E53/E54：完整 E40 的两个独立增量实验

不是联合模型，也不是从 E40 best 微调。两组都从原 E1 初始化重新训练，保留原 E40（40001）初始化，新增分支分别隔离53001/54001。固定640、batch16、workers8、seed1、200轮，原Detect/TAL/loss/训练增强不变。

- E53：原 neck model25 C3k2 后追加 hidden32、两级全网格 Haar 多频残差；LL上下文与三方向细节幅值生成软选择，实际细节保留符号；独立局部DW路径；仅32→128输出投影零初始化。
- E54：原 neck model16 C3k2 后追加 hidden48局部DW3与DW5→DW7(dilation3)上下文；空间softmax选择旁路/近/远；仅48→128输出投影零初始化。

E53 只直接改变 P2 Detect 输入。E54 的 model16 同时供 P2 top-down 和后续 bottom-up，不承诺其他检测尺度完全不变。

## 上传及解压

把 ZIP 上传至 `/root/autodl-tmp/`。保留完整目录，不把子目录合并到项目根、不覆盖 Ultralytics 源码：

```bash
cd /root/autodl-tmp
unzip E53E54_E40_FrequencyContext_Parallel_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
```

若同名目录已经存在，先确认是否为此包，避免 `unzip -o` 覆盖正在运行的代码。

## 门禁及正式双开

先确保没有其他训练占GPU。check依次跑；smoke真正同时跑两组，并等待双方完成训练更新和完整验证。13GiB每进程分配上限，smoke与formal相同，不自动降低batch。

```bash
bash /root/autodl-tmp/E53E54_E40_FrequencyContext_Parallel_AutoDL/run_pair.sh check
bash /root/autodl-tmp/E53E54_E40_FrequencyContext_Parallel_AutoDL/run_pair.sh smoke
bash /root/autodl-tmp/E53E54_E40_FrequencyContext_Parallel_AutoDL/run_pair.sh verify-gates
```

只有这些命令均成功后再启动：

```bash
nohup bash /root/autodl-tmp/E53E54_E40_FrequencyContext_Parallel_AutoDL/run_pair.sh run > /root/autodl-tmp/yolo11_project/launcher_e53_e54.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e53_e54.log
```

控制器打印两个 `e53_run.log`、`e54_run.log` 的完整路径；分别打开终端执行 `tail -f 对应完整路径`。
这里 Ctrl+C **仅停止 tail**。前台check/smoke的Ctrl+C会中断门禁；不要当日志查看来中断。没有自动关机。

可以在空闲时后台自动串联所有门禁和正式训练：将最后命令的 `run` 改成 `all`。失败即不进入正式训练，不跳过门禁。

## 报告 / 失败恢复

正式报告：`yolo11_project/comparison_reports/E53_RUN_时间/` 与 `E54_RUN_时间/`。

- `training.json`：200轮训练状态、开始/完成时间、epoch速度与残差统计。
- `official_metrics.json`、`comparison.json`、`REPORT.md`：原官方VisDrone评分，对照同seed E1及E40。
- `gt_rows.json`、`area_predictions.json`：几何覆盖及预测；不是官方AP-small。
- `completion.json`：只有训练及官方评分都成功才写入。
- `failure.json`：完整异常；评分失败不删除best/last。

一个正式任务失败不强杀另一个。控制器在双方退出后报告失败任务。门禁失败不能启动训练；OOM也不自动改batch。

补评分（将路径替换为对应组实际best.pt）：

```bash
bash /root/autodl-tmp/E53E54_E40_FrequencyContext_Parallel_AutoDL/e53/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e53a_e40_frequency_img640_seed1_200e/weights/best.pt
bash /root/autodl-tmp/E53E54_E40_FrequencyContext_Parallel_AutoDL/e54/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e54a_e40_selective_context_img640_seed1_200e/weights/best.pt
```

## 工程 / 科学边界

包包含独立Haar分析/伴随重建、不同halo边界对照、原损失/初始公共梯度、两步更新、真图路由、空GT、EMA-half、保存重载/fuse/标准Predictor、官方评分与TAL测试。服务器CUDA及并行真实数据smoke必须另外执行，不能用本地小测试替代。

新增参数仅18694 / 16665；少参数不等于少激活或更快。实测理论640主算子GFLOPs分别41.04324 / 40.13857（E40同口径39.92653）；归一化、Haar加减缩放、selection、数据移动不全计入传统MAC指标。实际显存和速度由smoke与正式日志确认。

借鉴小波多频建模和LSK局部/大核选择思想，不宣称首创这些算子、不保证抑制背景、不保证涨点或EI/SCI录用。只有正式同协议评分能判断单seed效果，多seed判断稳定性；不得把两组收益相加或直接拼权重。

相关工作：WTConv (ECCV 2024), https://arxiv.org/abs/2407.05848；LSKNet, https://arxiv.org/abs/2303.09030；SWT定义 https://pywavelets.readthedocs.io/en/latest/ref/swt-stationary-wavelet-transform.html。
没有拷贝第三方小波/LSK实现；原创工程实现与Ultralytics原许可分别说明于文件及LICENSE-Ultralytics.txt。
