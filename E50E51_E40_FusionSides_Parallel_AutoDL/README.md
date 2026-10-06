# E50 + E51：完整 E40 上两个独立实验

- E50：原网格联合核生成，替换 model23 的 P3 donor 重组；保留原 Concat24。
- E51：保持原 model23 最近邻，只处理 Concat24 的 backbone-P2 副本；原主干张量不改。
- 完整 E40、原 Detect/TAL/loss/decode/NMS 保留；两项独立，不先组合。
- 原始 yolo11s.pt 迁移重新训练；不从 E40/E49 best 微调。公共 E40 隔离40001，核网络隔离49001，gain=0，全局seed1。
- 640 / batch16 / workers8 / 200轮 / 原余弦日程 / close_mosaic10。每进程13GiB allocator上限，与smoke一致，不静默降batch。

## 上传解压与启动

将同名zip完整上传到 /root/autodl-tmp，保留现有E1-640 seed1报告/预测、E40 seed1报告和best.pt、数据与原始预训练。

```bash
cd /root/autodl-tmp
unzip E50E51_E40_FusionSides_Parallel_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E50E51_E40_FusionSides_Parallel_AutoDL/run_pair.sh check
bash /root/autodl-tmp/E50E51_E40_FusionSides_Parallel_AutoDL/run_pair.sh smoke
bash /root/autodl-tmp/E50E51_E40_FusionSides_Parallel_AutoDL/run_pair.sh verify-gates
nohup bash /root/autodl-tmp/E50E51_E40_FusionSides_Parallel_AutoDL/run_pair.sh run > launcher_e50_e51.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e50_e51.log
```

确认旧训练已退出：nvidia-smi。此包不停止其他实验。
check依次运行并实时显示输出/耗时；CPU通过不能代替服务器CUDA通过。
smoke同时执行真实batch16训练，至少8次有限成功更新，最多32次尝试；AMP跳步不计成功。
两任务在训练结束屏障后一起进入EMA-half batch32平方压力及全548图验证，训练模型/优化器保持存活。
验证loss使用AMP累计，不修改loss公式、不clamp/nan_to_num。代码/环境/配置/数据变更须重做门禁。
只有并行smoke通过才允许正式双开；若13GiB不足，停止，不降batch/分辨率或绕过凭据。
13GiB是PyTorch allocator限制，不是全部进程显存硬上限；真实余量另查。

## 分开监控

控制日志打印本次唯一 logs/pair_e50_e51_run_时间 目录，在两个终端分别：

```bash
tail -f /root/autodl-tmp/yolo11_project/logs/pair_e50_e51_run_实际时间/e50_run.log
tail -f /root/autodl-tmp/yolo11_project/logs/pair_e50_e51_run_实际时间/e51_run.log
```

这里Ctrl+C只退出tail，nohup训练不依赖本地电脑在线。不要中断前台check/smoke。
一个正式子任务失败不强杀另一个；不默认关机。双开每项可能变慢，不承诺睡醒完成，按实际ETA判断。

## 结果与补评

权重目录：

```text
runs/e50a_e40_native_grid_reassembly_img640_seed1_200e/weights/{best,last}.pt
runs/e51a_e40_local_contrast_img640_seed1_200e/weights/{best,last}.pt
```

报告：comparison_reports/E50_RUN_时间、E51_RUN_时间。
包含 training.json、transfer.json、area_predictions.json、gt_rows.json、official_metrics.json、comparison.json、REPORT.md。
完全成功有completion.json；异常有failure.json。训练完成与评分完成分别记录，评分失败不重训：

```bash
bash /root/autodl-tmp/E50E51_E40_FusionSides_Parallel_AutoDL/e50/run_e49.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e50a_e40_native_grid_reassembly_img640_seed1_200e/weights/best.pt
bash /root/autodl-tmp/E50E51_E40_FusionSides_Parallel_AutoDL/e51/run_e49.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e51a_e40_local_contrast_img640_seed1_200e/weights/best.pt
```

内部run_e49.py/sh名称是复用公共执行器，实际实验标签、路径、算子、凭据为E50/E51，不运行旧E49。
主对照E40同seed，累计收益另列E1。AP≥E40+0.25pp且AP50/AP75不降才考虑seed2；不是显著性或投稿承诺。

## 数值与创新边界

3×3 replicate邻域、跨通道共享核、FP32 softmax/逐邻域累计，不构建完整P2 C×9 unfold；输出返回输入dtype。
零残差保证初始预测及公共裁剪前梯度对齐，不保证后续训练轨迹相同。
借鉴CARAFE、FADE、FreqFusion，不宣称新的重组/高通原理；E51有符号gain不是固定线性纯高通，可能增强噪声。
独立核对Conv算术及共享DW两次调用。Profiler未计功能性邻域运算不是零成本，另列混合估计。
本地测试见LOCAL_VALIDATION.md；本地CPU/小CUDA/合成集成不能替代32GiB服务器真实并行smoke。

