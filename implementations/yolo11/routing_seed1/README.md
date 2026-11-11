# E60 路由消融：seed1，默认串行，可选并行

仅两项预先固定的消融，不是新优化模型：

- `uniform`：四尺度删除整个路由网络，四个增强深度固定各占1/4。
- `shared`：四尺度每个位置仅生成一组四深度权重，分类/回归共用。

两者均保留四尺度W96串行编码、P2/P3六个组内残差编码及其FP32保护、独立分类/回归输出投影、原Detect预测塔、TAL、损失、增强、SGD和200轮日程。全局seed1，从原始COCO yolo11s.pt重新初始化，非E60 best微调。共用E60参数逐项保持初值；shared仅将路由最后8个输出变4个，初值取原两任务logit卷积的均值，保存此规则。uniform实际删除51360参数，shared实际减少528参数，不保留假参数凑容量。

## 安装

上传zip和同名`.zip.sha256`到`/root/autodl-tmp`。SHA文件已用LF写入。

```bash
cd /root/autodl-tmp
sha256sum -c E60_Routing_Ablations_Seed1_AutoDL.zip.sha256 &&
mkdir -p /root/autodl-tmp/yolo11_project/experiment_code &&
test ! -e /root/autodl-tmp/yolo11_project/experiment_code/E60_Routing_Ablations_Seed1_Parallel_AutoDL &&
unzip -q E60_Routing_Ablations_Seed1_AutoDL.zip -d /root/autodl-tmp/yolo11_project/experiment_code
```

需要保留原E1 seed1报告/权重、E60 seed1报告（含actual_source.zip与预测）、原COCO weights/yolo11s.pt、VisDrone训练/验证数据及yolo11_src；不依赖已删除的E40/E41训练包或它们的报告。当前代码实体放在项目experiment_code内，不再依赖tmp外部源目录。

## 推荐：串行自动接续

已有E60在RTX4080 SUPER单开分别9.05、8.85、8.85小时。两项仍保留几乎全部编码计算，串行两项估计17–19小时加门禁/评分；不是保证。没有这两项同环境并行实测，不能承诺双开总时间更短。默认串行避免32GB双模型显存与计算争用，不降低batch/epochs。

```bash
code=/root/autodl-tmp/yolo11_project/experiment_code/E60_Routing_Ablations_Seed1_Parallel_AutoDL
bash "$code/run_pair.sh" check --execution serial
bash "$code/run_pair.sh" smoke --execution serial
```

两项门禁通过后：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash experiment_code/E60_Routing_Ablations_Seed1_Parallel_AutoDL/run_pair.sh run --execution serial \
  > launcher_e60_routing_ablations.log 2>&1 &
tail -f launcher_e60_routing_ablations.log
```

`all --execution serial`可自动串联check、smoke和正式训练。check/smoke失败不启动正式训练；正式uniform失败退出释放GPU后，仍运行已经通过门禁的独立shared。两项成功训练后统一串行官方val评估；一项训练/评分失败会明确记录，成功项仍评分/保留。绝不自动重训已存在runs、不隐藏负结果、不改用test-dev选结构。

## 可选并行（不是默认）

全部命令改为`--execution parallel`，必须重新运行check和联合smoke。串行门禁不能授权并行。

每进程15GiB allocator cap，双方真实batch16至少8次成功更新、48/52保留分支组实际梯度/更新，双方模型/优化器及双方EMA-half副本驻留下完整batch32验证548图与640平方输入。并行失败不自动降batch、不绕过门禁。若要改串行，重新跑串行check/smoke，不自动改变布局。

## 不刷屏的进度

```bash
/root/miniconda3/envs/yolo11/bin/python -u \
  /root/autodl-tmp/yolo11_project/experiment_code/E60_Routing_Ablations_Seed1_Parallel_AutoDL/watch_progress.py --follow
```

也可`--variant uniform`或`--variant shared`。Ctrl+C只结束这个查看器；后台nohup训练继续。服务器关机或停止实例会终止训练。

## 结果及评分

训练模型：

```text
runs/e60_ablation_uniform_img640_seed1_200e/
runs/e60_ablation_shared_img640_seed1_200e/
```

每次正式启动生成一个汇总结果根目录`comparison_reports/E60_ROUTING_ABLATIONS_RUN_时间_pid/`，含两项训练源快照、transfer、training、逐轮progress.log与两项官方eval子目录及总REPORT.md/summary.json。日志放`logs/e60_routing_ablations_run_时间_pid/`。通过`logs/latest_e60_routing_ablations.json`定位。check/smoke有各自独立目录，不是正式性能结果。

固定val-selected best；FP32/640/预测batch4/conf.001/NMS IoU.7/max_det500，复用冻结VisDrone原标注ignore计分。重新核验E1/E60原预测的官方AP/AP50/AP75/AR500后才报告差值。两个消融只seed1，不能宣称统计显著或稳定因果证明。uniform同时取消空间和任务自适应，shared检验任务专属权重，两个差值不能相加为独立贡献。

评分失败可单独`run_pair.sh eval --execution serial`恢复两项已有完整200轮权重的评分，不重训。

## 可选自动关机

默认不关机。只有本人明确在`run`或`all`命令后加`--auto-shutdown`才启用；最后评分/失败状态保存后，CPU helper等待控制器退出、GPU/其他训练评分进程释放，再倒计时60秒复查。只调用平台原shutdown命令，支持无shebang的ASCII脚本，不杀其他任务。创建项目内`KEEP_RUNNING_E60_ABLATIONS`文件可取消；命令返回不等于已确认停止计费。check/smoke不关机。

## 验证边界

见LOCAL_VALIDATION.md。本地合成小输入CPU/CUDA单测不等于服务器640真实数据和显存门禁，也不保证训练涨点。源码保存AGPL相关许可。
