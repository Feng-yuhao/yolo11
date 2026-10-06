# D44：E40 路由 / E42 通道 / E43 Dense 局部干预

仅使用已有 best.pt 做诊断，不训练、不更新权重、不改 Ultralytics 源码、不启动 E44/E45、不关机。
默认检查 E40 seed1、E40 seed2、E42 seed1、E43 seed1，各模型运行在独立进程中，防止同名模块串用。
所有原模块定义、D27 官方评分和 D38 推理参数代码均随包附带；无需安装进源目录。

## 解压与独立检查

上传 zip 到 `/root/autodl-tmp` 后：

```bash
cd /root/autodl-tmp
unzip -n D44_E40_RoutingDense_Diagnostic_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/D44_E40_RoutingDense_Diagnostic_AutoDL/run_d44.sh check
bash /root/autodl-tmp/D44_E40_RoutingDense_Diagnostic_AutoDL/run_d44.sh smoke
```

先等 check 完成再运行 smoke。两者会实时输出每个模型的进度。
check 包括非零残差下独立公式核对、未选尺度一致、异常退出恢复、缓存头重算与原前向一致、
NMS clone 和权重不变检查；smoke 是每模型真实验证集前 8 图，核对默认预测与正式保存预测。
没有 optimizer/backward，没有 2 epoch 训练。8 图不评分、不淘汰性能。
run 必须有同代码/权重/环境/数据/评分协议的 CUDA smoke 凭证；CPU check 不冒充它。

## 后台全量诊断与实时日志

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/D44_E40_RoutingDense_Diagnostic_AutoDL/run_d44.sh run > launcher_d44_routing_dense.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_d44_routing_dense.log
```

此处 Ctrl+C 只退出 tail；后台诊断继续。不要在前台 check/smoke 中按 Ctrl+C 后以为已经通过。
默认逐模型运行，不是四个模型同时占显存。FP32 / imgsz640 / batch4 / conf0.001 / iou0.7 / max_det500。
每批只计算一次 Backbone/Neck，再重算三个头状态；会显示图像进度、推理 ETA 和 reserved 显存。
评分阶段是 CPU 官方 AP，会显示当前模型、干预名和 class1/10..10/10；这不是重新训练。
默认 PyTorch 分配上限 6 GiB，并要求开始时另有 1 GiB 空闲余量；不杀别的任务、不自动降低 batch。
若实际 OOM 就停止该诊断，不颁发凭证，不改在跑的训练任务。

## 检查内容

| 模型 | 原状态 | 两项干预 |
|---|---|---|
| E40 seed1/seed2 | normal | uniform_depth：四深度平均；shared_task：两任务路由权重平均 |
| E42 seed1 | normal | channel_off_p2 / channel_off_p3：只把指定层新增通道 logits 置零 |
| E43 seed1 | normal | dense_bypass_p2 / dense_bypass_p3：只在指定层绕过 fuse2/fuse3，保留其余已训练参数 |

不同模型不互相交换权重。P4/P5 原预测塔保持原样，局部干预也不删除整条任务分支。
每项 AP 差值仅比较同一 checkpoint 的 normal，不能把与另一个已训练模型的差值归给局部干预。
默认预测先逐图复现原结果（坐标/分数 atol1e-5），全量默认官方四项指标与原报告差值须 <=0.002pp。
复现失败立即停止，禁止绕过此门禁后分析改进方向。

## 输出文件

`/root/autodl-tmp/yolo11_project/comparison_reports/DIAG_D44_RUN_时间/`

- `REPORT.md`：四个 checkpoint / 十二个状态的官方 AP、AP50、AP75、AR500，以及自身配对变化。
- `decision.json`：有限的方向性提示，不自动批准训练或宣布根因。
- `controller.json`：完成状态、起止时间。
- `e40/`、`e40s2/`、`e42/`、`e43/`：各自 metrics/check、三个预测 JSON、三个官方评分 JSON。
- 每个模型的 `routing_rows.json`、`routing_summary.json`：GT 中心深度权重、任务 JS/熵、
  GT 中心 3×3 邻域残差/原输入 RMS；E42 附加联合权重相对原空间权重的变化。
- 每个模型独立日志；失败时保留 `failure.json`，不会伪造完成。

全量默认会有 E40 seed2 复核。若仅需前三种子1模型，可在 check/smoke/run 都加
`--models e40,e42,e43`，但报告不得声称跨 seed 的路由依赖已被复核。

## 科学边界

关掉后变好，支持投入一个受控训练对照，不证明训练退化根因；关掉后变差可能是协同适应/分布外失配。
熵大、JS 小、参数非零都不直接证明能力不足或模块有用。不要选择最好的干预作为正式论文新模型。
不重复 D38/D23 的全阶段候选缺失树，不训练短探针，不使用 test-dev。
输入 sqrt(area) 的尺寸分组不是官方 AP-small，不能把总 AP 当作小目标 AP。

本地工程验证范围见 `LOCAL_VALIDATION.md`；服务器真实 8 图/548 图必须由用户运行。
