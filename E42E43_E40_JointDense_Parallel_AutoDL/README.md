# E42a / E43a：E40 的两个受控改进（seed1，640，正式200轮）

A=E42a：保留 E40 编码库与空间路由，新增通道条件的深度 logits；仅深度归一化。
B=E43a：保留 E40 空间路由；Z2/Z3 用 1×1+GN+SiLU 读取前面全部编码特征。
两者是不同干预轴，不保证正交或组合收益。新增参数分别213,072/198,864（相对E1）。
全部从原始 yolo11s.pt 严格迁移初始化，不从任何训练好的候选模型微调。
P2/P3入口改动；P4/P5、主干、Neck、原cv2/cv3/DFL、TAL算法、损失、增强均不改。
运行期间不编辑共享Ultralytics，不覆盖旧模型/结果，不自动关机。

## 上传解压

上传 ZIP 到 /root/autodl-tmp。第一次解压使用下面命令；不要向已运行的同名包覆盖解压。

```bash
cd /root/autodl-tmp
unzip -n E42E43_E40_JointDense_Parallel_AutoDL_v4.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh check
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh smoke
nohup bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh run > launcher_e42_e43.log 2>&1 &
tail -f launcher_e42_e43.log
```

check 应显示两项 CUDA CHECK PASSED；smoke 应显示两项 CUDA SMOKE PASSED 且控制器 FINISHED smoke exits均0。
没有通过不得启动run。CPU检查不是CUDA回执。Ctrl+C退出tail不会结束nohup后台训练。
控制器会打印每项真实训练日志的绝对路径；分别 `tail -f 该路径` 看批次速度。
启动后每30秒显示双方 epoch 和 GPU显存；每轮模型日志也显示时间/ETA。

## 显存/安全边界

训练batch16，验证batch32，640，workers8，SGD/AMP，200轮，patience0，close_mosaic10。
并发smoke包含双方同步进入的完整548图FP16 EMA验证、验证loss/NMS和额外square batch32。
v2修正：最多32个真实训练批次，等待至少8次成功优化更新；AMP overflow_skip不计入成功次数。
v3修正：E42通道MLP单独使用FP32池化/投影/反向，避免末层微小master权重被autocast舍入为0。
v4修正：验证 forward 仍为FP16；原始验证loss在AMP上下文中计算，避免纯FP16大规模BCE求和溢出。
第一验证批次记录原FP16调用与AMP调用的loss对照；原始logits、解码结果、loss、NMS分别检查。
非有限预测/AMP loss仍拒绝发回执，绝不clamp或nan_to_num；失败写出VALIDATION_FAILURE.json。
本改动只作用于工程smoke，不修改正式Trainer、validator、主loss公式或官方AP计算。
两套check新增640/batch2真实YOLO损失下的CUDA分阶段梯度检查，不能替代batch16真实数据smoke。
控制器遇到失败会直接打印对应日志末尾的traceback，不再只显示exit=1。
持续溢出、非有限loss、梯度缺失、参数未更新或显存超限仍拒绝发回执，不改变正式GradScaler或训练参数。
smoke和正式均固定14 GiB/进程的PyTorch分配器上限；空GPU至少28 GiB可用才启动。
留至少2 GiB实际空闲量，不静默降batch、不改分辨率。上限不覆盖所有CUDA库外部分配；短检查不能保证200轮绝无OOM。
一项正式失败不会杀另一项；控制器等待另一项退出后整体返回失败，并记录state.json。
smoke失败没有回执。要改为串行时，应先分别运行各自check/smoke，不能跳过回执。
两个独立smoke不需要同步barrier：`bash .../e42/run.sh smoke`、`bash .../e43/run.sh smoke`。
之后 `nohup bash .../run_pair.sh serial > launcher_e42_e43_serial.log 2>&1 &`。
串行仅解决两项合计峰值问题；如果某项自身超过14 GiB上限，当前串行入口也会拒绝。
这时需要明确调整分配器预算并重新check/smoke，不允许直接跳过门禁或偷偷降低batch。

## 结果/补评

自动训练结束后使用原D27官方ignore协议，分别评测并与E1 seed1及E40 seed1对比。
每项结果在 comparison_reports/E42_RUN_* 或 E43_RUN_*：
REPORT.md、training.json、transfer.json、official_metrics.json、comparison.json、area_predictions.json、gt_rows.json。
权重在 runs/e42a_joint_spatial_channel_task_head_img640_seed1_200e/weights 和对应e43目录。
评分导入/IO失败不重新训练：

```bash
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/e42/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e42a_joint_spatial_channel_task_head_img640_seed1_200e/weights/best.pt
```

E43同理。以Official AP/AP50/AP75作为主指标；raw coverage不是AP/一对一召回。
预算筛选：相对E40 AP至少+0.20 pp、AP50/AP75不下降，再补同seed2；不是显著性或论文录用门槛。
不能将P2收益或单seed改善冒充新增模块的稳定贡献。

## 已知机制风险

A的GAP通道描述可能受背景影响，联合权重也可能退化；C末层零初始化使前层最早第三次成功更新才可能获得梯度，数据/精度条件会继续延迟它。
B的编码可能更相关而非更多信息；原E40已经保留四深度特征，不能说此前完全丢失浅层。
保留原算法不意味着训练时匹配和共享网络轨迹不变。DenseNet/TOOD/DynamicHead思想需引用，不宣称概念首创。
本地验证与服务器CUDA回执分别报告；资源/效果都不保证。

已安装旧包时请先确认旧check/smoke/正式进程全部退出，再用 `unzip -o E42E43_E40_JointDense_Parallel_AutoDL_v4.zip` 更新。
代码签名已改变，旧回执不能复用。此次修复及验证边界见 V4_VALIDATION_FIX.md；此前弱梯度修复见 V3_AUDIT.md。
