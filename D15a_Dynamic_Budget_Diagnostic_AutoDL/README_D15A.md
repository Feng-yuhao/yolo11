# D15a：冻结 E1 的动态区域预算诊断

## 为什么下一步是它

现有证据中，唯一达到“百分点级、机制明确、重复可解释”的增益来自区域放大：

- E1 全图到固定四块融合：`mAP50-95 +3.03 pp`，`AP-small +3.81 pp`；
- D2 的预测引导 Top-2 已保留四块增益的约 87%/96%；
- E9b—E14a 的门控、平滑、注意力、蒸馏和局部语义模块都只有噪声级变化，或明显降低召回。

因此 D15a 不训练新模块，而先回答一个必须回答的问题：能否按图像难度动态分配 0/1/2 个局部块，在更低平均计算量下保留 Top-2 的小目标收益。

## D15a 做什么

```text
冻结 E1 全图预测
        │
        ├─ 四个区域的小目标/不确定性证据分数
        │
        └─ 每张图的第 1、第 2 块边际价值
                         │
           全局预算排序（不用标签/AP）
                         │
               每图选择 0 / 1 / 2 块
                         │
          原图裁剪→1024→冻结 E1→映射回原图→NMS
```

四个局部块只推理一次并缓存，随后复用到全部策略，避免不同策略重复推理造成浪费或随机差异。一次完整诊断会比较：

- 固定 Top-1、固定 Top-2、固定四块；
- 动态平均预算 0.50、0.75、1.00、1.25、1.50、1.75 块/图；
- 同预算下 `dynamic 1.00` 是否优于 `fixed Top-1`；
- `dynamic 1.50` 是否保留至少 85% 的固定 Top-2 增益。

D15a 不修改 Ultralytics，不训练、不写 checkpoint、不保存几十 MB 的预测 JSON，也绝不自动关机。

## 科学边界

D15a 是路线决策实验，不是可直接写成论文创新的最终方法。当前阈值只根据预测分数分布满足计算预算，没有查看标签或 AP；但它仍在验证集上完成预算校准。若门禁通过，E15a 必须在训练集上学习“纠错效用”（恢复 FN、纠正分类，扣除新增 FP 和计算成本），并在训练集上冻结阈值后只评估一次验证集。

## AutoDL 执行

先把整个 `D15a_Dynamic_Budget_Diagnostic_AutoDL` 文件夹或 zip 上传到 `/root/autodl-tmp/`。

```bash
conda activate yolo11
python /root/autodl-tmp/D15a_Dynamic_Budget_Diagnostic_AutoDL/install_d15a_dynamic_budget.py

cd /root/autodl-tmp/yolo11_project
python -u diagnose_d15a_dynamic_zoom_budget.py --check-only
python -u diagnose_d15a_dynamic_zoom_budget.py --smoke-images 8
```

两项均通过后运行完整诊断：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash run_d15a_dynamic_zoom_budget.sh > launcher_d15a_dynamic_budget.log 2>&1 &
tail -f launcher_d15a_dynamic_budget.log
```

结束标志：

```text
D15a completed. Report: .../metrics.txt
Automatic shutdown is disabled; the instance remains running.
```

同步回本地时至少保留完整的 `DIAG_D15A_e1_dynamic_zoom_budget_*` 报告目录。重点看 `metrics.txt` 中的 `recommended_next_action`，不要用 smoke 子集的 AP 做研究判断。

## 预先冻结的门禁

1. D1/D2 的 full、Top-1、Top-2、四块锚点误差都不超过 0.2 pp；
2. 固定 Top-2 相对 full 至少达到：`mAP +2.0 pp`、`AP-small +3.0 pp`、`AR-small +3.5 pp`；
3. 动态 1.00 块/图相对固定 Top-1 至少：`mAP +0.1 pp`、`AP-small +0.2 pp`，且 `AR-small` 不下降超过 0.2 pp；或者
4. 动态 1.50 块/图至少保留固定 Top-2 的 85% `mAP/AP-small` 增益，同时满足 `mAP +2.0 pp`、`AP-small +3.0 pp`、`AR-small +3.0 pp`。

只有满足这些预先写死的条件，才进入 E15a 可学习纠错效用路由器。否则保留区域放大作为系统上界，但停止继续为错误的动态代理投入训练成本。
