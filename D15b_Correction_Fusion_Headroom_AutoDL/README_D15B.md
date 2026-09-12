# D15b：纠错选择—融合上界诊断

## 本轮只回答一个决策问题

D15a 已证明动态 1.5 块/图可以获得 `mAP +2.445 pp`、`AP-small +3.422 pp`、`AR-small +5.439 pp`，但在置信度 0.25 时同时新增 2611 个 TP 和 3918 个 FP。

D15b 将下一步拆成两个互斥方向：

```text
方向 A：区域是否还选得不够准？ → 纠错效用路由器
方向 B：区域已经选对，但局部框误检太多？ → 跨视图候选校准器
```

它冻结 E1，一次缓存四块预测，不训练、不修改 Ultralytics、不修改 checkpoint、不保存大体积预测 JSON、绝不自动关机。

## 模式

- `proxy_*_standard`：当前可部署 D15a 代理与普通 NMS；
- `oracle_utility_*_standard`：使用验证标签选择能实际恢复小目标、同时少引入 FP 的区域，只是选择上界；
- `proxy_b1p50_oracle_validity`：区域仍由当前代理选择，但验证标签过滤局部 FP/重复框，只是融合上界；
- `oracle_utility_b1p50_oracle_validity`：选择和融合都使用标签，是联合理论上界。

所有带 `oracle` 的结果都不能作为论文方法精度，只能决定应该开发哪个组件。

## 固定门槛

- 选择分支：纠错 Oracle 相对当前动态 1.5，`mAP ≥ +0.20 pp` 或 `AP-small ≥ +0.50 pp`；
- 融合分支：Oracle validity 相对当前动态 1.5，同时满足 `mAP ≥ +0.30 pp`、`AP-small ≥ +0.50 pp`；
- 两者都过门槛时，优先开发 AP-small 上界更大的分支；
- 两者都不过门槛，停止学习式区域组件。

## AutoDL 命令

上传 ZIP 到 `/root/autodl-tmp/` 后：

```bash
cd /root/autodl-tmp
unzip -o D15b_Correction_Fusion_Headroom_AutoDL.zip

conda activate yolo11
python /root/autodl-tmp/D15b_Correction_Fusion_Headroom_AutoDL/install_d15b_headroom.py

cd /root/autodl-tmp/yolo11_project
python -u diagnose_d15b_correction_fusion_headroom.py --check-only
python -u diagnose_d15b_correction_fusion_headroom.py --smoke-images 8
```

两项通过后执行完整诊断：

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash run_d15b_correction_fusion_headroom.sh > launcher_d15b_headroom.log 2>&1 &
tail -f launcher_d15b_headroom.log
```

结束标志：

```text
D15b completed. Report: .../metrics.txt
Automatic shutdown is disabled; the instance remains running.
```

同步整个 `DIAG_D15B_e1_correction_fusion_headroom_*` 报告目录。只根据完整 548 张结果中的 `selected_branch` 决策，smoke 子集不参与研究判断。
