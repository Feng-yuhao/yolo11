# v4：验证 loss 精度检查修复（不是再次更改模块）

## 服务器实际日志证明了什么

最新附件中，E42/E43 两项 CUDA check 已通过；真实训练 smoke 各完成8次有限优化更新，
全部18组新参数有非零loss梯度及参数更新。双方随后在验证的第一个真实batch32退出。
v3把预测非有限、loss非有限合成一个报错，没有记录具体张量，是检查实现遗漏。
**旧日志不能区分本次服务器失败的具体非有限值；不把以下控制复现冒充失败批次的复现。**

## 已确认的检查缺陷

v3的工程验证代码让模型以FP16推理，然后在autocast之外调用原始loss。
Ultralytics 8.3.0 的v8DetectionLoss包含 `BCEWithLogitsLoss(...).sum()`；
纯FP16输出的求和可能先溢出，后续除以target_scores_sum也无法恢复。
这与训练时AMP自动使用FP32计算BCE/CE不同。

使用仓库**未改动的原E1 loss**、batch32、640对应P2–P5共34000个anchor、10类、
有限的FP16分类logit=-4、空GT，本地CUDA控制复现：

```
原FP16调用 loss items = [0.0, inf, 0.0]
AMP调用 loss items   = [0.0, 98735.203125, 0.0]
所有raw logits有限
```

该构造用于数值回归测试，不是模型效果实验，也不是实际服务器失败GT/预测。
数值与独立FP32 BCE求和公式一致；另以32张图各一个非空GT检查原CIoU/DFL/TAL loss有限。

## 修复范围

- 模型验证forward仍为纯FP16，batch32、640及完整548图/NMS检查保留。
- 仅原loss调用进入AMP上下文，不替换loss公式、GT、assignment、权重或预测。
- 分开验证每尺度raw logits、解码框/分数、总loss/三个loss分项、NMS输出。
- 第一个实际验证batch记录 `VALIDATION_PRECISION.json`（native_FP16与AMP分项、shape、GT数量）。
- 任意实际预测或AMP loss非有限仍停止；写 `VALIDATION_FAILURE.json`，包含阶段、shape/dtype、NaN/Inf数量和图片路径。
- 无nan_to_num、无输出裁剪、无放宽梯度/参数更新要求、无自动降batch。
- 每项全部通过才发新回执；旧回执因代码签名变更而失效。

## 没有改变的内容

E42/E43模块、参数初始化、FP32通道MLP修复、optimizer、200轮schedule、训练batch16、
seed1、数据增强、原检测loss、正式Trainer/validator、D27官方评分全部保持v3。
回归测试对v3归档的核心代码做字节对照，对原v1正式训练函数做AST对照，均通过。
正式训练没有套用这次工程验证helper，不涉及更改历史baseline评分或训练规则。

## 已执行与未执行

本地torch2.6/CUDA：13项自动回归测试通过，包括上面的完整batch32原始loss数值复现、
非空GT、故意坏预测/坏raw logits/坏loss拒绝测试及原弱梯度FP32 oracle。
v3服务器torch2.5的CUDA check和8次真实成功更新已在用户日志中通过。
**未自行运行服务器v4的batch16双开/548图完整smoke；不能保证其一定通过。**
服务器若仍失败，新日志/JSON将明确失败张量，不能据此继续猜根因或跳过门禁。

## 已安装v3时的更新命令

确认上一轮两项进程都已退出，上传v4到 `/root/autodl-tmp` 后：

```bash
cd /root/autodl-tmp
unzip -o E42E43_E40_JointDense_Parallel_AutoDL_v4.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh check && bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh smoke
```

只有两项 `CUDA SMOKE PASSED` 且 `FINISHED smoke: {'e42': 0, 'e43': 0}` 后才运行：

```bash
nohup bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh run > launcher_e42_e43.log 2>&1 &
tail -f launcher_e42_e43.log
```

失败时不得启动run，日志会自动输出traceback与分开的失败摘要。
