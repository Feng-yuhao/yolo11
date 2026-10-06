# v2：修复真实CUDA日志确认的smoke计数缺陷

2026-10-01用户日志：两项首8个真实batch均发生6次overflow_skip，仅2次finite_update。
v1把attempt固定8，同时要求至少3次成功更新，随后错误地报告“没有有效更新”。
此次失败不是日志中的OOM或非有限forward loss；真实CUDA batch16已被执行，但不代表完整smoke通过。

v2只修改工程smoke及其门禁：最多32个真实batch，等待至少8次成功更新、18组非零有限loss梯度和实际参数更新。
AMP溢出不计作成功；若在预算内仍达不到条件则明确失败，不发回执。
batch16、640、seed1、原GradScaler策略、正式200轮优化器/LR/增强、模型/公式/损失/TAL均不变。
保留batch32/548图验证、显存14GiB上限及双开失败隔离。不因本次错误修改核心模块。
存在已下载字体时复用，减少独立settings重复下载Arial.ttf造成的启动等待；不伪造字体。

本地回归验证包括用户日志序列、持续overflow拒绝、真实CPU GradScaler跳步/更新、两项正式函数AST与v1完全一致，
以及原检查和FP16模块对照。服务器CUDA v2尚未执行，不声明通过。

## 安装v2（先确认旧pair已结束）

上传E42E43_E40_JointDense_Parallel_AutoDL_v2.zip到/root/autodl-tmp；该ZIP内部仍是原实验包目录。
不向正在运行的包解压覆盖。用户提供的v1日志表明两个smoke均已退出，正式训练尚未启动。

```bash
cd /root/autodl-tmp
unzip -o E42E43_E40_JointDense_Parallel_AutoDL_v2.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh check
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh smoke
```

修改工程代码后旧回执指纹失效，因此需要重新check；不能删除门禁验证或手工改成passed。
双方CUDA CHECK PASSED、CUDA SMOKE PASSED且smoke退出码均0之后：

```bash
nohup bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh run > launcher_e42_e43_v2.log 2>&1 &
tail -f launcher_e42_e43_v2.log
```

原失败报告保留，不覆盖它们，也没有需要重新跑的正式训练。
