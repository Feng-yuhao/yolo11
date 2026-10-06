# v3 核心代码与AMP审核（2026-10-01）

## 原始失败，不隐瞒核心实现问题

本次服务器E42在32个真实批次里完成24次有限更新，但
`0.channel_router.1.`没有任何成功步的非零检测损失梯度。
E43完成8次有限更新后停在同步验证barrier，是E42失败的连带中止，不是E43自己的梯度报错。
这不是OOM，也不再是v1的成功步计数问题。

原实现仅在联合softmax/加权时转FP32，之前的通道MLP仍处于autocast。
新增末层从0开始学习，微小FP32 master权重在卷积前可能舍入为FP16零，切断前层梯度。
在本地CUDA独立受控fixture中，旧AMP写法前层非零梯度数为0，FP32参考为6144。
该fixture证明旧数值实现确实存在这一故障；原服务器失败日志未保存权重，不能声称重建了当时每个张量。

## 修复范围

- 只让新E42通道MLP的池化、两层投影和反向处于FP32区间。
- FP32均值累积不物化整个FP32高分辨率descriptor；投影读取原注册Parameter，不复制学习参数。
- 支持纯half EMA权重，不原地改变Parameter/buffer dtype、值或特征。
- 不关闭全模型AMP；不改GradScaler初值、batch、学习率、损失、TAL、增强、零初始化、参数形状和随机初始化。
- E43计算公式及公共E40结构不改；E42的数学联合路由公式不改，但数值计算策略明确改变了。
- 不删除/放宽18组loss梯度及真实参数更新门禁。
- check新增完整YOLO损失、640/batch2的CUDA分阶段更新检查；这是工程合成数据，不是性能probe。
- smoke仍要求batch16真实训练、至少8次成功更新、全部18组参与，及548图/验证batch32。
- 并行失败会立即展示子日志末尾；一项正式任务失败不杀另一项。

## 本地执行证据

环境：RTX4060 Laptop 8GiB，torch2.6.0+cu124，Python3.9.25，Ultralytics8.3.0。
没有连接或修改AutoDL，没有启动200轮正式训练。

- 旧故障在本地CPU FP16/CUDA autocast重现；修复后前向/梯度与独立FP32参考一致。
  loss-only SGD（无weight decay）更新原第一层Parameter；half EMA兼容且参数/输入不变。
- E42完整本地CUDA审核：`tmp/e42e43_v3_e42_CUDA_20261001_035809/audit.json`。
- E43完整本地CUDA审核：`tmp/e42e43_v3_e43_CUDA_20261001_040013/audit.json`。
- 实际原始YOLO检测损失、CUDA AMP、全网络SGD：E42/E43分别17/18次有限更新，全部18组有非零loss梯度和真实参数变化。
  零初始化导致初期梯度延迟，不能把“最早第三步”写成“必定第三步”。
- E1初始square/rect预测、原loss和原参数梯度严格对齐；任务输入/深度归一化oracle通过。
- 保存/重载/fuse/新进程/标准Predictor、纯FP16值保真、空GT通过。
- TAL FP32/FP16等价以及实际P2 FP32/AMP loss、梯度、更新检查通过。
- 原D27评分与几何入口单元检查通过；正式训练config/build_model/trainer_class/train与原发布版本AST一致。
- 修复后重新执行标准Trainer/EMA/best与last/矩形预测/官方评分集成（各4张合成图、160/batch2/2epoch），两套均通过：
  `E42_LOCAL-INTEGRATION_20261001_041226_254714`、`E43_LOCAL-INTEGRATION_20261001_041236_050065`。
  该测试获准读取本地已有pandas依赖，不是正式数据训练或性能证据。

## 尚未完成与资源边界

AutoDL是torch2.5.1、32GiB vGPU；本地小batch CUDA不能替代该环境的完整真实并行smoke。
服务器batch16双开、同时batch32验证、548图及200轮密集样本峰值仍要由用户运行门禁。
不得承诺绝无OOM/bug或一定涨点；首次正式训练尚未开始，本故障不代表已经跑坏了正式结果。
本修复没有改动E40/E41的旧代码或权重，不能据此推断历史所有负实验都是同一代码问题。

## 更新命令（旧作业退出后）

```bash
cd /root/autodl-tmp
unzip -o E42E43_E40_JointDense_Parallel_AutoDL_v3.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh check && \
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh smoke && \
bash /root/autodl-tmp/E42E43_E40_JointDense_Parallel_AutoDL/run_pair.sh verify
```

必须三项命令均退出0，且双方CUDA CHECK/SMOKE PASSED，才可独立启动正式后台命令。
不要单独输入行末`&&`后按回车等待；一次粘贴整个命令块。
