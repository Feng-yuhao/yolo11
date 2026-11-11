# 本地交付核验

本包的工程验证不构成服务器全量测试结果或跨域增益结论。正式启动会在服务器重新运行检查和六份真实图 smoke，全部通过才做完整 test。

已核对的资产：六份 VisDrone 原始 val-best SHA 与既有 `Paper_NoTrain_Supplement_AutoDL/inputs.json` 一致；四个模型实现文件与已完成 VisDrone test-dev 评估包逐字相同。没有把 Songdo 四类训练权重冒充源域十类权重。

本地验证覆盖：固定类别分数映射、car/van NMS 去重、不修改 raw 输出、非目标类别过滤时机、独立 NumPy greedy-NMS、小型 perfect/wrong-class/empty/subset COCO oracle、横竖图逆坐标、原始非连续 image_id、负增益种子保留、忙碌 GPU/未知 Python 任务拒绝、错误权重拒绝、BN 状态指纹回归。

真实工程 smoke：六份实际 VisDrone 权重分别在相同8张 Songdo test 图上执行 FP32/最大边640/batch4，其中包含竖图5282。只检验运行、坐标、映射和状态不变，不用 smoke AP 选模型。原始参数及 BN 缓冲在推理前后一致，原始权重文件哈希不变。

本地环境为 Windows / Torch2.6.0+cu124 / RTX4060 Laptop 8GiB，不等同于 AutoDL 的 Linux / Torch2.5.1+cu121 / RTX4080 SUPER。没有远程运行、正式训练、全1084图评分或真实关机。Linux进程枚举在单测中使用私有模拟/proc；不声称已做真实服务器准入测试。

本地审计日志位于项目 `PAPER_EXPERIMENTS/audit/visdrone_to_songdo_delivery_20261023/`；保留了修正本地临时目录权限前的失败记录。ZIP CRC、Python语法、Bash LF和代码/资产SHA在打包时核验；没有可用的本地 Bash，不声称运行了 bash -n。
