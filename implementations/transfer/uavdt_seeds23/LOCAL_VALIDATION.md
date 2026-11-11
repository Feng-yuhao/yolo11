# 本地工程验证

本地16项回归检查通过；包含新增紧凑JSON浮点/结构回读一致性、三seed负差值保留、6GiB磁盘准入和关机模拟。额外8项独立关机测试通过，使用实际结果哈希验证函数构造最小凭据，覆盖四任务完整性、文件篡改、父进程未退、新会话、倒计时取消、平台脚本变化、命令非零及成功路径。所有power命令runner均被mock，不执行真实关机。

四份真实VisDrone val-best（E1/E60 seed2/3）分别在相同固定8张UAVDT真实测试图（8段序列）完成FP32/640最大边/b4 smoke。原始权重SHA、头结构、未改参数/BN、raw映射NumPy oracle、逆坐标oracle均通过；COCO及native AP70两类计分在8图子集跑通。子集结果只做工程核验，不用于选seed/权重/阈值/映射。

本地数据复用已全文件SHA验证的UAVDT测试数据，重新核验全部文件大小/时间及原标注元数据SHA，确认stat签名未变；正式服务器all仍完整逐文件核验。推理、映射、模型、GT、忽略区域和计分核心文件由build_release与已完成seed1逐字节比对。仅common.write在两个大型预测JSON上去掉缩进空白，回读数值不变。

日志：PAPER_EXPERIMENTS/audit/visdrone_to_uavdt_seeds23_delivery_20261023/engineering_v2。Windows/Torch2.6+cu124/RTX4060 8GiB不是服务器Linux/Torch2.5.1+cu121/RTX4080 SUPER。尚未远程启动全量seed2/3、训练或真实关机。没有声称执行本地bash -n、真实Linux进程准入或Matlab官方程序。

最终仅更新控制器完成提示及本说明、打包资产核验，不改变已验证的推理/数据/计分路径。ZIP CRC、Python语法、Bash LF和逐文件manifest在build_release里核验。seed1参考绑定到已核验的原完成目录和ARTIFACT_SHA256；协议及GT不一致时拒绝汇总。
