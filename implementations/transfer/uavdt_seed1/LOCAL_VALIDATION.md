# 本地工程核验边界

已核对独立测试包全部33316个payload文件的SHA；原始20段测试序列、16592图、375884个DET对象及忽略区域原文均保留。
实际本地工程日志：PAPER_EXPERIMENTS/audit/visdrone_to_uavdt_seed1_delivery_20261023/engineering_v2。

小型回归覆盖：三类顺序、car/van最大分数合并、映射前后的原张量不变、NMS前排除非目标类别、跨类别重叠、top500前过滤、非有限输入拒绝、352x640原生padding逆坐标、忽略区域严格边界、原生AP70重复框/错误几何/阈值边界、80组独立标量匹配oracle、小型COCO perfect/wrong-class/empty、忙GPU拒绝、BN指纹变化、负增益保留及磁盘余量准入。

两份真实VisDrone seed1 best分别通过FP32/最大边640/b4的8张真实UAVDT图片smoke，固定来自8段不同序列。逐raw-anchor NumPy类别映射、NMS原始张量不变、逆坐标oracle、参数及BN前后指纹与原checkpoint SHA均核验；两种评分路径在同一8图子集跑通。子集AP不用于选映射/阈值/模型，不作全量收益结论。

本机Windows/Torch2.6.0+cu124/RTX4060 Laptop 8GiB，不等于服务器Linux/Torch2.5.1+cu121/RTX4080 SUPER。服务器仍须重新check和两份smoke后才完整评估；未运行服务器完整16592图评分、训练或关机。作者原生AP70是Python数学移植加独立小型oracle，不声称执行了Matlab。

第一轮完整数据SHA已通过，单元通过后因打包脚本更新触发了预期的代码哈希拒绝；失败日志保留。重打包后第二轮重新检查和两模型smoke均通过；只复用经文件stat和原标注SHA复核未变的全量数据凭据，不复用失败模型结果。之后仅补了控制器磁盘准入（单独回归）与本说明，没有修改已验证推理/模型/标注/计分代码。

ZIP重读CRC、代码语法与Bash LF由build_release.py复核。本地没有验证Linux进程枚举和bash -n执行，不冒称这些检查已运行。
