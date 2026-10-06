# E40 seed3 worker

这是已验证 E40 结构的第三个随机种子，不新增结构。对照已有
`e1_yolo11s_p2add_img640_seed3_20260926_172932_190728`，不重跑 E1。
640、batch16、workers8、200 轮及原始预训练迁移不变；E40新块仍用40001隔离初始化。

从包根 `run_pair.sh check`、`smoke`、`run` 启动；不要独立启动这个worker的smoke/run。
并行smoke与formal使用相同13GiB allocator上限，未通过并行门禁不启动正式训练。
不自动降低batch，不默认关机。结果及官方 AP/AP50/AP75 对比由训练入口自动生成。

本地 `test_pair_helpers.py` 为模拟门禁测试，CPU check 验证模型逻辑；
均不能代替服务器真实 batch16/640 训练与 batch32 全548图验证 smoke。
