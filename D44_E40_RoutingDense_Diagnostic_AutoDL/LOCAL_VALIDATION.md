# 本地验证记录（2026-10-01）

已实际执行，不是仅检查语法：

1. 最终代码的 CPU check：E40 seed1、E40 seed2、E42 seed1、E43 seed1 四份真实 best.pt 全部通过。
   报告：`yolo11_project/comparison_reports/DIAG_D44_CHECK_20261001_211354_801981/`。
2. 最终代码的本机 CUDA check：相同四份真实权重全部通过。
   本机 RTX4060 / torch2.6，显存分配上限4GiB；不等于 AutoDL vGPU32 / torch2.5。
   报告：`yolo11_project/comparison_reports/DIAG_D44_CHECK_20261001_211557_073402/`。
3. 四项额外单元测试通过：冻结评分/模块命名空间签名、JS 轴与解析值、矩形非对称 padding 的 GT 采样坐标、
   汇总报告不得自动批准训练。
4. check 在非零任务投影的合成结构上测试独立公式、未选尺度精确不变、干预关闭/异常退出恢复、
   输入与 state_dict 不变、无梯度；随后使用真实 checkpoint 在合成图像上通过实际
   YOLO → AutoBackend → 缓存头重算 → 原 NMS/坐标后处理 → GT 路由统计的集成路径。
5. 官方 D27 evaluator 字节归一化 SHA256 与正式评分的冻结版本一致，官方匹配/ignore/聚合单元测试通过。

明确未执行：服务器真实 8 图 smoke、548 图默认预测复现、548 图干预 AP、服务器6GiB峰值验证。
本地 check 不生成这些凭证，不填造正式诊断结论。服务器需按 README 分别执行 check、smoke、run。
没有启动训练、修改远程系统、修改任何原 best.pt / Ultralytics 源码。

若8图默认预测不能复现，则停止；不要降低复现要求强行继续。
验证期关闭路径的改善/下降只能作为推理敏感性证据，不是训练根因证明。
