# E60 v2：修复新残差分支的 AMP / EMA 精度风险

服务器 v1 smoke（2026-10-09 16:00）不是 OOM：reserved 13.24GiB，24 次有限更新。
失败项仅为 P2 三个新残差编码器的九个内层 kernel 参数组；没有开始正式200轮训练。
不能跳过门禁，也不能用这次工程失败宣称研究方向失败。

## 复现与解释边界

在本机 CUDA 上，使用完全相同的 FP32 参数，令输出投影权重为 1e-9：
原 v1 autocast 路径将 9,216 个非零投影值全部舍入为 FP16 的零，三个内层梯度全部为零；
FP32 对照有非零内层梯度。该机制与日志缺失的内层组一致。
服务器旧 smoke 没有保存每组极小权重/梯度的详细统计，故不声称已直接测得那次所有权重的实际范围。
此次修复消除已复现的风险，服务器复验仍为必需。

上一版的非零权重 FP16 检查使用正常幅值，未覆盖极小投影。
合成 EMA 集成发现的 FP16 下溢只被记录，未转成强制精度保护，这是工程审查遗漏。

## 修复范围

- 六个新增 GroupResidualEncoder 内部 Conv/GN/SiLU/project 在禁用 autocast 的 FP32 子区域计算。
- 仅这些新参数保持 FP32，包含原 Trainer 的 EMA.half() / strip 保存、设备移动及重载。
  不采用“先转 half 再 float”，避免已经损失的极小值无法恢复。
- 输出残差先转回原 bank dtype，再相加；原 E58 编码链/路由/native softmax/检测塔仍沿原精度路径。
- 零初始化、149,472 新参数、原始预训练迁移、隔离 RNG、seed1/640/b16/200轮、原 loss/TAL 均不变。
- 不扩大初始化、不调高学习率、不增加 smoke 次数或取消任何组的更新要求。
- smoke 每个尝试写 branch_gradient_audit.json；失败之前保留缺失组和投影精度信息。

## 补充测试

同权重极小投影 + FP16 输入下：修复前内层梯度为零，修复后与独立 FP32 对照相符且真实 SGD 更新。
检查 half/bfloat16/device transfer/save/reload 对新增 FP32 权重的逐值保真，以及原父模型参数仍可 half。
全检测 loss 的 FP32 更新、AMP 冷启动覆盖、原始链输入/空间路由、fuse、新进程和标准 Trainer/EMA 集成另行验证。
本地与服务器 PyTorch 版本不同；本地小样本通过不替代服务器 b16/640 和 b32/548 图真实 smoke。

FP32 子区域是 PyTorch 支持的 AMP 用法：[AMP 官方文档](https://docs.pytorch.org/docs/2.14/amp.html)。
实际速度/显存会变化，不能沿用 v1 估计；模型理论参数量和卷积计算量未变。

## 更新方式

v2 zip 解压到原父目录，内部仍是 E60a_E58_GroupResidualEncoding_AutoDL，不新建另一条实验路线。
保留失败报告；新的 manifest/signature 会拒绝旧 check/smoke 凭据。依次重新 check、smoke、verify-gates，
只有新 CUDA 门禁通过后才开始正式 run。自动关机仍需要显式 --auto-shutdown。
