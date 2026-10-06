# D47 本地工程验证

本地 Windows / Python 3.9 / Torch 2.6；CUDA 设备 RTX 4060 Laptop 8 GiB。

已实际执行的检查：

- CPU 与本地 CUDA 单元：独立解析 IoU=1、2/3、错类别、背景、重复 GT 归属；坐标平移/缩放；视频分组；采样预算与空候选。
- 按空间位置逐个核对四检测层的 64/128 通道 tower 特征和原候选索引；DFL uniform entropy。
- 质量网络零输出与原分数 bitwise 一致；原框/argmax 类别不变；两次真实优化更新、有限梯度。
- 新解释器 CPU 重载两种排序器 state_dict，核对输出。
- 分数变换与独立 sigmoid(logit(s)+delta) 对照；可进入 conf=.001 的最低支持分数。
- 原 NMS 的输入不被修改、改排序确实能选择另一框；官方 class-0 raster ignore 与 score-zero IoF；regular-positive 的优先规则。
- 真实 E40 seed1 权重：CPU、本地 CUDA，四张不同宽高合成图，含空 GT；冻结权重/BN digest、原预测逐图复现、三个分支零输出官方评分一致、候选缓存、拟合与重载评分通过。
- 真实 E40 seed2 权重：本地 CUDA 上完成同样集成，并额外验证固定10轮终点、train group holdout、full/DFL 权重加载、完整缓存 SHA 复用及 smoke 终点拒绝。

合成图中的 GT 仅用于工程覆盖，并包含由预测构造的 fixture，不是自然图像实验；这些 AP 数字绝对不能作为研究结果。
尚未执行 AutoDL/Torch2.5.1 的真实 train/val 全量、真实服务器 smoke 或有效性训练，不能保证涨点。
未修改原 Ultralytics 源码、checkpoint 或任何远程系统。用户仍须执行服务器 check、8-train/8-val smoke。

V2 读取修复新增 fixture 已通过：复现用户零高度 class-0 行；零宽/双零 ignore 行；原值保留与行号计数；独立几何 strip 的 ignore coverage；正常标注与原 loader 的所有数组及 metadata 一致；非法前景框/负尺寸/非整数类别/NaN/缺列拒绝；BOM 与末尾逗号兼容；冻结 val parser 仍拒绝零尺寸标注。修复后重新执行 CPU 和本地 CUDA 全套 self-test 通过。没有在本地执行真实 6471 张训练集扫描。
