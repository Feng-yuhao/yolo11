# E49a — E40 + shallow-guided residual reassembly

固定实验：seed1、640、batch16、workers8、SGD、200 epoch、cosine LR、close_mosaic=10。
父模型为完整 E40，不加入 E47/E48 的任何读出，不从 E40 best 微调。
原 E1 的初始化/迁移不变；E40 新参数隔离 40001，新重建分支隔离 49001。

## 精确干预

原 model23：model16 P3(C128,stride8) → nearest 2x。
新 model23 输入：[model16,model2]，后者是 backbone-P2(C128,stride4)。
两路各 128→32 投影、无可学习参数 GN4；P3 descriptor nearest 2x 后相加，
SiLU→DW3x3→SiLU→1x1 生成每个细网格位置的9个核权重（FP32 softmax）。
核跨128通道共享，3x3粗网格邻域，replicate 边界；四细位置使用相同核预测器，
但由各自真实浅层 descriptor 产生不同权重，不设置全局相位嵌入。

`Y = nearest(P3) + tanh(gain) * (reassemble(P3,kernel) - nearest(P3))`。
gain=0，其他新权重正常初始化。初始严格等于 E40；第一步 gain 有梯度，后续核网络有梯度。
初始公共参数裁剪前梯度数值一致，不保证新增参数纳入全局梯度裁剪后的更新或后续训练轨迹一致。
gain可为负，整个算子不能称为纯低通或凸组合。只有重组核是正权归一化。
无偏移、相关性搜索、高通分支、直接浅层值混合、额外attention或辅助loss。
原 Concat24 的 backbone-P2 值路径、C3k2 25、完整 E40/Detect26 保留。
P3–P5 直接预测路径结构保留，不宣称训练梯度完全隔离或收益必然叠加。

## 科学边界与相关工作

CARAFE：https://arxiv.org/abs/1905.02188
FADE：https://arxiv.org/abs/2207.10392
FADE 官方代码：https://github.com/poppinace/fade/blob/main/FADE_L2H.py

借鉴已有内容自适应重组/浅深联合核生成。四输出位置不同权重不是首创。
这是面向 E40 的窄范围结构适配，不能通过改名/隐藏宽度/零初始化宣称新原理。
最近邻不是已被证实的根因，Concat+C3k2 已有细位置建模能力。
不引入原图新像素，不保证可以恢复 P3 未编码的 tiny 目标。
与 E10（融合后增强）、E19（其他层已上采样特征的相关性移位搜索）有区别，仍有实质重叠。

## 对照与判断

主要对照：现有 E40 seed1；同时报告 E1 seed1，同 D27 官方 VisDrone 协议。
额外 AP≥0.25pp 且 AP50/AP75不下降是预算筛选目标，不是显著性/录用标准。
尺寸 raw/final@.75 是补充诊断，不替代 AP，不把输入px分组称为官方AP-small。
成功后复核 seed2；不根据 test-dev 的类别或分组继续设计/选择本模块。
失败只否定本固定配置，不证明所有引导重建均无效。

## 工程与资源

新增参数8778；含 E40 共184218个新参数。卷积增量约0.292G@640，
另列重组/残差算术估计，不把 THOP 未统计的 unfold/softmax/内存搬运冒充零开销。
低分辨率展开一次邻域，四相位复用；不物化完整高分辨率 C×9 邻域。
单开，smoke和正式均28GiB allocator上限，不改batch、不自动重试降配置。
check包含独立索引/边界/脉冲/梯度oracle、实际model16/model2路由、
原loss与公共梯度、激活E40恒等、保存/新进程重载/fuse、AMP与半精度值保真。
合成check先验证原始零起点gain的loss梯度和更新，再用可丢弃gain=0.1夹具验证完整核路径；
随机图初始很小的gain可能让FP32核更新低于ULP。夹具不用于正式初始化。
实际smoke从gain=0开始，必须观察真实数据下所有组loss梯度与更新，不用夹具绕过。
smoke要求8次真实成功更新（AMP overflow skip不计成功），最多32尝试，
同时验证所有19组新参数的非零loss梯度和实际更新、batch32完整548图验证。
本地小单元/合成2轮验证不等于服务器真实CUDA门禁或精度结果。
