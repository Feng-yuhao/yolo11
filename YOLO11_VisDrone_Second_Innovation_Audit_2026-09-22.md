# YOLO11 VisDrone 第二轮创新空间审计（固定版）

最后更新：2026-09-22  
用途：在《YOLO11 VisDrone 实验决策账本》基础上，对尚未直接正式测试的“问题类型”进行第二轮创新性审计。  
原则：本审计允许最终候选为 0；未经本审计明确放行，不新增 E 系列正式训练。

---

## 1. 审计前提

本文件继承第一份《YOLO11 VisDrone 实验决策账本》的全部硬约束：

- E1 是唯一 **ACTIVE BASELINE**；
- E21a / E25a 永久关闭；
- E27a 仅保留为数据协议研究；
- D28/E28 关闭；
- ROI / slicing / zoom / 局部专家只保留为性能上界证据；
- neck / attention / gate / 上采样 / P1 晚期注入 / 质量调节 / TAL 小修 / 回归小修等已关闭方向不得换名重开；
- 新评测排名变化不能单独构成重开条件；
- 当前没有被确认的最终创新模型。

本轮审计不再从“能设计什么模块”出发，而从“现有证据还剩哪些未解释问题”出发，并同时检查：

1. 历史是否已覆盖；
2. 现有诊断是否直接支持；
3. 2022–2026 文献是否已有高度同构方案；
4. 成功后是否仍有足够论文贡献；
5. 是否值得投入正式 200 epoch。

---

## 2. 第二轮问题空间覆盖矩阵

| 问题格子 | 现有实验/诊断证据 | 历史覆盖情况 | 2022–2026 最接近工作 | 审计结论 |
|---|---|---|---|---|
| **跨检测尺度/FPN层级训练梯度冲突** | D21 将 E21a 的 large 崩塌判为 shared-trajectory-like；多次出现 small/large 此消彼长。但 E25a 的 Gradient-Isolated RSG 失败，说明至少“切断 RSG 直接梯度”不能解决问题。 | 没有对 E1 原生 P2/P3/P4/P5 做通用 PCGrad 式正式训练，但 E21/E25 已触及训练轨迹与梯度隔离。 | **SDSDet (2024)** 明确提出 FPN 邻层 “gradient confusion”，并用 OGF-NEM 改变反向梯度路径进行梯度解耦；**Decoupled Feature Pyramid Learning (2023)** 又通过 level supervision 处理层级监督失衡。 | **REJECT BEFORE TRAINING**。问题本身有文献直接命名和专门机制，不能作为新的主创新。 |
| **尺度层级监督不均衡 / 上层监督不足** | E21 的尺度牺牲提供间接迹象，但没有独立因果证明。 | 未直接做“给不同 FPN 层单独加强监督”的 E 系列。 | **Decoupled Feature Pyramid Learning (2023)** 明确指出 FPN 上层监督不足会损害大目标，并提出 level supervision。 | **REJECT**。高度直接重合，且证据仅间接。 |
| **分类-定位任务冲突 / score 与 localization 不一致** | D22：IoU=.50 未匹配 small GT 中 classification/ranking 占主要比例；E20b 质量调节无效；E23/DFL 强化无效。 | 已做分类/质量/回归方向多轮诊断和正式实验。 | **TOOD** 直接做 task alignment；**VarifocalNet** 学 IoU-aware classification score；**GFL/DFL** 将质量估计与分布式框回归结合。 | **CLOSED / SATURATED**。既有历史失败，也有成熟文献。 |
| **tiny-object label assignment / IoU 尺度敏感** | D23.1 没有达到预注册的强 TAL deficit；E23a 简单加强 P2 box/DFL 也未解决。 | Assignment 已被 D23/D23.1 专门诊断。 | **NWD-RKA (2022)** 直接针对 aerial tiny objects 的 IoU 尺度敏感和正样本分配不平衡，并在 VisDrone 等数据上验证。 | **REJECT**。诊断不支持，文献又高度直接。 |
| **高 IoU 几何误差的特殊框表示/不确定性** | D23 显示 IoU=.75 的主要失败来自 raw geometry；D24 说明 P1/P2 signed boundary correction 接近随机；E16/E23 都未显著改善 raw geometry。 | 已试 P1 回归细节、TSDR、加强 CIoU/DFL、边界诊断。 | GFL/DFL、NWD 系列以及近年的 tiny-object localization loss 已大量覆盖“非 Dirac 分布、几何敏感度、Wasserstein/IoU 改造”等思路。 | **NO APPROVED METHOD**。问题真实，但当前没有低重合、低风险、由现有证据直接支持的实现。 |
| **NMS / dense competition / post-processing 冲突** | D23 @.75 中 NMS 是第二大失败来源之一，但 raw geometry 明显更大；D21 max_det 证明候选预算问题真实但解释力有限。 | 有 D21 max_det、D23 candidate-origin，未做 NMS-free 新模型。 | 近年 end-to-end / NMS-free 检测以及 UAV 小目标 NMS-free、local NMS 工作已明显增多。 | **REJECT AS MAINLINE**。不是当前主导瓶颈，且会把问题改造成另一类 detector/postprocess 论文。 |
| **早期下采样/频率/相位细节保真** | D24 对现有 P1/P2 边界方向证据弱；E15 已用 PixelUnshuffle 保留 P1 相位；D28 已按决策树关闭。 | E15/E16 已非常接近；D28 明确禁止。 | **SPD-Conv (2022)** 直接以 space-to-depth + non-strided conv 替代 stride/pooling，目标就是小目标与低分辨率；SDSDet 也研究 convolution spatial artifacts。 | **CLOSED**。不能重开。 |
| **ROI / slicing / zoom / 高分辨率局部重读** | D15/D18 明确证明收益很强，是现有最强性能上界证据。 | 已完成预算、兼容性、DCRV 等大量实验。 | **QueryDet (CVPR 2022)**：低分辨率 query 指引高分辨率稀疏计算；ClusDet/Cascaded Zoom 等同类路线成熟。 | **ARCHIVED DIAGNOSTIC**。有效但不作为当前主创新。 |
| **遮挡/截断属性监督** | VisDrone 本身有遮挡/截断属性，但现有历史没有将其作为主训练变量。 | 未直接正式做。 | **OGMN (2023)** 已在 VisDrone/UAVDT 上直接做 occlusion localization + occlusion-guided multitask detection。 | **REJECT**。数据证据虽天然存在，但文献重合过高。 |
| **类别层级 / 长尾 / 易混类别辅助监督** | D21.1 表明 E21 的 large 崩塌含强烈 class×scale 重分配；D22 也显示 confusion 重要。 | 未做专门 hierarchy head，但分类/score 路线已大量试验。 | **VistrongerDet** 已针对 VisDrone 的相似类别混淆、长尾和 group-softmax / multi-label；相关 UAV 长尾工作已成熟。 | **REJECT**。与现有 VisDrone 专项工作高度重合。 |
| **ignore-aware 训练** | D27 发现多个模型约 17% small FP 落入 ignore 区，说明训练/评测语义存在系统性不一致。 | E27a 已写代码但按决策树归档、停止正式训练。 | 属于数据协议/监督对齐，而不是结构创新；即使有效，也需要把公平基线一并修正。 | **ARCHIVED / NOT MAINLINE**。不用于扩大模型创新贡献。 |
| **上下文/场景先验、密集关系、排斥** | UAV 场景符合，但现有诊断并未证明这是主导剩余瓶颈。 | 语义融合、背景抑制、RCFA 等已覆盖相邻方向。 | Repulsion Loss、各类 context-aware / relation 模块以及大量 UAV/remote-sensing 工作已覆盖。 | **REJECT**。无直接新证据，文献拥挤。 |
| **跨分辨率蒸馏 / zoom teacher** | 切片教师有强性能，但 E14 probe 已失败。 | 已做 E14。 | 超分辨率/特征蒸馏/跨尺度 teacher 大量存在。 | **CLOSED**。 |
| **数据增强 / copy-paste / tiny-object curriculum** | 当前诊断没有指向 augmentation 为主要剩余瓶颈。 | 不是主线；若做会改变训练分布而非结构。 | aerial tiny-object 增强、copy-paste、scale-aware training 文献拥挤。 | **NOT APPROVED**。 |
| **视频/时序信息** | VisDrone 数据来源具视频属性，但当前工程与论文目标是单图 DET，现有诊断没有时序证据。 | 未测试。 | UAV video detection / tracking-by-detection 已形成独立研究问题。 | **OUT OF SCOPE**。这不是“修 E1”，而是改变任务定义。 |

---

## 3. 对“跨尺度梯度冲突”候选的最终审计

### 3.1 为什么它一开始看起来合理

当前实验确实出现过如下现象：

- E21a small/medium 小幅提升但 large 明显下降；
- D21 的 checkpoint transplant 不能把问题定位为单一 gate/SDG 参数，报告给出 `Shared-trajectory-like`；
- E25a 切断 RSG 的直接辅助梯度后，large 仍未恢复；
- 多个历史结构存在“某尺度涨、另一尺度掉”的现象。

因此，“不同检测尺度在共享参数上的优化方向可能存在冲突”是一个**合理研究假设**。

### 3.2 为什么它仍不能获得训练许可

关键反证来自文献边界，而不是假设本身错误：

**SDSDet (Image and Vision Computing, 2024)** 已经把 FPN 邻接尺度之间的问题直接表述为 **gradient confusion**，并设计 **OGF-NEM** 通过改变反向传播路径，阻止浅层检测头的不合理梯度向深层回流。它与“检测尺度间梯度冲突/梯度解耦”在问题定义和机制层面都高度接近。

此外，**Decoupled Feature Pyramid Learning for Multi-Scale Object Detection in Low-Altitude Remote Sensing Images (2023)** 明确指出高层 FPN 监督不足会造成大目标性能退化，并提出 **level supervision**；这进一步压缩了“按层重新分配训练监督”的创新空间。

更重要的是，你自己的 D21/E25 并没有证明“E1 原生 FPN 存在通用跨尺度梯度冲突”：

- D21 是围绕 E21a 失败机制的 off-manifold checkpoint 诊断；
- `Shared-trajectory-like` 只能说明损伤不像单一 gate 参数能解释；
- E25a 只否定了“RSG 直接辅助梯度污染”这一具体修复；
- 它们并不等价于对 E1 的 P2/P3/P4/P5 梯度夹角、共享参数冲突率做了因果证明。

因此这条路线同时存在：

1. **证据仍是间接的**；
2. **2024 已有高度同构的 remote-sensing detector 方法**；
3. 即使成功，论文很容易被归类成 SDSDet/gradient surgery 的 YOLO11 变体。

### 3.3 固定决策

```text
跨尺度梯度冲突：
允许作为“解释性诊断问题”存在；
不批准作为新的 E 系列主创新；
不做 PCGrad/OGF/gradient projection 正式200轮；
不以“YOLO11上的首次应用”作为创新理由。
```

---

## 4. 本轮审计最重要的结论

### 4.1 获准训练的新结构候选数量

```text
0
```

截至 2026-09-22，本轮没有找到一个同时满足以下六项的候选：

```text
历史未覆盖
+ 现有诊断直接支持
+ 2022–2026 文献相似度低
+ 从 E1 可干净验证
+ 实现风险可控
+ 成功后足以构成论文主贡献
```

这不是审计失败，而是本轮审计的正式结果。

### 4.2 为什么不能为了“继续实验”而强行选一个

当前剩余问题大体分成三类：

1. **证据强但文献重合高**  
   例如 ROI/high-res、跨尺度梯度解耦、tiny-object assignment。

2. **文献可能还能做变体，但你自己的证据不支持**  
   例如再设计新的 boundary/localization loss、frequency 模块、NMS-aware 模块。

3. **可以做，但已经改变研究问题**  
   例如时序检测、DETR/NMS-free 全模型迁移、数据集重标注或大规模数据工程。

因此继续在 E1 上随机设计“第 28 个模块”不符合当前准入规则。

---

## 5. 接下来允许做什么

### A. 立即禁止

```text
不要开新的200 epoch训练
不要写E28/E29结构代码
不要重开E21/E25
不要训练E27a主线
不要做PixelUnshuffle/ROI/gradient-surgery变体
不要因为某篇论文模块名字不同就重新包装已关闭方向
```

### B. 当前真正需要做的决策

研究方向必须在下面两种策略中二选一：

#### 方案 1：收束现有课题

承认当前没有足够强的新结构模型，转而把已有成果整理成：

- E0 → E1 的强小目标结构基线；
- E10a 等小幅正结果；
- E21/E25 等失败与尺度代价；
- D21–D27 系统错误分解；
- 官方 VisDrone 评测校准；
- 对“为什么常见 neck/attention/gate/assignment/regression 小改在该任务上收益有限”的系统实验结论。

这种方向的价值主要来自**系统诊断和经验性结论**，而不是声称一个大幅领先的新模块。

#### 方案 2：改变论文问题边界

如果论文必须以“新方法 + 明显性能提升”为主贡献，那么需要明确改变至少一个前提，例如：

- 换研究任务；
- 换模型范式；
- 换数据集/问题定义；
- 接受此前排除的高分辨率/多视图范式并把创新目标重新定义。

这属于新课题立项，不应继续沿用“E1 上再塞一个小模块”的实验编号逻辑。

---

## 6. 文献核查清单

1. SDSDet: A real-time object detector for small, dense, multi-scale remote sensing objects. Image and Vision Computing, 2024.  
   https://www.sciencedirect.com/science/article/pii/S0262885624000015

2. Decoupled Feature Pyramid Learning for Multi-Scale Object Detection in Low-Altitude Remote Sensing Images. IEEE JSTARS, 2023.  
   https://openreview.net/forum?id=zQMUdQk6o2

3. Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark. ISPRS JPRS, 2022.  
   https://www.sciencedirect.com/science/article/pii/S0924271622001599

4. QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection. CVPR 2022.  
   https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html

5. OGMN: Occlusion-guided multi-task network for object detection in UAV images. ISPRS JPRS, 2023.  
   https://www.sciencedirect.com/science/article/pii/S0924271623000989

6. No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects (SPD-Conv). ECML-PKDD 2022.  
   https://arxiv.org/abs/2208.03641

7. TOOD: Task-Aligned One-Stage Object Detection. ICCV 2021.  
   https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html

8. VarifocalNet: An IoU-Aware Dense Object Detector. CVPR 2021.  
   https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html

9. Generalized Focal Loss. NeurIPS 2020.  
   https://papers.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html

---

## 7. 固定结论

```text
SECOND AUDIT RESULT:
NO APPROVED NEW E-SERIES CANDIDATE
```

除非后续出现以下情况之一，否则不新增正式训练：

1. 找到一个此前账本未覆盖的问题；
2. 有现有实验直接证据指向该问题；
3. 检索后没有高度同构的近年工作；
4. 能清晰解释成功后相对 E1 的论文贡献；
5. 用户确认愿意接受该路线的风险。

否则，当前阶段应停止“继续找模块”，转入论文定位决策。

---

## 8. D30 后续补充：E18a 的论文候选资格与 E31 审查

### 8.1 D30 改变了什么

D30 使用与 D27 完全一致的 Official VisDrone evaluator 对既有正式预测做了统一重评，reproduction gate 通过，六个复现模型的 AP/AP50/AP75/AR500 与 D27 差异均为 0。它没有训练或修改 checkpoint。

主要结果：

| 模型 | Official AP | 相对 E1 | AP50 | AP75 |
|---|---:|---:|---:|---:|
| E18a | **32.8994** | **+0.5596 pp** | 55.3034 | 33.0363 |
| E21a | 32.8830 | +0.5432 pp | 55.4840 | 33.0637 |
| E15a | 32.7004 | +0.3606 pp | 55.0667 | 32.7306 |
| E10a | 32.5935 | +0.2537 pp | 54.8901 | 32.6714 |
| E17a | 32.3538 | +0.0140 pp | 54.6696 | 32.3951 |
| E1 | 32.3398 | — | 54.6828 | 32.2472 |

E18a 相对 E1 的 Official AP/AP50/AP75 分别为 +0.5596/+0.6205/+0.7891 pp；官方逐类别 AP 有 9/10 类上升。旧 COCO-style 尺度诊断中，small/medium/large 也均高于 E1。因此，E18a 是当前**表现最健康的既有单 seed 模型**。

D30 不能证明 E18a 的 +0.5596 pp 已超过随机种子波动，也不能推翻“新评测排名本身不重开关闭派生路线”的账本规则。它改变的是论文候选优先级，而不是新结构实验许可。

固定状态更新：

```text
E18a = PAPER CANDIDATE PENDING REPLICATION
E18a ≠ ACTIVE BASELINE FOR NEW DERIVATIVES
```

### 8.2 “DSSG 与 FusionReady 协同”的表述边界

按 Official AP 计算：

```text
E10a - E1 = +0.2537 pp
E17a - E1 = +0.0140 pp
E18a - E1 = +0.5596 pp
difference-in-differences = +0.2919 pp
```

这提供了单 seed 的正交组合信号，但不能直接证明 FusionReadyAdapter 本体产生了协同。E17a/E18a 的共同新增变量实际是一个完整系统：

```text
两处 FusionReadyAdapter
+ P3/P4 共用的 training-only auxiliary head
+ auxiliary loss
```

因此论文中最多表述为：

> 在 seed1 中，DSSG 与 FusionReady 系统的组合收益高于两者单独增益的简单相加，呈现互补信号；该结论需由匹配种子复现确认。

不能将 +0.2919 pp 单独归因于空间兼容门、adapter 或 auxiliary loss。

### 8.3 E31a（标量改通道 scale）不获许可

当前 FusionReadyAdapter 为：

```text
update = Conv1x1_c(DepthwiseConv([top, lateral]))
output_c = lateral_c + scalar_scale * spatial_gate * update_c
```

虽然 `layer_scale` 是标量、`spatial_gate` 是单通道，但最后的 `Conv1x1` 已为每个 lateral channel 产生独立的 `update_c`。静态通道缩放大部分可以被该层各输出通道的权重吸收：

```text
channel_scale_c * update_c
≈ 重新缩放 Conv1x1 第 c 个输出通道权重
```

所以把一个标量改成 `[1,C,1,1]` 的自由参数，主要增加的是重复参数化，并没有建立“原模块无法选择通道”的证据。它也不足以构成论文贡献。

固定决定：

```text
E31a CHANNEL-WISE STATIC SCALE = REJECT BEFORE TRAINING
```

### 8.4 E31b（输入相关 channel-spatial gate）暂不获许可

输入相关的 `channel_gate [B,C,1,1]` 确实增加了静态输出卷积不能完全吸收的新能力。但当前没有诊断证明 E18 的剩余误差来自“融合前通道兼容性不足”。而且 channel+spatial adaptive FPN/fusion 已是拥挤方向：ASFF、CCA-FPN、FI-FPN、AFPN 以及大量 remote-sensing attention fusion 方法都覆盖了相近问题。

E19a 已经说明，把 FusionReady 换成更复杂的可靠性/相关性对齐结构会下降到 E1 以下。它不能直接否定轻量 channel gate，但足以说明“提高 adapter 复杂度”没有现成正证据。

短 probe 也不能解决这一问题：E1/E10a/E17a/E18a 的最佳 checkpoint 分别出现在 78/78/73/86 epoch，20–30 epoch 排名不能可靠预测正式 200 epoch 结果。

固定决定：

```text
E31b CHANNEL-SPATIAL FUSION = NOT APPROVED
不做 30 epoch probe
不做 200 epoch 正式训练
```

### 8.5 唯一获准的下一训练步骤

当前只批准**复现已有候选**，不批准新结构：

```text
seed2 E1（相同初始化规则、超参数、200 epochs）
seed2 E18a（相同初始化规则、超参数、200 epochs）
统一使用 D30/D27 Official evaluator
```

必须成对训练；只跑 E18a seed2 再与 E1 seed1 比较，无法区分种子波动与模型增益。

预注册判定：

```text
若 seed2 的 E18a-E1 Official AP >= +0.30 pp，
且 AP50、AP75 均不低于配对 E1：
    E18a 保留为最终论文候选，随后补 seed3。

若 <= +0.15 pp，或 AP50/AP75 任一出现明显反向退化：
    E18a 不封版，不开发 E31 衍生版。

若位于 +0.15~+0.30 pp：
    不解释成成功或失败；补 seed3 后用三种子均值和标准差决策。
```

截至本补充结论，第二轮审计仍然没有批准新的 E 系列结构实验；它只确定了一个值得做匹配种子复现的既有论文候选。

---

## 9. E31 实际结果与 E32 最后一次训练许可

### 9.1 E31 状态

用户提供的服务器端 90 轮对齐结果表明：

```text
E18a 前90轮最佳 mAP50-95：34.086% @86
E31a 前90轮最佳 mAP50-95：33.829% @90
差值：-0.257 pp

E18a epoch81-90 均值：33.944%
E31a epoch81-90 均值：33.697%
差值：-0.247 pp
```

E31a 独立验证为 mAP50-95 34.049%、AP-small 22.956%、AP-medium 42.118%、AP-large 55.045%。它没有提高综合上限，主要表现为尺度间重新分配。E31b 前15轮也没有出现早期优势，已停止。

固定状态：

```text
E31a = CLOSED
E31b = CLOSED
禁止 E31a 90→120/130 补跑
禁止 E31c 或门控变体
```

### 9.2 E32 获得一次性许可的新直接证据

对 E18a 的 `e17_fusion_loss.py` 进行源代码审查后，发现当前辅助监督存在可直接复现的监督冲突：

1. P3/P4 使用同一个 `min_feature_side=1.5` 判定；一个较大 GT 可以同时成为两个层的正样本；
2. 不满足某层尺寸条件的真实 GT 不会进入 `valid`；
3. 该层的 negative mask 只屏蔽 `valid` GT 的中心邻域；
4. 因此，不满足该层尺寸条件的真实目标位置仍可能被辅助分类损失当作背景负样本。

这与“简单调整 P3/P4 权重”不同。它是当前 E18 辅助损失中可由实现直接确认的正负监督冲突，满足账本的重新审查条件。

### 9.3 E32 固定定义

实验名：

```text
E32a Scale-Consistent Auxiliary Supervision
```

父模型与推理结构：E18a 完全不变。仅替换 training-only auxiliary loss 的 GT 路由和归一化。

#### 标注规模审计

以现有 VisDrone val 标注按 1024 letterbox 比例估算：

```text
GT总数：38,759
当前P3 valid：22,706（58.58%）
当前P4 valid：8,955（23.10%）
当前P3/P4重复正监督：8,955（23.10%）
当前两个辅助层均不产生正监督：16,053（41.42%）
```

若所有真实 GT 都屏蔽 auxiliary negative：

```text
P3 negative-mask空间覆盖：2.04% → 2.86%（+0.82 pp）
P4 negative-mask空间覆盖：3.26% → 8.05%（+4.78 pp）
```

这说明 P4 上的辅助假负样本不是极少数边界情况。

#### 分类与回归分开的唯一归属

沿用现有 `min_feature_side=1.5`，不新增可调尺寸阈值：

```text
valid_P3 = min(w * W3, h * H3) >= 1.5
valid_P4 = min(w * W4, h * H4) >= 1.5
```

辅助分类归属：

```text
若 valid_P4：classification_owner = P4
否则：classification_owner = P3
```

每个真实 GT 都在且仅在一个辅助层产生分类正监督。小于 P3 的 1.5 cells 的 tiny GT 仍可在 P3 中提供中心类别监督，但不要求该低分辨率层回归其边界。

辅助回归归属：

```text
若 valid_P4：regression_owner = P4
否则若 valid_P3：regression_owner = P3
否则：regression_owner = NONE
```

因此，分类覆盖全部 GT，回归只覆盖可由对应层解析的 GT。按现有验证集估算：

```text
classification：P3 76.90%，P4 23.10%
regression：P3 35.48%，P4 23.10%，NONE 41.42%

在有regression owner的GT中：P3 60.56%，P4 39.44%
```

任何 GT 不得同时在 P3/P4 产生同一种辅助正监督。

#### 全层真实目标忽略

无论 GT 属于 P3、P4 或 NONE，它在 P3/P4 两层的中心 `3x3` 邻域都必须从辅助 negative classification 中排除。

```text
classification owner：只在该层产生分类positive
regression owner：只在该层产生regression
non-owner层：不产生positive/regression，也不作为negative
```

这不涉及 D27 的数据集 ignore region；它只处理 auxiliary head 内部真实标注目标的尺度不归属问题。

#### 损失归一化

禁止手工设置新的 P3/P4 权重。两个层的 classification-owned positive 合并后按总数取均值，regression-owned 样本合并后按总数取均值；两层 negative loss 各自取均值后再等权平均。

保持以下超参数不变：

```text
auxiliary gain = 0.1
negative_weight = 0.1
box_weight = 0.25
min_feature_side = 1.5
shared auxiliary head = unchanged
```

因此：

```text
L_aux = mean(all_classification_owned_positive)
      + 0.1 * mean(level_negative_means)
      + 0.25 * mean(all_regression_owned_samples)
```

### 9.4 必须通过的实现检查

正式训练前必须证明：

1. E32 与 E18a 架构、参数量、GFLOPs、推理输出格式完全相同；
2. 新初始化与 E18a 的同 seed 初始参数逐张量相同；
3. 每个 GT 的 classification owner 数量必须为1，regression owner数量只能为0或1；
4. 所有 GT 在两个辅助层的 negative mask 中均被正确屏蔽；
5. `aux gain=0` 时总损失和梯度与 E18a 主检测损失一致；
6. P3/P4 classification owner、regression owner、regression NONE、原本双重 valid 的 GT 数量写入报告；
7. train/backward、eval validation、恢复 train/backward 均通过；
8. TAL2 等价和 batch8/imgsz1024 smoke 通过；
9. 自动关机保持禁用。

若 P3 或 P4 的 classification owner 数量低于全部 GT 的 10%，或任一 regression owner 低于全部有 regression owner GT 的 10%，训练前停止，因为这说明固定路由使某一辅助层基本失去监督。

### 9.5 90轮 probe 协议

trainer 从一开始必须按 `epochs=200` 构建余弦学习率计划；在完成 epoch90 后通过正常 callback 停止并保存 optimizer、scheduler、EMA。禁止把总 epochs 设置成 90，否则不能与 E18a 前90轮比较，也不能公平续训到200。

epoch90 后，用 probe 期间最佳 checkpoint 运行同一个 D27/D30 Official evaluator。

预注册门槛：

```text
Official AP 相对 E18a >= +0.25 pp，且 AP50/AP75 均不下降：
    续训到200。

Official AP <= +0.10 pp 或下降：
    立即关闭E32，E18a进入多seed验证。

Official AP 位于 +0.10~+0.25 pp：
    仅当 E32 epoch81-90 的 Ultralytics mAP50-95均值
    相对 E18a 同轮次均值 >= +0.10 pp，且最佳epoch >=75，才续训；
    否则关闭。
```

200轮最终替代门槛：

```text
Official AP >= E18a +0.25 pp
AP50、AP75均不低于E18a
旧尺度辅助诊断中任一 AP-small/medium/large 不得下降超过0.50 pp
推理参数量、GFLOPs与E18a相同
```

E32 只有这一版，不派生 b/c/d，不搜索 1.5 阈值、loss gain 或 P3/P4 权重。

### 9.6 论文边界

尺度级监督、FPN level supervision 和 auxiliary deep supervision 已有大量先例，E32 不能声称首次提出尺度监督。若有效，只能主张：

> 针对所提出 pre-fusion adaptation 的训练期辅助头，使用尺度一致的唯一归属与非归属目标忽略，避免跨层重复正监督和辅助假负样本；该策略不增加推理开销。

它是应用型论文中的训练一致性贡献，不是独立的大理论创新。
