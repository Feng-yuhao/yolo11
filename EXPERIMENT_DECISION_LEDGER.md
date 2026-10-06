# YOLO11 VisDrone 实验决策账本

最后更新：2026-09-22  
适用范围：`yolo11_project` 的 E0–E27、D 系列诊断及后续实验决策。

## 1. 本账本的约束

这是后续实验规划的唯一状态来源。聊天中的临时建议不得覆盖本文件记录的正式结果。

状态定义：

- **ACTIVE BASELINE**：允许作为新独立方法的父模型。
- **SUPPORTING RESULT**：结果可写入论文或用于解释，但不得默认继续派生新版本。
- **CLOSED**：路线已关闭，不得再次推荐训练、补 seed 或“再修一个版本”。
- **ARCHIVED DIAGNOSTIC**：只保留为诊断或性能上界证据，不是论文主创新。

关闭路线只有满足以下任一条件才允许重新打开：

1. 发现可复现的实现 bug，且该 bug 会实质改变原实验结论；
2. 获得新的、直接针对原失败机制的证据，而不是新的猜想或不同评测排名；
3. 用户明确要求重新打开，并接受重复路线的时间与创新风险。

新评测协议、单个指标排名变化、换一个模块名称、或“也许组合后有效”均不构成重开条件。

## 2. 当前固定结论

| 项目 | 状态 | 固定结论 |
|---|---|---|
| E0 原始 YOLO11s | REFERENCE | 原始基准；D27 Official AP 29.8928%。 |
| E1 YOLO11s+P2 | **ACTIVE BASELINE** | 唯一可靠结构基线；D27 Official AP 32.3398%，相对 E0 +2.4470 pp。允许作为真正新路线的父模型。 |
| E10a DSSG | SUPPORTING RESULT | 有效但幅度小；D27 Official AP 32.5935%，相对 E1 +0.2537 pp。保留论文证据，但其大量后续派生均未形成稳定改进，不再默认作为父模型。 |
| E16a TSDR | CLOSED | 当前 Ultralytics mAP 较高，但 AP-small 低于 E10a；P1→回归并未验证预期的小目标机制。不得因单一总指标重新选为主线。 |
| E21a RSG | **CLOSED** | 预注册门槛失败，AP-small仅 +0.272 pp，AP-large -7.778 pp；D21/D21.1 已完成专门诊断，且 E25a 修复实验失败。D27 overall 排名不能重开该路线。 |
| E25a Gradient-Isolated RSG | **CLOSED** | 已执行 D21 预注册的修复方向；mAP 34.3116%、AP-small 23.0048%、AP-large 48.2382%，没有恢复尺度平衡。永久关闭 RSG 修复分支。 |
| E26a TERP | **CLOSED** | 独立反向任务路由正式200轮无效；mAP 33.8669%，接近 E1 且低于主要候选。 |
| E27/D27 Official audit | ARCHIVED DIAGNOSTIC | 建立官方 VisDrone overall 评测并发现约17%的small FP落在ignore区域；这是全模型共有的数据协议现象，不是E21特有机制。 |
| E27a ignore-aware训练 | ARCHIVED / DO NOT RUN AS MAINLINE | 代码仅保留为训练协议研究。它不能扩大结构创新贡献，当前不正式训练、不与E21组合。 |
| 切片、ROI、zoom、局部专家 | ARCHIVED DIAGNOSTIC | 证明重新读取高分辨率像素能提高性能；因相关工作拥挤且不符合当前结构创新目标，不再作为主线。 |
| D28/E28 PixelUnshuffle早期保真 | CLOSED BEFORE TRAINING | 与E15/SPD-Conv高度接近、证据不足、创新风险高；不做probe，不做正式训练。 |

当前没有被确认的最终创新模型。E1是基线，不等于论文最终方法。

## 3. E21 路线永久关闭的完整证据链

### E21a 正式结果

相对 E10a 的旧 COCO-style 尺度指标：

| 指标 | E10a | E21a | 变化 |
|---|---:|---:|---:|
| AP-small | 22.8917% | 23.1637% | +0.2720 pp |
| AP-medium | 42.4849% | 42.8387% | +0.3538 pp |
| AP-large | 56.6350% | 48.8575% | **-7.7775 pp** |
| Ultralytics mAP50-95 | 34.1129% | 34.3564% | +0.2435 pp |

small 未达到预注册的 +0.30 pp 门槛，large 远超允许的 -0.50 pp 下限，正式实验失败。

### D21

文件：`yolo11_project/comparison_reports/DIAG_D21_E21A_RSG_SCALE_BALANCE_20260919_102334_809540/decision.json`

- max_det 扩展不能解释 large 差距；
- gate-only checkpoint transplant 基本不能恢复 large；
- whole-SDG transplant 同样不能恢复；
- 反向 transplant 不能稳定复现损伤；
- 结论为 **Shared-trajectory-like**，且报告明确说明该诊断不是训练期因果证明。

### D21.1

文件：`yolo11_project/comparison_reports/DIAG_D21_1_E21A_ERROR_DECOMPOSITION_20260919_115055_591047/decision.json`

- large AP 差距在 IoU 0.50–0.75 平均为 9.0208 pp；
- 在 IoU 0.80–0.95 平均为 5.9126 pp；
- oracle same-class IoU 仅下降 0.00243；
- large FP 从 1532 降到 1280；
- 因而不能用简单定位恶化、FP爆炸或单一gate参数解释。

### E25a：真正执行过的修复实验

D21提出的 Gradient-Isolated RSG 后来已经作为 E25a 正式训练200轮，并非“尚未尝试”：

| 指标 | E10a | E25a | 变化 |
|---|---:|---:|---:|
| mAP50-95 | 34.1129% | 34.3116% | +0.1987 pp |
| AP-small | 22.8917% | 23.0048% | +0.1131 pp |
| AP-medium | 42.4849% | 42.4499% | -0.0350 pp |
| AP-large | 56.6350% | 48.2382% | **-8.3968 pp** |

梯度隔离没有救回large，small增益也进一步缩小。该修复假设已被正式否定。

### D27不能重开E21

D27显示E21a Official AP 32.8830%，在已评估模型中排名第一，但它只改变overall评测协议的解释，不满足上述任何重开条件：

- 没有发现E21或D21实现bug；
- 没有推翻预注册尺度门槛失败；
- 没有修复机制；
- ignore区域FP比例是多个模型共有现象。

因此禁止：E21 seed2、D27.1“救E21”、E22修复版、E21+E27、E21继续叠模块。

## 4. 其他路线状态

| 路线 | 代表实验 | 结论 | 状态 |
|---|---|---|---|
| 普通动态上采样 | E5a DySample | 相对E1没有可靠收益；不再因插入位置不同重复验证。 | CLOSED |
| P3语义/背景抑制 | E8b P3SDE、E9b PGALP、E9c | 正式或probe均未形成有效增益；位置、门控和loss修补不再扩展。 | CLOSED |
| 深层语义门控 | E10a、E10clean、E11、E12、E13 | E10a保留小幅正结果；删除、升级空间门控、局部注意力和分类专用分支均未继续提高。 | CLOSED FOR DERIVATIVES |
| zoom蒸馏 | E14a ERZD | probe明显下降；不再用切片/zoom作为教师救full-view。 | CLOSED |
| P1晚期细节注入 | E15a P1SDI、E16a TSDR | 总指标局部变化，但small机制不成立；不得改位置后当作全新路线反复训练。 | CLOSED |
| 融合前适配 | E17a、E18a、E19a | 单模块小幅变化，组合无叠加，显式对齐反而下降。 | CLOSED |
| SDQC/质量调节 | E20b | 与TSDR组合未提升；不再扩展质量门控。 | CLOSED |
| small分类/定位诊断 | D22、D23、D23.1 | 瓶颈随IoU变化；TAL强因果未成立；不支持再训练分类或assignment小修。 | CLOSED AS MODULE SOURCE |
| 回归监督增强 | E23a | 86轮机制审计双门槛失败；否定“简单加强现有P2 CIoU/DFL即可解决”。 | CLOSED |
| 边界证据细化 | D24 | P1/P2 signed correction方向接近随机；不写E24a。 | CLOSED |
| 任务排他反向路由 | E26a TERP | 正式200轮无效。 | CLOSED |

## 5. 早期已清理实验的历史边界

本地当前清理后的仓库未保留 E2–E7 的完整正式报告，只保留了历史命名与会话结论。为避免伪造精确数字，本账本只锁定可确认的边界：

- E2 SGP2-lite、E3 TASD、E4 PADB、E6 GSCG、E7 BDSI 均不得被当成“从未尝试的新方向”；
- 涉及切片预算、双向语义细节注入、背景/梯度抑制、旧可靠性融合的提案，必须先找到原归档或明确证明结构不同；
- 在缺少原报告时，不得编造指标，也不得以“记录不全”为理由重新训练。

## 6. 后续提案准入规则

任何新实验在写代码前必须同时提交：

1. **父模型**：默认只能是E1；若使用其他父模型，必须说明为什么没有重开关闭路线；
2. **历史差异**：逐项说明与E2–E26最接近实验的实质差异；
3. **文献边界**：列出最相近的至少3篇工作，并说明只能主张的增量创新；
4. **直接证据**：引用现有诊断中支持该机制的具体指标；没有直接证据只能做低成本审计，不能直接200轮；
5. **停止条件**：在看结果之前写死，失败后不得派生b/c/d修补版；
6. **论文价值**：即使成功，也必须说明相对E1的净创新收益如何构成论文，而不是只提高绝对分数。

当前状态：**没有获准训练的新E系列实验。先完成创新空间审计，再由用户确认唯一候选。**
