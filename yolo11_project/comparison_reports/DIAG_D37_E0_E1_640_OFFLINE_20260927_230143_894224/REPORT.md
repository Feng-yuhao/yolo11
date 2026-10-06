# D37：E0/E1 640 已保存预测诊断

不训练、不重新推理、不修改原文件。

## 解释边界

- Official metrics are cached frozen-D27 results, with exact prediction-hash checks.
- Candidate coverage uses converted YOLO GT; original VisDrone ignore regions are NOT available to this offline matcher.
- All GT may independently use the same prediction: coverage is NOT one-to-one recall or AP.
- Boxes are saved AFTER score filtering, NMS and max_det=500; absence cannot be attributed to raw geometry, TAL, a backbone stage or NMS alone.
- Wrong-class candidate availability is not proof of a classifier root cause: nearby/overlapping instances may also explain it.
- IoU=.50 coverage minus IoU=.75 coverage describes final-box precision, not an isolated regression failure.
- Input size is sqrt(original GT area)*640/max(original image width,height), not COCO AP-small.
- Density is other converted-label GT centres within a fixed 16-input-pixel radius; report it conditional on size.
- Density differences may still reflect class, within-bin size, occlusion and truncation; they are not an isolated effect of crowding.
- Three E1 seeds assess recurrent descriptive weaknesses, not candidate-model significance; E0 has only seed1.
- Validation diagnostics do not authorize selecting modules using test-dev or changing the official evaluation protocol.

## 官方分数（已核对预测哈希）

|模型|AP|AP50|AP75|AR500|
|---|---:|---:|---:|---:|
|E0_seed1|21.6422|37.5896|21.1884|30.8188|
|E1_seed1|25.6312|44.7255|25.0298|37.8147|
|E1_seed2|25.4107|44.3122|24.6989|37.6048|
|E1_seed3|25.4733|44.3375|24.8450|37.6421|

## 同类别最终候选覆盖率 @IoU=.75 / conf≥.001（非官方召回）

|输入尺寸√面积|GT|E0 s1|E1 s1|E1 s2|E1 s3|
|---|---:|---:|---:|---:|---:|
|lt8|12005|7.197|17.718|17.076|17.209|
|8_16|15103|36.145|47.997|47.712|47.918|
|16_24|6172|64.566|68.762|69.264|69.232|
|24_32|2547|78.563|80.016|79.505|78.916|
|ge32|2932|85.846|85.505|85.266|85.300|

## 同尺寸的拥挤分组（三个E1种子覆盖率均值，非因果效应）

拥挤定义：目标中心周围16输入像素内的其他GT数量。类别、遮挡和组内尺寸仍可能混杂。

|输入尺寸|邻居数|GT数量|同类覆盖@.50|同类覆盖@.75|
|---|---|---:|---:|---:|
|lt8|0|700|63.857|24.143|
|lt8|1_3|5247|57.747|19.478|
|lt8|ge4|6058|49.730|14.691|
|8_16|0|2944|84.556|61.368|
|8_16|1_3|9529|80.691|46.539|
|8_16|ge4|2630|76.996|37.617|
|16_24|0|2953|88.486|75.957|
|16_24|1_3|3042|87.070|63.346|
|16_24|ge4|177|77.213|53.107|
|24_32|0|1852|91.055|82.721|
|24_32|1_3|679|89.642|71.330|
|24_32|ge4|16|64.583|50.000|
|ge32|0|2712|91.544|86.271|
|ge32|1_3|216|90.123|74.537|
|ge32|ge4|4|58.333|50.000|

## 置信度降低能恢复多少高IoU候选（三seed均值）

该表不是AP；降低阈值也可能增加大量FP。

|输入尺寸|conf≥.05|conf≥.001|覆盖差值pp|
|---|---:|---:|---:|
|lt8|16.116|17.334|1.219|
|8_16|45.426|47.876|2.450|
|16_24|66.818|69.086|2.268|
|24_32|77.359|79.479|2.120|
|ge32|84.141|85.357|1.216|

## 三个种子共同的最终候选缺口

|输入尺寸|三seed均无同类@.75候选|三seed均无任意类@.75候选|三seed均有同类@.50但无同类@.75|
|---|---:|---:|---:|
|lt8|8945|8375|2572|
|8_16|5830|4821|2531|
|16_24|1312|921|539|
|24_32|362|217|135|
|ge32|278|122|76|

尺寸×拥挤、尺寸×类别、类别、置信度及三seed重复失败统计见 metrics.json。

没有从该离线报告批准任何新模块或正式训练。需要 raw-head/检测层来源/分类-NMS前后归因时，须另外读取模型原始输出。
