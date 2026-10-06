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

尺寸×拥挤、类别、置信度及三seed重复失败统计见 metrics.json。

没有从该离线报告批准任何新模块或正式训练。需要 raw-head/检测层来源/分类-NMS前后归因时，须另外读取模型原始输出。
