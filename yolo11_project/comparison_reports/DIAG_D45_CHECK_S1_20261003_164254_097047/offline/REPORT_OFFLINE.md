# D45 E47/E40 已有报告严格配对诊断

不训练、不读取 checkpoint、不使用 GPU。

匹配 seed=1，imgsz=640，GT=38599。

## 官方评分

|指标|E40|E47|E47−E40 (pp)|
|---|---:|---:|---:|
|AP50_95_percent|26.2225|26.4020|+0.1796|
|AP50_percent|45.3613|45.6455|+0.2842|
|AP75_percent|25.8445|26.0148|+0.1704|

## 官方逐类变化

|类别|E40 AP|E47 AP|ΔAP pp|ΔAP50 pp|ΔAP75 pp|
|---|---:|---:|---:|---:|---:|
|pedestrian|25.2398|25.3842|+0.1443|+0.0651|+0.4433|
|people|16.9483|16.7423|-0.2060|-0.2680|-0.3842|
|bicycle|8.1383|8.5877|+0.4494|+0.8640|+0.5577|
|car|59.2031|59.5001|+0.2970|+0.1679|+0.5864|
|van|28.9102|28.7678|-0.1424|-0.1798|-0.3435|
|truck|24.1918|25.0046|+0.8128|+1.2715|-0.7884|
|tricycle|16.0855|16.4902|+0.4047|+1.0030|+1.1049|
|awning-tricycle|8.8677|7.4697|-1.3980|-1.7042|-1.8233|
|bus|40.8371|42.8148|+1.9777|+3.6694|+0.0349|
|motor|24.7836|25.0643|+0.2807|+0.1107|+0.9534|

## 同一 GT 的尺寸分组得失（IoU≥0.75，非 AP）

|尺寸组|GT|raw-any 净变化 pp|final-correct 新覆盖|final-correct 损失|final-correct 净变化 pp|
|---|---:|---:|---:|---:|---:|
|all|38599|+0.0259|2547|2385|+0.4197|
|lt8|11819|+0.5669|655|572|+0.7023|
|8_16|15114|-0.4631|1248|1216|+0.2117|
|16_24|6186|-0.0162|421|381|+0.6466|
|24_32|2548|+0.2747|125|110|+0.5887|
|ge32|2932|+0.2387|98|106|-0.2729|

## 剩余缺点的互斥描述（不是根因）

优先判断最终同类别框是否存在，再依次归入 raw geometry / raw argmax label / postprocess candidate missing。
这不是严格嵌套损失漏斗；non_nested_threshold_counts 会单列反例。

|状态|E40 GT 数|E47 GT 数|E47−E40|
|---|---:|---:|---:|
|final_correct_present|18267|18429|+162|
|raw_geometry_missing|14312|14302|-10|
|raw_argmax_label_missing|2676|2657|-19|
|postprocess_candidate_missing|3344|3211|-133|

## 严格配对及解释边界

旧 GT 行不含坐标，不能声称已逐坐标核对；采用注释/图像尺寸 SHA、完整 GT 键集合、类别和输入尺度尺寸核验。

密度缺失则不编造密度诊断。详细逐类、尺寸×类别、改善/损失 GT 数及状态转移见 paired_slices.json。

- Official AP uses the formal VisDrone ignore handling; legacy final coverage is before official prediction ignore filtering.
- Size is sqrt(GT area)*640/max(original width,height), not official AP-small/medium/large.
- Coverage is independent candidate existence per GT, not one-to-one matching recall or AP.
- raw_correct uses raw argmax class, while final_correct uses emitted class; these stages are not guaranteed nested.
- Mutually exclusive failure groups use final-correct first, then raw geometry, then raw argmax label, then postprocessing. This ordering is descriptive, not causal.
- This is a matched-checkpoint descriptive comparison; it cannot establish training root cause, seed-variance cause, or which added path is causally responsible.
- Legacy GT rows omit coordinates. When absent, annotation/dimension SHA, exact GT keys, classes and sizes are checked, not nonexistent per-row coordinates.
- Differences across size/class slices do not prove a module mechanism; rare slices have limited support.
