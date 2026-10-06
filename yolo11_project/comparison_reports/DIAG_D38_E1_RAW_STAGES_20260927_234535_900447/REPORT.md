# D38 E1-640 raw-stage audit

Normal saved predictions reproduced. No training.

|size|GT|raw any @.75|raw correct @.75|conf @.75|max_nms @.75|NMS @.75|top500 @.75|
|---|---:|---:|---:|---:|---:|---:|---:|
|all|38599|62.613|55.823|54.921|54.921|48.043|46.980|
|lt8|11819|26.060|21.516|21.372|21.372|18.174|17.641|
|8_16|15114|68.473|60.507|59.772|59.772|49.332|47.949|
|16_24|6186|88.522|80.472|78.597|78.597|70.417|68.817|
|24_32|2548|94.309|87.402|85.518|85.518|80.887|79.984|
|ge32|2932|97.544|90.518|88.608|88.608|86.050|85.505|

Detailed density/class/stride breakdown: metrics.json; per-GT evidence: gt_stage_rows.json.

Interpretation boundaries:

- Coverage is independent GT candidate existence, not one-to-one recall or AP.
- Raw coverage uses decoded boxes, not a causal test of backbone, TAL or regression loss.
- Raw any-class versus argmax-correct gap is not isolated classification error; nearby instances can explain it.
- NMS/score/budget losses locate postprocessing removal, not proof that disabling that operation improves AP.
- Continuous XYXY diagnostic IoU is separate from frozen official MATLAB-style AP.
- Only nonignored class 1..10 official GT are included; raw candidates are not ignore-filtered to preserve the actual NMS path.
- Size is sqrt(area)*640/max(original H,W), not official AP-small; density is conditioned on size but remains confounded.
- No new training or module is authorized by this report. No test-dev is used.