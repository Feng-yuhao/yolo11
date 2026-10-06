# D42 E38 same-checkpoint ON/OFF audit

Full 548-image validation. No training.

|Metric (%)|E1 reference|E38 OFF|E38 ON|ON minus OFF (pp)|
|---|---:|---:|---:|---:|
|AP50_95_percent|25.631240|25.338101|25.373724|+0.035623|
|AP50_percent|44.725460|44.206345|44.259593|+0.053248|
|AP75_percent|25.029824|24.828913|24.851496|+0.022583|
|AR500_percent|37.814717|37.608454|37.673845|+0.065391|

ON saved predictions reproduced.

- Same-checkpoint inference intervention, not a training-time causal explanation.
- OFF uses the E38-trained base network, NOT the independently trained E1 weights.
- ON/OFF AP includes their different NMS consequences; scores themselves stay unchanged.
- Both modes use the same 640/FP32/batch4/conf0.001/iou0.7/max_det500 protocol.
- The long NMS time allowance only prevents timing-based truncation under shared GPU load.
- PyTorch allocator cap does not cover all CUDA driver/library allocations.
- Concurrent inference can slow E39 and cannot guarantee zero shared-resource interference.
- No E39 files, shared sources, checkpoints, optimizer, shutdown state or processes are changed.
