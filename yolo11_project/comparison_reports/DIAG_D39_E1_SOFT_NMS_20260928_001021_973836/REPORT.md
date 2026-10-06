# D39 E1-640 hard vs linear Soft-NMS

NO_CLEAR_OVERALL_AND_HIGH_IOU_BENEFIT_DO_NOT_AUTOMATICALLY_TRAIN_QUALITY_HEAD

|metric|hard|soft|delta pp|
|---|---:|---:|---:|
|AP50_95_percent|25.631240|25.755297|+0.124057|
|AP50_percent|44.725460|44.546644|-0.178815|
|AP75_percent|25.029824|25.379991|+0.350167|
|AR500_percent|37.814717|39.008943|+1.194227|

|size|GT|hard coverage .75|soft coverage .75|gained|lost|
|---|---:|---:|---:|---:|---:|
|all|38599|46.98049172258349|49.7474027824555|1196|128|
|lt8|11819|17.6410863863271|19.43480835942127|234|22|
|8_16|15114|47.948921529707555|51.97168188434564|677|69|
|16_24|6186|68.81668283220175|71.6618170061429|199|23|
|24_32|2548|79.98430141287284|81.71114599686028|53|9|
|ge32|2932|85.50477489768076|86.45975443383357|33|5|

## Boundaries

- Existing Soft-NMS is an inference intervention, not a new module or publication novelty.
- Candidate coverage is independent GT existence, not one-to-one recall/AP.
- Scores are deliberately decayed only in the soft arm; normal boxes/scores and weights are unchanged.
- Both arms use original-image official ignore rules for scoring and global max_det=500.
- Success cannot repair raw-box absence, prove a classifier root cause, or authorize another 200-epoch training.
- One fixed configuration, no val threshold search and no test-dev.