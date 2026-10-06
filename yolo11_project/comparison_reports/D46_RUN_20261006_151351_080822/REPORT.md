# D46 E1/E40 instance-competition audit

|Model|Official AP|raw same-class independent75|raw maximum one-to-one75|post-NMS maximum one-to-one75|
|---|---:|---:|---:|---:|
|e1_seed1|25.6312|55.823|55.818|48.019|
|e40_seed1|26.2225|55.988|55.988|48.312|
|e1_seed2|25.4107|55.437|55.431|47.734|
|e40_seed2|26.0639|55.991|55.983|48.377|

All original predictions and official scores reproduced before interpretation. No training.

## Read these files

- `all_models.json`: paired-seed summaries and size/class/density cells.
- `<model>/gt_rows.json`: signed center/size errors, same-class nearest-neighbor direction, surviving alternatives.
- `<model>/nms_events.json`: good-candidate overlap witnesses, GT association and ties/unresolved cases.
- `<model>/image_stats.json`: independent versus exact maximum one-to-one coverage at all stages.

## Scientific boundaries

- Maximum matching is score-independent geometric upper coverage, NOT achievable AP or official recall.
- Matched GT identity can vary among equal-cardinality solutions; only image totals are compared.
- Density/neighbor direction remain confounded; candidate geometry-selected bias is not training root cause.
- NMS witness counts are candidate events, not recoverable unique GT; rows also record surviving alternatives.
- Continuous geometry/argmax classes and pre-ignore candidates are distinct from official AP.
- Matched IDs, neighbor direction and NMS witnesses do not identify a training-time root cause.
- Geometry candidates are pre-official-ignore; official metrics retain original ignore rules.
- No test-dev evaluation, GT-oracle score rewriting, threshold search, checkpoint tuning or new training.
