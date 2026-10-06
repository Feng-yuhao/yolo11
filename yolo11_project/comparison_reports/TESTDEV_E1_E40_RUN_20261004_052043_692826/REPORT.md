# E1 / E40 test-dev paired three-seed evaluation

Finished: 2026-10-04T05:34:56+08:00

1610 test-dev images; frozen D27 MATLAB-v1.0.4-style scorer; AP values in percent, deltas in pp.

| Seed | E1 AP | E40 AP | delta AP | delta AP50 | delta AP75 |
|---|---:|---:|---:|---:|---:|
| 1 | 23.6827 | 23.7195 | +0.0368 | +0.0452 | +0.1351 |
| 2 | 23.2236 | 23.8695 | +0.6459 | +0.8106 | +0.7272 |
| 3 | 23.4693 | 24.1348 | +0.6655 | +0.7909 | +0.9441 |

Mean +/- sample standard deviation across three seeds:

- e1: 23.4586 +/- 0.2297
- e40: 23.9079 +/- 0.2103
- paired_delta_pp: 0.4494 +/- 0.3575

Per-class AP/AR and provenance: each model's official_metrics.json and prediction_manifest.json.
This is a held-out evaluation, not permission to tune repeatedly on test-dev.
No training was started, checkpoint changed, or automatic shutdown requested.
