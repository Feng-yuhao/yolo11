# Delivery validation / 2026-10-14

## Boundary

Server preflight was run by the user: Ultralytics8.4.177/Python3.10/
torch2.5.1+cu121, CUDA, both heads/main loss/common gradients exact at start;
eight FP32 and eight AMP finite updates, all 52 groups actually updated.
That receipt is engineering-only, not a 640/b16 real-data smoke.

New workflow local integration: Windows/Python3.9/torch2.6+cu124,
RTX4060 Laptop8GiB, **private synthetic128/b16**, 48 train/35 val images.
Local fixture tests never create a server gate. Synthetic AP=0 is expected
from two cold-start epochs on random images and is not a research result.

## Executed checks

Private `audit/yolo26_port/paired_local_v2/PRIVATE_LOCAL_VALIDATION.json`:

- Baseline and enhanced: native AMP checkpoint check with bundled26n passed;
  real dataloader/preprocessing/original progressive two-head criterion/optimizer.
- Eight finite actual optimizer updates each; E6052/52 gradient groups and
  52/52 actually updated groups. Losses remain finite; skipped scaled-gradient
  overflow attempts are not counted as updates.
- Native FP32 EMA autocast validation, b32 plus last3 images, all35 evaluated;
  separate square32 forward and finite loss/decoded/raw output guard.
- Saved smoke checkpoint + standard predictor on eight images; inverse rounded
  x/y resize coordinates independently verified, raw predictions not mutated.
- Native Trainer **two complete epochs** for both models, CSV/best/last,
  progressive-loss epoch update, checkpoint strip and standalone final-best
  validation completed. New process checkpoint loading did not need old11 code.
- Frozen official scorer unit tests, predictions and full35-image ignore scoring
  completed for both saved val-best checkpoints. Actual scoring outputs retained.
- Full private dataset audit checked independent original-annotation -> YOLO
  classes/coordinates, label ranges, pairing, train/val disjoint IDs and SHA hashes.
- Nine fast unit regressions passed: fixed200/16/640 protocol, pinned-source
  drift refusal, stale/local gate refusal, altered report refusal, automatic
  batch reduction refusal, nonfinite checkpoint refusal, nonfinite-loss refusal,
  rounded inverse coordinate consistency, and mocked serial controller smoke/run
  jobs/state/report completion. Controller mocks are explicitly private tests;
  they are NOT evidence that server jobs executed.

## Corrections retained before delivery

`paired_local_v1` is retained. The new independent inverse-coordinate oracle
initially reused old11's single isotropic scale. Actual8.4.177 accounts separately
for rounded resized width and height. The oracle was corrected to those actual
dimensions; the production native Predictor and official scoring formula were
not replaced, tolerance was not enlarged. V2 passed square and rectangle inputs.
The private harness also resolves output paths absolutely to avoid native
settings re-prefixing relative paths on Windows.

An initial unit invocation used the sandbox default temporary path/settings
directory and was denied write access. Test settings/temp paths were moved under
the authorized workspace; all nine assertions then passed. No server environment,
training code, model definition, batch or numerical loss was changed for this.

## Still requires user server execution

True VisDrone640/b16, all548 validation images atb32,32GiB peak memory,
servertorch2.5/native Trainer200 epochs and actual accuracy/benefit are unexecuted.
`run_pair.sh smoke` must pass before `run`; it does not launch formal training.
No remote writes/training/shutdown were performed by the assistant.
