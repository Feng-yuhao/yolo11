# E39 post-training official evaluation recovery

This package does not train, alter saved weights, overwrite source code, change
the E39 model, or install a third-party package named `evaluate`.

The original E39 evaluator, official evaluator, raw-geometry helpers and custom
checkpoint classes are bundled. Only the evaluator's two report-IO imports have
changed from `run_e39` to `recovery_common`. Official scoring, prediction options,
ignore rules and the E1 reproduction gate remain unchanged.

The original E39_RUN `failure.json` remains intact. A successful full 548-image
evaluation writes a separate E39_RECOVERY report and a hash-linked completion
receipt. The standalone queue accepts this specific repaired post-evaluation
import error only; all other failures, changed checkpoints/reports and incomplete
200-epoch training remain blocked. Original E40 files/check/smoke receipts are
not edited. E40 starts only after original gates verify and two GPU-idle checks.

## AutoDL

Upload E39_PostEval_Recovery_AutoDL.zip to /root/autodl-tmp.

```bash
cd /root/autodl-tmp
unzip -o E39_PostEval_Recovery_AutoDL.zip -d /root/autodl-tmp
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E39_PostEval_Recovery_AutoDL/run_recovery.sh inspect
nohup bash /root/autodl-tmp/E39_PostEval_Recovery_AutoDL/run_recovery.sh all > launcher_e39_recovery_e40.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e39_recovery_e40.log
```

`all` means **evaluate the saved E39 best**, then resume the previously authorized
E40 queue. It does not retrain E39. If you only want evaluation, use `eval` instead
of `all`. Any unsuccessful recovery or stale E40 gate prevents continuation.
Ctrl+C in `tail -f` only leaves the log viewer; it does not kill the nohup job.

Recovered artifacts:

- comparison_reports/E39_RECOVERY_*/official_metrics.json (official AP/AP50/AP75/AR500)
- comparison.json (E1 reproduction, candidate scores and differences)
- area_predictions.json, gt_rows.json, REPORT.md, recovery.json
- original E39_RUN/evaluation_recovery.json (links original training and failure)

Local CPU tests do not replace server CUDA tests or prove metric gains.
