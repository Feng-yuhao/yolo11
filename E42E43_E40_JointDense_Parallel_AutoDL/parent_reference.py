"""Verify the original E40 result and weight identity, not a new numeric guess."""
from pathlib import Path
import json


def reference_e40(args,dataset):
    import d27_official as official
    root=args.project/'comparison_reports/E40_RUN_20260929_205658_507620'
    report=json.loads((root/'comparison.json').read_text())
    training=json.loads((root/'training.json').read_text())
    if report['status']!='completed' or report['seed']!=1 or report['imgsz']!=640 or training['completed_epochs']!=200:
        raise RuntimeError('E40 parent is not the completed seed1/640/200e experiment')
    if report['dataset']!=dataset:
        raise RuntimeError('E40 parent dataset fingerprint differs')
    # Stored server path cannot be used verbatim for local Windows checks.
    best=args.project/'runs/e40a_spatial_task_head_img640_seed1_200e/weights/best.pt'
    if not best.is_file() or official.sha256(best)!=report['checkpoint_sha256']:
        raise RuntimeError('E40 parent best.pt missing or changed: '+str(best))
    expected={'AP50_95_percent':26.222452343110788,'AP50_percent':45.361316073,
              'AP75_percent':25.8444541332}
    for k,v in expected.items():
        if abs(report['candidate'][k]-v)>1e-6:
            raise RuntimeError('E40 reference metric mismatch: '+k)
    return report['candidate']

