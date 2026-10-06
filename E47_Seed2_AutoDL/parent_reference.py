"""Verify the original E40 result and weight identity, not a new numeric guess."""
from pathlib import Path
import json
from references import SEED, PARENT_REPORT, PARENT_RUN, PARENT_EXPECTED, require_scores


def reference_e40(args,dataset):
    import d27_official as official
    root=args.project/'comparison_reports'/PARENT_REPORT
    report=json.loads((root/'comparison.json').read_text())
    training=json.loads((root/'training.json').read_text())
    if report['status']!='completed' or report['seed']!=SEED or report['imgsz']!=640 or training['completed_epochs']!=200 or training['status']!='completed':
        raise RuntimeError('E40 parent is not the completed seed2/640/200e experiment')
    if report['dataset']!=dataset:
        raise RuntimeError('E40 parent dataset fingerprint differs')
    # Stored server path cannot be used verbatim for local Windows checks.
    best=args.project/'runs'/PARENT_RUN/'weights/best.pt'
    if not best.is_file() or official.sha256(best)!=report['checkpoint_sha256']:
        raise RuntimeError('E40 parent best.pt missing or changed: '+str(best))
    require_scores(report['candidate'],PARENT_EXPECTED,'E40 seed2')
    return report['candidate']

