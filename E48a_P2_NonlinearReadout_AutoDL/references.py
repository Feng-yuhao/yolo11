"""Completed E47 seed1 reference, never loaded as training initialization."""
import json


def reference_e47(args,dataset):
    import d27_official as official
    root=args.project/'comparison_reports/E47P_RUN_20261002_223526_450642'
    report=json.loads((root/'comparison.json').read_text())
    training=json.loads((root/'training.json').read_text())
    if (report['status']!='completed' or report['seed']!=1 or report['imgsz']!=640
            or training['completed_epochs']!=200 or report['dataset']!=dataset):
        raise RuntimeError('E47 seed1 reference training/protocol/dataset mismatch')
    best=args.project/'runs/e47a_p2_depth_readout_img640_seed1_200e/weights/best.pt'
    if not best.is_file() or official.sha256(best)!=report['checkpoint_sha256']:
        raise RuntimeError('E47 seed1 best checkpoint missing/changed: '+str(best))
    expected=dict(AP50_95_percent=26.402007638651753,AP50_percent=45.64554054709653,
                  AP75_percent=26.014810713031974)
    for key,value in expected.items():
        if abs(report['candidate'][key]-value)>1e-7: raise RuntimeError('E47 reference score drift: '+key)
    return report['candidate']
