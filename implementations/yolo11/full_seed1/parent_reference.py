"""Frozen paired seed1 references: identities, protocols and cached predictions."""
import json

E40_REPORT = 'E40_RUN_20260929_205658_507620'
E58_REPORT = 'E58_RUN_20261008_001519_559728'
E1_SHA = 'f1a5ba6642a01445a99b2c7c80d8e79299a4402159bc37d2a08bb31ff365df12'
E40_SHA = '0a9fdac26c4fe7a9161900c82e879f584026afa569f870c0159d41feca031337'
E58_SHA = 'd4afe6bcbedb415f5551a09e3f48e6f0d436d6a49d0d8080a5b9f6394fb7319e'
E1_EXPECTED = dict(AP50_95_percent=25.63123996261998,
                   AP50_percent=44.72545960536977, AP75_percent=25.02982351561637)
E40_EXPECTED = dict(AP50_95_percent=26.222452343110788,
                    AP50_percent=45.361316073, AP75_percent=25.8444541332)
E58_EXPECTED = dict(AP50_95_percent=26.58242057951359,
                    AP50_percent=45.79600679537345, AP75_percent=26.312034357842997)


def reference(args, dataset, name, sha, expected, run_name):
    import d27_official as official
    root = args.project/'comparison_reports'/name
    report = json.loads((root/'comparison.json').read_text())
    training = json.loads((root/'training.json').read_text())
    if (report['status'] != 'completed' or report['seed'] != 1 or report['imgsz'] != 640 or
            training['status'] != 'completed' or training['completed_epochs'] != 200 or
            training['params']['seed'] != 1 or report['checkpoint_sha256'] != sha):
        raise RuntimeError('Wrong completed seed1 reference: '+name)
    if report['dataset'] != dataset:
        raise RuntimeError('Original validation dataset fingerprint differs: '+name)
    best = args.project/'runs'/run_name/'weights/best.pt'
    if not best.is_file() or official.sha256(best) != sha:
        raise RuntimeError('Reference best.pt missing or changed: '+str(best))
    if not (root/'area_predictions.json').is_file():
        raise RuntimeError('Reference predictions missing; restore archived report: '+str(root))
    for key, value in expected.items():
        if abs(report['candidate'][key]-value) > 1e-6:
            raise RuntimeError('Reference metric mismatch '+name+' '+key)
    from run_experiment import BASE_REPORT
    base = json.loads((args.project/'comparison_reports'/BASE_REPORT/'metrics.json').read_text())
    e1 = args.project/'runs/e1_yolo11s_p2add_img640_seed1/weights/best.pt'
    if base['evaluation_weights']['sha256'] != E1_SHA or not e1.is_file() or official.sha256(e1) != E1_SHA:
        raise RuntimeError('Paired E1 seed1 checkpoint missing or changed')
    return report['candidate'], report


def reference_e40(args, dataset):
    return reference(args, dataset, E40_REPORT, E40_SHA, E40_EXPECTED,
                     'e40a_spatial_task_head_img640_seed1_200e')[0]


def reference_e58(args, dataset):
    return reference(args, dataset, E58_REPORT, E58_SHA, E58_EXPECTED,
                     'e58a_e57_allscale_w96_img640_seed1_200e')


def rescore_reference(args, records, name, expected):
    """Same scorer/annotations applied again; no reference-model inference."""
    import d27_official as official
    from d38_common import KEYS
    root = args.project/'comparison_reports'/name
    detections = official.load_predictions(root/'area_predictions.json', len(records))
    filtered, _ = official.filter_detections_and_audit(records, detections)
    result = official.official_evaluate(records, filtered)
    for key in KEYS:
        if abs(result[key]-expected[key]) > 1e-7:
            raise RuntimeError('Cached official reference did not reproduce '+name+' '+key)
    return result
