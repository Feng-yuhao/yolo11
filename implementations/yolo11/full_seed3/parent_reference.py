"""Frozen paired seed3 references; no nonexistent E58 seed3 comparator."""
import json

E40_REPORT = 'E40S3_RUN_20261002_223525_596764'
E1_SHA = '11e656c7cdeb9112d303bae0d68c32a874031a7734ad88fda1f37fa9c88778eb'
E40_SHA = '6a2a59025f61391d1c59ab8c28110a58e67953e9758f7946b13298f7f63ff267'
E1_EXPECTED = dict(AP50_95_percent=25.47334717171393,
                   AP50_percent=44.337464035543746, AP75_percent=24.84501913155771,
                   AR500_percent=37.64210458739394)
E40_EXPECTED = dict(AP50_95_percent=25.99458970292751,
                    AP50_percent=45.02249968263732, AP75_percent=25.587757918795475,
                    AR500_percent=38.065581593067975)


def reference_e40(args, dataset):
    import d27_official as official
    name,sha,expected = E40_REPORT,E40_SHA,E40_EXPECTED
    run_name = 'e40a_spatial_task_head_img640_seed3_200e'
    root = args.project/'comparison_reports'/name
    report = json.loads((root/'comparison.json').read_text())
    training = json.loads((root/'training.json').read_text())
    if (report['status'] != 'completed' or report['seed'] != 3 or report['imgsz'] != 640 or
            training['status'] != 'completed' or training['completed_epochs'] != 200 or
            training['params']['seed'] != 3 or report['checkpoint_sha256'] != sha):
        raise RuntimeError('Wrong completed seed3 reference: '+name)
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
    for key,value in E1_EXPECTED.items():
        if abs(report['baseline'][key]-value)>1e-7:
            raise RuntimeError('E40 report paired E1 seed3 metric mismatch: '+key)
    from run_experiment import BASE_REPORT
    base = json.loads((args.project/'comparison_reports'/BASE_REPORT/'metrics.json').read_text())
    e1 = args.project/'runs/e1_yolo11s_p2add_img640_seed3/weights/best.pt'
    if base['evaluation_weights']['sha256'] != E1_SHA or not e1.is_file() or official.sha256(e1) != E1_SHA:
        raise RuntimeError('Paired E1 seed3 checkpoint missing or changed')
    return report['candidate']


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
