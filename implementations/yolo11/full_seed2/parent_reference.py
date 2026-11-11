"""Frozen paired seed2 references: identities, protocols and cached predictions."""
import json

E40_REPORT = 'E40S2_RUN_20260930_130808_314014'
E58_REPORT = 'E58S2_RUN_20261008_162925_565240'
E1_SHA = '271901db9ff32c5b3a7735f218779eb27e1e1a6bc6604d3c77d37e338fd039d4'
E40_SHA = 'd5e21c610335dc44cd6b6c9a0766b7535c3cc5b68e75560d6c371509e78a8a09'
E58_SHA = '0024eac956589836c2215e35ad30e295286f1a958129322a0a05b74c6ae5f9a2'
E1_EXPECTED = dict(AP50_95_percent=25.41065733278743,
                   AP50_percent=44.312152547264084, AP75_percent=24.698933563022347)
E40_EXPECTED = dict(AP50_95_percent=26.063860003172476,
                    AP50_percent=45.21397062058152, AP75_percent=25.664076075372027)
E58_EXPECTED = dict(AP50_95_percent=26.534595145557994,
                    AP50_percent=45.94701265755563, AP75_percent=26.088941891839788)


def reference(args, dataset, name, sha, expected, run_name):
    import d27_official as official
    root = args.project/'comparison_reports'/name
    report = json.loads((root/'comparison.json').read_text())
    training = json.loads((root/'training.json').read_text())
    if (report['status'] != 'completed' or report['seed'] != 2 or report['imgsz'] != 640 or
            training['status'] != 'completed' or training['completed_epochs'] != 200 or
            training['params']['seed'] != 2 or report['checkpoint_sha256'] != sha):
        raise RuntimeError('Wrong completed seed2 reference: '+name)
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
    e1 = args.project/'runs/e1_yolo11s_p2add_img640_seed2/weights/best.pt'
    if base['evaluation_weights']['sha256'] != E1_SHA or not e1.is_file() or official.sha256(e1) != E1_SHA:
        raise RuntimeError('Paired E1 seed2 checkpoint missing or changed')
    return report['candidate'], report


def reference_e40(args, dataset):
    return reference(args, dataset, E40_REPORT, E40_SHA, E40_EXPECTED,
                     'e40a_spatial_task_head_img640_seed2_200e')[0]


def reference_e58(args, dataset):
    return reference(args, dataset, E58_REPORT, E58_SHA, E58_EXPECTED,
                     'e58a_e57_allscale_w96_img640_seed2_200e')


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
