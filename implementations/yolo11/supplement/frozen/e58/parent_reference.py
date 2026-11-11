"""Only matched seed2 val references; E57 seed1 is not a seed2 comparator."""
import json

E40_REPORT = 'E40S2_RUN_20260930_130808_314014'
E1_SHA = '271901db9ff32c5b3a7735f218779eb27e1e1a6bc6604d3c77d37e338fd039d4'
E40_SHA = 'd5e21c610335dc44cd6b6c9a0766b7535c3cc5b68e75560d6c371509e78a8a09'
E1_EXPECTED = dict(AP50_95_percent=25.41065733278743,
                   AP50_percent=44.312152547264084, AP75_percent=24.698933563022347)
E40_EXPECTED = dict(AP50_95_percent=26.063860003172476,
                    AP50_percent=45.21397062058152, AP75_percent=25.664076075372027)


def reference_e40(args,dataset):
    import d27_official as official
    root=args.project/'comparison_reports'/E40_REPORT
    report=json.loads((root/'comparison.json').read_text())
    training=json.loads((root/'training.json').read_text())
    if (report['status']!='completed' or report['seed']!=2 or report['imgsz']!=640 or
            training['completed_epochs']!=200 or training['params']['seed']!=2 or
            report['checkpoint_sha256']!=E40_SHA):
        raise RuntimeError('E40 parent is not the completed seed2/640/200e experiment')
    if report['dataset']!=dataset:
        raise RuntimeError('E40 parent dataset fingerprint differs')
    # Stored server path cannot be used verbatim for local Windows checks.
    best=args.project/'runs/e40a_spatial_task_head_img640_seed2_200e/weights/best.pt'
    if not best.is_file() or official.sha256(best)!=E40_SHA:
        raise RuntimeError('E40 parent best.pt missing or changed: '+str(best))
    from run_experiment import BASE_REPORT
    base=json.loads((args.project/'comparison_reports'/BASE_REPORT/'metrics.json').read_text())
    e1=args.project/'runs/e1_yolo11s_p2add_img640_seed2/weights/best.pt'
    if base['evaluation_weights']['sha256']!=E1_SHA or not e1.is_file() or official.sha256(e1)!=E1_SHA:
        raise RuntimeError('Paired E1 seed2 weights missing/changed')
    for k,v in E40_EXPECTED.items():
        if abs(report['candidate'][k]-v)>1e-6:
            raise RuntimeError('E40 reference metric mismatch: '+k)
    for k,v in E1_EXPECTED.items():
        if abs(report['baseline'][k]-v)>1e-7:
            raise RuntimeError('E1 seed2 reference metric mismatch: '+k)
    return report['candidate']
