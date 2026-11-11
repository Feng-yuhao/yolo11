"""Frozen FP32 predictor + same original-annotation VisDrone scorer, val only."""
from pathlib import Path
import common as c
from e60_evaluate import predict_records


def evaluate(args, best, out):
    import d27_official as official
    official.run_unit_tests()
    records, dataset = official.load_dataset(args.dataset)
    saved = c.read(args.project/'comparison_reports'/c.E60_REPORT/'comparison.json')
    c.require(dataset == saved['dataset'], 'Reference/current evaluation dataset differs')
    predictions, rows = predict_records(args, best, out, records)
    detections, audit = official.filter_detections_and_audit(records,
                            official.load_predictions(out/'area_predictions.json', len(records)))
    candidate = official.official_evaluate(records, detections)
    references = {}
    for model, report, expected in (('E1', c.E1_REPORT, saved['baseline']),
                                    ('E60', c.E60_REPORT, saved['candidate'])):
        directory = args.project/'comparison_reports'/report
        filtered, _ = official.filter_detections_and_audit(records,
                              official.load_predictions(directory/'area_predictions.json',len(records)))
        metrics = official.official_evaluate(records, filtered)
        for key in c.KEYS:
            c.require(abs(metrics[key]-expected[key])<1e-7, 'Original '+model+' official score did not reproduce: '+key)
        references[model] = metrics
    delta = {model:{key:candidate[key]-metrics[key] for key in c.KEYS}
             for model, metrics in references.items()}
    result = dict(status='completed',variant=args.variant,seed=1,split='validation',imgsz=640,
                  candidate=candidate,references=references,delta_pp=delta,dataset=dataset,ignore=audit,
                  checkpoint=str(best),checkpoint_sha256=official.sha256(best),
                  protocol=dict(precision='FP32',batch=4,conf=.001,iou=.7,max_det=500,original_annotations=True),
                  boundary='Fixed seed1 retrained ablation, not an optimization search or a multi-seed causal/significance proof. Uniform removes all adaptive routing, not spatial variation alone.')
    c.write(out/'comparison.json',result)
    c.write(out/'official_metrics.json',result)
    table = '|Metric|E1 seed1|E60 seed1|'+args.variant+' seed1|Delta vs E60 pp|\n|---|---:|---:|---:|---:|\n'
    for key in c.KEYS:
        table += '|'+key+'|'+ '|'.join(f'{v:.4f}' for v in
                      (references['E1'][key],references['E60'][key],candidate[key]))+f'|{delta["E60"][key]:+.4f}|\n'
    (out/'REPORT.md').write_text('# E60 routing ablation / '+args.variant+' / seed1\n\n'+table+
                               '\n'+result['boundary']+'\n',encoding='utf-8')
    c.write(out/'completion.json',dict(status='completed',official_evaluation_completed=True,finished_at=c.now()))
    print(c.now()+' '+args.variant+' OFFICIAL RESULT vs E60='+str(delta['E60'])+' REPORT='+str(out/'REPORT.md'),flush=True)
    return result
