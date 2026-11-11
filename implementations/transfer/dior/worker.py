"""One source checkpoint and one target per fresh process; no training."""
from pathlib import Path
import time
import traceback
from common import HERE,KEYS,SPECS,boot,now,package_audit,parser,read,require,sha,split_key,write
from target_data import audit,records,root_for
def run(args):
    package_audit();target,source_key=split_key(args.key)
    args.project,args.source,args.data_parent,args.report=(p.resolve() for p in (args.project,args.source,args.data_parent,args.report))
    binding=read(args.report/'binding.json');root=root_for(args.data_parent,target)
    require(binding['package_sha256']==sha(HERE/'manifest.json'),'Package changed')
    require(audit(root,target)['file_stat_sha256']==binding['datasets'][target]['file_stat_sha256'],'Target payload changed')
    spec=binding['weights'][source_key];weights=args.project/spec['relative_path'];require(sha(weights)==spec['sha256'],'Wrong weights')
    if args.phase=='evaluation':
        receipt=read(args.report/'smoke'/args.key/'completion.json')
        require(receipt['status']=='passed' and receipt['checkpoint_sha256']==spec['sha256'],'Missing bound smoke')
    env=boot(args,args.report/'settings'/args.key);require(env['source']==binding['source'],'Source changed')
    write(args.report/args.phase/(args.key+'_environment.json'),env)
    items=records(root)
    if args.phase=='smoke':items=items[:8]
    out=args.report/args.phase/args.key;start=time.monotonic()
    from inference import predict
    from storage import read_predictions
    from scoring import evaluate
    proof=predict(args,weights,args.key,items,out)
    predictions=read_predictions(out/'predictions.json.gz')
    require(len(predictions)==proof['prediction_count'],'Lossless gzip readback failed')
    gt=args.report/'target'/target/'gt.json';require(sha(gt)==binding['generated_GT'][target]['sha256'],'Evaluation GT changed')
    write(out/'scoring_progress.json',dict(stage='COCO-style AP',at=now()))
    scores=evaluate(gt,predictions,target,out,[r['id'] for r in items] if args.phase=='smoke' else None)
    if args.phase=='evaluation':
        require(all(scores[k]==SPECS[target][k] for k in ('images','positive_boxes','ignored_entries')),'Incomplete evaluation population')
    scores.update(status='completed',key=args.key,checkpoint_sha256=spec['sha256'],prediction_sha256=proof['prediction_sha256'],
                  seconds=time.monotonic()-start,finished_at=now(),gzip_readback_verified=True,engineering_only=args.phase=='smoke')
    write(out/'metrics.json',scores)
    if args.phase=='smoke':write(out/'completion.json',dict(status='passed',checkpoint_sha256=spec['sha256'],images=len(items),scoring_executed=True,smoke_only=True))
    require(audit(root,target)['file_stat_sha256']==binding['datasets'][target]['file_stat_sha256'] and sha(weights)==spec['sha256'],'Target/source changed during inference')
    print(f'{now()} {args.key} {args.phase} FINISHED',flush=True)
if __name__=='__main__':
    p=parser();p.add_argument('--report',type=Path,required=True);p.add_argument('--key',choices=KEYS,required=True);p.add_argument('--phase',choices=('smoke','evaluation'),required=True)
    try:run(p.parse_args())
    except BaseException:print(traceback.format_exc(),flush=True);raise SystemExit(1)
