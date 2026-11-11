"""One fresh CUDA process per fixed source checkpoint; no training branch."""
import sys
import time
import traceback
from common import HERE, KEYS, PROTOCOL, boot, now, package_audit, parser, read, require, sha, target_audit, write


def run(args):
    package_audit()
    args.project,args.source,args.dataset = (p.resolve() for p in (args.project,args.source,args.dataset))
    report = args.report.resolve()
    destination = report/args.phase/args.key
    binding = read(report/'binding.json')
    require(binding['package_sha256'] == sha(HERE/'manifest.json'), 'Receipt/code mismatch')
    require(target_audit(args.dataset,False)['test_file_stat_sha256'] == binding['target']['test_file_stat_sha256'],
            'Target payload changed since preflight')
    weights = args.project/binding['weights'][args.key]['relative_path']
    require(sha(weights) == binding['weights'][args.key]['sha256'], 'Source checkpoint changed')
    if args.phase == 'evaluation':
        smoke = read(report/'smoke'/args.key/'completion.json')
        require(smoke['status'] == 'passed' and smoke['checkpoint_sha256'] == binding['weights'][args.key]['sha256'],
                'Missing bound smoke; full evaluation not started')
    env = boot(args,report/'settings'/args.key)
    require(env['source'] == binding['source'], 'Inference source changed since preflight')
    write(report/args.phase/(args.key+'_environment.json'),env)
    from coco_eval import records,coco_score
    from inference import predict
    items = records(args.dataset)
    if args.phase == 'smoke':
        chosen = []
        for r in items:
            if r['sequence'] not in {c['sequence'] for c in chosen}: chosen.append(r)
            if len(chosen) == 8: break
        items = chosen
    started = time.monotonic()
    predictions,proof = predict(args,weights,args.key,items,destination)
    if args.phase == 'smoke':
        # No smoke AP used for tuning or checkpoint selection.
        write(destination/'completion.json',dict(status='passed',smoke_only=True,images=8,
              checkpoint_sha256=proof['checkpoint_sha256'],source_head_nc=10,finished_at=now()))
    else:
        print(f'{now()} {args.key} COCO SCORING START',flush=True)
        from uavdt_data import filter_ignored
        target = report/'target'
        require(sha(target/'gt_coco_style.json') == binding['generated_GT']['gt_sha256'] and
                sha(target/'ignore_by_image.json') == binding['generated_GT']['ignore_sha256'], 'Generated GT changed')
        kept,removed = filter_ignored(predictions,read(target/'ignore_by_image.json'))
        write(destination/'ignore_filtered_predictions.json',kept)
        write(destination/'ignore_filter_audit.json',dict(raw=len(predictions),kept=len(kept),removed=len(removed),
              rule=PROTOCOL['ignored_regions'],filtered_sha256=sha(destination/'ignore_filtered_predictions.json')))
        write(destination/'scoring_progress.json',dict(stage='COCO-style AP scoring',at=now()))
        scores = coco_score(target/'gt_coco_style.json',kept,destination)
        require(scores['images'] == 16592 and sum(scores['GT_counts'].values()) == 375884, 'Incomplete test evaluation')
        from native_ap70 import score as native_score
        def progress(done,total):
            write(destination/'scoring_progress.json',dict(stage='Native-math AP70 matching',done=done,total=total,at=now()))
        write(destination/'scoring_progress.json',dict(stage='Native-math AP70 matching',at=now()))
        native = native_score(read(target/'gt_coco_style.json'),kept,progress)
        write(destination/'native_ap70.json',native)
        scores.update(status='completed',key=args.key,seed=int(args.key[-1]),
                      checkpoint_sha256=proof['checkpoint_sha256'],prediction_sha256=proof['prediction_sha256'],
                      seconds=time.monotonic()-started,finished_at=now())
        write(destination/'coco_metrics.json',scores)
        write(destination/'scoring_progress.json',dict(stage='completed',at=now()))
        print(f'{now()} {args.key} AP={scores["AP_percent"]:.4f}% '
              f'AP50={scores["AP50_percent"]:.4f}% AP75={scores["AP75_percent"]:.4f}% FINISHED',flush=True)
    require(target_audit(args.dataset,False)['test_file_stat_sha256'] == binding['target']['test_file_stat_sha256'],
            'Target changed during prediction')
    require(sha(weights) == binding['weights'][args.key]['sha256'], 'Checkpoint changed during prediction')


if __name__ == '__main__':
    p = parser()
    p.add_argument('--key',choices=KEYS,required=True)
    p.add_argument('--phase',choices=('smoke','evaluation'),required=True)
    p.add_argument('--report',type=__import__('pathlib').Path,required=True)
    args = p.parse_args()
    try:
        run(args)
    except BaseException:
        print(traceback.format_exc(),flush=True)
        sys.exit(1)
