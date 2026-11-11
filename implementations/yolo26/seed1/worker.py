"""One isolated process per smoke, formal training or official scoring job."""
import argparse
import json
from pathlib import Path
import shutil
import traceback
import common as c
from protocol import write,now,CAP_GIB,DATA


def arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('smoke','train','eval'))
    parser.add_argument('--variant',choices=('baseline','e60'),required=True)
    parser.add_argument('--project',type=Path,default=c.PROJECT)
    parser.add_argument('--source',type=Path,default=c.SOURCE)
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--val-dataset',type=Path,default=DATA.parent/'VisDrone2019-DET-val')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--signature',type=Path)
    parser.add_argument('--local-test',action='store_true')
    parser.add_argument('--local-size',type=int,default=128)
    parser.add_argument('--local-epochs',type=int,default=2)
    return parser.parse_args()


def main():
    args=arguments(); c.runtime(args.project)
    if not args.local_test: c.resource_guard()
    audit=c.audit(args.source,args.local_test)
    from protocol import audit_training_source
    audit['training_source_files_checked']=audit_training_source(args.source)
    args.report.mkdir(parents=True,exist_ok=False)
    write(args.report/'runtime_audit.json',audit)
    source_copy=args.report/'actual_source'; source_copy.mkdir()
    for name in json.loads((c.HERE/'manifest.json').read_text()):
        if name.startswith('weights/'): continue
        src=c.HERE/name; dst=source_copy/name; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
    try:
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for these jobs')
        free,total=torch.cuda.mem_get_info(0)
        cap=4 if args.local_test else CAP_GIB
        if total<cap*2**30 or free<(3 if args.local_test else 28)*2**30:
            raise RuntimeError(f'Need {cap} GiB capacity and an idle GPU for this fixed layout; free={free/2**30:.2f} GiB')
        torch.cuda.set_per_process_memory_fraction(cap*2**30/total,0)
        torch.cuda.reset_peak_memory_stats(0)
        from protocol import deploy_amp_weight,verify_gate
        deploy_amp_weight(args)
        if args.local_test:
            import d27_official
            d27_official.EXPECTED_IMAGES=len(list((args.val_dataset/'annotations').glob('*.txt')))
        if args.mode=='train' and not args.local_test:
            if args.signature is None: raise RuntimeError('Use the controller; formal run requires bound real smoke')
            from protocol import signature
            current=signature(args)
            expected=json.loads(args.signature.read_text())
            if current['fingerprint']!=expected['fingerprint']:
                raise RuntimeError('Code/environment/data changed after controller preflight')
            verify_gate(args,current)
        from training26 import quiet_progress,smoke,train
        quiet_progress()
        print(f'{now()} {args.variant.upper()} {args.mode} START REPORT={args.report}',flush=True)
        if args.mode=='smoke': result=smoke(args,args.variant,args.report)
        elif args.mode=='train': result=train(args,args.variant,args.report)
        else:
            if args.weights is None: raise RuntimeError('eval requires fixed --weights')
            from scoring26 import evaluate
            result=evaluate(args,args.variant,args.weights,args.report)
        write(args.report/'completion.json',dict(status='completed',result=result,finished_at=now(),local_fixture=args.local_test))
        print(f'{now()} {args.variant.upper()} {args.mode} FINISHED',flush=True)
    except BaseException:
        write(args.report/'failure.json',dict(status='failed',error=traceback.format_exc(),finished_at=now()))
        raise


if __name__=='__main__': main()
