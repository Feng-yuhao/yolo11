#!/usr/bin/env python3
"""One isolated process per seed, architecture, phase and checkpoint namespace."""
from __future__ import annotations
import sys, argparse, traceback, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common7 import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',required=True,choices=['prepare','check','smoke','train','score'])
    p.add_argument('--key',required=True,choices=list(JOBS))
    p.add_argument('--project',type=Path,default=DEFAULT_PROJECT);p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--test-dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--cap-gib',type=float,default=28.)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pair',type=Path);p.add_argument('--split',choices=['val','test-dev'],default='val')
    args=p.parse_args();args.variant=JOBS[args.key]['variant'];args.workers=8
    for n in ['project','source','data','dataset','test_dataset','report','out'] :setattr(args,n,getattr(args,n).resolve())
    contained(args.report,args.project/'comparison_reports');contained(args.out,args.report)
    args.out.mkdir(parents=True,exist_ok=False)
    print(f'{now()} {args.key} {args.phase} START cap={args.cap_gib:g}GiB report={args.out}',flush=True)
    try:
        import engine7 as e
        e.setup(args)
        if args.phase=='prepare':e.prepare(args,args.out)
        elif args.phase=='check':e.check(args,args.out)
        elif args.phase=='smoke':e.smoke(args,args.out)
        elif args.phase=='train':e.train(args,args.out)
        else:
            from scoring7 import evaluate
            evaluate(args,args.out)
        print(f'{now()} {args.key} {args.phase} FINISHED',flush=True)
        return 0
    except BaseException as error:
        torch=sys.modules.get('torch')
        category='cuda_oom' if torch is not None and isinstance(error,torch.cuda.OutOfMemoryError) else 'gpu_headroom' if str(error).startswith('GPU headroom below') else 'peer_failed' if str(error).startswith('Peer failed during pair smoke') else 'error'
        value=dict(status='failed',key=args.key,phase=args.phase,at=now(),category=category,error=traceback.format_exc())
        write(args.out/'failure.json',value)
        if args.pair:
            write(args.pair/'failures'/(args.key+'.json'),value)
        print(value['error'],file=sys.stderr,flush=True)
        return 42 if category in ['cuda_oom','gpu_headroom'] else 1

if __name__=='__main__':sys.exit(main())
