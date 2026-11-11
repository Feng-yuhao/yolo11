"""One fresh process per check/smoke/train/score. No YOLO11 imports."""
import argparse
from pathlib import Path
import shutil
import traceback
import common as c
from protocol import (runtime, source_audit, signature, verify_gate, write, read, now, require,
                      DATA, CAP_GIB, EVAL_CAP_GIB, training_weights, evaluate, actual_run_name)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('check', 'smoke', 'train', 'eval'))
    parser.add_argument('--variant', choices=('baseline', 'e60'), default='e60')
    parser.add_argument('--split', choices=('val', 'test-dev'), default='val')
    parser.add_argument('--project', type=Path, default=c.PROJECT)
    parser.add_argument('--source', type=Path, default=c.SOURCE)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--val-dataset', type=Path, default=DATA.parent / 'VisDrone2019-DET-val')
    parser.add_argument('--dataset', type=Path, default=DATA.parent / 'VisDrone2019-DET-test-dev')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--signature', type=Path, required=True)
    parser.add_argument('--training-report', type=Path)
    parser.add_argument('--local-test', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--local-size', type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument('--local-epochs', type=int, default=2, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = arguments()
    runtime(args, args.variant)
    args.report.mkdir(parents=True, exist_ok=False)
    try:
        audit = source_audit(args)
        write(args.report / 'runtime_audit.json', audit)
        import torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
        free, total = torch.cuda.mem_get_info(0)
        cap = 4 if args.local_test else (EVAL_CAP_GIB if args.mode == 'eval' else CAP_GIB)
        minimum = 3 if args.local_test else cap
        require(free >= minimum * 2**30 and total >= cap * 2**30,
                f'Need a free GPU for serial {args.mode}; need={minimum}GiB free={free / 2**30:.2f}GiB')
        torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        current = signature(args)
        require(current['fingerprint'] == read(args.signature)['fingerprint'], 'Code/data/environment changed after preflight')
        if args.mode == 'train':
            verify_gate(args, current)
            require(not (args.project / 'runs' / actual_run_name(args, args.variant)).exists(),
                    'Existing formal run; will not overwrite, resume or restart')
        from training26 import quiet_progress
        quiet_progress()
        print(f'{now()} {args.variant.upper()} seed3 {args.mode} START REPORT={args.report}', flush=True)
        if args.mode == 'check':
            from checks26 import check
            result = check(args.project / 'weights/yolo26s.pt', 'cuda:0', args.report, args.local_test)
        elif args.mode in ('smoke', 'train'):
            import training26
            # Only seed3 run naming changes. Numerical training/model code stays frozen.
            training26.run_name_for = actual_run_name
            result = (training26.smoke if args.mode == 'smoke' else training26.train)(args, args.variant, args.report)
            if args.mode == 'train':
                result.update(seed=3, protocol_fingerprint=current['fingerprint'])
                write(args.report / 'training_completion.json', result)
        else:
            require(args.training_report is not None, 'eval requires matching completed training report')
            weights = training_weights(args, args.training_report, args.variant)
            result = evaluate(args, args.variant, weights, args.report, args.split)
        write(args.report / 'completion.json', dict(status='completed', seed=3, result=result,
              local_fixture=args.local_test, fingerprint=current['fingerprint'], finished_at=now()))
        print(f'{now()} {args.variant.upper()} seed3 {args.mode} FINISHED', flush=True)
    except BaseException:
        write(args.report / 'failure.json', dict(status='failed', error=traceback.format_exc(), finished_at=now()))
        raise


if __name__ == '__main__':
    main()
