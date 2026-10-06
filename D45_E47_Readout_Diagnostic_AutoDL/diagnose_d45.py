#!/usr/bin/env python3
"""Bounded E47 readout diagnosis. No training, remote mutation or shutdown."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from io_utils import HERE, KEYS, package_signature, read, resolve_reports, write


def report(out, metrics, offline):
    lines = ['# D45: E47新增深度读出诊断', '',
             '仅推理与已有缓存分析，不训练、不修改模型/源码，不使用test-dev。',
             '局部旁路说明已训练权重的条件性依赖，不证明训练根因或删路后重训收益。', '',
             '|状态|Official AP|AP50|AP75|AR500|AP相对同权重normal|',
             '|---|---:|---:|---:|---:|---:|']
    for state, value in metrics['scores'].items():
        numbers = [value['official'][k] for k in KEYS]
        lines.append('|' + state + '|' + '|'.join(f'{v:.6f}' for v in numbers) +
                     f"|{value['delta_vs_normal_pp'][KEYS[0]]:+.6f}|")
    lines.extend(['', '## 如何读取结果', '',
                  '- 先看offline/REPORT_OFFLINE.md：E47对配对E40的类别、尺寸、个体GT改善与损失。',
                  '- 再看同一E47权重的单点旁路；normal必须先复现原始预测与官方AP。',
                  '- 每状态*_geometry.json记录候选存在覆盖率；不是官方尺度AP或一对一Recall。',
                  '- readout_summary.json按P2/P3、尺寸、类别及成功/失败统计新增读出强度和残差关系。',
                  '- 新增RMS非零、路由熵大或残差相反，不单独证明收益/冲突；必须结合旁路结果。',
                  '- 四个单点变化不能相加；all_readout_off也不是重新训练的E40。',
                  '- 关闭后变差可来自协同适应；关闭后变好只生成待验证假设，不自动批准新训练。',
                  '- 本次是单seed；seed2完成后用相同设置复核关键模式，不能由一次诊断解释种子波动。', ''])
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    write(out / 'decision.json', dict(status='completed', automatic_training_authorized=False,
          seed=metrics['seed'], interpretation='Inference dependence and remaining errors, not root cause.',
          normal_reproduced=True, scores=metrics['scores'], offline_report=str(out / 'offline'),
          next_action='Review replicated AP/AP75 and paired deficits; do not launch a new variant automatically.'))


def child(args, mode, out):
    command = [sys.executable, '-u', str(HERE / 'diagnose_worker.py'), '--mode', mode,
               '--seed', str(args.seed), '--project', str(args.project), '--source', str(args.source),
               '--dataset', str(args.dataset), '--device', args.device,
               '--memory-cap-gib', str(args.memory_cap_gib), '--out', str(out)]
    if args.e47_report is not None:
        command += ['--e47-report', str(args.e47_report)]
    if args.e40_report is not None:
        command += ['--e40-report', str(args.e40_report)]
    if args.allow_busy_gpu:
        command.append('--allow-busy-gpu')
    log = out.parent / (mode + '.log')
    print(f'D45 {mode} START live-log={log}', flush=True)
    with log.open('w', encoding='utf-8') as handle:
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1, env=env)
        try:
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()  # this diagnostic child only, never any training
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print('Diagnostic child still exiting; no training process was touched.', flush=True)
            raise
    if code:
        raise RuntimeError(f'D45 {mode} failed exit={code}; see {log}. No training started.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('offline', 'check', 'smoke', 'run', 'all'), required=True)
    parser.add_argument('--seed', type=int, choices=(1, 2), default=1)
    parser.add_argument('--project', type=Path, default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--dataset', type=Path, default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--e47-report', type=Path)
    parser.add_argument('--e40-report', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--memory-cap-gib', type=float, default=6.)
    parser.add_argument('--allow-busy-gpu', action='store_true', help='Explicitly allow inference alongside another CUDA process; may slow training.')
    args = parser.parse_args()
    args.project = args.project.resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = args.project / 'comparison_reports' / f'DIAG_D45_{args.mode.upper()}_S{args.seed}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', zero_training=True, automatic_shutdown=False, seed=args.seed,
                 mode=args.mode, started_at=datetime.now().astimezone().isoformat(), report_dir=str(out))
    write(out / 'controller.json', state)
    print(f'D45 mode={args.mode} seed={args.seed} REPORT={out}; ZERO TRAINING', flush=True)
    started = time.perf_counter()
    try:
        package_signature()
        e47, e40 = resolve_reports(args.project, args.seed, args.e47_report, args.e40_report)
        from offline_analysis import summarize_pair
        offline = summarize_pair(e47, e40, out / 'offline')
        print(f'D45 OFFLINE COMPLETE {out / "offline/REPORT_OFFLINE.md"}; no CUDA was used.', flush=True)
        if args.mode != 'offline':
            modes = ('check', 'smoke', 'run') if args.mode == 'all' else (args.mode,)
            for mode in modes:
                child(args, mode, out / mode)
                if mode == 'run':
                    report(out, read(out / mode / 'metrics.json'), offline)
        state.update(status='completed', finished_at=datetime.now().astimezone().isoformat(),
                     elapsed_seconds=time.perf_counter() - started)
        write(out / 'controller.json', state)
        print(f'D45 {args.mode.upper()} FINISHED {state["finished_at"]} '
              f'elapsed={state["elapsed_seconds"]/60:.1f}min REPORT={out}', flush=True)
    except BaseException:
        state.update(status='failed', finished_at=datetime.now().astimezone().isoformat(), error=traceback.format_exc())
        write(out / 'controller.json', state)
        raise


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
