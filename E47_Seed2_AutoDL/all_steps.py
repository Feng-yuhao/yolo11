"""Run engineering gates then one formal job; a failed gate stops the pipeline."""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

STAGES = ('check', 'smoke', 'verify-gates', 'run')


def run_all(arguments, runner=None):
    runner = runner or subprocess.run
    script = Path(__file__).resolve().with_name('run_e47.py')
    if '--mode' in arguments or any(a.startswith('--mode=') for a in arguments):
        raise ValueError('all fixes the stage order; do not pass --mode')
    for stage in STAGES:
        print(f'[{datetime.now().isoformat(timespec="seconds")}] E47 seed2 stage={stage} START', flush=True)
        result = runner([sys.executable, '-u', str(script), '--mode', stage, *arguments],
                        check=False, stdin=subprocess.DEVNULL)
        if result.returncode:
            print(f'E47 seed2 STOP: {stage} failed exit={result.returncode}; '
                  'later stages were not started. Read the traceback above.', flush=True)
            return result.returncode
        print(f'[{datetime.now().isoformat(timespec="seconds")}] E47 seed2 stage={stage} FINISHED', flush=True)
    print('E47 seed2 ALL COMPLETED: formal training and official evaluation finished; no shutdown.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run_all(sys.argv[1:]))
