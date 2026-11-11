"""Read-only idle admission. Never kill a process or request shutdown."""
import csv
import io
import os
from pathlib import Path
import subprocess
from common import HERE, require


def capture(command):
    result = subprocess.run(command,capture_output=True,text=True,timeout=20)
    require(result.returncode == 0, 'Read-only GPU probe failed: '+result.stderr.strip())
    return result.stdout


def check_idle(proc_root=Path('/proc')):
    rows = list(csv.reader(io.StringIO(capture(['nvidia-smi','--query-gpu=memory.total,memory.free',
                                               '--format=csv,noheader,nounits']))))
    require(len(rows) == 1, 'Expected one GPU')
    total,free = [int(v.strip()) for v in rows[0]]
    require(0 <= free <= total and free >= 6144, 'Need at least 6GiB free for fixed FP32/b4 inference')
    gpu_pids = capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).split()
    require(not gpu_pids, 'GPU is in use; wait for current training/evaluation to finish. No process was stopped.')
    require(proc_root.is_dir(), 'Cannot verify Linux process state')
    jobs = []
    for directory in proc_root.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            stat = (directory/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0] in ('Z','X'):
                continue
            argv = [v.decode(errors='replace') for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError,ProcessLookupError):
            continue
        if not argv:
            continue
        name = Path(argv[0]).name.lower()
        if not ('python' in name or name in ('torchrun','yolo')):
            continue
        i = 1
        while i < len(argv) and argv[i].startswith('-') and argv[i] not in ('-m','-c'):
            i += 2 if argv[i] in ('-W','-X') else 1
        target = argv[i] if i < len(argv) else ''
        module = argv[i+1] if target == '-m' and i+1 < len(argv) else ''
        frontend = module in ('tensorboard','tensorboard.main','jupyterlab','jupyter_server','notebook')
        frontend = frontend or Path(target).name in ('jupyter-lab','jupyter-notebook','jupyter_server.py','tensorboard','supervisord')
        own_viewer = target not in ('-m','-c','') and Path(target).resolve() == HERE/'watch_progress.py'
        if not frontend and not own_viewer:
            jobs.append(dict(pid=int(directory.name),program=name))
    require(not jobs, 'Other Python tasks/kernels remain; finish them before source-only evaluation: '+str(jobs))
    return dict(total_MiB=total,free_MiB=free,other_GPU_PIDs=[],other_Python_jobs=[])
