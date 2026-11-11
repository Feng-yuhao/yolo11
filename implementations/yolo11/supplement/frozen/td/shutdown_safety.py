"""AutoDL shutdown probes only; no torch import and no import-time action.

Known UI services (including TensorBoard) do not block an otherwise idle GPU.
Unknown Python jobs, notebook kernels, training/evaluation and ANY CUDA PID do.
Actual shutdown is exclusively in shutdown_after.py, after the parent exits.
"""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp,path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def capture(command):
    result=subprocess.run(command,capture_output=True,text=True,timeout=20)
    if result.returncode:
        raise RuntimeError('GPU query unavailable: '+result.stderr.strip())
    return result.stdout


def gpu_snapshot():
    rows=list(csv.reader(io.StringIO(capture([
        'nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits']))))
    if len(rows)!=1 or len(rows[0])!=2:
        raise RuntimeError('Exactly one server GPU is required')
    total,free=[int(x.strip()) for x in rows[0]]
    if total<30*1024 or not 0<=free<=total:
        raise RuntimeError('Unexpected/unreliable GPU memory values')
    pids=[]
    for line in capture(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).splitlines():
        if not line.strip(): continue
        if not line.strip().isdigit():
            raise RuntimeError('GPU process enumeration is unreliable: '+line)
        pids.append(int(line.strip()))
    return dict(total_MiB=total,free_MiB=free,compute_pids=sorted(set(pids)))


def python_target(argv):
    """Handle python -u/-I SCRIPT as well as python -m MODULE; never print secrets."""
    i=1
    while i<len(argv):
        token=argv[i]
        if token=='-m': return ('module',argv[i+1] if i+1<len(argv) else '')
        if token=='-c': return ('inline','<inline>')
        if token in ('-W','-X'): i+=2; continue
        if token.startswith('-'): i+=1; continue
        return ('script',token)
    return ('interactive','<interactive>')


def classify(argv, package):
    if not argv: return None
    executable=Path(argv[0]).name.lower()
    if 'python' not in executable and executable not in ('torchrun','yolo','yolo.exe'):
        return None
    kind,target=python_target(argv)
    name=Path(target).name.lower()
    # Only known frontends/platform daemons are exempt, never ipykernel.
    service=(kind=='module' and target in ('tensorboard','tensorboard.main','jupyterlab','jupyter_server','notebook'))
    service=service or (kind=='script' and name in ('tensorboard','jupyter-lab','jupyter-notebook','jupyter_server.py','supervisord'))
    service=service or (kind=='script' and name=='main.py' and Path(target).parent.name.lower()=='tensorboard')
    own_viewer=(kind=='script' and Path(target).resolve()==(Path(package)/'watch_progress.py').resolve())
    return dict(program=executable,kind=kind,target=target,
                policy='ignored_readonly_frontend' if service or own_viewer else 'blocks_shutdown')


def process_identity(pid, proc_root=Path('/proc')):
    try:
        stat=(Path(proc_root)/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):
        return None
    if stat[0] in ('Z','X'): return None
    return dict(pid=int(pid),start_ticks=int(stat[19]))


def process_snapshot(package, proc_root=Path('/proc'), exclude_pid=None):
    proc_root=Path(proc_root)
    if not proc_root.is_dir(): raise RuntimeError('Linux /proc unavailable; cannot prove idle')
    exclude_pid=os.getpid() if exclude_pid is None else exclude_pid
    jobs=[]; ignored=[]
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or int(proc.name)==exclude_pid: continue
        try:
            identity=process_identity(int(proc.name),proc_root)
            if identity is None: continue
            argv=[v.decode('utf-8',errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError,ProcessLookupError): continue
        # Permission or incomplete scans fail closed; do not classify them as idle.
        item=classify(argv,package)
        if item is None: continue
        item.update(identity)
        (ignored if item['policy']=='ignored_readonly_frontend' else jobs).append(item)
    return dict(jobs=jobs,ignored_services=ignored)


def require_idle(gpu, processes, admission=False):
    if gpu['compute_pids'] or processes['jobs']:
        raise RuntimeError('Other GPU/Python tasks active: '+json.dumps(dict(gpu=gpu,processes=processes),ensure_ascii=False))
    needed=28*1024 if admission else gpu['total_MiB']-2048
    if gpu['free_MiB']<needed:
        raise RuntimeError('GPU memory not released: '+str(gpu))


def shutdown_invocation(command=Path('/usr/bin/shutdown')):
    command=Path(command)
    if not command.is_file(): raise RuntimeError('AutoDL shutdown helper missing: '+str(command))
    data=command.read_bytes()
    if not data or len(data)>65536 or b'\0' in data or data.startswith(b'\x7fELF'):
        raise RuntimeError('Not the expected container-owned AutoDL text shutdown helper')
    # Avoid accidentally invoking host/system poweroff. This is the observed
    # AutoDL helper, which logs to PID1 and stops its container supervisor.
    if not all(token in data for token in (b'user execute shutdown command in container',b'/proc/1/fd/1',b'supervisord')):
        raise RuntimeError('Unrecognized platform shutdown helper; no generic fallback')
    return dict(args=['/bin/bash',str(command)],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),
                no_shebang_supported=True,called=False)


def readiness(package):
    if sys.platform!='linux': raise RuntimeError('AutoDL shutdown preflight is Linux-only')
    command=shutdown_invocation()
    gpu=gpu_snapshot(); processes=process_snapshot(package)
    require_idle(gpu,processes,admission=True)
    return dict(status='shutdown_preflight_passed',command=command,gpu=gpu,processes=processes,
                torch_imported=False,shutdown_called=False)
