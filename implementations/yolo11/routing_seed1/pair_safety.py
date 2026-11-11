"""Joint smoke rendezvous: peer identity/liveness and retained residency."""
import os
from pathlib import Path
import time
from common import VARIANTS, read, write, require, now


def start_ticks(pid):
    path = Path('/proc')/str(pid)/'stat'
    if not path.is_file():
        raise RuntimeError('Cannot inspect worker process identity')
    return int(path.read_text().split(') ', 1)[1].split()[19])


def barrier(args, stage, payload=None, timeout=3600):
    if args.execution == 'serial':
        return
    require(args.pair is not None, 'Joint smoke requires a new controller session')
    directory = args.pair/'barriers'/stage
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/(args.variant+'.json'), dict(pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
          stage=stage, at=now(), payload=payload or {}))
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if (args.pair/'abort.json').exists():
            raise RuntimeError('Joint smoke peer/controller failed: '+str(args.pair/'abort.json'))
        registry = args.pair/'children.json'
        jobs = read(registry) if registry.is_file() else None
        if jobs is not None:
            complete = all((directory/(v+'.json')).is_file() for v in VARIANTS)
            if complete and stage == 'both_smoke_receipts_saved':
                # At the last rendezvous a peer may exit after publishing its
                # successful receipt. Earlier phases still require live peers.
                for v in VARIANTS:
                    marker = read(directory/(v+'.json'))
                    require((marker['pid'],marker['start_ticks']) ==
                            (jobs[v]['pid'],jobs[v]['start_ticks']), 'Stale final smoke barrier')
                return
            for variant in VARIANTS:
                job = jobs[variant]
                try:
                    actual = start_ticks(job['pid'])
                except RuntimeError:
                    raise RuntimeError('Joint smoke peer exited before '+stage)
                require(actual == job['start_ticks'], 'Joint smoke PID identity changed')
            if complete:
                for v in VARIANTS:
                    marker = read(directory/(v+'.json'))
                    require((marker['pid'],marker['start_ticks']) ==
                            (jobs[v]['pid'],jobs[v]['start_ticks']), 'Stale joint smoke barrier')
                print(now()+' '+args.variant+' JOINT '+stage+' both workers resident', flush=True)
                return
        time.sleep(.25)
    raise RuntimeError('Timed out at joint smoke barrier: '+stage)
