"""File barrier used ONLY for the real concurrent engineering smoke."""
import json, os, time
from pathlib import Path
def synchronize(phase):
    directory=os.environ.get('PAIR_SMOKE_DIR')
    if not directory: raise RuntimeError('Run real smoke through run_pair.sh smoke')
    root=Path(directory); variant=os.environ['PAIR_VARIANT']
    (root/(variant+'_'+phase)).write_text(str(os.getpid()))
    deadline=time.monotonic()+600
    while not all((root/(v+'_'+phase)).is_file() for v in ('e50','e51')):
        if (root/'aborted').exists(): raise RuntimeError('Parallel smoke peer failed')
        if time.monotonic()>deadline: raise RuntimeError('Parallel smoke barrier timed out: '+phase)
        time.sleep(.5)
