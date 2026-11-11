"""CPU-only helper; no shutdown while a controller/worker is still active."""
import argparse
import os
from pathlib import Path
import time
from shutdown_safety import auto_shutdown

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent',type=int,required=True)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True)
    a = p.parse_args()
    while (Path('/proc')/str(a.parent)).exists():
        time.sleep(5)
    auto_shutdown(True,a.session,a.project/'KEEP_RUNNING_E60_ABLATIONS',delay=60)
