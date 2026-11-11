"""CPU-only helper: wait for parent exit, then use existing guarded shutdown."""
import argparse
import os
from pathlib import Path
import time
import common as c
from shutdown_safety import auto_shutdown


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=int,required=True)
    parser.add_argument('--session',type=Path,required=True)
    args=parser.parse_args()
    start=time.monotonic()
    while (Path('/proc')/str(args.parent)).exists():
        if time.monotonic()-start>180:
            c.write(args.session/'shutdown_status.json',dict(status='skipped_parent_still_alive',enabled=True))
            return
        time.sleep(2)
    result=auto_shutdown(True,args.session,c.PROJECT/'KEEP_RUNNING_E0_S23_E60',delay=60)
    state=c.read(args.session/'state.json');state['shutdown']=result
    c.write(args.session/'state.json',state)


if __name__=='__main__':main()
