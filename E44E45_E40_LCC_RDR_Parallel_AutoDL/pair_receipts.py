"""Parallel run requires BOTH receipts from the SAME concurrent full smoke."""
import json
from pathlib import Path


def require_parallel_smoke(project):
    roots=[]
    for name in ('e44','e45'):
        receipt=json.loads((Path(project)/'comparison_reports'/f'{name}_smoke_gate.json').read_text())
        evidence=json.loads((Path(receipt['report'])/'smoke.json').read_text())
        validation=evidence.get('validation',{})
        if evidence.get('status')!='passed' or validation.get('parallel_pair') is not True:
            raise RuntimeError('Parallel run needs run_pair.sh smoke, not separate single-task smoke: '+name)
        roots.append(Path(validation['barrier_directory']))
    if roots[0]!=roots[1]:
        raise RuntimeError('The two smoke receipts were not produced in the same parallel test')
    state=json.loads((roots[0]/'state.json').read_text())
    if state.get('mode')!='smoke' or state.get('exit_codes')!={'e44':0,'e45':0} or (roots[0]/'abort').exists():
        raise RuntimeError('Parallel smoke controller did not complete successfully')
    if not all((roots[0]/(name+'.ready')).is_file() for name in ('e44','e45')):
        raise RuntimeError('Parallel validation barrier evidence missing')
    return str(roots[0])
