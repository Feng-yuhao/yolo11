"""Offline tests and bounded local CUDA units; NOT a server smoke receipt."""
import argparse
import ast
import json
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cpu')
    p.add_argument('--result-file',type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    for file in root.glob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    from checks import strict_math
    from extension import routing_audit,branch_groups
    from depth_readout import added_conv_flops,ADDED_PARAMETERS
    from spatial_task_head import NEW_PARAMETERS
    with strict_math(): result=routing_audit(args.device)
    assert len(set(branch_groups()))==26
    assert ADDED_PARAMETERS==73728 and NEW_PARAMETERS==249168
    assert added_conv_flops(640)==2359296000
    from smoke_progress import SmokeProgress
    progress=SmokeProgress()
    for i in range(8):
        progress.observe(dict(updated=True,status='finite_update',bad_parameter_count=0))
    progress.require_success()
    stalled=SmokeProgress()
    for i in range(8):
        stalled.observe(dict(updated=False,status='overflow_skip',bad_parameter_count=1))
    try: stalled.require_success()
    except RuntimeError: pass
    else: raise AssertionError('AMP-skipped updates incorrectly accepted')
    import d27_official as official
    import d38_common as common
    official.run_unit_tests(); common.tests()
    if args.result_file:
        args.result_file.parent.mkdir(parents=True,exist_ok=True)
        args.result_file.write_text(json.dumps(dict(status='passed',torch=torch.__version__,
            device=args.device,synthetic_unit_only=True,server_smoke_executed=False,
            result=result),indent=2),encoding='utf-8')
    print('E47 UNIT/STATIC TESTS PASSED device='+args.device+' '+str(result),flush=True)


if __name__=='__main__': main()
