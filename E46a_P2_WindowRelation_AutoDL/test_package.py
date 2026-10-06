"""Repeatable offline unit/static tests; no server training or dataset writes."""
import ast
from pathlib import Path
import torch

def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parent
    for file in root.glob('*.py'): ast.parse(file.read_text(encoding='utf-8'),filename=str(file))
    from extension import routing_audit
    from window_context import WindowContext,context_flops
    from spatial_task_head import CONTEXT_PARAMETERS
    result=routing_audit('cpu')
    assert sum(p.numel() for p in WindowContext().parameters())==CONTEXT_PARAMETERS==93320
    assert sum(context_flops(160,160).values())==5452595200
    from smoke_progress import SmokeProgress
    progress=SmokeProgress()
    for i in range(8): progress.observe(dict(updated=True,status='finite_update',bad_parameter_count=0))
    progress.require_success()
    import d27_official as official
    import d38_common as common
    official.run_unit_tests(); common.tests()
    print('E46 OFFLINE UNIT/STATIC TESTS PASSED',result,flush=True)

if __name__=='__main__': main()
