"""Small local CUDA full-graph tests. NOT a 32GiB real-data smoke or gate."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT=Path(__file__).resolve().parents[1]
if len(sys.argv)!=2 or sys.argv[1] not in ('e53','e54'): raise ValueError('Specify e53 or e54')
variant=sys.argv[1]
sys.path.insert(0,str(Path(__file__).resolve().parent/variant))
import run_experiment
args=SimpleNamespace(project=ROOT/'yolo11_project',source=ROOT/'yolo11_src',
    data=Path('/not-present-local-dataset.yaml'),dataset=Path('/not-present-local-val'),
    device='cpu',mode='check',workers=8)
core=run_experiment.setup(args)
run_experiment.protocol_check(core,args)
if not torch.cuda.is_available(): raise RuntimeError('Local CUDA unavailable')
args.device='cuda:0'; torch.cuda.set_device(0)
out=ROOT/'tmp'/('e53e54_local_cuda_'+variant)
out.mkdir(parents=True,exist_ok=True)
from checks import check_all
result=check_all(core,args,out)
result.update(boundary='Small synthetic local CUDA, not server batch16/640 concurrent smoke',server_gate_issued=False)
run_experiment.write(out/'result.json',result)
print('LOCAL SMALL CUDA PASSED',variant,flush=True)
