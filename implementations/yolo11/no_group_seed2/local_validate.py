"""Developer-only small CUDA test; no server smoke gate and no shutdown."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import run_experiment as run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    options=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    args=SimpleNamespace(project=root/'yolo11_project',source=root/'yolo11_src',
                         data=root/'tmp/missing_local_dataset.yaml',dataset=root/'tmp/missing_local_images',
                         device='cpu',workers=8,variant='e58',mode='check')
    core=run.setup(args)
    run.protocol_check(core,args)
    args.device=options.device
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA unavailable')
    torch.cuda.set_device(options.device)
    torch.cuda.reset_peak_memory_stats()
    out=root/'tmp'/('e58_seed2_local_cuda_'+run.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    from checks import check_all
    result=check_all(core,args,out)
    result.update(local_small_cuda_only=True,not_server_smoke=True,
                  peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
    run.write(out/'local_check.json',result)
    print('LOCAL SMALL CUDA ONLY; no server receipt issued. '+str(out),flush=True)


if __name__=='__main__':main()
