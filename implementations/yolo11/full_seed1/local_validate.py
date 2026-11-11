"""Developer small-fixture CPU/CUDA test: NO server gate, NO remote shutdown."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import run_experiment as run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--integration',action='store_true')
    options=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    args=SimpleNamespace(project=root/'yolo11_project',source=root/'yolo11_src',
                         data=root/'tmp/missing_local_dataset.yaml',dataset=root/'tmp/missing_local_images',
                         device='cpu',workers=8,variant='e60',mode='check')
    core=run.setup(args)
    run.protocol_check(core,args)
    args.device=options.device
    import torch
    if options.device!='cpu':
        if not torch.cuda.is_available():raise RuntimeError('Local CUDA unavailable')
        torch.cuda.set_device(options.device); torch.cuda.reset_peak_memory_stats()
    out=root/'tmp'/('e60_local_'+('integration' if options.integration else options.device.replace(':',''))+
                      '_'+run.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    from checks import check_all,integration
    if options.integration:
        if args.device!='cpu': raise RuntimeError('Synthetic integration entry uses CPU')
        integration(core,args,out)
    else:
        result=check_all(core,args,out)
        result.update(local_fixture_only=True,not_server_smoke=True)
        if options.device!='cpu': result['peak_reserved_GiB']=torch.cuda.max_memory_reserved()/2**30
        run.write(out/'local_check.json',result)
    print('LOCAL FIXTURE ONLY: no server receipt issued. '+str(out),flush=True)


if __name__=='__main__':main()
