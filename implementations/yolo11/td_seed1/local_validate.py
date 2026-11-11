"""Developer fixtures only. No server receipts, remote connection or shutdown."""
import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import time


def integration(core, args, out):
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics.utils import yaml_save
    from ultralytics import YOLO
    from p2_tal_chunked import chunked_assignment
    from run_experiment import trainer_class, config
    from checks import require
    from evaluation import predict
    import d27_official as official
    fixture = out / 'fixture'
    for split in ('train','val'):
        (fixture / 'images' / split).mkdir(parents=True)
        (fixture / 'labels' / split).mkdir(parents=True)
        for i in range(4):
            Image.fromarray(np.random.default_rng(i).integers(0,255,(128,160,3),dtype=np.uint8)).save(
                fixture / 'images' / split / f'{i}.jpg')
            (fixture / 'labels' / split / f'{i}.txt').write_text(
                '3 0.5 0.5 0.9 0.9\n3 0.5 0.5 0.4 0.4\n0 0.15 0.15 0.08 0.09\n')
    yaml = fixture / 'data.yaml'
    names = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
    yaml_save(yaml,dict(path=str(fixture.resolve()),train='images/train',val='images/val',nc=10,names=names))
    params = config(core,args)
    params.update(data=str(yaml),device=args.device,epochs=2,imgsz=160,batch=2,workers=0,
                  amp=False,project=str(out / 'runs'),name='integration',plots=False,mosaic=0.,
                  close_mosaic=0,warmup_epochs=0.,nbs=2)
    report = dict(report_dir=str(out),_clock=time.perf_counter())
    trainer = trainer_class(core,report)(overrides=params)
    with chunked_assignment():
        trainer.train()
    require(trainer.best.is_file() and trainer.last.is_file(),'Native Trainer failed to save both checkpoints')
    from tood_td_head import assert_topology, output_magnitudes
    for checkpoint in (trainer.best,trainer.last):
        loaded = YOLO(str(checkpoint),task='detect')
        assert_topology(loaded.model)
        require(all(v>0 for v in output_magnitudes(loaded.model)),'Saved EMA TD adapter underflowed to zero')
    of = out / 'official_fixture'
    (of / 'images').mkdir(parents=True); (of / 'annotations').mkdir()
    import shutil
    for i in range(4):
        shutil.copy2(fixture / 'images/val' / f'{i}.jpg',of / 'images' / f'{i}.jpg')
        (of / 'annotations' / f'{i}.txt').write_text('64,51,32,26,1,4,0,0\n')
    previous = official.EXPECTED_IMAGES
    try:
        official.EXPECTED_IMAGES = 4
        records,_ = official.load_dataset(of)
    finally:
        official.EXPECTED_IMAGES = previous
    result = predict(args,trainer.best,records,out / 'official_predictions')
    filtered,_ = official.filter_detections_and_audit(records,
        official.load_predictions(out / 'official_predictions/area_predictions.json',4))
    metrics = official.official_evaluate(records,filtered)
    return dict(status='passed',local_only=True,not_server_smoke=True,epochs=2,
                native_Trainer_save_EMA_reload=True,standard_predictor=result,
                toy_official_metrics=metrics,no_remote_or_shutdown=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True); p.add_argument('--project',type=Path,required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--integration',action='store_true')
    args = p.parse_args()
    args.project,args.source = args.project.resolve(),args.source.resolve()
    args.data = args.project / 'MISSING_LOCAL_DATA.yaml'
    args.dataset = args.test_dataset = args.project / 'MISSING_LOCAL_IMAGES'
    args.mode,args.workers,args.weights = 'check',8,None
    out = Path(__file__).resolve().parents[2] / 'audit' / (
        'TOOD_TD_LOCAL_' + ('integration' if args.integration else args.device.replace(':','')) +
        '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    args.report_dir = args.report_root = out
    from run_experiment import setup, protocol_check
    from shutdown_safety import write
    try:
        core = setup(args,local=True)
        protocol_check(core,args,out,local=True)
        if args.integration:
            result = integration(core,args,out)
        else:
            from checks import check_all
            result = check_all(core,args,out)
        result.update(local_only=True,not_server_smoke=True,no_remote_or_shutdown=True)
        write(out / 'local_result.json',result)
    except BaseException:
        import traceback
        write(out / 'failure.json',dict(local_only=True,error=traceback.format_exc()))
        raise
    print('LOCAL FIXTURE ONLY; no server gate: ' + str(out),flush=True)


if __name__ == '__main__':
    main()
