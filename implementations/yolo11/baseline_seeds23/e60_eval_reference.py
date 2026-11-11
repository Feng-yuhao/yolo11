#!/usr/bin/env python3
"""E60 seed1 frozen val-best -> official test-dev. No training or shutdown."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import platform
import os
from pathlib import Path
import sys
import traceback

import base_eval as base

HERE = Path(__file__).resolve().parent
REF = HERE / 'references'
REVISION = 'testdev_e60_seed1_v1'
KEY = 'e60_seed1'
SPEC = dict(weights='runs/e60a_e58_group_residual_encoding_img640_seed1_200e/weights/best.pt',
            sha256='bf38299c8723f5fe399e2f94fa4042df2ff6ae51dcb99fceece8bc85ed00dc17',
            nl=4, stride=(4, 8, 16, 32))
BASELINES = {
    'e0_seed1': 'bc5d1304c7bf4e0c81db853d0d7a39dd46485940ec2b8c2da63df8c3eda934b8',
    'e1_seed1': 'f1a5ba6642a01445a99b2c7c80d8e79299a4402159bc37d2a08bb31ff365df12',
    'e40_seed1': '0a9fdac26c4fe7a9161900c82e879f584026afa569f870c0159d41feca031337',
    'e58_seed1': 'd4afe6bcbedb415f5551a09e3f48e6f0d436d6a49d0d8080a5b9f6394fb7319e',
}
base.SPECS = {KEY: SPEC}
require, read, write, sha, digest, now = base.require, base.read, base.write, base.sha, base.digest, base.now
KEYS = base.KEYS


def prepare(args, strict=True):
    require((args.source/'ultralytics/__init__.py').is_file(), 'Missing pinned yolo11_src')
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.insert(0, str(HERE))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(args.project/'comparison_reports/testdev_e60_seed1_settings')
    import torch
    import ultralytics
    import spatial_task_head
    from ultralytics.utils.torch_utils import init_seeds
    spatial_task_head.configure('e60')
    require(Path(ultralytics.__file__).resolve().parent == args.source.resolve()/'ultralytics',
            'Imported a different Ultralytics installation')
    if strict:
        original = read(REF/'e60_seed1_training.json')['signature']
        require(ultralytics.__version__ == '8.3.0', 'Ultralytics version differs')
        require(torch.__version__ == original['torch'] and torch.version.cuda == original['cuda'],
                'Torch/CUDA version differs from completed E60 evaluation')
    init_seeds(1, deterministic=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return torch


def model_audit(model, key):
    import torch
    import spatial_task_head as head_code
    import e58_spatial_task_head as parent
    head = model.model.model[-1]
    require(key == KEY and type(head) is head_code.SpatialTaskDetect and head.enabled and
            head.experiment_variant == 'e60', 'Wrong/disabled E60 trained head')
    require(head.nc == 10 and head.nl == 4 and tuple(head.stride.tolist()) == SPEC['stride'] and
            not head.end2end, 'Wrong E60 class/scale/stride')
    blocks = head.task_blocks
    require(len(blocks) == 4 and tuple(b.reduce[0].in_channels for b in blocks) == (128,128,256,512)
            and all(b.reduce[0].out_channels == 96 for b in blocks), 'E60 channel/width drift')
    for i, block in enumerate(blocks):
        require(len(block.depths) == 3, 'E60 original four-depth bank changed')
        if i < 2:
            require(type(block) is head_code.GroupResidualTaskBlock and len(block.refinements) == 3,
                    'Missing trained P2/P3 group residual encoders')
            for refine in block.refinements:
                require(type(refine) is head_code.GroupResidualEncoder and
                        len(refine.kernels) == 3 and refine.project.weight.shape == (96,96,1,1),
                        'E60 group residual topology drift')
                require(all(p.dtype == torch.float32 for p in refine.parameters()),
                        'E60 FP32 sidecar checkpoint precision was lost')
        else:
            require(type(block) is parent.WideTaskBlock and not hasattr(block, 'refinements'),
                    'P4/P5 must retain the original E58 implementation')
    for name, parameter in model.model.named_parameters():
        require(torch.isfinite(parameter).all().item(), 'Non-finite checkpoint tensor: '+name)
    return dict(head=type(head).__module__+'.'+type(head).__name__, nc=10, detection_scales=4,
                stride=list(SPEC['stride']), hidden=[96]*4, refinements=[3,3,0,0],
                fp32_sidecars=True, parameter_count=sum(p.numel() for p in model.model.parameters()))


base.model_audit = model_audit


def cached_references(protocol):
    summary = read(REF/'seed1_testdev_summary.json')
    require(summary['status'] == 'completed' and summary['seed'] == 1 and summary['split'] == 'test-dev',
            'Cached reference is not completed seed1 test-dev')
    cached = {}
    for key, weight_sha in BASELINES.items():
        metric = read(REF/(key+'_official_metrics.json'))
        pred = read(REF/(key+'_prediction_manifest.json'))
        binding = metric['binding']
        require(metric['status'] == pred['status'] == 'completed' and binding == pred['binding'],
                'Incomplete/mismatched cached '+key)
        checkpoint = binding['checkpoint_sha256']
        require((checkpoint[key] if isinstance(checkpoint, dict) else checkpoint) == weight_sha,
                'Wrong cached val-best checkpoint '+key)
        require(binding['dataset_annotation_sha256'] == base.TEST_ANN_SHA and
                binding['dataset_image_sha256'] == base.TEST_IMAGE_SHA and
                binding['image_count'] == pred['seen_images'] == 1610 and not binding.get('smoke_only',False),
                'Cached reference is not the full original test-dev '+key)
        require(binding['protocol'] == dict(protocol, device=binding['protocol']['device']),
                'Cached FP32/batch4/conf/NMS protocol differs '+key)
        require(metric['prediction_sha256'] == pred['sha256'] == summary['models'][key]['prediction_sha256'],
                'Cached prediction SHA differs '+key)
        require(metric['official'] == summary['models'][key]['official'], 'Cached scores differ '+key)
        require(pred['backend']['matmul_allow_tf32'] is False and
                pred['backend']['cudnn_allow_tf32'] is True,
                'Cached TF32 backend differs '+key)
        cached[key] = dict(official=metric['official'], binding=binding,
                           checkpoint_sha256=weight_sha, prediction_sha256=pred['sha256'],
                           provenance='Frozen completed same-seed test-dev; not rerun')
    return cached


def preflight(args, torch):
    package = base.audit_package()
    source = base.source_binding(args)
    prior_source = read(REF/'seed1_testdev_summary.json')['binding']['source']
    for name, value in source.items():
        # Unrelated YAML registrations accumulated in tasks.py. The actual
        # inference modules must still match the cached comparisons exactly.
        if name != 'nn/tasks.py':
            require(value == prior_source[name], 'Cached comparison inference source differs: '+name)
    training = read(REF/'e60_seed1_training.json')
    completion = read(REF/'e60_seed1_completion.json')
    comparison = read(REF/'e60_seed1_comparison.json')
    require(training['status'] == completion['status'] == comparison['status'] == 'completed' and
            training['completed_epochs'] == completion['trained_epochs'] == 200 and
            completion['official_evaluation_completed'], 'E60 formal200/scoring incomplete')
    require(training['variant'] == comparison['variant'] == 'e60' and
            training['params']['seed'] == comparison['seed'] == 1 and
            training['params']['imgsz'] == comparison['imgsz'] == 640 and
            comparison['checkpoint_sha256'] == SPEC['sha256'], 'Wrong E60 provenance')
    checkpoint = args.project/SPEC['weights']
    require(checkpoint.is_file(), 'Missing E60 val-best: '+str(checkpoint))
    require(sha(checkpoint) == SPEC['sha256'], 'E60 checkpoint differs from completed val-selected best.pt')
    reference = read(REF/'val_smoke.json')
    require(reference['checkpoint_sha256'] == SPEC['sha256'] and
            [x['id'] for x in reference['images']] == list(range(1,9)), 'Wrong val reproduction reference')
    records, meta = base.audit_dataset(args.dataset)
    protocol = base.protocol(args.device)
    cached = cached_references(protocol)
    env = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0)
               if torch.cuda.is_available() else None)
    binding = dict(revision=REVISION, seed=1, package=package['sha256_normalized'], source=source,
                   checkpoint_sha256=SPEC['sha256'], val_smoke_sha256=sha(REF/'val_smoke.json'),
                   protocol=protocol, dataset_annotation_sha256=base.TEST_ANN_SHA,
                   dataset_image_sha256=base.TEST_IMAGE_SHA, image_count=1610,
                   environment=env, allocator_cap_GiB=6)
    print('E60 frozen val-best / completed200 / original1610 test-dev / cached E0+E1+E40+E58 PASSED',flush=True)
    return records, meta, {KEY:str(checkpoint.resolve())}, binding, cached


def gate_path(args, mode):
    return args.project/'comparison_reports/testdev_e60_seed1_gates'/(mode+'.json')


def save_gate(args, mode, binding, out):
    write(gate_path(args,mode),dict(status='passed',signature=digest(binding),report=str(out),finished_at=now()))


def verify_gates(args, binding):
    for mode in ('check','smoke'):
        path = gate_path(args,mode)
        require(path.is_file(),'Run '+mode+' first: '+str(path))
        receipt = read(path)
        require(receipt['status'] == 'passed' and receipt['signature'] == digest(binding),
                'Stale check/smoke: code, weights, data or environment changed')
    print('E60 CHECK AND REAL 8-IMAGE SMOKE RECEIPTS VALID' if binding.get('revision') == REVISION
          else 'LOCAL SYNTHETIC gates valid (NOT server evidence)',flush=True)


def smoke(args, records, weights, binding, out):
    val, meta = base.load_dataset(args.val_dataset,548)
    reference = read(REF/'val_smoke.json')
    require(meta['annotation_dimension_sha256'] == reference['annotation_dimension_sha256'],
            'Wrong original val annotations/dimensions')
    for record,item in zip(val[:8],reference['images']):
        require((record.image_id,record.stem,record.width,record.height) ==
                (item['id'],Path(item['file_name']).stem,item['width'],item['height']), 'Val image mapping differs')
    actual,_ = base.predict(args,KEY,weights[KEY],val[:8],out/'val_reproduction')
    reproduction = base.compare_predictions(actual,reference['predictions'])
    base.predict(args,KEY,weights[KEY],records[:8],out/'testdev8')
    metrics,audit = base.score(records[:8],out/'testdev8')
    write(out/'smoke.json',dict(val_reproduction=reproduction,testdev8_official=metrics,ignore_audit=audit,
                              boundary='8-image engineering only; not full accuracy'))
    save_gate(args,'smoke',binding,out)
    print('E60 ORIGINAL VAL REPRODUCTION AND TEST-DEV8 CUDA SMOKE PASSED',flush=True)


def full(args, records, meta, weights, binding, cached, out):
    verify_gates(args,binding)
    folder = out/KEY
    _,prediction = base.predict(args,KEY,weights[KEY],records,folder)
    print(f'{now()} E60 OFFICIAL SCORING (CPU, all10 classes)',flush=True)
    metrics,audit = base.score(records,folder)
    require(sha(weights[KEY]) == SPEC['sha256'],'Read-only evaluation changed checkpoint')
    result = dict(status='completed',official=metrics,ignore_audit=audit,binding=binding,
                  checkpoint=str(weights[KEY]),checkpoint_sha256=SPEC['sha256'],
                  prediction_sha256=prediction['sha256'],finished_at=now())
    write(folder/'official_metrics.json',result)
    prediction['binding'] = binding
    write(folder/'prediction_manifest.json',prediction)
    models = dict(cached,**{KEY:result})
    deltas = {key:{m:metrics[m]-ref['official'][m] for m in KEYS} for key,ref in cached.items()}
    summary = dict(status='completed',revision=REVISION,seed=1,
                   split='test-dev' if binding.get('revision') == REVISION else 'SYNTHETIC_ONLY',
                   dataset=meta,models=models,deltas=deltas,binding=binding,finished_at=now(),
                   boundary='Fixed val-best; no training/test-set epoch or threshold selection. '
                            'Single seed, not stability proof. Overall AP is not AP-small.')
    write(out/'summary.json',summary)
    table = '|Model seed1|Official AP|AP50|AP75|AR500|\n|---|---:|---:|---:|---:|\n'
    for key,item in models.items():
        table += '|'+key+'|'+'|'.join(f'{item["official"][m]:.4f}' for m in KEYS)+'|\n'
    detail = '\n'.join(f'- E60 minus {key}: AP {v[KEYS[0]]:+.4f} pp; '
                       f'AP50 {v[KEYS[1]]:+.4f}; AP75 {v[KEYS[2]]:+.4f}.' for key,v in deltas.items())
    (out/'REPORT.md').write_text('# E60 seed1 official VisDrone test-dev\n\n'+table+'\n'+detail+
                               '\n\n'+summary['boundary']+'\n',encoding='utf-8')
    write(out/'COMPLETED.json',dict(status='completed',finished_at=now(),summary_sha256=sha(out/'summary.json')))
    print(f'{now()} ALL COMPLETED REPORT={out/"REPORT.md"}\n'+detail,flush=True)


@contextmanager
def exclusive(args):
    import fcntl
    path = args.project/'logs/testdev_e60_seed1.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another E60 seed1 test-dev controller is active')
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('check','smoke','verify-gates','run','all'))
    parser.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    parser.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    parser.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev'))
    parser.add_argument('--val-dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    parser.add_argument('--device',default='0',choices=('0',))
    args = parser.parse_args()
    args.project = args.project.resolve(); args.source = args.source.resolve()
    with exclusive(args):
        out = args.project/'comparison_reports'/('TESTDEV_E60S1_'+args.mode.upper().replace('-','_')+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        out.mkdir(parents=True,exist_ok=False)
        write(args.project/'logs/latest_testdev_e60_seed1.json',dict(report_dir=str(out),started_at=now()))
        print(f'{now()} {args.mode.upper()} START REPORT={out}; NO TRAINING; SHUTDOWN DISABLED',flush=True)
        try:
            torch = prepare(args)
            records,meta,weights,binding,cached = preflight(args,torch)
            write(out/'input_manifest.json',dict(binding=binding,weights=weights,dataset=meta))
            modes = ('check','smoke','run') if args.mode == 'all' else (args.mode,)
            for mode in modes:
                print(f'{now()} STAGE {mode} START',flush=True)
                if mode == 'check':
                    from ultralytics import YOLO
                    result = base.units()
                    model = YOLO(weights[KEY],task='detect')
                    result[KEY] = model_audit(model,KEY)
                    del model
                    gc.collect()
                    write(out/'checks.json',result)
                    save_gate(args,'check',binding,out)
                    print('E60 PACKAGE / COORDINATES / OFFICIAL SCORER / CHECKPOINT CHECK PASSED',flush=True)
                elif mode == 'verify-gates':
                    verify_gates(args,binding)
                else:
                    base.cuda_limit(args,torch)
                    if mode == 'smoke':
                        path = gate_path(args,'check')
                        require(path.is_file() and read(path)['status'] == 'passed' and
                                read(path)['signature'] == digest(binding),'Run matching check before smoke')
                        smoke(args,records,weights,binding,out)
                    else:
                        full(args,records,meta,weights,binding,cached,out)
                print(f'{now()} STAGE {mode} FINISHED',flush=True)
            print(f'{now()} FINISHED; automatic shutdown DISABLED',flush=True)
        except BaseException:
            write(out/'FAILED.json',dict(status='failed',at=now(),traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
