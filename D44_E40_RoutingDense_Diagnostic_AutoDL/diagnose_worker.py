#!/usr/bin/env python3
"""One process per checkpoint/module namespace. Eval only; no source/checkpoint edits."""
from __future__ import annotations
import argparse
import contextlib
from datetime import datetime
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SPECS = {
    'e40': ('e40', 'E40_RUN_20260929_205658_507620', 'e40a_spatial_task_head_img640_seed1_200e', 1),
    'e40s2': ('e40', 'E40S2_RUN_20260930_130808_314014', 'e40a_spatial_task_head_img640_seed2_200e', 2),
    'e42': ('e42', 'E42_RUN_20261001_044531_789674', 'e42a_joint_spatial_channel_task_head_img640_seed1_200e', 1),
    'e43': ('e43', 'E43_RUN_20261001_044532_044824', 'e43a_dense_depth_task_head_img640_seed1_200e', 1),
}
KEYS = ('AP50_95_percent','AP50_percent','AP75_percent','AR500_percent')
BOUNDARY = 'Frozen-checkpoint, off-manifold inference intervention; not training-time causal proof, not a new trained model.'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path, normalize=False):
    h = hashlib.sha256()
    if normalize:
        h.update(Path(path).read_bytes().replace(b'\r\n',b'\n'))
    else:
        with Path(path).open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def package_signature():
    manifest = read(HERE/'manifest.json')
    for name, expected in manifest.items():
        if sha(HERE/name, True) != expected: raise RuntimeError('Package hash mismatch: '+name)
    return manifest


def setup(a):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    os.environ['YOLO_CONFIG_DIR']=str(a.project/'comparison_reports/d44_settings')
    sys.path.insert(0,str(a.source.resolve()))
    sys.path.insert(0,str(HERE/'models'/SPECS[a.model][0]))
    import torch
    import ultralytics
    import spatial_task_head
    if ultralytics.__version__ != '8.3.0' or not Path(ultralytics.__file__).resolve().is_relative_to(a.source.resolve()):
        raise RuntimeError('Requires the project Ultralytics 8.3.0; specify --source')
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(SPECS[a.model][3], deterministic=True)
    torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    device = a.device
    if device != 'cpu':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
        device = 'cuda:'+device if device.isdigit() else device
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info()
        cap = int(a.memory_cap_gib*2**30)
        if cap >= total or free < cap+1024**3:
            raise RuntimeError(f'Need cap={a.memory_cap_gib} GiB + 1 GiB headroom; free={free/2**30:.2f} GiB. Do not kill other jobs.')
        torch.cuda.set_per_process_memory_fraction(cap/total)
    a.device = device
    return torch


def sources(a):
    _, report_name, run, seed = SPECS[a.model]
    report = a.project/'comparison_reports'/report_name
    comparison = read(report/'comparison.json')
    training = read(report/'training.json')
    weights = a.project/'runs'/run/'weights/best.pt'
    if comparison.get('status')!='completed' or comparison.get('seed')!=seed or comparison.get('imgsz')!=640:
        raise RuntimeError('Wrong formal comparison report: '+str(report))
    if training.get('status')!='completed' or training.get('completed_epochs')!=200:
        raise RuntimeError('Wrong formal training report: '+str(report))
    if sha(weights)!=comparison['checkpoint_sha256']: raise RuntimeError('Checkpoint hash mismatch: '+str(weights))
    baseline = read(a.project/'comparison_reports/e1_yolo11s_p2add_img640_seed1_20260926_102750_718906/metrics.json')
    source_audit = {}
    for name, expected in baseline['p2_architecture']['source_code_sha256'].items():
        if name.startswith('nn/'):
            actual=sha(a.source/'ultralytics'/name,True)
            source_audit[name]=dict(expected=expected,actual=actual,matches=actual==expected)
            if name!='nn/tasks.py' and actual!=expected:
                raise RuntimeError('Original layer implementation drift: '+name)
    return weights, report, comparison, source_audit


def make_predictor(a, model, records, observer, options):
    """Run backbone/neck ONCE per batch, re-run only head for two interventions."""
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from interventions import MODES, intervene, state_digest
    from routing_stats import RoutingStats
    by_path={str(r.image_path.resolve()):r for r in records}

    class Predictor(DetectionPredictor):
        def setup_model(self, model, verbose=False):
            super().setup_model(model,verbose=verbose)
            if any(module.training for module in self.model.model.modules()):
                raise RuntimeError('Diagnostic model must be entirely in eval mode')
            self.head=self.model.model.model[-1]
            if (self.head.nl,self.head.reg_max,list(self.head.f))!=(4,16,[25,16,19,22]):
                raise RuntimeError('Wrong saved four-level Detect topology')
            if not getattr(self.head,'enabled',False): raise RuntimeError('Task head is disabled in saved model')
            self.capture=None
            def cache(module, inputs):
                value=inputs[0]
                if len(value)!=4: raise RuntimeError('Wrong cached head feature count')
                self.capture=tuple(t.detach().clone() for t in value)
            self.handle=self.head.register_forward_pre_hook(cache)
            self.rows={mode:[] for mode in MODES[a.model]}
            self.d44_seen={mode:set() for mode in MODES[a.model]}
            self.stats=RoutingStats(a.model)
            self.state_before=state_digest(self.model.model)
            self.first=True

        def inference(self, im, *args, **kwargs):
            if im.dtype!=torch.float32: raise RuntimeError('Diagnostic must use FP32 inference')
            context=[by_path[str(Path(p).resolve())] for p in self.batch[0]]
            self.stats.set_context(context, im.shape[2:])
            self.capture=None
            with intervene(self.head,a.model,'normal', self.stats if observer else None):
                normal=super().inference(im,*args,**kwargs)
            if self.capture is None: raise RuntimeError('Head input cache not captured')
            features=self.capture
            if any(t.dtype!=torch.float32 for t in features): raise RuntimeError('Cached features not FP32')
            results={'normal':normal}
            self.handle.remove()  # direct head replay must not overwrite its own input cache
            try:
                with torch.inference_mode():
                    if self.first:
                        replay=self.head(list(features))
                        torch.testing.assert_close(replay[0],normal[0],rtol=0,atol=1e-5)
                        for x,y in zip(replay[1],normal[1]): torch.testing.assert_close(x,y,rtol=0,atol=0)
                    for mode in MODES[a.model][1:]:
                        # Current heads are nonmutating; guards also verify this in the first batch.
                        inputs=[x.clone() for x in features]
                        with intervene(self.head,a.model,mode):
                            value=self.head(inputs)
                        if self.first:
                            for x,y in zip(inputs,features): torch.testing.assert_close(x,y,rtol=0,atol=0)
                        results[mode]=value
                    if self.first:
                        restored=self.head(list(features))
                        torch.testing.assert_close(restored[0],normal[0],rtol=0,atol=1e-5)
                        self.first=False
            finally:
                def cache(module, inputs): self.capture=tuple(t.detach().clone() for t in inputs[0])
                self.handle=self.head.register_forward_pre_hook(cache)
            self.pending=results
            return normal

        def postprocess(self,preds,img,orig_imgs):
            answer=None
            for mode in MODES[a.model]:
                value=self.pending[mode]
                if (not isinstance(value,(tuple,list)) or len(value)!=2 or len(value[1])!=4 or
                        value[0].ndim!=3 or value[0].shape[1]!=14 or not torch.isfinite(value[0]).all()):
                    raise RuntimeError('Invalid detector output '+mode)
                # Ultralytics NMS performs in-place xywh conversion; NEVER pass the cached decoded tensor.
                decoded=value[0].detach().clone()
                guard=value[0].detach().clone() if len(self.d44_seen[mode])==0 else None
                result=super().postprocess((decoded,value[1]),img,orig_imgs)
                if guard is not None: torch.testing.assert_close(value[0],guard,rtol=0,atol=0)
                for item in result:
                    rec=by_path[str(Path(item.path).resolve())]
                    if rec.image_id in self.d44_seen[mode]: raise RuntimeError('Repeated image '+mode)
                    self.d44_seen[mode].add(rec.image_id)
                    for x1,y1,x2,y2,score,cls in item.boxes.data.cpu().tolist():
                        self.rows[mode].append(dict(image_id=rec.image_id,category_id=int(cls)+1,
                            bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)],score=score))
                if mode=='normal': answer=result
            self.pending=None; self.capture=None
            return answer

        def finish(self):
            self.handle.remove()
            expected={r.image_id for r in records}
            for mode,seen in self.d44_seen.items():
                if seen!=expected: raise RuntimeError('Missing images '+mode)
            if state_digest(self.model.model)!=self.state_before:
                raise RuntimeError('Evaluation/interventions changed registered weights/buffers')
            if any(p.grad is not None for p in self.model.model.parameters()):
                raise RuntimeError('Diagnostic unexpectedly generated gradients')
            return self.rows,self.stats

    p=Predictor(overrides={**model.overrides,**options,'mode':'predict'},_callbacks=model.callbacks)
    return p


def fixture_test(a, weights):
    """Actual YOLO/AutoBackend/NMS/cache integration on in-memory synthetic images."""
    import numpy as np
    from ultralytics import YOLO
    import d27_official as official
    import d38_common as common
    images=[np.random.default_rng(710+i).integers(0,256,(h,w,3),dtype=np.uint8)
            for i,(h,w) in enumerate(((97,131),(85,127)))]
    records=[]
    for i,image in enumerate(images):
        h,w=image.shape[:2]
        gt=np.array([[15,20,20,18,0,1]],dtype=np.float64)
        records.append(official.ImageRecord(i+1,f'image{i}',Path(f'image{i}.jpg'),Path('unused.txt'),
            w,h,np.zeros((0,8)),gt,np.zeros((0,4)),np.zeros((0,4))))
    model=YOLO(str(weights),task='detect')
    opts={**common.prediction_options(a.device),'imgsz':128}
    p=make_predictor(a,model,records,True,opts)
    model.predict(images,batch=2,predictor=p,stream=False,**opts)
    rows,stats=p.finish()
    if len(stats.rows)!=4: raise RuntimeError('GT-center statistics integration failed')
    del p,model
    return dict(actual_predictor=True,cached_head_replay=True,nms_clone=True,
                synthetic_images=2,synthetic_size=128,statistics=True,
                boundary='Synthetic integration only; not real VisDrone or CUDA smoke receipt')


def signature(a, comparison, dataset):
    import torch
    import d38_common as common
    names=('nn/tasks.py','nn/modules/head.py','nn/modules/block.py','nn/modules/conv.py',
           'utils/ops.py','engine/predictor.py','models/yolo/detect/predict.py','nn/autobackend.py')
    gpu=None
    if a.device!='cpu':
        p=torch.cuda.get_device_properties(a.device)
        gpu=dict(name=p.name,total_memory=p.total_memory,capability=list(torch.cuda.get_device_capability(a.device)))
    return dict(model=a.model,package=package_signature(),weights=comparison['checkpoint_sha256'],
                saved_predictions=sha(a.project/'comparison_reports'/SPECS[a.model][1]/'area_predictions.json'),
                dataset=dataset['annotation_dimension_sha256'],
                source={name:sha(a.source/'ultralytics'/name,True) for name in names},
                torch=torch.__version__,cuda=torch.version.cuda,gpu=gpu,device=a.device,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                protocol=common.prediction_options(a.device),batch=4,memory_cap_gib=a.memory_cap_gib)


def run(a, out):
    torch=setup(a)
    from interventions import self_test, MODES
    import d27_official as official
    import d38_common as common
    from ultralytics import YOLO
    import numpy as np
    weights,report,comparison,source_audit=sources(a)
    checks=self_test(a.model,a.device)
    with contextlib.redirect_stdout(io.StringIO()):
        official_tests=official.run_unit_tests()
    common.tests()
    fixture=fixture_test(a,weights)
    checks['fixture']=fixture
    checks['scope']='Synthetic tests + real saved checkpoint; no training'
    write(out/'check.json',dict(status='passed',checks=checks,official=official_tests,source=source_audit))
    if a.mode=='check':
        print(f'D44 {a.model} CHECK PASSED device={a.device}',flush=True); return
    if a.device=='cpu': raise RuntimeError('Real smoke/full run require CUDA; CPU check is local engineering only')
    records,dataset=official.load_dataset(a.dataset)
    if dataset['annotation_dimension_sha256']!=comparison['dataset']['annotation_dimension_sha256']:
        raise RuntimeError('Dataset annotation/dimension fingerprint differs from formal report')
    # Verify original ordering/identity via the corresponding E1 archive, not filename guesses.
    base_report = ('e1_yolo11s_p2add_img640_seed1_20260926_102750_718906' if SPECS[a.model][3]==1
                   else 'e1_yolo11s_p2add_img640_seed2_20260926_172907_911324')
    images=read(a.project/'comparison_reports'/base_report/'area_ground_truth.json')['images']
    if len(images)!=len(records) or any((r.image_id,r.stem,r.width,r.height)!=
            (im['id'],Path(im['file_name']).stem,im['width'],im['height']) for r,im in zip(records,images)):
        raise RuntimeError('Original image ID mapping differs')
    sig=signature(a,comparison,dataset)
    gate=a.project/'comparison_reports'/f'd44_{a.model}_smoke_gate.json'
    if a.mode=='run':
        receipt=read(gate) if gate.is_file() else {}
        if receipt.get('signature')!=sig: raise RuntimeError('Missing/stale real smoke gate; run smoke first')
        evidence=read(Path(receipt['report'])/'metrics.json')
        if (evidence.get('status')!='completed' or not evidence.get('subset_only') or
                evidence.get('image_count')!=8 or not evidence.get('default_saved_prediction_reproduction') or
                (Path(receipt['report'])/'failure.json').exists()):
            raise RuntimeError('Invalid real smoke evidence')
    selected=records[:8] if a.mode=='smoke' else records
    started=time.perf_counter()
    metrics=dict(status='running',model=a.model,zero_training=True,subset_only=a.mode=='smoke',
                 image_count=len(selected),signature=sig,boundary=BOUNDARY,
                 source_report=str(report),checkpoint=str(weights),started_at=datetime.now().astimezone().isoformat())
    write(out/'metrics.json',metrics)
    opts=common.prediction_options(a.device)
    model=YOLO(str(weights),task='detect')
    p=make_predictor(a,model,selected,True,opts)
    for start in range(0,len(selected),4):
        chunk=selected[start:start+4]
        model.predict([str(r.image_path) for r in chunk],batch=len(chunk),predictor=p,stream=False,**opts)
        done=start+len(chunk); elapsed=time.perf_counter()-started
        if start==0 or done%40==0 or done==len(selected):
            eta=elapsed/done*(len(selected)-done)
            print(f'D44 {a.model} IMAGES {done}/{len(selected)} all 3 head modes '
                  f'elapsed={elapsed/60:.1f}min inference_remaining~{eta/60:.1f}min '
                  f'reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB',flush=True)
    predictions,stats=p.finish()
    for mode,rows in predictions.items(): write(out/(mode+'_predictions.json'),rows)
    write(out/'routing_rows.json',stats.rows); write(out/'routing_summary.json',stats.summary())
    del model,p
    gc.collect(); torch.cuda.empty_cache()
    saved=official.load_predictions(report/'area_predictions.json',len(records))
    normal=official.load_predictions(out/'normal_predictions.json',len(records))
    for rec in selected:
        np.testing.assert_allclose(normal[rec.image_id-1][:,:6],saved[rec.image_id-1][:,:6],rtol=0,atol=1e-5,
                                   err_msg=f'Default prediction reproduction failed: {a.model}/{rec.stem}')
    metrics['default_saved_prediction_reproduction']=True
    if a.mode=='run':
        scores={}
        for mode in MODES[a.model]:
            print(f'D44 {a.model} OFFICIAL SCORING {mode} START (CPU; GPU released)',flush=True)
            raw=normal if mode=='normal' else official.load_predictions(out/(mode+'_predictions.json'),len(records))
            filtered,ignore=official.filter_detections_and_audit(records,raw)
            score=official.official_evaluate(records,filtered)
            if mode=='normal':
                difference={k:score[k]-comparison['candidate'][k] for k in KEYS}
                if max(abs(v) for v in difference.values())>.002:
                    raise RuntimeError('Default official AP reproduction failed: '+str(difference))
                metrics['official_reproduction_delta_pp']=difference
            delta={k:score[k]-scores['normal']['official'][k] for k in KEYS} if mode!='normal' else {k:0. for k in KEYS}
            scores[mode]=dict(official=score,delta_vs_same_checkpoint_normal_pp=delta,ignore=ignore)
            write(out/(mode+'_official.json'),scores[mode])
            print(f'D44 {a.model} OFFICIAL {mode} AP={score[KEYS[0]]:.6f} delta={delta}',flush=True)
            del raw,filtered; gc.collect()
        metrics['scores']=scores
    else:
        metrics['scores_not_computed']='8-image smoke is functional only, not performance screening'
    if sha(weights)!=comparison['checkpoint_sha256']: raise RuntimeError('Checkpoint changed during diagnostic')
    metrics.update(status='completed',finished_at=datetime.now().astimezone().isoformat(),
                   elapsed_seconds=time.perf_counter()-started,weights_unchanged=True,
                   no_gradients=True,cached_head_replay_verified=True)
    write(out/'metrics.json',metrics)
    if a.mode=='smoke': write(gate,dict(signature=sig,report=str(out)))
    print(f'D44 {a.model} {a.mode.upper()} COMPLETED REPORT={out}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('check','smoke','run'),required=True)
    p.add_argument('--model',choices=tuple(SPECS),required=True)
    p.add_argument('--project',type=Path,default=Path('/root/autodl-tmp/yolo11_project'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/yolo11_src'))
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-val'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--memory-cap-gib',type=float,default=6.)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.project=a.project.resolve(); a.source=a.source.resolve()
    if not 2<=a.memory_cap_gib<=12: raise ValueError('Memory cap must be between 2 and 12 GiB')
    a.out.mkdir(parents=True,exist_ok=False)
    print(f'D44 {a.model} {a.mode} START {datetime.now().astimezone().isoformat()} REPORT={a.out}',flush=True)
    try:
        package_signature()
        run(a,a.out)
    except BaseException:
        write(a.out/'failure.json',dict(status='failed',error=traceback.format_exc(),zero_training=True))
        raise


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    sys.stderr.reconfigure(encoding='utf-8',errors='replace')
    main()
