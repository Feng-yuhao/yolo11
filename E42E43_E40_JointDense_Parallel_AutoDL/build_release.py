"""Bulk mechanical adaptation of validated E40 harness; never touches old packages."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'E40a_SpatialTaskHead_AutoDL'


def replace(text, old, new):
    if old not in text:
        raise RuntimeError('Template no longer matches: ' + old[:90])
    return text.replace(old, new)


def build():
    common = ('e1_640_core.py','e1_reference.yaml','p2_tal_chunked.py','d27_official.py','d38_common.py','LICENSE-Ultralytics.txt')
    for name, variant, total, experiment in (
        ('e42','joint',213072,'e42a_joint_spatial_channel_task_head_img640_seed1_200e'),
        ('e43','dense',198864,'e43a_dense_depth_task_head_img640_seed1_200e')):
        target = ROOT / name
        target.mkdir(exist_ok=True)
        for f in common:
            (target/f).write_bytes((SOURCE/f).read_bytes())
        for f in ('extension.py','validation_smoke.py','smoke_progress.py','precision_audit.py','amp_branch_audit.py'):
            (target/f).write_bytes((ROOT/f).read_bytes())
        module = (SOURCE/'spatial_task_head.py').read_text(encoding='utf-8')
        module = replace(module, 'NEW_PARAMETERS = 175440', f"NEW_PARAMETERS = {total}\nVARIANT = '{variant}'")
        module = replace(module, 'head.__class__ = SpatialTaskDetect',
                         'from extension import install\n    blocks = install(blocks, VARIANT)\n    head.__class__ = SpatialTaskDetect')
        (target/'spatial_task_head.py').write_text(module,encoding='utf-8')
        checks = (SOURCE/'checks.py').read_text(encoding='utf-8')
        begin=checks.index('def routing_unit('); end=checks.index('\ndef head_route_oracle(',begin)
        checks=checks[:begin]+"def routing_unit(device):\n    from extension import routing_audit\n    from spatial_task_head import VARIANT\n    return routing_audit(device,VARIANT)\n\n"+checks[end:]
        checks=replace(checks,"groups=[f'{i}.{part}.' for i in range(2) for part in ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]",
                       "from extension import branch_groups\n        from spatial_task_head import VARIANT\n        groups=branch_groups(VARIANT)")
        checks=replace(checks,"require(any(p.grad.abs().sum()>0 for p in selected),'Zero group gradient '+group)",
                       "if not (step==0 and 'channel_router.1.' in group):\n                        require(any(p.grad.abs().sum()>0 for p in selected),'Zero group gradient '+group)\n                    else:\n                        require(all(p.grad.count_nonzero()==0 for p in selected),'Channel front should wait for nonzero final weights')")
        checks=replace(checks,'all 14 branch groups updated','all 18 branch groups updated')
        checks=replace(checks,'unit = routing_unit(args.device)',
                       "unit = routing_unit(args.device)\n    from spatial_task_head import VARIANT\n    if VARIANT == 'joint':\n        from precision_audit import channel_precision_audit\n        unit['channel_precision']=channel_precision_audit(args.device)\n        print('E42 CHECK weak channel gradients match FP32 oracle under AMP',flush=True)\n    if args.device!='cpu':\n        from amp_branch_audit import audit_detector_amp\n        unit['staged_detection_AMP']=audit_detector_amp(core,args)")
        checks=replace(checks,"optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)",
                       "# Gate probe updates the new head only to keep synthetic TAL positives stable.\n        # Check nonzero LOSS gradients independently: weight decay alone is not evidence.\n        # Formal/integration optimizer coverage is independently checked on ALL parameters.\n        optimizer=torch.optim.SGD(new.parameters(),lr=.01,momentum=.937,weight_decay=.0005)")
        checks=replace(checks,'for step in range(2):','for step in range(12):')
        checks=replace(checks,"with chunked_assignment():\n            for step", "seen_gradients=set()\n        with chunked_assignment():\n            for step")
        checks=replace(checks,"if not (step==0 and 'channel_router.1.' in group):\n                        require(any(p.grad.abs().sum()>0 for p in selected),'Zero group gradient '+group)\n                    else:\n                        require(all(p.grad.count_nonzero()==0 for p in selected),'Channel front should wait for nonzero final weights')",
                       "if any(p.grad.abs().sum()>0 for p in selected): seen_gradients.add(group)\n                    if step==0 and 'channel_router.1.' in group:\n                        require(all(p.grad.count_nonzero()==0 for p in selected),'Channel front should wait for nonzero final weights')")
        checks=replace(checks,'optimizer.step(); optimizer.zero_grad(set_to_none=True)\n            empty=',
                       "optimizer.step(); model.zero_grad(set_to_none=True)\n                if step>=1 and len(seen_gradients)==len(groups) and all(any(not torch.equal(p,before[n]) for n,p in new.named_parameters() if n.startswith(g)) for g in groups):\n                    break\n            require(set(groups)==seen_gradients,'Missing nonzero gradients: '+str(set(groups)-seen_gradients))\n            empty=")
        checks=replace(checks,'e1_yolo11s_p2add_img640_seed1/weights/best.pt','e1_yolo11s_p2add_img640_seed1/weights/best.pt')
        (target/'checks.py').write_text(checks.replace('E40',name.upper()),encoding='utf-8')
        runner=(SOURCE/'run_e40.py').read_text(encoding='utf-8')
        runner=replace(runner,"EXPERIMENT = 'e40a_spatial_task_head_img640_seed1_200e'",f"EXPERIMENT = '{experiment}'")
        runner=replace(runner,'torch.cuda.set_device(args.device)',
                       "torch.cuda.set_device(args.device)\n        if args.mode in ('smoke','run'):\n            total=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory\n            torch.cuda.set_per_process_memory_fraction(14.*2**30/total)\n            print('Fixed 14 GiB allocator cap for smoke AND formal; batch16 unchanged',flush=True)")
        runner=replace(runner,"groups=[f'{i}.{part}.' for i in range(2) for part in ('reduce','depths.0','depths.1','depths.2','router','out_cls','out_reg')]",
                       "from extension import branch_groups\n    from spatial_task_head import VARIANT\n    groups=branch_groups(VARIANT)")
        runner=replace(runner,"not any(v['updated'] for v in updates)","sum(v['updated'] for v in updates)<3")
        runner=replace(runner,"start = time.perf_counter(); updates=[]", "start = time.perf_counter(); updates=[]; seen_loss_gradients=set()")
        runner=replace(runner,"updates.append(amp_step(model,optimizer,scaler))",
                       "from extension import branch_groups\n            from spatial_task_head import VARIANT\n            nonzero={g for g in branch_groups(VARIANT) if any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for n,p in branch(model).named_parameters() if n.startswith(g))}\n            updates.append(amp_step(model,optimizer,scaler))\n            if updates[-1]['updated']: seen_loss_gradients.update(nonzero)")
        runner=replace(runner,"groups=branch_groups(VARIANT)\n    for group", "groups=branch_groups(VARIANT)\n    if set(groups)!=seen_loss_gradients: raise RuntimeError('No nonzero successful-step LOSS gradient in '+str(set(groups)-seen_loss_gradients))\n    for group")
        runner=replace(runner,"os.environ['YOLO_AUTOINSTALL'] = 'false'", "from smoke_progress import seed_existing_font\n    seed_existing_font(args.project,os.environ['YOLO_CONFIG_DIR'])\n    os.environ['YOLO_AUTOINSTALL'] = 'false'")
        runner=replace(runner,"start = time.perf_counter(); updates=[]; seen_loss_gradients=set()",
                       "from smoke_progress import SmokeProgress,stress_batches,MAX_ATTEMPTS,TARGET_UPDATES\n    from extension import branch_groups\n    from spatial_task_head import VARIANT\n    groups=branch_groups(VARIANT)\n    progress=SmokeProgress()\n    start = time.perf_counter(); updates=[]; seen_loss_gradients=set()")
        runner=replace(runner,'for i,batch in smoke_batches(loader):','for i,batch in stress_batches(loader):')
        runner=replace(runner,"if updates[-1]['updated']: seen_loss_gradients.update(nonzero)",
                       "progress.observe(updates[-1])\n            if updates[-1]['updated']: seen_loss_gradients.update(nonzero)")
        runner=replace(runner,"SMOKE batch {i+1}/8 loss=", "SMOKE attempt {i+1}/{MAX_ATTEMPTS} successful={progress.successful}/{TARGET_UPDATES} loss=")
        runner=replace(runner,"f'elapsed={time.perf_counter()-start:.1f}s',flush=True)\n    if len(updates)!=8 or sum(v['updated'] for v in updates)<3:\n        raise RuntimeError('Smoke did not perform a finite optimizer update')",
                       "f'elapsed={time.perf_counter()-start:.1f}s',flush=True)\n            all_changed=all(any(not torch.equal(p,branch_before[n]) for n,p in branch(model).named_parameters() if n.startswith(g)) for g in groups)\n            if progress.ready and set(groups)==seen_loss_gradients and all_changed:\n                break\n    progress.require_success()")
        runner=replace(runner,"# Real forward after updates, not just a synthetic/checkpoint-only test.",
                       "# Retain the real train model/optimizer while testing EMA FP16 val batch32.\n    from validation_smoke import validation_stress\n    validation=validation_stress(model,args,data,params,out)\n    # Real forward after updates, not just a synthetic/checkpoint-only test.")
        runner=replace(runner,"result=dict(status='passed',real_batches=8", "result=dict(validation=validation,status='passed',real_batches=progress.attempts,successful_optimizer_updates=progress.successful,AMP_overflow_skips=progress.skipped,max_real_batches=MAX_ATTEMPTS")
        runner=replace(runner,'other_job_reserve_GiB=6.', 'other_job_reserve_GiB=2.')
        runner=replace(runner,"len(result.get('updated_groups',[]))!=14", "len(result.get('updated_groups',[]))!=18 or result.get('validation',{}).get('validation_images')!=548 or result.get('validation',{}).get('validation_batch')!=32")
        runner=replace(runner,"get('validation_batch')!=32", "get('validation_batch')!=32 or result.get('validation',{}).get('loss_precision')!='autocast_float16' or result.get('validation',{}).get('raw_logits_and_decoded_finite') is not True")
        runner=replace(runner,"result.get('real_batches')!=8", "not (8<=result.get('real_batches',0)<=32) or result.get('successful_optimizer_updates',0)<8")
        runner=replace(runner,"'Leave E39 running and retry smoke after it releases GPU. '", "'No smoke receipt issued; do not start formal training. '")
        runner=replace(runner,"out/'training.json',{k:v for k,v in report.items() if k!='_clock'})", "out/'training.json',{k:v for k,v in report.items() if k!='_clock'})")
        runner=runner.replace('e40_settings',name+'_settings').replace('e40_cpu_',name+'_cpu_').replace('e40_cuda_',name+'_cuda_').replace('e40_smoke_',name+'_smoke_').replace('E40',name.upper())
        # Keep filename run_e40.py for tested imports; experiment names, reports,
        # settings, receipts and package directories are variant-specific.
        (target/'run_e40.py').write_text(runner,encoding='utf-8')
        evaluation=(SOURCE/'evaluate.py').read_text(encoding='utf-8')
        evaluation=replace(evaluation,"keys=common.KEYS\n    delta=", "from parent_reference import reference_e40\n    parent=reference_e40(args,dataset)\n    keys=common.KEYS\n    parent_delta={k:candidate[k]-parent[k] for k in keys}\n    delta=")
        evaluation=replace(evaluation,"seed=1,imgsz=640,checkpoint=", "parent_E40=parent,delta_vs_E40_pp=parent_delta,seed=1,imgsz=640,checkpoint=")
        evaluation=replace(evaluation,"screening_passed=delta['AP50_95_percent']>=.5 and delta['AP75_percent']>=.3 and delta['AP50_percent']>=0", "screening_passed=parent_delta['AP50_95_percent']>=.20 and parent_delta['AP75_percent']>=0 and parent_delta['AP50_percent']>=0")
        evaluation=replace(evaluation,"dataset=dataset,checkpoint_sha256=comparison['checkpoint_sha256']", "parent_E40=parent,delta_vs_E40_pp=parent_delta,dataset=dataset,checkpoint_sha256=comparison['checkpoint_sha256']")
        evaluation=replace(evaluation,"table='|Metric|E1|E40|Delta pp|", "table='|Metric|E1|"+name.upper()+"|Delta pp|")
        evaluation=replace(evaluation,"'\\nSingle-seed screening only", "'\\nDelta vs E40: '+json.dumps(parent_delta)+'\\n\\nSingle-seed screening only")
        (target/'evaluate.py').write_text(evaluation.replace('E40 formal',name.upper()+' formal').replace('E40 EVAL',name.upper()+' EVAL').replace('E40 OFFICIAL',name.upper()+' OFFICIAL'),encoding='utf-8')
        for f in ('smoke_memory.py','parent_reference.py'):
            (target/f).write_bytes((ROOT/f).read_bytes())
        shell=(ROOT/'run_one.sh').read_text(encoding='utf-8').replace('VARIANT_NAME',name)
        (target/'run.sh').write_text(shell,encoding='utf-8')


def release():
    # Pure release operation: source files must already exist and have passed checks.
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
           p.suffix in ('.py','.sh','.yaml','.md','.txt') and '.lock' not in p.name]
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    for folder in ('e42','e43'):
        sub={str(p.relative_to(ROOT/folder)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files if p.parent==ROOT/folder}
        (ROOT/folder/'manifest.json').write_text(json.dumps(sub,indent=2)+'\n',encoding='utf-8')
    archive=ROOT.parent/(ROOT.name+'_v4.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files+[ROOT/'manifest.json',ROOT/'e42/manifest.json',ROOT/'e43/manifest.json']:
            z.writestr(ROOT.name+'/'+str(p.relative_to(ROOT)).replace('\\','/'),p.read_bytes().replace(b'\r\n',b'\n'))
    with zipfile.ZipFile(archive) as z:
        for name,expected in manifest.items():
            assert hashlib.sha256(z.read(ROOT.name+'/'+name)).hexdigest()==expected,name
        assert z.testzip() is None
    print('READY',archive,archive.stat().st_size,'bytes',flush=True)


if __name__=='__main__':
    import sys
    if '--release-only' not in sys.argv:
        build()
    if '--build-only' not in sys.argv:
        release()
