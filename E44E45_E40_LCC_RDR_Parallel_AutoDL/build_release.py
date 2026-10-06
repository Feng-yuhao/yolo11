"""Mechanical adaptation of the validated v4 E43 harness; old sources untouched."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'E42E43_E40_JointDense_Parallel_AutoDL'
TEMPLATE=SOURCE/'e43'
RUNS={'e44':'e44a_local_channel_calibration_img640_seed1_200e',
      'e45':'e45a_residual_depth_reuse_img640_seed1_200e'}


def change(text,old,new):
    if old not in text: raise RuntimeError('Template drift: '+old[:100])
    return text.replace(old,new)


def build():
    for name,variant,total,group_count in (('e44','local',182352,26),('e45','reuse',198480,18)):
        dest=ROOT/name; dest.mkdir(exist_ok=True)
        for file in ('e1_640_core.py','e1_reference.yaml','p2_tal_chunked.py','d27_official.py',
                     'd38_common.py','LICENSE-Ultralytics.txt','smoke_memory.py','smoke_progress.py','parent_reference.py'):
            (dest/file).write_bytes((TEMPLATE/file).read_bytes())
        for file in ('extension.py','precision_audit.py'):
            (dest/file).write_bytes((ROOT/file).read_bytes())
        for file in ('validation_smoke.py','amp_branch_audit.py'):
            text=(SOURCE/file).read_text(encoding='utf-8')
            text=text.replace('E42E43_SMOKE_BARRIER','E44E45_SMOKE_BARRIER')
            text=text.replace("('e42', 'e43')","('e44', 'e45')")
            text=text.replace("barrier('e42' if VARIANT == 'joint' else 'e43')","barrier('e44' if VARIANT == 'local' else 'e45')")
            if file=='amp_branch_audit.py':
                begin=text.index('                # Double zero-init')
                end=text.index('            changed =',begin)
                text=text[:begin]+text[end:]
            else:
                text=change(text,'return dict(validation_batch=32,',
                            "return dict(parallel_pair=bool(os.environ.get('E44E45_SMOKE_BARRIER')),barrier_directory=os.environ.get('E44E45_SMOKE_BARRIER'),validation_batch=32,")
            (dest/file).write_text(text,encoding='utf-8')
        module=(TEMPLATE/'spatial_task_head.py').read_text(encoding='utf-8')
        module=change(module,'NEW_PARAMETERS = 198864',f'NEW_PARAMETERS = {total}')
        module=change(module,"VARIANT = 'dense'",f"VARIANT = '{variant}'")
        (dest/'spatial_task_head.py').write_text(module,encoding='utf-8')
        checks=(TEMPLATE/'checks.py').read_text(encoding='utf-8')
        begin=checks.index("    if VARIANT == 'joint':"); end=checks.index("    if args.device!='cpu':",begin)
        checks=checks[:begin]+"    from precision_audit import precision_audit\n    unit['new_branch_precision']=precision_audit(args.device,VARIANT)\n    print('CHECK new branch weak gradients match FP32 oracle',flush=True)\n"+checks[end:]
        checks=change(checks,"                    if step==0 and 'channel_router.1.' in group:\n                        require(all(p.grad.count_nonzero()==0 for p in selected),'Channel front should wait for nonzero final weights')",'')
        checks=change(checks,'    unit = routing_unit(args.device)',
                      '    with strict_math():\n        unit = routing_unit(args.device)')
        checks=change(checks,"    unit['new_branch_precision']=precision_audit(args.device,VARIANT)",
                      "    with strict_math():\n        unit['new_branch_precision']=precision_audit(args.device,VARIANT)")
        checks=change(checks,'            else: torch.testing.assert_close(q.grad,p.grad,rtol=0,atol=0)',
                      "            else:\n                from extension import compare_gradients\n                compare_gradients(q.grad,p.grad,'base.'+n,exact=args.device=='cpu')")
        checks=checks.replace('all 18 branch groups updated',f'all {group_count} branch groups updated')
        checks=checks.replace('main loss/base gradients exact at start;', 'main loss exact / base gradients numerically equal at start;')
        checks=change(checks,"    flops=get_flops(fresh,imgsz=640); base_flops=get_flops(base,imgsz=640)",
                      "    flops=get_flops(fresh,imgsz=640); base_flops=get_flops(base,imgsz=640)\n    from spatial_task_head import SpatialTaskBlock\n    parent=copy.deepcopy(fresh)\n    for block in branch(parent):\n        block.__class__=SpatialTaskBlock\n        for key in ('calibrators','reuse2','reuse3'):\n            if hasattr(block,key): delattr(block,key)\n    parent_flops=get_flops(parent,imgsz=640)\n    expected_extra=.21504 if VARIANT=='local' else .73728\n    require(abs(flops-parent_flops-expected_extra)<1e-5,'New convs missing from profiler '+str((flops,parent_flops,expected_extra)))")
        checks=change(checks,'GFLOPs640=flops,E1_GFLOPs640=base_flops)',
                      'GFLOPs640=flops,E1_GFLOPs640=base_flops,E40_GFLOPs640=parent_flops,added_conv_GFLOPs640=flops-parent_flops)')
        (dest/'checks.py').write_text(checks.replace('E43',name.upper()),encoding='utf-8')
        runner=(TEMPLATE/'run_e40.py').read_text(encoding='utf-8')
        runner=change(runner,"EXPERIMENT = 'e43a_dense_depth_task_head_img640_seed1_200e'",f"EXPERIMENT = '{RUNS[name]}'")
        runner=change(runner,"len(result.get('updated_groups',[]))!=18",f"len(result.get('updated_groups',[]))!={group_count}")
        runner=change(runner,"report.update(status='completed',finished_at=datetime.now().isoformat(),", "report.update(status='evaluating',training_finished_at=datetime.now().isoformat(),")
        runner=change(runner,"    evaluate(args,best,out)\n", "    evaluate(args,best,out)\n    report.update(status='completed',finished_at=datetime.now().isoformat())\n    write(out/'training.json',{k:v for k,v in report.items() if k!='_clock'})\n    write(out/'completion.json',dict(status='completed',trained_epochs=200,official_evaluation_completed=True,finished_at=report['finished_at']))\n")
        runner=runner.replace('e43_',name+'_').replace('E43',name.upper())
        runner=runner.replace('leave the running E39 untouched','do not run CUDA check alongside other GPU jobs')
        (dest/'run_e40.py').write_text(runner,encoding='utf-8')
        evaluation=(TEMPLATE/'evaluate.py').read_text(encoding='utf-8').replace('E43',name.upper())
        evaluation=change(evaluation,"screening_passed=parent_delta['AP50_95_percent']>=.20", "screening_passed=candidate['AP50_95_percent']>=26.44")
        (dest/'evaluate.py').write_text(evaluation,encoding='utf-8')
        shell=(TEMPLATE/'run.sh').read_text(encoding='utf-8').replace('.e43_execution.lock','.'+name+'_execution.lock')
        shell=change(shell,'exec 9>"$PACKAGE/.'+name+'_execution.lock"',
                     'PROJECT="${EXPERIMENT_PROJECT:-/root/autodl-tmp/yolo11_project}"\n'
                     'ARGS=("$@")\n'
                     'for ((i=0; i<${#ARGS[@]}; i++)); do\n'
                     '  if [[ "${ARGS[i]}" == --project ]]; then PROJECT="${ARGS[i+1]:?Missing project}"; fi\n'
                     '  if [[ "${ARGS[i]}" == --project=* ]]; then PROJECT="${ARGS[i]#--project=}"; fi\n'
                     'done\n'
                     '[[ -d "$PROJECT" ]] || { echo "Missing project: $PROJECT" >&2; exit 3; }\n'
                     'exec 9>"$PROJECT/.'+name+'_execution.lock"')
        (dest/'run.sh').write_text(shell,encoding='utf-8')
    controller=(SOURCE/'run_pair.py').read_text(encoding='utf-8')
    controller=controller.replace('e42','e44').replace('e43','e45').replace('E42E43','E44E45')
    for old,new in (('e44a_joint_spatial_channel_task_head_img640_seed1_200e',RUNS['e44']),
                    ('e45a_dense_depth_task_head_img640_seed1_200e',RUNS['e45'])):
        controller=change(controller,old,new)
    controller=change(controller,"    say('Logs: '+str(logs))", "    say('Logs: '+str(logs))\n    write(args.project/'logs'/('e44e45_latest_'+mode+'.json'),dict(directory=str(logs),logs={n:str(logs/(n+'_'+mode+'.log')) for n in NAMES}))")
    controller=change(controller,"    idle()\n    if args.mode=='serial':", "    if args.mode=='run':\n        from pair_receipts import require_parallel_smoke\n        require_parallel_smoke(args.project)\n    idle()\n    if args.mode=='serial':")
    (ROOT/'run_pair.py').write_text(controller,encoding='utf-8')
    (ROOT/'run_pair.sh').write_bytes((SOURCE/'run_pair.sh').read_bytes())


def release():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.yaml','.md','.txt'))
    sha=lambda p: hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    manifests=[ROOT/'manifest.json']
    for name in RUNS:
        path=ROOT/name/'manifest.json'
        path.write_text(json.dumps({p.name:sha(p) for p in files if p.parent==ROOT/name},indent=2)+'\n',encoding='utf-8')
        manifests.append(path)
    release_name=ROOT.name if ROOT.name.endswith('_v2') else ROOT.name+'_v2'
    archive=ROOT.parent/(release_name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files+manifests:
            z.writestr(release_name+'/'+p.relative_to(ROOT).as_posix(),p.read_bytes().replace(b'\r\n',b'\n'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,expected in manifest.items():
            assert hashlib.sha256(z.read(release_name+'/'+name)).hexdigest()==expected,name
    print('READY',archive,archive.stat().st_size,'bytes',flush=True)


if __name__=='__main__':
    import sys
    if '--release-only' not in sys.argv: build()
    if '--build-only' not in sys.argv: release()
