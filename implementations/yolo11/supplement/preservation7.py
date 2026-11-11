"""Torch-free completion seal. Used independently by controller and guard."""
from common7 import *

def result_paths(project,report):
    project,report=Path(project),Path(report)
    paths=[report/'summary.json',report/'REPORT.md',report/'per_seed.csv',report/'paired_differences.csv',
           report/'admission.json',report/'execution_plan.json',report/'input_files.json']
    for key,job in JOBS.items():
        root=report/'jobs'/key
        paths += [root/'train'/n for n in ['training_completion.json','training.json','transfer.json','actual_experiment_source.zip','actual_ultralytics_source.zip']]
        paths += [root/phase/n for phase in ['check','smoke'] for n in [phase+'.json',phase+'_receipt.json']]
        paths += [root/split/n for split in ['val','test-dev'] for n in ['official_metrics.json','prediction_manifest.json','area_predictions.json','ignore_audit.json']]
        run=project/'runs'/job['run']
        paths += [run/n for n in ['results.csv','args.yaml','weights/best.pt','weights/last.pt']]
    return paths

def seal_results(project,report,proof=None):
    if proof is None:completed_inventory(project,report)
    files=[]
    for p in result_paths(project,report):
        require(p.is_file(),'Required preserved result missing: '+str(p))
        files.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(status='complete_and_verified',at=now(),formal_runs=7,official_scores=14,
                package_sha256=package_audit(),files=files,inventory_sha256=canonical(files))
    write(Path(report)/'COMPLETION_SEAL.json',result)
    return result

def verify_seal(project,report,expected_sha):
    report=Path(report);p=report/'COMPLETION_SEAL.json'
    require(sha(p)==expected_sha,'Completion seal changed')
    seal=read(p)
    require(seal['status']=='complete_and_verified' and seal['formal_runs']==7 and seal['official_scores']==14,'Incomplete seal')
    require(seal['inventory_sha256']==canonical(seal['files']) and seal['package_sha256']==package_audit(),'Wrong preservation/package identity')
    expected={str(p.resolve()) for p in result_paths(project,report)}
    require({v['path'] for v in seal['files']}==expected and len(seal['files'])==len(expected),'Seal omits required files')
    for value in seal['files']:
        p=Path(value['path']);require(p.is_file() and p.stat().st_size==value['bytes'] and sha(p)==value['sha256'],'Preserved result missing/changed: '+str(p))
    completed_inventory(project,report)
    return seal
