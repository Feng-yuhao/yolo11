"""Mechanical freezing of actual evidence; run locally, never on server."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    reports=ROOT/'yolo11_project/comparison_reports'
    ref=HERE/'references';ref.mkdir(exist_ok=True)
    original=reports/'baseline_yolo11s_img640_seed1_20260926_073621_610152'
    shutil.copy2(original/'yolo11s_img640_seed1.py',HERE/'e0_core.py')
    shutil.copy2(original/'metrics.json',ref/'e0_seed1_training.json')
    evaluator=ROOT/'TESTDEV_E60_Seed1_AutoDL'
    for name in ('base_eval.py','d27_official.py','spatial_task_head.py','e58_spatial_task_head.py',
                 'e57_spatial_task_head.py','parent_spatial_task_head.py','LICENSE-Ultralytics.txt'):
        shutil.copy2(evaluator/name,HERE/name)
    shutil.copy2(evaluator/'run_eval.py',HERE/'e60_eval_reference.py')
    shutil.copy2(ROOT/'E60_Seed3_AutoShutdown_AutoDL/shutdown_safety.py',HERE/'shutdown_safety.py')
    for name in ('testdev_input_manifest.json','e1_val_metadata.json','seed1_testdev_summary.json'):
        shutil.copy2(evaluator/'references'/name,ref/name)
    val_template=read(evaluator/'references/val_smoke.json')
    paths={1:'E60_RUN_20261010_133029_103161',2:'E60S2_RUN_20261011_120724_723053',
           3:'E60S3_RUN_20261011_220512_228555'}
    for seed,name in paths.items():
        folder=reports/name
        for item in ('training','comparison','completion'):
            shutil.copy2(folder/(item+'.json'),ref/('e60_seed%d_%s.json'%(seed,item)))
        # Check frozen inference classes against EACH actual trained snapshot.
        with zipfile.ZipFile(folder/'actual_source.zip') as archive:
            for item in ('spatial_task_head.py','e58_spatial_task_head.py','e57_spatial_task_head.py',
                         'parent_spatial_task_head.py','d27_official.py'):
                if archive.read(item).replace(b'\r\n',b'\n')!=(HERE/item).read_bytes().replace(b'\r\n',b'\n'):
                    raise RuntimeError('Actual seed%d code differs: %s'%(seed,item))
        predictions=read(folder/'area_predictions.json')
        write(ref/('e60_seed%d_val_smoke.json'%seed),dict(
            checkpoint_sha256=read(folder/'comparison.json')['checkpoint_sha256'],
            annotation_dimension_sha256=val_template['annotation_dimension_sha256'],
            images=val_template['images'],predictions=[p for p in predictions if p['image_id']<=8]))
    cache={
        'e0_seed1':reports/'TESTDEV_E0_E58_RUN_20261008_150524_396506/e0_seed1',
        'e60_seed1':reports/'TESTDEV_E60S1_ALL_20261010_230739_867165/e60_seed1',
    }
    for seed in (1,2,3):
        cache['e1_seed%d'%seed]=reports/('TESTDEV_E1_E40_RUN_20261004_052043_692826/e1_seed%d'%seed)
    identities={}
    for key,folder in cache.items():
        metrics=read(folder/'official_metrics.json')
        pred=read(folder/'prediction_manifest.json')
        write(ref/(key+'_testdev_metrics.json'),metrics)
        write(ref/(key+'_testdev_prediction.json'),pred)
        identity=metrics['binding']['checkpoint_sha256']
        identities[key]=identity[key] if isinstance(identity,dict) else identity
    write(ref/'identities.json',dict(cached_testdev=identities))
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json')
    entries={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
    write(HERE/'manifest.json',dict(revision='e0_s23_parallel_e60_testdev_v1',sha256_normalized=entries))
    target=ROOT/(HERE.name+'.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in files+[HERE/'manifest.json']:
            archive.write(p,HERE.name+'/'+p.relative_to(HERE).as_posix())
    print(target)


if __name__=='__main__':main()
