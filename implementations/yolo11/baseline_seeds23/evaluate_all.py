"""Serial official VAL then held-out TEST-DEV; frozen val-best, no training."""
import argparse
import gc
from pathlib import Path
import statistics
import sys

import common as c
import base_eval as base
import e60_eval_reference as e60

ORDER=('e0_seed2','e0_seed3','e60_seed2','e60_seed3')
KEYS=base.KEYS


def setup(strict=True):
    from types import SimpleNamespace
    args=SimpleNamespace(project=c.PROJECT,source=c.SOURCE,dataset=c.DATA_ROOT/'VisDrone2019-DET-test-dev',
                         val_dataset=c.DATA_ROOT/'VisDrone2019-DET-val',device='0')
    torch=e60.prepare(args,strict=strict)
    # These dependencies are frozen actual E60 code, not attached to E0.
    original=e60.model_audit
    def audit(model,key):
        if key.startswith('e60_'):
            return original(model,e60.KEY)
        from train_e0 import audit_model
        return audit_model(model.model)
    base.model_audit=audit
    base.REF=c.REF
    return args,torch


def cached(key,protocol):
    metric=c.read(c.REF/(key+'_testdev_metrics.json'))
    pred=c.read(c.REF/(key+'_testdev_prediction.json'))
    binding=metric['binding']
    c.require(metric['status']==pred['status']=='completed' and binding==pred['binding'],'Incomplete cached '+key)
    c.require(binding['dataset_annotation_sha256']==base.TEST_ANN_SHA and
              binding['dataset_image_sha256']==base.TEST_IMAGE_SHA and
              binding['image_count']==pred['seen_images']==1610 and not binding.get('smoke_only',False),
              'Wrong cached test split '+key)
    c.require(binding['protocol']==dict(protocol,device=binding['protocol']['device']),'Cached inference protocol differs '+key)
    c.require(metric['prediction_sha256']==pred['sha256'],'Cached prediction SHA differs '+key)
    c.require(pred['backend']['matmul_allow_tf32'] is False and pred['backend']['cudnn_allow_tf32'] is True,
              'Cached TF32 differs '+key)
    checkpoint=binding['checkpoint_sha256']
    actual=checkpoint[key] if isinstance(checkpoint,dict) else checkpoint
    expected=c.read(c.REF/'identities.json')['cached_testdev'][key]
    c.require(actual==expected,'Cached val-best SHA differs '+key)
    return metric


def preflight(args):
    c.package_audit();c.protocol_audit()
    source=base.source_binding(args)
    prior=c.read(c.REF/'seed1_testdev_summary.json')['binding']['source']
    for name,value in source.items():
        if name!='nn/tasks.py':c.require(value==prior[name],'Cached inference source differs '+name)
    records,meta=base.audit_dataset(args.dataset)
    val,val_meta=base.load_dataset(args.val_dataset,548)
    c.require(val_meta['annotation_dimension_sha256']==
              '058c87e779a61a09933bebe35af1c05c807a444dc842b4be12124dd36134c787','Wrong original val annotations')
    models={key:cached(key,base.protocol(args.device)) for key in
            ('e0_seed1','e60_seed1','e1_seed1','e1_seed2','e1_seed3')}
    for seed in (2,3):c.e60_identity(seed)
    return records,meta,val,val_meta,models,source


def engineering(out):
    args,torch=setup()
    records,meta,val,val_meta,_,source=preflight(args)
    base.cuda_limit(args,torch)
    result=base.units()
    # E0 checks use the actual unchanged original architecture in train_e0.
    for seed in (2,3):
        key='e60_seed%d'%seed
        identity=c.e60_identity(seed)
        base.SPECS={key:dict(nl=4,stride=(4,8,16,32))}
        actual,_=base.predict(args,key,identity['weights'],val[:8],out/key/'val8')
        expected=c.read(c.REF/('e60_seed%d_val_smoke.json'%seed))
        result[key]=base.compare_predictions(actual,expected['predictions'])
        base.predict(args,key,identity['weights'],records[:8],out/key/'testdev8')
        small,_=base.score(records[:8],out/key/'testdev8')
        result[key]['testdev8']=small
    c.write(out/'eval_checks.json',result)
    c.write(c.PROJECT/'comparison_reports/e0_s23_pair_gates/eval_check.json',dict(status='passed',
            package=c.package_audit(),source=source,identities={str(s):c.e60_identity(s) for s in (2,3)},
            dataset=meta,report=str(out),finished_at=c.now()))
    print('E60 seeds2/3 frozen val-best, real val8 reproduction/testdev8 and official coordinate checks PASSED',flush=True)


def summarize(models):
    # Fixed seeds123, same-seed deltas. NO seed dropping and no inference of AP-small.
    stats={}
    for kind in ('e0','e1','e60'):
        stats[kind]={}
        for metric in KEYS:
            v=[models['%s_seed%d'%(kind,s)]['official'][metric] for s in (1,2,3)]
            stats[kind][metric]=dict(seeds=[1,2,3],values=v,mean=statistics.mean(v),sample_std=statistics.stdev(v))
    deltas={}
    for baseline in ('e0','e1'):
        deltas['e60_minus_'+baseline]={}
        for metric in KEYS:
            v=[models['e60_seed%d'%s]['official'][metric]-models['%s_seed%d'%(baseline,s)]['official'][metric]
               for s in (1,2,3)]
            deltas['e60_minus_'+baseline][metric]=dict(values_pp=v,mean_pp=statistics.mean(v),
                                                     sample_std_pp=statistics.stdev(v))
    return dict(status='completed',split='test-dev',seeds=[1,2,3],models=models,statistics=stats,paired_deltas=deltas,
                boundary='E60-E0 = total architecture gain; E60-E1 = added head gain. '
                         'Three seeds are descriptive evidence, not a publication or significance guarantee. '
                         'Val-selected best.pt; no test checkpoint/threshold selection. Overall AP is not AP-small.')


def markdown(summary):
    lines=['# Official VisDrone test-dev: E0 / E1 / E60, seeds 1/2/3','',
           '|Model|AP mean ± SD|AP50 mean ± SD|AP75 mean ± SD|','|---|---:|---:|---:|']
    for kind in ('e0','e1','e60'):
        values=[summary['statistics'][kind][m] for m in KEYS[:3]]
        lines.append('|'+kind.upper()+'|'+'|'.join('%.4f ± %.4f'%(v['mean'],v['sample_std']) for v in values)+'|')
    lines+=['','All values above are percentages; SD is sample SD (n=3).','']
    for key,metrics in summary['paired_deltas'].items():
        v=metrics[KEYS[0]]
        lines.append('- %s: AP mean %+.4f pp, paired SD %.4f pp; individual gains %s.'%
                     (key,v['mean_pp'],v['sample_std_pp'],', '.join('%+.4f'%x for x in v['values_pp'])))
    lines+=['','|Model / seed|AP|AP50|AP75|AR500|','|---|---:|---:|---:|---:|']
    for kind in ('e0','e1','e60'):
        for seed in (1,2,3):
            key='%s_seed%d'%(kind,seed)
            lines.append('|'+key+'|'+'|'.join('%.4f'%summary['models'][key]['official'][m] for m in KEYS)+'|')
    return '\n'.join(lines+['',summary['boundary'],''])


def full(out):
    args,torch=setup()
    records,meta,val,val_meta,models,source=preflight(args)
    gate=c.read(c.PROJECT/'comparison_reports/e0_s23_pair_gates/eval_check.json')
    c.require(gate['status']=='passed' and gate['package']==c.package_audit() and gate['source']==source,
              'Repeat matching check; inference package/source changed')
    base.cuda_limit(args,torch)
    val_scores={}
    for key in ORDER:  # Explicit serial loop, never launches concurrent inference.
        seed=int(key[-1])
        identity=c.completed_e0(seed) if key.startswith('e0_') else c.e60_identity(seed)
        base.SPECS={key:dict(nl=3 if key.startswith('e0_') else 4,
                            stride=(8,16,32) if key.startswith('e0_') else (4,8,16,32))}
        folder=out/key
        # Newly trained E0 has no old val8 predictions: score ALL 548 before
        # touching test-dev. E60 additionally reproduces its original val8.
        predictions,_=base.predict(args,key,identity['weights'],val,folder/'val')
        if key.startswith('e60_'):
            expected=c.read(c.REF/('e60_seed%d_val_smoke.json'%seed))
            base.compare_predictions([x for x in predictions if x['image_id']<=8],expected['predictions'])
        val_metrics,val_audit=base.score(val,folder/'val')
        val_scores[key]=val_metrics
        c.write(folder/'val/official_metrics.json',dict(status='completed',official=val_metrics,ignore_audit=val_audit,
                checkpoint_sha256=identity['checkpoint_sha256'],split='val'))
        if key.startswith('e60_'):
            reference=c.read(c.REF/('e60_seed%d_comparison.json'%seed))['candidate']
            c.require(all(abs(val_metrics[m]-reference[m])<1e-7 for m in KEYS),'E60 original full val scoring did not reproduce')
        _,prediction=base.predict(args,key,identity['weights'],records,folder/'testdev')
        print(c.now()+' '+key+' OFFICIAL TEST-DEV SCORING (CPU, all10 classes)',flush=True)
        metrics,audit=base.score(records,folder/'testdev')
        c.require(c.sha(identity['weights'])==identity['checkpoint_sha256'],'Read-only evaluation changed checkpoint')
        binding=dict(key=key,checkpoint_sha256=identity['checkpoint_sha256'],protocol=base.protocol(args.device),
                     dataset_annotation_sha256=base.TEST_ANN_SHA,dataset_image_sha256=base.TEST_IMAGE_SHA,
                     image_count=1610,smoke_only=False,source=source,package=c.package_audit())
        result=dict(status='completed',official=metrics,binding=binding,ignore_audit=audit,
                    checkpoint=identity['weights'],prediction_sha256=prediction['sha256'],finished_at=c.now())
        prediction['binding']=binding
        c.write(folder/'testdev/official_metrics.json',result)
        c.write(folder/'testdev/prediction_manifest.json',prediction)
        models[key]=result
        c.write(out/'progress.json',dict(completed=list(models),current=key,finished_at=c.now()))
        gc.collect();torch.cuda.empty_cache()
    summary=summarize(models)
    summary.update(dataset=meta,new_val_results=val_scores,finished_at=c.now())
    c.write(out/'summary.json',summary)
    (out/'REPORT.md').write_text(markdown(summary),encoding='utf-8')
    c.write(out/'COMPLETED.json',dict(status='completed',summary_sha256=c.sha(out/'summary.json'),finished_at=c.now()))
    print(c.now()+' ALL FOUR SERIAL EVALUATIONS COMPLETED REPORT='+str(out/'REPORT.md'),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('check','run'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    try:
        (engineering if args.mode=='check' else full)(args.out)
    except BaseException:
        import traceback
        c.write(args.out/'FAILED.json',dict(status='failed',error=traceback.format_exc(),at=c.now()))
        raise


if __name__=='__main__':main()
