"""Complete saved results are required before shutdown is even considered."""
from common import HERE,KEYS,SPECS,read,require,sha,split_key

def verify_outputs(report,require_optin=True):
    state=read(report/'state.json')
    require(state['status']=='completed' and (state['auto_shutdown'] or not require_optin),'Only completed opt-in run may shut down')
    expected=['check']+['smoke_'+k for k in KEYS]+['evaluation_'+k for k in KEYS]
    require(all(state['jobs'].get(k,{}).get('exit_code')==0 for k in expected),'A check/smoke/evaluation is missing or failed')
    require(sha(report/'ARTIFACT_SHA256.json')==state['artifact_manifest_sha256'],'Saved artifact index changed')
    for n,d in read(report/'ARTIFACT_SHA256.json').items():
        p=report/n;require(p.is_file() and p.resolve().is_relative_to(report.resolve()) and sha(p)==d,'Saved artifact changed: '+n)
    binding=read(report/'binding.json');specs=read(HERE/'references/checkpoints.json')
    for k in KEYS:
        target,source_key=split_key(k);out=report/'evaluation'/k
        p=read(out/'prediction_manifest.json');s=read(out/'metrics.json');n=SPECS[target]['images']
        require(p['status']=='completed' and p['seen_images']==len(set(p['seen_image_ids']))==n,'Incomplete seen IDs')
        require(p['image_ids_sha256']==binding['datasets'][target]['image_ids_sha256'],'Wrong evaluation image population')
        require(p['parameters_and_BN_unchanged'] and p['inference_state_sha256_before']==p['inference_state_sha256_after'],'Model/BN changed')
        require(p['checkpoint_sha256']==s['checkpoint_sha256']==specs[source_key]['sha256'],'Wrong checkpoint proof')
        require(p['prediction_sha256']==s['prediction_sha256']==sha(out/'predictions.json.gz'),'Saved predictions changed')
        require(s['status']=='completed' and not s['engineering_only'] and s['gzip_readback_verified'],'Score not complete')
        require(all(s[f]==SPECS[target][f] for f in ('images','positive_boxes','ignored_entries')),'Incomplete scoring population')
        require(all(s[f] is not None for f in ('AP_percent','AP50_percent','AP75_percent')),'Main AP not reported')
        if target=='tiny':require(all(s['native_tiny'].get(f'AP{t}_tiny_percent') is not None for t in (25,50,75)),'Native TinyPerson AP missing')
    summary=read(report/'summary.json')
    require(summary['status']=='completed' and set(summary['targets'])=={'dior','tiny'} and (report/'REPORT.md').is_file(),'Two-target summary missing')
    return True
