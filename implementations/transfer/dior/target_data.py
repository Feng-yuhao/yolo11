"""Read immutable installed test packages; produce separately bound evaluation GT."""
import hashlib
import importlib.util
import json
import os
from pathlib import PurePosixPath
from common import HERE,SPECS,read,require,sha,write

def root_for(parent,target):return parent/SPECS[target]['folder']
def image_id_sha(ids):return hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()
def audit(root,target,full=False):
    root=root.resolve();ref=read(HERE/'references/datasets.json')[target]
    require(sha(root/'SHA256SUMS.json')==ref['manifest_sha256'],'Wrong target payload manifest')
    require(sha(root/'PROTOCOL.json')==ref['protocol_sha256'],'Wrong conversion protocol')
    protocol=read(root/'PROTOCOL.json');m=read(root/'SHA256SUMS.json')
    # Scan each parent once: preserve the exact same per-file size/mtime binding
    # without repeatedly resolving35k deep Windows paths in every worker.
    groups={};table={}
    for name in m:
        relative=PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,'Unsafe manifest path')
        groups.setdefault(relative.parent.as_posix(),[]).append(relative.name)
    for parent,names in groups.items():
        directory=root/parent
        require(directory.resolve().is_relative_to(root) and not directory.is_symlink(),'Unsafe payload directory')
        for ancestor in directory.parents:
            if ancestor==root:break
            require(not ancestor.is_symlink(),'Symlinked payload ancestor')
        with os.scandir(directory) as scan:entries={entry.name:entry for entry in scan}
        for name in names:
            entry=entries.get(name);require(entry is not None and entry.is_file(follow_symlinks=False),'Missing/unsafe target payload')
            s=entry.stat(follow_symlinks=False);relative=name if parent=='.' else parent+'/'+name
            table[relative]=(relative,s.st_size,s.st_mtime_ns)
    stats=[table[name] for name in sorted(m)]
    if full:
        print('TARGET full SHA256/decode/supervision check: '+target,flush=True)
        loader=importlib.util.spec_from_file_location('isolated_data_verifier',HERE/'dataset_verify.py')
        mod=importlib.util.module_from_spec(loader);loader.loader.exec_module(mod)
        proof=mod.verify(root,decode=True)
        print('TARGET full check passed: '+target,flush=True)
        require(tuple(proof[k] for k in ('images','positive_boxes','ignored_entries'))==tuple(SPECS[target][k] for k in ('images','positive_boxes','ignored_entries')),'Wrong target counts')
    else:proof=None
    gt=root/protocol['canonical_annotation'];require(sha(gt)==ref['gt_sha256'],'Target GT changed')
    records=read(gt)['images']
    return dict(root=str(root),manifest_sha256=ref['manifest_sha256'],gt_sha256=sha(gt),
                images=len(records),image_ids_sha256=image_id_sha([r['id'] for r in records]),
                file_stat_sha256=hashlib.sha256(json.dumps(stats,separators=(',',':')).encode()).hexdigest(),
                full_check=proof)
def records(root):
    p=read(root/'PROTOCOL.json');gt=read(root/p['canonical_annotation'])
    return [dict(id=i['id'],path=root/i['file_name'],width=i['width'],height=i['height']) for i in gt['images']]
def prepare_gt(root,target,out):
    p=read(root/'PROTOCOL.json');gt=read(root/p['canonical_annotation'])
    if target=='tiny':
        for a in gt['annotations']:
            a['source_annotation_id']=a['id'];a['id']+=1
            require(a['id']>0,'COCO matching zero-sentinel ID')
    write(out/'gt.json',gt)
    proof=dict(sha256=sha(out/'gt.json'),images=len(gt['images']),
               positive_boxes=sum(not a.get('ignore',False) for a in gt['annotations']),
               ignored_entries=sum(bool(a.get('ignore',False)) for a in gt['annotations']),
               image_ids_sha256=image_id_sha([i['id'] for i in gt['images']]),
               geometry_unchanged=True,tiny_annotation_id_offset=1 if target=='tiny' else 0)
    write(out/'proof.json',proof);return proof
