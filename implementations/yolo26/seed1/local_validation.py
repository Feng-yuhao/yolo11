"""Private synthetic integration harness. Never issues a server smoke receipt."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import json
import numpy as np
from PIL import Image,ImageDraw
import yaml
from protocol import NAMES,write


def fixture(root):
    rng=np.random.default_rng(163)
    for split,count in (('train',48),('val',35)):
        for folder in ('images','labels','annotations'): (root/split/folder).mkdir(parents=True)
        for i in range(count):
            h,w=(128,160) if split=='val' and i%2 else (160,160)
            image=Image.fromarray(rng.integers(20,150,(h,w,3),dtype=np.uint8)); draw=ImageDraw.Draw(image)
            labels=[]; annotations=[]
            for j,(x,y,bw,bh) in enumerate(((7,11,8,7),(47,35,19,23),(60,54,48,50),(21,23,100,96))):
                cls=(i+j)%10
                draw.rectangle((x,y,x+bw,y+bh),outline=(255,255,255),width=2)
                labels.append(f'{cls} {(x+bw/2)/w:.9f} {(y+bh/2)/h:.9f} {bw/w:.9f} {bh/h:.9f}')
                annotations.append(f'{x},{y},{bw},{bh},1,{cls+1},0,0')
            stem=f'{split}_{i:04d}'
            image.save(root/split/'images'/(stem+'.jpg'))
            (root/split/'labels'/(stem+'.txt')).write_text('\n'.join(labels)+'\n')
            (root/split/'annotations'/(stem+'.txt')).write_text('\n'.join(annotations)+'\n')
    path=root/'fixture.yaml'
    path.write_text(yaml.safe_dump(dict(path=str(root.resolve()),train='train/images',val='val/images',names=NAMES)),encoding='utf-8')
    return path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--stage',choices=('smoke','train','eval','all'),default='all')
    args=parser.parse_args(); here=Path(__file__).resolve().parent
    args.root=args.root.resolve(); args.source=args.source.resolve()
    args.root.mkdir(parents=True,exist_ok=True)
    data=args.root/'data/fixture.yaml'
    if not data.exists(): data=fixture(args.root/'data')
    project=args.root/'project'; (project/'weights').mkdir(parents=True,exist_ok=True)
    shutil.copyfile(here/'weights/yolo26s.pt',project/'weights/yolo26s.pt')
    reports=project/'comparison_reports'; reports.mkdir(exist_ok=True)
    stages=('smoke','train','eval') if args.stage=='all' else (args.stage,)
    summary={}
    for variant in ('baseline','e60'):
        for stage in stages:
            out=reports/(variant+'_'+stage)
            cmd=[sys.executable,'-u',str(here/'worker.py'),stage,'--variant',variant,'--report',str(out),
                 '--project',str(project),'--source',str(args.source),'--data',str(data),
                 '--val-dataset',str(args.root/'data/val'),'--local-test','--local-size','128','--local-epochs','2']
            if stage=='eval': cmd+=['--weights',str(project/'runs'/('local_integration_'+variant+'_2e')/'weights/best.pt')]
            result=subprocess.run(cmd)
            if result.returncode: raise RuntimeError('Local '+variant+' '+stage+' failed')
            summary[variant+'_'+stage]=json.loads((out/'completion.json').read_text())['result']
    write(args.root/'PRIVATE_LOCAL_VALIDATION.json',dict(local_only=True,real_VisDrone_tested=False,
          server_gate_issued=False,results=summary))


if __name__=='__main__': main()
