"""Download public paper artifacts and verify their exact SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def download(row,directory):
    relative=Path(row.get("destination",row["asset"]))
    target=(directory/relative).resolve()
    if not target.is_relative_to(directory.resolve()):raise ValueError("Unsafe artifact destination")
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==row["bytes"] and sha(target)==row["sha256"]:
            print("Verified existing",relative);return
        raise RuntimeError("Existing file differs; choose a new output directory: "+str(target))
    temporary=target.with_name(target.name+".partial")
    request=urllib.request.Request(row["url"],headers={"User-Agent":"ST-MDRF-artifact-download"})
    with urllib.request.urlopen(request,timeout=120) as source,temporary.open("wb") as output:
        for block in iter(lambda:source.read(1024*1024),b""):output.write(block)
    if temporary.stat().st_size!=row["bytes"] or sha(temporary)!=row["sha256"]:
        raise RuntimeError("Download verification failed; .partial retained: "+str(temporary))
    temporary.replace(target);print("Downloaded and verified",relative)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group",choices=["weights","predictions","pretrained"],default="weights")
    p.add_argument("--model");p.add_argument("--seed",type=int,choices=[1,2,3])
    p.add_argument("--asset",help="Select a prediction archive or pretrained file by exact asset name")
    p.add_argument("--output",type=Path,default=ROOT)
    p.add_argument("--list",action="store_true")
    a=p.parse_args()
    filename={"weights":"weights.json","predictions":"prediction_archives.json","pretrained":"pretrained.json"}[a.group]
    data=json.loads((ROOT/"manifests"/filename).read_text(encoding="utf-8"))
    rows=data["checkpoints"] if a.group=="weights" else data["archives"] if a.group=="predictions" else data["files"]
    if a.model:rows=[r for r in rows if r.get("model")==a.model]
    if a.seed:rows=[r for r in rows if r.get("seed")==a.seed]
    if a.asset:rows=[r for r in rows if r["asset"]==a.asset]
    if not rows:p.error("No matching artifact")
    for row in rows:
        if a.list:print(row["asset"],row["bytes"],row["sha256"])
        else:download(row,a.output)


if __name__=="__main__":main()
