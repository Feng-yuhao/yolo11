"""Verify the prepared target population, original image bytes and canonical GT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--target",choices=["songdo","dior","uavdt"],required=True)
    p.add_argument("--dataset-root",type=Path,required=True);a=p.parse_args()
    with (ROOT/f"data/manifests/{a.target}_test.csv").open(encoding="utf-8",newline="") as handle:rows=list(csv.DictReader(handle))
    expected={r["file_name"] for r in rows};folder=a.dataset_root/"images/test"
    actual={p.name for p in folder.glob("*.jpg")}
    if expected!=actual:raise RuntimeError("Missing or extra official target images")
    from PIL import Image
    checked=0
    for row in rows:
        file=folder/row["file_name"]
        if row["sha256"]:
            if sha(file)!=row["sha256"]:raise RuntimeError("Image SHA256 differs: "+row["file_name"])
            checked+=1
        with Image.open(file) as image:
            if image.size!=(int(row["width"]),int(row["height"])):raise RuntimeError("Image dimensions changed")
    gt={"songdo":"annotations_coco/test.json","dior":"annotations_coco/test_vehicle.json","uavdt":"annotations/gt_coco_style.json"}[a.target]
    binding=next(r for r in json.loads((ROOT/"data/manifests/target_GT_bindings.json").read_text(encoding="utf-8")) if r["dataset"]==a.target)
    if sha(a.dataset_root/gt)!=binding["GT_sha256"]:raise RuntimeError("Canonical GT bytes differ; review conversion/version before scoring")
    print(json.dumps({"target":a.target,"images":len(rows),"image_hashes_checked":checked,"GT_sha256_matches_original":True}))


if __name__=="__main__":main()
