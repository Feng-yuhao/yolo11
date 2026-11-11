"""Preserved COCO scoring and UAVDT strict-ignore/native AP70 supplement."""
import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",choices=["songdo","dior","uavdt"],required=True)
    p.add_argument("--gt",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--ignore",type=Path,help="UAVDT ignore_by_image.json from prepare_data.py")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    directory={"songdo":"songdo","dior":"dior","uavdt":"uavdt_seeds23"}[a.target]
    sys.path.insert(0,str(ROOT/"implementations/transfer"/directory))
    opener=gzip.open if a.predictions.suffix==".gz" else open
    with opener(a.predictions,"rt",encoding="utf-8") as handle:predictions=json.load(handle)
    a.output.mkdir(parents=True)
    if a.target=="dior":
        from scoring import evaluate
        metrics=evaluate(a.gt,predictions,"dior",a.output)
    else:
        from coco_eval import coco_score
        if a.target=="uavdt":
            if a.ignore is None:p.error("UAVDT requires --ignore; the original ignore semantics cannot be dropped")
            from uavdt_data import filter_ignored
            import native_ap70
            ignored=json.loads(a.ignore.read_text(encoding="utf-8"))
            predictions,removed=filter_ignored(predictions,ignored)
            (a.output/"ignore_filter.json").write_text(json.dumps({"kept":len(predictions),"removed":len(removed),"rule":"strict containment; GT retained"},indent=2),encoding="utf-8")
            del removed
        metrics=coco_score(a.gt,predictions,a.output)
        if a.target=="uavdt":metrics["native_class_agnostic_AP70"]=native_ap70.score(json.loads(a.gt.read_text(encoding="utf-8")),predictions)
    (a.output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__=="__main__":main()
