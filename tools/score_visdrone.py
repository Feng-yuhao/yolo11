"""Score fixed predictions with the preserved MATLAB-v1.0.4 VisDrone port."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"implementations/yolo11/full_seed1"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True);p.add_argument("--predictions",type=Path,required=True)
    p.add_argument("--split",choices=["val","test-dev"],required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    import d27_official as official
    official.EXPECTED_IMAGES=548 if a.split=="val" else 1610
    records,metadata=official.load_dataset(a.dataset_root)
    temporary=None;path=a.predictions
    try:
        if path.suffix==".gz":
            temporary=tempfile.TemporaryDirectory();path=Path(temporary.name)/"predictions.json"
            with gzip.open(a.predictions,"rb") as original,path.open("wb") as target:
                import shutil;shutil.copyfileobj(original,target)
        predictions=official.load_predictions(path,len(records))
        filtered,audit=official.filter_detections_and_audit(records,predictions)
        metrics=official.official_evaluate(records,filtered)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({"official":metrics,"ignore_audit":audit,"dataset":metadata,"scorer_revision":official.REVISION},indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps({k:v for k,v in metrics.items() if not isinstance(v,dict)},indent=2))
    finally:
        if temporary is not None:temporary.cleanup()


if __name__=="__main__":main()
