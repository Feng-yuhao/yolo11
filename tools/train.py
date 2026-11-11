"""Original recipes and preserved architecture builders; --execute starts training."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from st_mdrf.runtime import specification,boot,build_model,sha


def training_label_fingerprint(images):
    from ultralytics.data.utils import img2label_paths
    files=sorted(p.absolute() for p in Path(images).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"})
    if len(files)!=6471:raise RuntimeError("Paper training requires the original 6471-image training split")
    labels=img2label_paths([str(p) for p in files]);digest=hashlib.sha256()
    for image,label in zip(files,labels):digest.update(image.name.encode()+b"\0"+Path(label).read_bytes()+b"\0")
    expected="7ac9962a06c0150d9e4ba052827ba41d2bec7f487ec9ccd2a72cfafa4eccedd5"
    if digest.hexdigest()!=expected:raise RuntimeError("Training labels differ from the original experiment; inspect the data preparation report")
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True);p.add_argument("--seed",type=int,choices=[1,2,3],required=True)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--pretrained",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="0")
    p.add_argument("--execute",action="store_true",help="Explicitly run the 200-epoch training recipe")
    a=p.parse_args();a.data=a.data.resolve();a.pretrained=a.pretrained.resolve();a.output=a.output.resolve()
    spec=specification(a.model,a.seed)
    import yaml
    recorded=yaml.safe_load((ROOT/spec["training_args"]).read_text(encoding="utf-8"))
    if recorded["seed"]!=a.seed or recorded["epochs"]!=200 or recorded["batch"]!=16:raise RuntimeError("Archived training recipe mismatch")
    print(json.dumps({"model":a.model,"seed":a.seed,"epochs":200,"batch":16,"imgsz":640,"training_will_start":a.execute,"data":str(a.data.resolve()),"output":str(a.output.resolve()),"recipe":spec["training_args"]},indent=2))
    if not a.execute:return
    if a.device=="cpu":p.error("The paper's AMP training recipe requires CUDA; this entry does not substitute CPU training")
    if a.output.exists():p.error("Output already exists; choose a new output directory")
    init_manifest=json.loads((ROOT/"manifests/pretrained.json").read_text(encoding="utf-8"))
    auxiliary="yolo26n.pt" if spec["family"]=="yolo26" else "yolov8n.pt"
    expected_aux=next(r for r in init_manifest["files"] if Path(r["destination"]).name==auxiliary)
    aux_path=a.pretrained.parent/auxiliary
    if not aux_path.is_file() or sha(aux_path)!=expected_aux["sha256"]:raise RuntimeError("Download the native AMP auxiliary model with tools/download.py --group pretrained")
    spec,torch,_=boot(a.model,a.seed)
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no training started")
    from ultralytics.data.utils import check_det_dataset
    data=check_det_dataset(str(a.data.resolve()),autodownload=False)
    label_sha=training_label_fingerprint(data["train"])
    val_count=sum(p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"} for p in Path(data["val"]).rglob("*"))
    if val_count!=548:raise RuntimeError("Paper validation requires all 548 original validation images")
    os.chdir(a.pretrained.parent)
    from ultralytics.utils import DEFAULT_CFG_DICT
    extra=set(recorded)-set(DEFAULT_CFG_DICT)-{"save_dir"}
    if extra:raise RuntimeError("Unknown archived recipe fields: "+str(sorted(extra)))
    options={k:v for k,v in recorded.items() if k in DEFAULT_CFG_DICT}
    options.update(data=str(a.data.resolve()),device=a.device,project=str(a.output.resolve()/"runs"),name=f"{a.model}_seed{a.seed}",exist_ok=False,resume=False)
    a.output.mkdir(parents=True)
    (a.output/"input_binding.json").write_text(json.dumps({"training_label_sha256":label_sha,"pretrained_sha256":sha(a.pretrained),"configuration":spec,"options":options},indent=2),encoding="utf-8")
    if spec["family"]=="yolo26":
        # Reuse the original paired trainer and native dual-branch loss logic.
        from types import SimpleNamespace
        import training26
        expected="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
        if sha(a.pretrained)!=expected:raise RuntimeError("Different YOLO26 pretrained initialization")
        init=a.output/"weights";init.mkdir();shutil.copy2(a.pretrained,init/"yolo26s.pt")
        task=SimpleNamespace(project=a.output.resolve(),data=a.data.resolve(),source=ROOT/"vendor/ultralytics26",device=a.device,local_test=False)
        report=a.output/"reports";report.mkdir()
        options["model"]=str(ROOT/"vendor/ultralytics26/ultralytics/cfg/models/26/yolo26-p2.yaml")
        trainer=training26.PairedTrainer(task,"e60" if a.model.endswith("st-mdrf") else "baseline",report,options)
        trainer.train()
    else:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from p2_tal_chunked import chunked_assignment
        class PaperTrainer(DetectionTrainer):
            def get_model(self,cfg=None,weights=None,verbose=True):
                if weights is not None:raise RuntimeError("Use the original pretrained initialization, not trained best weights")
                target,_,audit=build_model(a.model,a.seed,a.pretrained)
                (a.output/"initialization.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
                return target
        options["model"]=str(ROOT/"vendor/ultralytics83/ultralytics/cfg/models/11/yolo11.yaml" if a.model=="yolo11s" else ROOT/"implementations/yolo11/full_seed1/e1_reference.yaml")
        trainer=PaperTrainer(overrides=options)
        with chunked_assignment():trainer.train()


if __name__=="__main__":main()
