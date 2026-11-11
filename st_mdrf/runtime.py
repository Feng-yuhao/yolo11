"""Load each architecture in a fresh process, preserving original pickle module names."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
NAMES=("pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor")
_ACTIVE=None


def experiments():
    return json.loads((ROOT/"configs/experiments.json").read_text(encoding="utf-8"))


def specification(model,seed):
    matches=[r for r in experiments() if r["model"]==model and r["seed"]==seed]
    if len(matches)!=1:raise ValueError("Unknown model/seed; see configs/experiments.json")
    return matches[0]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()


def boot(model,seed):
    global _ACTIVE
    spec=specification(model,seed)
    family=spec["family"]
    vendor=ROOT/"vendor"/("ultralytics26" if family=="yolo26" else "ultralytics83")
    os.environ.setdefault("YOLO_CONFIG_DIR",str(ROOT/".runtime/settings"))
    os.environ.setdefault("YOLO_AUTOINSTALL","false")
    os.environ.setdefault("YOLO_VERBOSE","false")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD","1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    (ROOT/".runtime/settings").mkdir(parents=True,exist_ok=True)
    if "ultralytics" in sys.modules:
        if _ACTIVE==(model,seed):
            return spec,sys.modules["torch"],sys.modules["ultralytics"]
        raise RuntimeError("Use a fresh Python process for each model/seed; do not mix frozen imports")
    sys.path[:0]=[str(vendor),str(ROOT/spec["implementation"])]
    import torch
    import ultralytics
    expected="8.4.177" if family=="yolo26" else "8.3.0"
    if ultralytics.__version__!=expected or not Path(ultralytics.__file__).resolve().is_relative_to(vendor.resolve()):
        raise RuntimeError("Wrong frozen Ultralytics source")
    from ultralytics.utils import SETTINGS
    SETTINGS.update({k:False for k in ["sync","clearml","comet","dvc","hub","mlflow","neptune","raytune","wandb"] if k in SETTINGS})
    if family=="yolo26":importlib.import_module("e60_adapter")
    elif model=="tood-td":importlib.import_module("tood_td_head")
    elif model not in ["yolo11s","yolo11s-p2"]:
        head=importlib.import_module("spatial_task_head")
        head.configure({"st-mdrf":"e60","no-group":"e58","uniform":"uniform","shared":"shared"}[model])
    _ACTIVE=(model,seed)
    return spec,torch,ultralytics


def checkpoint_entry(model,seed):
    value=json.loads((ROOT/"manifests/weights.json").read_text(encoding="utf-8"))
    return next(r for r in value["checkpoints"] if r["model"]==model and r["seed"]==seed)


def load_model(model,seed,weights=None,device="cpu"):
    spec,torch,_=boot(model,seed)
    entry=checkpoint_entry(model,seed)
    path=Path(weights) if weights is not None else ROOT/entry["destination"]
    if not path.is_file():raise FileNotFoundError("Download the checkpoint first with tools/download.py: "+str(path))
    if sha(path)!=entry["sha256"]:raise RuntimeError("Checkpoint SHA256 differs from the paper artifact")
    from ultralytics import YOLO
    wrapper=YOLO(str(path),task="detect")
    wrapper.model.to(device).float().eval()
    if tuple(wrapper.model.names.values())!=NAMES:raise RuntimeError("Wrong class order")
    if spec["family"]=="yolo26":
        wrapper.model.end2end=False
        wrapper.model.model[-1].max_det=500
    return wrapper,spec


def build_model(model,seed,pretrained):
    """Original pretraining transfer and original isolated module initialization."""
    spec,torch,_=boot(model,seed)
    pretrained=Path(pretrained).resolve()
    expected="646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b" if spec["family"]=="yolo26" else "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
    if not pretrained.is_file() or sha(pretrained)!=expected:raise RuntimeError("Use the original pretrained checkpoint from the release manifest")
    if spec["family"]=="yolo26":
        original=importlib.import_module("models26")
        baseline,full,audit=original.build_pair(pretrained)
        return (full if model.endswith("st-mdrf") else baseline),spec,audit
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel,yaml_model_load
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(seed,deterministic=True)
    if model=="yolo11s":
        config=yaml_model_load(str(ROOT/"vendor/ultralytics83/ultralytics/cfg/models/11/yolo11.yaml"));config["scale"]="s"
        target=DetectionModel(config,nc=10,verbose=False)
        target.load(YOLO(str(pretrained)).model)
        target.names=dict(enumerate(NAMES))
        return target,spec,{"initialization":"original COCO-pretrained YOLO11s, ten-class target"}
    core=importlib.import_module("e1_640_core")
    core.PRETRAINED=pretrained;core.torch=torch;core.YOLO=YOLO
    config_path=ROOT/spec["implementation"]/"e1_reference.yaml"
    if not config_path.is_file():config_path=ROOT/"implementations/yolo11/full_seed1/e1_reference.yaml"
    config=yaml_model_load(str(config_path));config["scale"]="s"
    target=DetectionModel(config,nc=10,verbose=False)
    core.assert_p2_model(target)
    audit=core.load_p2_pretrained(target)
    target.names=dict(enumerate(NAMES))
    if model!="yolo11s-p2":
        head=importlib.import_module("tood_td_head" if model=="tood-td" else "spatial_task_head")
        target=head.attach(target)
    return target,spec,audit
