"""Explicit same-operator standard-26 -> P2 transfer, shared by both candidates."""
import copy
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
import ultralytics
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from e60_adapter import attach

WEIGHT_SHA = '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor')
MAPPING = dict([(i, i) for i in range(17)] + [(17, 23), (19, 25), (20, 26), (22, 28)])


def load_config():
    path = Path(ultralytics.__file__).resolve().parent / 'cfg/models/26/yolo26-p2.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg.update(scale='s', nc=10)
    return cfg, path


def transfer(model, source):
    if (len(source.model) != 24 or source.model[-1].nl != 3 or
            source.model[-1].reg_max != 1 or list(source.model[-1].f) != [16, 19, 22]):
        raise RuntimeError('Unexpected official standard YOLO26s checkpoint')
    if source.is_fused():
        raise RuntimeError('Need the unfused official training checkpoint')
    audit = []
    for source_i, target_i in MAPPING.items():
        old, new = source.model[source_i], model.model[target_i]
        if type(old) is not type(new):
            raise RuntimeError(f'Mapping operator mismatch {source_i}->{target_i}')
        old_state, new_state = old.state_dict(), new.state_dict()
        if set(old_state) != set(new_state) or any(
                old_state[k].shape != new_state[k].shape for k in old_state):
            raise RuntimeError(f'Mapping tensor topology mismatch {source_i}->{target_i}')
        new.load_state_dict(old_state, strict=True)
        if any(not torch.equal(new.state_dict()[k], old_state[k]) for k in old_state):
            raise RuntimeError('Transferred tensor equality failed')
        audit.append(dict(source_layer=source_i, target_layer=target_i,
                          operator=type(old).__name__, tensors=len(old_state),
                          tensor_keys=sorted(old_state)))
    return dict(mapping=audit, prediction_towers_transferred=False,
                reason='P2 changes input channels and tower widths; no implicit by-index head load',
                new_neck_layers=[19, 20, 22],
                pretrained_history='Official YOLO26 weights include Objects365 and COCO training; differs from YOLO11',
                target_names=list(NAMES))


def build_pair(weights):
    from common import sha256
    if sha256(weights) != WEIGHT_SHA:
        raise RuntimeError('Untrusted or different pretrained checkpoint')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    source = (checkpoint.get('ema') or checkpoint['model']).float()
    cfg, path = load_config()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1)
        baseline = DetectionModel(cfg, ch=3, nc=10, verbose=False)
    baseline.names = dict(enumerate(NAMES))
    baseline.args = SimpleNamespace(**dict(DEFAULT_CFG_DICT, epochs=200, imgsz=640,
                                           batch=16, optimizer='SGD', lr0=.01,
                                           box=7.5, cls=.5, dfl=1.5, nms=None,
                                           conf=.001, iou=.7, max_det=500))
    baseline.end2end = False  # inference branch only; both native heads still train
    baseline.model[-1].max_det = 500
    audit = transfer(baseline, source)
    del checkpoint, source
    baseline.yolo26_port_variant = 'baseline'
    enhanced = copy.deepcopy(baseline)
    rng = torch.get_rng_state().clone()
    attach(enhanced)
    if not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError('Adapter changed the global training RNG stream')
    enhanced.yolo26_port_variant = 'e60'
    base_state, new_state = baseline.state_dict(), enhanced.state_dict()
    if any(not torch.equal(value, new_state[key]) for key, value in base_state.items()):
        raise RuntimeError('Common initialized parameters changed during attach')
    return baseline, enhanced, dict(audit, config=str(path),
                                    common_initialization_identical=True,
                                    new_rng_isolated=True)
