"""Fail-closed E35 architecture, real loss, gradient, update and serialization checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

LEAVES = ("model.4.m.0", "model.6.m.0.m.0", "model.6.m.0.m.1")
PREFIXES = tuple(x + "." for x in LEAVES)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_config(cfg, reference):
    expected = copy.deepcopy(reference)
    for index in (4, 6):
        require(expected["backbone"][index][2] == "C3k2", "Reference is not E1")
        expected["backbone"][index][2] = "C3k2PKI"
    for key in ("backbone", "head", "nc", "scales"):
        require(cfg.get(key) == expected.get(key), f"E35 architecture drift: {key}")
    require(cfg.get("scale") == "s", "E35 must use s scale")


def assert_architecture(model):
    from ultralytics.nn.modules import Detect
    from ultralytics.nn.modules.e35a_pki import C3k2PKI, E35PKIBlock
    require(len(model.model) == 27, "Expected exactly 27 top-level layers")
    require(model.yaml.get("scale") == "s", "Wrong model scale")
    require([i for i, m in enumerate(model.model) if type(m) is C3k2PKI] == [4, 6],
            "Only backbone stages 4 and 6 may be PKI")
    leaves = {n: m for n, m in model.named_modules() if type(m) is E35PKIBlock}
    require(tuple(leaves) == LEAVES, f"Wrong PKI leaves: {list(leaves)}")
    for name, block in leaves.items():
        require(block.channels == 64, f"Wrong inner channels: {name}")
        require([m.kernel_size for m in block.inception.context] == [(k, k) for k in (5, 7, 9, 11)],
                f"Wrong context kernels: {name}")
        require(all(m.dilation == (1, 1) and m.groups == 64 for m in block.inception.context),
                "PKI must be dense, non-dilated depthwise convolution")
    head = model.model[-1]
    require(type(head) is Detect and head.nc == 10 and head.nl == 4, "Detect implementation changed")
    require(head.f == [25, 16, 19, 22], "Detect input indices changed")
    require(model.stride.tolist() == [4., 8., 16., 32.], "Detect strides changed")
    require(model.model[10].__class__.__name__ == "C2PSA", "P5 changed")
    return dict(pki_leaves=list(leaves), channels=[64, 64, 64], detect_strides=model.stride.tolist())


def synthetic_batch(device="cpu"):
    return dict(img=torch.rand(2, 3, 128, 128, device=device),
                batch_idx=torch.tensor([0., 0., 1., 1.], device=device),
                cls=torch.tensor([[0.], [3.], [5.], [9.]], device=device),
                bboxes=torch.tensor([[.25, .3, .10, .12], [.62, .6, .30, .35],
                                     [.45, .55, .12, .14], [.7, .25, .4, .32]], device=device))


def gradient_audit(source, device="cpu"):
    from ultralytics.utils.loss import v8DetectionLoss
    model = copy.deepcopy(source).float().to(device).train()
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    require(type(model.init_criterion()) is v8DetectionLoss, "Non-standard detection loss selected")
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.937)
    rows = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, items = model(synthetic_batch(device))
        require(torch.isfinite(loss).all().item(), "Non-finite detection loss")
        loss.sum().backward()
        norms = {}
        for name, parameter in model.named_parameters():
            if name.startswith(PREFIXES):
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                        f"Missing/non-finite PKI gradient: {name}")
                norm = float(parameter.grad.float().norm())
                require(norm > 0, f"Blocked PKI parameter: {name}")
                norms[name] = norm
        before = {n: p.detach().clone() for n, p in model.named_parameters()
                  if n.startswith(PREFIXES) and n.endswith("weight")}
        optimizer.step()
        for name, old in before.items():
            require(not torch.equal(old, model.get_parameter(name)), f"PKI parameter not updated: {name}")
        rows.append(dict(step=step + 1, loss=float(loss.detach().sum()),
                         loss_items=items.detach().cpu().tolist(), gradient_norms=norms))
    return rows


def serialization_audit(source, directory):
    model = copy.deepcopy(source).float().cpu().eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        expected, raw = model(image)
    require([list(x.shape[-2:]) for x in raw] == [[32, 40], [16, 20], [8, 10], [4, 5]],
            "Rectangular output shapes changed")
    path = Path(directory) / "e35_roundtrip.pt"
    torch.save({"model": model}, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)["model"]
    assert_architecture(restored)
    with torch.no_grad():
        actual = restored(image)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fused = restored.fuse(verbose=False).eval()(image)[0]
        torch.testing.assert_close(fused, expected, rtol=2e-4, atol=2e-4)
    path.unlink()
    return dict(checkpoint_roundtrip=True, rectangular_forward=True, fuse_equivalence=True)


def install_test(runtime):
    from ultralytics.nn.tasks import DetectionModel, yaml_model_load
    from ultralytics.utils import yaml_load
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cfg = yaml_model_load(str(runtime / "yolo11s-p2-pki.yaml"))
    cfg["scale"] = "s"
    validate_config(cfg, yaml_load(runtime / "e1_reference.yaml"))
    torch.manual_seed(1)
    model = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False)
    result = assert_architecture(model)
    with tempfile.TemporaryDirectory(prefix="e35_test_") as directory:
        result.update(serialization_audit(model, directory))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--install-test":
        raise SystemExit("Usage: e35a_checks.py --install-test RUNTIME")
    install_test(Path(sys.argv[2]))
