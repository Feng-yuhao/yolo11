#!/usr/bin/env python3
"""Isolated inference and pinned D27 scoring of six val-selected checkpoints.

No training, remote control, weight mutation, or test-based checkpoint selection.
CUDA runs only in short-lived child processes. Parent scoring uses NumPy/CPU.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import d27_official as official

PACKAGE = Path(__file__).resolve().parent
REVISION = "testdev_e1_e40_paired3_v1"
KEYS = ("AP50_95_percent", "AP50_percent", "AP75_percent", "AR500_percent")
PREDICTION_OPTIONS = dict(imgsz=640, conf=.001, iou=.7, max_det=500,
                          half=False, augment=False, verbose=False, save=False,
                          agnostic_nms=False, classes=None)
EXPECTED_IMAGES = 1610
GIB = 1024 ** 3


def inference_protocol():
    return dict(**PREDICTION_OPTIONS, batch=4, device="0",
                letterbox="unchanged Ultralytics 8.3.0 predictor pre_transform",
                deterministic_algorithms=True, deterministic_warn_only=True,
                cudnn_deterministic=True, cudnn_benchmark=False)


def configure_numeric(torch):
    # Match the project's deterministic formal-training/evaluation policy.
    # No backward pass or training RNG is touched in another process.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, payload):
    """Atomic receipt/result publication; a partial file is never a success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_records(dataset, expected=EXPECTED_IMAGES, hash_images=True):
    dataset = Path(dataset).resolve()
    # Only this independent process's unchanged vendored loader is parameterized.
    # This does not edit the shared scorer or affect another process's imports.
    previous = official.EXPECTED_IMAGES
    official.EXPECTED_IMAGES = expected
    try:
        records, summary = official.load_dataset(dataset)
    finally:
        official.EXPECTED_IMAGES = previous
    supported = {".jpg", ".jpeg", ".png"}
    images = [p for p in (dataset / "images").iterdir() if p.is_file() and p.suffix.lower() in supported]
    stems = [p.stem for p in images]
    require(len(stems) == len(set(stems)), "Duplicate image stems/extensions in dataset")
    require(set(stems) == {r.stem for r in records}, "Images and original annotations are not one-to-one")
    require(all(np.all(r.raw_gt[:, 5] == np.floor(r.raw_gt[:, 5])) for r in records), "Noninteger GT class IDs")
    require(all(np.all(np.isin(r.raw_gt[:, 4], [0, 1])) for r in records), "GT score/ignore flag is not 0/1")
    table = []
    sha = hashlib.sha256()
    for i, record in enumerate(records, 1):
        image_sha = official.sha256(record.image_path) if hash_images else None
        table.append(dict(id=record.image_id, stem=record.stem, path=str(record.image_path),
                          width=record.width, height=record.height, sha256=image_sha))
        if hash_images:
            sha.update((record.image_path.name + "\0" + image_sha + "\0").encode())
        if i == 1 or i % 200 == 0 or i == len(records):
            print(f"DATA audit {i}/{len(records)}: original annotations / image identity", flush=True)
    summary.update(split="test-dev", image_content_sha256=sha.hexdigest() if hash_images else None)
    return records, summary, table


def vendor_check():
    manifest = read(PACKAGE / "models.json")
    require(manifest["revision"] == REVISION, "Manifest revision mismatch")
    for filename, expected in manifest["vendor_sha256"].items():
        require(official.sha256(PACKAGE / filename) == expected, "Changed pinned vendor file: " + filename)
    specs = manifest["models"]
    require([(s["model"], s["seed"]) for s in specs] ==
            [(m, seed) for seed in (1, 2, 3) for m in ("E1", "E40")], "Expected exactly three E1/E40 seed pairs")
    require(len({s["sha256"] for s in specs}) == 6, "Duplicate checkpoints in manifest")
    return specs


def resolve_specs(project):
    specs = []
    for item in vendor_check():
        path = (project / item["weights"]).resolve()
        require(path.is_relative_to(project.resolve()) and path.name == "best.pt", "Unsafe/non-best checkpoint path")
        require(path.is_file(), "Missing val-selected checkpoint: " + str(path))
        require(official.sha256(path) == item["sha256"], "Checkpoint SHA mismatch: " + str(path))
        specs.append({**item, "path": str(path)})
        print(f"WEIGHTS {item['key']}: SHA verified", flush=True)
    return specs


def setup_source(source, project):
    require((source / "ultralytics" / "__init__.py").is_file(), "Missing pinned yolo11_src: " + str(source))
    sys.path.insert(0, str(source))
    os.environ["YOLO_CONFIG_DIR"] = str(project / "comparison_reports" / "testdev_settings")
    import ultralytics
    require(Path(ultralytics.__file__).resolve().is_relative_to(source.resolve()), "Imported installed package, not requested source")
    require(ultralytics.__version__ == "8.3.0", "Expected project Ultralytics 8.3.0")
    return ultralytics


def environment(source, project):
    ultralytics = setup_source(source, project)
    import torch
    source_sha = hashlib.sha256()
    for path in sorted((source / "ultralytics").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            source_sha.update((path.relative_to(source).as_posix() + "\0" + official.sha256(path)).encode())
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                numpy=np.__version__, ultralytics=ultralytics.__version__, source=str(source),
                source_sha256=source_sha.hexdigest(), gpu=gpu)


def preflight(args):
    require(args.device == "0", "Production test-dev uses CUDA device 0; CPU is reserved for scoring/local tests")
    require(1 <= args.memory_cap_gib <= 8, "Inference cap must be between 1 and 8 GiB; no batch/precision fallback")
    specs = resolve_specs(args.project)
    records, dataset, images = load_records(args.dataset)
    payload = dict(revision=REVISION, dataset=dataset, images=images, models=specs,
                   protocol=inference_protocol(),
                   resources=dict(memory_cap_gib=args.memory_cap_gib, free_margin_gib=3, cpu_threads=2),
                   environment=environment(args.source, args.project),
                   package_sha256={p.name: official.sha256(p) for p in sorted(PACKAGE.iterdir())
                                   if p.is_file() and p.suffix in {".py", ".json", ".sh"}})
    return records, payload


def gate_path(args, mode):
    return args.project / "comparison_reports" / f"testdev_{mode}_passed.json"


def require_gates(args, payload, smoke=True):
    for mode in (["check", "smoke"] if smoke else ["check"]):
        path = gate_path(args, mode)
        require(path.is_file(), f"Missing {mode} gate; run run_testdev.sh {mode} first")
        receipt = read(path)
        require(receipt.get("status") == "passed" and receipt.get("binding") == digest(payload),
                f"{mode} gate does not match current code / weights / images / environment / cap; rerun {mode}")


def binding_for(payload, spec, smoke=False):
    # No absolute file paths in identity, so CPU rescoring can use downloaded files.
    return dict(revision=REVISION, key=spec["key"], checkpoint_sha256=spec["sha256"],
                dataset_annotation_sha256=payload["dataset"]["annotation_dimension_sha256"],
                dataset_image_sha256=payload["dataset"]["image_content_sha256"],
                protocol=payload["protocol"], source_sha256=payload["environment"]["source_sha256"],
                image_count=8 if smoke else EXPECTED_IMAGES, smoke_only=smoke)


def validate_manifest(payload):
    require(payload.get("revision") == REVISION, "Wrong input manifest revision")
    expected = vendor_check()
    require(len(payload.get("models", [])) == 6, "Expected all six fixed models in report")
    for saved, pinned in zip(payload["models"], expected):
        require(all(saved.get(key) == pinned[key] for key in ("key", "model", "seed", "sha256")),
                "Report models differ from the six val-selected checkpoints")
    require(payload.get("dataset", {}).get("images") == EXPECTED_IMAGES, "Not complete test-dev")
    images = payload.get("images", [])
    require(len(images) == EXPECTED_IMAGES and [r["id"] for r in images] == list(range(1, EXPECTED_IMAGES + 1)),
            "Missing or reordered image-ID table")
    require(len({r["stem"] for r in images}) == EXPECTED_IMAGES, "Duplicate image names")
    protocol = payload.get("protocol", {})
    require(protocol == inference_protocol(), "Report inference protocol differs")


def prediction_valid(out, expected):
    meta_path, predictions = out / "prediction_manifest.json", out / "predictions.json"
    if not meta_path.is_file() or not predictions.is_file():
        return False
    meta = read(meta_path)
    require(meta.get("binding") == expected, "Cached predictions belong to another split/model/protocol: " + str(out))
    require(meta.get("status") == "completed" and meta.get("seen_images") == expected["image_count"], "Incomplete predictions")
    require(meta.get("sha256") == official.sha256(predictions), "Corrupt/changed cached predictions: " + str(out))
    return True


class PredictionWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        self.handle = self.tmp.open("w", encoding="utf-8")
        self.handle.write("[")
        self.count = 0

    def add(self, row):
        self.handle.write(("," if self.count else "") + json.dumps(row, separators=(",", ":"), allow_nan=False))
        self.count += 1

    def complete(self):
        self.handle.write("]\n")
        self.handle.close()
        os.replace(self.tmp, self.path)

    def close(self):
        if not self.handle.closed:
            self.handle.close()


class NmsTimeoutGuard(logging.Handler):
    def emit(self, record):
        if "NMS time limit" in record.getMessage():
            raise RuntimeError("NMS timeout under contention; refusing partial detections. Retry after training releases GPU.")


def validate_model(yolo, spec):
    from ultralytics.nn.modules.head import Detect
    from spatial_task_head import SpatialTaskDetect, NEW_PARAMETERS
    head = yolo.model.model[-1]
    require(len(yolo.model.model) == 27 and head.nc == 10 and head.nl == 4 and head.reg_max == 16,
            "Not the pinned four-level E1 detector")
    require(list(head.f) == [25, 16, 19, 22] and head.stride.tolist() == [4., 8., 16., 32.], "Wrong feature sources/strides")
    require(yolo.names == {i - 1: name for i, name in official.CLASS_NAMES.items()}, "Class-index mapping mismatch")
    if spec["model"] == "E40":
        require(type(head) is SpatialTaskDetect and head.enabled and len(head.task_blocks) == 2,
                "E40 checkpoint is not the original enabled SpatialTaskDetect")
        require(sum(p.numel() for p in head.task_blocks.parameters()) == NEW_PARAMETERS, "E40 branch topology drift")
    else:
        require(type(head) is Detect and not hasattr(head, "task_blocks"), "E1 unexpectedly has custom branches")


def predict_model(args, payload, spec, out, smoke=False, cpu_test=False):
    """Called only by inference workers (or explicit local synthetic integration)."""
    setup_source(args.source, args.project)
    import torch
    import spatial_task_head  # exact class module required BEFORE loading E40
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import LOGGER
    torch.set_num_threads(2)
    configure_numeric(torch)
    if not cpu_test:
        require(torch.cuda.is_available(), "CUDA unavailable")
        free, total = torch.cuda.mem_get_info(0)
        cap = float(payload["resources"]["memory_cap_gib"])
        require(args.allow_busy_gpu or free / total >= .90, "GPU is busy; use --allow-busy-gpu for authorized concurrent evaluation")
        require(free >= (cap + 3) * GIB, f"Need cap+3GiB free ({cap + 3:.1f}); currently {free / GIB:.2f}GiB")
        torch.cuda.set_per_process_memory_fraction(cap * GIB / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        print(f"INFERENCE ONLY cap={cap:.1f}GiB free={free / GIB:.2f}GiB; batch4/FP32 unchanged", flush=True)
    image_table = payload["images"][:8] if smoke else payload["images"]
    require(len(image_table) == (8 if smoke else EXPECTED_IMAGES) or cpu_test, "Wrong inference image count")
    require(official.sha256(Path(spec["path"])) == spec["sha256"], "Worker checkpoint changed")
    model = YOLO(spec["path"], task="detect")
    validate_model(model, spec)
    by_path = {str(Path(row["path"]).resolve()): row for row in image_table}
    seen = set()
    shapes = {}
    out.mkdir(parents=True, exist_ok=True)
    writer = PredictionWriter(out / "predictions.json")
    class Predictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs):
            require(isinstance(preds, (tuple, list)) and len(preds) == 2 and len(preds[1]) == 4,
                    "Wrong detector output format")
            require(img.dtype == torch.float32 and preds[0].dtype == torch.float32, "FP32 inference was not preserved")
            require(torch.isfinite(preds[0]).all().item(), "Nonfinite decoded output")
            shape_key = "x".join(map(str, img.shape))
            shapes[shape_key] = shapes.get(shape_key, 0) + 1
            results = super().postprocess(preds, img, orig_imgs)
            require(len(results) == img.shape[0], "Lost images after NMS")
            for value in results:
                row = by_path[str(Path(value.path).resolve())]
                require(row["id"] not in seen, "Repeated inference image")
                require(tuple(value.orig_shape) == (row["height"], row["width"]), "Prediction coordinate dimensions mismatch")
                seen.add(row["id"])
                for x1, y1, x2, y2, score, cls in value.boxes.data.cpu().tolist():
                    require(all(math.isfinite(v) for v in (x1,y1,x2,y2,score,cls)), "Nonfinite final boxes")
                    require(cls == int(cls) and 0 <= cls < 10 and 0 <= score <= 1, "Invalid class/score")
                    require(0 <= x1 <= x2 <= row["width"] and 0 <= y1 <= y2 <= row["height"], "Boxes not in original image coordinates")
                    writer.add(dict(image_id=row["id"], category_id=int(cls)+1,
                                    bbox=[x1,y1,max(0,x2-x1),max(0,y2-y1)], score=score))
            return results
    device = "cpu" if cpu_test else args.device
    options = dict(**PREDICTION_OPTIONS, device=device)
    predictor = Predictor(overrides={**model.overrides, **options, "mode": "predict"}, _callbacks=model.callbacks)
    guard = NmsTimeoutGuard()
    LOGGER.addHandler(guard)
    started = time.monotonic()
    try:
        for start in range(0, len(image_table), 4):
            chunk = image_table[start:start+4]
            # stream is a predict() argument, NEVER a get_cfg override.
            values = model.predict([row["path"] for row in chunk], batch=len(chunk),
                                   predictor=predictor, stream=False, **options)
            del values
            done = start + len(chunk)
            if start == 0 or done % 40 == 0 or done == len(image_table):
                elapsed = time.monotonic() - started
                eta = elapsed / done * (len(image_table) - done)
                peak = torch.cuda.max_memory_reserved(0) / GIB if not cpu_test else 0
                print(f"{spec['key']} {done}/{len(image_table)} elapsed={elapsed:.1f}s ETA={eta:.1f}s peak_reserved={peak:.2f}GiB {now()}", flush=True)
        require(seen == {row["id"] for row in image_table}, "Incomplete inference coverage")
        writer.complete()
        peak = torch.cuda.max_memory_reserved(0) / GIB if not cpu_test else 0
        result = dict(status="completed", binding=binding_for(payload, spec, smoke), seen_images=len(seen),
                      predictions=writer.count, sha256=official.sha256(out / "predictions.json"),
                      actual_tensor_shapes=shapes, peak_reserved_gib=peak, elapsed_seconds=time.monotonic()-started,
                      finished_at=now(), model_class=type(model.model.model[-1]).__name__, cuda_smoke=smoke and not cpu_test,
                      backend=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                                   cudnn_deterministic=torch.backends.cudnn.deterministic,
                                   cudnn_benchmark=torch.backends.cudnn.benchmark,
                                   matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
        write(out / "prediction_manifest.json", result)
        return result
    finally:
        LOGGER.removeHandler(guard)
        writer.close()
        model.predictor = None
        del predictor, model
        gc.collect()
        if not cpu_test:
            torch.cuda.empty_cache()


def spawn_inference(args, out, spec, smoke=False):
    command = [sys.executable, "-u", str(PACKAGE / "testdev.py"), "--mode", "infer",
               "--project", str(args.project), "--source", str(args.source),
               "--device", args.device, "--report", str(out), "--model-key", spec["key"]]
    if args.allow_busy_gpu:
        command.append("--allow-busy-gpu")
    if smoke:
        command.append("--smoke-worker")
    print(f"START {spec['key']} {'8-image smoke' if smoke else 'test-dev inference'} {now()}", flush=True)
    require(subprocess.run(command).returncode == 0, "Inference worker failed: " + spec["key"])
    # Child exit releases its CUDA context before CPU scoring / next model.
    print(f"END {spec['key']}; evaluator CUDA context released {now()}", flush=True)


def filter_for_score(records, raw):
    """Same class-0 filter as D27, without unrelated val FP diagnostics."""
    require(len(records) == len(raw), "Record/prediction length mismatch")
    filtered, removed = [], 0
    for i, (record, detections) in enumerate(zip(records, raw), 1):
        keep = np.array([official.official_region_coverage(row[:4], record.ignore_rects,
                        record.width, record.height) < .5 for row in detections], dtype=bool)
        removed += int(np.count_nonzero(~keep))
        filtered.append(detections[keep, :6].copy())
        if i % 200 == 0 or i == len(records):
            print(f"CPU ignore filter {i}/{len(records)}", flush=True)
    return filtered, dict(raw_predictions=sum(len(r) for r in raw), removed_by_class0=removed,
                          kept_predictions=sum(len(r) for r in filtered))


def score_one(records, payload, spec, out):
    binding = binding_for(payload, spec)
    require(prediction_valid(out, binding), "Missing complete predictions for " + spec["key"])
    score_path = out / "official_metrics.json"
    prediction_sha = official.sha256(out / "predictions.json")
    if score_path.is_file():
        value = read(score_path)
        require(value.get("binding") == binding and value.get("prediction_sha256") == prediction_sha,
                "Saved official score identity mismatch")
        if value.get("status") == "completed":
            print("Reusing verified CPU score: " + spec["key"], flush=True)
            return value
    print("CPU SCORE " + spec["key"] + " " + now(), flush=True)
    start = time.monotonic()
    raw = official.load_predictions(out / "predictions.json", len(records))
    filtered, audit = filter_for_score(records, raw)
    metrics = official.official_evaluate(records, filtered)  # unchanged D27 port
    value = dict(status="completed", binding=binding, prediction_sha256=prediction_sha,
                 official=metrics, ignore=audit, cpu_seconds=time.monotonic()-start, finished_at=now())
    write(score_path, value)
    print(f"{spec['key']} AP={metrics['AP50_95_percent']:.4f} AP50={metrics['AP50_percent']:.4f} AP75={metrics['AP75_percent']:.4f}", flush=True)
    del raw, filtered
    gc.collect()
    return value


def save_summary(out, payload, results):
    paired = []
    for seed in (1, 2, 3):
        base = results[f"e1_seed{seed}"]["official"]
        candidate = results[f"e40_seed{seed}"]["official"]
        paired.append(dict(seed=seed, delta_pp={key: candidate[key] - base[key] for key in KEYS}))
    stats = {}
    for model in ("e1", "e40"):
        stats[model] = {key: dict(mean=statistics.mean(results[f"{model}_seed{s}"]["official"][key] for s in (1,2,3)),
                                sample_std=statistics.stdev(results[f"{model}_seed{s}"]["official"][key] for s in (1,2,3)))
                        for key in KEYS}
    stats["paired_delta_pp"] = {key: dict(mean=statistics.mean(p["delta_pp"][key] for p in paired),
                                        sample_std=statistics.stdev(p["delta_pp"][key] for p in paired)) for key in KEYS}
    summary = dict(status="completed", revision=REVISION, split="test-dev", images=EXPECTED_IMAGES,
                   input_manifest_sha256=official.sha256(out / "input_manifest.json"),
                   models=results, paired=paired, statistics=stats, finished_at=now(),
                   boundary="Val-selected weights; no test selection, ensemble, statistical significance or small-object AP claim.")
    write(out / "summary.json", summary)
    lines = ["# E1 / E40 test-dev paired three-seed evaluation", "", f"Finished: {summary['finished_at']}",
             "", "1610 test-dev images; frozen D27 MATLAB-v1.0.4-style scorer; AP values in percent, deltas in pp.",
             "", "| Seed | E1 AP | E40 AP | delta AP | delta AP50 | delta AP75 |", "|---|---:|---:|---:|---:|---:|"]
    for p in paired:
        seed, d = p["seed"], p["delta_pp"]
        lines.append(f"| {seed} | {results[f'e1_seed{seed}']['official'][KEYS[0]]:.4f} | {results[f'e40_seed{seed}']['official'][KEYS[0]]:.4f} | {d[KEYS[0]]:+.4f} | {d[KEYS[1]]:+.4f} | {d[KEYS[2]]:+.4f} |")
    lines += ["", "Mean +/- sample standard deviation across three seeds:", ""]
    for label in ("e1", "e40", "paired_delta_pp"):
        item = stats[label][KEYS[0]]
        lines.append(f"- {label}: {item['mean']:.4f} +/- {item['sample_std']:.4f}")
    lines += ["", "Per-class AP/AR and provenance: each model's official_metrics.json and prediction_manifest.json.",
              "This is a held-out evaluation, not permission to tune repeatedly on test-dev.",
              "No training was started, checkpoint changed, or automatic shutdown requested."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(out / "COMPLETED.json", dict(status="completed", finished_at=now(), summary_sha256=official.sha256(out / "summary.json")))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["check", "smoke", "verify-gates", "run", "score", "infer"])
    parser.add_argument("--project", type=Path, default=Path("/root/autodl-tmp/yolo11_project"))
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-tmp/yolo11_src"))
    parser.add_argument("--dataset", type=Path, default=Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019-DET-test-dev"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--memory-cap-gib", type=float, default=4.)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--report", type=Path, help="Existing report for resumable run / CPU-only score")
    parser.add_argument("--model-key", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("project", "source", "dataset"):
        setattr(args, name, getattr(args, name).resolve())
    if args.report:
        args.report = args.report.resolve()
    return args


def main():
    args = parse_args()
    if args.mode == "infer":
        require(args.report is not None and args.model_key is not None, "Internal worker requires report/model-key")
        payload = read(args.report / "input_manifest.json")
        validate_manifest(payload)
        spec = next(s for s in payload["models"] if s["key"] == args.model_key)
        predict_model(args, payload, spec, args.report / spec["key"], args.smoke_worker)
        return
    # CPU-only recovery needs the unchanged scorer, dataset and saved predictions,
    # but does not need CUDA, the six .pt files, or a training environment.
    if args.mode == "score":
        require(args.report is not None, "CPU score requires --report /absolute/report/directory")
        vendor_check()
        payload = read(args.report / "input_manifest.json")
        validate_manifest(payload)
        records, dataset, _ = load_records(args.dataset)
        for key in ("annotation_dimension_sha256", "image_content_sha256"):
            require(dataset[key] == payload["dataset"][key], "Scoring dataset differs: " + key)
        results = {s["key"]: score_one(records, payload, s, args.report / s["key"]) for s in payload["models"]}
        save_summary(args.report, payload, results)
        print(f"CPU scoring COMPLETED: {args.report / 'REPORT.md'}", flush=True)
        return
    records, payload = preflight(args)
    if args.mode == "check":
        from test_testdev import run_tests
        run_tests()
        gate = dict(status="passed", binding=digest(payload), checked_at=now(), cuda_smoke=False)
        write(gate_path(args, "check"), gate)
        print("TEST-DEV CHECK PASSED (CPU structure/scorer; CUDA smoke still required)", flush=True)
        return
    require_gates(args, payload, smoke=args.mode != "smoke")
    if args.mode == "verify-gates":
        print("TEST-DEV CHECK + SIX-WEIGHT CUDA SMOKE RECEIPTS VALID", flush=True)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.report or args.project / "comparison_reports" / f"TESTDEV_E1_E40_{args.mode.upper()}_{stamp}"
    if args.report:
        require(args.mode == "run" and out.is_dir(), "--report is only for existing run recovery")
        require(digest(read(out / "input_manifest.json")) == digest(payload), "Resume report identity differs from current input")
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / "input_manifest.json", payload)
    print(f"TEST-DEV {args.mode} START {now()} REPORT={out}", flush=True)
    try:
        smoke = args.mode == "smoke"
        for spec in payload["models"]:
            model_out = out / spec["key"]
            if prediction_valid(model_out, binding_for(payload, spec, smoke)):
                print("Reusing verified complete predictions: " + spec["key"], flush=True)
                continue
            spawn_inference(args, out, spec, smoke)
            require(prediction_valid(model_out, binding_for(payload, spec, smoke)), "Worker did not publish complete predictions")
        if smoke:
            write(gate_path(args, "smoke"), dict(status="passed", binding=digest(payload), report=str(out),
                                               checked_at=now(), cuda_smoke=True, images_per_weight=8))
            print("TEST-DEV SIX-WEIGHT CUDA SMOKE PASSED; no smoke AP used for conclusions", flush=True)
        else:
            print("All six GPU prediction passes complete; now CPU scoring only", flush=True)
            results = {s["key"]: score_one(records, payload, s, out / s["key"]) for s in payload["models"]}
            save_summary(out, payload, results)
            print(f"TEST-DEV COMPLETED {now()} REPORT={out / 'REPORT.md'}", flush=True)
    except BaseException:
        write(out / "failure.json", dict(status="failed_or_interrupted", at=now(), traceback=traceback.format_exc(),
                                        recovery=f"run_testdev.sh run --allow-busy-gpu --report {out}"))
        raise


if __name__ == "__main__":
    main()
