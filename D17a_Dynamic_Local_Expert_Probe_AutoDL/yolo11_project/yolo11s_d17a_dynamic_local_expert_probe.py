#!/usr/bin/env python3
"""D17a dynamic scale-specialized local expert 30-epoch probe.

E1 remains the full-view detector and D15a remains the frozen 1.5-crop router.
A separate copy of E1 is fine-tuned only on train-split crops selected by that
router. The backbone is frozen; only the local expert neck/head adapts to the
zoomed distribution. Final metrics always use the original 548-image val split
with full-view E1 plus routed local predictions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import diagnose_d16a_cvrc_oof as route
import evaluate_e1_e5a_slicing as core


ROOT = Path("/root/autodl-tmp/yolo11_project")
DATA = Path("/root/autodl-tmp/project/VisDrone2019/VisDrone2019.yaml")
E1 = ROOT / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
RUN_NAME = "d17a_local_expert_probe30_img1024_seed1"
RUN_DIR = ROOT / "runs" / RUN_NAME
GENERATED = ROOT / "generated_d17a_local_expert"
REPORTS = ROOT / "comparison_reports"
SCRIPT_REVISION = "d17a_dynamic_route_local_expert_freeze10_probe30_v1"

EPOCHS = 30
BATCH = 8
WORKERS = 8
FREEZE = 10
LR0 = 0.003
PRIMARY_BUDGET = 1.50
VISIBLE_FRACTION = 0.70
INTERNAL_MARGIN_FRACTION = 0.005
REFERENCE_TOLERANCE = 0.002
MAP_GATE = 0.002
APS_GATE = 0.003
APM_FLOOR = -0.005
MIN_FREE_GIB = 8.0


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    core.atomic_json(out / "metrics.json", report)
    core.atomic_text(out / "metrics.txt", report_text(report))


def report_text(report: dict) -> str:
    lines = [
        "D17a dynamic scale-specialized local expert probe",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "Full-view E1 and D15a routing are frozen. Only a separate local expert is trained.",
        "All final metrics use the original 548-image validation split.",
        "",
        "[modes]",
    ]
    for name, row in report.get("modes", {}).items():
        overall = row.get("overall", {})
        fixed = row.get("fixed_operating_point", {}).get("overall", {})
        lines.append(
            f"{name}: mAP50_95={overall.get('mAP50_95')}, AP_small={overall.get('AP_small')}, "
            f"AR_small={overall.get('AR_small')}, AP_medium={overall.get('AP_medium')}, "
            f"AP_large={overall.get('AP_large')}, TP={fixed.get('tp')}, FP={fixed.get('fp')}"
        )
    lines.extend([
        "", "[decision]", json.dumps(report.get("decision"), ensure_ascii=False, indent=2),
        "", "[dataset]", json.dumps(report.get("generated_dataset"), ensure_ascii=False, indent=2),
        "", "[full_metadata]", json.dumps(report, ensure_ascii=False, indent=2, default=core.json_default),
    ])
    return "\n".join(lines) + "\n"


def find_completed(pattern: str, required_branch: str | None = None) -> tuple[Path, dict]:
    rows = []
    for path in sorted(REPORTS.glob(pattern)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        branch = value.get("decision", value.get("decisions", {})).get("selected_branch")
        if value.get("status") == "completed" and (required_branch is None or branch == required_branch):
            rows.append((path, value))
    if not rows:
        raise FileNotFoundError(f"No completed report found: {pattern}")
    return rows[-1]


def split_images(dataset: dict, split: str, limit: int = 0) -> list[Path]:
    source = dataset[split]
    sources = source if isinstance(source, list) else [source]
    images = []
    for item in sources:
        path = Path(item)
        if path.is_dir():
            images.extend(
                child.resolve() for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in core.SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                value = line.strip()
                if value:
                    candidate = Path(value)
                    images.append((candidate if candidate.is_absolute() else path.parent / candidate).resolve())
        else:
            raise ValueError(f"Unsupported {split} source: {path}")
    images = sorted(set(images))
    if not images:
        raise RuntimeError(f"No images found for split {split}")
    return images[:limit] if limit else images


def read_labels(images: list[Path]) -> dict[str, list[dict]]:
    result = {}
    for image_path, label_value in zip(images, core.img2label_paths([str(path) for path in images])):
        label_path = Path(label_value)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        image = core.cv2.imread(str(image_path), core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        height, width = image.shape[:2]
        rows = []
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5:
                raise ValueError(f"Invalid label row in {label_path}")
            cls, xc, yc, bw, bh = values
            box_w, box_h = bw * width, bh * height
            rows.append({
                "class": int(cls),
                "box": [xc * width - box_w / 2, yc * height - box_h / 2,
                        xc * width + box_w / 2, yc * height + box_h / 2],
            })
        result[str(image_path)] = rows
    return result


def metadata(images: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in images:
        image = core.cv2.imread(str(path), core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {path}")
        height, width = image.shape[:2]
        rows[str(path)] = {"width": width, "height": height}
    return rows


def predict_full_simple(model, images: list[Path], report: dict, label: str) -> dict:
    predictions = {}
    for start in range(0, len(images), core.PREDICT_BATCH):
        chunk = images[start:start + core.PREDICT_BATCH]
        results = core.model_predict(model, [str(path) for path in chunk], len(chunk))
        for path, result in zip(chunk, results):
            predictions[str(path)] = core.result_tensor(result)
        del results
        done = min(start + core.PREDICT_BATCH, len(images))
        if start == 0 or done % 500 == 0 or done == len(images):
            print(f"{label}: {done}/{len(images)}", flush=True)
            report["progress"] = {"stage": label, "images": done}
            save(report)
    return predictions


def crop_labels(labels: list[dict], tile: dict) -> list[str]:
    crop_w, crop_h = tile["x1"] - tile["x0"], tile["y1"] - tile["y0"]
    margin = max(2.0, min(crop_w, crop_h) * INTERNAL_MARGIN_FRACTION)
    output = []
    for row in labels:
        x1, y1, x2, y2 = row["box"]
        original_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        ix1, iy1 = max(x1, tile["x0"]), max(y1, tile["y0"])
        ix2, iy2 = min(x2, tile["x1"]), min(y2, tile["y1"])
        visible_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if original_area <= 0 or visible_area / original_area < VISIBLE_FRACTION:
            continue
        lx1, ly1 = ix1 - tile["x0"], iy1 - tile["y0"]
        lx2, ly2 = ix2 - tile["x0"], iy2 - tile["y0"]
        if tile["x0"] > 0 and lx1 <= margin:
            continue
        if tile["y0"] > 0 and ly1 <= margin:
            continue
        if tile["x1"] < tile["full_w"] and lx2 >= crop_w - margin:
            continue
        if tile["y1"] < tile["full_h"] and ly2 >= crop_h - margin:
            continue
        width, height = lx2 - lx1, ly2 - ly1
        if width < 1.0 or height < 1.0:
            continue
        xc, yc = (lx1 + lx2) / 2 / crop_w, (ly1 + ly2) / 2 / crop_h
        output.append(f"{row['class']} {xc:.8f} {yc:.8f} {width/crop_w:.8f} {height/crop_h:.8f}")
    return output


def dataset_signature(images: list[Path], selections: dict, labels: dict) -> str:
    digest = hashlib.sha256()
    for path in images:
        digest.update(str(path).encode())
        digest.update(bytes(selections[str(path)]))
        digest.update(json.dumps(labels[str(path)], sort_keys=True).encode())
    digest.update(SCRIPT_REVISION.encode())
    return digest.hexdigest()


def materialize_split(split: str, images: list[Path], labels: dict, selections: dict, target: Path, report: dict) -> dict:
    image_dir, label_dir = target / "images" / split, target / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    crops, objects, empty = 0, 0, 0
    for index, path in enumerate(images, 1):
        image = core.cv2.imread(str(path), core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {path}")
        tiles = core.make_tiles(image)
        token = hashlib.sha1(str(path).encode()).hexdigest()[:10]
        for tile_id in selections[str(path)]:
            tile = tiles[tile_id]
            stem = f"{path.stem}_{token}_t{tile_id}"
            image_path, label_path = image_dir / f"{stem}.jpg", label_dir / f"{stem}.txt"
            rows = crop_labels(labels[str(path)], tile)
            if not core.cv2.imwrite(str(image_path), tile["image"], [int(core.cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"Failed to write {image_path}")
            label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            crops += 1
            objects += len(rows)
            empty += int(not rows)
        if index == 1 or index % 500 == 0 or index == len(images):
            print(f"D17a materialize {split}: {index}/{len(images)}", flush=True)
            report["progress"] = {"stage": f"materialize_{split}", "images": index}
            save(report)
    return {"source_images": len(images), "crops": crops, "objects": objects, "empty_crops": empty}


def build_local_dataset(model, dataset: dict, smoke: bool, report: dict) -> Path:
    target = GENERATED / ("smoke" if smoke else "formal")
    manifest_path = target / "manifest.json"
    limits = {"train": 24 if smoke else 0, "val": 12 if smoke else 0}
    all_rows = {}
    for split in ("train", "val"):
        images = split_images(dataset, split, limits[split])
        labels = read_labels(images)
        meta = metadata(images)
        full = predict_full_simple(model, images, report, f"D17a route {split}")
        proxy_rows = route.build_proxy_rows(images, full, meta)
        selections, budget = route.allocate_budget(proxy_rows, PRIMARY_BUDGET)
        signature = dataset_signature(images, selections, labels)
        all_rows[split] = {
            "images": images, "labels": labels, "selections": selections,
            "budget": budget, "signature": signature,
        }
        del full
        gc.collect()
    expected = {split: all_rows[split]["signature"] for split in ("train", "val")}
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signatures") == expected and old.get("status") == "completed":
            report["generated_dataset"] = old
            print(f"Reusing verified local dataset: {target}", flush=True)
            return target / "dataset.yaml"
        raise RuntimeError(f"Existing generated dataset does not match D17a: {target}")
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty generated dataset: {target}")
    target.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val"):
        row = all_rows[split]
        stats[split] = materialize_split(split, row["images"], row["labels"], row["selections"], target, report)
        stats[split]["budget"] = row["budget"]
    normalized_names = core.normalize_names(dataset["names"])
    names = [normalized_names[index] for index in range(10)]
    yaml_text = "path: " + str(target) + "\ntrain: images/train\nval: images/val\nnames:\n"
    yaml_text += "".join(f"  {index}: {name}\n" for index, name in enumerate(names))
    (target / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    manifest = {
        "status": "completed", "script_revision": SCRIPT_REVISION,
        "created_at": now(), "signatures": expected, "stats": stats,
        "visible_fraction": VISIBLE_FRACTION,
        "internal_margin_fraction": INTERNAL_MARGIN_FRACTION,
        "uses_validation_labels_for_training": False,
    }
    core.atomic_json(manifest_path, manifest)
    report["generated_dataset"] = manifest
    return target / "dataset.yaml"


def local_predictions(model, images: list[Path], selections: dict, report: dict, label: str) -> dict:
    predictions = {}
    for image_index, path in enumerate(images, 1):
        image = core.cv2.imread(str(path), core.cv2.IMREAD_COLOR)
        tiles = core.make_tiles(image)
        selected = selections[str(path)]
        mapped = []
        if selected:
            results = core.model_predict(model, [tiles[index]["image"] for index in selected], len(selected))
            mapped = [core.map_tile_detections(result, tiles[index]) for result, index in zip(results, selected)]
            del results
        tensors = [value for value in mapped if len(value)]
        predictions[str(path)] = core.nms_detections(core.torch.cat(tensors, dim=0)) if tensors else core.empty_detections()
        if image_index == 1 or image_index % 100 == 0 or image_index == len(images):
            print(f"{label}: {image_index}/{len(images)}", flush=True)
            report["progress"] = {"stage": label, "images": image_index}
            save(report)
    return predictions


def merged_predictions(images, full: dict, local: dict) -> dict:
    return {
        str(path): core.nms_detections(core.torch.cat([full[str(path)], local[str(path)]], dim=0))
        for path in images
    }


def evaluate(name, predictions, images, path_ids, gt, names) -> dict:
    print(f"D17a evaluating {name}", flush=True)
    detections = core.predictions_to_coco(predictions, images, path_ids)
    metrics = core.coco_metrics(gt, detections, names)
    metrics["fixed_operating_point"] = core.fixed_operating_point(gt, detections, names)
    metrics["prediction_count"] = len(detections)
    return metrics


def train_local_expert(data_yaml: Path, smoke: bool, report: dict) -> Path:
    name = "smoke2_d17a_local_expert" if smoke else RUN_NAME
    run_dir = ROOT / "runs" / name
    best = run_dir / "weights/best.pt"
    if best.is_file():
        print(f"Reusing existing local expert: {best}", flush=True)
        return best
    if run_dir.exists():
        raise RuntimeError(f"Incomplete run directory exists; inspect it before retry: {run_dir}")
    expert = core.YOLO(str(E1))
    before = core.sha256(E1)
    report["status"] = "training_local_expert"
    save(report)
    expert.train(
        data=str(data_yaml), epochs=1 if smoke else EPOCHS, imgsz=core.IMGSZ,
        batch=BATCH, device=0, workers=WORKERS, project=str(ROOT / "runs"),
        name=name, exist_ok=False, pretrained=True, optimizer="SGD",
        lr0=LR0, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=1.0 if smoke else 2.0, warmup_momentum=0.8,
        warmup_bias_lr=0.1, cos_lr=True, patience=0, amp=True,
        seed=1, deterministic=True, freeze=FREEZE, val=True, plots=True,
        conf=0.001, iou=0.70, max_det=500, close_mosaic=0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.05, scale=0.20, shear=0.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )
    del expert
    core.clear_gpu()
    if core.sha256(E1) != before:
        raise RuntimeError("Frozen E1 changed while training local expert")
    if not best.is_file():
        raise FileNotFoundError(best)
    return best


def delta(one: dict, zero: dict) -> dict:
    keys = ("mAP50_95", "mAP50", "mAP75", "AR_all", "AP_small", "AR_small", "AP_medium", "AR_medium", "AP_large", "AR_large")
    return {key: one[key] - zero[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke2", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_D17A" if args.check_only else "SMOKE2_D17A" if args.smoke2 else "d17a"
    out = REPORTS / f"{prefix}_local_expert_probe_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing", "script_revision": SCRIPT_REVISION,
        "started_at": now(), "report_dir": str(out),
        "is_formal_probe": not args.check_only and not args.smoke2,
        "protocol": {
            "full_detector": str(E1), "router": "D15a proxy 1.5 crops/image",
            "local_initialization": str(E1), "epochs": EPOCHS, "freeze": FREEZE,
            "imgsz": core.IMGSZ, "batch": BATCH, "mosaic": 0.0,
            "validation_labels_used_for_training": False,
            "automatic_shutdown": False,
        },
        "modes": {},
    }
    save(report)
    started = time.perf_counter()
    try:
        core.preflight(report, ["e1"], args.check_only)
        d15b_path, d15b = find_completed("DIAG_D15B_e1_correction_fusion_headroom_*/metrics.json")
        d16b_path, d16b = find_completed("DIAG_D16B_e1_onesided_cvrc_*/metrics.json", "stop_cvrc_keep_d15a")
        report["references"] = {"d15b": str(d15b_path), "d16b": str(d16b_path)}
        free_gib = shutil.disk_usage("/root/autodl-tmp").free / 1024 ** 3
        report["preflight"].update({"free_data_disk_GiB": free_gib, "e1_sha256": core.sha256(E1)})
        if not args.check_only and free_gib < MIN_FREE_GIB:
            raise RuntimeError(f"D17a needs at least {MIN_FREE_GIB} GiB free, found {free_gib:.2f}")
        test_image = core.np.zeros((100, 160, 3), dtype=core.np.uint8)
        test_tiles = core.make_tiles(test_image)
        test_labels = crop_labels([{"class": 0, "box": [50, 30, 70, 50]}], test_tiles[0])
        if not test_labels:
            raise RuntimeError("Synthetic crop-label check failed")
        report["status"] = "preflight_passed"
        save(report)
        if args.check_only:
            report.update(status="check_passed", finished_at=now(), total_seconds=time.perf_counter() - started)
            save(report)
            print("D17a source/report/disk/crop-label checks passed. No training or shutdown.", flush=True)
            return
        dataset = core.check_det_dataset(str(DATA), autodownload=False)
        expected_names = core.normalize_names(dataset["names"])
        if [expected_names[index] for index in range(10)] != core.EXPECTED_CLASSES:
            raise RuntimeError("VisDrone class order changed")
        e1_model = core.YOLO(str(E1))
        core.assert_model(e1_model, expected_names, "e1")
        local_yaml = build_local_dataset(e1_model, dataset, args.smoke2, report)
        del e1_model
        core.clear_gpu()
        local_best = train_local_expert(local_yaml, args.smoke2, report)
        report["local_expert"] = {"best": str(local_best), "sha256": core.sha256(local_best)}
        images, path_ids, gt, names = core.prepare_dataset(report, 12 if args.smoke2 else 0)
        meta = route.image_meta(gt)
        e1_model = core.YOLO(str(E1))
        expert_model = core.YOLO(str(local_best))
        core.assert_model(e1_model, names, "e1")
        core.assert_model(expert_model, names, "local_expert")
        full = predict_full_simple(e1_model, images, report, "D17a val full E1")
        proxy_rows = route.build_proxy_rows(images, full, meta)
        selections, budget_meta = route.allocate_budget(proxy_rows, PRIMARY_BUDGET)
        report["val_router"] = budget_meta
        e1_local = local_predictions(e1_model, images, selections, report, "D17a val local E1")
        expert_local = local_predictions(expert_model, images, selections, report, "D17a val local expert")
        standard = merged_predictions(images, full, e1_local)
        proposed = merged_predictions(images, full, expert_local)
        report["modes"]["d15a_e1_local"] = evaluate("d15a_e1_local", standard, images, path_ids, gt, names)
        report["modes"]["d17a_specialized_local"] = evaluate("d17a_specialized_local", proposed, images, path_ids, gt, names)
        gain = delta(report["modes"]["d17a_specialized_local"]["overall"], report["modes"]["d15a_e1_local"]["overall"])
        if args.smoke2:
            report["decision"] = {"disabled": "Smoke metrics cannot drive decisions"}
        else:
            reference = d15b["models"]["e1"]["modes"]["proxy_b1p50_standard"]["overall"]
            anchor = report["modes"]["d15a_e1_local"]["overall"]
            parity = {key: anchor[key] - reference[key] for key in ("mAP50_95", "AP_small", "AR_small", "AP_medium", "AP_large")}
            parity_pass = all(abs(value) <= REFERENCE_TOLERANCE for value in parity.values())
            passed = bool(parity_pass and gain["mAP50_95"] >= MAP_GATE and gain["AP_small"] >= APS_GATE and gain["AP_medium"] >= APM_FLOOR)
            report["decision"] = {
                "d15a_reproduction": {"delta": parity, "tolerance": REFERENCE_TOLERANCE, "passed": parity_pass},
                "d17a_minus_d15a": gain,
                "gate": {"mAP50_95": MAP_GATE, "AP_small": APS_GATE, "AP_medium_minimum": APM_FLOOR},
                "passed": passed,
                "selected_branch": "extend_local_expert_training" if passed else "stop_local_expert_keep_d15a",
            }
        if core.sha256(E1) != report["preflight"]["e1_sha256"]:
            raise RuntimeError("Frozen E1 changed during D17a")
        report["e1_checkpoint_unchanged"] = True
        report.update(status="completed", finished_at=now(), total_seconds=time.perf_counter() - started)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        save(report)
        print(f"D17a completed. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(status="failed_or_interrupted", finished_at=now(), total_seconds=time.perf_counter() - started, error=traceback.format_exc())
        save(report)
        core.atomic_text(out / "error.log", report["error"])
        print(f"D17a failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        core.clear_gpu()
        try:
            core.atomic_json(out / "shutdown_status.json", {"time": now(), "enabled": False, "status": "disabled_by_user_policy", "reason": report.get("status")})
            save(report)
        except Exception:
            pass
        print("Automatic shutdown is disabled; the instance remains running.", flush=True)


if __name__ == "__main__":
    main()
