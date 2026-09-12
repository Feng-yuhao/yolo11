#!/usr/bin/env python3
"""D2 frozen E1 adaptive local-budget diagnostic.

This is a decision experiment, not a proposed publishable method.  It measures
whether a deployable prediction-guided proxy can retain most of the fixed
four-tile D1 gain with only one or two local crops.  Ground-truth oracle routes
are explicitly diagnostic upper bounds and are never reported as deployable
accuracy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import evaluate_e1_e5a_slicing as core


ROOT = Path("/root/autodl-tmp/yolo11_project")
REPORTS = ROOT / "comparison_reports"
WEIGHT = ROOT / "runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt"
D1_GLOB = "DIAG_SLICING_both_*/metrics.json"

ROUTER_CONF = 0.05
ROUTER_SMALL_EFFECTIVE_SIDE = 32.0
ROUTER_UNCERTAINTY_WEIGHT = 0.5
RECOVERY_GATE = 0.80
PARITY_TOLERANCE = 0.002
ROUTES = {
    "proxy_top1": ("proxy", 1),
    "proxy_top2": ("proxy", 2),
    "oracle_top1": ("oracle", 1),
    "oracle_top2": ("oracle", 2),
}
SCRIPT_REVISION = "prediction_guided_budget_diagnostic_d2_v1"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    core.atomic_json(out / "metrics.json", report)
    lines = [
        "D2 E1 adaptive local-budget diagnostic",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "All metrics are 0-1. Multiply by 100 for percentages.",
        "Oracle modes use validation ground truth for route selection and are diagnostic only.",
        "This uses converted YOLO labels, not official VisDrone ignore-region evaluation.",
        "",
    ]
    modes = report.get("models", {}).get("e1", {}).get("modes", {})
    for name, row in modes.items():
        overall = row.get("overall", {})
        lines.append(
            f"{name}: "
            + ", ".join(
                f"{key}={overall.get(key)}"
                for key in (
                    "mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small",
                    "AP_medium", "AP_large",
                )
            )
        )
    lines.extend([
        "", "[decisions]",
        json.dumps(report.get("decisions"), ensure_ascii=False, indent=2),
        "", "[full_metadata]",
        json.dumps(report, ensure_ascii=False, indent=2, default=core.json_default),
    ])
    core.atomic_text(out / "metrics.txt", "\n".join(lines) + "\n")


def latest_d1_reference() -> tuple[Path, dict]:
    completed = []
    for path in sorted(REPORTS.glob(D1_GLOB)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            value.get("status") == "completed"
            and value.get("is_full_diagnostic") is True
            and value.get("models", {}).get("e1", {}).get("modes", {}).get("combined")
        ):
            completed.append((path, value))
    if not completed:
        raise FileNotFoundError(
            "No completed full D1 DIAG_SLICING_both report was found."
        )
    return completed[-1]


def image_meta(gt: dict) -> dict[str, dict]:
    return {
        str(row["file_name"]): {
            "image_id": int(row["id"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in gt["images"]
    }


def assigned_tile(cx: float, cy: float, width: int, height: int) -> int:
    column = 0 if cx < width / 2 else 1
    row = 0 if cy < height / 2 else 1
    return row * 2 + column


def proxy_scores(detections, meta: dict) -> tuple[list[float], list[int]]:
    scores = [0.0] * 4
    counts = [0] * 4
    resize_scale = core.IMGSZ / max(meta["width"], meta["height"])
    for x1, y1, x2, y2, confidence, _cls in detections.tolist():
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        effective_side = math.sqrt(width * height) * resize_scale
        if confidence < ROUTER_CONF or effective_side >= ROUTER_SMALL_EFFECTIVE_SIDE:
            continue
        tile = assigned_tile(
            (x1 + x2) / 2, (y1 + y2) / 2, meta["width"], meta["height"]
        )
        evidence = math.sqrt(max(confidence, 0.0))
        scale_deficit = max(
            0.0, 1.0 - effective_side / ROUTER_SMALL_EFFECTIVE_SIDE
        )
        uncertainty = 4.0 * confidence * (1.0 - confidence)
        scores[tile] += evidence * (1.0 + scale_deficit) * (
            1.0 + ROUTER_UNCERTAINTY_WEIGHT * uncertainty
        )
        counts[tile] += 1
    return scores, counts


def oracle_scores(annotations: list[dict], meta: dict) -> tuple[list[float], list[int]]:
    scores = [0.0] * 4
    counts = [0] * 4
    resize_scale = core.IMGSZ / max(meta["width"], meta["height"])
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        effective_side = math.sqrt(width * height) * resize_scale
        if effective_side >= ROUTER_SMALL_EFFECTIVE_SIDE:
            continue
        tile = assigned_tile(
            x + width / 2, y + height / 2, meta["width"], meta["height"]
        )
        scale_deficit = max(
            0.0, 1.0 - effective_side / ROUTER_SMALL_EFFECTIVE_SIDE
        )
        scores[tile] += 1.0 + scale_deficit
        counts[tile] += 1
    return scores, counts


def top_indices(scores: list[float], count: int) -> list[int]:
    return sorted(range(4), key=lambda index: (-scores[index], index))[:count]


def center_covered(annotation: dict, selected: list[int], tiles: list[dict]) -> bool:
    x, y, width, height = annotation["bbox"]
    cx, cy = x + width / 2, y + height / 2
    return any(
        tiles[index]["x0"] <= cx <= tiles[index]["x1"]
        and tiles[index]["y0"] <= cy <= tiles[index]["y1"]
        for index in selected
    )


def tile_geometry(width: int, height: int) -> list[dict]:
    tile_width, xs = core.axis_tiles(width)
    tile_height, ys = core.axis_tiles(height)
    return [
        {
            "x0": x0, "y0": y0,
            "x1": x0 + tile_width, "y1": y0 + tile_height,
        }
        for y0 in ys for x0 in xs
    ]


def build_routes(
    images: list[Path], full: dict, gt: dict, meta_by_path: dict
) -> tuple[dict, dict]:
    annotations_by_image = {}
    for annotation in gt["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    selections = {name: {} for name in ROUTES}
    audit_rows = []
    coverage = {
        name: {"covered_tiny": 0, "total_tiny": 0} for name in ROUTES
    }
    agreements = {"top1_exact": 0, "top2_overlap_sum": 0}

    for path in images:
        key = str(path)
        meta = meta_by_path[key]
        annotations = annotations_by_image.get(meta["image_id"], [])
        p_scores, p_counts = proxy_scores(full[key], meta)
        o_scores, o_counts = oracle_scores(annotations, meta)
        p1, p2 = top_indices(p_scores, 1), top_indices(p_scores, 2)
        o1, o2 = top_indices(o_scores, 1), top_indices(o_scores, 2)
        route_row = {
            "proxy_top1": p1, "proxy_top2": p2,
            "oracle_top1": o1, "oracle_top2": o2,
        }
        for name, selected in route_row.items():
            selections[name][key] = selected
        agreements["top1_exact"] += int(p1 == o1)
        agreements["top2_overlap_sum"] += len(set(p2).intersection(o2))

        geometry = tile_geometry(meta["width"], meta["height"])
        resize_scale = core.IMGSZ / max(meta["width"], meta["height"])
        tiny_annotations = [
            ann for ann in annotations
            if math.sqrt(ann["bbox"][2] * ann["bbox"][3]) * resize_scale
            < ROUTER_SMALL_EFFECTIVE_SIDE
        ]
        for name, selected in route_row.items():
            coverage[name]["total_tiny"] += len(tiny_annotations)
            coverage[name]["covered_tiny"] += sum(
                center_covered(ann, selected, geometry) for ann in tiny_annotations
            )
        audit_rows.append({
            "image": key,
            "proxy_scores": p_scores,
            "proxy_candidate_counts": p_counts,
            "oracle_scores": o_scores,
            "oracle_tiny_counts": o_counts,
            "selections": route_row,
        })

    count = len(images)
    statistics = {
        "images": count,
        "proxy_oracle_top1_exact_fraction": agreements["top1_exact"] / count,
        "proxy_oracle_top2_mean_intersection_fraction": (
            agreements["top2_overlap_sum"] / (2 * count)
        ),
        "tiny_center_coverage": {
            name: {
                **row,
                "fraction": row["covered_tiny"] / row["total_tiny"]
                if row["total_tiny"] else None,
            }
            for name, row in coverage.items()
        },
    }
    return selections, {"statistics": statistics, "per_image": audit_rows}


def predict_route(
    model,
    images: list[Path],
    full: dict,
    selections: dict,
    route_name: str,
    report: dict,
):
    combined = {}
    core.torch.cuda.reset_peak_memory_stats(0)
    core.torch.cuda.synchronize()
    started = time.perf_counter()
    for image_index, path in enumerate(images, 1):
        image = core.cv2.imread(str(path), core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        tiles = core.make_tiles(image)
        selected = selections[str(path)]
        crops = [tiles[index]["image"] for index in selected]
        results = core.model_predict(model, crops, len(crops))
        if len(results) != len(selected):
            raise RuntimeError(f"{route_name} result count mismatch")
        mapped = [
            core.map_tile_detections(result, tiles[index])
            for result, index in zip(results, selected)
        ]
        local = core.torch.cat(mapped, dim=0) if mapped else core.empty_detections()
        local = core.nms_detections(local)
        candidates = core.torch.cat([full[str(path)], local], dim=0)
        combined[str(path)] = core.nms_detections(candidates)
        del image, tiles, crops, results, mapped, local, candidates
        if image_index == 1 or image_index % 50 == 0 or image_index == len(images):
            print(f"{route_name}: {image_index}/{len(images)}", flush=True)
            report["progress"] = {
                "stage": route_name, "images": image_index,
            }
            save(report)
    core.torch.cuda.synchronize()
    return combined, {
        "local_and_merge_seconds": time.perf_counter() - started,
        "local_crops_per_image": len(next(iter(selections.values()))),
        "local_forward_images": len(images) * len(next(iter(selections.values()))),
        "peak_memory": core.memory_peak(),
    }


def evaluate_compact(
    mode_name: str,
    predictions: dict,
    images: list[Path],
    path_ids: dict,
    gt: dict,
    names: dict,
) -> dict:
    print(f"Evaluating {mode_name} on CPU.", flush=True)
    detections = core.predictions_to_coco(predictions, images, path_ids)
    metrics = core.coco_metrics(gt, detections, names)
    metrics["fixed_operating_point"] = core.fixed_operating_point(gt, detections, names)
    metrics["prediction_count"] = len(detections)
    return metrics


def build_decisions(report: dict) -> dict:
    modes = report["models"]["e1"]["modes"]
    full = modes["full"]["overall"]
    reference = report["d1_all4_reference"]
    ref_full = reference["full"]
    ref_all4 = reference["combined_all4"]
    parity_keys = ("mAP50_95", "AP_small", "AR_small", "AP_medium", "AP_large")
    parity_delta = {key: full[key] - ref_full[key] for key in parity_keys}
    parity_passed = all(abs(value) <= PARITY_TOLERANCE for value in parity_delta.values())
    all4_gain = {
        key: ref_all4[key] - ref_full[key]
        for key in ("mAP50_95", "AP_small")
    }
    routes = {}
    for name in ROUTES:
        overall = modes[name]["overall"]
        gain = {
            key: overall[key] - full[key]
            for key in (
                "mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small",
                "AP_medium", "AP_large",
            )
        }
        recovery = {
            key: gain[key] / all4_gain[key] if all4_gain[key] > 0 else None
            for key in ("mAP50_95", "AP_small")
        }
        routes[name] = {
            "combined_minus_full": gain,
            "fraction_of_d1_all4_gain": recovery,
            "oracle_not_deployable": name.startswith("oracle_"),
        }
    proxy2 = routes["proxy_top2"]["fraction_of_d1_all4_gain"]
    proceed = bool(
        parity_passed
        and proxy2["mAP50_95"] >= RECOVERY_GATE
        and proxy2["AP_small"] >= RECOVERY_GATE
    )
    return {
        "full_reproduction": {
            "current_minus_d1_full": parity_delta,
            "absolute_tolerance": PARITY_TOLERANCE,
            "passed": parity_passed,
        },
        "d1_all4_gain": all4_gain,
        "routes": routes,
        "learned_router_gate": {
            "required_fraction_of_all4_gain": RECOVERY_GATE,
            "proxy_top2_passed": proceed,
            "proceed_to_learned_router": proceed,
            "note": (
                "Oracle accuracy is diagnostic only and never participates in the gate."
            ),
        },
    }


def save_summary_csv(report: dict) -> None:
    out = Path(report["report_dir"])
    core.save_csv(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-images", type=int, default=0, metavar="N")
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    if args.smoke_images < 0 or args.smoke_images > 548:
        parser.error("--smoke-images must be between 0 and 548")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_D2" if args.check_only else "SMOKE_D2" if args.smoke_images else "DIAG_D2"
    out = REPORTS / f"{prefix}_e1_adaptive_budget_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing",
        "purpose": "frozen E1 prediction-guided Top-K local crop budget diagnostic",
        "script_revision": SCRIPT_REVISION,
        "started_at": now(),
        "report_dir": str(out),
        "is_full_diagnostic": not args.check_only and not args.smoke_images,
        "protocol": {
            "weight": str(WEIGHT),
            "data": str(core.DATA),
            "imgsz": core.IMGSZ,
            "full_prediction_batch": core.PREDICT_BATCH,
            "grid": "2x2 with 20% overlap",
            "routes": ROUTES,
            "proxy_router": {
                "confidence_floor": ROUTER_CONF,
                "effective_small_side_px": ROUTER_SMALL_EFFECTIVE_SIDE,
                "uncertainty_weight": ROUTER_UNCERTAINTY_WEIGHT,
                "uses_ground_truth": False,
            },
            "oracle_router": {
                "effective_small_side_px": ROUTER_SMALL_EFFECTIVE_SIDE,
                "uses_ground_truth": True,
                "claim_policy": "diagnostic upper bound only; never a deployable result",
            },
            "recovery_gate": RECOVERY_GATE,
            "prediction_json_saved": False,
            "notes": [
                "No training and no checkpoint updates.",
                "D1 fixed four-tile combined metrics are loaded as the reference upper bound.",
                "Proxy routing uses only frozen full-image E1 detections.",
                "Oracle routing uses validation labels only to estimate router headroom.",
                "Metrics use converted YOLO labels, not official VisDrone ignored-region evaluation.",
            ],
        },
        "models": {"e1": {"modes": {}}},
    }
    save(report)
    started = time.perf_counter()
    success = False
    try:
        if core.SCRIPT_REVISION != "fixed_2x2_slicing_upper_bound_v2":
            raise RuntimeError(f"Unexpected D1 core revision: {core.SCRIPT_REVISION}")
        core.preflight(report, ["e1"], args.check_only)
        d1_path, d1 = latest_d1_reference()
        d1_modes = d1["models"]["e1"]["modes"]
        report["d1_all4_reference"] = {
            "path": str(d1_path),
            "script_revision": d1.get("script_revision"),
            "full": d1_modes["full"]["overall"],
            "combined_all4": d1_modes["combined"]["overall"],
        }
        images, path_ids, gt, names = core.prepare_dataset(report, args.smoke_images)
        model = core.YOLO(str(WEIGHT))
        before_hash = core.sha256(WEIGHT)
        report["models"]["e1"].update({
            "weight": str(WEIGHT),
            "sha256": before_hash,
            "structure": core.assert_model(model, names, "e1"),
        })
        report["status"] = "preflight_passed"
        save(report)
        if args.check_only:
            report.update(status="check_passed", finished_at=now())
            success = True
            print("D2 preflight passed. No prediction, training, or shutdown.", flush=True)
            return

        full, full_profile = core.predict_full(model, images, report, "e1")
        full_metrics = evaluate_compact("full", full, images, path_ids, gt, names)
        full_metrics["profile"] = full_profile
        report["models"]["e1"]["modes"]["full"] = full_metrics
        save(report)

        selections, routing_audit = build_routes(images, full, gt, image_meta(gt))
        core.atomic_json(out / "routing_audit.json", routing_audit)
        report["routing_statistics"] = routing_audit["statistics"]
        save(report)

        for route_name in ROUTES:
            report["status"] = f"predicting_{route_name}"
            save(report)
            combined, route_profile = predict_route(
                model, images, full, selections[route_name], route_name, report
            )
            metrics = evaluate_compact(
                route_name, combined, images, path_ids, gt, names
            )
            metrics["profile"] = {
                "seconds": full_profile["seconds"] + route_profile["local_and_merge_seconds"],
                "full_seconds": full_profile["seconds"],
                **route_profile,
                "forward_images": len(images) + route_profile["local_forward_images"],
                "peak_memory": {
                    key: max(
                        full_profile["peak_memory"][key],
                        route_profile["peak_memory"][key],
                    )
                    for key in ("allocated_GiB", "reserved_GiB")
                },
            }
            report["models"]["e1"]["modes"][route_name] = metrics
            del combined
            gc.collect()
            save(report)

        after_hash = core.sha256(WEIGHT)
        if after_hash != before_hash:
            raise RuntimeError("Frozen E1 checkpoint changed during D2 evaluation")
        report["models"]["e1"]["checkpoint_unchanged"] = True
        report["decisions"] = build_decisions(report) if not args.smoke_images else {
            "disabled": "Smoke subset metrics must not drive research decisions."
        }
        report.update(
            status="completed", finished_at=now(),
            total_seconds=time.perf_counter() - started,
        )
        save_summary_csv(report)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        save(report)
        success = True
        print(f"\nD2 completed. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(
            status="failed_or_interrupted", finished_at=now(),
            total_seconds=time.perf_counter() - started,
            error=traceback.format_exc(),
        )
        save(report)
        core.atomic_text(out / "error.log", report["error"])
        print(f"D2 failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            save(report)
        except Exception:
            pass
        enabled = bool(args.shutdown and not args.check_only and not args.smoke_images)
        try:
            core.request_shutdown(out, enabled, report.get("status", "unknown"))
        except Exception:
            if success:
                raise


if __name__ == "__main__":
    main()
