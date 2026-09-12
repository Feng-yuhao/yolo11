#!/usr/bin/env python3
"""D15b frozen-E1 correction-selection and fusion-headroom diagnostic.

The experiment performs no training and never changes E1.  It reuses one full
prediction and four cached local-tile predictions per validation image, then
separates two questions that D15a could not answer:

1. Selection headroom: would a ground-truth-only correction-utility oracle pick
   meaningfully better tiles than the deployable D15a proxy at the same budget?
2. Fusion headroom: how much could be gained if a verifier removed invalid and
   duplicate local candidates while retaining their original confidence scores?

Every oracle is explicitly non-deployable and is used only to select the next
research branch.  The deployable anchors never use ground truth for routing.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
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
D15A_GLOB = "DIAG_D15A_e1_dynamic_zoom_budget_*/metrics.json"

ROUTER_CONF = 0.05
ROUTER_SMALL_EFFECTIVE_SIDE = 32.0
ROUTER_UNCERTAINTY_WEIGHT = 0.5
PRIMARY_BUDGET = 1.50

UTILITY_CONFS = (0.10, 0.25)
UTILITY_IOUS = (0.50, 0.75)
SMALL_AREA = 32.0**2
SMALL_TP_WEIGHT = 4.0
SMALL_FP_WEIGHT = 1.0
NONSMALL_FP_WEIGHT = 0.25
ORACLE_VALIDITY_IOU = 0.50

REFERENCE_TOLERANCE = 0.002
SELECTION_MAP_GATE = 0.002
SELECTION_APS_GATE = 0.005
FUSION_MAP_GATE = 0.003
FUSION_APS_GATE = 0.005

SCRIPT_REVISION = "d15b_correction_selection_fusion_headroom_v1"
EXPECTED_CORE_REVISION = "fixed_2x2_slicing_upper_bound_v2"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save(report: dict) -> None:
    report["updated_at"] = now()
    out = Path(report["report_dir"])
    core.atomic_json(out / "metrics.json", report)
    core.atomic_text(out / "metrics.txt", report_text(report))


def report_text(report: dict) -> str:
    lines = [
        "D15b frozen-E1 correction-selection and fusion-headroom diagnostic",
        f"status: {report.get('status')}",
        f"revision: {SCRIPT_REVISION}",
        f"started_at: {report.get('started_at')}",
        f"finished_at: {report.get('finished_at')}",
        "",
        "All AP/AR values are 0-1; multiply by 100 for percentage points.",
        "Modes containing 'oracle' use validation ground truth and are NOT deployable.",
        "Oracle results are branch-selection upper bounds, not method accuracy.",
        "This uses converted YOLO labels, not official VisDrone ignore handling.",
        "",
        "[modes]",
    ]
    modes = report.get("models", {}).get("e1", {}).get("modes", {})
    for name, row in modes.items():
        overall = row.get("overall", {})
        profile = row.get("profile", {})
        lines.append(
            f"{name}: crops/image={profile.get('local_crops_per_image')}, "
            f"forward_multiplier={profile.get('forward_image_multiplier')}, "
            + ", ".join(
                f"{key}={overall.get(key)}"
                for key in (
                    "mAP50_95", "mAP50", "mAP75", "AP_small", "AR_small",
                    "AP_medium", "AP_large",
                )
            )
        )
    lines.extend([
        "",
        "[decisions]",
        json.dumps(report.get("decisions"), ensure_ascii=False, indent=2),
        "",
        "[full_metadata]",
        json.dumps(report, ensure_ascii=False, indent=2, default=core.json_default),
    ])
    return "\n".join(lines) + "\n"


def latest_d15a_reference() -> tuple[Path, dict]:
    completed = []
    for path in sorted(REPORTS.glob(D15A_GLOB)):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        modes = value.get("models", {}).get("e1", {}).get("modes", {})
        required = {"full", "fixed_top1", "fixed_top2", "adaptive_b1p50"}
        if (
            value.get("status") == "completed"
            and value.get("is_full_diagnostic") is True
            and required.issubset(modes)
        ):
            completed.append((path, value))
    if not completed:
        raise FileNotFoundError("No completed full D15a report was found")
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


def annotation_records(gt: dict) -> dict[str, dict]:
    path_by_id = {int(row["id"]): str(row["file_name"]) for row in gt["images"]}
    rows = {path: [] for path in path_by_id.values()}
    for annotation in gt["annotations"]:
        x, y, width, height = map(float, annotation["bbox"])
        rows[path_by_id[int(annotation["image_id"])]].append({
            "id": int(annotation["id"]),
            "category_id": int(annotation["category_id"]),
            "box": core.np.asarray([x, y, x + width, y + height], dtype=core.np.float32),
            "area": float(annotation["area"]),
            "is_small": float(annotation["area"]) < SMALL_AREA,
        })
    indexed = {}
    for path, annotations in rows.items():
        by_class = {}
        for annotation in annotations:
            by_class.setdefault(annotation["category_id"], []).append(annotation)
        indexed[path] = {
            "rows": annotations,
            "by_class": by_class,
            "small_ids": {ann["id"] for ann in annotations if ann["is_small"]},
        }
    return indexed


def assigned_tile(cx: float, cy: float, width: int, height: int) -> int:
    return (0 if cy < height / 2 else 1) * 2 + (0 if cx < width / 2 else 1)


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
        scale_deficit = max(0.0, 1.0 - effective_side / ROUTER_SMALL_EFFECTIVE_SIDE)
        uncertainty = 4.0 * confidence * (1.0 - confidence)
        scores[tile] += evidence * (1.0 + scale_deficit) * (
            1.0 + ROUTER_UNCERTAINTY_WEIGHT * uncertainty
        )
        counts[tile] += 1
    return scores, counts


def build_proxy_rows(images: list[Path], full: dict, meta_by_path: dict) -> list[dict]:
    rows = []
    for path in images:
        key = str(path)
        scores, counts = proxy_scores(full[key], meta_by_path[key])
        order = sorted(range(4), key=lambda index: (-scores[index], index))
        rows.append({
            "image": key,
            "tile_scores": scores,
            "candidate_counts": counts,
            "ranked_tiles": order,
            "ranked_scores": [scores[index] for index in order],
        })
    return rows


def allocate_proxy_budget(
    rows: list[dict], budget: float
) -> tuple[dict[str, list[int]], dict]:
    requested = int(round(budget * len(rows)))
    marginals = []
    for row in rows:
        for rank in (0, 1):
            marginals.append({
                "score": float(row["ranked_scores"][rank]),
                "rank": rank,
                "image": row["image"],
            })
    marginals.sort(key=lambda item: (-item["score"], item["rank"], item["image"]))
    ranks_by_image: dict[str, set[int]] = {}
    for item in marginals[:requested]:
        ranks_by_image.setdefault(item["image"], set()).add(item["rank"])
    selections = {}
    distribution = {"0": 0, "1": 0, "2": 0}
    for row in rows:
        ranks = ranks_by_image.get(row["image"], set())
        if 1 in ranks and 0 not in ranks:
            raise RuntimeError("Proxy budget violated the rank-prefix constraint")
        count = len(ranks)
        selections[row["image"]] = row["ranked_tiles"][:count]
        distribution[str(count)] += 1
    return selections, {
        "target_crops_per_image": budget,
        "actual_total_crops": sum(map(len, selections.values())),
        "actual_crops_per_image": sum(map(len, selections.values())) / len(rows),
        "images_by_crop_count": distribution,
        "last_selected_score": marginals[requested - 1]["score"],
        "first_rejected_score": marginals[requested]["score"],
        "uses_ground_truth": False,
    }


def predict_all_tiles(model, images: list[Path], report: dict):
    cache = {}
    geometry_example = None
    core.torch.cuda.reset_peak_memory_stats(0)
    core.torch.cuda.synchronize()
    started = time.perf_counter()
    for image_index, path in enumerate(images, 1):
        image = core.cv2.imread(str(path), core.cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV failed to read {path}")
        tiles = core.make_tiles(image)
        if geometry_example is None:
            geometry_example = [
                {key: value for key, value in tile.items() if key != "image"}
                for tile in tiles
            ]
        results = core.model_predict(model, [tile["image"] for tile in tiles], len(tiles))
        if len(results) != 4:
            raise RuntimeError("Expected four tile predictions per image")
        cache[str(path)] = [
            core.map_tile_detections(result, tile)
            for result, tile in zip(results, tiles)
        ]
        del results, tiles, image
        if image_index == 1 or image_index % 50 == 0 or image_index == len(images):
            print(f"D15b cached local tiles: {image_index}/{len(images)}", flush=True)
            report["progress"] = {"stage": "cache_all_four_tiles", "images": image_index}
            save(report)
    core.torch.cuda.synchronize()
    return cache, {
        "seconds": time.perf_counter() - started,
        "peak_memory": core.memory_peak(),
        "forward_images": 4 * len(images),
        "tiles_per_image": 4,
        "example_geometry": geometry_example,
    }


def compose_standard(full_data, mapped_tiles: list, selected: tuple[int, ...]):
    tensors = [mapped_tiles[index] for index in selected]
    local = (
        core.nms_detections(core.torch.cat(tensors, dim=0))
        if tensors else core.empty_detections()
    )
    combined = core.nms_detections(core.torch.cat([full_data, local], dim=0))
    return combined, local


def match_summary(detections, annotation_index: dict, conf: float, iou: float) -> dict:
    data = detections.detach().float().cpu().numpy()
    if len(data):
        valid = core.np.isfinite(data).all(axis=1) & (data[:, 4] >= conf)
        data = data[valid]
        if len(data):
            data = data[core.np.argsort(-data[:, 4], kind="stable")]
    used: set[int] = set()
    matched_iou = {}
    fp_all = 0
    fp_small = 0
    for detection in data:
        cls = int(detection[5]) + 1
        candidates = [
            ann for ann in annotation_index["by_class"].get(cls, ())
            if ann["id"] not in used
        ]
        best_ann = None
        best_iou = -1.0
        if candidates:
            boxes = core.np.stack([ann["box"] for ann in candidates])
            overlaps = core.box_iou_numpy(detection[:4], boxes)
            best_index = int(core.np.argmax(overlaps))
            best_iou = float(overlaps[best_index])
            best_ann = candidates[best_index]
        if best_ann is not None and best_iou >= iou:
            used.add(best_ann["id"])
            matched_iou[best_ann["id"]] = best_iou
        else:
            fp_all += 1
            width = max(0.0, float(detection[2] - detection[0]))
            height = max(0.0, float(detection[3] - detection[1]))
            if width * height < SMALL_AREA:
                fp_small += 1
    small_ids = annotation_index["small_ids"]
    matched_ids = set(matched_iou)
    matched_small = matched_ids.intersection(small_ids)
    return {
        "matched_ids": matched_ids,
        "matched_small_ids": matched_small,
        "matched_iou": matched_iou,
        "tp_all": len(matched_ids),
        "tp_small": len(matched_small),
        "fn_small": len(small_ids - matched_small),
        "fp_all": fp_all,
        "fp_small": fp_small,
        "fp_nonsmall": fp_all - fp_small,
    }


def signature_summaries(detections, annotation_index: dict) -> dict:
    return {
        f"conf{conf:.2f}_iou{iou:.2f}": match_summary(
            detections, annotation_index, conf, iou
        )
        for conf in UTILITY_CONFS for iou in UTILITY_IOUS
    }


def correction_delta(before: dict, after: dict) -> dict:
    rows = []
    for key in before:
        one, two = before[key], after[key]
        recovered = two["matched_small_ids"] - one["matched_small_ids"]
        lost = one["matched_small_ids"] - two["matched_small_ids"]
        delta_tp_small = two["tp_small"] - one["tp_small"]
        delta_fp_small = two["fp_small"] - one["fp_small"]
        delta_fp_nonsmall = two["fp_nonsmall"] - one["fp_nonsmall"]
        rows.append({
            "recovered_small": len(recovered),
            "lost_small": len(lost),
            "delta_tp_small": delta_tp_small,
            "delta_fp_small": delta_fp_small,
            "delta_fp_nonsmall": delta_fp_nonsmall,
            "utility": (
                SMALL_TP_WEIGHT * delta_tp_small
                - SMALL_FP_WEIGHT * delta_fp_small
                - NONSMALL_FP_WEIGHT * delta_fp_nonsmall
            ),
        })
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
    }


def build_oracle_rows(
    images: list[Path], full: dict, tile_cache: dict, annotations_by_path: dict,
    proxy_by_path: dict, report: dict,
) -> list[dict]:
    rows = []
    for image_index, path in enumerate(images, 1):
        key = str(path)
        annotation_index = annotations_by_path[key]
        prediction_cache = {(): full[key]}
        summary_cache = {(): signature_summaries(full[key], annotation_index)}

        def prediction(selected: tuple[int, ...]):
            selected = tuple(sorted(selected))
            if selected not in prediction_cache:
                prediction_cache[selected] = compose_standard(
                    full[key], tile_cache[key], selected
                )[0]
            return prediction_cache[selected]

        def summaries(selected: tuple[int, ...]):
            selected = tuple(sorted(selected))
            if selected not in summary_cache:
                summary_cache[selected] = signature_summaries(
                    prediction(selected), annotation_index
                )
            return summary_cache[selected]

        selected: list[int] = []
        marginal_utilities = []
        marginal_components = []
        initial_tile_utilities = [None] * 4
        initial_tile_components = [None] * 4
        for _stage in range(2):
            before_key = tuple(sorted(selected))
            candidates = []
            for tile in range(4):
                if tile in selected:
                    continue
                after_key = tuple(sorted([*selected, tile]))
                delta = correction_delta(summaries(before_key), summaries(after_key))
                if _stage == 0:
                    initial_tile_utilities[tile] = float(delta["utility"])
                    initial_tile_components[tile] = delta
                candidates.append((
                    delta["utility"],
                    delta["recovered_small"],
                    -delta["lost_small"],
                    -delta["delta_fp_small"],
                    proxy_by_path[key]["tile_scores"][tile],
                    -tile,
                    tile,
                    delta,
                ))
            best = max(candidates)
            selected.append(best[6])
            marginal_utilities.append(float(best[0]))
            marginal_components.append(best[7])
        rows.append({
            "image": key,
            "ranked_tiles": selected,
            "marginal_utilities": marginal_utilities,
            "marginal_components": marginal_components,
            "initial_tile_utilities": initial_tile_utilities,
            "initial_tile_components": initial_tile_components,
        })
        if image_index == 1 or image_index % 50 == 0 or image_index == len(images):
            print(f"D15b correction-utility audit: {image_index}/{len(images)}", flush=True)
            report["progress"] = {"stage": "correction_utility", "images": image_index}
            save(report)
    return rows


def allocate_oracle_budget(
    rows: list[dict], budget: float
) -> tuple[dict[str, list[int]], dict]:
    requested = int(round(budget * len(rows)))
    by_path = {row["image"]: row for row in rows}
    heap = []
    for row in rows:
        heapq.heappush(heap, (
            -float(row["marginal_utilities"][0]), row["image"], 0
        ))
    counts = {row["image"]: 0 for row in rows}
    selected_utilities = []
    for _ in range(requested):
        if not heap:
            raise RuntimeError("Oracle marginal heap exhausted")
        negative_utility, key, rank = heapq.heappop(heap)
        if rank != counts[key]:
            raise RuntimeError("Oracle budget violated the rank-prefix constraint")
        counts[key] += 1
        selected_utilities.append(-negative_utility)
        if counts[key] < 2:
            next_rank = counts[key]
            heapq.heappush(heap, (
                -float(by_path[key]["marginal_utilities"][next_rank]), key, next_rank
            ))
    selections = {
        key: by_path[key]["ranked_tiles"][:count] for key, count in counts.items()
    }
    distribution = {"0": 0, "1": 0, "2": 0}
    for count in counts.values():
        distribution[str(count)] += 1
    return selections, {
        "target_crops_per_image": budget,
        "actual_total_crops": requested,
        "actual_crops_per_image": requested / len(rows),
        "images_by_crop_count": distribution,
        "last_selected_utility": selected_utilities[-1],
        "first_available_rejected_utility": -heap[0][0] if heap else None,
        "uses_ground_truth": True,
        "claim_policy": "diagnostic upper bound only",
    }


def oracle_validity_filter(local, annotation_index: dict):
    if local.numel() == 0:
        return local, {"before": 0, "kept": 0, "invalid_removed": 0, "duplicate_removed": 0}
    data = local.detach().float().cpu()
    best_by_gt = {}
    valid_candidates = 0
    for index, detection in enumerate(data):
        cls = int(detection[5].item()) + 1
        candidates = annotation_index["by_class"].get(cls, ())
        if not candidates:
            continue
        boxes = core.np.stack([ann["box"] for ann in candidates])
        overlaps = core.box_iou_numpy(detection[:4].numpy(), boxes)
        best_index = int(core.np.argmax(overlaps))
        best_iou = float(overlaps[best_index])
        if best_iou < ORACLE_VALIDITY_IOU:
            continue
        valid_candidates += 1
        gt_id = candidates[best_index]["id"]
        rank = (best_iou, float(detection[4].item()), -index)
        previous = best_by_gt.get(gt_id)
        if previous is None or rank > previous[0]:
            best_by_gt[gt_id] = (rank, index)
    indices = sorted(value[1] for value in best_by_gt.values())
    filtered = data[indices].contiguous() if indices else core.empty_detections()
    before = len(data)
    kept = len(filtered)
    return filtered, {
        "before": before,
        "kept": kept,
        "invalid_removed": before - valid_candidates,
        "duplicate_removed": valid_candidates - kept,
    }


def compose_route(
    images: list[Path], full: dict, tile_cache: dict, selections: dict,
    annotations_by_path: dict, oracle_validity: bool,
) -> tuple[dict, dict]:
    predictions = {}
    started = time.perf_counter()
    crops = 0
    counts = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0}
    filter_totals = {"before": 0, "kept": 0, "invalid_removed": 0, "duplicate_removed": 0}
    for path in images:
        key = str(path)
        selected = tuple(selections[key])
        crops += len(selected)
        counts[str(len(selected))] += 1
        _combined, local = compose_standard(full[key], tile_cache[key], selected)
        if oracle_validity:
            local, filter_row = oracle_validity_filter(local, annotations_by_path[key])
            for name, value in filter_row.items():
                filter_totals[name] += value
        candidates = core.torch.cat([full[key], local], dim=0)
        predictions[key] = core.nms_detections(candidates)
        del _combined, local, candidates
    return predictions, {
        "merge_seconds": time.perf_counter() - started,
        "local_forward_images": crops,
        "local_crops_per_image": crops / len(images),
        "images_by_crop_count": counts,
        "oracle_validity_filter": oracle_validity,
        "oracle_filter_counts": filter_totals if oracle_validity else None,
    }


def evaluate_compact(
    name: str, predictions: dict, images: list[Path], path_ids: dict,
    gt: dict, names: dict,
) -> dict:
    print(f"D15b evaluating {name} on CPU.", flush=True)
    detections = core.predictions_to_coco(predictions, images, path_ids)
    metrics = core.coco_metrics(gt, detections, names)
    metrics["fixed_operating_point"] = core.fixed_operating_point(gt, detections, names)
    metrics["prediction_count"] = len(detections)
    return metrics


def route_profile(
    full_profile: dict, tile_profile: dict, compose_profile: dict, image_count: int,
) -> dict:
    local_count = compose_profile["local_forward_images"]
    estimated_local = tile_profile["seconds"] * (
        local_count / max(1, tile_profile["forward_images"])
    )
    return {
        **compose_profile,
        "seconds": full_profile["seconds"] + estimated_local + compose_profile["merge_seconds"],
        "timing_is_estimate": True,
        "forward_images": image_count + local_count,
        "forward_image_multiplier": (image_count + local_count) / image_count,
        "diagnostic_actual_forward_images": (
            full_profile["forward_images"] + tile_profile["forward_images"]
        ),
        "peak_memory": {
            key: max(full_profile["peak_memory"][key], tile_profile["peak_memory"][key])
            for key in ("allocated_GiB", "reserved_GiB")
        },
    }


def fixed_selections(rows: list[dict], count: int) -> dict[str, list[int]]:
    return {row["image"]: row["ranked_tiles"][:count] for row in rows}


def average_ranks(values: list[float]):
    values = core.np.asarray(values, dtype=core.np.float64)
    order = core.np.argsort(values, kind="stable")
    ranks = core.np.empty(len(values), dtype=core.np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def safe_correlation(one: list[float], two: list[float], ranked: bool) -> float | None:
    a = average_ranks(one) if ranked else core.np.asarray(one, dtype=core.np.float64)
    b = average_ranks(two) if ranked else core.np.asarray(two, dtype=core.np.float64)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return None
    return float(core.np.corrcoef(a, b)[0, 1])


def routing_agreement(proxy_rows: list[dict], oracle_rows: list[dict]) -> dict:
    proxy_by_path = {row["image"]: row for row in proxy_rows}
    top1 = 0
    top2_intersection = 0
    proxy_marginals = []
    oracle_marginals = []
    positive_oracle = 0
    for row in oracle_rows:
        proxy = proxy_by_path[row["image"]]
        top1 += int(proxy["ranked_tiles"][0] == row["ranked_tiles"][0])
        top2_intersection += len(
            set(proxy["ranked_tiles"][:2]).intersection(row["ranked_tiles"][:2])
        )
        for tile in range(4):
            proxy_marginals.append(float(proxy["tile_scores"][tile]))
            oracle_marginals.append(float(row["initial_tile_utilities"][tile]))
            positive_oracle += int(row["initial_tile_utilities"][tile] > 0)
    return {
        "images": len(oracle_rows),
        "top1_exact_fraction": top1 / len(oracle_rows),
        "top2_mean_intersection_fraction": top2_intersection / (2 * len(oracle_rows)),
        "marginal_score_pearson": safe_correlation(proxy_marginals, oracle_marginals, False),
        "marginal_score_spearman": safe_correlation(proxy_marginals, oracle_marginals, True),
        "positive_initial_tile_utility_fraction": positive_oracle / len(oracle_marginals),
    }


def metric_delta(one: dict, zero: dict) -> dict:
    keys = (
        "mAP50_95", "mAP50", "mAP75", "AR_all", "AP_small", "AR_small",
        "AP_medium", "AR_medium", "AP_large", "AR_large",
    )
    return {key: one[key] - zero[key] for key in keys}


def reproduction(report: dict) -> dict:
    modes = report["models"]["e1"]["modes"]
    reference = report["d15a_reference"]["modes"]
    mapping = {
        "full": "full",
        "proxy_top1_standard": "fixed_top1",
        "proxy_top2_standard": "fixed_top2",
        "proxy_b1p50_standard": "adaptive_b1p50",
    }
    keys = ("mAP50_95", "AP_small", "AR_small", "AP_medium", "AP_large")
    checks = {}
    for current_name, reference_name in mapping.items():
        current = modes[current_name]["overall"]
        expected = reference[reference_name]
        delta = {key: current[key] - expected[key] for key in keys}
        checks[current_name] = {
            "d15a_mode": reference_name,
            "current_minus_d15a": delta,
            "absolute_tolerance": REFERENCE_TOLERANCE,
            "passed": all(abs(value) <= REFERENCE_TOLERANCE for value in delta.values()),
        }
    return {"checks": checks, "passed": all(row["passed"] for row in checks.values())}


def build_decisions(report: dict) -> dict:
    modes = report["models"]["e1"]["modes"]
    reproduce = reproduction(report)
    selection = metric_delta(
        modes["oracle_utility_b1p50_standard"]["overall"],
        modes["proxy_b1p50_standard"]["overall"],
    )
    fusion = metric_delta(
        modes["proxy_b1p50_oracle_validity"]["overall"],
        modes["proxy_b1p50_standard"]["overall"],
    )
    joint = metric_delta(
        modes["oracle_utility_b1p50_oracle_validity"]["overall"],
        modes["proxy_b1p50_standard"]["overall"],
    )
    selection_pass = bool(
        selection["mAP50_95"] >= SELECTION_MAP_GATE
        or selection["AP_small"] >= SELECTION_APS_GATE
    )
    fusion_pass = bool(
        fusion["mAP50_95"] >= FUSION_MAP_GATE
        and fusion["AP_small"] >= FUSION_APS_GATE
    )
    if not reproduce["passed"]:
        branch = "stop_reproduction_failed"
        action = "STOP: D15a anchors did not reproduce; fix the evaluation pipeline first."
    elif fusion_pass and (
        not selection_pass or fusion["AP_small"] >= selection["AP_small"] + 0.002
    ):
        branch = "cross_view_local_candidate_verifier"
        action = (
            "Train a lightweight cross-view local-candidate verifier on VisDrone train. "
            "Keep E1 and the D15a 1.5-crop selector frozen; learn only residual score/quality "
            "calibration from full-view and local-view evidence."
        )
    elif selection_pass:
        branch = "decomposed_correction_utility_router"
        action = (
            "Train a decomposed correction-utility router on VisDrone train. Predict small-FN "
            "recovery and FP risk separately; freeze the budget threshold on train before val."
        )
    else:
        branch = "stop_learned_regional_components"
        action = (
            "Keep D15a as an engineering upper-bound result, but do not train a router or "
            "verifier: neither oracle shows enough headroom."
        )
    return {
        "d15a_reproduction": reproduce,
        "selection_headroom": {
            "comparison": "oracle_utility_b1p50_standard minus proxy_b1p50_standard",
            "delta": selection,
            "gate": {
                "mAP50_95_gain_or": SELECTION_MAP_GATE,
                "AP_small_gain": SELECTION_APS_GATE,
            },
            "passed": selection_pass,
        },
        "fusion_headroom": {
            "comparison": "proxy_b1p50_oracle_validity minus proxy_b1p50_standard",
            "delta": fusion,
            "gate": {
                "mAP50_95_gain_and": FUSION_MAP_GATE,
                "AP_small_gain": FUSION_APS_GATE,
            },
            "passed": fusion_pass,
        },
        "joint_oracle_headroom": joint,
        "selected_branch": branch,
        "recommended_next_action": action,
        "oracle_claim_boundary": (
            "Oracle modes use validation labels and cannot be reported as deployable accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-images", type=int, default=0, metavar="N",
        help="Pipeline test on the first N images; its metrics cannot drive decisions.",
    )
    args = parser.parse_args()
    if args.smoke_images < 0 or args.smoke_images > 548:
        parser.error("--smoke-images must be between 0 and 548")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "CHECK_D15B" if args.check_only else "SMOKE_D15B" if args.smoke_images else "DIAG_D15B"
    out = REPORTS / f"{prefix}_e1_correction_fusion_headroom_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing",
        "purpose": "frozen E1 correction-selection versus fusion headroom decision",
        "script_revision": SCRIPT_REVISION,
        "started_at": now(),
        "report_dir": str(out),
        "is_full_diagnostic": not args.check_only and not args.smoke_images,
        "protocol": {
            "weight": str(WEIGHT),
            "data": str(core.DATA),
            "imgsz": core.IMGSZ,
            "grid": "2x2 with 20% overlap",
            "primary_budget_crops_per_image": PRIMARY_BUDGET,
            "proxy_router": {
                "confidence_floor": ROUTER_CONF,
                "effective_small_side_px": ROUTER_SMALL_EFFECTIVE_SIDE,
                "uncertainty_weight": ROUTER_UNCERTAINTY_WEIGHT,
                "uses_ground_truth": False,
            },
            "correction_utility_oracle": {
                "confidence_thresholds": list(UTILITY_CONFS),
                "iou_thresholds": list(UTILITY_IOUS),
                "small_area_px2": SMALL_AREA,
                "formula": (
                    "mean(4*delta_TP_small - delta_FP_small "
                    "- 0.25*delta_FP_non_small)"
                ),
                "uses_ground_truth": True,
                "claim_policy": "diagnostic branch-selection upper bound only",
            },
            "oracle_validity_fusion": {
                "correct_class_iou_min": ORACLE_VALIDITY_IOU,
                "keeps_original_confidence": True,
                "one_local_candidate_per_gt": True,
                "removes_full_view_false_positives": False,
                "uses_ground_truth": True,
                "claim_policy": "diagnostic verifier headroom only",
            },
            "prediction_json_saved": False,
            "training_performed": False,
            "checkpoint_updates": False,
            "automatic_shutdown": False,
        },
        "models": {"e1": {"modes": {}}},
    }
    save(report)
    started = time.perf_counter()
    success = False
    model = None
    try:
        if core.SCRIPT_REVISION != EXPECTED_CORE_REVISION:
            raise RuntimeError(f"Unexpected slicing core revision: {core.SCRIPT_REVISION}")
        core.preflight(report, ["e1"], args.check_only)
        reference_path, reference = latest_d15a_reference()
        reference_modes = reference["models"]["e1"]["modes"]
        report["d15a_reference"] = {
            "path": str(reference_path),
            "script_revision": reference.get("script_revision"),
            "modes": {
                name: reference_modes[name]["overall"]
                for name in ("full", "fixed_top1", "fixed_top2", "adaptive_b1p50")
            },
        }
        images, path_ids, gt, names = core.prepare_dataset(report, args.smoke_images)
        annotations_by_path = annotation_records(gt)
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
            print("D15b preflight passed. No prediction, training, or shutdown.", flush=True)
            return

        full, full_profile = core.predict_full(model, images, report, "D15b E1")
        full_metrics = evaluate_compact("full", full, images, path_ids, gt, names)
        full_metrics["profile"] = {
            **full_profile,
            "local_forward_images": 0,
            "local_crops_per_image": 0.0,
            "forward_image_multiplier": 1.0,
            "diagnostic_actual_forward_images": len(images),
            "timing_is_estimate": False,
        }
        report["models"]["e1"]["modes"]["full"] = full_metrics
        save(report)

        tile_cache, tile_profile = predict_all_tiles(model, images, report)
        report["diagnostic_cache_profile"] = tile_profile
        proxy_rows = build_proxy_rows(images, full, image_meta(gt))
        proxy_by_path = {row["image"]: row for row in proxy_rows}
        oracle_rows = build_oracle_rows(
            images, full, tile_cache, annotations_by_path, proxy_by_path, report
        )
        oracle_by_path = {row["image"]: row for row in oracle_rows}
        proxy_budget, proxy_budget_meta = allocate_proxy_budget(proxy_rows, PRIMARY_BUDGET)
        oracle_budget, oracle_budget_meta = allocate_oracle_budget(oracle_rows, PRIMARY_BUDGET)
        report["routing_analysis"] = {
            "agreement": routing_agreement(proxy_rows, oracle_rows),
            "proxy_budget": proxy_budget_meta,
            "oracle_budget": oracle_budget_meta,
        }
        audit = {
            "script_revision": SCRIPT_REVISION,
            "routing_analysis": report["routing_analysis"],
            "per_image": [
                {
                    **proxy_by_path[str(path)],
                    "oracle_ranked_tiles": oracle_by_path[str(path)]["ranked_tiles"],
                    "oracle_marginal_utilities": oracle_by_path[str(path)]["marginal_utilities"],
                    "oracle_marginal_components": oracle_by_path[str(path)]["marginal_components"],
                    "proxy_budget_selection": proxy_budget[str(path)],
                    "oracle_budget_selection": oracle_budget[str(path)],
                }
                for path in images
            ],
        }
        core.atomic_json(out / "correction_routing_audit.json", audit)
        save(report)

        routes = {
            "proxy_top1_standard": (fixed_selections(proxy_rows, 1), False),
            "oracle_utility_top1_standard": (fixed_selections(oracle_rows, 1), False),
            "proxy_top2_standard": (fixed_selections(proxy_rows, 2), False),
            "oracle_utility_top2_standard": (fixed_selections(oracle_rows, 2), False),
            "proxy_b1p50_standard": (proxy_budget, False),
            "oracle_utility_b1p50_standard": (oracle_budget, False),
            "proxy_b1p50_oracle_validity": (proxy_budget, True),
            "oracle_utility_b1p50_oracle_validity": (oracle_budget, True),
        }
        for route_name, (selections, use_oracle_validity) in routes.items():
            report["status"] = f"evaluating_{route_name}"
            save(report)
            predictions, compose_profile = compose_route(
                images, full, tile_cache, selections, annotations_by_path,
                use_oracle_validity,
            )
            metrics = evaluate_compact(
                route_name, predictions, images, path_ids, gt, names
            )
            metrics["profile"] = route_profile(
                full_profile, tile_profile, compose_profile, len(images)
            )
            metrics["uses_ground_truth_for_route"] = route_name.startswith("oracle_utility")
            metrics["uses_ground_truth_for_fusion"] = use_oracle_validity
            report["models"]["e1"]["modes"][route_name] = metrics
            del predictions
            gc.collect()
            save(report)

        after_hash = core.sha256(WEIGHT)
        if after_hash != before_hash:
            raise RuntimeError("Frozen E1 checkpoint changed during D15b")
        report["models"]["e1"]["checkpoint_unchanged"] = True
        if args.smoke_images:
            report["decisions"] = {
                "disabled": "Smoke-subset metrics must not drive research decisions."
            }
        else:
            report["decisions"] = build_decisions(report)
        report.update(
            status="completed",
            finished_at=now(),
            total_seconds=time.perf_counter() - started,
        )
        core.save_csv(report)
        shutil.copy2(Path(__file__), out / Path(__file__).name)
        save(report)
        success = True
        print(f"D15b completed. Report: {out / 'metrics.txt'}", flush=True)
    except BaseException:
        report.update(
            status="failed_or_interrupted",
            finished_at=now(),
            total_seconds=time.perf_counter() - started,
            error=traceback.format_exc(),
        )
        save(report)
        core.atomic_text(out / "error.log", report["error"])
        print(f"D15b failure saved: {out}", file=sys.stderr, flush=True)
        raise
    finally:
        if model is not None:
            del model
        core.clear_gpu()
        try:
            core.atomic_json(out / "shutdown_status.json", {
                "time": now(),
                "enabled": False,
                "status": "disabled_by_user_policy",
                "reason": report.get("status", "unknown"),
            })
            save(report)
        except Exception:
            if success:
                raise
        print("Automatic shutdown is disabled; the instance remains running.", flush=True)


if __name__ == "__main__":
    main()
