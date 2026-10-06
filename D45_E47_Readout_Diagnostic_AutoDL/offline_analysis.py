"""Strict, stdlib-only paired analysis of existing E47/E40 reports.

No checkpoint loading, training, GPU work, or writes into source reports.
Coverage is per-GT candidate existence, not AP / one-to-one recall / cause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

IOU = 0.75
METRICS = ("raw_any", "raw_p2", "raw_p3", "raw_correct", "final_any", "final_correct")
AP_KEYS = ("AP50_95_percent", "AP50_percent", "AP75_percent")
STAGES = ("final_correct_present", "raw_geometry_missing", "raw_argmax_label_missing", "postprocess_candidate_missing")
SIZE_BINS = ((0, math.inf, "all"), (0, 8, "lt8"), (8, 16, "8_16"),
             (16, 24, "16_24"), (24, 32, "24_32"), (32, math.inf, "ge32"))
BOUNDARIES = [
    "Official AP uses the formal VisDrone ignore handling; legacy final coverage is before official prediction ignore filtering.",
    "Size is sqrt(GT area)*640/max(original width,height), not official AP-small/medium/large.",
    "Coverage is independent candidate existence per GT, not one-to-one matching recall or AP.",
    "raw_correct uses raw argmax class, while final_correct uses emitted class; these stages are not guaranteed nested.",
    "Mutually exclusive failure groups use final-correct first, then raw geometry, then raw argmax label, then postprocessing. This ordering is descriptive, not causal.",
    "This is a matched-checkpoint descriptive comparison; it cannot establish training root cause, seed-variance cause, or which added path is causally responsible.",
    "Legacy GT rows omit coordinates. When absent, annotation/dimension SHA, exact GT keys, classes and sizes are checked, not nonexistent per-row coordinates.",
    "Differences across size/class slices do not prove a module mechanism; rare slices have limited support.",
]


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _number(value: Any, label: str, lo: float | None = None, hi: float | None = None) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), "Non-numeric " + label)
    number = float(value)
    _require(math.isfinite(number), "Non-finite " + label)
    _require(lo is None or number >= lo, "Out-of-range " + label)
    _require(hi is None or number <= hi, "Out-of-range " + label)
    return number


def _integer(value: Any, label: str, lo: int = 0, hi: int | None = None) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), "Non-integer " + label)
    _require(value >= lo and (hi is None or value <= hi), "Out-of-range " + label)
    return value


def _unique_object(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        _require(key not in result, "Duplicate JSON object field: " + key)
        result[key] = value
    return result


def _read(path: Path) -> Any:
    _require(path.is_file(), "Missing required input: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Invalid JSON number " + x)))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _index_rows(rows: Any, label: str, image_count: int) -> dict:
    _require(isinstance(rows, list), label + " gt_rows is not an array")
    result = {}
    for row in rows:
        _require(isinstance(row, dict), label + " has a non-object GT row")
        for field in ("image_id", "gt_index", "cls", "size") + METRICS:
            _require(field in row, label + " missing GT field " + field)
        key = (_integer(row["image_id"], label + ".image_id", 1, image_count),
               _integer(row["gt_index"], label + ".gt_index"))
        _require(key not in result, label + " duplicate GT key " + str(key))
        _integer(row["cls"], label + ".cls", 1, 10)
        _number(row["size"], label + ".size", 0)
        for field in METRICS:
            _number(row[field], label + "." + field, 0, 1)
        if "gt_xywh" in row:
            _require(isinstance(row["gt_xywh"], list) and len(row["gt_xywh"]) == 4,
                     label + " gt_xywh must have four coordinates")
            for n, value in enumerate(row["gt_xywh"]):
                _number(value, label + ".gt_xywh[" + str(n) + "]", 0)
        if "density" in row:
            _integer(row["density"], label + ".density")
        result[key] = row
    return result


def _stage(row: dict) -> str:
    if row["final_correct"] >= IOU:
        return STAGES[0]
    if row["raw_any"] < IOU:
        return STAGES[1]
    if row["raw_correct"] < IOU:
        return STAGES[2]
    return STAGES[3]


def _summarize(group: list[tuple[dict, dict]]) -> dict:
    count = len(group)
    result: dict[str, Any] = {"gt_count": count, "iou_threshold": IOU, "coverage": {}}
    for metric in METRICS:
        good47 = good40 = gained = lost = same_good = same_bad = 0
        iou_delta_sum = 0.0
        for a, b in group:
            pa, pb = a[metric] >= IOU, b[metric] >= IOU
            good47 += pa
            good40 += pb
            gained += pa and not pb
            lost += pb and not pa
            same_good += pa and pb
            same_bad += not pa and not pb
            iou_delta_sum += a[metric] - b[metric]
        result["coverage"][metric] = {
            "e47_good_count": good47, "e40_good_count": good40,
            "e47_percent": 100 * good47 / count if count else None,
            "e40_percent": 100 * good40 / count if count else None,
            "improved_count": gained, "lost_count": lost, "net_count": gained - lost,
            "delta_pp": 100 * (gained - lost) / count if count else None,
            "unchanged_good_count": same_good, "unchanged_bad_count": same_bad,
            "mean_iou_delta": iou_delta_sum / count if count else None,
        }
    e47 = {stage: 0 for stage in STAGES}
    e40 = {stage: 0 for stage in STAGES}
    matrix = {old: {new: 0 for new in STAGES} for old in STAGES}
    non_nested = {model: {"final_correct_without_raw_argmax_correct": 0,
                          "final_any_without_raw_any": 0,
                          "raw_correct_without_raw_any": 0,
                          "final_correct_without_final_any": 0} for model in ("e47", "e40")}
    for a, b in group:
        new, old = _stage(a), _stage(b)
        e47[new] += 1
        e40[old] += 1
        matrix[old][new] += 1
        for label, row in (("e47", a), ("e40", b)):
            non_nested[label]["final_correct_without_raw_argmax_correct"] += row["final_correct"] >= IOU and row["raw_correct"] < IOU
            non_nested[label]["final_any_without_raw_any"] += row["final_any"] >= IOU and row["raw_any"] < IOU
            non_nested[label]["raw_correct_without_raw_any"] += row["raw_correct"] >= IOU and row["raw_any"] < IOU
            non_nested[label]["final_correct_without_final_any"] += row["final_correct"] >= IOU and row["final_any"] < IOU
    result["failure_stage_ordering"] = {
        "e47_counts": e47, "e40_counts": e40,
        "e47_percent": {key: 100 * val / count if count else None for key, val in e47.items()},
        "e40_percent": {key: 100 * val / count if count else None for key, val in e40.items()},
        "delta_count": {key: e47[key] - e40[key] for key in STAGES},
        "transition_matrix_e40_to_e47": matrix,
        "non_nested_threshold_counts": non_nested,
    }
    return result


def _metrics(value: Any, label: str) -> dict:
    _require(isinstance(value, dict), label + " candidate metrics are not an object")
    for key in AP_KEYS:
        _require(key in value, label + " missing metric " + key)
        _number(value[key], label + "." + key, 0, 100)
    _require(isinstance(value.get("per_class"), dict), label + " missing per_class")
    _require(set(value["per_class"]) == {str(i) for i in range(1, 11)}, label + " incomplete ten-class metrics")
    for cls, item in value["per_class"].items():
        _require(isinstance(item, dict) and isinstance(item.get("name"), str), label + " invalid class " + cls)
        for key in AP_KEYS:
            _require(key in item, label + " class " + cls + " missing metric " + key)
            _number(item[key], label + ".per_class." + cls + "." + key, 0, 100)
        _integer(item.get("images_with_class_weight"), label + ".class_weight", 0)
    _require(isinstance(value.get("class_image_weights"), dict), label + " missing class_image_weights")
    _require(set(value["class_image_weights"]) == {str(i) for i in range(1, 11)}, label + " incomplete class_image_weights")
    for cls in (str(i) for i in range(1, 11)):
        weight = _integer(value["class_image_weights"][cls], label + ".class_image_weights." + cls)
        _require(weight == value["per_class"][cls]["images_with_class_weight"], label + " inconsistent per-class availability weight " + cls)
    _require(isinstance(value.get("aggregation_note"), str), label + " missing official aggregation_note")
    return value


def summarize_pair(e47_dir: str | Path, e40_dir: str | Path, out_dir: str | Path) -> dict:
    """Write strict paired reports only after all source consistency checks pass."""
    a_dir, b_dir, output = (Path(path).resolve() for path in (e47_dir, e40_dir, out_dir))
    _require(a_dir != b_dir, "E47 and E40 reports must be distinct")
    for source in (a_dir, b_dir):
        _require(output != source and source not in output.parents, "Output must not be inside an original report")
    files = ("comparison.json", "official_metrics.json", "gt_rows.json")
    hashes = {label: {name: _sha(directory / name) for name in files}
              for label, directory in (("e47", a_dir), ("e40", b_dir))}
    ac, bc = _read(a_dir / files[0]), _read(b_dir / files[0])
    for label, comp in (("e47", ac), ("e40", bc)):
        _require(isinstance(comp, dict) and comp.get("status") == "completed", label + " report is not completed")
        _integer(comp.get("seed"), label + ".seed", 0)
        _integer(comp.get("imgsz"), label + ".imgsz", 1)
        _require(isinstance(comp.get("dataset"), dict), label + " missing dataset metadata")
        _require(isinstance(comp.get("ignore"), dict), label + " missing ignore audit metadata")
        _require(isinstance(comp.get("checkpoint_sha256"), str) and len(comp["checkpoint_sha256"]) == 64,
                 label + " missing checkpoint SHA")
    _require(ac["seed"] == bc["seed"], "Not a matched seed pair")
    _require(ac["imgsz"] == bc["imgsz"] == 640, "D45 requires matched imgsz=640")
    for field in ("annotation_dimension_sha256", "images", "evaluated_gt_rows", "raw_class_counts",
                  "score_zero_counts", "class0_ignore_regions", "class11_others", "gt_rows_dropped_by_class0_region"):
        _require(field in ac["dataset"] and field in bc["dataset"], "Missing dataset fingerprint field " + field)
        _require(ac["dataset"][field] == bc["dataset"][field], "Dataset / ignored-GT mismatch: " + field)
    fingerprint = ac["dataset"]["annotation_dimension_sha256"]
    _require(isinstance(fingerprint, str) and len(fingerprint) == 64, "Invalid annotation/dimension SHA")
    _require(ac["ignore"].get("fp_audit_note") == bc["ignore"].get("fp_audit_note") and
             isinstance(ac["ignore"].get("fp_audit_note"), str), "Ignore audit convention differs or is missing")
    ca, cb = _metrics(ac.get("candidate"), "e47"), _metrics(bc.get("candidate"), "e40")
    _require(ca["class_image_weights"] == cb["class_image_weights"], "Official class availability weights differ")
    _require(ca["aggregation_note"] == cb["aggregation_note"], "Official aggregation convention differs")
    for label, directory, candidate in (("e47", a_dir, ca), ("e40", b_dir, cb)):
        official = _read(directory / "official_metrics.json")
        _require(isinstance(official, dict) and official.get("status") == "completed", label + " official report incomplete")
        _require(official.get("candidate") == candidate, label + " official/comparison metric mismatch")
        _require(official.get("dataset") == (ac if label == "e47" else bc)["dataset"], label + " official/comparison dataset mismatch")
    image_count = _integer(ac["dataset"]["images"], "dataset.images", 1)
    expected_count = _integer(ac["dataset"]["evaluated_gt_rows"], "dataset.evaluated_gt_rows", 1)
    ai = _index_rows(_read(a_dir / "gt_rows.json"), "e47", image_count)
    bi = _index_rows(_read(b_dir / "gt_rows.json"), "e40", image_count)
    _require(len(ai) == len(bi) == expected_count, "GT row count does not match evaluated_gt_rows")
    if set(ai) != set(bi):
        missing = sorted(set(bi) - set(ai))[:5]
        extra = sorted(set(ai) - set(bi))[:5]
        raise ValueError("GT key sets differ; no inner join allowed; missing=" + str(missing) + " extra=" + str(extra))
    coords = density = True
    pairs = []
    for key in sorted(ai):
        a, b = ai[key], bi[key]
        _require(a["cls"] == b["cls"], "GT class differs at " + str(key))
        _require(math.isclose(a["size"], b["size"], rel_tol=0, abs_tol=1e-9), "GT size differs at " + str(key))
        if "gt_xywh" in a and "gt_xywh" in b:
            _require(all(math.isclose(x, y, rel_tol=0, abs_tol=1e-9) for x, y in zip(a["gt_xywh"], b["gt_xywh"])),
                     "GT coordinates differ at " + str(key))
        else:
            coords = False
        if "density" in a and "density" in b:
            _require(a["density"] == b["density"], "GT density differs at " + str(key))
        else:
            density = False
        pairs.append((a, b))
    per_class = {}
    for cls in (str(i) for i in range(1, 11)):
        a, b = ca["per_class"][cls], cb["per_class"][cls]
        _require(a["name"] == b["name"], "Class names differ: " + cls)
        _require(a["images_with_class_weight"] == b["images_with_class_weight"], "Class weight differs: " + cls)
        per_class[cls] = {"name": a["name"], "e47": {k: a[k] for k in AP_KEYS},
                          "e40": {k: b[k] for k in AP_KEYS}, "delta_pp": {k: a[k] - b[k] for k in AP_KEYS}}
    slices = {}
    for lo, hi, name in SIZE_BINS:
        subset = [(a, b) for a, b in pairs if lo <= a["size"] < hi]
        slices[name] = _summarize(subset)
        if name != "all":
            for cls in range(1, 11):
                slices[name + "/class_" + str(cls)] = _summarize([(a, b) for a, b in subset if a["cls"] == cls])
    for cls in range(1, 11):
        slices["class_" + str(cls)] = _summarize([(a, b) for a, b in pairs if a["cls"] == cls])
    if density:
        for lo, hi, name in ((0, 2, "0_1"), (2, 5, "2_4"), (5, math.inf, "ge5")):
            slices["density_" + name] = _summarize([(a, b) for a, b in pairs if lo <= a["density"] < hi])
    summary = {
        "status": "completed", "kind": "cached_matched_report_descriptive_comparison", "seed": ac["seed"],
        "imgsz": ac["imgsz"], "inputs": {"e47": str(a_dir), "e40": str(b_dir)}, "source_sha256": hashes,
        "checkpoint_sha256": {"e47": ac["checkpoint_sha256"], "e40": bc["checkpoint_sha256"]},
        "pairing": {"stable_key": ["image_id", "gt_index"], "gt_count": len(pairs),
                    "annotation_dimension_sha256": fingerprint, "exact_key_set_match": True,
                    "class_and_size_match": True, "per_row_coordinates_checked": coords,
                    "coordinate_note": "All paired gt_xywh checked" if coords else "Legacy rows lack coordinates; SHA/classes/sizes/keys checked instead",
                    "density_available_in_both_reports": density},
        "official": {"e47": {key: ca[key] for key in AP_KEYS}, "e40": {key: cb[key] for key in AP_KEYS},
                     "delta_pp": {key: ca[key] - cb[key] for key in AP_KEYS}, "per_class": per_class},
        "overall": slices["all"], "boundaries": BOUNDARIES,
        "missing_optional_data": [] if density else ["No matched per-GT density: density slices were not fabricated"],
        "interpretation": "Positive/negative paired transitions show checkpoint-level tradeoffs. Added-path dependence requires separately reproduced intervention inference; neither establishes training root cause.",
    }
    for label, directory in (("e47", a_dir), ("e40", b_dir)):
        for name in files:
            _require(_sha(directory / name) == hashes[label][name], "Input changed during analysis: " + str(directory / name))
    lines = ["# D45 E47/E40 已有报告严格配对诊断", "", "不训练、不读取 checkpoint、不使用 GPU。", "",
             f"匹配 seed={ac['seed']}，imgsz=640，GT={len(pairs)}。", "",
             "## 官方评分", "", "|指标|E40|E47|E47−E40 (pp)|", "|---|---:|---:|---:|"]
    for key in AP_KEYS:
        lines.append(f"|{key}|{cb[key]:.4f}|{ca[key]:.4f}|{ca[key]-cb[key]:+.4f}|")
    lines += ["", "## 官方逐类变化", "", "|类别|E40 AP|E47 AP|ΔAP pp|ΔAP50 pp|ΔAP75 pp|", "|---|---:|---:|---:|---:|---:|"]
    for cls in (str(i) for i in range(1, 11)):
        item = per_class[cls]
        delta = item["delta_pp"]
        lines.append(f"|{item['name']}|{item['e40']['AP50_95_percent']:.4f}|{item['e47']['AP50_95_percent']:.4f}|"
                     f"{delta['AP50_95_percent']:+.4f}|{delta['AP50_percent']:+.4f}|{delta['AP75_percent']:+.4f}|")
    lines += ["", "## 同一 GT 的尺寸分组得失（IoU≥0.75，非 AP）", "",
              "|尺寸组|GT|raw-any 净变化 pp|final-correct 新覆盖|final-correct 损失|final-correct 净变化 pp|",
              "|---|---:|---:|---:|---:|---:|"]
    for _, _, name in SIZE_BINS:
        item = slices[name]
        fc = item["coverage"]["final_correct"]
        raw = item["coverage"]["raw_any"]["delta_pp"]
        rf = "N/A" if raw is None else f"{raw:+.4f}"
        ff = "N/A" if fc["delta_pp"] is None else f"{fc['delta_pp']:+.4f}"
        lines.append(f"|{name}|{item['gt_count']}|{rf}|{fc['improved_count']}|{fc['lost_count']}|{ff}|")
    lines += ["", "## 剩余缺点的互斥描述（不是根因）", "",
              "优先判断最终同类别框是否存在，再依次归入 raw geometry / raw argmax label / postprocess candidate missing。",
              "这不是严格嵌套损失漏斗；non_nested_threshold_counts 会单列反例。", "",
              "|状态|E40 GT 数|E47 GT 数|E47−E40|", "|---|---:|---:|---:|"]
    stages = slices["all"]["failure_stage_ordering"]
    for stage in STAGES:
        lines.append(f"|{stage}|{stages['e40_counts'][stage]}|{stages['e47_counts'][stage]}|{stages['delta_count'][stage]:+d}|")
    lines += ["", "## 严格配对及解释边界", "",
              "旧 GT 行不含坐标，不能声称已逐坐标核对；采用注释/图像尺寸 SHA、完整 GT 键集合、类别和输入尺度尺寸核验。"
              if not coords else "双方 gt_xywh 逐条核对通过。", "",
              "密度缺失则不编造密度诊断。详细逐类、尺寸×类别、改善/损失 GT 数及状态转移见 paired_slices.json。", ""]
    lines += ["- " + item for item in BOUNDARIES]
    _atomic_json(output / "paired_slices.json", {"iou_threshold": IOU, "slices": slices, "boundaries": BOUNDARIES})
    _atomic_json(output / "offline_summary.json", summary)
    _atomic_text(output / "REPORT_OFFLINE.md", "\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e47-report", required=True, type=Path)
    parser.add_argument("--e40-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize_pair(args.e47_report, args.e40_report, args.out)
    print("D45 OFFLINE COMPLETED " + json.dumps({"gt_count": result["pairing"]["gt_count"],
          "official_delta_pp": result["official"]["delta_pp"], "out": str(args.out.resolve())}), flush=True)


if __name__ == "__main__":
    main()
