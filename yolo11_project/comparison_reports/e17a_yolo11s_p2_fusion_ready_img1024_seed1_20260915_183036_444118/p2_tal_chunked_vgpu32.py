"""Two-image TAL chunks for the pinned 32GB vGPU training protocol.

This module reuses the audited tests from p2_tal_chunked.py but changes only
the TaskAlignedAssigner execution chunk from one image to two images.  Full
model forward, padded GT, loss normalization and checkpoint classes remain
unchanged.  Do not use until its full equivalence test and 1024 batch-8 Smoke
have passed on the target vGPU.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
from pathlib import Path

import torch
import ultralytics
from ultralytics.utils.tal import TaskAlignedAssigner

import p2_tal_chunked as legacy


REVISION = "tal_chunk2_vgpu32_830_v1"
CHUNK_SIZE = 2
EXPECTED_LEGACY_SHA256 = "16e34b5c937c3224f0017da732999e567635f8c040b9327306c8f4179c77d6b6"
_ORIGINAL_FORWARD = legacy._ORIGINAL_FORWARD
_ACTIVE_STATS = None
_LEGACY_IMPLEMENTATION_INFO = legacy.implementation_info


def _self_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def implementation_info():
    legacy_path = Path(inspect.getfile(legacy)).resolve()
    if hashlib.sha256(legacy_path.read_bytes()).hexdigest() != EXPECTED_LEGACY_SHA256:
        raise RuntimeError(f"Audited legacy TAL helper changed: {legacy_path}")
    base = _LEGACY_IMPLEMENTATION_INFO()
    base.update(
        revision=REVISION,
        helper_sha256=_self_sha256(),
        matching_images_per_chunk=CHUNK_SIZE,
        acceleration_target="NVIDIA vGPU-32GB",
        note="Only matching chunk size changes from 1 to 2.",
    )
    return base


@torch.no_grad()
def _chunk2_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    if type(self) is not TaskAlignedAssigner:
        return _ORIGINAL_FORWARD(
            self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )
    batch_size, max_boxes = pd_scores.shape[0], gt_bboxes.shape[1]
    if _ACTIVE_STATS is not None:
        _ACTIVE_STATS["calls"] += 1
        _ACTIVE_STATS["images"] += int(batch_size)
        _ACTIVE_STATS["max_padded_gt_per_image"] = max(
            _ACTIVE_STATS["max_padded_gt_per_image"], int(max_boxes)
        )
        _ACTIVE_STATS["max_prediction_locations"] = max(
            _ACTIVE_STATS["max_prediction_locations"], int(pd_scores.shape[1])
        )
    if gt_bboxes.dtype != torch.float32 or pd_bboxes.dtype != torch.float32:
        raise RuntimeError("Expected original detection-loss FP32 box/GT protocol")
    if batch_size <= CHUNK_SIZE or max_boxes == 0:
        if _ACTIVE_STATS is not None:
            _ACTIVE_STATS["max_forward_batch"] = max(
                _ACTIVE_STATS["max_forward_batch"], int(batch_size)
            )
        return _ORIGINAL_FORWARD(
            self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )
    outputs = []
    try:
        for start in range(0, batch_size, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, batch_size)
            if _ACTIVE_STATS is not None:
                _ACTIVE_STATS["max_forward_batch"] = max(
                    _ACTIVE_STATS["max_forward_batch"], stop - start
                )
            sl = slice(start, stop)
            outputs.append(_ORIGINAL_FORWARD(
                self, pd_scores[sl], pd_bboxes[sl], anc_points,
                gt_labels[sl], gt_bboxes[sl], mask_gt[sl],
            ))
        return tuple(
            torch.cat([row[column] for row in outputs], dim=0) for column in range(5)
        )
    finally:
        self.bs, self.n_max_boxes = batch_size, max_boxes


@contextlib.contextmanager
def chunked_assignment():
    global _ACTIVE_STATS
    implementation_info()
    if TaskAlignedAssigner.forward is not _ORIGINAL_FORWARD or _ACTIVE_STATS is not None:
        raise RuntimeError("Unexpected or nested TAL patch")
    stats = {
        "revision": REVISION,
        "matching_images_per_chunk": CHUNK_SIZE,
        "calls": 0,
        "images": 0,
        "max_forward_batch": 0,
        "max_padded_gt_per_image": 0,
        "max_prediction_locations": 0,
    }
    _ACTIVE_STATS = stats
    TaskAlignedAssigner.forward = _chunk2_forward
    try:
        yield stats
    finally:
        TaskAlignedAssigner.forward = _ORIGINAL_FORWARD
        _ACTIVE_STATS = None


def verify_equivalence(cfg, device="cpu", baseline_cfg=None):
    old_chunk = legacy.chunked_assignment
    old_info = legacy.implementation_info
    legacy.chunked_assignment = chunked_assignment
    legacy.implementation_info = implementation_info
    try:
        result = legacy.verify_equivalence(cfg, device=device, baseline_cfg=baseline_cfg)
    finally:
        legacy.chunked_assignment = old_chunk
        legacy.implementation_info = old_info
    result["implementation"] = implementation_info()
    if result.get("status") != "passed":
        raise RuntimeError("Chunk-2 TAL equivalence did not pass")
    return result


def assert_patch_restoration():
    try:
        with chunked_assignment():
            raise ValueError("intentional restoration test")
    except ValueError:
        pass
    if TaskAlignedAssigner.forward is not _ORIGINAL_FORWARD:
        raise AssertionError("Exception left the TAL patch active")
