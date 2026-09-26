"""Image-wise TAL execution for the pinned Ultralytics 8.3.0 detector.

No source-file patches, new label rules, microbatching, or per-image loss means.
The original forward is called on each image WITH the original batch's padding.
An explicit context restores the original class method, even after an exception.
Checkpoints contain the original Ultralytics classes, not a custom model class.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import ultralytics
from ultralytics.utils.tal import TaskAlignedAssigner

REVISION = 'imagewise_tal_830_v1'
EXPECTED_TAL_SHA256 = '102b2ef79c44a8e22d9b13033786130f716c52222c40d5b2d29642dd37ef15f9'
_ORIGINAL_FORWARD = TaskAlignedAssigner.forward
_ACTIVE_STATS = None


def implementation_info():
    source = Path(inspect.getfile(TaskAlignedAssigner))
    digest = hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if ultralytics.__version__ != '8.3.0' or digest != EXPECTED_TAL_SHA256:
        raise RuntimeError('Image-wise TAL requires the unmodified pinned 8.3.0 tal.py. '
                           f'Got version={ultralytics.__version__}, sha256={digest}.')
    return dict(revision=REVISION, original_tal_sha256=digest,
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                forward_batch_unchanged=True, matching_images_per_chunk=1,
                GT_padding_unchanged=True, batch_loss_normalization_unchanged=True,
                checkpoint_classes_unchanged=True)


@torch.no_grad()
def _imagewise_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
    # Never change rotated detection or a different custom assigner subclass.
    if type(self) is not TaskAlignedAssigner:
        return _ORIGINAL_FORWARD(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    batch_size, max_boxes = pd_scores.shape[0], gt_bboxes.shape[1]
    if _ACTIVE_STATS is not None:
        _ACTIVE_STATS['calls'] += 1
        _ACTIVE_STATS['images'] += int(batch_size)
        _ACTIVE_STATS['max_forward_batch'] = max(_ACTIVE_STATS['max_forward_batch'], int(batch_size))
        _ACTIVE_STATS['max_padded_gt_per_image'] = max(_ACTIVE_STATS['max_padded_gt_per_image'], int(max_boxes))
        _ACTIVE_STATS['max_prediction_locations'] = max(_ACTIVE_STATS['max_prediction_locations'],
                                                       int(pd_scores.shape[1]))
    # The pinned detection loss preprocesses GT in FP32, including during AMP.
    # Half-GT custom protocols have batch-dependent dtype promotion in TAL and
    # are deliberately rejected, not silently changed or retried after OOM.
    if gt_bboxes.dtype != torch.float32 or pd_bboxes.dtype != torch.float32:
        raise RuntimeError('Expected the original detection-loss FP32 box/GT protocol.')
    if batch_size <= 1 or max_boxes == 0:
        return _ORIGINAL_FORWARD(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
    outputs = []
    try:
        for index in range(batch_size):
            sl = slice(index, index + 1)
            # Do not trim padded GT, remove empty images, change local target
            # indices, or move tensors to CPU. Each call uses the original math.
            outputs.append(_ORIGINAL_FORWARD(
                self, pd_scores[sl], pd_bboxes[sl], anc_points,
                gt_labels[sl], gt_bboxes[sl], mask_gt[sl],
            ))
        return tuple(torch.cat([row[column] for row in outputs], dim=0) for column in range(5))
    finally:
        self.bs, self.n_max_boxes = batch_size, max_boxes


@contextlib.contextmanager
def chunked_assignment():
    """Single-process, non-nested scope; restores original forward on all exits."""
    global _ACTIVE_STATS
    implementation_info()
    if TaskAlignedAssigner.forward is not _ORIGINAL_FORWARD or _ACTIVE_STATS is not None:
        raise RuntimeError('Unexpected or nested TAL patch; refusing to mix implementations.')
    stats = dict(revision=REVISION, calls=0, images=0, max_forward_batch=0,
                 max_padded_gt_per_image=0, max_prediction_locations=0)
    _ACTIVE_STATS = stats
    TaskAlignedAssigner.forward = _imagewise_forward
    try:
        yield stats
    finally:
        TaskAlignedAssigner.forward = _ORIGINAL_FORWARD
        _ACTIVE_STATS = None


def _compare_tensors(reference, candidate, label, *, exact=False, rtol=1e-6, atol=1e-7):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(f'{label}: shape/dtype differ: '
                             f'{reference.shape}/{reference.dtype} vs {candidate.shape}/{candidate.dtype}')
    if reference.is_floating_point() and not (torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        raise AssertionError(f'{label}: non-finite result')
    equal = torch.equal(reference, candidate)
    if exact or not reference.is_floating_point():
        if not equal:
            raise AssertionError(f'{label}: discrete/exact outputs differ')
    else:
        torch.testing.assert_close(reference, candidate, rtol=rtol, atol=atol, equal_nan=False)
    difference = float((reference.float() - candidate.float()).abs().max().item()) if reference.numel() else 0.0
    return dict(name=label, exact_equal=equal, max_abs_difference=difference)


def _assignment_inputs(device, scenario, half_scores=False):
    batch, classes, side = 8, 10, 24
    y, x = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing='ij')
    anchors = torch.stack((x.flatten(), y.flatten()), -1).float() + .5
    points = anchors.shape[0]
    scores = torch.rand(batch, points, classes, device=device) * .8 + .1
    if half_scores:
        scores = scores.half()
        anchors = anchors.half()
    centers = anchors.float()[None].expand(batch, -1, -1)
    radii = torch.rand(batch, points, 2, device=device) * 3 + 1
    boxes = torch.cat((centers - radii, centers + radii), -1)
    counts = {
        'random': [16] * 8, 'mixed_empty': [0, 1, 16, 4, 0, 9, 2, 0],
        'dense': [128, 11, 70, 0, 35, 128, 3, 1],
        'ties_and_conflicts': [2, 1, 0, 6, 0, 2, 1, 0],
        'all_empty': [0] * 8,
    }[scenario]
    n = max(counts)
    gt = torch.zeros(batch, n, 4, device=device, dtype=torch.float32)
    labels = torch.zeros(batch, n, 1, device=device)
    mask = torch.zeros(batch, n, 1, device=device)
    for i, count in enumerate(counts):
        lt = torch.rand(count, 2, device=device) * 18
        wh = torch.rand(count, 2, device=device) * 5 + 1
        gt[i, :count] = torch.cat((lt, lt + wh), -1)
        labels[i, :count, 0] = torch.arange(count, device=device) % classes
        mask[i, :count] = 1
    if scenario == 'ties_and_conflicts':
        scores.fill_(.5)
        boxes[:] = torch.tensor([3., 3., 18., 18.], device=device)
        for i, count in enumerate(counts):
            gt[i, :count] = torch.tensor([3., 3., 18., 18.], device=device)
            labels[i, :count] = 0
    return scores, boxes, anchors, labels, gt, mask


def _test_assignments(device):
    rows = []
    for half in (False, True):
        for scenario in ('random', 'mixed_empty', 'dense', 'ties_and_conflicts', 'all_empty'):
            inputs = _assignment_inputs(device, scenario, half_scores=half)
            before = [x.clone() for x in inputs]
            original = TaskAlignedAssigner(topk=10, num_classes=10, alpha=.5, beta=6.)
            changed = TaskAlignedAssigner(topk=10, num_classes=10, alpha=.5, beta=6.)
            reference = original(*inputs)
            with chunked_assignment():
                candidate = changed(*inputs)
            checks = [_compare_tensors(a, b, name, exact=index in (0, 1, 3, 4))
                      for index, (a, b, name) in enumerate(zip(
                          reference, candidate,
                          ('labels', 'boxes', 'scores', 'foreground_mask', 'local_gt_indices')))]
            for a, b in zip(inputs, before):
                if not torch.equal(a, b):
                    raise AssertionError('Assignment unexpectedly changed its input.')
            if changed.bs != 8 or changed.n_max_boxes != inputs[4].shape[1]:
                raise AssertionError('Assigner batch metadata not restored.')
            positive_count = int(reference[3].bool().sum().item())
            if scenario != 'all_empty' and positive_count == 0:
                raise AssertionError('Nonempty equivalence case did not exercise positive matches.')
            rows.append(dict(case=scenario, score_dtype=str(inputs[0].dtype),
                             positive_assignments=positive_count, checks=checks))
            print(f'TAL equivalence: {scenario}, {inputs[0].dtype}: passed', flush=True)
    return rows


def _loss_batch(device, size, step):
    # Batch stays eight, with a mixture of empty and dense images.
    counts = [0, 1, 5, 13, 2, 0, 8, 3] if step == 0 else [4, 0, 2, 11, 0, 7, 3, 1]
    indices, classes, boxes = [], [], []
    for i, count in enumerate(counts):
        xy = torch.rand(count, 2, device=device) * .55 + .225
        wh = torch.rand(count, 2, device=device) * .15 + .10
        indices.extend([i] * count)
        classes.extend([j % 10 for j in range(count)])
        boxes.append(torch.cat((xy, wh), -1))
    return dict(img=torch.rand(8, 3, size, size, device=device),
                batch_idx=torch.tensor(indices, device=device, dtype=torch.float32),
                cls=torch.tensor(classes, device=device, dtype=torch.float32).reshape(-1, 1),
                bboxes=torch.cat(boxes))


def _test_loss_and_gradients(device, cfg, model_label='P2'):
    """Real P2 model, BN, full-batch detection loss, backward and two SGD steps."""
    from ultralytics.nn.tasks import DetectionModel
    rows = []
    for use_amp in ((False, True) if device.type == 'cuda' else (False,)):
        torch.manual_seed(284)
        reference = DetectionModel(copy.deepcopy(cfg), nc=10, verbose=False).to(device).train()
        reference.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
        candidate = copy.deepcopy(reference)
        opts = [torch.optim.SGD(m.parameters(), lr=.001, momentum=.937) for m in (reference, candidate)]
        # Fixed-scale AMP comparison includes unscaling before gradient checks.
        scale = 128.0 if use_amp else 1.0
        for step in range(2):
            batch = _loss_batch(device, 128, step)
            results = []
            for index, (model, optimizer) in enumerate(zip((reference, candidate), opts)):
                optimizer.zero_grad(set_to_none=True)
                scope = chunked_assignment() if index else contextlib.nullcontext()
                with scope:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        total, components = model(batch)
                    (total * scale).backward()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.div_(scale)
                results.append((total.detach(), components))
            checks = [
                _compare_tensors(results[0][0], results[1][0], 'full_batch_loss'),
                _compare_tensors(results[0][1], results[1][1], 'box_cls_dfl_loss'),
            ]
            gradient_max = 0.
            for (name, a), (_, b) in zip(reference.named_parameters(), candidate.named_parameters()):
                if (a.grad is None) != (b.grad is None):
                    raise AssertionError(f'{name}: gradient presence differs')
                if a.grad is not None:
                    row = _compare_tensors(a.grad, b.grad, 'gradient:' + name, rtol=1e-5, atol=1e-7)
                    gradient_max = max(gradient_max, row['max_abs_difference'])
            for optimizer in opts:
                optimizer.step()
            update_max = 0.
            for name, a in reference.state_dict().items():
                row = _compare_tensors(a, candidate.state_dict()[name], 'updated_state:' + name,
                                       rtol=1e-6, atol=1e-7)
                update_max = max(update_max, row['max_abs_difference'])
            rows.append(dict(model=model_label, amp=use_amp, step=step + 1, forward_batch=8, image_size=128,
                             checks=checks, gradient_max_abs_difference=gradient_max,
                             updated_state_max_abs_difference=update_max))
            print(f'TAL real {model_label} loss/gradient/update: AMP={use_amp}, step={step + 1}: passed', flush=True)
        del reference, candidate, opts, results, batch, total, components
    return rows


def verify_equivalence(cfg, device='cpu', baseline_cfg=None):
    """Bounded real-tensor regression, not proof for every possible future input."""
    info = implementation_info()
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested for equivalence verification but unavailable.')
    if TaskAlignedAssigner.forward is not _ORIGINAL_FORWARD:
        raise RuntimeError('Run verification outside the training patch context.')
    started = time.perf_counter()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    threads = torch.get_num_threads()
    try:
        # Preserve application RNG states; trainer also reseeds before actual fit.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(173)
            if device.type == 'cpu':
                torch.set_num_threads(min(4, threads))
            assignments = _test_assignments(device)
            gradients = _test_loss_and_gradients(device, cfg)
            baseline_steps = _test_loss_and_gradients(
                device, baseline_cfg, model_label='original YOLO11s') if baseline_cfg is not None else []
        if TaskAlignedAssigner.forward is not _ORIGINAL_FORWARD:
            raise AssertionError('Original forward was not restored after verification.')
        return dict(status='passed', device=str(device), torch=torch.__version__,
                    gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    implementation=info, seconds=time.perf_counter() - started,
                    assignment_cases=assignments, real_model_steps=gradients,
                    original_yolo11s_model_steps=baseline_steps,
                    tolerances=dict(discrete_outputs='exact; also shapes/dtypes',
                                    float_rtol=1e-6, float_atol=1e-7, gradient_rtol=1e-5),
                    limits='Synthetic FP32-GT cases and batch8 real-model 128px steps; '
                           'not a 1024px memory test or a guarantee of identical final AP.')
    finally:
        torch.set_num_threads(threads)


def assert_patch_restoration():
    """Small failure-path check without allocating any CUDA tensors."""
    try:
        with chunked_assignment():
            raise ValueError('intentional restoration test')
    except ValueError:
        pass
    if TaskAlignedAssigner.forward is not _ORIGINAL_FORWARD:
        raise AssertionError('Exception left a global TAL patch active.')
