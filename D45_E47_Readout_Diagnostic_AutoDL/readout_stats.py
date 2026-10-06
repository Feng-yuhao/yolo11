"""GT-local E47 readout measurements; descriptions, never causal conclusions.

The observer consumes tensors captured in the actual forward. It does not call
the encoder, router or output projections again. Center route weights and the
3x3 mean route weights are reported separately. RMS and residual cosine use a
3x3 neighborhood (cropped, not zero-padded, at map borders).
"""
import math

import numpy as np
import torch
from torch.nn import functional as F


SIZE_GROUPS = ((0., math.inf, 'all'), (0., 8., 'lt8'), (8., 16., '8_16'),
               (16., 24., '16_24'), (24., 32., '24_32'), (32., math.inf, 'ge32'))
SCALARS = ('task_JS_nats', 'cls_entropy_nats', 'reg_entropy_nats',
           'task_JS_3x3_nats', 'cls_entropy_3x3_nats', 'reg_entropy_3x3_nats',
           'input_RMS', 'cls_e40_residual_RMS', 'reg_e40_residual_RMS',
           'cls_new_readout_RMS', 'reg_new_readout_RMS',
           'cls_e40_to_input_ratio', 'reg_e40_to_input_ratio',
           'cls_new_to_input_ratio', 'reg_new_to_input_ratio',
           'cls_new_to_e40_ratio', 'reg_new_to_e40_ratio',
           'cls_residual_cosine', 'reg_residual_cosine')
VECTORS = ('cls_depth_weights', 'reg_depth_weights',
           'cls_depth_weights_3x3', 'reg_depth_weights_3x3')


def js(p, q):
    """Jensen-Shannon divergence over the leading depth axis, in nats."""
    mid = (p + q) * .5
    return .5 * ((p * (p.clamp_min(1e-12).log() - mid.clamp_min(1e-12).log())).sum(0)
                 + (q * (q.clamp_min(1e-12).log() - mid.clamp_min(1e-12).log())).sum(0))


def _pool(value):
    return F.avg_pool2d(value, 3, stride=1, padding=1, count_include_pad=False)


def _mean_square(value):
    return _pool(value.float().square().mean(1, keepdim=True))[:, 0]


def _ratio(numerator, denominator):
    # Zero reference is an undefined ratio, not fabricated zero contribution.
    return None if denominator == 0. else float(numerator / denominator)


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError('Nonfinite readout statistic')
    return result


class ReadoutStats:
    """Observer API: (level, block, x, decomposition, base_outputs, updates)."""
    def __init__(self):
        self.context = None
        self.rows = []
        self._keys = set()

    def set_context(self, records, shape):
        shape = tuple(int(v) for v in shape)
        if len(shape) != 2 or min(shape) <= 0 or any(v % 32 for v in shape):
            raise RuntimeError('Expected actual positive, stride-32-aligned input H/W')
        if not records:
            raise RuntimeError('Empty input context')
        ids = set()
        for rec in records:
            if int(rec.width) <= 0 or int(rec.height) <= 0:
                raise RuntimeError('Invalid original image dimensions')
            if rec.image_id in ids:
                raise RuntimeError('Repeated image in input context')
            ids.add(rec.image_id)
            gt = np.asarray(rec.eval_gt)
            if gt.ndim != 2 or gt.shape[1] != 6 or not np.isfinite(gt).all():
                raise RuntimeError('Expected finite official eval_gt [N,6]')
            if (gt[:, 2:4] < 0).any():
                raise RuntimeError('Negative GT dimensions')
        self.context = (list(records), shape)

    @torch.no_grad()
    def __call__(self, level, block, x, decomposition, base_outputs, readout_updates):
        if self.context is None:
            raise RuntimeError('ReadoutStats context must be set before forward')
        if level not in (0, 1):
            raise RuntimeError('ReadoutStats only supports P2/P3')
        records, shape = self.context
        stride = 4 if level == 0 else 8
        if x.ndim != 4 or len(x) != len(records) or x.shape[1] != 128:
            raise RuntimeError('ReadoutStats input/context shape mismatch')
        if tuple(x.shape[-2:]) != (shape[0] // stride, shape[1] // stride):
            raise RuntimeError('ReadoutStats level/input stride mismatch')
        if len(decomposition) != 4 or len(base_outputs) != 2:
            raise RuntimeError('Malformed captured forward tensors')
        if readout_updates is None or len(readout_updates) != 2:
            raise RuntimeError('No captured new readout: do not invent zero contribution')
        _, _, weights, bank = decomposition
        expected = (len(x), 2, 4, x.shape[-2], x.shape[-1])
        if tuple(weights.shape) != expected or len(bank) != 4:
            raise RuntimeError('Wrong task/depth axes in captured route weights')
        tensors = [x, weights, *base_outputs, *readout_updates]
        if any(t is None or t.device != x.device or not torch.isfinite(t).all() for t in tensors):
            raise RuntimeError('Missing, cross-device or nonfinite captured tensor')
        if any(tuple(t.shape) != tuple(x.shape) for t in (*base_outputs, *readout_updates)):
            raise RuntimeError('Residual/output tensor shape mismatch')
        weights = weights.float()
        if (weights < 0).any():
            raise RuntimeError('Negative depth probability')
        torch.testing.assert_close(weights.sum(2), torch.ones_like(weights[:, :, 0]),
                                   rtol=1e-4, atol=1e-5)
        base_square = _mean_square(x)
        old = [t.float() - x.float() for t in base_outputs]
        new = [t.float() for t in readout_updates]
        old_square = [_mean_square(t) for t in old]
        new_square = [_mean_square(t) for t in new]
        dot = [_pool((a * b).mean(1, keepdim=True))[:, 0] for a, b in zip(old, new)]
        pooled_weights = _pool(weights.reshape(len(x), 8, *x.shape[-2:])).reshape_as(weights)
        for index, rec in enumerate(records):
            gt = np.asarray(rec.eval_gt)
            gt = gt[(gt[:, 4] == 0) & (gt[:, 5] >= 1) & (gt[:, 5] <= 10)]
            if not len(gt):
                continue
            # Pinned Ultralytics LetterBox: rounded unpadded size, then rounded
            # asymmetric top/left borders. Use ACTUAL rect-batch tensor shape.
            gain = min(shape[0] / rec.height, shape[1] / rec.width)
            left = round((shape[1] - round(rec.width * gain)) / 2 - .1)
            top = round((shape[0] - round(rec.height * gain)) / 2 - .1)
            cx = (gt[:, 0] + gt[:, 2] * .5) * gain + left
            cy = (gt[:, 1] + gt[:, 3] * .5) * gain + top
            xx_np = np.clip(np.floor(cx / stride).astype(np.int64), 0, x.shape[-1] - 1)
            yy_np = np.clip(np.floor(cy / stride).astype(np.int64), 0, x.shape[-2] - 1)
            xx = torch.as_tensor(xx_np, dtype=torch.long, device=x.device)
            yy = torch.as_tensor(yy_np, dtype=torch.long, device=x.device)
            center_weights = weights[index, :, :, yy, xx]
            mean_weights = pooled_weights[index, :, :, yy, xx]
            entropy = lambda w: -(w * w.clamp_min(1e-12).log()).sum(1)
            ent, mean_ent = entropy(center_weights), entropy(mean_weights)
            divergence = js(center_weights[0], center_weights[1])
            mean_divergence = js(mean_weights[0], mean_weights[1])
            # Transfer compact sampled arrays ONCE per image, not thousands of
            # individual CUDA scalars. Do not retain full maps or GPU tensors.
            small = torch.stack([base_square[index, yy, xx].clamp_min(0).sqrt(),
                                 *[t[index, yy, xx].clamp_min(0).sqrt() for t in old_square],
                                 *[t[index, yy, xx].clamp_min(0).sqrt() for t in new_square],
                                 *[t[index, yy, xx] for t in dot],
                                 divergence, mean_divergence, ent[0], ent[1], mean_ent[0], mean_ent[1]], 0).cpu().numpy()
            local_routes = torch.stack((center_weights, mean_weights), 0).cpu().numpy()
            sizes = np.sqrt(gt[:, 2] * gt[:, 3]) * 640 / max(rec.height, rec.width)
            for j in range(len(gt)):
                key = (int(rec.image_id), j, int(level))
                if key in self._keys:
                    raise RuntimeError('Duplicate image/GT/level statistics; possible warm-up contamination')
                self._keys.add(key)
                input_rms = _number(small[0, j])
                row = dict(image_id=int(rec.image_id), gt_index=j, level=f'P{level + 2}',
                           cls=int(gt[j, 5]), size=_number(sizes[j]),
                           input_shape=list(shape), feature_cell=[int(xx_np[j]), int(yy_np[j])],
                           input_center_xy=[_number(cx[j]), _number(cy[j])],
                           input_RMS=input_rms, task_JS_nats=_number(small[7, j]),
                           task_JS_3x3_nats=_number(small[8, j]))
                for task, name in enumerate(('cls', 'reg')):
                    old_rms = _number(small[1 + task, j])
                    new_rms = _number(small[3 + task, j])
                    denominator = old_rms * new_rms
                    cosine = None if denominator == 0. else _number(small[5 + task, j] / denominator)
                    if cosine is not None:
                        cosine = max(-1., min(1., cosine))  # round-off only
                    row.update({f'{name}_entropy_nats': _number(small[9 + task, j]),
                                f'{name}_entropy_3x3_nats': _number(small[11 + task, j]),
                                f'{name}_depth_weights': [_number(v) for v in local_routes[0, task, :, j]],
                                f'{name}_depth_weights_3x3': [_number(v) for v in local_routes[1, task, :, j]],
                                f'{name}_e40_residual_RMS': old_rms,
                                f'{name}_new_readout_RMS': new_rms,
                                f'{name}_e40_to_input_ratio': _ratio(old_rms, input_rms),
                                f'{name}_new_to_input_ratio': _ratio(new_rms, input_rms),
                                f'{name}_new_to_e40_ratio': _ratio(new_rms, old_rms),
                                f'{name}_residual_cosine': cosine})
                self.rows.append(row)

    def summary(self, geometry_rows=None):
        """Group descriptions, optionally conditioned on NORMAL final IoU>=.75.

        This is GT coverage, not AP/one-to-one matching; conditioning on an
        outcome cannot identify which parameter or statistic caused the outcome.
        """
        geometry = None
        if geometry_rows is not None:
            if not isinstance(geometry_rows, (list, tuple)):
                raise RuntimeError('Expected a list of normal geometry GT rows')
            geometry = {}
            for row in geometry_rows:
                key = (int(row['image_id']), int(row['gt_index']))
                if key in geometry:
                    raise RuntimeError('Duplicate normal geometry GT key')
                # E47 and the D45 worker use flat final_correct; the earlier
                # D38 audit used nested iou.top500. Do not silently substitute
                # raw correctness or any-class geometry for either field.
                value = row.get('final_correct')
                if value is None and isinstance(row.get('iou'), dict):
                    value = row['iou'].get('top500', row['iou'].get('final_correct'))
                if value is None or not math.isfinite(float(value)):
                    raise RuntimeError('Missing finite normal final-correct IoU')
                geometry[key] = (int(row['cls']), float(row['size']), float(value))
            for row in self.rows:
                key = (row['image_id'], row['gt_index'])
                if key not in geometry:
                    raise RuntimeError('Statistics GT absent from normal geometry')
                cls, size, _ = geometry[key]
                if cls != row['cls'] or not math.isclose(size, row['size'], rel_tol=1e-8, abs_tol=1e-8):
                    raise RuntimeError('Statistics/geometry GT class or size mismatch')

        def summarize(items):
            result = dict(gt_count=len(items))
            for key in SCALARS:
                values = [row[key] for row in items if row[key] is not None]
                result[key] = dict(valid_count=len(values), undefined_count=len(items) - len(values),
                                   mean=None if not values else _number(np.mean(values)),
                                   median=None if not values else _number(np.median(values)))
            for key in VECTORS:
                result[key] = None if not items else [_number(v) for v in np.mean([r[key] for r in items], axis=0)]
            return result

        output = {'_metadata': dict(spatial_neighborhood='3x3 cropped at borders; RMS and local cosine',
                    route_weights='center and 3x3 mean distributions reported separately; entropy in nats',
                    residual_definition='actual E40 base_outputs minus x; captured FP32 new readout sum',
                    scientific_boundary='Descriptive only. Negative residual cosine is not proof of harm. '
                                        'Outcome-conditioned groups do not establish root causes.',
                    geometry_conditioned=geometry is not None,
                    outcome_definition='normal final_correct IoU>=0.75 before official prediction-ignore filtering; '
                                       'not official matching/AP and not one-to-one recall')}
        for level in ('P2', 'P3'):
            level_rows = [r for r in self.rows if r['level'] == level]
            groups = [(name, [r for r in level_rows if lo <= r['size'] < hi]) for lo, hi, name in SIZE_GROUPS]
            groups += [('class_' + str(c), [r for r in level_rows if r['cls'] == c]) for c in range(1, 11)]
            for name, items in groups:
                output[level + '/' + name] = summarize(items)
                if geometry is not None:
                    for success, label in ((True, 'normal_final_correct75'), (False, 'normal_final_missing75')):
                        chosen = [r for r in items if (geometry[(r['image_id'], r['gt_index'])][2] >= .75) == success]
                        output[level + '/' + name + '/' + label] = summarize(chosen)
        return output
