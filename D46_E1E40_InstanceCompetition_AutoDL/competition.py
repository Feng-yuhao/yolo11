"""Pure diagnostic geometry. Continuous XYXY, NOT official AP matching."""
from collections import deque
import numpy as np


def iou(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    inter = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(2)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def maximum_matching(adj):
    """Exact maximum cardinality bipartite matching, iterative augmenting paths.

    Candidate IDs are global raw-anchor IDs. No greedy IoU approximation.
    """
    owner, assigned = {}, {}
    for root in range(len(adj)):
        queue, visited, parent = deque([root]), {root}, {}
        endpoint = None
        while queue and endpoint is None:
            g = queue.popleft()
            for c in adj[g]:
                c = int(c)
                if c in parent:
                    continue
                parent[c] = g
                if c not in owner:
                    endpoint = c
                    break
                old = owner[c]
                if old not in visited:
                    visited.add(old)
                    queue.append(old)
        if endpoint is not None:
            c = endpoint
            while True:
                g = parent[c]
                previous = assigned.get(g)
                owner[c], assigned[g] = g, c
                if previous is None:
                    break
                c = previous
    return assigned


def size_bin(s):
    return 'lt8' if s < 8 else '8_16' if s < 16 else '16_24' if s < 24 else '24_32' if s < 32 else 'ge32'


def raw_geometry(gt, gc, boxes, labels, scores, input_scale, level):
    """Bounded GT x 1024 chunk; retain all raw candidates (no conf prefilter).

    Best same-class argmax candidate is geometry-selected, not deployment-selected.
    Diagnostic selection is explicitly distinct from official matched detections.
    """
    n = len(gt)
    best, best_id = np.zeros(n), np.full(n, -1, dtype=int)
    best_any = np.zeros(n)
    adj, adj_any = [[] for _ in range(n)], [[] for _ in range(n)]
    for start in range(0, len(boxes), 1024):
        v = iou(gt, boxes[start:start + 1024])
        if not v.shape[1]:
            continue
        best_any = np.maximum(best_any, v.max(1))
        for g in range(n):
            adj_any[g].extend((np.flatnonzero(v[g] >= .75) + start).tolist())
        v *= gc[:, None] == labels[None, start:start + 1024]
        ids = v.argmax(1)
        val = v[np.arange(n), ids]
        improved = val > best
        best_id[improved], best[improved] = start + ids[improved], val[improved]
        for g in range(n):
            adj[g].extend((np.flatnonzero(v[g] >= .75) + start).tolist())
    centers = (gt[:, :2] + gt[:, 2:]) / 2
    wh = np.maximum(gt[:, 2:] - gt[:, :2], 1e-9)
    distances = np.linalg.norm((centers[:, None] - centers[None]) * input_scale, axis=2)
    np.fill_diagonal(distances, np.inf)
    rows = []
    for g in range(n):
        c = int(best_id[g])
        neighbors = np.flatnonzero((gc == gc[g]) & (distances[g] <= 16))
        j = int(neighbors[np.argmin(distances[g, neighbors])]) if len(neighbors) else -1
        r = dict(gt_index=g, cls=int(gc[g]), size=float(np.sqrt(wh[g].prod()) * input_scale),
                 density=int((distances[g] <= 16).sum()), nearest_same_class_gt=j,
                 raw_any_iou=float(best_any[g]), raw_correct_iou=float(best[g]), best_candidate=c)
        if c >= 0:
            b = boxes[c]
            bc, bw = (b[:2] + b[2:]) / 2, np.maximum(b[2:] - b[:2], 1e-9)
            error = bc - centers[g]
            r.update(best_score=float(scores[c]), level=int(level[c]),
                     center_dx_norm=float(error[0] / wh[g, 0]), center_dy_norm=float(error[1] / wh[g, 1]),
                     log_width_error=float(np.log(bw[0] / wh[g, 0])),
                     log_height_error=float(np.log(bw[1] / wh[g, 1])))
            r['same_class_neighbor_cosine'] = None
            r['neighborward_projection'] = None
            r['error_unit_direction'] = None
            r['neighbor_unit_direction'] = None
            if j >= 0:
                direction = centers[j] - centers[g]
                length = float(np.linalg.norm(error) * np.linalg.norm(direction))
                r['same_class_neighbor_cosine'] = float(error @ direction / length) if length > 1e-12 else None
                r['neighborward_projection'] = float(error @ direction / max(float(direction @ direction), 1e-12))
                if length > 1e-12:
                    r['error_unit_direction'] = (error/np.linalg.norm(error)).tolist()
                    r['neighbor_unit_direction'] = (direction/np.linalg.norm(direction)).tolist()
        r['size_bin'] = size_bin(r['size'])
        r['density_bin'] = '0' if r['density'] == 0 else '1_3' if r['density'] < 4 else 'ge4'
        rows.append(r)
    return rows, adj, adj_any


def stage_match(rows, adj, adj_any, stages):
    stats = {}
    for name, ids in stages.items():
        valid = None if ids is None else set(map(int, ids))
        a = adj if valid is None else [[c for c in edges if c in valid] for edges in adj]
        aa = adj_any if valid is None else [[c for c in edges if c in valid] for edges in adj_any]
        matched, matched_any = maximum_matching(a), maximum_matching(aa)
        stats[name] = dict(gt=len(rows), independent=sum(bool(e) for e in a),
                           maximum_one_to_one=len(matched), independent_any=sum(bool(e) for e in aa),
                           maximum_one_to_one_any=len(matched_any))
        for g, r in enumerate(rows):
            r[name + '_independent75'] = bool(a[g])
            r[name + '_matched75'] = g in matched
    # Which particular GT loses a cardinality tie is nonunique. Do not stratify
    # this chosen maximum assignment as if it were the unique failure identity.
    return stats


def nms_witnesses(network_boxes, labels, scores, pre_ids, kept_ids, threshold=.7):
    """First kept overlap witness in the pinned torchvision NMS output order.

    Compare network-coordinate class-offset float32 boxes, exactly as NMS does.
    This is NOT an invented causal suppressor for score ties/boundary ambiguity.
    It does not run NMS again or choose a different ordering.
    """
    import torch
    from torchvision.ops import box_iou
    pre = set(map(int, pre_ids))
    survivors = set(map(int, kept_ids))
    removed = np.array(sorted(pre - survivors), dtype=int)
    keep = np.array(kept_ids, dtype=int)
    if not len(keep):
        return {int(c): dict(witness=-1, ambiguous=True, reason='no_reconstructed_witness') for c in removed}
    offset = network_boxes + labels[:, None] * 7680
    result = {}
    for start in range(0, len(removed), 1024):
        ids = removed[start:start + 1024]
        values = box_iou(torch.from_numpy(offset[keep].astype(np.float32)),
                         torch.from_numpy(offset[ids].astype(np.float32))).numpy()
        valid = (values > threshold) & (labels[keep, None] == labels[None, ids])
        valid &= scores[keep, None] >= scores[None, ids]
        has = valid.any(0)
        first = valid.argmax(0)
        for k, c in enumerate(ids):
            if not has[k]:
                result[int(c)] = dict(witness=-1, ambiguous=True, reason='no_reconstructed_witness')
            else:
                s = int(keep[first[k]])
                gap = float(scores[s] - scores[c])
                overlap = float(values[first[k], k])
                result[int(c)] = dict(witness=s, overlap=overlap, score_gap=gap,
                                      ambiguous=gap == 0 or abs(overlap-threshold) < 1e-6,
                                      reason='tie_or_threshold' if gap == 0 or abs(overlap-threshold) < 1e-6 else 'strict_overlap')
    return result


def witness_events(gt, gc, boxes, scores, labels, witnesses):
    """Diagnostic own-GT attribution. Not the official score-ordered matcher.

    Include every NMS-removed candidate with IoU>=.75 to at least one same-class GT.
    The survivor must IoU>=.50 to its own class GT to claim cross-instance witness.
    """
    events = []
    ids = np.array(list(witnesses), dtype=int)
    for start in range(0, len(ids), 1024):
        sub = ids[start:start + 1024]
        v = iou(gt, boxes[sub]) * (gc[:, None] == labels[None, sub])
        for k, c in enumerate(sub):
            good = np.flatnonzero(v[:, k] >= .75)
            if not len(good):
                continue
            g = int(good[np.argmax(v[good, k])])
            lost_owner_ties = int(np.isclose(v[good, k], v[g, k], rtol=0, atol=1e-9).sum())
            w = witnesses[int(c)]
            s = w['witness']
            ev = dict(candidate=int(c), gt=g, lost_iou=float(v[g, k]), multi_gt_good=len(good),
                      lost_owner_ties=lost_owner_ties, **w)
            if s >= 0:
                u = iou(gt, boxes[[s]])[:, 0] * (gc == labels[s])
                j = int(u.argmax())
                witness_owner_ties = int(np.isclose(u, u[j], rtol=0, atol=1e-9).sum()) if u[j] >= .5 else 0
                ev.update(witness_gt=j if u[j] >= .5 else -1, witness_own_iou=float(u[j]),
                          witness_iou_to_lost_gt=float(u[g]),
                          witness_owner_ties=witness_owner_ties,
                          lower_quality_on_lost_gt=bool(u[g] < v[g, k]))
                ev['kind'] = ('ambiguous' if w['ambiguous'] else
                              'ambiguous_gt_owner' if lost_owner_ties > 1 or witness_owner_ties > 1 else
                              'same_instance' if j == g and u[j] >= .5 else
                              'cross_instance' if j != g and u[j] >= .5 else 'unassigned_witness')
            else:
                ev['kind'] = 'unresolved'
            events.append(ev)
    return events


def summarize(rows, images, events):
    slices = {}
    for r in rows:
        for key in ('all', r['size_bin'], r['size_bin'] + '/class_' + str(r['cls']) + '/density_' + r['density_bin']):
            slices.setdefault(key, []).append(r)
    summary = {}
    for key, group in slices.items():
        geometry_fail = [r for r in group if .5 <= r['raw_correct_iou'] < .75]
        eligible = [r for r in geometry_fail if r.get('same_class_neighbor_cosine') is not None]
        summary[key] = dict(gt=len(group), geometric_near_misses=len(geometry_fail),
                            neighbor_direction_eligible=len(eligible),
                            neighborward_fraction=(sum(r['same_class_neighbor_cosine'] > 0 for r in eligible)/len(eligible)) if eligible else None,
                            mean_neighbor_cosine=float(np.mean([r['same_class_neighbor_cosine'] for r in eligible])) if eligible else None,
                            raw_correct75=sum(r['raw_independent75'] for r in group)/len(group),
                            final_correct75=sum(r['final_independent75'] for r in group)/len(group))
    stages = {}
    for im in images:
        for name, stats in im['matching'].items():
            target = stages.setdefault(name, {k: 0 for k in stats})
            for k, v in stats.items():
                target[k] += v
    kinds = {}
    for ev in events:
        kinds[ev['kind']] = kinds.get(ev['kind'], 0) + 1
    distinct_lost = {(e['image_id'], e['gt']) for e in events if e['kind'] == 'cross_instance' and not e['gt_has_surviving75']}
    poor_witness_lost = {(e['image_id'], e['gt']) for e in events if e.get('lower_quality_on_lost_gt') and
                         not e['ambiguous'] and not e['gt_has_surviving75']}
    return dict(slices=summary, matching=stages, nms_good_candidate_events=kinds,
                cross_instance_unique_gt_without_nms_alternative=len(distinct_lost),
                poorer_witness_unique_gt_without_nms_alternative=len(poor_witness_lost),
                neighbor_direction_control=direction_control(rows),
                event_count=len(events), image_count=len(images),
                boundaries=['Maximum matching is score-independent geometric upper coverage, NOT achievable AP or official recall.',
                            'Matched GT identity can vary among equal-cardinality solutions; only image totals are compared.',
                            'Density/neighbor direction remain confounded; candidate geometry-selected bias is not training root cause.',
                            'NMS witness counts are candidate events, not recoverable unique GT; rows also record surviving alternatives.',
                            'Continuous geometry/argmax classes and pre-ignore candidates are distinct from official AP.'])


def direction_control(rows):
    """Descriptive permutation control, no automatic significance/causal verdict.

    Swap center-error directions only within image/class/2-input-pixel size band/
    density-bin of geometric near misses. Preserve neighbor directions and error
    direction marginals; singleton strata cannot contribute, report coverage.
    """
    eligible = [r for r in rows if .5 <= r['raw_correct_iou'] < .75 and r.get('error_unit_direction') is not None]
    groups = {}
    for r in eligible:
        key = (r['image_id'], r['cls'], int(r['size']//2), r['density_bin'])
        groups.setdefault(key, []).append(r)
    rng = np.random.RandomState(46001)
    actual_total, null_total, count, groups_used = 0., np.zeros(100), 0, 0
    for group in groups.values():
        if len(group) < 2:
            continue
        error = np.array([r['error_unit_direction'] for r in group])
        neighbor = np.array([r['neighbor_unit_direction'] for r in group])
        actual_total += float((error*neighbor).sum())
        for k in range(100):
            null_total[k] += float((error[rng.permutation(len(group))]*neighbor).sum())
        count += len(group); groups_used += 1
    return dict(eligible_near_misses=len(eligible), exchangeable_near_misses=count, strata_used=groups_used,
                excluded_singletons=len(eligible)-count, repetitions=100,
                observed_mean_cosine=actual_total/count if count else None,
                permuted_mean_cosine=float(null_total.mean()/count) if count else None,
                observed_minus_permuted=(actual_total-float(null_total.mean()))/count if count else None,
                boundary='Descriptive within-stratum direction control, not a causal test or a training gate; limited exchangeable coverage may make it inconclusive.')
