"""Frozen-candidate experiment, not an end-to-end detector or a causal test."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

REVISION = 'd47_frozen_rank_v1'
FEATURE_DIM = 225  # reg64 + cls128 + distribution16 + geometry2 + level4 + class10 + score1
DFL_START = 192
BOUND = 2.0  # bound on correction in score-logit space; fixed, no val search
CONF = .001
SUPPORT = 1.0 / (1.0 + math.exp(-(math.log(CONF / (1-CONF)) - BOUND)))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def holdout(stem):
    # Group frames from the same video, not individual adjacent frames.
    group = stem.split('_d_')[0]
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10 == 0


def features(raw, maps, sample_index):
    """Identical feature construction for training and inference; no GT input."""
    rows, offset = [], 0
    for level, (reg, cls, logits) in enumerate(maps):
        r = reg[sample_index].flatten(1).T.float()
        c = cls[sample_index].flatten(1).T.float()
        require(r.shape[1] == 64 and c.shape[1] == 128, 'Pinned E40 tower channels differ')
        n = len(r)
        p = logits[sample_index, :64].float().reshape(4, 16, -1).softmax(1)
        top = p.topk(2, dim=1).values
        bins = torch.arange(16, device=p.device).reshape(1, 16, 1)
        mean = (p*bins).sum(1, keepdim=True)
        var = (p*(bins-mean).square()).sum(1)/225.
        entropy = -(p*p.clamp_min(1e-12).log()).sum(1)/math.log(16)
        dfl = torch.stack((top[:, 0], top[:, 1], entropy, var), 1).reshape(16, n).T
        piece = raw[:, offset:offset+n]
        s, labels = piece[4:].max(0)
        size = (piece[2:4].T.float() / [4., 8., 16., 32.][level]).clamp_min(1e-6).log()
        lv = F.one_hot(torch.full((n,), level, device=p.device), 4).float()
        cl = F.one_hot(labels, 10).float()
        score = torch.logit(s.float().clamp(1e-7, 1-1e-7))[:, None]/10.
        rows.append(torch.cat((r, c, dfl, size, lv, cl, score), 1))
        offset += n
    x = torch.cat(rows)
    require(offset == raw.shape[1] and x.shape[1] == FEATURE_DIM, 'Feature/anchor ordering mismatch')
    require(torch.isfinite(x).all().item(), 'Nonfinite frozen features')
    return x.detach()


def ious_xyxy(boxes, gt):
    if not len(boxes) or not len(gt):
        return np.zeros((len(boxes), len(gt)), np.float64)
    wh = np.maximum(0., np.minimum(boxes[:, None, 2:], gt[None, :, 2:]) -
                    np.maximum(boxes[:, None, :2], gt[None, :, :2]))
    inter = wh.prod(2)
    a = np.maximum(0., boxes[:, 2:]-boxes[:, :2]).prod(1)
    b = np.maximum(0., gt[:, 2:]-gt[:, :2]).prod(1)
    return inter/np.maximum(a[:, None]+b[None]-inter, 1e-12)


def targets(boxes, labels, eval_gt):
    """Original-image CONTINUOUS IoU, same predicted class. Not NWD or TAL targets."""
    require(boxes.shape == (len(labels), 4), 'Expected original-image xyxy')
    gt = eval_gt[(eval_gt[:, 4] == 0) & (eval_gt[:, 5] >= 1) & (eval_gt[:, 5] <= 10)]
    q = np.zeros(len(boxes), np.float32)
    owner = np.full(len(boxes), -1, np.int64)
    for cls in range(1, 11):
        ci = np.flatnonzero(labels == cls)
        gi = np.flatnonzero(gt[:, 5] == cls)
        if not len(ci) or not len(gi):
            continue
        g = gt[gi, :4].copy(); g[:, 2:] += g[:, :2]
        overlap = ious_xyxy(boxes[ci], g)
        best = overlap.argmax(1)
        exact_best = overlap[np.arange(len(ci)), best]
        q[ci] = exact_best
        # Tied duplicate GT excluded from pair supervision, quality target unchanged.
        unique = (np.abs(overlap-exact_best[:, None]) < 1e-8).sum(1) == 1
        ok = unique & (q[ci] >= .5)
        owner[ci[ok]] = gi[best[ok]]
    return q, owner


def sample_candidates(q, owner, score, rng):
    """At most 512 rows/image: foreground diversity + hard and random negatives."""
    pos = np.flatnonzero(q >= .5)
    priority = []
    for g in np.unique(owner[owner >= 0]):
        ids = np.flatnonzero(owner == g)
        priority.extend((ids[np.argmax(q[ids])], ids[np.argmax(score[ids])]))
    priority = np.unique(priority)
    if len(priority) > 256:
        priority = rng.choice(priority, 256, replace=False)
    extra = np.setdiff1d(pos, priority)
    extra = rng.choice(extra, min(len(extra), 256-len(priority)), replace=False)
    selected = np.concatenate((priority, extra))
    neg = np.flatnonzero(q < .5)
    hard = neg[np.argsort(-score[neg], kind='stable')[:128]]
    other = np.setdiff1d(neg, hard)
    random = rng.choice(other, min(len(other), 256-len(hard)), replace=False)
    return np.sort(np.concatenate((selected, hard, random))).astype(np.int64)


def pairs(q, owner):
    """At most four lower-quality comparisons per GT; no cross-instance ordering."""
    result = []
    for g in np.unique(owner[owner >= 0]):
        ids = np.flatnonzero(owner == g)
        high = ids[np.argmax(q[ids])]
        low = ids[q[high]-q[ids] >= .05]
        low = low[np.argsort(q[low], kind='stable')[:4]]
        result.extend((int(high), int(x)) for x in low)
    return np.asarray(result, np.int64).reshape(-1, 2)


class Ranker(nn.Module):
    def __init__(self, variant, mean, std):
        super().__init__()
        require(variant in ('full', 'dfl'), 'Wrong ranker variant')
        self.variant = variant
        start = 0 if variant == 'full' else DFL_START
        self.register_buffer('mean', torch.as_tensor(mean[start:], dtype=torch.float32))
        self.register_buffer('std', torch.as_tensor(std[start:], dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(47001)
            self.net = nn.Sequential(nn.Linear(len(self.mean), 64), nn.SiLU(),
                                     nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        start = 0 if self.variant == 'full' else DFL_START
        z = ((x[:, start:].float()-self.mean)/self.std).clamp(-10, 10)
        return self.net(z).squeeze(1)


def correction(logit):
    return BOUND*logit.tanh()


def corrected_score(score, delta):
    """Stable sigmoid(logit(score)+delta), with bitwise identity at delta=0."""
    t = delta.expm1()
    return score + score*(1-score)*t/(1+score*t)


def calibrate(raw, x, ranker):
    """Class-agnostic factor: original argmax class and all box values preserved."""
    result = raw.clone()
    score = raw[4:].amax(0)
    ids = torch.where(score > SUPPORT)[0]
    if len(ids):
        chunks = []
        for part in ids.split(4096):
            chunks.append(corrected_score(score[part], correction(ranker(x[part]))))
        new = torch.cat(chunks)
        factor = new/score[ids]
        result[4:, ids] = raw[4:, ids]*factor[None]
        # FP32 scaling may round two almost-equal classes into a tie. Preserve
        # the ORIGINAL winner (inference uses single-label argmax, not multi-label).
        labels = raw[4:, ids].argmax(0)
        part = result[4:, ids]
        tied = part >= new[None]
        tied[labels, torch.arange(len(ids), device=ids.device)] = False
        part = torch.where(tied, torch.nextafter(new, torch.zeros_like(new))[None], part)
        part[labels, torch.arange(len(ids), device=ids.device)] = new
        # At EXACT identity initialization retain every original class probability.
        identity = new == score[ids]
        part[:, identity] = raw[4:, ids[identity]]
        result[4:, ids] = part
    require(torch.equal(result[:4], raw[:4]), 'Calibration modified boxes')
    require(torch.equal(result[4:].argmax(0), raw[4:].argmax(0)), 'Calibration changed class identity')
    require(torch.isfinite(result).all().item(), 'Nonfinite calibrated prediction')
    return result


def objective(logits, score, q, pair):
    positive = q >= .5
    negative = ~positive
    terms = F.binary_cross_entropy_with_logits(logits, q, reduction='none')
    groups = [terms[m].mean() for m in (positive, negative) if m.any()]
    require(bool(groups), 'No supervised rows')
    quality = torch.stack(groups).mean()
    adjusted = torch.logit(score.clamp(1e-7, 1-1e-7)) + correction(logits)
    if len(pair):
        margin = adjusted[pair[:, 0]]-adjusted[pair[:, 1]]
        weight = q[pair[:, 0]]-q[pair[:, 1]]
        ordering = (F.softplus(-margin)*weight).sum()/weight.sum().clamp_min(1e-8)
        accuracy = (margin > 0).float().mean().detach()
    else:
        ordering = logits.sum()*0.
        accuracy = logits.new_tensor(0.)
    return quality + ordering, dict(quality=quality.detach(), ranking=ordering.detach(),
                                    pair_accuracy=accuracy, pairs=len(pair))
