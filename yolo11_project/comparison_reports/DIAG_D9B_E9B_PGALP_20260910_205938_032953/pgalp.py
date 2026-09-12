"""Assignment-guided PGALP used by the E9c causal probe.

The inference graph is intentionally identical to E9b PGALP. The only extra
state is a transient reliability logit consumed by the training criterion.
"""

from __future__ import annotations

from .pgalp import PGALP


class AGPGALP(PGALP):
    """PGALP that exposes its one-channel reliability logit to the loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reliability_logits = None

    def routing(self, p2, p3, low_small):
        import torch
        import torch.nn.functional as F

        p3_aligned = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        high_energy = (p2 - low_small).abs().mean(dim=1, keepdim=True)
        guide = torch.cat((self.p2_reduce(p2), self.p3_reduce(p3_aligned), high_energy), dim=1)
        logits = self.router(guide)
        self._reliability_logits = logits[:, :1]
        margin = self.reliability_margin
        reliability = margin + (1.0 - 2.0 * margin) * torch.sigmoid(logits[:, :1])
        scale_weights = torch.softmax(logits[:, 1:], dim=1)
        return reliability, scale_weights

    def pop_reliability_logits(self):
        logits = self._reliability_logits
        self._reliability_logits = None
        return logits

    def clear_reliability_logits(self):
        self._reliability_logits = None

    def clear_prior_logits(self):
        self.clear_reliability_logits()


__all__ = ("AGPGALP",)
