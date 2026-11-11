# Ultralytics YOLO 🚀, AGPL-3.0 license
"""TASD fusion modules for the VisDrone P2 research branch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("TASDFuse", "TASDPriorFuse")


class TASDFuse(nn.Module):
    """Fuse ``[upsampled P3 semantics, backbone P2 details]`` without a bare Concat.

    This E3a implementation is deliberately self-contained: it is optimized only
    by the normal detection loss and does *not* claim GT heatmap supervision. A
    P2-conditioned 3x3 local router first calibrates compressed P3 semantics. The
    calibrated semantics then selects the high-frequency residual of P2. The two
    refined streams are concatenated so the following original C3k2 sees exactly
    the same channel count as in the E1 P2 baseline.

    Args:
        channels (list[int] | tuple[int, int]): Channels of P3 and P2 inputs.
        hidden (int): Compressed channels used by the local router.
        max_align_gain (float): Upper bound of the semantic residual scale.
        max_detail_gain (float): Upper bound of the detail residual scale.
        layer_scale (float): Small initial residual gain; keeps initialization
            close to the E1 bare-Concat baseline.
    """

    def __init__(
        self,
        channels,
        hidden=16,
        max_align_gain=0.5,
        max_detail_gain=0.5,
        layer_scale=1e-2,
    ):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError(f"TASDFuse expects two input channel counts, got {channels!r}.")
        semantic_channels, detail_channels = map(int, channels)
        hidden = int(hidden)
        if semantic_channels <= 0 or detail_channels <= 0:
            raise ValueError(f"Input channels must be positive, got {channels!r}.")
        if not 0 < hidden <= min(semantic_channels, detail_channels):
            raise ValueError(f"Invalid hidden={hidden} for channels={channels!r}.")
        if max_align_gain <= 0 or max_detail_gain <= 0:
            raise ValueError("Residual gain bounds must be positive.")
        if not 0 <= layer_scale < min(max_align_gain, max_detail_gain):
            raise ValueError("layer_scale must be non-negative and smaller than both gain bounds.")

        self.channels = (semantic_channels, detail_channels)
        self.hidden = hidden
        self.output_channels = semantic_channels + detail_channels
        self.max_align_gain = float(max_align_gain)
        self.max_detail_gain = float(max_detail_gain)
        self.has_gt_prior_supervision = False

        self.semantic_embed = Conv(semantic_channels, hidden, 1, 1)
        self.detail_embed = Conv(detail_channels, hidden, 1, 1)

        # Nine probabilities choose a local 3x3 semantic neighborhood at each
        # position. The P2 stream participates in predicting these probabilities.
        self.align_context = Conv(3 * hidden, hidden, 3, 1)
        self.align_logits = nn.Conv2d(hidden, 9, 1, 1, 0, bias=True)
        self.align_out = nn.Conv2d(hidden, semantic_channels, 1, 1, 0, bias=False)

        # The implicit foreground router is an ablation precursor to the later
        # explicitly supervised tiny-object prior. Zero logits start at M=0.5.
        self.prior_context = Conv(3 * hidden, hidden, 3, 1)
        self.prior_logits = nn.Conv2d(hidden, 1, 1, 1, 0, bias=True)

        self.align_gain_raw = nn.Parameter(
            torch.tensor(self._inverse_tanh_gain(layer_scale, self.max_align_gain), dtype=torch.float32)
        )
        self.detail_gain_raw = nn.Parameter(
            torch.tensor(self._inverse_tanh_gain(layer_scale, self.max_detail_gain), dtype=torch.float32)
        )

        nn.init.zeros_(self.align_logits.weight)
        nn.init.constant_(self.align_logits.bias, -2.0)
        self.align_logits.bias.data[4] = 2.0  # center-biased, near-identity local routing
        nn.init.zeros_(self.prior_logits.weight)
        nn.init.zeros_(self.prior_logits.bias)

    @staticmethod
    def _inverse_tanh_gain(value, maximum):
        """Return the raw scalar whose bounded tanh gain is ``value``."""
        if value == 0:
            return 0.0
        ratio = min(max(float(value) / float(maximum), -0.999999), 0.999999)
        return math.atanh(ratio)

    @staticmethod
    def _local_mix(feature, probabilities):
        """Apply a deterministic spatially varying 3x3 mixture to a small tensor."""
        height, width = feature.shape[-2:]
        padded = F.pad(feature, (1, 1, 1, 1), mode="constant", value=0.0)
        mixed = torch.zeros_like(feature)
        index = 0
        for row in range(3):
            for column in range(3):
                shifted = padded[..., row : row + height, column : column + width]
                mixed = mixed + probabilities[:, index : index + 1] * shifted
                index += 1
        return mixed

    def forward(self, x):
        """Return a refined, channel-preserving replacement for bare Concat."""
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("TASDFuse expects [upsampled_p3, backbone_p2].")
        semantic, detail = x
        if semantic.ndim != 4 or detail.ndim != 4:
            raise ValueError("TASDFuse inputs must be BCHW tensors.")
        if semantic.shape[1] != self.channels[0] or detail.shape[1] != self.channels[1]:
            raise ValueError(
                f"TASDFuse channel mismatch: {(semantic.shape[1], detail.shape[1])} vs {self.channels}."
            )
        if semantic.shape[0] != detail.shape[0] or semantic.shape[-2:] != detail.shape[-2:]:
            raise ValueError(f"TASDFuse shape mismatch: {tuple(semantic.shape)} vs {tuple(detail.shape)}.")

        semantic_small = self.semantic_embed(semantic)
        detail_small = self.detail_embed(detail)
        difference = torch.abs(semantic_small - detail_small)

        align_features = self.align_context(torch.cat((semantic_small, detail_small, difference), dim=1))
        # FP32 softmax avoids AMP underflow; cast back immediately to avoid
        # promoting the high-resolution feature computations to FP32.
        probabilities = self.align_logits(align_features).float().softmax(dim=1).to(semantic.dtype)
        semantic_mixed = self._local_mix(semantic_small, probabilities)
        align_correction = self.align_out(semantic_mixed - semantic_small)
        align_gain = self.max_align_gain * torch.tanh(self.align_gain_raw)
        semantic_aligned = semantic + align_gain.to(semantic.dtype) * align_correction

        padded_detail = F.pad(detail, (1, 1, 1, 1), mode="constant", value=0.0)
        detail_low = F.avg_pool2d(padded_detail, kernel_size=3, stride=1, padding=0)
        detail_high = detail - detail_low
        detail_low_small = self.detail_embed(detail_low)
        prior_features = self.prior_context(
            torch.cat((semantic_mixed, detail_low_small, torch.abs(semantic_mixed - detail_low_small)), dim=1)
        )
        prior_logits = self.prior_logits(prior_features)
        self._capture_prior_logits(prior_logits)
        prior = torch.sigmoid(prior_logits)
        detail_gain = self.max_detail_gain * torch.tanh(self.detail_gain_raw)
        detail_refined = detail + detail_gain.to(detail.dtype) * (2.0 * prior.to(detail.dtype) - 1.0) * detail_high

        return torch.cat((semantic_aligned, detail_refined), dim=1)

    def _capture_prior_logits(self, logits):
        """Optional hook used by the explicitly supervised E3b subclass."""


class TASDPriorFuse(TASDFuse):
    """E3b TASD fusion with a training-only tiny-object prior loss interface.

    Its inference computation and trainable parameters are identical to
    :class:`TASDFuse`.  The only difference is that the one-channel router
    logits are exposed to the registered E3b criterion and cleared immediately
    after the loss is computed.  The criterion supervises these logits using
    the *current augmented batch boxes*, so Mosaic and affine transforms stay
    spatially consistent with the feature map.
    """

    def __init__(self, channels, hidden=16, max_align_gain=0.5, max_detail_gain=0.5, layer_scale=1e-2):
        super().__init__(channels, hidden, max_align_gain, max_detail_gain, layer_scale)
        self.has_gt_prior_supervision = True
        self._last_prior_logits = None

    def _capture_prior_logits(self, logits):
        self._last_prior_logits = logits

    def pop_prior_logits(self):
        """Return and clear the most recent router logits without detaching."""
        logits = self._last_prior_logits
        self._last_prior_logits = None
        return logits

    def clear_prior_logits(self):
        """Drop any temporary forward tensor before deepcopy/checkpointing."""
        self._last_prior_logits = None
