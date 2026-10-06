"""Scale-adaptive distribution-quality calibration head for E20a.

The module leaves the four E1 detection feature maps and the original
classification/regression towers unchanged.  It reads detached DFL
distributions, predicts one localization-quality logit per location, and uses
that quality only to calibrate class confidence.  Detaching the DFL statistics
prevents the auxiliary quality objective from changing box regression.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .head import Detect

__all__ = ("SDQCDetect",)


class SDQCDetect(Detect):
    """YOLO Detect head with a shared distribution-quality calibrator."""

    def __init__(self, nc=80, ch=()):
        super().__init__(nc=nc, ch=ch)
        # Four statistics for each of four box sides, plus one level code.
        quality_features = 4 * 4 + 1
        hidden = 32
        self.quality_predictor = nn.Sequential(
            nn.Conv2d(quality_features, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        # q starts at 0.9 and gamma at sigmoid(-2)=0.119.  Their product is
        # close to identity (0.9**0.119 ~= 0.9875), while gamma still receives
        # usable gradients.  The final convolution starts distribution-neutral.
        nn.init.kaiming_uniform_(self.quality_predictor[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.quality_predictor[0].bias)
        nn.init.zeros_(self.quality_predictor[2].weight)
        nn.init.constant_(self.quality_predictor[2].bias, math.log(9.0))
        self.quality_gamma_logit = nn.Parameter(torch.tensor(-2.0))

        # Training raw output has one extra quality channel.  Inference still
        # returns exactly 4+nc channels, so Ultralytics NMS remains unchanged.
        self.no = self.nc + self.reg_max * 4 + 1

    @property
    def quality_gamma(self):
        """Bound the learned calibration exponent to (0, 1)."""
        return self.quality_gamma_logit.sigmoid()

    def _quality_features(self, box_logits, level_index):
        """Return detached DFL confidence/uncertainty statistics as BCHW."""
        batch, _, height, width = box_logits.shape
        prob = box_logits.detach().float().view(batch, 4, self.reg_max, height, width).softmax(2)
        top2 = prob.topk(2, dim=2).values
        bins = torch.arange(self.reg_max, device=prob.device, dtype=prob.dtype).view(1, 1, -1, 1, 1)
        mean = (prob * bins).sum(2, keepdim=True)
        variance = (prob * (bins - mean).square()).sum(2) / float((self.reg_max - 1) ** 2)
        entropy = -(prob.clamp_min(1e-8).log() * prob).sum(2) / math.log(float(self.reg_max))
        # Per side: top-1, top-2, normalized entropy, normalized variance.
        features = torch.stack((top2[:, :, 0], top2[:, :, 1], entropy, variance), dim=2)
        features = features.reshape(batch, 16, height, width)
        if self.nl > 1:
            level_value = -1.0 + 2.0 * float(level_index) / float(self.nl - 1)
        else:
            level_value = 0.0
        level = features.new_full((batch, 1, height, width), level_value)
        return torch.cat((features, level), 1).to(dtype=box_logits.dtype)

    def forward(self, x):
        """Return raw box/class/quality tensors in train and calibrated scores in eval."""
        if self.end2end:
            raise RuntimeError("SDQCDetect is defined only for the standard one-to-many YOLO11 head")
        outputs = []
        for index in range(self.nl):
            box = self.cv2[index](x[index])
            cls = self.cv3[index](x[index])
            quality = self.quality_predictor(self._quality_features(box, index))
            outputs.append(torch.cat((box, cls, quality), 1))
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)

    def _inference(self, x):
        """Decode boxes and multiply class probability by learned localization quality."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            from ultralytics.utils.tal import make_anchors

            self.anchors, self.strides = (value.transpose(0, 1) for value in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls, quality = x_cat.split((self.reg_max * 4, self.nc, 1), 1)
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        calibrated = cls.sigmoid() * quality.sigmoid().pow(self.quality_gamma)
        return torch.cat((dbox, calibrated), 1)
