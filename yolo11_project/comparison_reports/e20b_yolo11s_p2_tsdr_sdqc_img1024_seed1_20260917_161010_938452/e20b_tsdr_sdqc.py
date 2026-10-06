"""E20b head: the frozen E16a TSDR mechanism plus SDQC score calibration."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .tsdr import TSDRDetect

__all__ = ("TSDRSDQCDetect",)


class TSDRSDQCDetect(TSDRDetect):
    """Keep E16a regression-only P1 routing and add detached DFL quality prediction."""

    def __init__(self, nc=80, gate_hidden=16, max_gain=0.25, ch=()):
        super().__init__(nc=nc, gate_hidden=gate_hidden, max_gain=max_gain, ch=ch)
        hidden = 32
        self.quality_predictor = nn.Sequential(
            nn.Conv2d(17, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.kaiming_uniform_(self.quality_predictor[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.quality_predictor[0].bias)
        nn.init.zeros_(self.quality_predictor[2].weight)
        nn.init.constant_(self.quality_predictor[2].bias, math.log(9.0))
        self.quality_gamma_logit = nn.Parameter(torch.tensor(-2.0))
        self.no = self.nc + self.reg_max * 4 + 1

    @property
    def quality_gamma(self):
        return self.quality_gamma_logit.sigmoid()

    def _quality_features(self, box_logits, level_index):
        batch, _, height, width = box_logits.shape
        probability = box_logits.detach().float().view(batch, 4, self.reg_max, height, width).softmax(2)
        top2 = probability.topk(2, dim=2).values
        bins = torch.arange(self.reg_max, device=probability.device, dtype=probability.dtype).view(1, 1, -1, 1, 1)
        mean = (probability * bins).sum(2, keepdim=True)
        variance = (probability * (bins - mean).square()).sum(2) / float((self.reg_max - 1) ** 2)
        entropy = -(probability.clamp_min(1e-8).log() * probability).sum(2) / math.log(float(self.reg_max))
        statistics = torch.stack((top2[:, :, 0], top2[:, :, 1], entropy, variance), dim=2)
        statistics = statistics.reshape(batch, 16, height, width)
        level_value = -1.0 + 2.0 * float(level_index) / float(max(self.nl - 1, 1))
        level = statistics.new_full((batch, 1, height, width), level_value)
        return torch.cat((statistics, level), 1).to(dtype=box_logits.dtype)

    def forward(self, inputs):
        self._validate_inputs(inputs)
        p1, p2_sem, p3_sem, p4, p5 = inputs
        detail, logits, gate, gain = self.routing(p1, p2_sem, p3_sem)
        p2_box = p2_sem + gain * gate * detail
        self._tsdr_gate_logits = logits if self.training else None

        if self.collect_diagnostics:
            with torch.no_grad():
                residual = p2_box - p2_sem
                self._diagnostic = {
                    "gate_mean": float(gate.float().mean().cpu()),
                    "gate_std": float(gate.float().std(unbiased=False).cpu()),
                    "gate_min": float(gate.float().min().cpu()),
                    "gate_max": float(gate.float().max().cpu()),
                    "detail_rms": float(detail.float().square().mean().sqrt().cpu()),
                    "residual_rms": float(residual.float().square().mean().sqrt().cpu()),
                    "p2_rms": float(p2_sem.float().square().mean().sqrt().cpu()),
                }

        features = [p2_sem, p3_sem, p4, p5]
        outputs = []
        for index, feature in enumerate(features):
            box_feature = p2_box if index == 0 else feature
            box = self.cv2[index](box_feature)
            cls = self.cv3[index](feature)
            quality = self.quality_predictor(self._quality_features(box, index))
            outputs.append(torch.cat((box, cls, quality), 1))
        if self.training:
            return outputs
        prediction = self._inference(outputs)
        return prediction if self.export else (prediction, outputs)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([tensor.view(shape[0], self.no, -1) for tensor in x], 2)
        if self.dynamic or self.shape != shape:
            from ultralytics.utils.tal import make_anchors

            self.anchors, self.strides = (value.transpose(0, 1) for value in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls, quality = x_cat.split((self.reg_max * 4, self.nc, 1), 1)
        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h, grid_w = shape[2], shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            decoded = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            decoded = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        score = cls.sigmoid() * quality.sigmoid().pow(self.quality_gamma)
        return torch.cat((decoded, score), 1)
