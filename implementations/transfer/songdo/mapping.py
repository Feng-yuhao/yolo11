"""Fixed semantic score mapping, before native best-class NMS. No learned state."""
from common import require


def map_scores(raw):
    import torch
    require(raw.ndim == 3 and raw.shape[1] == 14 and raw.shape[2] != 6,
            'Expected source10 decoded BCN predictions, NOT a target4 or end2end head')
    require(raw.dtype == torch.float32 and bool(torch.isfinite(raw).all()), 'Need finite FP32 source output')
    require(bool(((raw[:, 4:] >= 0) & (raw[:, 4:] <= 1)).all()), 'Expected sigmoid scores')
    # Allocate new storage: native NMS can modify its input box coordinates.
    car = torch.maximum(raw[:, 4+3], raw[:, 4+4])
    scores = torch.stack((car, raw[:, 4+8], raw[:, 4+5], raw[:, 4+9]), dim=1)
    return torch.cat((raw[:, :4], scores), dim=1)


def mapped_nms(raw):
    from ultralytics.utils.ops import non_max_suppression
    return non_max_suppression(map_scores(raw), conf_thres=.001, iou_thres=.7,
                               nc=4, multi_label=False, agnostic=False, max_det=500,
                               max_nms=30000)


def inverse_boxes(rows, input_hw, original_hw):
    """Independent inverse for pinned 8.3.0 LetterBox/scale_boxes, including portrait."""
    output = rows.clone()
    ih, iw = input_hw
    oh, ow = original_hw
    gain = min(ih / oh, iw / ow)
    pad_x = round((iw - ow * gain) / 2 - .1)
    pad_y = round((ih - oh * gain) / 2 - .1)
    output[:, :4] -= output.new_tensor([pad_x, pad_y, pad_x, pad_y])
    output[:, :4] /= gain
    output[:, [0, 2]] = output[:, [0, 2]].clamp(0, ow)
    output[:, [1, 3]] = output[:, [1, 3]].clamp(0, oh)
    return output
