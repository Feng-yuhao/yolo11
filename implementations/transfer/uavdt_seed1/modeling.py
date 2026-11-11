"""Load original trained ten-class checkpoints; never construct a new target head."""
import hashlib
from common import SOURCE_CLASSES, require


def audit(model, key):
    import torch
    from ultralytics.nn.modules.head import Detect
    import spatial_task_head as e60
    import e58_spatial_task_head as e58
    head = model.model[-1]
    require(len(model.model) == 27 and head.nc == 10 and head.nl == 4 and head.reg_max == 16,
            'Wrong original source E1 topology')
    require(model.names == dict(enumerate(SOURCE_CLASSES)), 'Wrong source ten-class names/order')
    require(head.f == [25, 16, 19, 22] and model.stride.tolist() == [4., 8., 16., 32.], 'Wrong source levels/strides')
    require(not head.end2end, 'Do not substitute YOLO26/end2end')
    if key.startswith('e60_'):
        require(type(head) is e60.SpatialTaskDetect and head.enabled and head.experiment_variant == 'e60',
                'Wrong/disabled ST-MDRF source head')
        require(len(head.task_blocks) == 4 and
                sum(p.numel() for p in head.task_blocks.parameters()) == e60.NEW_PARAMETERS == 1494144,
                'Wrong E60 branch count')
        for i, block in enumerate(head.task_blocks):
            require(block.reduce[0].out_channels == 96 and
                    block.reduce[0].in_channels == (128,128,256,512)[i] and len(block.depths) == 3,
                    'Source width/depth changed')
            if i < 2:
                require(type(block) is e60.GroupResidualTaskBlock and len(block.refinements) == 3,
                        'Missing source P2/P3 group encoders')
                for refine in block.refinements:
                    require(type(refine) is e60.GroupResidualEncoder and len(refine.kernels) == 3 and
                            all(p.dtype == torch.float32 for p in refine.parameters()), 'FP32 sidecar lost')
            else:
                require(type(block) is e58.WideTaskBlock and not hasattr(block, 'refinements'), 'Not E60: E61/E62/E63 forbidden')
    else:
        require(key.startswith('e1_') and type(head) is Detect and not hasattr(head, 'task_blocks'), 'Not original E1')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite source weight')
    return dict(nc=10, classes=list(SOURCE_CLASSES), head=type(head).__module__+'.'+type(head).__name__,
                parameters=sum(p.numel() for p in model.parameters()), source_head_replaced=False)


def fingerprint(model):
    """Snapshot after inference-only fuse; catch parameter or BN-buffer updates."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
