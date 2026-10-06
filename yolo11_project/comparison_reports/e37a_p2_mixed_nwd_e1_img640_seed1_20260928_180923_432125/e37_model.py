"""Original E1 graph and tensor names; only its training criterion differs."""
from ultralytics.nn.tasks import DetectionModel
from e37_loss import MixedNWDDetectionLoss


class E37DetectionModel(DetectionModel):
    def init_criterion(self):
        return MixedNWDDetectionLoss(self)
