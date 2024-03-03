from ..products.edgesam_stage2_loss import EdgeSamStage2Loss 
from .interface import LossFactoryInterface

class EncoderDistillationLossFactory(LossFactoryInterface):
    def __init__(self
                 , seg_loss_weight: float
                 , ce_loss_weight: float) -> None:
        self.seg_loss_weight = seg_loss_weight
        self.ce_loss_weight = ce_loss_weight

    def create_loss(self) -> EdgeSamStage2Loss:
        return EdgeSamStage2Loss(
            self.seg_loss_weight
            , self.ce_loss_weight
        )