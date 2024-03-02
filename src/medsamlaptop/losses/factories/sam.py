from ..products.sam import SAMLoss
from .interface import LossFactoryInterface

class SamLossFactory(LossFactoryInterface):
    def __init__(self
                 , seg_loss_weight: float
                 , ce_loss_weight: float
                 , iou_loss_weight: float) -> None:
        self.seg_loss_weight = seg_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.iou_loss_weight = iou_loss_weight

    def create_loss(self) -> SAMLoss:
        return SAMLoss(self.seg_loss_weight
                       , self.ce_loss_weight
                       , self.iou_loss_weight)