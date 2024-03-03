from ..products.encoder_distillation import EncoderDistillationLoss
from .interface import LossFactoryInterface

class EncoderDistillationLossFactory(LossFactoryInterface):
    def __init__(self) -> None:
        pass

    def create_loss(self) -> EncoderDistillationLoss:
        return EncoderDistillationLoss()