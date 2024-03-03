from .interface import TrainerFactoryInterface
from ..products.encoder import EncoderDistiller
import torch

class EncoderDistillerFactory(TrainerFactoryInterface):
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def create_trainer(self
                       , model: torch.nn.Module
                       , optimizer: torch.optim.Optimizer
                       , loss_fn: torch.nn.Module
                       , lr_scheduler: torch.optim.lr_scheduler.LRScheduler) -> EncoderDistiller:
        return EncoderDistiller(
            model
            , optimizer
            , loss_fn
            , lr_scheduler
            , self.device
        )
