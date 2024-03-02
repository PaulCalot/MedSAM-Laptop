from .interface import TrainerFactoryInterface
from ..products.sam import SamTrainer
import torch

class SamTrainerFactory(TrainerFactoryInterface):
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def create_trainer(self
                       , model: torch.nn.Module
                       , optimizer: torch.optim.Optimizer
                       , loss_fn: torch.nn.Module
                       , lr_scheduler: torch.optim.lr_scheduler.LRScheduler) -> SamTrainer:
        return SamTrainer(
            model
            , optimizer
            , loss_fn
            , lr_scheduler
            , self.device
        )
