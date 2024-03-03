from .interface import TrainerFactoryInterface
from ..products.edgesam_stage2_distillation import EdgeSamStage2Distillation
import torch

class EdgeSamStage2DistillationFactory(TrainerFactoryInterface):
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def create_trainer(self
                       , model: torch.nn.Module
                       , optimizer: torch.optim.Optimizer
                       , loss_fn: torch.nn.Module
                       , lr_scheduler: torch.optim.lr_scheduler.LRScheduler) -> EdgeSamStage2Distillation:
        return EdgeSamStage2Distillation(
            model
            , optimizer
            , loss_fn
            , lr_scheduler
            , self.device
        )
