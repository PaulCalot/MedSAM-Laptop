import torch
from .interface import SchedulerFactoryInterface

class SamSchedulerFactory(SchedulerFactoryInterface):
    def __init__(self) -> None:
        pass

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.9,
                    patience=5,
                    cooldown=0
                )
