import abc
import torch

class SchedulerFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        pass