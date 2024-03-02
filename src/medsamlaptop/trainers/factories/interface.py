import abc
import torch
from ..products.interface import BaseTrainer

class TrainerFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_trainer(self
                       , model: torch.nn.Module
                       , optimizer: torch.optim.Optimizer
                       , loss_fn: torch.nn.Module
                       , lr_scheduler: torch.optim.lr_scheduler.LRScheduler) -> BaseTrainer:
        pass