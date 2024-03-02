import abc
import torch

class OptimizerFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass