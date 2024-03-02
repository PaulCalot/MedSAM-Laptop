import abc
import torch

class LossFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_loss(self) -> torch.nn.Module:
        pass