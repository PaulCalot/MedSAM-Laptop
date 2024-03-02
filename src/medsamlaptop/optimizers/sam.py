from .interface import OptimizerFactoryInterface
import torch

class SamOptimizerFactory(OptimizerFactoryInterface):
    def __init__(self
                 , learning_rate: float = 0.00005
                 , weight_decay: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.AdamW:
        return torch.optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=self.weight_decay,
            )