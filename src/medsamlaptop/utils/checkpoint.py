import torch
import pathlib
from typing import Optional, Dict

class Checkpoint:
    def __init__(self
                 , model_weights: Dict[str, torch.Tensor]
                 , epoch: int
                 , optimizer_state: Dict[str, torch.Tensor] # torch.optim.Optimizer
                 , loss: float
                 , best_loss: float) -> None:
        self.model_weights = model_weights
        self.epoch = epoch
        self.optimizer_state = optimizer_state
        self.loss = loss
        self.best_loss = best_loss

    @staticmethod
    def load(path: pathlib.Path):
        return Checkpoint(**torch.load(path))
    
    def save(self, path: pathlib.Path):
        checkpoint = {
            "model": self.model_weights,
            "epoch": self.epoch,
            "optimizer": self.optimizer_state,
            "loss": self.loss,
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, path)
