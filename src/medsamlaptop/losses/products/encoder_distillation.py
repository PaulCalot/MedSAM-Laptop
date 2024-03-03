from torch.nn import (Module, MSELoss)

class EncoderDistillationLoss(Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = MSELoss()

    def forward(self, pred, truth):
        return self.loss(pred, truth)
