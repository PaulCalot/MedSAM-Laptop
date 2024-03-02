import torch
import abc

class SegmentAnythingModelInterface(torch.nn.Module, abc.ABC):
    def __init__(self):
        super(SegmentAnythingModelInterface, self).__init__()
    
    abc.abstractmethod
    def forward(self):
        pass

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Must return the masks
        """
        pass