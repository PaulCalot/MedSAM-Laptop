from ..factories import FactoryInterface
from ..products import SegmentAnythingModelInterface
import torch
import pathlib

class SegmentAnythingModelFacade:
    def __init__(self
                 , factory: FactoryInterface) -> None:
        self.factory = factory
        self.model: SegmentAnythingModelInterface = factory.create_model()

    # TODO: 
    # This facade will be useful for everything related to 
    # snapshotting the weights, reloading weights, and even plotting
        
    def load_checkpoint_from_path(self, path: pathlib.Path):
        # TODO: may be add try / except
        checkpoint = torch.load(
                path,
                map_location="cpu"
        )
        self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint, strict=True)

    def get_model(self):
        return self.model
    
    def set_model(self, model: SegmentAnythingModelInterface):
        self.model = model