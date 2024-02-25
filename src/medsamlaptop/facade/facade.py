from ..models import ModelFactoryInterface
from ..data import DatasetFactoryInterface
from ..models.products import SegmentAnythingModelInterface
from ..data.products import DatasetInterface

import torch
import pathlib

class SegmentAnythingPipeFacade:
    def __init__(self
                 , model_factory: ModelFactoryInterface
                 , data_factory: DatasetFactoryInterface) -> None:
        self.model: SegmentAnythingModelInterface = model_factory.create_model()
        self.dataset: DatasetInterface = data_factory.create_dataset()

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

    def get_dataset(self):
        return self.dataset