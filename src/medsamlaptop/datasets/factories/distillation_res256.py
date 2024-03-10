import pathlib
from .interface import DatasetFactoryInterface
from ..products import EncoderDistillationDataset

class Distillation256Factory(DatasetFactoryInterface):
    def __init__(self, path_to_data: pathlib.Path) -> None:
        self.path_to_data = path_to_data

    def create_dataset(self) -> EncoderDistillationDataset:
        return EncoderDistillationDataset(
            self.path_to_data
            , data_root=self.path_to_data.parent
            , image_size=256
        )
