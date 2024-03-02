import pathlib
from .interface import DatasetFactoryInterface
from ..products import NpyDataset

class Npy1024Factory(DatasetFactoryInterface):
    def __init__(self, path_to_data: pathlib.Path) -> None:
        self.path_to_data = path_to_data

    def create_dataset(self) -> NpyDataset:
        return NpyDataset(
            data_root=self.path_to_data
            , image_size=1024
            , gt_size=256
            , bbox_shift=5
            , data_aug=True
        )
