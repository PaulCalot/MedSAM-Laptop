import pathlib
from .interface import DatasetFactoryInterface
from ..products import Stage2DistillationDataset

class Stage2Distillation1024Factory(DatasetFactoryInterface):
    def __init__(self, path_to_data: pathlib.Path) -> None:
        self.path_to_data = path_to_data

    def create_dataset(self) -> Stage2DistillationDataset:
        return Stage2DistillationDataset(
            data_root=self.path_to_data
            , image_size=1024
            , gt_size=256
            , bbox_shift=5
            , data_aug=True
        )
