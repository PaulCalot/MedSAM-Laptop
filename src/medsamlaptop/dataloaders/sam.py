from torch.utils.data import (
    Dataset
    , DataLoader
)
from .interface import DataloaderFactoryInterface

class SamDataloaderFactory(DataloaderFactoryInterface):
    def __init__(self
                 , batch_size: int
                 , num_workers: int
                 , shuffle: bool = True
                 , pin_memory: bool = True):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def create_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(dataset
                          , batch_size=self.batch_size
                          , num_workers=self.num_workers
                          , shuffle=self.shuffle
                          , pin_memory=self.pin_memory)
