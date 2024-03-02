import abc
from torch.utils.data import (
    Dataset
    , DataLoader
)

class DataloaderFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_dataloader(self, dataset: Dataset) -> DataLoader:
        pass