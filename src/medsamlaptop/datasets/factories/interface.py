import abc
from ..products import DatasetInterface

class DatasetFactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_dataset(self) -> DatasetInterface:
        pass