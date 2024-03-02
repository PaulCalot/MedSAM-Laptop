import torch
import abc
import pathlib

# TODO
class DatasetInterface(abc.ABC, torch.utils.data.Dataset): 
    @abc.abstractmethod
    def __init__(self, data_root: pathlib.Path) -> None:
        pass