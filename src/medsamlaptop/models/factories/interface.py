import abc
from ..products import SegmentAnythingModelInterface
# TODO: at some point we may want to make it one step further cleaner 
# by adding interface to encoder(s) / decoder
# and factory
# this way, we are certain it will be clean

class FactoryInterface(abc.ABC):
    @abc.abstractmethod
    def create_model(self) -> SegmentAnythingModelInterface:
        pass