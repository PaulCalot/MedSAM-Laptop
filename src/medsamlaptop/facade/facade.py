import pathlib
import torch
from torch.utils.data.dataloader import DataLoader
from ..models import ModelFactoryInterface
from ..models.products import SegmentAnythingModelInterface
from ..datasets import DatasetFactoryInterface
from ..datasets.products import DatasetInterface
# TODO: add type for optimizer, etc.
from ..utils.checkpoint import Checkpoint
from .meta_factory import MetaFactory

class TrainSegmentAnythingPipeFacade:
    def __init__(self
                 , meta_factory: MetaFactory) -> None:
        self.model: SegmentAnythingModelInterface = meta_factory.create_model()
        self.dataset: DatasetInterface = meta_factory.create_dataset()
        self.dataloader: DataLoader = meta_factory.create_dataloader(self.dataset)
        self.optimizer = meta_factory.create_optimizer(self.model)
        self.scheduler = meta_factory.create_scheduler(self.optimizer)
        self.loss = meta_factory.create_loss()
        self.trainer = meta_factory.create_trainer(
            self.model
            , self.optimizer
            , self.loss
            , self.scheduler
        )
    
    def __str__(self) -> str:
        representation_parts = [
                "Training parameters:\n\t"
                , f"\tModel:{str(self.model.__class__)}"
                , f"\tDataset:{str(self.dataset)}"
                , f"\tOptimizer:{str(self.optimizer)}"
                , f"\tScheduler:{str(self.scheduler)}"
                , f"\tLoss:{str(self.loss)}"
                , f"\tTrainer:{str(self.trainer)}"]
        representation = "\n".join(representation_parts)
        return representation

    def load_checkpoint(self, checkpoint: Checkpoint):
        self.model.load_state_dict(checkpoint.model_weights, strict=True)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)

    def train(self
              , saving_dir: pathlib.Path
              , num_epochs: int
              , start_epoch: int
              , best_loss: float):
        self.trainer.train(
            self.dataloader
            , saving_dir
            , num_epochs
            , start_epoch
            , best_loss
        )
    # ------------- Setter and getter -------------- #
    def get_model(self):
        return self.model

    def set_model(self, model: SegmentAnythingModelInterface):
        self.model = model

    def get_dataset(self):
        return self.dataset

class InferSegmentAnythingPipeFacade:
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