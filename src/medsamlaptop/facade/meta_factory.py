import torch
import pathlib
from typing import Dict, Tuple
from torch.utils.data import (
    Dataset
    , DataLoader
)
# TODO: redo imports
from medsamlaptop import datasets as medsamlaptop_datasets
from medsamlaptop import dataloaders as medsamlaptop_dataloaders
from medsamlaptop import models as medsamlaptop_models
from medsamlaptop import losses as medsamlaptop_losses
from medsamlaptop import schedulers as medsamlaptop_schedulers
from medsamlaptop import optimizers as medsamlaptop_optimizers
from medsamlaptop import trainers as medsamlaptop_trainers
from medsamlaptop import constants
from medsamlaptop.trainers.products.interface import BaseTrainer
from medsamlaptop.datasets.products.interface import DatasetInterface
from medsamlaptop.models.products.interface import SegmentAnythingModelInterface

# TODO: Replace dictionnary by something better that involves setting the factory types explicitely
class MetaFactory:
    AVAILABLE_BUILDS = (
        (constants.TRAIN_RUN_TYPE, constants.EDGE_SAM_NAME)
        , (constants.TRAIN_RUN_TYPE, constants.MED_SAM_LITE_NAME)
        , (constants.TRAIN_RUN_TYPE, constants.MED_SAM_NAME)
        , (constants.ENCODER_DISTILLATION_RUN_TYPE, constants.EDGE_SAM_NAME)
        , (constants.ENCODER_DISTILLATION_RUN_TYPE, constants.MED_SAM_LITE_NAME)
        , (constants.EDGE_SAM_STAGE_2_DISTILLATION_RUN_TYPE, constants.EDGE_SAM_NAME)
    )
    def __init__(self
                 , run_type: str
                 , model_type: str
                 , data_root: pathlib.Path
                 , lr: float
                 , weight_decay: float
                 , device: torch.device
                 , kwargs_loss: Dict
                 , kwargs_dataloaders: Dict
                 , pretrained_checkpoint: pathlib.Path) -> None:
        assert (run_type, model_type) in self.AVAILABLE_BUILDS, f"Build type not implemented: {(run_type, model_type)} - authorized: {self.AVAILABLE_BUILDS}"
        self.build_type = (run_type, model_type)
        self.model_type = model_type
        self.run_type = run_type
        
        # params
        self.lr = lr
        self.weight_decay = weight_decay
        self.data_root = data_root
        self.device = device
        self.kwargs_loss = kwargs_loss
        self.kwargs_dataloaders = kwargs_dataloaders
        self.pretrained_checkpoint = pretrained_checkpoint
    
        # init factories
        self.factories = self._get_factories()

    def _get_factories(self):
        match self.build_type:
            case (constants.TRAIN_RUN_TYPE, constants.EDGE_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedEdgeSAMFactory()
                    , "dataset": medsamlaptop_datasets.Npy1024Factory(self.data_root)
                    , "dataloader": medsamlaptop_dataloaders.SamDataloaderFactory(**self.kwargs_dataloaders)
                    , "optimizer": medsamlaptop_optimizers.SamOptimizerFactory(self.lr, self.weight_decay)
                    , "scheduler": medsamlaptop_schedulers.SamSchedulerFactory()
                    , "loss": medsamlaptop_losses.SamLossFactory(**self.kwargs_loss)
                    , "trainer": medsamlaptop_trainers.SamTrainerFactory(self.device)
                }
            case (constants.TRAIN_RUN_TYPE, constants.MED_SAM_LITE_NAME):
                return {
                    "model": medsamlaptop_models.MedSAMLiteFactory()
                    , "dataset": medsamlaptop_datasets.Npy256Factory(self.data_root)
                    , "dataloader": medsamlaptop_dataloaders.SamDataloaderFactory(**self.kwargs_dataloaders)
                    , "optimizer": medsamlaptop_optimizers.SamOptimizerFactory(self.lr, self.weight_decay)
                    , "scheduler": medsamlaptop_schedulers.SamSchedulerFactory()
                    , "loss": medsamlaptop_losses.SamLossFactory(**self.kwargs_loss)
                    , "trainer": medsamlaptop_trainers.SamTrainerFactory(self.device)
                }
            case (constants.TRAIN_RUN_TYPE, constants.MED_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedSAMFactory()
                    , "dataset": medsamlaptop_datasets.Npy1024Factory(self.data_root)
                    , "dataloader": medsamlaptop_dataloaders.SamDataloaderFactory(**self.kwargs_dataloaders)
                    , "optimizer": medsamlaptop_optimizers.SamOptimizerFactory(self.lr, self.weight_decay)
                    , "scheduler": medsamlaptop_schedulers.SamSchedulerFactory()
                    , "loss": medsamlaptop_losses.SamLossFactory(**self.kwargs_loss)
                    , "trainer": medsamlaptop_trainers.SamTrainerFactory(self.device)
                }
            case (constants.ENCODER_DISTILLATION_RUN_TYPE, constants.EDGE_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedEdgeSAMFactory()
                    , "dataset": medsamlaptop_datasets.Distillation1024Factory(self.data_root)
                    # NOTE: for now same as Sam
                    , "dataloader": medsamlaptop_dataloaders.SamDataloaderFactory(**self.kwargs_dataloaders)
                    , "optimizer": medsamlaptop_optimizers.SamOptimizerFactory(self.lr, self.weight_decay)
                    , "scheduler": medsamlaptop_schedulers.SamSchedulerFactory()          
                    , "loss": medsamlaptop_losses.EncoderDistillationLossFactory()
                    , "trainer": medsamlaptop_trainers.EncoderDistillerFactory(self.device) # TODO
                }
            case (constants.ENCODER_DISTILLATION_RUN_TYPE, constants.MED_SAM_LITE_NAME):
                return {
                    "model": medsamlaptop_models.MedSAMLiteFactory()
                    , "dataset": medsamlaptop_datasets.Distillation256Factory(self.data_root)
                    , "dataloader": medsamlaptop_dataloaders.SamDataloaderFactory(**self.kwargs_dataloaders)
                    , "optimizer": medsamlaptop_optimizers.SamOptimizerFactory(self.lr, self.weight_decay)
                    , "scheduler": medsamlaptop_schedulers.SamSchedulerFactory()          
                    , "loss": medsamlaptop_losses.EncoderDistillationLossFactory()
                    , "trainer": medsamlaptop_trainers.EncoderDistillerFactory(self.device) # TODO
                }
            case (constants.EDGE_SAM_STAGE_2_DISTILLATION_RUN_TYPE, constants.EDGE_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedEdgeSAMFactory()
                    , "dataset": medsamlaptop_datasets.Stage2Distillation1024Factory(self.data_root)
                    # NOTE: for now same as Sam
                    , "dataloader": medsamlaptop_dataloaders.SamDataloaderFactory(**self.kwargs_dataloaders)
                    , "optimizer": medsamlaptop_optimizers.SamOptimizerFactory(self.lr, self.weight_decay)
                    , "scheduler": medsamlaptop_schedulers.SamSchedulerFactory()          
                    , "loss": medsamlaptop_losses.EdgeSamStage2LossFactory(seg_loss_weight=0.5, ce_loss_weight=0.5) # TODO: weights should be set outside
                    , "trainer": medsamlaptop_trainers.EdgeSamStage2DistillationFactory(self.device) # TODO
                }
            case _:
                raise NotImplementedError(f"Build type {self.build_type} not implemented")

    def create_model(self) -> torch.nn.Module:
        # NOTE: we add here the load of the checkpoint
        # however, if in the future it should be different between models
        # then it is important to do it at the ModelFactory model
        # which is sensible
        model: SegmentAnythingModelInterface = self.factories["model"].create_model()
        if(self.pretrained_checkpoint.is_file()):
            self.load_model_checkpoint_from_path(model)

        # TODO: may be this is not the best thing...
        if(self.run_type == constants.ENCODER_DISTILLATION_RUN_TYPE):
            return model.get_encoder()
        elif(self.run_type == constants.EDGE_SAM_STAGE_2_DISTILLATION_RUN_TYPE):
            model.freeze_prompt_encoder()
        return model

    def create_datasets(self) -> Tuple[DatasetInterface, DatasetInterface]:
        dataset = self.factories["dataset"].create_dataset()
        train_set, valid_set = torch.utils.data.random_split(dataset
                                      , [0.8, 0.2]
                                      , generator=torch.Generator().manual_seed(42))
        return (train_set, valid_set)
    
    def create_dataloader(self, dataset: Dataset) -> DataLoader:
        return self.factories["dataloader"].create_dataloader(dataset)
    
    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return self.factories["optimizer"].create_optimizer(model)

    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return self.factories["scheduler"].create_scheduler(optimizer)

    def create_loss(self) -> torch.nn.Module:
        return self.factories["loss"].create_loss()

    def create_trainer(self
                       , model: torch.nn.Module
                       , optimizer: torch.optim.Optimizer
                       , loss_fn: torch.nn.Module
                       , lr_scheduler: torch.optim.lr_scheduler.LRScheduler)-> BaseTrainer:
        trainer_factory: medsamlaptop_trainers.TrainerFactoryInterface = self.factories["trainer"]
        return trainer_factory.create_trainer(
            model
            , optimizer
            , loss_fn
            , lr_scheduler
        )

    def load_model_checkpoint_from_path(self, model: torch.nn.Module):
        # TODO: may be add try / except
        checkpoint = torch.load(
                self.pretrained_checkpoint,
                map_location="cpu"
        )
        self.load_model_checkpoint(model, checkpoint)

    def load_model_checkpoint(self, model: torch.nn.Module, checkpoint):
        model.load_state_dict(checkpoint, strict=True)
