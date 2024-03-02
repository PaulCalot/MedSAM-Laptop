from medsamlaptop import data as medsamlaptop_data
from medsamlaptop import models as medsamlaptop_models
from medsamlaptop import constants

class MetaFactory:
    AVAILABLE_BUILDS = (
        (constants.TRAIN_RUN_TYPE, constants.EDGE_SAM_NAME)
        , (constants.TRAIN_RUN_TYPE, constants.MED_SAM_LITE_NAME)
        , (constants.TRAIN_RUN_TYPE, constants.MED_SAM_NAME)
        , (constants.ENCODER_DISTILLATION_RUN_TYPE, constants.EDGE_SAM_NAME)
    )
    def __init__(self
                 , run_type
                 , model_type
                 , data_root) -> None:
        assert (run_type, model_type) in self.AVAILABLE_BUILDS, f"Build type not implemented: {(run_type, model_type)} - authorized: {self.AVAILABLE_BUILDS}"
        self.build_type = (run_type, model_type)
        self.model_type = model_type
        self.run_type = run_type
        self.data_root = data_root
        self.factories = self._get_factories()

    def _get_factories(self):
        match self.build_type:
            case (constants.TRAIN_RUN_TYPE, constants.EDGE_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedEdgeSAMFactory()
                    , "dataset": medsamlaptop_data.Npy1024Factory(self.data_root)
                }
            case (constants.TRAIN_RUN_TYPE, constants.MED_SAM_LITE_NAME):
                return {
                    "model": medsamlaptop_models.MedSAMLiteFactory()
                    , "dataset": medsamlaptop_data.Npy256Factory(self.data_root)
                }
            case (constants.TRAIN_RUN_TYPE, constants.MED_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedSAMFactory()
                    , "dataset": medsamlaptop_data.Npy1024Factory(self.data_root)
                }
            case (constants.TRAIN_RUN_TYPE, constants.EDGE_SAM_NAME):
                return {
                    "model": medsamlaptop_models.MedEdgeSAMFactory()
                    , "dataset": medsamlaptop_data.Distillation1024Factory(self.data_root)
                }
            case _:
                raise NotImplementedError(f"Build type {self.build_type} not implemented")

    def create_model(self):
        return self.factories["model"].create_model()
    
    def create_dataset(self):
        return self.factories["dataset"].create_dataset()