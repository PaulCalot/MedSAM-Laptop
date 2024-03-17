"""
This script takes a dataset as inputs and apply a teacher model,
then saving its outputs for later use, typically in a distillation setup.
"""
import os
import torch
import datetime
import argparse
import pathlib
import sys
import numpy as np
import tqdm
from torch.utils.data import (
    DataLoader
)

# local packages
from medsamlaptop import constants
from medsamlaptop import facade as medsamlaptop_facade
from medsamlaptop import models as medsamlaptop_models
from medsamlaptop.models.products.interface import SegmentAnythingModelInterface
from medsamlaptop.datasets.products.npy import NpyDataset # this is a concrete class
from medsamlaptop import datasets as medsamlaptop_datasets
from medsamtools import user
from utils.my_async import AsyncArraySaver

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root", type=pathlib.Path, default=pathlib.Path("./data/npy"),
    help="Path to the npy data root."
)
parser.add_argument(
    "--pretrained_checkpoint", type=pathlib.Path, default=pathlib.Path("lite_medsam.pth"),
    help="Path to the pretrained checkpoint."
)
parser.add_argument(
    "--model_type", default=constants.MED_SAM_NAME,
    help="Type of backbone model.",
    choices=[constants.MED_SAM_NAME]
)
parser.add_argument(
    "--run_type", default=constants.FULL_INFERENCE_RUN_TYPE,
    help="Type of inference run.",
    choices=[constants.FULL_INFERENCE_RUN_TYPE, constants.ENCODER_INFERENCE_RUN_TYPE]
)
parser.add_argument(
    "--device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "--batch_size", type=int, default=8,
    help="Batch size for intererence"
)
parser.add_argument(
    "--num_workers", type=int, default=4,
    help="Number of workers to load data"
)
args = parser.parse_args()
args.data_root = user.get_path_to_data() / args.data_root
# TODO: make it better, here we suppose that a mistake is still a teacher_gts...
name_folder = "encoder_gts" if args.run_type == constants.ENCODER_INFERENCE_RUN_TYPE else "teacher_gts"
args.output_dir = args.data_root / name_folder
args.output_dir.mkdir(exist_ok=True, parents=True)
args.pretrained_checkpoint = user.get_path_to_pretrained_models() / args.pretrained_checkpoint

print("Using paths:")
print("Data: {}".format(args.data_root))
print("Outputs: {}".format(args.output_dir))
print("Pretrained: {}".format(args.pretrained_checkpoint))

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

if(args.model_type=="MedSAM"):
    model_factory: medsamlaptop_models.ModelFactoryInterface = medsamlaptop_models.MedSAMFactory()
    dataset_factory: medsamlaptop_datasets.DatasetFactoryInterface = medsamlaptop_datasets.Npy1024Factory(args.data_root)
else:
    print(f"{args.model_type} not implemented.")
    sys.exit(1)
facade = medsamlaptop_facade.InferSegmentAnythingPipeFacade(
                model_factory
                , dataset_factory)

if args.pretrained_checkpoint.is_file():
    facade.load_checkpoint_from_path(args.pretrained_checkpoint)
else:
    print("Checkpoint not found. Terminating.")
    sys.exit(1) # abnormal termination

model: SegmentAnythingModelInterface = facade.get_model().eval()
dataset: NpyDataset = facade.get_dataset()
assert(isinstance(dataset, NpyDataset))
dataloader = DataLoader(dataset
          , batch_size=args.batch_size
          , num_workers=args.num_workers
          , shuffle=False
          , pin_memory=True)
async_saver = AsyncArraySaver(args.output_dir, max_workers=args.num_workers)
with torch.no_grad():
    model = model.to(args.device)
    for batch_ in tqdm.tqdm(dataloader):
        img_tensors = batch_["image"].to(args.device)
        names = batch_["image_name"]
        if(args.run_type == constants.ENCODER_INFERENCE_RUN_TYPE):
            image_embedding = model.image_encoder(img_tensors)  # output: (1, 256, 64, 64)
            image_embedding = image_embedding.cpu().numpy()
            async_saver.save_arrays(image_embedding, names)
            # np.save(args.output_dir / name # name already contains .npy
            #         , image_embedding)
        elif(args.run_type == constants.FULL_INFERENCE_RUN_TYPE):
            boxes = batch_["bboxes"].to(args.device)
            (masks, iou_predictions) = model(img_tensor
                                             , boxes)
            masks = masks.cpu().numpy()
            async_saver.save_arrays(masks, names)
            # np.save(args.output_dir / name
            #         , masks)

