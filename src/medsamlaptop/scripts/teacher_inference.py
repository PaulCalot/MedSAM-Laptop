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

# local packages
from medsamlaptop import constants
from medsamlaptop import facade as medsamlaptop_facade
from medsamlaptop.models.products.interface import SegmentAnythingModelInterface
from medsamlaptop.data.products.npy import NpyDataset # this is a concrete class
from medsamtools import user

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
    "--model_type", default="MedSAM",
    help="Type of backbone model : MedSAM. Default: MedSAM",
    choices=["MedSAM"]
)
parser.add_argument(
    "--device", type=str, default="cuda:0",
    help="Device to train on."
)

args = parser.parse_args()
args.data_root = user.get_path_to_data() / args.data_root
args.output_dir = args.data_root / "encoder_gts"
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

meta_factory = medsamlaptop_facade.MetaFactory(
    run_type= constants.TRAIN_RUN_TYPE # for now
    , model_type=args.model_type
    , data_root=args.data_root
)
facade = medsamlaptop_facade.MetaSegmentAnythingPipeFacade(meta_factory)

if args.pretrained_checkpoint.is_file():
    facade.load_checkpoint_from_path(args.pretrained_checkpoint)
else:
    print("Checkpoint not found. Terminating.")
    sys.exit(1) # abnormal termination

model: SegmentAnythingModelInterface = facade.get_model().eval()
dataset: NpyDataset = facade.get_dataset()
assert(isinstance(dataset, NpyDataset))
with torch.no_grad():
    model = model.to(args.device)
    for i in tqdm.tqdm(range(len(dataset))):
        # TODO: this output is a dictionnary, may be we want an interface
        output = dataset[i]
        img_tensor = output["image"].to(args.device)
        name = output["image_name"]
        image_embedding = model.image_encoder(torch.unsqueeze(img_tensor, 0))  # output: (1, 256, 64, 64)
        image_embedding = image_embedding.cpu().numpy()
        np.save(args.output_dir / name # name already contains .npy
                , image_embedding)