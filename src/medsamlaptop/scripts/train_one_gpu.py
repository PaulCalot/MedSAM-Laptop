import os
import torch
import datetime
import argparse
import pathlib

# local packages
from medsamlaptop import constants
from medsamlaptop import facade as medsamlaptop_facade
from medsamlaptop import trainers
from medsamlaptop import losses
from medsamlaptop.utils.checkpoint import Checkpoint
from medsamtools import user

# local imports
import utils as script_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=pathlib.Path, default=pathlib.Path("./data/npy"),
    help="Path to the npy data root."
)
parser.add_argument(
    "-pretrained_checkpoint", type=pathlib.Path, default=pathlib.Path("lite_medsam.pth"),
    help="Path to the pretrained Lite-MedSAM checkpoint."
)
parser.add_argument(
    "-resume", type=pathlib.Path, default=pathlib.Path('workdir/medsam_lite_latest.pth'),
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=pathlib.Path, default=pathlib.Path("./workdir"),
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=10,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=8,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-lr", type=float, default=0.00005,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."

)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)
parser.add_argument(
    "--model_type", default="medSAMLite",
    help="Type of backbone model",
    choices=["edgeSAM", "medSAMLite"]
)

args = parser.parse_args()
args.work_dir = user.get_path_to_results() / args.work_dir
args.work_dir.mkdir(exist_ok=True, parents=True)

args.resume = user.get_path_to_results() / args.resume
args.data_root = user.get_path_to_data() / args.data_root
args.pretrained_checkpoint = user.get_path_to_pretrained_models() / args.pretrained_checkpoint

print("Using paths:")
print("Saving results (workdir): {}".format(args.work_dir))
print("Checkpoint to resume: {}".format(args.resume))
print("Data: {}".format(args.data_root))
print("Pretrained: {}".format(args.pretrained_checkpoint))

torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

if args.sanity_check:
    print("SANITY CHECK...")
    # TODO: Check if can delete this as we now have the unit testing of the dataset
    # Actually, it is still required. Indeed, this is for checking if all data is fine before
    # training as to not "waste" 1 epoch to see that one image is missing the ground truth
    # It is not replaced with the unittest
    script_utils.checks.perform_dataset_sanity_check(args.data_root)

meta_factory = medsamlaptop_facade.MetaFactory(
    run_type= constants.TRAIN_RUN_TYPE # for now
    , model_type=args.model_type
    , data_root=args.data_root
    , lr=args.lr
    , weight_decay=args.weight_decay
    , device=torch.device(args.device)
    , kwargs_loss={
        "seg_loss_weight": args.seg_loss_weight
        , "ce_loss_weight": args.ce_loss_weight
        , "iou_loss_weight": args.iou_loss_weight
    }
    , pretrained_checkpoint=args.pretrained_checkpoint
)
facade = medsamlaptop_facade.TrainSegmentAnythingPipeFacade(meta_factory)

train_dataset = facade.get_dataset()

# TODO: change this ! Dataloader should also have its own factory
train_loader = torch.utils.data.DataLoader(train_dataset
                                , batch_size=args.batch_size
                                , shuffle=True
                                , num_workers=args.num_workers
                                , pin_memory=True)
if args.resume.is_file():
    print(f"Resuming from checkpoint {args.resume}...", end=" ")
    checkpoint = Checkpoint.load(args.resume)
    facade.load_checkpoint(checkpoint)
    start_epoch = checkpoint.epoch
    best_loss = checkpoint.loss
else:
    start_epoch = 0
    best_loss = 1e10

saving_dir = args.work_dir / "{}_training".format(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
saving_dir.mkdir()

facade.train(
    train_loader
    , saving_dir
    , num_epochs=args.num_epochs
    , start_epoch=start_epoch
    , best_loss=1e10
)
