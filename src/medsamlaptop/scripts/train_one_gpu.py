import os
import torch
import datetime
import argparse
import pathlib

# local packages
from medsamlaptop import facade as medsamlaptop_facade
from medsamlaptop import models as medsamlaptop_models
from medsamlaptop import trainers
from medsamlaptop import losses
from medsamlaptop import data as medsamlaptop_data
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
    help="Type of backbone model : edgeSAM, medSAMLite. Default: medSAMLite"
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
    script_utils.checks.perform_dataset_sanity_check(args.data_root)

if(args.model_type=="edgeSAM"):
    model_factory: medsamlaptop_models.ModelFactoryInterface = medsamlaptop_models.MedEdgeSAMFactory()
    dataset_factory: medsamlaptop_data.DatasetFactoryInterface = medsamlaptop_data.Npy1024Factory(args.data_root)
elif(args.model_type=="medSAMLite"):
    model_factory  = medsamlaptop_models.MedSAMLiteFactory()
    dataset_factory = medsamlaptop_data.Npy256Factory(args.data_root)
facade = medsamlaptop_facade.SegmentAnythingPipeFacade(
                model_factory
                , dataset_factory)

if args.pretrained_checkpoint.is_file():
    facade.load_checkpoint_from_path(args.pretrained_checkpoint)

model = facade.get_model()
print(f"MedSAM Lite size: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=args.weight_decay,
)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)

train_dataset = facade.get_dataset()
train_loader = torch.utils.data.DataLoader(train_dataset
                                , batch_size=args.batch_size
                                , shuffle=True
                                , num_workers=args.num_workers
                                , pin_memory=True)

loss_fn = losses.SAMLoss(
    args.seg_loss_weight
    , args.ce_loss_weight
    , args.iou_loss_weight
)

if args.resume.is_file():
    print(f"Resuming from checkpoint {args.resume}...", end=" ")
    checkpoint = Checkpoint.load(args.resume)
    facade.load_checkpoint(checkpoint.model_weights)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    start_epoch = checkpoint.epoch
    best_loss = checkpoint.loss
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    best_loss = 1e10

trainer = trainers.MedSamTrainer(
    model
    , train_loader
    , None # val_loader
    , optimizer
    , lr_scheduler
    , loss_fn
    , device=args.device
)

saving_dir = args.work_dir / "{}_training".format(datetime.datetime.now().strftime('%Y%m%d_%H%M'))
saving_dir.mkdir()

print(f"New training, will save at: {saving_dir}")

trainer.train(
    saving_dir
    , num_epochs=args.num_epochs
    , start_epoch=start_epoch
    , best_loss=1e10
)
