# MedSAM-Laptop
CVPR 2024 Competition which aims to adapt MedSAM to run on a CPU of a laptop.

## Installation
*Setup will be clean once code is in terminal phase*

The installation is the same as for MedSAM (litteMedSAM). 
Once the env is setup, you can simply do :
```shell
pip install -e .
```
From the root of this repo.

You also need to configure a configuration file, in the following named *user.cfg*. It should contain the following paths :
```cfg
[camerone]
path_to_data=/media/camerone/data-disk/DATA/CVPR-2024-MEDSAM-LAPTOP/
path_to_results=/media/camerone/data-disk/RESULTS/CVPR-2024-MEDSAM-LAPTOP/
path_to_pretrained_models=/media/camerone/data-disk/MODELS/
path_to_logs=/media/camerone/data-disk/LOGS/CVPR-2024-MEDSAM-LAPTOP/
```
Replace with your own path and make the file available from the environment variable under the name "USERCFG". It is highly encouragered to write in the *.bashrc* (or your equivalent), to make it permanent:
```
export USERCFG="/home/camerone/Documents/repertoires/user.cfg"
```
Replace with your own path.

## Use
```shell
python src/medsamlaptop/scripts/train_one_gpu.py \
    -data_root FLARE22Train/data/npy/CT_Abd/ \
    -pretrained_checkpoint little-med-sam/lite_medsam.pth \
    -work_dir DEV \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10 \
    -device cuda:0 \
    -model_type medSAMLite

python src/medsamlaptop/scripts/train_one_gpu.py \
    -data_root FLARE22Train/data/npy/CT_Abd/ \
    -pretrained_checkpoint edge-sam/edge_sam_3x.pth \
    -work_dir DEV \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10 \
    -device cuda:0 \
    -model_type edgeSAM
```

Replace with your own path, for the pretrained checkpoint and root of the data npy files.