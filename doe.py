# from medsamtools.user import get_path_to_data
import os
import tqdm

commands = [
"""python scripts/train_one_gpu.py \
    --data_root 'data_frac=0.1_ts=2024-03-15 18:12:13.169376.json' \
    --pretrained_checkpoint edge-sam/edge_sam_3x.pth \
    --work_dir ENCODER_DISTILLATION \
    --num_workers 4 \
    --batch_size 8 \
    --num_epochs 10 \
    --device cuda:0 \
    --model_type edgeSAM \
    --run_type encoder-distillation
""",
"""
python scripts/train_one_gpu.py \
    --data_root 'data_frac=0.1_ts=2024-03-15 18:12:13.169376.json' \
    --pretrained_checkpoint little-med-sam/lite_medsam.pth \
    --work_dir ENCODER_DISTILLATION \
    --num_workers 4 \
    --batch_size 8 \
    --num_epochs 10 \
    --device cuda:0 \
    --model_type medSAMLite \
    --run_type encoder-distillation
"""
]
# root_dir = get_path_to_data()
for command in tqdm.tqdm(commands):
    print(f"\n #-----------NEW TRAINING---------# \n")
    os.system(command)
    # break
