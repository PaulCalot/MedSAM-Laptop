from medsamtools.user import get_path_to_data
import os
import tqdm

command_pattern = """python scripts/teacher_inference.py \
    --data_root {} \
    --pretrained_checkpoint med-sam/medsam_vit_b.pth \
    --model_type MedSAM \
    --device cuda:0 \
    --run_type encoder-inference
"""

root_dir = get_path_to_data()
lst = (root_dir / "PROCESSED").glob("*")

for dataset_directory_path in tqdm.tqdm(lst):
    if(not dataset_directory_path.is_dir()):
        continue
    print(f"\n #-----------{dataset_directory_path}---------# \n")
    os.system(command_pattern.format(dataset_directory_path))
