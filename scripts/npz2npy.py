from os.path import join, basename
import pandas as pd
import sys
from tqdm import tqdm
import numpy as np
import cv2
import multiprocessing as mp
import argparse
import pathlib
import json
import datetime
# local 
from medsamtools import user
from utils.name_mapper import (
    NameMapper
    , get_handler
    , datasets_paths
)


def resize_longest_side(image, target_length=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)
    
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image, target_length=256):
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_length - h
    padw = target_length - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))
    
    return image_padded

do_resize_256 = False
def convert_npz_to_npy(input_):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_path : str
        Name of the npz file to be converted
    """
    npz_path, npy_root_path, new_name = input_
    try:
        name = basename(npz_path).split(".npz")[0]
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        imgs = npz["imgs"]
        gts = npz["gts"]
        img_root_saving_dir = npy_root_path / "imgs"
        gts_root_saving_dir = npy_root_path / "gts"
        img_root_saving_dir.mkdir(parents=True, exist_ok=True)
        gts_root_saving_dir.mkdir(parents=True, exist_ok=True)
        if len(gts.shape) > 2: ## 3D image
            for i in range(imgs.shape[0]):
                new_name_img = img_root_saving_dir / f"{new_name}_{i:03d}.npy"
                new_name_gts = gts_root_saving_dir / f"{new_name}_{i:03d}.npy"
                if(new_name_img.is_file() and new_name_gts.is_file()):
                    continue 
                img_i = imgs[i, :, :]
                gt_i = gts[i, :, :]
                if do_resize_256:
                    img_i = resize_longest_side(img_i)
                    gt_i = cv2.resize(gt_i, (img_i.shape[1], img_i.shape[0]), interpolation=cv2.INTER_NEAREST)
                    img_i = pad_image(img_i)
                    gt_i = pad_image(gt_i)

                img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)

                img_01 = (img_3c - img_3c.min()) / np.clip(
                    img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)

                gt_i = gts[i, :, :]

                gt_i = np.uint8(gt_i)
                assert img_01.shape[:2] == gt_i.shape

                np.save(new_name_img, img_01)
                np.save(new_name_gts, gt_i)
        else: ## 2D image
            new_name_img = img_root_saving_dir / f"{new_name}.npy"
            new_name_gts = gts_root_saving_dir / f"{new_name}.npy"
            if(new_name_img.is_file() and new_name_gts.is_file()):
                return
            if len(imgs.shape) < 3:
                img_3c = np.repeat(imgs[:, :, None], 3, axis=-1)
            else:
                img_3c = imgs

            if do_resize_256:
                img_3c = resize_longest_side(img_3c)
                #gts = resize_longest_side(gts)
                gts = cv2.resize(gts, (img_3c.shape[1], img_3c.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_3c = pad_image(img_3c)
                gts = pad_image(gts)

            img_01 = (img_3c - img_3c.min()) / np.clip(
                img_3c.max() - img_3c.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            assert img_01.shape[:2] == gts.shape
            np.save(new_name_img, img_01)
            np.save(new_name_gts, gts)
    except Exception as e:
        print(e)
        print(npz_path)

root_path = user.get_path_to_data()
root_npz_directory = root_path / "NPZ"
root_saving_directory = root_path / "PROCESSED"
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mapper_path", type=str)
    argparser.add_argument("--fraction", type=float, default=1.0)
    args = argparser.parse_args()
    args.mapper_path = pathlib.Path(args.mapper_path).resolve() 

    if(not args.mapper_path.is_file()):
        sys.exit(1)

    df = pd.read_csv(args.mapper_path)
    df = df.sample(frac=args.fraction)

    def generate_inputs_from_df(df:pd.DataFrame):
        EXPECTED_COLUMNS = ["primary_key", "dataset", "modality", "anatomy", "target_task", "old_name", "new_name", "id"]
        assert(all( [col in df.columns for col in EXPECTED_COLUMNS] ))
        inputs = []
        for index, row in df.iterrows():
            tupl = (
                    (root_npz_directory / datasets_paths[row["dataset"]]) / "{}.npz".format(row["old_name"])
                    , root_saving_directory / row["dataset"]
                    , row["new_name"]
                    )
            inputs.append(tupl)
        return inputs

    inputs = generate_inputs_from_df(df)
    with open(root_path / f"data_frac={args.fraction}_ts={datetime.datetime.now()}.json", "w") as f:
        json.dump(
                {"data": df["new_name"].to_list()}
                , f
        )
    num_workers = 4
    with mp.Pool(num_workers) as p:
        r = list(tqdm(p.imap(convert_npz_to_npy, inputs), total=len(inputs)))
