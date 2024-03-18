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

def count_2D_npy_images(input_):
    """
    Convert npz files to npy files for training

    Parameters
    ----------
    npz_path : str
        Name of the npz file to be converted
    """
    npz_path = input_
    try:
        npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        imgs = npz["imgs"]
        gts = npz["gts"]
        if len(gts.shape) > 2: ## 3D image
            return imgs.shape[0]
        else:
            return 1
    except Exception as e:
        print(e)
        print(npz_path)
        return 0
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
            tupl = (root_npz_directory / datasets_paths[row["dataset"]]) / "{}.npz".format(row["old_name"])
            inputs.append(tupl)
        return inputs

    inputs = generate_inputs_from_df(df)
    with open(root_path / f"data_frac={args.fraction}_ts={datetime.datetime.now()}.json", "w") as f:
        json.dump(
                {"data": df["new_name"].to_list()}
                , f
        )
    num_workers = 8
    with mp.Pool(num_workers) as p:
        r = list(tqdm(p.imap(count_2D_npy_images, inputs), total=len(inputs)))
        print(sum(r))

