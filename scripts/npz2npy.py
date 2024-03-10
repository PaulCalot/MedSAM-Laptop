import glob
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
import numpy as np
import cv2
import multiprocessing as mp
from tqdm import tqdm

from medsamtools import user
from utils.name_mapper import (
    NameMapper
    , get_handler
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
    #name = npz_path.split(".npz")[0]
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

                np.save(img_root_saving_dir / f"{new_name}_{i:03d}.npy", img_01)
                np.save(gts_root_saving_dir / f"{new_name}_{i:03d}.npy", gt_i)
        else: ## 2D image
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
            np.save(img_root_saving_dir / f"{new_name}.npy", img_01)
            np.save(gts_root_saving_dir / f"{new_name}.npy", gts)
            # np.save(join(npy_dir, "imgs", name + ".npy"), img_01)
            # np.save(join(npy_dir, "gts", name + ".npy"), gts)
    except Exception as e:
        print(e)
        print(npz_path)

# Target : 
# Pivot format <dataset a-zA-Z0-9>_<modality>_<anatomy>_<id> 
# with id relative to the dataset (for now)
# we need converter for each dataset 
# What should be the image ID, should it be unique ?
# What size should it be given ?
# we will create a unique ID of image, which means we will allow up to 1M images
# and will create a mapper : old name => new name
# or rather: new unique id => dataset, id etc.
# There should be some way to make sure this is repeatalbe (that is id are unique throughout each system)
# what would it allow us to do, that to have unique ID ?

# modality id : 4 0-9

datasets_paths = {
    "hc18": "US/hc18"
    , "Breast-Ultrasound": "US/Breast-Ultrasound"
    , "autoPET": "PET/autoPET"
    , "Intraretinal-Cystoid-Fluid": "OCT/Intraretinal-Cystoid-Fluid"
    , "NeurIPS22CellSeg": "Microscopy/NeurIPS22CellSeg"
    , "m2caiSeg": "Endoscopy/m2caiSeg"
    , "Kvasir-SEG": "Endoscopy/Kvasir-SEG"
    , "CholecSeg8k": "Endoscopy/CholecSeg8k"
    , "CT_AbdTumor": "CT/CT_AbdTumor"
    , "AbdomenCT1K": "CT/AbdomenCT1K"
    , "AMOD": "CT/AMOS"
}

possibilities_CholecSeg_8K = [
    "AbdominalWall"
    , "Blood"
    , "ConnectiveTissue"
    , "CysticDuct"
    , "Fat"
    , "Gallbladder"
    , "GastrointestinalTract"
    , "Grasper"
    , "HepaticVein"
    , "LhookElectrocautery"
    , "Liver"
]

possibilities_CT_AbdTmor = [
        "Adrenal"
        , "case" # this one is weird => is it another kind of format ???
        , "colon" 
        , "hepaticvessel"
        , "liver"
        , "pancreas"
        , "PETCT" # this one is weird too
]

# To use the NameMapper, we need to create, for each data point, the dictionnaru with various inputs
# for now, we will not use any mapper to associate to an modality an id
# nor will we convert the images
# we just want to generate the csv

root_path = user.get_path_to_data()
mapper_name = "MAPPER"
root_saving_directory = root_path / "PROCESSED"
if __name__ == "__main__":
    inputs = []
    with NameMapper(root_path / f"{mapper_name}.csv") as mapper:
        for name_dataset, rel_path in datasets_paths.items():
            print(f"Processing {name_dataset}...")
            path_to_dataset = root_path / rel_path
            lst_paths = sorted(path_to_dataset.glob("*.npz"))
            dataset_handler = get_handler(name_dataset)
            for k, path_ in tqdm(enumerate(lst_paths)):
                try:
                    (status, dico) = dataset_handler(path_.stem)
                except Exception as e:
                    print(f"[{k}] Exception: {name_dataset} - {path_.stem} - {e}")
                    continue
                if(status == "KO"):
                    print(f"[{k}] FAIL: {name_dataset} - {path_.stem} (reason: {dico})")
                else:
                    new_name = mapper.get_new_name(**dico)
                inputs.append( # npz_path, npy_root_path, new_name
                    (path_, root_saving_directory / name_dataset, new_name)
                )
    # npz_dir = "train_npz"
    # npy_dir = "train_npy"
    num_workers = 8
    # do_resize_256 = False # whether to resize images and masks to 256x256
    # makedirs(npy_dir, exist_ok=True)
    # makedirs(join(npy_dir, "imgs"), exist_ok=True)
    # makedirs(join(npy_dir, "gts"), exist_ok=True)
    # npz_paths = glob.glob(join(npz_dir, "**/*.npz"), recursive=True)
    with mp.Pool(num_workers) as p:
        r = list(tqdm(p.imap(convert_npz_to_npy, inputs), total=len(inputs)))
    
            
