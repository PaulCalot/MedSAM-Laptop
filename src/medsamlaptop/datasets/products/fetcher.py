import pathlib
import re
import logging
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)
# TODO : move this out 

processed_root_dir = "PROCESSED/"
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
    , "ISIC2018": "Dermoscopy/ISIC2018"
    , "IDRiD": "Fundus/IDRiD"
    , "PAPILA": "Fundus/PAPILA"
    , "CDD-CESM": "Mammography/CDD-CESM"
    , "AMOSMR": "MR/AMOSMR"
    , "BraTS_FLAIR": "MR/BraTS_FLAIR"
    , "BraTS_T1": "MR/BraTS_T1"
    , "BraTS_T1CE": "MR/BraTS_T1CE"
    , "CervicalCancer": "MR/CervicalCancer"
    , "crossmoda": "MR/crossmoda"
    , "Heart": "MR/Heart"
    , "ISLES2022_ADC": "MR/ISLES2022_ADC"
    , "ISLES2022_DWI": "MR/ISLES2022_DWI"
    , "ProstateADC": "MR/ProstateADC"
    , "ProstateT2": "MR/ProstateT2"
    , "QIN-PROSTATE-Lesion": "MR/QIN-PROSTATE-Lesion"
    , "QIN-PROSTATE-Prostate": "MR/QIN-PROSTATE-Prostate"
    , "SpineMR": "MR/SpineMR"
    , "WMH_FLAIR": "MR/WMH_FLAIR"
    , "WMH_T1": "MR/WMH_T1"
    , "Chest-Xray-Masks-and-Labels": "XRay/Chest-Xray-Masks-and-Labels"
    , "COVID-19-Radiography-Database": "XRay/COVID-19-Radiography-Database"
    , "COVID-QU-Ex-lungMask_CovidInfection": "XRay/COVID-QU-Ex-lungMask_CovidInfection"
    , "COVID-QU-Ex-lungMask_Lung": "XRay/COVID-QU-Ex-lungMask_Lung"
    , "Pneumothorax-Masks": "XRay/Pneumothorax-Masks"
}

def remove_non_alphanumeric(s):
    return ''.join([char for char in s if char.isalnum()])

id_to_root_path_mapper = {}
for key, index in datasets_paths.items():
    id_to_root_path_mapper[remove_non_alphanumeric(key).lower()] = key
id_to_root_path_mapper["unittest_CT"] = "CT_Abd"

# TODO: fix this problem - should not have the same in anatomy (my bad)
NAME_PATTERN = r'(?P<dataset>[a-zA-Z0-9]+)_(?P<modality>[a-zA-Z0-9]+)_(?P<anatomy>[a-zA-Z0-9\ ]+)_(?P<id>[a-zA-Z0-9]+)'

class IncorrectDataNameFormat(Exception):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"Name was: {self.name}, was expecting: {NAME_PATTERN}"

def dictionary_of_lists_to_list_of_dictionaries(dict_of_lists):
    return [dict(zip(dict_of_lists.keys(), values)) for values in zip(*dict_of_lists.values())]

class JsonDataParser:
    def __init__(self
                 , json_data_file: Dict
                 , root_path: pathlib.Path) -> None:
        assert("data" in json_data_file.keys())
        self.data = json_data_file["data"]
        self.root_dir = root_path

    def get_paths(self, subfolders):
        assert(len(subfolders) > 0)
        paths = {}
        for name in self.data:
            (status, info_name) = self.make_paths_for_name(
                                        self.parse_name(name)
                                        , subfolders)
            if(status == "KO"):
                logger.warning(f"Failed to load data for name = {name} (error = {info_name})")
            elif(status == "OK_2D"):
                paths[name] = info_name
            elif(status == "OK_3D"):
                lst_info_name = dictionary_of_lists_to_list_of_dictionaries(info_name)
                subfolder_1 = subfolders[0]
                for elem in lst_info_name:
                    name_ = elem[subfolder_1].stem
                    paths[name_] = elem
            else:
                logger.warning(f"Unexpected status: {status}")
        return paths

    def parse_name(self, name: str):
        try:
            dataset_id = self.try_dataset_parser(name)
            return ("OK", {"name": name
                           , "dataset": dataset_id})
        except IncorrectDataNameFormat as e:
            return ("KO", e)
        
    def make_paths_for_name(self, input: Tuple[str, Dict | Exception], subfolders: List[str]):
        status, info = input
        if(status == "KO" or isinstance(info, Exception)):
            return input

        assert(all( [expected_key in info.keys() for expected_key in ("dataset", "name")] ))
        dataset_id = info["dataset"]
        name = info["name"]

        if(not dataset_id in id_to_root_path_mapper):
            return ("KO", f"Missing dataset key in mapper: {dataset_id}")
        dataset_name = id_to_root_path_mapper[dataset_id]
        paths = {}
        status_ = None
        for subfolder in subfolders:
            path = ((self.root_dir / processed_root_dir) / dataset_name) / subfolder
            # path = (((self.root_dir / processed_root_dir) / dataset_name) / subfolder) / f"{name}.npy"
            status, output = self.handle_path(path, name)
            if(status == "KO"): 
                return ("KO", output)
            elif((status_ is not None) and (status_ != status)):
                return ("KO", f"Mismatch in subfolder types: {path} / {name}")
            else:
                status_ = status
            paths[subfolder] = output 
        return (status_, paths)
    
    def handle_path(self, root_dir: pathlib.Path, name: str):
        path_2D = root_dir / f"{name}.npy"
        if(path_2D.is_file()):
            return ("OK_2D", path_2D)
        
        # check if 3D
        list_path_3D = list(root_dir.glob(f"{name}_*.npy")) # TODO: not ideal
        # list_path_3D = list(root_dir.glob(name + r"_[0-9]{3}.npy"))
        if(len(list_path_3D) > 0):
            return ("OK_3D", list_path_3D)
        return ("KO", f"Data does not exist: {root_dir} - {name}")

    def try_dataset_parser(self, name:str):
        match = re.match(NAME_PATTERN, name)
        if(match):
            return match.groupdict()["dataset"]
        raise IncorrectDataNameFormat(name)
