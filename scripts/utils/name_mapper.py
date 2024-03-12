import re
import pandas as pd
import pathlib
from typing import Optional, Dict, Tuple

class NameMapper:
    # TODO: add getter for old name etc?
    # TODO: what of race conditions ?
    PK_COLUMNS = ["dataset", "old_name"]
    EXPECTED_COLUMNS = ["primary_key", "dataset", "modality", "anatomy", "target_task", "old_name", "new_name", "id"]

    def __init__(self
                 , path_to_mapper: pathlib.Path):
        assert path_to_mapper.suffix == ".csv", "Input path should be a .csv file" 
        self.path_to_mapper = path_to_mapper
        if(not self.path_to_mapper.is_file()):
            df_mapper = pd.DataFrame(columns=self.EXPECTED_COLUMNS)
        else:
            df_mapper = pd.read_csv(self.path_to_mapper)
        assert(all([ col in df_mapper.columns for col in self.EXPECTED_COLUMNS]))
        self.df_mapper = df_mapper

    def get_new_name(self
                , dataset: str
                , old_name: str
                , modality: Optional[str] = 'UNK'
                , anatomy: Optional[str] = 'UNK'
                , target_task: Optional[str] = 'UNK') -> str:
        # TODO: should we check for dataset /  old_name validity ?
        dataset = remove_non_alphanumeric(dataset).lower()
        old_name_formatted = remove_non_alphanumeric(old_name).lower()
        primary_key = f"{dataset}_{old_name_formatted}"
        if(primary_key in self.df_mapper["primary_key"].values):
            # NOTE in theory there should be only one
            new_name = self.df_mapper[self.df_mapper["primary_key"] == primary_key]["new_name"].iloc[0]
            return new_name

        new_id = self.generate_new_id()
        new_name = f"{dataset}_{modality}_{anatomy}_{new_id}"
        new_row = {
            "primary_key": primary_key
            , "dataset": dataset
            , "modality": modality
            , "anatomy": anatomy
            , "target_task": target_task
            , "old_name": old_name
            , "new_name": new_name
            , "id": new_id 
        }
        self.df_mapper.loc[self.df_mapper.shape[0]] = new_row
        return new_name

    def generate_new_id(self) -> str:
        # works like an auto-increment on database
        return "{:07d}".format(len(self.df_mapper)+1) # starts at 1

    def _save(self):
        self.df_mapper.to_csv(self.path_to_mapper)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._save()
        assert self.df_mapper["id"].nunique() == len(self.df_mapper), "ID is not unique (Race condition ?)"

    # def iterate(self):
    #     for index, rows in self.df_mapper.itterows():
    #         old_name, dataset, new_name =  

def get_handler(dataset):
    dataset_to_authorized_pattern = {
        "hc18": handle_hc18
        , "Breast-Ultrasound": handle_breastultrasound
        , "autoPET": handle_autopet
        , "Intraretinal-Cystoid-Fluid": handle_intraretinalcystoidfluid
        , "NeurIPS22CellSeg": handle_neurips22cellseg
        , "m2caiSeg": handle_m2caiseg
        , "Kvasir-SEG": handle_kvasirseg
        , "CholecSeg8k": handle_cholecseg8k
        , "CT_AbdTumor": handle_abdtumor
        , "AbdomenCT1K": handle_abdomenct1K
        , "AMOD": handle_amod
    }
    return dataset_to_authorized_pattern[dataset]

# TODO: clean that
# ---------------- UTILS 
dataset_to_authorized_pattern = {
    "hc18": [r"US_hc18_(?P<NUM>[0-9]{3})_(?P<RANK>[0-9]{0,1})HC"]
    , "Breast-Ultrasound": [r"US_Breast-Ultrasound_(?P<TYPE>benign|malignant|normal) \((?P<NUM>[0-9]+\))"]
    , "autoPET": [r"PET_Lesion_PETCT_(?P<ID>[a-zA-Z0-9]{10})"]
    , "Intraretinal-Cystoid-Fluid": [r"OCT_Intraretinal-Cystoid-Fluid_(?P<SOMETHING>\w)"]
    , "NeurIPS22CellSeg": [r"Microscopy_NeurIPS22CellSeg_cell_(?P<ID>[0-9]{5})"]
    , "m2caiSeg": [r"Endoscopy_m2caiSeg_(?P<ID>[0-9]{3})_(?P<FRAME>[0-9]+)"] # this is an hypothesis... from the da
    , "Kvasir-SEG": [r"Endoscopy_Kvasir-SEG_(?P<ID>[a-z0-9]{25})"]
    # TODO : we don't enforce from there the possiblitiees that we saw - we should do it afterwards
    , "CholecSeg8k": [r"Endoscopy_CholecSeg8k_(?P<ORGAN>[a-zA-Z]*)_video(?P<ID>[0-9]+)_frame_(?P<FRAME>[0-9]+)"]
    # TODO: once again, we are kind of lazy, in practice this dataset is already a merge without clean pivotal format
    , "CT_AbdTumor": [r"CT_AbdTumor_(?P<ORGAN>[a-zA-Z]+)_(?P<ID>[a-zA-Z0-9]+)"]
    , "AbdomenCT1K": [r"CT_AbdomenCT_Case_(?P<ID>[0-9]{5})"]
    , "AMOD": [r"CT_AMOS_amos_(?P<ID>[0-9]{4})"]
}

def handle_amod(name: str):
    patterns = dataset_to_authorized_pattern["AMOD"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "AMOD"
                    , "old_name": name
                    , "modality": "CT"
                    # TODO
                    # , "anatomy": ""
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_abdomenct1K(name: str):
    patterns = dataset_to_authorized_pattern["AbdomenCT1K"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "AbdomenCT1K"
                    , "old_name": name
                    , "modality": "CT"
                    # TODO
                    # , "anatomy": ""
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_abdtumor(name: str):
    patterns = dataset_to_authorized_pattern["CT_AbdTumor"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "AbdTumor"
                    , "old_name": name
                    , "modality": "CT"
                    # TODO
                    # , "anatomy": ""
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_cholecseg8k(name: str):
    patterns = dataset_to_authorized_pattern["CholecSeg8k"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "CholecSeg8k"
                    , "old_name": name
                    , "modality": "Endoscopy"
                    # TODO
                    # , "anatomy": ""
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_kvasirseg(name: str):
    patterns = dataset_to_authorized_pattern["Kvasir-SEG"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "Kvasir-SEG"
                    , "old_name": name
                    , "modality": "Endoscopy"
                    # TODO
                    # , "anatomy": ""
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_m2caiseg(name: str):
    patterns = dataset_to_authorized_pattern["m2caiSeg"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "m2caiSeg"
                    , "old_name": name
                    , "modality": "Endoscopy"
                    # TODO
                    # , "anatomy": ""
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_neurips22cellseg(name: str):
    patterns = dataset_to_authorized_pattern["NeurIPS22CellSeg"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "NeurIPS22CellSeg"
                    , "old_name": name
                    , "modality": "Microscopy"
                    # TODO
                    # , "anatomy": "retina"
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_intraretinalcystoidfluid(name: str):
    patterns = dataset_to_authorized_pattern["Intraretinal-Cystoid-Fluid"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "Intraretinal-Cystoid-Fluid"
                    , "old_name": name
                    , "modality": "OCT" 
                    , "anatomy": "retina"
                    # , "target_task": ""
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_autopet(name: str):
    patterns = dataset_to_authorized_pattern["autoPET"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "autoPET"
                    , "old_name": name
                    , "modality": "PET" # /CT" TODO: is this also CT ?
                    , "target_task": "automated PET lesion segmentation"
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_breastultrasound(name: str) -> Tuple[str, Dict|str]:
    patterns = dataset_to_authorized_pattern["Breast-Ultrasound"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            dico = match.groupdict()
            breast_type = dico["TYPE"]
            return ("OK", {
                    "dataset": "Breast-Ultrasound"
                    , "old_name": name
                    , "modality": "US"
                    , "anatomy": f"{breast_type} breast"
                    , "target_task": "breast cancer detection"
                    })
    return ("KO", f"Name not recognized: {name}")

def handle_hc18(name: str):
    patterns = dataset_to_authorized_pattern["hc18"]
    for pattern in patterns:
        match = re.match(pattern, name)
        if(match):
            # dico = match.groupdict()
            return ("OK", {
                    "dataset": "hc18"
                    , "old_name": name
                    , "modality": "US"
                    , "anatomy": "fetus"
                    , "target_task": "fetal head circumference"
                    })
    return ("KO", f"Name not recognized: {name}")

def remove_non_alphanumeric(s):
    return ''.join([char for char in s if char.isalnum()])

