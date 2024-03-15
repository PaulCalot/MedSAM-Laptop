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
        dataset_formatted = remove_non_alphanumeric(dataset).lower()
        old_name_formatted = remove_non_alphanumeric(old_name).lower()
        primary_key = f"{dataset_formatted}_{old_name_formatted}"
        if(primary_key in self.df_mapper["primary_key"].values):
            # NOTE in theory there should be only one
            new_name = self.df_mapper[self.df_mapper["primary_key"] == primary_key]["new_name"].iloc[0]
            return new_name

        new_id = self.generate_new_id()
        new_name = f"{dataset_formatted}_{modality}_{anatomy}_{new_id}"
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
    dataset_to_handler = {
        "hc18": handle_hc18,
        "Breast-Ultrasound": handle_breastultrasound,
        "autoPET": handle_autopet,
        "Intraretinal-Cystoid-Fluid": handle_intraretinalcystoidfluid,
        "NeurIPS22CellSeg": handle_neurips22cellseg,
        "m2caiSeg": handle_m2caiseg,
        "Kvasir-SEG": handle_kvasirseg,
        "CholecSeg8k": handle_cholecseg8k,
        "CT_AbdTumor": handle_abdtumor,
        "AbdomenCT1K": handle_abdomenct1K,
        "AMOD": handle_amod,
        "ISIC2018": handle_ISIC2018,
        "IDRiD": handle_IDRiD,
        "PAPILA": handle_PAPILA,
        "CDD-CESM": handle_CDD_CESM,
        "AMOSMR": handle_AMOSMR,
        "BraTS_FLAIR": handle_BraTS_FLAIR,
        "BraTS_T1": handle_BraTS_T1,
        "BraTS_T1CE": handle_BraTS_T1CE,
        "CervicalCancer": handle_CervicalCancer,
        "crossmoda": handle_crossmoda,
        "Heart": handle_Heart,
        "ISLES2022_ADC": handle_ISLES2022_ADC,
        "ISLES2022_DWI": handle_ISLES2022_DWI,
        "ProstateADC": handle_ProstateADC,
        "ProstateT2": handle_ProstateT2,
        "QIN-PROSTATE-Lesion": handle_QIN_PROSTATE_Lesion,
        "QIN-PROSTATE-Prostate": handle_QIN_PROSTATE_Prostate,
        "SpineMR": handle_SpineMR,
        "WMH_FLAIR": handle_WMH_FLAIR,
        "WMH_T1": handle_WMH_T1,
        "Chest-Xray-Masks-and-Labels": handle_Chest_Xray_Masks_and_Labels,
        "COVID-19-Radiography-Database": handle_COVID_19_Radiography_Database,
        "COVID-QU-Ex-lungMask_CovidInfection": handle_COVID_QU_Ex_lungMask_CovidInfection,
        "COVID-QU-Ex-lungMask_Lung": handle_COVID_QU_Ex_lungMask_Lung,
        "Pneumothorax-Masks": handle_Pneumothorax_Masks
    }
    return dataset_to_handler.get(dataset, lambda name: ("KO", f"No handler for dataset: {dataset}"))

# TODO: clean that
# ---------------- UTILS 

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
dataset_to_authorized_pattern = {
    "hc18": [r"US_hc18_(?P<NUM>[0-9]{3})_(?P<RANK>[0-9]{0,1})HC"]
    , "Breast-Ultrasound": [r"US_Breast-Ultrasound_(?P<TYPE>benign|malignant|normal) \((?P<NUM>[0-9]+\))"]
    , "autoPET": [r"PET_Lesion_PETCT_(?P<ID>[a-zA-Z0-9]{10})"]
    , "Intraretinal-Cystoid-Fluid": [r"OCT_Intraretinal-Cystoid-Fluid_(?P<SOMETHING>\w)"]
    , "NeurIPS22CellSeg": [r"Microscopy_NeurIPS22CellSeg_cell_(?P<ID>[0-9]{5})"]
    , "m2caiSeg": [r"Endoscopy_m2caiSeg_(?P<ID>[0-9]{3})_(?P<FRAME>[0-9]+)"]
    , "Kvasir-SEG": [r"Endoscopy_Kvasir-SEG_(?P<ID>[a-z0-9]{25})"]
    # TODO : we don't enforce from there the possiblitiees that we saw - we should do it afterwards
    , "CholecSeg8k": [r"Endoscopy_CholecSeg8k_(?P<ORGAN>[a-zA-Z]*)_video(?P<ID>[0-9]+)_frame_(?P<FRAME>[0-9]+)"]
    # TODO: once again, we are kind of lazy, in practice this dataset is already a merge without clean pivotal format
    , "CT_AbdTumor": [r"CT_AbdTumor_(?P<ORGAN>[a-zA-Z]+)_(?P<ID>[a-zA-Z0-9]+)"]
    , "AbdomenCT1K": [r"CT_AbdomenCT_Case_(?P<ID>[0-9]{5})"]
    , "AMOD": [r"CT_AMOS_amos_(?P<ID>[0-9]{4})"]
    , "ISIC2018": [r"Dermoscopy_ISIC2018_ISIC_(?P<ID>[0-9]{7})"]
    , "IDRiD": [r"Fundus_IDRiD_IDRiD_(?P<ID>[0-9]{2})"]
    , "PAPILA": [r"Fundus_PAPILA_(?P<TYPE>OpticCup|OpticDisc)_RET(?P<ID>[0-9]{3})(?P<EYE>[A-Z]{2})"]
    , "CDD-CESM": [r"Mammo_CDD-CESM_(?P<PATIENT_ID>P[0-9]+)_(?P<SIDE>[RL])_(?P<TYPE>DM|CM)_(?P<VIEW>CC|MLO)_(?P<NUM>[0-9]+)"]
    , "AMOSMR": [r"MR_AMOSMR_amos_(?P<ID>[0-9]{4})"]
    , "BraTS_FLAIR": [r"MR_BraTS_FLAIR_BraTS-GLI-(?P<ID>[0-9]{5})-(?P<NUM>[0-9]{3})"]
    , "BraTS_T1": [r"MR_BraTS_T1_BraTS-GLI-(?P<ID>[0-9]{5})-(?P<NUM>[0-9]{3})"]
    , "BraTS_T1CE": [r"MR_BraTS_T1CE_BraTS-GLI-(?P<ID>[0-9]{5})-(?P<NUM>[0-9]{3})"]
    , "CervicalCancer": [r"MR_CervicalCancer_CCTH-(?P<ID>[A-Z0-9]{3})"]
    , "crossmoda": [r"MR_crossmoda_crossmoda2023_(?P<LOCATION>[a-z]{3})_(?P<ID>[0-9]{1,3})_ceT1"]
    , "Heart": [r"MR_Heart_la_(?P<ID>[0-9]{3})"]
    , "ISLES2022_ADC": [r"MR_ISLES2022_ADC_sub-strokecase(?P<ID>[0-9]{4})"]
    , "ISLES2022_DWI": [r"MR_ISLES2022_DWI_sub-strokecase(?P<ID>[0-9]{4})"]
    , "ProstateADC": [r"MR_Prostate_ADC-(?P<SERIES>[\-A-Za-z]+)_(?P<ID>[0-9]{2,4})"
                      , r"MR_Prostate_ADC-(?P<SERIES>[A-Za-z]+)-(?P<ID>[0-9]{2,4})"]
    , "ProstateT2": [r"MR_Prostate_T2-(?P<SERIES>[A-Za-z]+)-(?P<ID>[0-9]{4})"
                     , r"MR_Prostate_T2-(?P<SERIES>[A-Za-z\-]+)_(?P<ID>[0-9]{2})"
                     , r"MR_Prostate_NCI-(?P<SERIES>[A-Za-z0-9]+)-(?P<IDbis>[0-9]{2})-(?P<ID>[0-9]{4})"]
    , "QIN-PROSTATE-Lesion": [r"MR_QIN-PROSTATE-Lesion_PCAMPMRI-(?P<ID>[0-9]{5})_(?P<SEQUENCE>[0-1])_(?P<SUBSEQUENCE>[0-2])_MR"]
    , "QIN-PROSTATE-Prostate": [r"MR_QIN-PROSTATE-Prostate_PCAMPMRI-(?P<ID>[0-9]{5})_(?P<SEQUENCE>[0-1])_(?P<SUBSEQUENCE>[0-2])_MR"]
    , "SpineMR": [r"MR_SpineMR_spine_case(?P<ID>[0-9]+)"]
    , "WMH_FLAIR": [r"MR_WMH_FLAIR_(?P<LOCATION>[A-Za-z\_]+)_(?P<IDbis>[A-Za-z0-9]{4,5})_(?P<ID>[0-9]+)"
                    , r"MR_WMH_FLAIR_(?P<LOCATION>[A-Za-z]+)_(?P<ID>[0-9]+)"]
    , "WMH_T1": [r"MR_WMH_T1_(?P<LOCATION>[A-Za-z\_]+)_(?P<IDbis>[A-Za-z0-9]{4,5})_(?P<ID>[0-9]+)"
                    , r"MR_WMH_T1_(?P<LOCATION>[A-Za-z]+)_(?P<ID>[0-9]+)"]
    , "Chest-Xray-Masks-and-Labels": [r"XRay_Chest-Xray-Masks-and-Labels_(?P<SOURCE>[A-Z]+)_(?P<ID>[0-9]{4})_(?P<UNIQUE>[0-1])"]
    , "COVID-19-Radiography-Database": [r"XRay_COVID-19-Radiography-Database_(?P<TYPE>[A-Za-z \_]+)-(?P<ID>[0-9]+)"]
    , "COVID-QU-Ex-lungMask_CovidInfection": [r"XRay_COVID-QU-Ex-lungMask_covid_(?P<ID>[0-9]+)"]
    , "COVID-QU-Ex-lungMask_Lung": [r"XRay_COVID-QU-Ex-lungMask_(?P<TYPE>Non_COVID|Normal|covid|non_COVID) \((?P<ID>[0-9]+)\)"
                                    , r"XRay_COVID-QU-Ex-lungMask_(?P<TYPE>Non_COVID|Normal|covid)_(?P<ID>[0-9]+)"]
    , "Pneumothorax-Masks": [r"XRay_Pneumothorax-Masks_(?P<ID>[0-9]+)_(?P<TYPE>train|test)_(?P<FLAG>[0-1])"]
}

def handle_generic(name: str, dataset: str) -> Tuple[str, Dict]:
    patterns = dataset_to_authorized_pattern.get(dataset, [])
    modality = datasets_paths.get(dataset, "").split('/')[0]
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            dico = match.groupdict()
            return_info = {
                "dataset": dataset,
                "old_name": name,
                "modality": modality
            }
            # TODO : ajouter pour les autres datasets non encore présents ici
            if dataset == "AMOD":
                return_info.update({"anatomy": "abdomen", "target_task": "organ segmentation"})
            elif dataset == "AbdomenCT1K":
                return_info.update({"anatomy": "abdomen", "target_task": "organ segmentation"})
            elif dataset == "CT_AbdTumor":
                return_info.update({"anatomy": "abdomen", "target_task": "tumor detection and segmentation"})
            elif dataset == "CholecSeg8k":
                return_info.update({"anatomy": "abdomen", "target_task": "surgical tool segmentation"})
            elif dataset == "Kvasir-SEG":
                return_info.update({"anatomy": "gastrointestinal", "target_task": "polyp segmentation"})
            elif dataset == "m2caiSeg":
                return_info.update({"anatomy": "abdomen", "target_task": "surgical tool segmentation"})
            elif dataset == "NeurIPS22CellSeg":
                return_info.update({"anatomy": "cell", "target_task": "cell segmentation"})
            elif dataset == "Intraretinal-Cystoid-Fluid":
                return_info.update({"anatomy": "retina", "target_task": "cystoid fluid segmentation"})
            elif dataset == "autoPET":
                return_info.update({"anatomy": "various", "target_task": "automated PET lesion segmentation"})
            elif dataset == "Breast-Ultrasound":
                return_info.update({"anatomy": "breast", "target_task": "breast cancer detection"})
            elif dataset == "hc18":
                return_info.update({"anatomy": "fetus", "target_task": "fetal head circumference estimation"})
            elif dataset == "ISIC2018":
                return_info.update({"anatomy": "skin", "target_task": "lesion classification"})
            elif dataset == "IDRiD":
                return_info.update({"anatomy": "retina", "target_task": "retinopathy detection"})
            elif dataset == "PAPILA":
                return_info.update({"anatomy": "optic disc", "target_task": "optic disc segmentation"})
            elif dataset == "CDD-CESM":
                return_info.update({"anatomy": "breast", "target_task": "cancer detection"})
            elif dataset == "AMOSMR":
                return_info.update({"anatomy": "brain", "target_task": "lesion segmentation"})
            elif dataset in ["BraTS_FLAIR", "BraTS_T1", "BraTS_T1CE"]:
                return_info.update({"anatomy": "brain", "target_task": "tumor segmentation"})
            elif dataset == "CervicalCancer":
                return_info.update({"anatomy": "cervical", "target_task": "cancer detection"})
            elif dataset == "crossmoda":
                return_info.update({"anatomy": "brain", "target_task": "modality conversion segmentation"})
            elif dataset == "Heart":
                return_info.update({"anatomy": "heart", "target_task": "structure segmentation"})
            elif dataset in ["ISLES2022_ADC", "ISLES2022_DWI"]:
                return_info.update({"anatomy": "brain", "target_task": "stroke lesion segmentation"})
            elif dataset in ["ProstateADC", "ProstateT2"]:
                return_info.update({"anatomy": "prostate", "target_task": "cancer detection"})
            elif dataset in ["QIN-PROSTATE-Lesion", "QIN-PROSTATE-Prostate"]:
                return_info.update({"anatomy": "prostate", "target_task": "lesion segmentation"})
            elif dataset == "SpineMR":
                return_info.update({"anatomy": "spine", "target_task": "structure segmentation"})
            elif dataset == "WMH_FLAIR":
                return_info.update({"anatomy": "brain", "target_task": "white matter hyperintensities segmentation"})
            elif dataset == "WMH_T1":
                return_info.update({"anatomy": "brain", "target_task": "tissue differentiation"})
            elif dataset == "Chest-Xray-Masks-and-Labels":
                return_info.update({"anatomy": "chest", "target_task": "lung segmentation"})
            elif dataset == "COVID-19-Radiography-Database":
                return_info.update({"anatomy": "chest", "target_task": "COVID-19 detection"})
            elif dataset == "COVID-QU-Ex-lungMask_CovidInfection":
                return_info.update({"anatomy": "chest", "target_task": "COVID-19 infection segmentation"})
            elif dataset == "COVID-QU-Ex-lungMask_Lung":
                return_info.update({"anatomy": "chest", "target_task": "lung segmentation"})
            elif dataset == "Pneumothorax-Masks":
                return_info.update({"anatomy": "chest", "target_task": "pneumothorax segmentation"})
            return ("OK", return_info)
    return ("KO", f"Name not recognized: {name}")

def handle_amod(name: str):
    return handle_generic(name, "AMOD")

def handle_abdomenct1K(name: str):
    return handle_generic(name, "AbdomenCT1K")

def handle_abdtumor(name: str):
    return handle_generic(name, "CT_AbdTumor")

def handle_cholecseg8k(name: str):
    return handle_generic(name, "CholecSeg8k")

def handle_kvasirseg(name: str):
    return handle_generic(name, "Kvasir-SEG")

def handle_m2caiseg(name: str):
    return handle_generic(name, "m2caiSeg")

def handle_neurips22cellseg(name: str):
    return handle_generic(name, "NeurIPS22CellSeg")

def handle_intraretinalcystoidfluid(name: str):
    return handle_generic(name, "Intraretinal-Cystoid-Fluid")

def handle_autopet(name: str):
    return handle_generic(name, "autoPET")

def handle_breastultrasound(name: str):
    return handle_generic(name, "Breast-Ultrasound")

def handle_hc18(name: str):
    return handle_generic(name, "hc18")

def handle_ISIC2018(name: str):
    return handle_generic(name, "ISIC2018")

def handle_IDRiD(name: str):
    return handle_generic(name, "IDRiD")

def handle_PAPILA(name: str):
    return handle_generic(name, "PAPILA")

def handle_CDD_CESM(name: str):
    return handle_generic(name, "CDD-CESM")

def handle_AMOSMR(name: str):
    return handle_generic(name, "AMOSMR")

def handle_BraTS_FLAIR(name: str):
    return handle_generic(name, "BraTS_FLAIR")

def handle_BraTS_T1(name: str):
    return handle_generic(name, "BraTS_T1")

def handle_BraTS_T1CE(name: str):
    return handle_generic(name, "BraTS_T1CE")

def handle_CervicalCancer(name: str):
    return handle_generic(name, "CervicalCancer")

def handle_crossmoda(name: str):
    return handle_generic(name, "crossmoda")

def handle_Heart(name: str):
    return handle_generic(name, "Heart")

def handle_ISLES2022_ADC(name: str):
    return handle_generic(name, "ISLES2022_ADC")

def handle_ISLES2022_DWI(name: str):
    return handle_generic(name, "ISLES2022_DWI")

def handle_ProstateADC(name: str):
    return handle_generic(name, "ProstateADC")

def handle_ProstateT2(name: str):
    return handle_generic(name, "ProstateT2")

def handle_QIN_PROSTATE_Lesion(name: str):
    return handle_generic(name, "QIN-PROSTATE-Lesion")

def handle_QIN_PROSTATE_Prostate(name: str):
    return handle_generic(name, "QIN-PROSTATE-Prostate")

def handle_SpineMR(name: str):
    return handle_generic(name, "SpineMR")

def handle_WMH_FLAIR(name: str):
    return handle_generic(name, "WMH_FLAIR")

def handle_WMH_T1(name: str):
    return handle_generic(name, "WMH_T1")

def handle_Chest_Xray_Masks_and_Labels(name: str):
    return handle_generic(name, "Chest-Xray-Masks-and-Labels")

def handle_COVID_19_Radiography_Database(name: str):
    return handle_generic(name, "COVID-19-Radiography-Database")

def handle_COVID_QU_Ex_lungMask_CovidInfection(name: str):
    return handle_generic(name, "COVID-QU-Ex-lungMask_CovidInfection")

def handle_COVID_QU_Ex_lungMask_Lung(name: str):
    return handle_generic(name, "COVID-QU-Ex-lungMask_Lung")

def handle_Pneumothorax_Masks(name: str):
    return handle_generic(name, "Pneumothorax-Masks")

# def handle_amod(name: str):
#     patterns = dataset_to_authorized_pattern["AMOD"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "AMOD"
#                     , "old_name": name
#                     , "modality": "CT"
#                     # TODO
#                     # , "anatomy": ""
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_abdomenct1K(name: str):
#     patterns = dataset_to_authorized_pattern["AbdomenCT1K"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "AbdomenCT1K"
#                     , "old_name": name
#                     , "modality": "CT"
#                     # TODO
#                     # , "anatomy": ""
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_abdtumor(name: str):
#     patterns = dataset_to_authorized_pattern["CT_AbdTumor"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "AbdTumor"
#                     , "old_name": name
#                     , "modality": "CT"
#                     # TODO
#                     # , "anatomy": ""
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_cholecseg8k(name: str):
#     patterns = dataset_to_authorized_pattern["CholecSeg8k"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "CholecSeg8k"
#                     , "old_name": name
#                     , "modality": "Endoscopy"
#                     # TODO
#                     # , "anatomy": ""
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_kvasirseg(name: str):
#     patterns = dataset_to_authorized_pattern["Kvasir-SEG"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "Kvasir-SEG"
#                     , "old_name": name
#                     , "modality": "Endoscopy"
#                     # TODO
#                     # , "anatomy": ""
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_m2caiseg(name: str):
#     patterns = dataset_to_authorized_pattern["m2caiSeg"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "m2caiSeg"
#                     , "old_name": name
#                     , "modality": "Endoscopy"
#                     # TODO
#                     # , "anatomy": ""
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_neurips22cellseg(name: str):
#     patterns = dataset_to_authorized_pattern["NeurIPS22CellSeg"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "NeurIPS22CellSeg"
#                     , "old_name": name
#                     , "modality": "Microscopy"
#                     # TODO
#                     # , "anatomy": "retina"
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_intraretinalcystoidfluid(name: str):
#     patterns = dataset_to_authorized_pattern["Intraretinal-Cystoid-Fluid"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "Intraretinal-Cystoid-Fluid"
#                     , "old_name": name
#                     , "modality": "OCT" 
#                     , "anatomy": "retina"
#                     # , "target_task": ""
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_autopet(name: str):
#     patterns = dataset_to_authorized_pattern["autoPET"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "autoPET"
#                     , "old_name": name
#                     , "modality": "PET" # /CT" TODO: is this also CT ?
#                     , "target_task": "automated PET lesion segmentation"
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_breastultrasound(name: str) -> Tuple[str, Dict|str]:
#     patterns = dataset_to_authorized_pattern["Breast-Ultrasound"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             dico = match.groupdict()
#             breast_type = dico["TYPE"]
#             return ("OK", {
#                     "dataset": "Breast-Ultrasound"
#                     , "old_name": name
#                     , "modality": "US"
#                     , "anatomy": f"{breast_type} breast"
#                     , "target_task": "breast cancer detection"
#                     })
#     return ("KO", f"Name not recognized: {name}")

# def handle_hc18(name: str):
#     patterns = dataset_to_authorized_pattern["hc18"]
#     for pattern in patterns:
#         match = re.match(pattern, name)
#         if(match):
#             # dico = match.groupdict()
#             return ("OK", {
#                     "dataset": "hc18"
#                     , "old_name": name
#                     , "modality": "US"
#                     , "anatomy": "fetus"
#                     , "target_task": "fetal head circumference"
#                     })
#     return ("KO", f"Name not recognized: {name}")

def remove_non_alphanumeric(s):
    return ''.join([char for char in s if char.isalnum()])

