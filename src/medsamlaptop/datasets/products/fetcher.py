import pathlib
import re
import logging
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)
# TODO : move this out 

processed_root_dir = "PROCESSED/"
id_to_root_path_mapper = {
    "hc18": "hc18"
    , "breastultrasound": "Breast-Ultrasound" 
    , "autopet": "autoPET"
    , "intraretinalcystoidfluid": "Intraretinal-Cystoid-Fluid"
    , "Neurips22cellseg": "NeurIPS22CellSeg"
    , "m2caiseg": "m2caiSeg"
    , "kvasirseg": "Kvasir-SEG" 
    , "cholecseg8k": "CholecSeg8k"
    , "ctabdtumor": "CT_AbdTumor"
    , "abdomenct1k": "AbdomenCT1K"
    , "amod": "AMOD"
    # Unittest - to be handled differently - TODO: move this out
    , "unittest_CT": "CT_Abd"
}

NAME_PATTERN = r'(?P<dataset>[a-zA-Z0-9]+)_(?P<modality>[a-zA-Z0-9]+)_(?P<anatomy>[a-zA-Z0-9]+)_(?P<id>[a-zA-Z0-9]+)_'

class IncorrectDataNameFormat(Exception):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"Name was: {self.name}, was expecting: {NAME_PATTERN}"

class JsonDataParser:
    def __init__(self
                 , json_data_file: Dict
                 , root_path: pathlib.Path) -> None:
        assert("data" in json_data_file.keys())
        self.data = json_data_file["data"]
        self.root_dir = root_path

    def get_paths(self, subfolders):
        paths = {}
        for name in self.data:
            (status, info_name) = self.make_paths_for_name(
                                        self.parse_name(name)
                                        , subfolders)
            if(status == "KO"):
                logger.warning(f"Failed to load data for name = {name} (error = {info_name})")
            else:
                paths[name] = info_name
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
        for subfolder in subfolders:
            path = (((self.root_dir / processed_root_dir) / dataset_name) / subfolder) / f"{name}.npy"
            if(not path.is_file()):
                return ("KO", f"Data does not exist: {path}")
            paths[subfolder] = path
        return ("OK", paths)

    def try_dataset_parser(self, name:str):
        match = re.match(NAME_PATTERN, name)
        if(match):
            return match.groupdict()["dataset"]
        raise IncorrectDataNameFormat(name)
