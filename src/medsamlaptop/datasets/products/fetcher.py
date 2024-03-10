import pathlib
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)
# TODO : move this out 
id_to_root_path_mapper = {
    # Exemple : "mfile odality<1>": "path-...-..."   
    "FLARE22": "FLARE22Train/data/npy/CT_Abd/"
    # Unittest
    , "unittest_CT": "CT_Abd"
}

# name should look like : <something with _ and a-zA-Z0-9>_XXXX-YYY.npy
# the XXXX is the modality
# TODO: this does not work because the first part also validates for modality_id
NAME_PATTERN = r'(?P<root_name>\w)_(?P<modality_id>\d{4})-(?P<data_id>\d{3}).npy'

class IncorrectDataNameFormat(Exception):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"Name was: {self.name}, was expecting: {NAME_PATTERN}"

class JsonDataParser:
    def __init__(self
                 , json_data_file: Dict
                 , root_path: pathlib.Path) -> None:
        assert("modality" in json_data_file.keys())
        assert("data" in json_data_file.keys())
        self.modalities =  json_data_file["modality"]
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
            modality_id = self.try_modality_parser(name)
            return ("OK", {"name": name
                           , "modality": modality_id})
        except IncorrectDataNameFormat as e:
            return ("KO", e)
        
    def make_paths_for_name(self, input: Tuple[str, Dict | Exception], subfolders: List[str]):
        status, info = input
        if(status == "KO" or isinstance(info, Exception)):
            return input # ("KO", <error>)

        assert(all( [expected_key in info.keys() for expected_key in ("modality", "name")] ))
        modality_id = info["modality"]
        name = info["name"]

        # NOTE: two steps, we suppose that the jsn contains the link between the IDs and some modality name
        # then we go and fetch it using the reference
        modality_name = self.modalities[modality_id]
        modality_root_path = id_to_root_path_mapper[modality_name]
        paths = {}
        for subfolder in subfolders:
            path = ((self.root_dir / modality_root_path) / subfolder) / name
            if(not path.is_file()):
                return ("KO", f"Data does not exist: {path}")
            paths[subfolder] = path
        return ("OK", paths)

    def try_modality_parser(self, name:str):
        # match = re.match(NAME_PATTERN, name)
        # if(match):
        #     return match.groupdict()["modality_id"]
        if(len(name) >= 12):
            # TODO: this is really really stupid
            modality = name[-12: -8]
            return modality
        raise IncorrectDataNameFormat(name)
