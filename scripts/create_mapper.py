from tqdm import tqdm
import sys 
import random
from medsamtools import user
from utils.name_mapper import (
    NameMapper
    , get_handler
    , datasets_paths
)

root_path = user.get_path_to_data()
mapper_name = "MAPPER"
root_saving_directory = root_path / "PROCESSED"
root_npz_directory = root_path / "NPZ"
if __name__ == "__main__":
    with NameMapper(root_path / f"{mapper_name}.csv") as mapper:
        for name_dataset, rel_path in datasets_paths.items():
            print(f"Processing {name_dataset}...")
            path_to_dataset = root_npz_directory / rel_path
            lst_paths = sorted(path_to_dataset.glob("*.npz"))
            dataset_handler = get_handler(name_dataset)
            for k, path_ in tqdm(enumerate(lst_paths)):
                try:
                    (status, dico) = dataset_handler(path_.stem)
                except Exception as e:
                    print(f"[{k}] Exception: {name_dataset} - {path_.stem} - {e}")
                    sys.exit(1)
                    # continue
                if(status == "KO"):
                    print(f"[{k}] FAIL: {name_dataset} - {path_.stem} (reason: {dico})")
                    sys.exit(1)
                else:
                    new_name = mapper.get_new_name(**dico)
