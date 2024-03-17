from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class AsyncArraySaver:
    def __init__(self, root_saving_path, max_workers=5):
        self.root_path = Path(root_saving_path)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def save_array(self, path, array):
        np.save(path, array)

    def save_arrays(self, arrays, names):
        for array, name in zip(arrays, names):
            file_path = self.root_path / name
            self.executor.submit(self.save_array, file_path, array)

    def close(self):
        self.executor.shutdown(wait=True)

