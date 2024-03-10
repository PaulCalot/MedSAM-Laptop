import unittest
import pathlib
import torch
import json

from medsamlaptop import datasets as medsamlaptop_datasets
from medsamlaptop.datasets.products.fetcher import JsonDataParser

class TestJsonDataParser(unittest.TestCase):
    root_path_to_test_data = pathlib.Path(pathlib.Path(__file__).resolve() / "../ref_data").resolve()
    path_to_json = root_path_to_test_data / "data_example.json"
    with open(path_to_json, "r") as f:
        json_data_file = json.load(f)
    json_parser = JsonDataParser(json_data_file, root_path_to_test_data / "npy/")

    def test_loading(self):
        subfolders = ["gts", "encoder_gts", "imgs", "teacher_gts"]
        paths = self.json_parser.get_paths(subfolders)
        
        expected_keys = ["CT_Abd_FLARE22_Tr_0001-000.npy"
                         , "CT_Abd_FLARE22_Tr_0001-005.npy"
                         , "CT_Abd_FLARE22_Tr_0001-010.npy"
                         , "CT_Abd_FLARE22_Tr_0001-028.npy"]
        for expected_key in expected_keys:
            self.assertIn(expected_key, paths.keys())
            dico = paths[expected_key]
            self.assertTrue(all([subfolder in dico.keys() for subfolder in subfolders]))
            for subfolder in subfolders:
                path_value = dico[subfolder]
                self.assertIsInstance(path_value, pathlib.Path)
                self.assertTrue(path_value.is_file())
                self.assertEqual(path_value
                                 , self.root_path_to_test_data / f"npy/CT_Abd/{subfolder}/{expected_key}")

class TestNpyDataset(unittest.TestCase):
    data_root = pathlib.Path(pathlib.Path(__file__).resolve() / "../ref_data/npy/CT_Abd").resolve()
    dataset_factory = medsamlaptop_datasets.Npy256Factory(path_to_data=data_root)
    dataset = dataset_factory.create_dataset()

    def test_output_shape(self):
        for idx in range(len(self.dataset)):
            expected_keys = ["image", "gt2D", "bboxes", "image_name", "new_size", "original_size"]
            output = self.dataset[idx]
            self.assertIsInstance(output, dict, "Output of NPY dataset should be a dictionnary")
            keys = list(output.keys())
            for key in expected_keys:
                self.assertIn(key, keys, f"{key} should be in {keys}")
            self.assertEqual(output['image'].shape, (3, 256, 256), "Shape of 'image' is incorrect")
            self.assertEqual(output['gt2D'].shape, (1, 256, 256), "Shape of 'gt2D' is incorrect")
            self.assertEqual(output['bboxes'].shape, (1, 1, 4), "Shape of 'bboxes' is incorrect") 
            self.assertIsInstance(output['image_name'], str, "Type of 'image_name' should be a string")
            self.assertEqual(output['new_size'].shape, (2, ), "Shape of 'new_size' is incorrect")
            self.assertEqual(output['original_size'].shape, (2, ), "Shape of 'original_size' is incorrect")

class TestEncoderDistillationDataset(unittest.TestCase):
    data_root = pathlib.Path(pathlib.Path(__file__).resolve() / "../ref_data/npy/CT_Abd").resolve()
    dataset_factory = medsamlaptop_datasets.Distillation1024Factory(path_to_data=data_root)
    dataset = dataset_factory.create_dataset()
    def test_output_shape(self):
        for idx in range(len(self.dataset)):
            expected_keys = ["image", "encoder_gts", "image_name", "new_size", "original_size"]
            output = self.dataset[idx]
            self.assertIsInstance(output, dict, "Output of Distillaiton dataset should be a dictionnary")
            keys = list(output.keys())
            for key in expected_keys:
                self.assertIn(key, keys, f"{key} should be in {keys}")
            self.assertEqual(output['image'].shape, (3, 1024, 1024), "Shape of 'image' is incorrect")
            self.assertEqual(output['encoder_gts'].shape, (256, 64, 64), "Shape of 'encoder_gts' is incorrect")
            self.assertIsInstance(output['image_name'], str, "Type of 'image_name' should be a string")
            self.assertEqual(output['new_size'].shape, (2, ), "Shape of 'new_size' is incorrect")
            self.assertEqual(output['original_size'].shape, (2, ), "Shape of 'original_size' is incorrect")

class TestStage2DistillationDataset(unittest.TestCase):
    data_root = pathlib.Path(pathlib.Path(__file__).resolve() / "../ref_data/npy/CT_Abd").resolve()
    dataset_factory = medsamlaptop_datasets.Stage2Distillation1024Factory(path_to_data=data_root)
    dataset = dataset_factory.create_dataset()

    def test_output_shape(self):
        for idx in range(len(self.dataset)):
            expected_keys = ["image", "teacher_gt2D", "bboxes", "image_name", "new_size", "original_size"]
            output = self.dataset[idx]
            self.assertIsInstance(output, dict, "Output of NPY dataset should be a dictionnary")
            keys = list(output.keys())
            for key in expected_keys:
                self.assertIn(key, keys, f"{key} should be in {keys}")
            self.assertEqual(output['image'].shape, (3, 1024, 1024), "Shape of 'image' is incorrect")
            self.assertEqual(output['teacher_gt2D'].shape, (1, 256, 256), "Shape of 'gt2D' is incorrect")
            self.assertIn(output['teacher_gt2D'].dtype, [torch.float16, torch.float32, torch.float64], f"Type of 'gt2D' is incorrect, should be float, got {output['teacher_gt2D'].dtype}")
            self.assertEqual(output['bboxes'].shape, (1, 1, 4), "Shape of 'bboxes' is incorrect") 
            self.assertIsInstance(output['image_name'], str, "Type of 'image_name' should be a string")
            self.assertEqual(output['new_size'].shape, (2, ), "Shape of 'new_size' is incorrect")
            self.assertEqual(output['original_size'].shape, (2, ), "Shape of 'original_size' is incorrect")

if __name__ == '__main__':
    unittest.main()
