import unittest

from medsamlaptop import models as medsamlaptop_models
import torch

class TestLittleMedSAM(unittest.TestCase):
    batch_size = 8
    image_size = 256
    input_image = torch.rand(batch_size, 3, image_size, image_size) # (B, 3, 256, 256)
    input_box = torch.rand(batch_size, 1, 4) # (B, 1, 4)
    expected_low_res_masks_size = torch.Size([batch_size, 1, image_size, image_size])
    expected_iou_predictions_size = torch.Size([batch_size, 1])
    factory = medsamlaptop_models.MedSAMLiteFactory()
    facade = medsamlaptop_models.SegmentAnythingModelFacade(factory)
    model = facade.get_model()
    def test_output_shape(self):
        low_res_masks, iou_predictions = self.model(self.input_image, self.input_box)        
        self.assertEqual(self.expected_low_res_masks_size
                         , low_res_masks.shape
                         , "Low resolution mask shape error")
        self.assertEqual(self.expected_iou_predictions_size
                         , iou_predictions.shape
                         , "IoU shape error")

class TestEdgeSAM(unittest.TestCase):
    # NOTE: requires, in inputs, image of size 1024 !!
    batch_size = 8
    image_size = 1024
    output_res = 256
    input_image = torch.rand(batch_size, 3, image_size, image_size) # (B, 3, 256, 256)
    input_box = torch.rand(batch_size, 1, 4) # (B, 1, 4)
    expected_low_res_masks_size = torch.Size([batch_size, 1, output_res, output_res])
    expected_iou_predictions_size = torch.Size([batch_size, 1])
    factory = medsamlaptop_models.MedEdgeSAMFactory()
    facade = medsamlaptop_models.SegmentAnythingModelFacade(factory)
    model = facade.get_model()
    def test_output_shape(self):
        low_res_masks, iou_predictions = self.model(self.input_image, self.input_box)        
        self.assertEqual(self.expected_low_res_masks_size
                         , low_res_masks.shape
                         , "Low resolution mask shape error")
        self.assertEqual(self.expected_iou_predictions_size
                         , iou_predictions.shape
                         , "IoU shape error")

if __name__ == '__main__':
    unittest.main()