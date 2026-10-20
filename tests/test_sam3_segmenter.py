"""Offline checks for SAM 3 region alignment and overlapping concept prompts."""
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

from deploy.sam3_segmenter import Sam3Segmenter


class Inputs(dict):
    def to(self, device):
        return self

    @property
    def pixel_values(self):
        return self["pixel_values"]


class Sam3Tests(unittest.TestCase):
    def segmenter(self, results):
        segmenter = Sam3Segmenter.__new__(Sam3Segmenter)
        segmenter.device = "cpu"
        segmenter.confidence = 0.4
        segmenter.max_det = 60
        segmenter.prompts = ["animal", "horse"][:len(results)]
        segmenter.model = Mock()
        segmenter.processor = Mock(side_effect=lambda **kw: Inputs(
            {"pixel_values": torch.zeros(1)} if "images" in kw else
            {"input_ids": torch.ones(1, dtype=torch.long)}))
        segmenter.processor.post_process_instance_segmentation.side_effect = [[r] for r in results]
        return segmenter

    def test_duplicate_concepts_keep_best_mask_and_label(self):
        mask = torch.zeros((1, 8, 10), dtype=torch.bool)
        mask[:, 2:6, 2:7] = True
        def result(score):
            return dict(boxes=torch.tensor([[2, 2, 7, 6]]), masks=mask,
                        scores=torch.tensor([score]))
        segmenter = self.segmenter([result(0.6), result(0.9)])
        boxes, masks, labels = segmenter(Image.new("RGB", (10, 8)))
        np.testing.assert_array_equal(boxes, [[2, 2, 7, 6]])
        np.testing.assert_array_equal(masks, mask.numpy())
        self.assertEqual(labels, ["horse"])
        self.assertEqual(segmenter.model.get_vision_features.call_count, 1)

    def test_empty_results_keep_original_dimensions(self):
        segmenter = self.segmenter([dict(boxes=torch.empty(0, 4),
                                         masks=torch.empty(0, 8, 10), scores=torch.empty(0))])
        boxes, masks, labels = segmenter(Image.new("RGB", (10, 8)))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(masks.shape, (0, 8, 10))
        self.assertEqual(labels, [])


if __name__ == "__main__":
    unittest.main()
