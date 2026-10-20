"""Panoptic region conversion must preserve instance identity and geometry."""
import unittest

import numpy as np
import torch

from deploy.mask2former_segmenter import panoptic_regions


class PanopticRegionsTests(unittest.TestCase):
    def test_instances_stuff_void_and_score_order(self):
        result = dict(segmentation=torch.tensor([[0, 3, 3, 9], [0, 5, 5, 9]]),
                      segments_info=[dict(id=3, label_id=0, score=0.8),
                                     dict(id=5, label_id=0, score=0.9),
                                     dict(id=9, label_id=1, score=0.7)])
        boxes, masks, labels = panoptic_regions(result, {0: "person", 1: "wall"}, (4, 2))
        self.assertEqual(labels, ["person", "person", "wall"])
        np.testing.assert_array_equal(boxes, [[1, 1, 3, 2], [1, 0, 3, 1], [3, 0, 4, 2]])
        self.assertEqual(masks.dtype, np.bool_)
        self.assertFalse(masks[:, :, 0].any())
        self.assertFalse((masks[0] & masks[1]).any())
        limited = panoptic_regions(result, {0: "person", 1: "wall"}, (4, 2), max_det=1)
        np.testing.assert_array_equal(limited[0], boxes[:1])

    def test_empty_and_absent_segments(self):
        result = dict(segmentation=torch.full((2, 4), -1),
                      segments_info=[dict(id=7, label_id=0, score=0.9)])
        boxes, masks, labels = panoptic_regions(result, {0: "person"}, (4, 2))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(masks.shape, (0, 2, 4))
        self.assertEqual(labels, [])

    def test_wrong_resolution_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            panoptic_regions(dict(segmentation=torch.zeros(2, 4), segments_info=[]), {}, (2, 4))


if __name__ == "__main__":
    unittest.main()
