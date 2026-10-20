"""Offline checks for RF-DETR mask/box/class alignment."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from deploy.rfdetr_segmenter import RFDETRSegmenter, detection_regions


class RFDETRTests(unittest.TestCase):
    def test_sort_filter_and_limit_preserve_alignment(self):
        masks = np.zeros((3, 4, 6), bool)
        masks[0, 0:2, 0:2] = True
        masks[2, 2:4, 3:6] = True
        detections = SimpleNamespace(xyxy=np.array([[0, 0, 2, 2], [0, 0, 6, 4], [3, 2, 6, 4]]),
                                     mask=masks, class_id=np.array([1, 2, 3]),
                                     confidence=np.array([0.5, 0.99, 0.9]))
        boxes, actual, labels = detection_regions(detections, {1: "person", 2: "cat", 3: "cup"}, (6, 4), 1)
        np.testing.assert_array_equal(boxes, [[3, 2, 6, 4]])
        np.testing.assert_array_equal(actual, masks[2:3])
        self.assertEqual(labels, ["cup"])

    def test_empty_detection(self):
        result = detection_regions(SimpleNamespace(xyxy=np.empty((0, 4)), mask=None), {}, (6, 4))
        self.assertEqual(result[0].shape, (0, 4))
        self.assertEqual(result[1].shape, (0, 4, 6))
        self.assertEqual(result[2], [])

    def test_missing_or_wrong_resolution_masks_rejected(self):
        for mask in (None, np.ones((1, 2, 3), bool)):
            with self.assertRaises(ValueError):
                detection_regions(SimpleNamespace(xyxy=np.zeros((1, 4)), mask=mask), {}, (6, 4))

    def test_wrapper_passes_image_and_threshold(self):
        segmenter = RFDETRSegmenter.__new__(RFDETRSegmenter)
        segmenter.model = Mock()
        segmenter.model.predict.return_value = SimpleNamespace(xyxy=np.empty((0, 4)))
        segmenter.names, segmenter.confidence, segmenter.max_det = {}, 0.4, 60
        segmenter(Image.new("RGBA", (6, 4)))
        call = segmenter.model.predict.call_args
        self.assertEqual(call.args[0].mode, "RGB")
        self.assertEqual(call.kwargs, {"threshold": 0.4})


if __name__ == "__main__":
    unittest.main()
