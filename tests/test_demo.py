"""Offline checks for device routing, mask forwarding and image pipelining."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

from demo import (collect_images, iter_detected, output_base, predict_relations,
                  save_outputs, select_devices)


class DemoTests(unittest.TestCase):
    def test_device_routing(self):
        self.assertEqual(select_devices(2), ("cuda:0", "cuda:1"))
        self.assertEqual(select_devices(1), ("cuda:0", "cuda:0"))
        self.assertEqual(select_devices(0), ("cpu", "cpu"))

    def test_both_relation_predictions_receive_same_masks(self):
        model = Mock()
        model.predict.side_effect = [[], {"spatial": [], "semantic": []}]
        image, boxes, masks = object(), object(), object()
        with redirect_stdout(io.StringIO()):
            predict_relations(model, image, boxes, masks, ["person", "horse"], 3)
        self.assertEqual(model.predict.call_count, 2)
        for call in model.predict.call_args_list:
            self.assertIs(call.args[0], image)
            self.assertIs(call.args[1], boxes)
            self.assertIs(call.kwargs["masks"], masks)

    def test_next_detection_overlaps_consumption_in_order(self):
        second_started = threading.Event()
        release_second = threading.Event()
        threads = []

        def detect(path):
            threads.append(threading.get_ident())
            if path == "second":
                second_started.set()
                if not release_second.wait(timeout=5):
                    raise TimeoutError("Consumer did not release the detector")
            return path

        results = iter_detected(["first", "second", "third"], detect)
        try:
            self.assertEqual(next(results), "first")
            # While the caller processes image 1, the detector starts image 2.
            self.assertTrue(second_started.wait(timeout=2))
        finally:
            release_second.set()
        self.assertEqual(list(results), ["second", "third"])
        self.assertEqual(len(set(threads)), 1)
        self.assertNotEqual(threads[0], threading.get_ident())

    def test_worker_errors_are_not_silently_skipped(self):
        def fail(path):
            raise ValueError("missing masks")

        with self.assertRaisesRegex(ValueError, "missing masks"):
            list(iter_detected(["bad_image"], fail))

    def test_empty_input(self):
        self.assertEqual(list(iter_detected([], Mock())), [])

    def test_external_directory_and_duplicate_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            a, b = root / "scene_a/raw_1.png", root / "scene_b/raw_1.png"
            for p in (a, b):
                p.parent.mkdir(parents=True)
                p.touch()
            (root / "notes.txt").touch()
            self.assertEqual(collect_images(image_dir=root), [a, b])
            out = Path(tmp) / "output"
            self.assertNotEqual(output_base(a, root, out), output_base(b, root, out))
            self.assertNotEqual(output_base(a, None, out), output_base(b, None, out))

    def test_save_mask_arrows_and_machine_readable_scores(self):
        import numpy as np
        from PIL import Image
        from relsgg.api import Triplet

        image = Image.new("RGB", (240, 160), (100, 100, 100))
        boxes = np.array([[10, 20, 90, 130], [145, 25, 225, 135]], np.float32)
        masks = np.zeros((2, 160, 240), bool)
        masks[0, 20:130, 10:90] = True
        masks[1, 25:135, 145:225] = True
        relation = Triplet(0, boxes[0], "holding", 0.85, 1, boxes[1], "person", "cup")
        predictions = dict(relations=[relation], spatial=[], semantic=[relation])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "scene/raw_1.png"
            with redirect_stdout(io.StringIO()):
                save_outputs(image, boxes, masks, ["person", "cup"], predictions,
                             base, Path("source.png"), {"full_vocabulary": True})
            data = json.loads(Path(f"{base}.json").read_text())
            self.assertEqual(data["predictions"]["relations"][0]["score"], 0.85)
            self.assertEqual(data["predictions"]["relations"][0]["subject_idx"], 0)
            self.assertEqual(set(data["visuals"]), {"masks", "relations", "spatial", "semantic"})
            for path in data["visuals"].values():
                with Image.open(path) as preview:
                    self.assertEqual(preview.format, "PNG")
            with Image.open(data["visuals"]["masks"]) as only_masks:
                with Image.open(data["visuals"]["relations"]) as arrows:
                    self.assertFalse(np.array_equal(np.asarray(only_masks), np.asarray(arrows)))
        # Rendering must not alter the source pixels or mask geometry.
        self.assertEqual(image.getpixel((50, 50)), (100, 100, 100))
        self.assertEqual(int(masks.sum()), 17600)

    def test_render_zero_detections(self):
        import numpy as np
        from PIL import Image
        from deploy.demo_visuals import render_regions

        preview = render_regions(Image.new("RGB", (100, 80)), np.zeros((0, 4)),
                                 np.zeros((0, 80, 100), bool), [], [], "No detections")
        self.assertEqual(preview.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
