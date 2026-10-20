"""Offline report and process orchestration checks (no model downloads)."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from deploy.segmentation_benchmark import MODELS, run_benchmark, write_report


class BenchmarkTests(unittest.TestCase):
    def test_successful_run_aligns_duplicate_filenames_by_full_source(self):
        from demo import output_base
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "input"
            paths = [sources / "a/photo.png", sources / "b/photo.png"]
            for path in paths:
                path.parent.mkdir(parents=True)
                Image.new("RGB", (16, 12)).save(path)
            args = SimpleNamespace(output_dir=root / "results", image_dir=sources,
                                   model="relation", topk=3, detector="yolo.pt",
                                   mask2former_model="mask-model", rfdetr_size="medium")

            def fake_run(command, **kwargs):
                target = Path(command[command.index("--output-dir") + 1])
                for index, path in enumerate(paths):
                    base = output_base(path, sources, target)
                    base.parent.mkdir(parents=True, exist_ok=True)
                    result = dict(source=str(path.resolve()), boxes=[], labels=[],
                                  settings=dict(segmentation_ms=10, relation_ms=20, first_image=index == 0),
                                  predictions=dict(relations=[], spatial=[], semantic=[]),
                                  visuals={name: str(path) for name in ("masks", "relations", "spatial", "semantic")})
                    Path(f"{base}.json").write_text(json.dumps(result))
                return SimpleNamespace(returncode=0)

            with patch("deploy.segmentation_benchmark.subprocess.run", side_effect=fake_run):
                run_benchmark(args, paths)
            report = Path((args.output_dir / "latest_report.txt").read_text())
            html = report.read_text()
            self.assertEqual(html.count("<section "), 2)
            self.assertEqual(html.count("30.0 / 30.0 / 30.0 ms"), 3)
            with (report.parent / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual({r["source"] for r in rows}, {str(p) for p in paths})
            self.assertTrue(all(r["status"] == "ok" for r in rows))

    def test_three_columns_geometry_previews_relations_and_missing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "photo.png"
            Image.new("RGB", (16, 12)).save(image)
            triplet = dict(subject_idx=0, object_idx=1, subject_label="person", object_label="cup",
                           predicate="holding <unsafe>", score=0.9)
            result = dict(boxes=[[0, 0, 2, 2], [4, 4, 8, 8]],
                          settings=dict(segmentation_ms=10, relation_ms=20, first_image=True),
                          predictions=dict(relations=[triplet], spatial=[], semantic=[triplet]),
                          visuals={name: str(image) for name in ("masks", "relations", "spatial", "semantic")})
            records = {backend: {str(image.resolve()): result} for backend in ("yolo", "rfdetr")}
            report = write_report([image], records, {"mask2former": "Failed <model>"}, root)
            html = report.read_text()
            for title in MODELS.values():
                self.assertIn(f"<th>{title}</th>", html)
            self.assertIn("holding &lt;unsafe&gt;", html)
            self.assertIn("Failed &lt;model&gt;", html)
            self.assertIn("data:image/jpeg;base64,", html)
            self.assertIn("N/A (no warm samples)", html)
            with (root / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["status"] for r in rows], ["ok", "missing", "ok"])
            self.assertEqual(rows[0]["total_ms"], "30")

    def test_failing_worker_still_runs_all_models_and_creates_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "photo.png"
            Image.new("RGB", (16, 12)).save(image)
            args = SimpleNamespace(output_dir=root / "results", image_dir=None,
                                   model="relation", topk=3, detector="yolo.pt",
                                   mask2former_model="mask-model", rfdetr_size="medium")
            with patch("deploy.segmentation_benchmark.subprocess.run", return_value=SimpleNamespace(returncode=1)) as run:
                with self.assertRaisesRegex(SystemExit, "incomplete"):
                    run_benchmark(args, [image])
            self.assertEqual(run.call_count, 3)
            for call, backend in zip(run.call_args_list, MODELS):
                command = call.args[0]
                self.assertEqual(command[command.index("--segmenter") + 1], backend)
                self.assertIn("--measure", command)
                self.assertIn(str(image.resolve()), command)
            pointer = args.output_dir / "latest_report.txt"
            self.assertTrue(Path(pointer.read_text()).is_file())


if __name__ == "__main__":
    unittest.main()
