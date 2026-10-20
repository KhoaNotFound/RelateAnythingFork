"""The notebook must surface command failures before trying to read a report."""
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class NotebookTests(unittest.TestCase):
    def cells(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/segmentation_benchmark.ipynb"
        return json.loads(path.read_text())["cells"]

    def test_failed_pull_prevents_sync(self):
        code = "".join(self.cells()[1]["source"])
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                exec(compile(code, "notebook setup", "exec"), {})
        self.assertEqual(run.call_count, 1)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_failed_run_without_report_raises_command_error(self):
        code = "".join(self.cells()[2]["source"])
        display_module = ModuleType("IPython.display")
        display_module.HTML = Mock()
        display_module.display = Mock()
        completed = subprocess.CompletedProcess(["uv", "run"], 1)
        with patch.dict(sys.modules, {"IPython.display": display_module}), \
                patch("subprocess.run", return_value=completed), patch.object(Path, "exists", return_value=False):
            with self.assertRaises(subprocess.CalledProcessError):
                exec(compile(code, "notebook run", "exec"), {})
        display_module.display.assert_not_called()


if __name__ == "__main__":
    unittest.main()
