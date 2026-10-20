# Three-model comparison on Kaggle

Use the existing RelSGG environment, enable Internet and GPU, and pull branch
`test`. Install the two optional backends with `uv sync --extra benchmark`.

In a Kaggle notebook cell:

```python
from pathlib import Path
import subprocess
from IPython.display import HTML, display

repo = Path('/kaggle/working/RelateAnythingFork')
subprocess.run(['git', 'pull', '--ff-only', 'origin', 'test'], cwd=repo, check=True)
subprocess.run(['uv', 'sync', '--extra', 'benchmark'], cwd=repo, check=True)
subprocess.run(['uv', 'run', '--extra', 'benchmark', 'demo.py', '--benchmark'], cwd=repo, check=True)
pointer = Path('/kaggle/working/segmentation_benchmark/latest_report.txt')
if not pointer.is_file():
    raise RuntimeError('Benchmark did not generate a report; inspect output above.')
report = pointer.read_text().strip()
display(HTML(filename=report))
```

Unlike notebook `!` commands, `check=True` stops the cell if pulling, dependency
installation or inference fails. The included notebook also displays fresh
partial reports on backend failure and rejects stale reports from earlier runs.

If `git pull` says local `pyproject.toml` edits would be overwritten, save those
edits first, then rerun the cell above:

```python
subprocess.run(['git', 'stash', 'push', '-m', 'kaggle-pyproject-before-benchmark',
                '--', 'pyproject.toml'], cwd=repo, check=True)
```

The local configuration edits remain in the stash (`git stash list` and
`git stash show -p` to inspect). Review and merge any Kaggle-specific dependency
adjustments with the updated configuration before restoring them; blindly
restoring the old file would remove the new `benchmark` extra.

A bare `uv run demo.py` on Kaggle runs all three backends against
`/kaggle/working/data/02_2_3_obj`, with `02_2_3obj` accepted as an alternate name.
Use an explicit directory elsewhere or when the dataset has another name:

```bash
uv run demo.py --benchmark --image-dir /path/to/data --output-dir /path/to/results
```

Shell subprocesses cannot publish a rich HTML display directly into the parent
notebook. The `display(HTML(...))` lines in the same cell render the report after
the command completes. A ready-to-use notebook is in
`notebooks/segmentation_benchmark.ipynb`.

Every image has three columns: YOLOE-26m prompt-free, Mask2Former Swin-S COCO
panoptic, and RF-DETR Seg Medium. Columns show mask overlays, object IDs, relation
arrows, textual triplets and scores, plus expandable spatial/semantic predictions.
IDs and colors are local to each model. A filename filter helps navigate larger
datasets. The portable HTML embeds preview images; full PNGs and prediction JSON
remain in backend subdirectories. CSV includes counts and timing per image/model.

Models run in separate sequential processes to release GPU memory between runs.
The display is side by side, not concurrent execution of three GPU models. The
same RelSGG model, full vocabulary, top-k (default 3 per graph), confidence 0.4 and
limit of 60 regions are used. `--topk`, `--rfdetr-size`, `--detector` and
`--mask2former-model` can override these defaults and are recorded in results.

Timing synchronizes CUDA, disables image prefetch, excludes model loading and
rendering, and measures segmentation plus both relation calls. The first image
is reported as cold and excluded from aggregate mean, median and p95. These are
observed sequential latencies, not a warmed throughput benchmark. Mask2Former
includes stuff regions and the others output instances, so region counts need
that context. Scores and counts are not accuracy; no ground truth means no mAP,
IoU or relation recall is computed.

Each invocation creates a fresh run directory. A failed backend is reported as
missing for unfinished images, the other models still run, and the command exits
nonzero after writing the partial report. See the corresponding model log in the
run directory. `latest_report.txt` points to the latest report, including partial
runs. No pretrained inference has been verified locally; run it on Kaggle.
