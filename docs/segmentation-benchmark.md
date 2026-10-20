# Three-model comparison on Kaggle

Use the existing RelSGG environment, enable Internet and GPU, and pull branch
`test`. Install the two optional backends with `uv sync --extra benchmark`.

In a Kaggle notebook cell:

```python
%cd /kaggle/working/RelateAnythingFork
!git pull origin test
!uv sync --extra benchmark
!uv run demo.py

from pathlib import Path
from IPython.display import HTML, display
report = Path('/kaggle/working/segmentation_benchmark/latest_report.txt').read_text().strip()
display(HTML(filename=report))
```

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
