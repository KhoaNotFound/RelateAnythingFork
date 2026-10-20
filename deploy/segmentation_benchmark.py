"""Isolated model runs and a portable, per-image comparison report."""
import base64
import csv
from html import escape
import io
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

MODELS = {"yolo": "YOLOE-26", "mask2former": "Mask2Former", "rfdetr": "RF-DETR"}


def preview(path):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((720, 1000))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img loading="lazy" src="data:image/jpeg;base64,{encoded}">'


def relation_list(triplets):
    items = []
    for t in triplets:
        subject = t.get("subject_label") or "object"
        obj = t.get("object_label") or "object"
        text = (f'#{t["subject_idx"]} {subject} → {t["predicate"]} → '
                f'#{t["object_idx"]} {obj} ({t["score"]:.3f})')
        items.append(f"<li>{escape(text)}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else "<p>No relations returned.</p>"


def write_report(paths, records, failures, output_dir):
    """records[backend][absolute source path] is a saved demo result."""
    import numpy as np
    rows, summaries = [], []
    for backend, title in MODELS.items():
        for source in paths:
            data = records.get(backend, {}).get(str(source.resolve()))
            if data is None:
                rows.append(dict(model=title, source=str(source), status="missing", regions="",
                                 relations="", segmentation_ms="", relation_ms="", total_ms="", first_image=""))
                continue
            settings = data["settings"]
            seg, rel = settings["segmentation_ms"], settings["relation_ms"]
            rows.append(dict(model=title, source=str(source), status="ok", regions=len(data["boxes"]),
                             relations=len(data["predictions"]["relations"]), segmentation_ms=seg,
                             relation_ms=rel, total_ms=seg + rel, first_image=settings["first_image"]))
        measured = [r for r in rows if r["model"] == title and r["status"] == "ok" and not r["first_image"]]
        completed = sum(r["model"] == title and r["status"] == "ok" for r in rows)
        timing = (f'{np.mean([r["total_ms"] for r in measured]):.1f} / '
                  f'{np.median([r["total_ms"] for r in measured]):.1f} / '
                  f'{np.percentile([r["total_ms"] for r in measured], 95):.1f} ms') if measured else "N/A (no warm samples)"
        summaries.append(f'<tr><td>{title}</td><td>{completed}/{len(paths)}</td><td>{timing}</td></tr>')
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parts = ['''<!doctype html><html><head><meta charset="utf-8"><title>Segmentation comparison</title>
<style>body{font:15px system-ui;background:#101827;color:#e5edf9;margin:24px}
table{width:100%;border-collapse:collapse;table-layout:fixed}td,th{padding:12px;border:1px solid #35445c;vertical-align:top}
img{max-width:100%;height:auto}section{margin:32px 0}h2{overflow-wrap:anywhere}li{margin:8px 0;overflow-wrap:anywhere}
details{margin:12px 0}summary{cursor:pointer}p{line-height:1.5}.error{color:#ffb6ab}input{padding:10px;width:60%}
</style></head><body><h1>YOLOE · Mask2Former · RF-DETR</h1>
<p>Same input images and RelSGG checkpoint. Image-only segmentation; full relation vocabulary.
Mask colors and object IDs are local to each model, not cross-model matches.</p>
<p>Qualitative and latency comparison, not an accuracy benchmark: no ground truth provided.
Mask2Former includes stuff regions; the other backends predict object instances. Relation scores are not accuracy.</p>
<p>GPU-synchronized sequential timing excludes model loading, image file reading and rendering.
Relation timing includes both combined and decomposed predictions (two RelSGG calls).
The first image per model is marked cold and excluded from summary timings. No throughput claim.</p>
<table><tr><th>Model</th><th>Images completed</th><th>Total mean / median / p95 (excluding first)</th></tr>''',
             "".join(summaries), '</table><p><input id="filter" placeholder="Filter image filename" oninput="document.querySelectorAll(\'section\').forEach(s=>s.hidden=!s.dataset.name.includes(this.value.toLowerCase()))"></p>']
    for source in paths:
        parts.append(f'<section data-name="{escape(str(source).lower(), quote=True)}"><h2>{escape(str(source))}</h2>')
        parts.append('<details><summary>Original image</summary>' + preview(source) + '</details><table><tr>')
        for title in MODELS.values():
            parts.append(f"<th>{title}</th>")
        parts.append("</tr><tr>")
        for backend in MODELS:
            data = records.get(backend, {}).get(str(source.resolve()))
            if data is None:
                parts.append(f'<td class="error">No result. {escape(failures.get(backend, "Missing output"))}</td>')
                continue
            s = data["settings"]
            parts.append(f'<td><p>{len(data["boxes"])} regions · segmentation {s["segmentation_ms"]:.1f} ms · '
                         f'relations {s["relation_ms"]:.1f} ms {"· cold first image" if s["first_image"] else ""}</p>')
            parts.append(f'<details><summary>Run settings</summary><pre>{escape(json.dumps(s, indent=2))}</pre></details>')
            parts.append('<h3>Masks</h3>' + preview(data["visuals"]["masks"]))
            parts.append('<h3>Relations</h3>' + preview(data["visuals"]["relations"]))
            parts.append(relation_list(data["predictions"]["relations"]))
            for graph in ("spatial", "semantic"):
                parts.append(f'<details><summary>{graph.title()}</summary>' +
                             preview(data["visuals"][graph]) + relation_list(data["predictions"][graph]) + '</details>')
            parts.append("</td>")
        parts.append("</tr></table></section>")
    parts.append("</body></html>")
    report = output_dir / "report.html"
    report.write_text("".join(parts), encoding="utf-8")
    return report


def run_benchmark(args, paths):
    from demo import output_base

    # Each invocation gets fresh outputs: failed reruns cannot reuse stale results.
    output = args.output_dir.resolve() / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    records, failures = {}, {}
    for backend, title in MODELS.items():
        target = output / backend
        cmd = [sys.executable, str(root / "demo.py"), "--segmenter", backend, "--measure",
               "--output-dir", str(target), "--model", args.model, "--topk", str(args.topk),
               "--detector", args.detector, "--mask2former-model", args.mask2former_model,
               "--rfdetr-size", args.rfdetr_size]
        cmd += ["--image-dir", str(args.image_dir.resolve())] if args.image_dir else ["--images", *[str(p.resolve()) for p in paths]]
        log = output / f"{backend}.log"
        print(f"Running {title} on {len(paths)} images. Log: {log}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            failures[backend] = f"Process exited {process.returncode}; see {backend}.log"
            print(failures[backend], flush=True)
        records[backend] = {}
        for path in paths:
            result = Path(f"{output_base(path, args.image_dir, target)}.json")
            if result.exists():
                data = json.loads(result.read_text(encoding="utf-8"))
                records[backend][str(path.resolve())] = data
    report = write_report(paths, records, failures, output)
    # Stable pointer for the notebook display cell; HTML embeds its own thumbnails.
    (args.output_dir.resolve() / "latest_report.txt").write_text(str(report), encoding="utf-8")
    print(f"\nReport: {report}\nMetrics: {output / 'metrics.csv'}", flush=True)
    print("Kaggle display cell:\nfrom pathlib import Path\nfrom IPython.display import HTML, display\n"
          f"display(HTML(filename=Path({str(args.output_dir.resolve() / 'latest_report.txt')!r}).read_text().strip()))")
    if failures:
        raise SystemExit("Benchmark incomplete; report includes failures. See model logs.")
