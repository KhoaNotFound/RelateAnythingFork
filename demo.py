"""Full-vocabulary mask demo, with a two-GPU pipeline for multiple images.

    python demo.py
    python demo.py --images photo1.jpg photo2.jpg photo3.jpg

With two visible GPUs, segmentation uses GPU 0 and relations use GPU 1.
For one image the stages are sequential; with multiple images detection of
the next image overlaps relation prediction of the current image.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def select_devices(gpu_count):
    if gpu_count >= 2:
        return "cuda:0", "cuda:1"
    if gpu_count == 1:
        return "cuda:0", "cuda:0"
    return "cpu", "cpu"


def iter_detected(paths, detect):
    """One detector worker; at most one extra image is prefetched."""
    paths = iter(paths)
    first = next(paths, None)
    if first is None:
        return
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(detect, first)
        for path in paths:
            current = pending.result()
            pending = worker.submit(detect, path)
            yield current
        yield pending.result()


def predict_relations(model, image, boxes, masks, labels, topk):
    options = dict(masks=masks, box_labels=labels, topk=topk)
    triplets = model.predict(image, boxes, **options)
    for triplet in triplets:
        print(triplet)
    graphs = model.predict(image, boxes, decompose=True, **options)
    print("Spatial:", graphs["spatial"])
    print("Semantic:", graphs["semantic"])
    return {"relations": triplets, **graphs}


def collect_images(images=None, image_dir=None):
    if image_dir is not None:
        if not image_dir.is_dir():
            raise ValueError(f"Image directory not found: {image_dir}")
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
        paths = sorted(p for p in image_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in extensions)
    else:
        paths = images or [Path(__file__).resolve().parent / "assets/reel/images/horse.jpg"]
    if not paths:
        raise ValueError("No supported images found")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Image not found: {path}")
    return paths


def output_base(path, image_dir, output_dir):
    if image_dir is not None:
        # Preserve source subdirectories and extensions: no raw_1 collisions.
        return output_dir / path.resolve().relative_to(image_dir.resolve())
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
    return output_dir / f"{path.name}-{digest}"


def save_outputs(image, boxes, masks, labels, predictions, base, source, settings):
    from deploy.demo_visuals import save_visuals

    visuals = save_visuals(image, boxes, masks, labels, predictions, base)
    result = dict(source=str(source.resolve()), settings=settings,
                  boxes=boxes.tolist(), labels=labels, visuals=visuals,
                  predictions={name: [asdict(t) for t in trips]
                               for name, trips in predictions.items()})
    path = Path(f"{base}.json")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                               default=lambda value: value.tolist()), encoding="utf-8")
    print(f"Saved: {visuals['relations']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images", nargs="+", type=Path)
    source.add_argument("--image-dir", type=Path, help="Recursively read images outside or inside the repo")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/demo_output"))
    parser.add_argument("--detector", default="yolo26m-seg.pt")
    parser.add_argument("--model", default="maelic/relsgg-vits16plus")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    if args.topk < 1:
        parser.error("--topk must be positive")
    try:
        paths = collect_images(args.images, args.image_dir)
    except ValueError as error:
        parser.error(str(error))
    resolved_output = args.output_dir.resolve()
    # Do not recursively pick up prior output images on a subsequent run.
    if args.image_dir and resolved_output.is_relative_to(args.image_dir.resolve()):
        parser.error("--output-dir must be outside --image-dir")

    from PIL import Image
    import numpy as np
    import torch
    from ultralytics import YOLO
    from relsgg import RelateAnything

    gpu_count = torch.cuda.device_count()
    detector_device, relation_device = select_devices(gpu_count)
    print(f"Visible GPUs: {gpu_count}")
    for idx in range(gpu_count):
        print(f"  cuda:{idx}: {torch.cuda.get_device_name(idx)}")
    print(f"Segmentation: {detector_device}; relations: {relation_device}")
    if gpu_count < 2:
        print("Two GPUs are not visible; using the available GPU or CPU.")
    if len(paths) == 1:
        print("One image: stages use their assigned devices sequentially.")
    else:
        print("Pipelining next-image segmentation with current-image relations.")

    # The detector is created and used exclusively by the single worker.
    detector = None

    def detect(path):
        nonlocal detector
        if detector is None:
            detector = YOLO(args.detector)
            if detector.task != "segment":
                raise ValueError("--detector must be an instance-segmentation checkpoint")
        with Image.open(path) as source:
            image = source.convert("RGB")
        detections = detector.predict(
            image, device=detector_device, conf=0.4, max_det=60,
            retina_masks=True, verbose=False,
        )[0]
        if detections.boxes is None or len(detections.boxes) == 0:
            return (path, image, np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, image.height, image.width), dtype=bool), [])
        if detections.masks is None:
            raise ValueError(f"{path}: no masks returned; refusing to silently use boxes only")
        # GPU inference is complete; the public relation API takes NumPy inputs.
        boxes = detections.boxes.xyxy.cpu().numpy()
        masks = (detections.masks.data > 0.5).cpu().numpy()
        if masks.shape != (len(boxes), image.height, image.width):
            raise ValueError(f"{path}: masks do not align with boxes/original image")
        labels = [detections.names[int(c)] for c in detections.boxes.cls.cpu().tolist()]
        return path, image, boxes, masks, labels

    model = RelateAnything.from_pretrained(
        args.model, device=relation_device, full_vocabulary=True,
    )
    print(f"Relation vocabulary: {len(model.predicates)} predicates")
    # Keep the full vocabulary for both predictions; do not call set_vocabulary.
    settings = dict(detector=args.detector, relation_model=args.model,
                    detector_device=detector_device, relation_device=relation_device,
                    full_vocabulary=True, vocabulary_size=len(model.predicates),
                    topk=args.topk, detector_conf=0.4)
    for path, image, boxes, masks, labels in iter_detected(paths, detect):
        print(f"\nImage: {path}")
        if len(boxes) < 2:
            print("Fewer than two objects detected; no relations to predict.")
            predictions = {"relations": [], "spatial": [], "semantic": []}
        else:
            predictions = predict_relations(model, image, boxes, masks, labels, args.topk)
        base = output_base(path, args.image_dir, args.output_dir)
        save_outputs(image, boxes, masks, labels, predictions, base, path, settings)


if __name__ == "__main__":
    main()
