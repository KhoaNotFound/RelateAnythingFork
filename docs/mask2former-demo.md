# Image-only Mask2Former demo

The default `demo.py` backend is `facebook/mask2former-swin-small-coco-panoptic`.
It takes only an image and predicts foreground instances and background regions
from its COCO panoptic classes. No text prompts or external detector are needed.

```bash
python demo.py --images photo.jpg --output-dir runs/mask2former_demo
python demo.py --image-dir /path/to/images --output-dir runs/mask2former_demo
```

With no image arguments, the demo uses `assets/reel/images/horse.jpg`.
The first run downloads the segmentation and RelSGG checkpoints from Hugging Face.
Transformers is already a project dependency. Supply a compatible local panoptic
checkpoint using `--mask2former-model /path/to/model` if needed.

Each predicted segment becomes a boolean mask at the original image resolution.
Its bounding box is derived from that mask. Separate instances of the same class
remain separate; stuff regions are not explicitly fused by class. The highest
scoring 60 nonempty regions are passed to RelSGG. The relation vocabulary remains
the full vocabulary from the RelSGG checkpoint. JSON and mask/relation PNGs are
saved in the output directory; JSON records the segmenter checkpoint used.

For a comparison on the same image:

```bash
python demo.py --segmenter yolo --images photo.jpg --output-dir runs/yolo_demo
```

The checkpoint has a fixed segmentation label set; objects outside it may be
missed. Better segmentation does not by itself establish better relation accuracy.
Compare both pipelines on the same images and hardware.

Reference: https://huggingface.co/facebook/mask2former-swin-small-coco-panoptic

## Kaggle

Enable a GPU accelerator and Internet in the notebook settings. In the existing
repository checkout, pull the updated branch and use the environment already
prepared for RelSGG:

```python
%cd /kaggle/working/RelateAnythingFork
!git pull origin test
!python demo.py --segmenter mask2former --image-dir /kaggle/input/YOUR_DATASET/images --output-dir /kaggle/working/mask2former_output
```

For a first smoke run with the bundled image:

```python
!python demo.py --segmenter mask2former --output-dir /kaggle/working/mask2former_smoke
```

The first run downloads the checkpoints. One visible GPU runs both stages;
two visible GPUs put segmentation on GPU 0 and RelSGG on GPU 1. No local GPU
setup or SAM 3 weights are required. This integration has offline conversion
and demo tests; end-to-end model inference must still be verified on Kaggle.
