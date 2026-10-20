# RF-DETR Segmentation on Kaggle

In the existing RelSGG environment, enable GPU and Internet, then run:

```python
%cd /kaggle/working/RelateAnythingFork
!git pull origin test
!python -m pip install "rfdetr==1.11.0"
!python demo.py --segmenter rfdetr --rfdetr-size medium --image-dir "/kaggle/working/data/02_2_3obj" --output-dir "/kaggle/working/rfdetr_02_2_3obj"
```

The backend uses RFDETRSegMedium with its pretrained COCO weights. It only needs
images; no text prompts or boxes are supplied by the user. Weights download on
the first run. `--rfdetr-size` also supports `nano`, `small`, and `large`.
Version 1.11.0 supports Transformers 5, as used by this project. For a fresh
installation of the entire project, the equivalent optional extra is `.[rfdetr]`.

RF-DETR supplies original-resolution instance masks, pixel xyxy boxes and COCO
class IDs. Empty masks are dropped and up to 60 regions are retained in descending
confidence order (threshold 0.4). Labels, masks and boxes keep the same ordering
when passed to RelSGG. The full relation vocabulary is used as in the other demo
backends. Output JSON records the RF-DETR variant; mask and relation PNGs use the
existing directory layout. Input subdirectories are scanned recursively.

Mask2Former remains the default when `--segmenter` is omitted. Use separate output
directories to compare the backends on the same images. Offline adapter and demo
tests are provided; pretrained inference and dependency installation must still
be verified in the Kaggle environment.

API reference: https://rfdetr.roboflow.com/latest/learn/run/segmentation/
