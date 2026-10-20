# SAM 3 segmentation demo

`demo.py --segmenter sam3` uses Meta's `facebook/sam3` through Transformers. SAM 3
finds instances of the object concepts supplied with `--sam3-prompts`. These
are object names, separate from the full relation vocabulary used by RelSGG.
The default concepts are a small example list; supply concepts appropriate to
your images for broader coverage.

```bash
python demo.py --segmenter sam3 --images photo.jpg --sam3-prompts person horse "wooden fence"
python demo.py --segmenter sam3 --image-dir /path/to/images --output-dir runs/sam3_demo --sam3-prompts person car bicycle
```

The Hugging Face account used by the environment must have access to
[`facebook/sam3`](https://huggingface.co/facebook/sam3). Authenticate with
`hf auth login` before downloading its gated weights. A local Transformers
checkpoint directory can also be supplied with `--sam3-model /path/to/sam3`.
Transformers is already a core dependency; this backend does not need Ultralytics.

The image features are computed once per image and reused across prompts.
Duplicate instances across prompts are suppressed at mask IoU above 0.8,
retaining the label of the higher scoring prompt. Up to 60 instances are passed
to RelSGG, with masks at the original image resolution. Output JSON records the
backend and prompts for comparison. GPU routing and image pipelining follow the
existing demo behavior.

To run the YOLOE comparison:

```bash
python demo.py --segmenter yolo --detector yoloe-26m-seg-pf.pt --images photo.jpg --output-dir runs/yolo_demo
```

API reference: [SAM 3 in Transformers](https://huggingface.co/docs/transformers/model_doc/sam3).
