"""Image-only panoptic regions for RelSGG."""
import numpy as np
import torch


def panoptic_regions(result, id2label, image_size, max_det=60):
    """Convert segment IDs to aligned masks, pixel xyxy boxes and labels."""
    width, height = image_size
    segmentation = result["segmentation"].cpu().numpy()
    if segmentation.shape != (height, width):
        raise ValueError("Panoptic map does not match original image dimensions")
    boxes, masks, labels = [], [], []
    seen = set()
    for segment in sorted(result["segments_info"], key=lambda s: s["score"], reverse=True):
        segment_id = segment["id"]
        if segment_id in seen:
            continue
        seen.add(segment_id)
        mask = segmentation == segment_id
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        masks.append(mask)
        labels.append(id2label[segment["label_id"]])
        if len(boxes) >= max_det:
            break
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool), labels)


class Mask2FormerSegmenter:
    def __init__(self, model_id, device, confidence=0.4, max_det=60):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.confidence = confidence
        self.max_det = max_det

    @torch.inference_mode()
    def __call__(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        result = self.processor.post_process_panoptic_segmentation(
            outputs, threshold=self.confidence, mask_threshold=0.5,
            target_sizes=[(image.height, image.width)],
            # Keep separate predicted regions, including stuff, for relation nodes.
            label_ids_to_fuse=set(),
        )[0]
        return panoptic_regions(result, self.model.config.id2label, image.size, self.max_det)
