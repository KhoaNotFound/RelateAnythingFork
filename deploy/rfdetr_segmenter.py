"""Image-only RF-DETR instance segmentation for the relation demo."""
import numpy as np


def detection_regions(detections, names, image_size, max_det=60):
    width, height = image_size
    boxes = np.asarray(detections.xyxy, dtype=np.float32)
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0, height, width), bool), []
    if detections.mask is None:
        raise ValueError("RF-DETR returned boxes without instance masks")
    masks = np.asarray(detections.mask, dtype=bool)
    if masks.shape != (len(boxes), height, width):
        raise ValueError("RF-DETR masks do not align with boxes/original image")
    valid = masks.reshape(len(boxes), -1).any(axis=1)
    order = np.argsort(-np.asarray(detections.confidence), kind="stable")
    order = order[valid[order]][:max_det]
    labels = [names[int(detections.class_id[i])] for i in order]
    return boxes[order], masks[order], labels


class RFDETRSegmenter:
    def __init__(self, size, device, confidence=0.4, max_det=60):
        try:
            import rfdetr
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ModuleNotFoundError as error:
            if error.name != "rfdetr":
                raise
            raise ImportError('Install RF-DETR with: python -m pip install "rfdetr==1.11.0"') from error
        classes = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall",
                   "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge"}
        self.model = getattr(rfdetr, classes[size])(device=device)
        self.names = COCO_CLASSES
        self.confidence = confidence
        self.max_det = max_det

    def __call__(self, image):
        detections = self.model.predict(image.convert("RGB"), threshold=self.confidence)
        return detection_regions(detections, self.names, image.size, self.max_det)
