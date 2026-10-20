"""Text-prompted SAM 3 regions for the native relation demo."""
import numpy as np
import torch


class Sam3Segmenter:
    def __init__(self, model_id, device, prompts, confidence=0.4, max_det=60):
        from transformers import Sam3Model, Sam3Processor

        self.model = Sam3Model.from_pretrained(model_id).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.device = device
        self.prompts = list(dict.fromkeys(p.strip() for p in prompts if p.strip()))
        if not self.prompts:
            raise ValueError("SAM 3 needs at least one nonempty object prompt")
        self.confidence = confidence
        self.max_det = max_det

    @torch.inference_mode()
    def __call__(self, image):
        # Encode each image once, then query its features for each concept.
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_vision_features(pixel_values=inputs.pixel_values)
        candidates = []
        for prompt in self.prompts:
            text = self.processor(text=prompt, return_tensors="pt").to(self.device)
            outputs = self.model(vision_embeds=features, **text)
            result = self.processor.post_process_instance_segmentation(
                outputs, threshold=self.confidence, mask_threshold=0.5,
                target_sizes=[[image.height, image.width]],
            )[0]
            for box, mask, score in zip(result["boxes"], result["masks"], result["scores"]):
                mask = mask.cpu().numpy().astype(bool)
                if mask.shape != (image.height, image.width):
                    raise ValueError("SAM 3 mask does not match original image dimensions")
                if mask.any():
                    candidates.append((float(score), box.cpu().numpy(), mask, prompt))

        # Related prompts can return the same instance. Suppress by mask IoU,
        # rather than box overlap, so nearby objects keep their own masks.
        kept = []
        for candidate in sorted(candidates, key=lambda c: c[0], reverse=True):
            mask = candidate[2]
            if any(np.count_nonzero(mask & other[2]) /
                   np.count_nonzero(mask | other[2]) > 0.8 for other in kept):
                continue
            kept.append(candidate)
            if len(kept) >= self.max_det:
                break
        if not kept:
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0, image.height, image.width), bool), [])
        return (np.asarray([c[1] for c in kept], dtype=np.float32),
                np.stack([c[2] for c in kept]), [c[3] for c in kept])
