"""Crack detection and segmentation (component 5).

Wraps a YOLO segmentation model trained on the Ultralytics Crack-seg dataset
(4,029 real photographs of cracks on roads and walls, single `crack` class) and
returns a pixel mask per photograph.

This is the piece component 6 has been waiting for. Measuring crack width needs
a mask, and until now the only one available was the classical stand-in in
crack.py, which was measured at up to 3 mm error on known line widths. With a
real mask the measurement chain is good to +0.08 mm.

Weights are not bundled. Point `CrackSegmenter` at a .pt file: either the
OpenSistemas Crack-seg weights or your own fine-tune.
"""

from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = "models/yolov8n-crack-seg-best.pt"


class CrackSegmenter:
    """Crack masks from a YOLO segmentation checkpoint."""

    def __init__(self, weights=DEFAULT_WEIGHTS, confidence=0.40, device="cpu"):
        """0.40 is the measured operating point.

        Swept against 60 BD3 crack photographs and 60 plain/algae ones (BD3 is
        cross-dataset for these Crack-seg weights): 0.25 gave 100% recall at
        15% false positives, 0.40 kept 100% recall at 10%, and 0.70 dropped
        recall to 82% for only 5%. 0.40 is where recall is still free.
        """
        from ultralytics import YOLO

        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"no crack weights at {path}. Download Crack-seg weights or "
                f"fine-tune your own, then pass weights=...")
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.device = device

    def predict(self, image):
        """Return a combined binary mask for every crack found.

        `image` is a BGR array or a path. The mask is full image resolution and
        0/255, which is what crack.crack_width expects.
        """
        result = self.model.predict(image, conf=self.confidence,
                                    device=self.device, verbose=False)[0]

        height, width = result.orig_shape
        if result.masks is None or len(result.masks) == 0:
            return {"mask": np.zeros((height, width), np.uint8),
                    "instances": 0, "max_confidence": 0.0}

        import cv2
        combined = np.zeros((height, width), np.uint8)
        for polygon in result.masks.xy:
            if len(polygon) >= 3:
                cv2.fillPoly(combined, [polygon.astype(np.int32)], 255)

        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None \
            else np.array([])
        return {
            "mask": combined,
            "instances": int(len(result.masks)),
            "max_confidence": float(scores.max()) if scores.size else 0.0,
            "coverage": float((combined > 0).mean()),
        }

    def has_crack(self, image, min_confidence=None):
        out = self.predict(image)
        floor = self.confidence if min_confidence is None else min_confidence
        return out["instances"] > 0 and out["max_confidence"] >= floor
