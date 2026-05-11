from __future__ import annotations

import os
from typing import List

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None


class YoloService:
    def __init__(self) -> None:
        self.model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.model = None

    def _ensure_model(self) -> None:
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed")
        if self.model is None:
            self.model = YOLO(self.model_path)

    def detect(self, image_bgr: np.ndarray) -> List[dict]:
        self._ensure_model()
        results = self.model.predict(image_bgr, verbose=False)
        if not results:
            return []

        first = results[0]
        boxes = []
        if first.boxes is None:
            return boxes

        for box in first.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            label = first.names.get(cls_id, str(cls_id)) if hasattr(first, "names") else str(cls_id)

            x1, y1, x2, y2 = xyxy
            boxes.append(
                {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "label": label,
                    "confidence": conf,
                }
            )

        return boxes
