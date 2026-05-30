from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    mp = None


class FaceService:
    def __init__(self) -> None:
        self.backend = None
        self.detector = None
        self.mesh = None
        self.cascade = None

    def _ensure_models(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        if self.backend is not None:
            return

        # MediaPipe 0.10+ no longer exposes `mp.solutions` in this install.
        if mp is not None and hasattr(mp, "solutions"):
            try:
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
                self.backend = "mediapipe"
                return
            except Exception:
                # Fall back to OpenCV if MediaPipe solutions are unavailable
                # or fail to initialize in this environment.
                self.detector = None
                self.mesh = None

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("OpenCV face detector cascade could not be loaded")
        self.backend = "opencv"

    @staticmethod
    def _crop_embedding(crop: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        vector = resized.astype(np.float32).flatten()
        vector -= float(vector.mean())

        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm

    def detect_faces(self, image_bgr: np.ndarray) -> List[dict]:
        self._ensure_models()
        boxes = []

        if self.backend == "mediapipe":
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            result = self.detector.process(image_rgb)
            if not result.detections:
                return boxes

            height, width, _ = image_bgr.shape
            for detection in result.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0.0, bbox.xmin) * width
                y = max(0.0, bbox.ymin) * height
                w = bbox.width * width
                h = bbox.height * height
                boxes.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "confidence": float(detection.score[0]) if detection.score else 0.0,
                    }
                )
            return boxes

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        for x, y, w, h in faces:
            boxes.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                    "confidence": 1.0,
                }
            )
        return boxes

    def extract_embedding(self, image_bgr: np.ndarray, bbox: dict) -> Optional[np.ndarray]:
        self._ensure_models()
        x1 = max(int(bbox["x"]), 0)
        y1 = max(int(bbox["y"]), 0)
        x2 = max(int(bbox["x"] + bbox["width"]), x1 + 1)
        y2 = max(int(bbox["y"] + bbox["height"]), y1 + 1)

        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if self.backend == "mediapipe" and self.mesh is not None:
            image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mesh_result = self.mesh.process(image_rgb)
            if not mesh_result.multi_face_landmarks:
                return None

            landmarks = mesh_result.multi_face_landmarks[0].landmark
            vector = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32).flatten()
            if vector.size == 0:
                return None

            norm = np.linalg.norm(vector)
            if norm == 0:
                return None
            return vector / norm

        return self._crop_embedding(crop)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def match_face(
        self, embedding: Optional[np.ndarray], known: List[Tuple[int, str, Optional[np.ndarray]]]
    ) -> Optional[Tuple[int, str, float]]:
        if embedding is None:
            return None
        best = None
        for face_id, name, known_emb in known:
            if known_emb is None:
                continue
            score = self.cosine_similarity(embedding, known_emb)
            if best is None or score > best[2]:
                best = (face_id, name, score)
        return best
