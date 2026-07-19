from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    mp = None
    BaseOptions = None
    mp_vision = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
DEFAULT_LANDMARKER_MODEL_PATH = os.path.join(_BACKEND_DIR, "assets", "face_landmarker.task")
FACE_LANDMARKER_MODEL_PATH = os.getenv("FACE_LANDMARKER_MODEL_PATH", DEFAULT_LANDMARKER_MODEL_PATH)

# Embedding envelope format. Bumping this whenever the vector layout changes lets
# match_face() refuse to compare embeddings across incompatible spaces instead of
# silently producing a meaningless cosine similarity.
EMBEDDING_VERSION = 2
EMBEDDING_METHOD = "landmark_hog_v1"

# Iris centers from the 478-point MediaPipe face mesh topology. Stable across
# expression changes, which makes them good rotation/scale alignment anchors.
LEFT_IRIS_IDX = 473
RIGHT_IRIS_IDX = 468

ALIGN_SIZE = 128
# Fraction of ALIGN_SIZE the inter-ocular distance is stretched to.
TARGET_EYE_FRACTION = 0.32
# Vertical placement (from the top) of the eye line in the aligned crop.
TARGET_EYE_Y_FRACTION = 0.42

# Raw HOG cosine similarities for aligned faces cluster tightly at the high end
# (same person ~0.95-1.0, different people ~0.75-0.85 in local testing) because
# HOG on any two aligned, frontal faces shares a lot of generic structure. These
# constants rescale that narrow band into a full 0-1 range so a threshold in the
# conventional 0.6-0.7 zone behaves the way it would for a typical face-embedding
# model. Re-tune with real enrollment data if matches feel off.
MATCH_SCORE_LOW = 0.75
MATCH_SCORE_HIGH = 0.95


class FaceService:
    def __init__(self) -> None:
        self.backend = None
        self.detector = None
        self.cascade = None

        self.landmarker = None
        self.landmarker_unavailable_reason: Optional[str] = None

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
                self.backend = "mediapipe"
                return
            except Exception:
                # Fall back to OpenCV if MediaPipe solutions are unavailable
                # or fail to initialize in this environment.
                self.detector = None

        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("OpenCV face detector cascade could not be loaded")
        self.backend = "opencv"

    def _ensure_landmarker(self) -> None:
        if self.landmarker is not None or self.landmarker_unavailable_reason is not None:
            return

        if mp_vision is None:
            self.landmarker_unavailable_reason = "mediapipe tasks API is not available"
            return
        if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
            self.landmarker_unavailable_reason = (
                f"face landmarker model not found at {FACE_LANDMARKER_MODEL_PATH}"
            )
            return

        try:
            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH),
                num_faces=1,
                min_face_detection_confidence=0.5,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - defensive
            self.landmarker_unavailable_reason = f"failed to load face landmarker: {exc}"

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

    @staticmethod
    def _padded_crop(image_bgr: np.ndarray, bbox: dict, margin: float = 0.35) -> Optional[np.ndarray]:
        height, width = image_bgr.shape[:2]
        pad_x = bbox["width"] * margin
        pad_y = bbox["height"] * margin
        x1 = max(int(bbox["x"] - pad_x), 0)
        y1 = max(int(bbox["y"] - pad_y), 0)
        x2 = min(int(bbox["x"] + bbox["width"] + pad_x), width)
        y2 = min(int(bbox["y"] + bbox["height"] + pad_y), height)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_bgr[y1:y2, x1:x2]
        return crop if crop.size else None

    def _landmarks_for_crop(self, crop: np.ndarray):
        self._ensure_landmarker()
        if self.landmarker is None:
            return None
        image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    @staticmethod
    def _align_face(crop: np.ndarray, landmarks) -> Optional[np.ndarray]:
        height, width = crop.shape[:2]
        left_eye = np.array(
            [landmarks[LEFT_IRIS_IDX].x * width, landmarks[LEFT_IRIS_IDX].y * height]
        )
        right_eye = np.array(
            [landmarks[RIGHT_IRIS_IDX].x * width, landmarks[RIGHT_IRIS_IDX].y * height]
        )
        inter_ocular = float(np.linalg.norm(left_eye - right_eye))
        if inter_ocular < 1e-3:
            return None

        eye_center = (left_eye + right_eye) / 2.0
        dx, dy = left_eye - right_eye
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        scale = (ALIGN_SIZE * TARGET_EYE_FRACTION) / inter_ocular

        rotation = cv2.getRotationMatrix2D(tuple(eye_center), angle_deg, scale)
        rotation[0, 2] += ALIGN_SIZE / 2 - eye_center[0]
        rotation[1, 2] += ALIGN_SIZE * TARGET_EYE_Y_FRACTION - eye_center[1]

        return cv2.warpAffine(
            crop,
            rotation,
            (ALIGN_SIZE, ALIGN_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    _hog = None
    _face_mask = None
    _clahe = None

    @classmethod
    def _get_hog(cls):
        if cls._hog is None:
            cls._hog = cv2.HOGDescriptor(
                _winSize=(ALIGN_SIZE, ALIGN_SIZE),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )
        return cls._hog

    @classmethod
    def _get_face_mask(cls):
        if cls._face_mask is None:
            mask = np.zeros((ALIGN_SIZE, ALIGN_SIZE), dtype=np.uint8)
            center = (ALIGN_SIZE // 2, int(ALIGN_SIZE * 0.46))
            axes = (int(ALIGN_SIZE * 0.30), int(ALIGN_SIZE * 0.40))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            cls._face_mask = mask
        return cls._face_mask

    @classmethod
    def _get_clahe(cls):
        if cls._clahe is None:
            cls._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return cls._clahe

    @classmethod
    def _hog_vector(cls, gray: np.ndarray) -> Optional[np.ndarray]:
        feat = cls._get_hog().compute(gray)
        if feat is None:
            return None
        feat = feat.flatten().astype(np.float32)
        feat -= float(feat.mean())
        norm = np.linalg.norm(feat)
        if norm == 0:
            return None
        return feat / norm

    def extract_embedding(self, image_bgr: np.ndarray, bbox: dict) -> Optional[dict]:
        self._ensure_models()
        crop = self._padded_crop(image_bgr, bbox)
        if crop is None:
            return None

        landmarks = self._landmarks_for_crop(crop)
        if landmarks is not None:
            aligned = self._align_face(crop, landmarks)
            if aligned is not None:
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                gray = self._get_clahe().apply(gray)
                gray = cv2.bitwise_and(gray, gray, mask=self._get_face_mask())

                v_normal = self._hog_vector(gray)
                v_mirror = self._hog_vector(cv2.flip(gray, 1))
                vectors = [v for v in (v_normal, v_mirror) if v is not None]
                if vectors:
                    return {
                        "version": EMBEDDING_VERSION,
                        "method": EMBEDDING_METHOD,
                        "vectors": vectors,
                    }

        # Fall back to the coarse legacy descriptor if landmark alignment isn't
        # available (missing model, no landmarks found, degenerate geometry).
        # This is intentionally tagged as legacy so match_face() never conflates
        # it with the more discriminative aligned/HOG embeddings above.
        return self._crop_embedding(crop)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _calibrate(raw_score: float) -> float:
        calibrated = (raw_score - MATCH_SCORE_LOW) / (MATCH_SCORE_HIGH - MATCH_SCORE_LOW)
        return float(min(1.0, max(0.0, calibrated)))

    @staticmethod
    def _embedding_vectors(raw) -> Tuple[Optional[int], List[np.ndarray]]:
        """Returns (version, vectors) for either the new envelope format or a
        legacy raw ndarray, so callers never need to know the storage details."""
        if isinstance(raw, dict) and raw.get("version") == EMBEDDING_VERSION:
            return EMBEDDING_VERSION, list(raw.get("vectors") or [])
        if isinstance(raw, np.ndarray):
            return 1, [raw]
        return None, []

    def match_face(
        self, embedding, known: List[Tuple[int, str, object]]
    ) -> Optional[Tuple[int, str, float, float]]:
        """Returns (face_id, name, match_score, raw_similarity) for the best
        match, where match_score is what should be compared against the
        configured threshold. Returns None if there's nothing to compare."""
        query_version, query_vectors = self._embedding_vectors(embedding)
        if query_version is None or not query_vectors:
            return None

        best = None
        for face_id, name, known_raw in known:
            if known_raw is None:
                continue
            known_version, known_vectors = self._embedding_vectors(known_raw)
            if known_version is None or not known_vectors or known_version != query_version:
                # Different embedding spaces (e.g. legacy vs. new) aren't
                # comparable - skip rather than produce a meaningless score.
                continue

            raw_score = max(
                self.cosine_similarity(qv, kv) for qv in query_vectors for kv in known_vectors
            )
            score = self._calibrate(raw_score) if query_version == EMBEDDING_VERSION else raw_score

            if best is None or score > best[2]:
                best = (face_id, name, score, raw_score)
        return best

    @staticmethod
    def dump_embedding(embedding) -> Optional[bytes]:
        if embedding is None:
            return None
        return pickle.dumps(embedding)

    @staticmethod
    def load_embedding(blob: Optional[bytes]):
        if not blob:
            return None
        return pickle.loads(blob)
