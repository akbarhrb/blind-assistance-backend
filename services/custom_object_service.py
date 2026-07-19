from __future__ import annotations

import pickle
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

# Minimum RANSAC-inlier matches required before a reference object counts as
# recognized in a frame. Raw ORB/BF matches are noisy; requiring a geometric
# homography to agree on this many points filters out coincidental matches.
MIN_MATCH_COUNT = 12
# Lowe's ratio test threshold for accepting a keypoint match as "good".
LOWE_RATIO = 0.75
ORB_FEATURES = 500


class CustomObjectService:
    """Recognizes user-enrolled custom objects in a camera frame by matching
    ORB keypoint descriptors against each object's stored reference image,
    then fitting a homography to localize a bounding box."""

    def __init__(self) -> None:
        self._orb = None
        self._matcher = None

    def _ensure_models(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed")
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        if self._matcher is None:
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def extract_descriptor(self, image_bgr: np.ndarray) -> Optional[dict]:
        """Extracts keypoints/descriptors from a reference object image for
        storage. Returns None if too few features were found to be useful."""
        self._ensure_models()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < MIN_MATCH_COUNT:
            return None

        height, width = gray.shape[:2]
        return {
            "version": 1,
            "keypoints": [
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
                for kp in keypoints
            ],
            "descriptors": descriptors,
            "shape": (width, height),
        }

    @staticmethod
    def _to_keypoints(raw_keypoints) -> List["cv2.KeyPoint"]:
        return [
            cv2.KeyPoint(
                x=float(kp[0]),
                y=float(kp[1]),
                size=float(kp[2]),
                angle=float(kp[3]),
                response=float(kp[4]),
                octave=int(kp[5]),
                class_id=int(kp[6]),
            )
            for kp in raw_keypoints
        ]

    def match(
        self, frame_bgr: np.ndarray, known: List[Tuple[int, str, dict]]
    ) -> List[dict]:
        """known: list of (object_id, name, descriptor_dict). Returns boxes in
        the same shape as YoloService.detect() (x/y/width/height/label/confidence)."""
        self._ensure_models()
        if not known:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self._orb.detectAndCompute(gray, None)
        if frame_descriptors is None or len(frame_keypoints) < 2:
            return []

        frame_height, frame_width = frame_bgr.shape[:2]
        boxes = []

        for object_id, name, raw in known:
            if not raw or raw.get("descriptors") is None:
                continue

            ref_descriptors = raw["descriptors"]
            ref_keypoints = self._to_keypoints(raw["keypoints"])
            ref_width, ref_height = raw["shape"]
            if len(ref_keypoints) < MIN_MATCH_COUNT:
                continue

            try:
                pairs = self._matcher.knnMatch(ref_descriptors, frame_descriptors, k=2)
            except cv2.error:
                continue

            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance]
            if len(good) < MIN_MATCH_COUNT:
                continue

            src_pts = np.float32([ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if homography is None or mask is None:
                continue

            inliers = int(mask.sum())
            if inliers < MIN_MATCH_COUNT:
                continue

            corners = np.float32(
                [[0, 0], [ref_width, 0], [ref_width, ref_height], [0, ref_height]]
            ).reshape(-1, 1, 2)
            projected = cv2.perspectiveTransform(corners, homography)
            xs = projected[:, 0, 0]
            ys = projected[:, 0, 1]

            x1 = float(np.clip(xs.min(), 0, frame_width))
            x2 = float(np.clip(xs.max(), 0, frame_width))
            y1 = float(np.clip(ys.min(), 0, frame_height))
            y2 = float(np.clip(ys.max(), 0, frame_height))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue

            confidence = float(min(1.0, inliers / max(len(ref_keypoints), MIN_MATCH_COUNT)))

            boxes.append(
                {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "label": name,
                    "confidence": confidence,
                    "object_id": object_id,
                }
            )

        return boxes

    @staticmethod
    def dump_descriptor(descriptor: Optional[dict]) -> Optional[bytes]:
        if descriptor is None:
            return None
        return pickle.dumps(descriptor)

    @staticmethod
    def load_descriptor(blob: Optional[bytes]):
        if not blob:
            return None
        return pickle.loads(blob)
