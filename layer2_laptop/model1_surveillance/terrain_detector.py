"""Terrain segmentation for traversability map generation."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


class TerrainDetector:
    """Segments image into traversable mask + visualization map."""

    def __init__(self) -> None:
        # Broad vegetation/soil range for field scenes.
        self.open_lower = np.array([20, 20, 25], dtype=np.uint8)
        self.open_upper = np.array([110, 255, 255], dtype=np.uint8)

    def detect(self, image: np.ndarray, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> Dict[str, np.ndarray]:
        corrected = self._imu_compensate(image, pitch_deg, roll_deg)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)

        open_mask = cv2.inRange(hsv, self.open_lower, self.open_upper)
        # Secondary mask catches low-saturation bright traversable zones common in washed FPV feeds.
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        neutral_open = cv2.inRange(v, 75, 255) & cv2.inRange(s, 0, 95)
        open_mask = cv2.bitwise_or(open_mask, neutral_open)

        open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        open_ratio = float(np.count_nonzero(open_mask)) / float(open_mask.size)
        if open_ratio < 0.02 or open_ratio > 0.95:
            # Fallback adaptive segmentation when color thresholds are unreliable.
            gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, adaptive = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            open_mask = adaptive

        terrain_map = corrected.copy()
        blocked = open_mask == 0
        partial = cv2.dilate(open_mask, np.ones((9, 9), np.uint8)) > 0
        terrain_map[blocked] = (20, 20, 20)
        terrain_map[np.logical_and(~blocked, partial)] = (70, 140, 90)

        return {"open_mask": open_mask, "terrain_map": terrain_map}

    def _imu_compensate(self, image: np.ndarray, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """Light affine correction to reduce tilt distortion impact in prototype mode."""
        h, w = image.shape[:2]
        tx = float(roll_deg) * 0.5
        ty = float(pitch_deg) * 0.5
        matrix = np.float32([[1.0, 0.0, tx], [0.0, 1.0, ty]])
        return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
