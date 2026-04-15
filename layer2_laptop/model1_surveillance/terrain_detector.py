"""Terrain segmentation for traversability map generation."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


class TerrainDetector:
    """Segments image into traversable mask + visualization map."""

    def __init__(self) -> None:
        # HSV thresholds for synthetic/field grass-earth zones.
        self.open_lower = np.array([35, 30, 30], dtype=np.uint8)
        self.open_upper = np.array([95, 255, 255], dtype=np.uint8)

    def detect(self, image: np.ndarray, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> Dict[str, np.ndarray]:
        corrected = self._imu_compensate(image, pitch_deg, roll_deg)
        hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)

        open_mask = cv2.inRange(hsv, self.open_lower, self.open_upper)
        open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        open_mask = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

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