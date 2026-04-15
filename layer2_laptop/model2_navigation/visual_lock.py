"""Visual target lock using ORB feature matching."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


class VisualLock:
    def __init__(self, min_good_matches: int = 14, ratio_test: float = 0.75) -> None:
        self.orb = cv2.ORB_create(1200)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_good_matches = min_good_matches
        self.ratio_test = ratio_test
        self.target_descriptors: List[np.ndarray] = []

    def set_target_images(self, images: List[np.ndarray]) -> None:
        self.target_descriptors.clear()
        for img in images:
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, descriptors = self.orb.detectAndCompute(gray, None)
            if descriptors is not None and len(descriptors) > 0:
                self.target_descriptors.append(descriptors)

    def detect(self, frame: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        if frame is None or not self.target_descriptors:
            return False, 0.0, {"offset_px": 0.0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, frame_des = self.orb.detectAndCompute(gray, None)
        if frame_des is None or not keypoints:
            return False, 0.0, {"offset_px": 0.0}

        best_good = []
        for target_des in self.target_descriptors:
            pairs = self.matcher.knnMatch(target_des, frame_des, k=2)
            good = []
            for pair in pairs:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.ratio_test * n.distance:
                    good.append(m)
            if len(good) > len(best_good):
                best_good = good

        if len(best_good) < self.min_good_matches:
            return False, 0.0, {"offset_px": 0.0}

        xs = [keypoints[m.trainIdx].pt[0] for m in best_good]
        obj_x = float(sum(xs) / len(xs))
        frame_center = frame.shape[1] / 2.0
        offset = obj_x - frame_center

        confidence = min(1.0, len(best_good) / float(self.min_good_matches * 2))
        return True, confidence, {"offset_px": offset}