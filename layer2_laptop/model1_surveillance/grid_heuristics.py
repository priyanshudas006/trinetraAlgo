"""Grid construction + traversability heuristics."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


class GridHeuristics:
    """Builds the 20x20 (configurable) planning grid."""

    def __init__(self, grid_size: int = 20, obstacle_inflation_px: int = 10) -> None:
        self.grid_size = grid_size
        self.obstacle_inflation_px = obstacle_inflation_px

    def build(self, open_mask: np.ndarray) -> List[List[dict]]:
        h, w = open_mask.shape[:2]

        blocked_mask = cv2.bitwise_not(open_mask)
        kernel = np.ones((self.obstacle_inflation_px, self.obstacle_inflation_px), dtype=np.uint8)
        inflated_blocked = cv2.dilate(blocked_mask, kernel)
        inflated_open = cv2.bitwise_not(inflated_blocked)

        dist_to_blocked = cv2.distanceTransform(inflated_open, cv2.DIST_L2, 3)
        max_dist = float(np.max(dist_to_blocked)) if float(np.max(dist_to_blocked)) > 0 else 1.0

        cell_h = h // self.grid_size
        cell_w = w // self.grid_size
        grid: List[List[dict]] = []

        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                cell_open = inflated_open[y1:y2, x1:x2]
                cell_dist = dist_to_blocked[y1:y2, x1:x2]

                open_ratio = float(np.count_nonzero(cell_open)) / float(cell_open.size)
                mean_dist = float(np.mean(cell_dist)) / max_dist

                if open_ratio >= 0.75:
                    status = "SAFE"
                elif open_ratio >= 0.35:
                    status = "PARTIAL"
                else:
                    status = "BLOCKED"

                # Lower heuristic preferred; combines roughness + obstacle proximity.
                heuristic = round((1.0 - open_ratio) + (1.0 - mean_dist), 3)

                row.append(
                    {
                        "row": r,
                        "col": c,
                        "status": status,
                        "heuristic": heuristic,
                        "lat": None,
                        "lon": None,
                        "center_px": (x1 + (cell_w // 2), y1 + (cell_h // 2)),
                    }
                )
            grid.append(row)
        return grid