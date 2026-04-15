"""Boundary extraction for surveillance patrol mode."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


class BoundaryExtractor:
    def __init__(self, grid_size: int = 20) -> None:
        self.grid_size = grid_size

    def extract(self, grid: List[List[dict]]) -> List[dict]:
        boundary_nodes: List[dict] = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                node = grid[r][c]
                if node["status"] == "BLOCKED":
                    continue
                if self._touches_blocked(grid, r, c):
                    boundary_nodes.append(node)
        return boundary_nodes

    def order_nodes(self, nodes: List[dict]) -> List[dict]:
        if not nodes:
            return []
        remaining = nodes.copy()
        ordered = [remaining.pop(0)]
        while remaining:
            last = ordered[-1]
            next_idx = min(
                range(len(remaining)),
                key=lambda i: abs(remaining[i]["row"] - last["row"]) + abs(remaining[i]["col"] - last["col"]),
            )
            ordered.append(remaining.pop(next_idx))
        return ordered

    def visualize(self, grid: List[List[dict]], ordered: List[dict], cell_px: int = 35) -> np.ndarray:
        image = np.zeros((self.grid_size * cell_px, self.grid_size * cell_px, 3), dtype=np.uint8)
        color = {"SAFE": (50, 160, 70), "PARTIAL": (0, 180, 220), "BLOCKED": (40, 40, 40)}

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                node = grid[r][c]
                x1, y1 = c * cell_px, r * cell_px
                x2, y2 = x1 + cell_px, y1 + cell_px
                cv2.rectangle(image, (x1, y1), (x2, y2), color[node["status"]], -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (20, 20, 20), 1)

        for i in range(1, len(ordered)):
            p1 = (ordered[i - 1]["col"] * cell_px + (cell_px // 2), ordered[i - 1]["row"] * cell_px + (cell_px // 2))
            p2 = (ordered[i]["col"] * cell_px + (cell_px // 2), ordered[i]["row"] * cell_px + (cell_px // 2))
            cv2.line(image, p1, p2, (255, 255, 255), 2)

        return image

    def _touches_blocked(self, grid: List[List[dict]], row: int, col: int) -> bool:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if nr < 0 or nr >= self.grid_size or nc < 0 or nc >= self.grid_size:
                continue
            if grid[nr][nc]["status"] == "BLOCKED":
                return True
        return False