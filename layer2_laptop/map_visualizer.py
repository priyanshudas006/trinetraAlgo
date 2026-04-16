"""Real-time OpenCV mission map visualizer."""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
try:
    from .utils.debug import debug_log
except ImportError:
    try:
        from layer2_laptop.utils.debug import debug_log
    except ImportError:
        from utils.debug import debug_log


class MapVisualizer:
    def __init__(self, window_name: str = "TRINETRA Real-Time Map", refresh_hz: float = 8.0) -> None:
        self.window_name = window_name
        self.refresh_hz = max(2.0, refresh_hz)
        self._lock = threading.Lock()
        self._base_image: Optional[np.ndarray] = None
        self._grid: Optional[List[List[dict]]] = None
        self._path: List[dict] = []
        self._hazards: List[dict] = []
        self._rover: Optional[dict] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_rover_log_ts = 0.0

    def set_base_image(self, image: np.ndarray) -> None:
        with self._lock:
            self._base_image = image.copy() if image is not None else None

    def set_grid(self, grid: List[List[dict]]) -> None:
        with self._lock:
            self._grid = grid

    def set_path(self, waypoints: List[dict]) -> None:
        with self._lock:
            self._path = list(waypoints or [])

    def update_rover(self, lat: float, lon: float) -> None:
        with self._lock:
            self._rover = {"lat": float(lat), "lon": float(lon)}
        now = time.time()
        if now - self._last_rover_log_ts >= 0.75:
            self._last_rover_log_ts = now
            debug_log("MAP", "Rover position updated")

    def add_hazard(self, lat: float, lon: float, status: str) -> None:
        with self._lock:
            self._hazards.append({"lat": float(lat), "lon": float(lon), "status": status, "ts": time.time()})
            self._hazards = self._hazards[-250:]
        debug_log("MAP", f"Plotting hazard at ({lat:.6f},{lon:.6f})")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        cv2.destroyWindow(self.window_name)

    def _render_loop(self) -> None:
        delay = 1.0 / self.refresh_hz
        while self._running:
            frame = self._compose_frame()
            if frame is not None:
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
            time.sleep(delay)

    def _compose_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            base = self._base_image.copy() if self._base_image is not None else None
            grid = self._grid
            path = list(self._path)
            hazards = list(self._hazards)
            rover = dict(self._rover) if self._rover else None

        if base is None:
            return None

        canvas = cv2.resize(base, (1000, 700))
        if grid:
            self._draw_grid(canvas, grid)
        if path:
            self._draw_path(canvas, path)
        for hz in hazards:
            px = self._latlon_to_pixel(hz["lat"], hz["lon"], grid, canvas.shape[1], canvas.shape[0])
            if px is None:
                continue
            color = (0, 0, 255) if hz["status"] == "RED" else (0, 220, 255)
            cv2.circle(canvas, px, 6, color, -1)
        if rover:
            px = self._latlon_to_pixel(rover["lat"], rover["lon"], grid, canvas.shape[1], canvas.shape[0])
            if px is not None:
                cv2.circle(canvas, px, 8, (0, 255, 0), -1)
                cv2.putText(canvas, "ROVER", (px[0] + 6, px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.putText(canvas, "BLUE=PATH GREEN=ROVER RED/YELLOW=HAZARD", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    @staticmethod
    def _draw_grid(canvas: np.ndarray, grid: List[List[dict]]) -> None:
        if not grid:
            return
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        if rows == 0 or cols == 0:
            return
        h, w = canvas.shape[:2]
        cell_w = max(1, w // cols)
        cell_h = max(1, h // rows)
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 1)

    def _draw_path(self, canvas: np.ndarray, waypoints: List[dict]) -> None:
        pts = []
        for wp in waypoints:
            px = self._latlon_to_pixel(wp.get("lat"), wp.get("lon"), self._grid, canvas.shape[1], canvas.shape[0])
            if px:
                pts.append(px)
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (255, 120, 0), 2)
        for p in pts:
            cv2.circle(canvas, p, 3, (255, 120, 0), -1)

    @staticmethod
    def _latlon_to_pixel(lat: float, lon: float, grid: Optional[List[List[dict]]], w: int, h: int) -> Optional[Tuple[int, int]]:
        if grid is None or lat is None or lon is None:
            return None
        best = None
        best_d = float("inf")
        for row in grid:
            for node in row:
                nlat = node.get("lat")
                nlon = node.get("lon")
                if nlat is None or nlon is None:
                    continue
                d = (float(nlat) - float(lat)) ** 2 + (float(nlon) - float(lon)) ** 2
                if d < best_d:
                    best_d = d
                    best = node
        if best is None:
            return None
        cx, cy = best.get("center_px", (0, 0))
        gx, gy = int(cx), int(cy)
        # center_px is based on 1000x1000 layout in planner pipeline.
        px = int((gx / 1000.0) * w)
        py = int((gy / 1000.0) * h)
        return (max(0, min(w - 1, px)), max(0, min(h - 1, py)))
