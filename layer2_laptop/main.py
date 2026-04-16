"""TRINETRA laptop AI orchestrator."""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer1_drone.drone_stream import DroneStream
try:
    from .model1_surveillance.boundary_extractor import BoundaryExtractor
    from .model1_surveillance.grid_heuristics import GridHeuristics
    from .model1_surveillance.node_latlon import NodeLatLon
    from .model1_surveillance.path_planner import PathPlanner
    from .model1_surveillance.terrain_detector import TerrainDetector
    from .navigation_controller import MissionState, NavigationController
    from .map_visualizer import MapVisualizer
    from .rover_api import RoverAPI
    from .ui_controller import UIController
    from .utils.config import (
        DRONE_OCR_EVERY_N_FRAMES,
        DRONE_OCR_INTERVAL_SECONDS,
        DRONE_DEFAULT_ALT,
        DRONE_DEFAULT_LAT,
        DRONE_DEFAULT_LON,
        DRONE_AUTODETECT_MAX_SOURCES,
        DRONE_BLOCKED_SOURCES,
        DRONE_OSD_KEYWORDS,
        DRONE_SIMULATION,
        DRONE_SOURCE_FALLBACK,
        DRONE_VIDEO_SOURCE,
        EMERGENCY_FALLBACK,
        ROVER_BASE_URL,
        ROVER_CAMERA_URL,
        ROVER_SIMULATION,
        ROVER_TIMEOUT_SECONDS,
        STRICT_REAL_DATA,
    )
except ImportError:
    from model1_surveillance.boundary_extractor import BoundaryExtractor
    from model1_surveillance.grid_heuristics import GridHeuristics
    from model1_surveillance.node_latlon import NodeLatLon
    from model1_surveillance.path_planner import PathPlanner
    from model1_surveillance.terrain_detector import TerrainDetector
    from navigation_controller import MissionState, NavigationController
    from map_visualizer import MapVisualizer
    from rover_api import RoverAPI
    from ui_controller import UIController
    from utils.config import (
        DRONE_OCR_EVERY_N_FRAMES,
        DRONE_OCR_INTERVAL_SECONDS,
        DRONE_DEFAULT_ALT,
        DRONE_DEFAULT_LAT,
        DRONE_DEFAULT_LON,
        DRONE_AUTODETECT_MAX_SOURCES,
        DRONE_BLOCKED_SOURCES,
        DRONE_OSD_KEYWORDS,
        DRONE_SIMULATION,
        DRONE_SOURCE_FALLBACK,
        DRONE_VIDEO_SOURCE,
        EMERGENCY_FALLBACK,
        ROVER_BASE_URL,
        ROVER_CAMERA_URL,
        ROVER_SIMULATION,
        ROVER_TIMEOUT_SECONDS,
        STRICT_REAL_DATA,
    )


class TrinetraSystem:
    def __init__(self) -> None:
        self.grid_size = 20
        self.image_w = 1000
        self.image_h = 1000

        self.mode = "target"
        blocked_sources = []
        if DRONE_BLOCKED_SOURCES:
            for token in DRONE_BLOCKED_SOURCES.split(","):
                token = token.strip()
                if token.isdigit():
                    blocked_sources.append(int(token))
        self.drone = DroneStream(
            simulation=DRONE_SIMULATION,
            video_source=DRONE_VIDEO_SOURCE,
            ocr_interval_s=DRONE_OCR_INTERVAL_SECONDS,
            ocr_every_n_frames=DRONE_OCR_EVERY_N_FRAMES,
            fallback_lat=DRONE_DEFAULT_LAT,
            fallback_lon=DRONE_DEFAULT_LON,
            fallback_altitude=DRONE_DEFAULT_ALT,
            allow_source_fallback=DRONE_SOURCE_FALLBACK,
            blocked_sources=blocked_sources,
            strict_real_data=STRICT_REAL_DATA,
            emergency_fallback=EMERGENCY_FALLBACK,
        )
        self.rover_api = RoverAPI(
            ip=ROVER_BASE_URL,
            camera_url=ROVER_CAMERA_URL,
            timeout_s=ROVER_TIMEOUT_SECONDS,
            simulation=ROVER_SIMULATION,
            strict_real_data=STRICT_REAL_DATA,
            emergency_fallback=EMERGENCY_FALLBACK,
        )

        self.terrain_detector = TerrainDetector()
        self.grid_builder = GridHeuristics(grid_size=self.grid_size, obstacle_inflation_px=12)
        self.boundary_extractor = BoundaryExtractor(grid_size=self.grid_size)
        self.map_visualizer = MapVisualizer()
        self.drone_autodetect_max_sources = DRONE_AUTODETECT_MAX_SOURCES
        self.drone_osd_keywords = [k.strip() for k in DRONE_OSD_KEYWORDS.split(",") if k.strip()]

        self.drone_data: Optional[dict] = None
        self.grid: Optional[List[List[dict]]] = None
        self.terrain_data: Optional[dict] = None
        self.selected_target: Optional[dict] = None
        self.target_images: List = []
        self._last_open_ratio: float = 0.0

        self._planner: Optional[PathPlanner] = None
        self._nav: Optional[NavigationController] = None
        self._mission_thread: Optional[threading.Thread] = None
        self._state_cb: Optional[Callable[[str], None]] = None

    def set_ui_state_callback(self, callback: Callable[[str], None]) -> None:
        self._state_cb = callback

    def _emit_state(self, state: MissionState) -> None:
        if self._state_cb:
            self._state_cb(state.value)

    def set_mode(self, mode: str) -> None:
        if mode in ("target", "surveillance"):
            self.mode = mode

    def load_drone_snapshot(self, force_refresh: bool = True) -> Tuple[bool, str]:
        try:
            if self.drone_data is None or force_refresh:
                self.drone_data = self.drone.capture_snapshot()

            image = self.drone_data["image"]
            self.terrain_data = self.terrain_detector.detect(image, pitch_deg=self.drone_data["pitch"], roll_deg=self.drone_data["roll"])
            self.grid = self.grid_builder.build(self.terrain_data["open_mask"])
            self._last_open_ratio = float(np.count_nonzero(self.terrain_data["open_mask"])) / float(self.terrain_data["open_mask"].size)

            mapper = NodeLatLon(
                drone_lat=self.drone_data["lat"],
                drone_lon=self.drone_data["lon"],
                altitude=self.drone_data["altitude"],
                fov_h=90.0,
                fov_v=90.0,
                grid_size=self.grid_size,
                image_w=self.image_w,
                image_h=self.image_h,
                yaw_deg=self.drone_data["yaw"],
            )
            for row in self.grid:
                for node in row:
                    mapper.calculate(node)

            self.selected_target = None
            self.map_visualizer.set_base_image(image)
            self.map_visualizer.set_grid(self.grid)
            self.map_visualizer.start()
            source = "SIMULATION" if DRONE_SIMULATION else "WEBCAM"
            active = self.drone.get_active_source()
            return True, f"Drone snapshot loaded and map processed ({source}, source={active})"
        except Exception as exc:
            return False, f"Failed to process drone snapshot: {exc}"

    def get_drone_frame(self):
        try:
            snap = self.drone.capture_snapshot()
            return snap.get("image")
        except Exception:
            return None

    def auto_select_drone_source(self) -> Tuple[bool, str]:
        try:
            ok, msg = self.drone.auto_select_source(
                max_sources=self.drone_autodetect_max_sources,
                osd_keywords=self.drone_osd_keywords,
            )
            return ok, msg
        except Exception as exc:
            return False, f"Auto source detect failed: {exc}"

    def cycle_drone_source(self, step: int, max_sources: int = 8) -> Tuple[bool, str]:
        try:
            current = self.drone.get_configured_source()
            if not isinstance(current, int):
                current = 0
            total = max(1, int(max_sources))
            for _ in range(total):
                next_idx = (current + step) % total
                current = next_idx
                if next_idx in self.drone.blocked_sources:
                    continue
                self.drone.set_video_source(next_idx)
                frame = self.get_drone_frame()
                if frame is not None:
                    return True, f"Drone source switched to index {next_idx}"
            return False, "No usable drone source found while cycling"
        except Exception as exc:
            return False, f"Failed to switch drone source: {exc}"

    def get_display_map(self):
        if self.terrain_data is None or self.grid is None:
            return None
        # If segmentation confidence is low (typical with indoor FPV debug feed),
        # show raw drone frame with grid overlay instead of nearly-black terrain map.
        if self._last_open_ratio < 0.03 and self.drone_data is not None:
            canvas = cv2.resize(self.drone_data["image"].copy(), (self.image_w, self.image_h))
            cv2.putText(
                canvas,
                "LOW TERRAIN CONFIDENCE - SHOWING RAW DRONE FEED",
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 215, 255),
                2,
            )
        else:
            canvas = cv2.resize(self.terrain_data["terrain_map"].copy(), (self.image_w, self.image_h))
        cell_w = self.image_w // self.grid_size
        cell_h = self.image_h // self.grid_size

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                x1, y1 = c * cell_w, r * cell_h
                x2, y2 = x1 + cell_w, y1 + cell_h
                node = self.grid[r][c]
                if self._last_open_ratio >= 0.03 and node["status"] == "BLOCKED":
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (45, 45, 45), -1)
                elif self._last_open_ratio >= 0.03 and node["status"] == "PARTIAL":
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 170, 220), -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 70), 1)

        if self.selected_target is not None:
            sx1 = self.selected_target["col"] * cell_w
            sy1 = self.selected_target["row"] * cell_h
            sx2 = sx1 + cell_w
            sy2 = sy1 + cell_h
            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), (0, 0, 255), 3)
            cv2.putText(canvas, "TARGET", (sx1 + 2, sy1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return canvas

    def select_target_from_map(self) -> Tuple[bool, str]:
        if self.grid is None:
            ok, msg = self.load_drone_snapshot(force_refresh=True)
            if not ok:
                return False, msg

        display = self.get_display_map()
        if display is None:
            return False, "Map not available"

        selected = {"node": None}
        cell_w = self.image_w // self.grid_size
        cell_h = self.image_h // self.grid_size

        def on_click(event, x, y, _flags, _param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            row = min(self.grid_size - 1, y // cell_h)
            col = min(self.grid_size - 1, x // cell_w)
            node = self.grid[row][col]
            if node["status"] == "BLOCKED":
                return
            selected["node"] = node

        window = "TRINETRA Target Selection"
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_click)

        while True:
            frame = display.copy()
            if selected["node"] is not None:
                r, c = selected["node"]["row"], selected["node"]["col"]
                cv2.rectangle(frame, (c * cell_w, r * cell_h), ((c + 1) * cell_w, (r + 1) * cell_h), (0, 0, 255), 3)
            cv2.imshow(window, frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 and selected["node"] is not None:
                break
            if key in (ord("q"), 27):
                cv2.destroyWindow(window)
                return False, "Target selection canceled"

        cv2.destroyWindow(window)
        self.selected_target = selected["node"]
        return True, f"Target selected: ({self.selected_target['row']}, {self.selected_target['col']})"

    def set_target_images(self, paths: List[str]) -> Tuple[bool, str]:
        images = []
        for path in paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        if len(images) < 2 or len(images) > 5:
            return False, "Please provide between 2 and 5 valid images"
        self.target_images = images
        return True, f"Loaded {len(images)} target images"

    def start_mission(self) -> Tuple[bool, str]:
        if self._mission_thread and self._mission_thread.is_alive():
            return False, "Mission already running"

        if self.grid is None:
            ok, msg = self.load_drone_snapshot(force_refresh=True)
            if not ok:
                return False, msg

        self._planner = PathPlanner()
        self._planner.set_grid(self.grid, self.grid_size)

        if self.mode == "surveillance":
            boundary = self.boundary_extractor.extract(self.grid)
            ordered = self.boundary_extractor.order_nodes(boundary)
            if not ordered:
                return False, "No surveillance boundary path available from real Model1 output"
            self._planner.build_path(ordered)
            mission_target = ordered[-1]
            enable_vision = False
        else:
            if self.selected_target is None:
                return False, "Select a target node first"
            if len(self.target_images) < 2:
                return False, "Upload 2-5 target images first"

            rover_state = self.rover_api.get_state()
            start_node = self._planner.nearest_node(rover_state["lat"], rover_state["lon"], traversable_only=True)
            if start_node is None:
                return False, "Could not map rover location to grid"

            path = self._planner.astar(start_node["row"], start_node["col"], self.selected_target["row"], self.selected_target["col"])
            if not path:
                return False, "No valid path found to selected target"
            mission_target = self.selected_target
            enable_vision = True

        self._nav = NavigationController(
            planner=self._planner,
            rover_api=self.rover_api,
            backend_url="",
            gps_tolerance_m=3.5,
            vision_switch_m=6.0,
            vision_conf_threshold=0.55,
            camera_interval_s=0.6,
            enable_vision=enable_vision,
            map_visualizer=self.map_visualizer,
            state_cb=self._emit_state,
        )

        if self.target_images:
            self._nav.set_target_images(self.target_images)

        def run_mission() -> None:
            result = self._nav.start(mission_target)
            if result == MissionState.ERROR and self._state_cb:
                err = self._nav.last_error or "Unknown error"
                self._state_cb(f"ERROR: {err}")

        self._mission_thread = threading.Thread(target=run_mission, daemon=True)
        self._mission_thread.start()
        return True, f"Mission started in {self.mode} mode"

    def stop_mission(self) -> Tuple[bool, str]:
        if self._nav is None:
            return True, "No active mission"
        self._nav.stop()
        return True, "Mission stop requested"

    def get_camera_frame(self):
        try:
            return self.rover_api.get_camera_frame()
        except Exception:
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description="TRINETRA laptop orchestrator")
    parser.add_argument("--headless", action="store_true", help="Run a non-UI smoke mission and exit")
    parser.add_argument("--headless-seconds", type=float, default=5.0, help="Headless mission runtime before stop")
    args = parser.parse_args()

    system = TrinetraSystem()

    if args.headless:
        ok, msg = system.load_drone_snapshot(force_refresh=True)
        print(f"[HEADLESS] load_drone_snapshot: ok={ok} msg='{msg}'")
        if not ok:
            raise SystemExit(1)

        system.set_mode("surveillance")
        ok, msg = system.start_mission()
        print(f"[HEADLESS] start_mission: ok={ok} msg='{msg}'")
        if not ok:
            raise SystemExit(1)

        time.sleep(max(1.0, args.headless_seconds))
        state = system._nav.state.value if system._nav else "UNKNOWN"
        system.stop_mission()
        print(f"[HEADLESS] stop_mission: state='{state}'")
        return

    ui = UIController(system)
    ui.start()


if __name__ == "__main__":
    main()
