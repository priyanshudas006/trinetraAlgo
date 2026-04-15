"""Runtime navigation loop with state machine."""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable, List, Optional

try:
    from .model2_navigation.heading_calculator import HeadingCalculator
    from .model2_navigation.visual_lock import VisualLock
    from .model2_navigation.waypoint_sender import WaypointSender
    from .model3_sensor.threshold_checker import ThresholdChecker
    from .model3_sensor.backend_poster import BackendPoster
except ImportError:
    from model2_navigation.heading_calculator import HeadingCalculator
    from model2_navigation.visual_lock import VisualLock
    from model2_navigation.waypoint_sender import WaypointSender
    from model3_sensor.threshold_checker import ThresholdChecker
    from model3_sensor.backend_poster import BackendPoster


class MissionState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    NAVIGATING = "NAVIGATING"
    AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
    VISION_LOCK = "VISION_LOCK"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class NavigationController:
    def __init__(
        self,
        planner,
        rover_api,
        backend_url: str = "http://localhost:3000/api/hazards",
        gps_tolerance_m: float = 3.0,
        vision_switch_m: float = 6.0,
        vision_conf_threshold: float = 0.55,
        camera_interval_s: float = 0.6,
        enable_vision: bool = True,
        state_cb: Optional[Callable[[MissionState], None]] = None,
    ) -> None:
        self.planner = planner
        self.rover_api = rover_api
        self.heading = HeadingCalculator()
        self.visual = VisualLock()
        self.sensor = ThresholdChecker()
        self.backend = BackendPoster(url=backend_url)
        self.sender = WaypointSender(rover_ip=rover_api.ip)

        self.gps_tolerance_m = gps_tolerance_m
        self.vision_switch_m = vision_switch_m
        self.vision_conf_threshold = vision_conf_threshold
        self.camera_interval_s = camera_interval_s
        self.enable_vision = enable_vision

        self.target_node: Optional[dict] = None
        self.running = False
        self.state: MissionState = MissionState.IDLE
        self.state_cb = state_cb
        self.last_error: Optional[str] = None
        self._last_camera_ts = 0.0
        self._vision_fail_count = 0
        self._smooth_lat: Optional[float] = None
        self._smooth_lon: Optional[float] = None
        self._smooth_heading: Optional[float] = None

    def set_target_images(self, images: List) -> None:
        self.visual.set_target_images(images)
        self.rover_api.set_target_signature(images)

    def set_state(self, state: MissionState) -> None:
        self.state = state
        if self.state_cb:
            self.state_cb(state)

    def start(self, target_node: dict) -> MissionState:
        self.target_node = target_node
        self.running = True
        self._vision_fail_count = 0

        self.set_state(MissionState.PLANNING)
        self.rover_api.set_target(target_node["lat"], target_node["lon"])
        self.sender.push_plan(self.planner.waypoints)

        result = self._run_loop()
        self.running = False
        return result

    def stop(self) -> None:
        self.running = False
        self.rover_api.send_command("STOP")
        if self.state not in (MissionState.COMPLETE, MissionState.ERROR):
            self.set_state(MissionState.IDLE)

    def _run_loop(self) -> MissionState:
        waypoint_index = 0

        try:
            while self.running:
                if not self.rover_api.is_link_healthy(max_failures=5):
                    self.rover_api.send_command("STOP")
                    raise RuntimeError("ESP32 link unhealthy: repeated API/command failures")

                waypoint, waypoint_index = self.planner.get_next_waypoint(waypoint_index)
                if waypoint is None:
                    self.rover_api.send_command("STOP")
                    self.set_state(MissionState.COMPLETE)
                    return MissionState.COMPLETE

                self.set_state(MissionState.NAVIGATING)
                state = self.rover_api.get_state()
                sensor_payload = self.rover_api.get_sensor()

                lat, lon, heading = self._smooth_state(state["lat"], state["lon"], state["heading"])
                dist_to_waypoint = self.heading.haversine_distance(lat, lon, waypoint["lat"], waypoint["lon"])
                dist_to_target = self.heading.haversine_distance(lat, lon, self.target_node["lat"], self.target_node["lon"])

                sensor_with_status = self.sensor.enrich_payload(
                    {
                        "lat": lat,
                        "lon": lon,
                        "metal": sensor_payload.get("metal", 0.0),
                        "gas": sensor_payload.get("gas", 0.0),
                    }
                )
                if sensor_with_status["status"] in ("YELLOW", "RED"):
                    self.backend.post(sensor_with_status)

                if sensor_payload.get("obstacle", False):
                    self.set_state(MissionState.AVOIDING_OBSTACLE)
                    self.rover_api.send_command("STOP")
                    self.rover_api.send_command("RIGHT")
                    time.sleep(0.3)
                    curr_node = self.planner.nearest_node(lat, lon, traversable_only=True)
                    if curr_node is None:
                        raise RuntimeError("Unable to localize rover to grid for replanning")
                    self.planner.mark_blocked(waypoint["row"], waypoint["col"])
                    replanned = self.planner.replan(curr_node["row"], curr_node["col"], self.target_node["row"], self.target_node["col"])
                    if not replanned:
                        raise RuntimeError("Replanning failed: no feasible path")
                    waypoint_index = 0
                    continue

                if dist_to_waypoint <= self.gps_tolerance_m:
                    self.planner.mark_visited(waypoint_index)
                    waypoint_index += 1
                    continue

                if self.enable_vision and dist_to_target <= self.vision_switch_m:
                    self.set_state(MissionState.VISION_LOCK)
                    if self._vision_step():
                        if not self.rover_api.trigger_servo():
                            raise RuntimeError("Servo actuation failed")
                        self.rover_api.send_command("STOP")
                        self.set_state(MissionState.COMPLETE)
                        return MissionState.COMPLETE
                    if self._vision_fail_count >= 8:
                        # Vision fallback: resume GPS guidance temporarily.
                        self._vision_fail_count = 0
                    time.sleep(0.15)
                    continue

                bearing = self.heading.calculate_bearing(lat, lon, waypoint["lat"], waypoint["lon"])
                error = self.heading.heading_error(heading, bearing)
                cmd = self.heading.get_motion_command(error, dist_to_waypoint, turn_threshold=8, stop_distance=self.gps_tolerance_m)
                self.rover_api.send_command(cmd)
                time.sleep(0.35)

        except Exception as exc:
            self.last_error = str(exc)
            self.rover_api.send_command("STOP")
            self.set_state(MissionState.ERROR)
            return MissionState.ERROR

        self.set_state(MissionState.IDLE)
        return MissionState.IDLE

    def _vision_step(self) -> bool:
        now = time.time()
        if now - self._last_camera_ts < self.camera_interval_s:
            return False
        self._last_camera_ts = now

        try:
            frame = self.rover_api.get_camera_frame()
        except Exception:
            self._vision_fail_count += 1
            return False
        if frame is None:
            self._vision_fail_count += 1
            return False
        detected, confidence, details = self.visual.detect(frame)

        if not detected or confidence < self.vision_conf_threshold:
            self._vision_fail_count += 1
            self.rover_api.send_command("FORWARD")
            return False

        self._vision_fail_count = 0
        offset = details.get("offset_px", 0.0)
        if abs(offset) <= 22:
            self.rover_api.send_command("STOP")
            return True
        self.rover_api.send_command("LEFT" if offset < 0 else "RIGHT")
        return False

    def _smooth_state(self, lat: float, lon: float, heading: float) -> tuple[float, float, float]:
        alpha = 0.35
        if self._smooth_lat is None:
            self._smooth_lat = float(lat)
            self._smooth_lon = float(lon)
            self._smooth_heading = float(heading)
            return self._smooth_lat, self._smooth_lon, self._smooth_heading

        self._smooth_lat = (1.0 - alpha) * self._smooth_lat + alpha * float(lat)
        self._smooth_lon = (1.0 - alpha) * self._smooth_lon + alpha * float(lon)
        current = self._smooth_heading if self._smooth_heading is not None else float(heading)
        delta = self.heading.heading_error(current, float(heading))
        self._smooth_heading = (current + (alpha * delta)) % 360.0
        return self._smooth_lat, self._smooth_lon, self._smooth_heading
