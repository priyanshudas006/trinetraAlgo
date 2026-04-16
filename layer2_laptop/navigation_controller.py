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
    from .utils.debug import debug_log
except ImportError:
    from model2_navigation.heading_calculator import HeadingCalculator
    from model2_navigation.visual_lock import VisualLock
    from model2_navigation.waypoint_sender import WaypointSender
    from model3_sensor.threshold_checker import ThresholdChecker
    from model3_sensor.backend_poster import BackendPoster
    try:
        from layer2_laptop.utils.debug import debug_log
    except ImportError:
        from utils.debug import debug_log


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
        backend_url: str = "",
        gps_tolerance_m: float = 3.0,
        vision_switch_m: float = 6.0,
        vision_conf_threshold: float = 0.55,
        camera_interval_s: float = 0.6,
        enable_vision: bool = True,
        map_visualizer=None,
        state_cb: Optional[Callable[[MissionState], None]] = None,
    ) -> None:
        self.planner = planner
        self.rover_api = rover_api
        self.heading = HeadingCalculator()
        self.visual = VisualLock()
        self.sensor = ThresholdChecker()
        self.backend = BackendPoster(url=backend_url)
        self.sender = WaypointSender(rover_ip=rover_api.ip)
        self.map_visualizer = map_visualizer

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

        try:
            self.set_state(MissionState.PLANNING)
            self.rover_api.set_target(target_node["lat"], target_node["lon"])
            debug_log("PATH", f"Total waypoints: {len(self.planner.waypoints)}")
            for i, waypoint in enumerate(self.planner.waypoints):
                lat = float(waypoint.get("lat") or 0.0)
                lon = float(waypoint.get("lon") or 0.0)
                debug_log("WAYPOINT", f"{i} -> Lat={lat:.6f}, Lon={lon:.6f}")
            if not self.sender.push_plan(self.planner.waypoints):
                raise RuntimeError("Failed to upload waypoint plan to ESP32 endpoint /mission/path")
            if self.map_visualizer is not None:
                self.map_visualizer.set_path(self.planner.waypoints)
        except Exception as exc:
            self.last_error = str(exc)
            debug_log("ERROR", str(exc))
            self.rover_api.send_command("STOP")
            self.set_state(MissionState.ERROR)
            self.running = False
            return MissionState.ERROR

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
                self._validate_state_payload(state)
                self._validate_sensor_payload(sensor_payload)

                lat, lon, heading = self._smooth_state(state["lat"], state["lon"], state["heading"])
                if self.map_visualizer is not None:
                    self.map_visualizer.update_rover(lat, lon)
                dist_to_waypoint = self.heading.haversine_distance(lat, lon, waypoint["lat"], waypoint["lon"])
                dist_to_target = self.heading.haversine_distance(lat, lon, self.target_node["lat"], self.target_node["lon"])
                bearing = self.heading.calculate_bearing(lat, lon, waypoint["lat"], waypoint["lon"])
                error = self.heading.heading_error(heading, bearing)
                debug_log(
                    "NAV",
                    (
                        f"Current=({lat:.6f},{lon:.6f}) Target=({waypoint['lat']:.6f},{waypoint['lon']:.6f}) "
                        f"Dist={dist_to_waypoint:.2f}m Bearing={bearing:.2f} Error={error:.2f}"
                    ),
                )

                sensor_with_status = self.sensor.enrich_payload(
                    {
                        "lat": lat,
                        "lon": lon,
                        "metal": sensor_payload["metal"],
                        "gas": sensor_payload["gas"],
                    }
                )
                if sensor_with_status["status"] in ("YELLOW", "RED"):
                    hazard_type = self._hazard_type(sensor_with_status["metal"], sensor_with_status["gas"])
                    debug_log(
                        "HAZARD",
                        f"{hazard_type} detected at ({lat:.6f},{lon:.6f}) -> {sensor_with_status['status']}",
                    )
                    self.backend.post(sensor_with_status)
                    if self.map_visualizer is not None:
                        self.map_visualizer.add_hazard(lat, lon, sensor_with_status["status"])

                if sensor_payload["obstacle"]:
                    self.set_state(MissionState.AVOIDING_OBSTACLE)
                    if not self.rover_api.send_command("STOP"):
                        raise RuntimeError("Failed to send STOP command to ESP32")
                    if not self.rover_api.send_command("RIGHT"):
                        raise RuntimeError("Failed to send RIGHT command to ESP32")
                    time.sleep(0.3)
                    curr_node = self.planner.nearest_node(lat, lon, traversable_only=True)
                    if curr_node is None:
                        raise RuntimeError("Unable to localize rover to grid for replanning")
                    self.planner.mark_blocked(waypoint["row"], waypoint["col"])
                    replanned = self.planner.replan(curr_node["row"], curr_node["col"], self.target_node["row"], self.target_node["col"])
                    if not replanned:
                        raise RuntimeError("Replanning failed: no feasible path")
                    if self.map_visualizer is not None:
                        self.map_visualizer.set_path(self.planner.waypoints)
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

                cmd = self.heading.get_motion_command(error, dist_to_waypoint, turn_threshold=8, stop_distance=self.gps_tolerance_m)
                if not self.rover_api.send_command(cmd):
                    raise RuntimeError(f"ESP32 command send failed: {cmd}")
                time.sleep(0.35)

        except Exception as exc:
            self.last_error = str(exc)
            debug_log("ERROR", str(exc))
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
            if not self.rover_api.send_command("FORWARD"):
                raise RuntimeError("ESP32 command send failed during vision lock: FORWARD")
            return False

        self._vision_fail_count = 0
        offset = details.get("offset_px", 0.0)
        if abs(offset) <= 22:
            if not self.rover_api.send_command("STOP"):
                raise RuntimeError("ESP32 command send failed during vision lock: STOP")
            return True
        turn_cmd = "LEFT" if offset < 0 else "RIGHT"
        if not self.rover_api.send_command(turn_cmd):
            raise RuntimeError(f"ESP32 command send failed during vision lock: {turn_cmd}")
        return False

    def _hazard_type(self, metal: float, gas: float) -> str:
        metal_alert = float(metal) >= float(self.sensor.metal_low)
        gas_alert = float(gas) >= float(self.sensor.gas_low)
        if metal_alert and gas_alert:
            return "COMBINED"
        if gas_alert:
            return "GAS"
        if metal_alert:
            return "METAL"
        return "UNKNOWN"

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

    @staticmethod
    def _validate_state_payload(state: dict) -> None:
        for key in ("lat", "lon", "heading"):
            if key not in state or state[key] is None:
                raise RuntimeError(f"Rover state missing required field: {key}")

    @staticmethod
    def _validate_sensor_payload(sensor_payload: dict) -> None:
        for key in ("metal", "gas", "obstacle"):
            if key not in sensor_payload or sensor_payload[key] is None:
                raise RuntimeError(f"Sensor payload missing required field: {key}")
