"""Rover API adapter with optional built-in simulator."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import requests


@dataclass
class RoverState:
    lat: float
    lon: float
    heading: float


class _RoverSimulator:
    def __init__(self, lat: float = 29.8996, lon: float = 78.0995) -> None:
        self.lat = lat
        self.lon = lon
        self.heading = 0.0
        self.speed_mps = 1.0
        self.turn_rate_deg = 10.0
        self.last_tick = time.time()
        self.last_cmd = "STOP"
        self.target_lat: Optional[float] = None
        self.target_lon: Optional[float] = None
        self.target_signature: Optional[np.ndarray] = None
        self.obstacles = [(29.8999, 78.0999), (29.9002, 78.1002)]
        self.servo_triggered = False

    def send_command(self, cmd: str) -> None:
        self._tick()
        self.last_cmd = cmd

    def get_state(self) -> dict:
        self._tick()
        return {"lat": self.lat, "lon": self.lon, "heading": self.heading}

    def get_sensor(self) -> dict:
        obstacle = any(self._distance_m(self.lat, self.lon, o_lat, o_lon) < 4.0 for o_lat, o_lon in self.obstacles)
        gas = 250.0 + (450.0 if obstacle else 60.0)
        metal = 220.0 + (520.0 if obstacle else 70.0)
        return {"metal": metal, "gas": gas, "obstacle": obstacle}

    def get_camera_frame(self) -> np.ndarray:
        frame = np.full((480, 640, 3), 90, dtype=np.uint8)
        cv2.putText(frame, "SIM CAMERA", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)

        if self.target_lat is not None and self.target_lon is not None and self.target_signature is not None:
            dist = self._distance_m(self.lat, self.lon, self.target_lat, self.target_lon)
            if dist < 7.0:
                bearing = self._bearing_deg(self.lat, self.lon, self.target_lat, self.target_lon)
                error = self._normalize_angle(bearing - self.heading)
                x = int(320 + max(-180, min(180, error * 3)))
                x = max(40, min(600, x))
                y = 240
                patch = cv2.resize(self.target_signature, (90, 90))
                y1, y2 = y - 45, y + 45
                x1, x2 = x - 45, x + 45
                frame[y1:y2, x1:x2] = patch
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        return frame

    def set_target(self, lat: float, lon: float) -> None:
        self.target_lat = lat
        self.target_lon = lon

    def set_target_signature(self, images: List[np.ndarray]) -> None:
        self.target_signature = images[0] if images else None

    def trigger_servo(self) -> bool:
        self.servo_triggered = True
        return True

    def _tick(self) -> None:
        now = time.time()
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now

        if self.last_cmd == "LEFT":
            self.heading = (self.heading - self.turn_rate_deg * dt * 8.0) % 360.0
            return
        if self.last_cmd == "RIGHT":
            self.heading = (self.heading + self.turn_rate_deg * dt * 8.0) % 360.0
            return
        if self.last_cmd != "FORWARD":
            return

        d = self.speed_mps * dt
        heading_rad = math.radians(self.heading)
        north = math.cos(heading_rad) * d
        east = math.sin(heading_rad) * d

        self.lat += north / 111111.0
        self.lon += east / (111111.0 * math.cos(math.radians(self.lat)))

    @staticmethod
    def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius = 6371000.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return radius * c

    @staticmethod
    def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(p2)
        y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle


class RoverAPI:
    """Hardware API with simulation fallback.

    JSON contracts:
    - state: {lat, lon, heading}
    - sensor: {metal, gas, obstacle}
    - command POST: {cmd: FORWARD|LEFT|RIGHT|STOP}
    """

    def __init__(
        self,
        ip: str,
        timeout_s: float = 2.0,
        simulation: bool = False,
        camera_url: Optional[str] = None,
        camera_timeout_s: float = 0.35,
    ) -> None:
        self.ip = ip.rstrip("/")
        self.timeout_s = timeout_s
        self.camera_url = (camera_url or f"{self.ip}/camera").rstrip("/")
        self.camera_timeout_s = camera_timeout_s
        self.simulation = simulation
        self.sim = _RoverSimulator() if simulation else None
        self._last_state = {"lat": 0.0, "lon": 0.0, "heading": 0.0}
        self._last_sensor = {"metal": 0.0, "gas": 0.0, "obstacle": False}
        self._camera_cap: Optional[cv2.VideoCapture] = None
        self._last_camera_frame: Optional[np.ndarray] = None
        self._state_failures = 0
        self._sensor_failures = 0
        self._command_failures = 0

    def get_state(self) -> dict:
        if self.simulation and self.sim is not None:
            return self.sim.get_state()
        try:
            response = requests.get(f"{self.ip}/state", timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
            self._last_state = {
                "lat": self._coerce_float(payload, ("lat", "latitude"), self._last_state["lat"]),
                "lon": self._coerce_float(payload, ("lon", "lng", "longitude"), self._last_state["lon"]),
                "heading": self._coerce_float(payload, ("heading", "yaw", "hdg"), self._last_state["heading"]),
            }
            self._state_failures = 0
        except Exception:
            self._state_failures += 1
        return dict(self._last_state)

    def send_command(self, cmd: str) -> bool:
        if cmd not in ("FORWARD", "LEFT", "RIGHT", "STOP"):
            return False
        if self.simulation and self.sim is not None:
            self.sim.send_command(cmd)
            return True
        try:
            response = requests.post(f"{self.ip}/command", json={"cmd": cmd, "command": cmd}, timeout=self.timeout_s)
            ok = response.status_code < 300
            self._command_failures = 0 if ok else (self._command_failures + 1)
            return ok
        except Exception:
            self._command_failures += 1
            return False

    def get_sensor(self) -> dict:
        if self.simulation and self.sim is not None:
            return self.sim.get_sensor()
        try:
            response = requests.get(f"{self.ip}/sensor", timeout=self.timeout_s)
            response.raise_for_status()
            payload = response.json()
            self._last_sensor = {
                "metal": self._coerce_float(payload, ("metal", "metal_value", "metalValue"), self._last_sensor["metal"]),
                "gas": self._coerce_float(payload, ("gas", "mq2", "gas_value", "gasValue"), self._last_sensor["gas"]),
                "obstacle": self._coerce_bool(payload, ("obstacle", "obstacle_detected", "hasObstacle"), self._last_sensor["obstacle"]),
            }
            self._sensor_failures = 0
        except Exception:
            self._sensor_failures += 1
        return dict(self._last_sensor)

    def get_camera_frame(self) -> np.ndarray:
        if self.simulation and self.sim is not None:
            return self.sim.get_camera_frame()
        if self._is_stream_source(self.camera_url):
            image = self._read_stream_frame()
            if image is not None:
                self._last_camera_frame = image
                return image
        else:
            image = self._read_snapshot_frame()
            if image is not None:
                self._last_camera_frame = image
                return image

        if self._last_camera_frame is None:
            raise RuntimeError("Camera frame unavailable")
        return self._last_camera_frame

    def set_target(self, lat: float, lon: float) -> None:
        if self.simulation and self.sim is not None:
            self.sim.set_target(lat, lon)
            return
        requests.post(f"{self.ip}/mission/target", json={"lat": lat, "lon": lon}, timeout=self.timeout_s)

    def set_target_signature(self, images: List[np.ndarray]) -> None:
        if self.simulation and self.sim is not None:
            self.sim.set_target_signature(images)

    def trigger_servo(self) -> bool:
        if self.simulation and self.sim is not None:
            return self.sim.trigger_servo()
        try:
            response = requests.post(f"{self.ip}/servo", json={"action": "TRIGGER"}, timeout=self.timeout_s)
            return response.status_code < 300
        except Exception:
            return False

    def close(self) -> None:
        if self._camera_cap is not None:
            self._camera_cap.release()
            self._camera_cap = None

    def _read_stream_frame(self) -> Optional[np.ndarray]:
        if self._camera_cap is None:
            self._camera_cap = cv2.VideoCapture(self.camera_url)
            if not self._camera_cap.isOpened():
                self._camera_cap.release()
                self._camera_cap = None
                return None
            self._camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._camera_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.camera_timeout_s * 1000))
            self._camera_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.camera_timeout_s * 1000))
        ok, frame = self._camera_cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _read_snapshot_frame(self) -> Optional[np.ndarray]:
        try:
            response = requests.get(self.camera_url, timeout=self.camera_timeout_s)
            response.raise_for_status()
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            return image
        except Exception:
            return None

    @staticmethod
    def _is_stream_source(url: str) -> bool:
        u = url.lower()
        return u.startswith(("rtsp://", "udp://")) or "/stream" in u or u.endswith(".mjpg")

    @staticmethod
    def _coerce_float(payload: dict, keys: tuple[str, ...], default: float) -> float:
        for key in keys:
            if key in payload:
                try:
                    return float(payload[key])
                except Exception:
                    continue
        return float(default)

    @staticmethod
    def _coerce_bool(payload: dict, keys: tuple[str, ...], default: bool) -> bool:
        for key in keys:
            if key in payload:
                value = payload[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value != 0
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(default)

    def is_link_healthy(self, max_failures: int = 4) -> bool:
        if self.simulation:
            return True
        return (
            self._state_failures < max_failures
            and self._sensor_failures < max_failures
            and self._command_failures < max_failures
        )
