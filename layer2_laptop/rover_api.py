"""Rover API adapter with optional built-in simulator."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
import logging
try:
    from .utils.debug import debug_log
except ImportError:
    try:
        from layer2_laptop.utils.debug import debug_log
    except ImportError:
        from utils.debug import debug_log


LOGGER = logging.getLogger(__name__)


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
        strict_real_data: bool = True,
        emergency_fallback: bool = False,
    ) -> None:
        self.ip = ip.rstrip("/")
        self.timeout_s = timeout_s
        self.camera_url = (camera_url or f"{self.ip}/camera").rstrip("/")
        self.camera_timeout_s = camera_timeout_s
        self.simulation = simulation
        self.strict_real_data = bool(strict_real_data)
        self.emergency_fallback = bool(emergency_fallback)
        self.sim = _RoverSimulator() if simulation else None
        if self.strict_real_data and self.simulation:
            raise RuntimeError("STRICT_REAL_DATA enabled: ROVER_SIMULATION must be false")
        self._last_state: Optional[dict] = None
        self._last_sensor: Optional[dict] = None
        self._camera_cap: Optional[cv2.VideoCapture] = None
        self._last_camera_frame: Optional[np.ndarray] = None
        self._state_failures = 0
        self._sensor_failures = 0
        self._command_failures = 0
        self._request_retries = 2

    def get_state(self) -> dict:
        if self.simulation and self.sim is not None:
            return self.sim.get_state()
        payload = self._request_json("GET", "/state")
        if payload is None:
            self._state_failures += 1
            if self.emergency_fallback and self._last_state is not None:
                return dict(self._last_state)
            raise RuntimeError("Rover state missing from ESP32 endpoint /state")

        self._last_state = {
            "lat": self._required_float(payload, ("lat", "latitude"), "/state"),
            "lon": self._required_float(payload, ("lon", "lng", "longitude"), "/state"),
            "heading": self._required_float(payload, ("heading", "yaw", "hdg"), "/state"),
        }
        self._state_failures = 0
        return dict(self._last_state)

    def send_command(self, cmd: str) -> bool:
        if cmd not in ("FORWARD", "LEFT", "RIGHT", "STOP"):
            debug_log("ERROR", f"ESP32 failed: invalid command '{cmd}'")
            return False
        debug_log("ESP32", f"Sending command: {cmd}")
        if self.simulation and self.sim is not None:
            self.sim.send_command(cmd)
            debug_log("ESP32", "Response: 200 (simulation)")
            return True
        payload = {"cmd": cmd, "command": cmd}
        for _ in range(self._request_retries + 1):
            try:
                response = requests.post(f"{self.ip}/command", json=payload, timeout=self.timeout_s)
                debug_log("ESP32", f"Response: {response.status_code}")
                if response.status_code < 300:
                    self._command_failures = 0
                    return True
            except Exception as error:
                debug_log("ERROR", f"ESP32 failed: {error}")
            time.sleep(0.05)
        self._command_failures += 1
        if self._command_failures >= 4 and cmd != "STOP":
            # Best-effort failsafe stop after repeated command send failures.
            try:
                requests.post(f"{self.ip}/command", json={"cmd": "STOP", "command": "STOP"}, timeout=self.timeout_s)
            except Exception as error:
                debug_log("ERROR", f"ESP32 failed: {error}")
                LOGGER.warning("Failed to send failsafe STOP to rover")
        return False

    def get_sensor(self) -> dict:
        if self.simulation and self.sim is not None:
            sensor = self.sim.get_sensor()
            debug_log("SENSOR", f"Gas={sensor['gas']}, Metal={sensor['metal']}")
            return sensor
        payload = self._request_json("GET", "/sensor")
        if payload is None:
            self._sensor_failures += 1
            if self.emergency_fallback and self._last_sensor is not None:
                return dict(self._last_sensor)
            raise RuntimeError("Sensor data missing from ESP32 endpoint /sensor")

        self._last_sensor = {
            "metal": self._required_float(payload, ("metal", "metal_value", "metalValue"), "/sensor"),
            "gas": self._required_float(payload, ("gas", "mq2", "gas_value", "gasValue"), "/sensor"),
            "obstacle": self._required_bool(payload, ("obstacle", "obstacle_detected", "hasObstacle"), "/sensor"),
        }
        self._sensor_failures = 0
        debug_log("SENSOR", f"Gas={self._last_sensor['gas']}, Metal={self._last_sensor['metal']}")
        return dict(self._last_sensor)

    def get_camera_frame(self) -> np.ndarray:
        if self.simulation and self.sim is not None:
            return self.sim.get_camera_frame()
        for url in self._camera_candidates():
            if self._is_stream_source(url):
                image = self._read_stream_frame(url)
            else:
                image = self._read_snapshot_frame(url)
            if image is not None:
                self._last_camera_frame = image
                if url != self.camera_url:
                    self.camera_url = url
                return image

        if self.emergency_fallback and self._last_camera_frame is not None:
            return self._last_camera_frame
        raise RuntimeError("Camera frame unavailable from live rover source")

    def set_target(self, lat: float, lon: float) -> None:
        if self.simulation and self.sim is not None:
            self.sim.set_target(lat, lon)
            return
        try:
            response = requests.post(f"{self.ip}/mission/target", json={"lat": lat, "lon": lon}, timeout=self.timeout_s)
            response.raise_for_status()
        except Exception as error:
            raise RuntimeError(f"Failed to send mission target to ESP32: {error}") from error

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

    def _read_stream_frame(self, url: str) -> Optional[np.ndarray]:
        if getattr(self, "_camera_cap_url", None) != url and self._camera_cap is not None:
            self._camera_cap.release()
            self._camera_cap = None
        if self._camera_cap is None:
            self._camera_cap = cv2.VideoCapture(url)
            if not self._camera_cap.isOpened():
                self._camera_cap.release()
                self._camera_cap = None
                return None
            self._camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._camera_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.camera_timeout_s * 1000))
            self._camera_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.camera_timeout_s * 1000))
            self._camera_cap_url = url
        ok, frame = self._camera_cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _read_snapshot_frame(self, url: str) -> Optional[np.ndarray]:
        try:
            response = requests.get(url, timeout=max(1.5, self.camera_timeout_s))
            response.raise_for_status()
            ctype = (response.headers.get("Content-Type") or "").lower()
            if "image" not in ctype and response.content[:2] != b"\xff\xd8":
                return None
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

    def _camera_candidates(self) -> List[str]:
        parsed = urlparse(self.camera_url)
        if not parsed.scheme or not parsed.netloc:
            return [self.camera_url]
        root = f"{parsed.scheme}://{parsed.netloc}"
        host = parsed.hostname or ""
        out = [self.camera_url]
        # Snapshot-first fallbacks for ESP32-CAM firmwares that expose /capture but not /stream.
        out.extend(
            [
                f"{root}/capture",
                f"{root}/capture?_cb=1",
                f"{root}/stream",
            ]
        )
        if host:
            out.extend(
                [
                    f"{parsed.scheme}://{host}:81/stream",
                    f"{parsed.scheme}://{host}:81/capture",
                ]
            )
        # keep order + remove duplicates
        seen = set()
        uniq = []
        for u in out:
            if u not in seen:
                seen.add(u)
                uniq.append(u)
        return uniq

    @staticmethod
    def _required_float(payload: dict, keys: tuple[str, ...], endpoint: str) -> float:
        for key in keys:
            if key in payload:
                try:
                    return float(payload[key])
                except Exception:
                    break
        raise RuntimeError(f"Invalid/missing numeric field {keys} from ESP32 {endpoint} payload")

    @staticmethod
    def _required_bool(payload: dict, keys: tuple[str, ...], endpoint: str) -> bool:
        for key in keys:
            if key in payload:
                value = payload[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value != 0
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                break
        raise RuntimeError(f"Invalid/missing boolean field {keys} from ESP32 {endpoint} payload")

    def _request_json(self, method: str, path: str) -> Optional[dict]:
        url = f"{self.ip}{path}"
        for _ in range(self._request_retries + 1):
            try:
                if method == "GET":
                    response = requests.get(url, timeout=self.timeout_s)
                else:
                    response = requests.request(method, url, timeout=self.timeout_s)
                response.raise_for_status()
                return response.json()
            except Exception as error:
                debug_log("ERROR", f"ESP32 {method} {path} failed: {error}")
                time.sleep(0.05)
        return None
