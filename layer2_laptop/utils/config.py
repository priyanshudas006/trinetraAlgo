"""Central configuration values."""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:3000")
BACKEND_HAZARD_URL = os.getenv("BACKEND_HAZARD_URL", f"{BACKEND_BASE_URL}/api/hazards")
STRICT_REAL_DATA = _env_bool("STRICT_REAL_DATA", True)
EMERGENCY_FALLBACK = _env_bool("EMERGENCY_FALLBACK", False)

ROVER_BASE_URL = os.getenv("ROVER_BASE_URL", "http://192.168.1.20")
ROVER_CAMERA_URL = os.getenv("ROVER_CAMERA_URL", "http://192.168.137.189/")
ROVER_TIMEOUT_SECONDS = float(os.getenv("ROVER_TIMEOUT_SECONDS", "2.0"))
ROVER_SIMULATION = _env_bool("ROVER_SIMULATION", False)
MAP_VISUALIZER_ENABLED = _env_bool("MAP_VISUALIZER_ENABLED", True)

DRONE_VIDEO_SOURCE = os.getenv("DRONE_VIDEO_SOURCE", "0")
DRONE_OCR_INTERVAL_SECONDS = float(os.getenv("DRONE_OCR_INTERVAL_SECONDS", "0.6"))
DRONE_OCR_EVERY_N_FRAMES = int(os.getenv("DRONE_OCR_EVERY_N_FRAMES", "10"))
DRONE_SIMULATION = _env_bool("DRONE_SIMULATION", False)
DRONE_SOURCE_FALLBACK = _env_bool("DRONE_SOURCE_FALLBACK", False)
DRONE_DEFAULT_LAT = float(os.getenv("DRONE_DEFAULT_LAT", "29.9000"))
DRONE_DEFAULT_LON = float(os.getenv("DRONE_DEFAULT_LON", "78.1000"))
DRONE_DEFAULT_ALT = float(os.getenv("DRONE_DEFAULT_ALT", "20.0"))
DRONE_AUTODETECT_MAX_SOURCES = int(os.getenv("DRONE_AUTODETECT_MAX_SOURCES", "6"))
DRONE_OSD_KEYWORDS = os.getenv("DRONE_OSD_KEYWORDS", "BATT,HOR,GPS,ALT").strip()
DRONE_BLOCKED_SOURCES = os.getenv("DRONE_BLOCKED_SOURCES", "").strip()
