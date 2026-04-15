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

ROVER_BASE_URL = os.getenv("ROVER_BASE_URL", "http://192.168.1.20")
ROVER_CAMERA_URL = os.getenv("ROVER_CAMERA_URL", f"{ROVER_BASE_URL}/camera")
ROVER_TIMEOUT_SECONDS = float(os.getenv("ROVER_TIMEOUT_SECONDS", "2.0"))
ROVER_SIMULATION = _env_bool("ROVER_SIMULATION", True)

DRONE_VIDEO_SOURCE = os.getenv("DRONE_VIDEO_SOURCE", "0")
DRONE_OCR_INTERVAL_SECONDS = float(os.getenv("DRONE_OCR_INTERVAL_SECONDS", "0.6"))
DRONE_OCR_EVERY_N_FRAMES = int(os.getenv("DRONE_OCR_EVERY_N_FRAMES", "10"))
DRONE_SIMULATION = _env_bool("DRONE_SIMULATION", True)
