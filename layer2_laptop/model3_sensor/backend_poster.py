"""Hazard backend client with validation + retry."""

from __future__ import annotations

import os
import time
from typing import Dict

import requests


class BackendPoster:
    def __init__(self, url: str = "", timeout_s: float = 3.0, retries: int = 3) -> None:
        self.url = url or os.getenv("BACKEND_HAZARD_URL", "http://localhost:3000/api/hazards")
        self.timeout_s = timeout_s
        self.retries = retries

    def post(self, data: Dict) -> bool:
        if not self._is_valid(data):
            return False

        for attempt in range(self.retries):
            try:
                response = requests.post(self.url, json=data, timeout=self.timeout_s)
                if response.status_code < 300:
                    return True
            except Exception:
                pass
            time.sleep(0.2 * (attempt + 1))
        return False

    def _is_valid(self, payload: Dict) -> bool:
        required = ("lat", "lon", "status")
        if not all(key in payload for key in required):
            return False
        if payload["status"] not in ("GREEN", "YELLOW", "RED"):
            return False
        try:
            float(payload["lat"])
            float(payload["lon"])
        except Exception:
            return False
        return True
