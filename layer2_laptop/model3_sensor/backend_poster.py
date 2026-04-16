"""Local hazard logger with validation + retry-safe writes."""

from __future__ import annotations

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict


class BackendPoster:
    def __init__(self, url: str = "", timeout_s: float = 3.0, retries: int = 3) -> None:
        # `url` kept for compatibility; now treated as local output path.
        default_path = os.getenv("LOCAL_HAZARD_LOG", "TRINETRA/logs/hazards.jsonl")
        self.url = url or default_path
        self.timeout_s = timeout_s
        self.retries = retries
        self._lock = threading.Lock()

    def post(self, data: Dict) -> bool:
        if not self._is_valid(data):
            return False

        for attempt in range(self.retries):
            try:
                if self._append_local(data):
                    return True
            except Exception:
                pass
            time.sleep(0.2 * (attempt + 1))
        return False

    def _append_local(self, payload: Dict) -> bool:
        path = Path(self.url)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return True

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
