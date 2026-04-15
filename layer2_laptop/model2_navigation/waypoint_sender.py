"""Waypoint pre-send utility."""

from __future__ import annotations

from typing import List

import requests


class WaypointSender:
    """Optional utility: uploads the full mission path to rover before live control loop."""

    def __init__(self, rover_ip: str, timeout_s: float = 3.0) -> None:
        self.rover_ip = rover_ip
        self.timeout_s = timeout_s

    def push_plan(self, waypoints: List[dict]) -> bool:
        try:
            response = requests.post(f"{self.rover_ip}/mission/path", json={"waypoints": waypoints}, timeout=self.timeout_s)
            return response.status_code < 300
        except Exception:
            return False