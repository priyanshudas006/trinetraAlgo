"""Sensor ingest endpoint for ESP32 push mode."""

from __future__ import annotations

from flask import Flask, request

try:
    from .backend_poster import BackendPoster
    from .threshold_checker import ThresholdChecker
except ImportError:
    from backend_poster import BackendPoster
    from threshold_checker import ThresholdChecker

app = Flask(__name__)
latest = {}
checker = ThresholdChecker()
poster = BackendPoster()


@app.route("/sensor", methods=["POST"])
def sensor():
    global latest
    payload = request.get_json(silent=True) or {}
    latest = checker.enrich_payload(payload)
    if "lat" in latest and "lon" in latest and latest.get("status") in ("YELLOW", "RED"):
        poster.post(latest)
    return {"status": "ok", "latest": latest.get("status", "UNKNOWN")}


@app.route("/sensor/latest", methods=["GET"])
def sensor_latest():
    return latest or {"status": "empty"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
