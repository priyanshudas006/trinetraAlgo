"""Sensor ingest endpoint for ESP32 push mode."""

from __future__ import annotations

from flask import Flask, request

try:
    from .backend_poster import BackendPoster
    from .threshold_checker import ThresholdChecker
    from ..utils.debug import debug_log
except ImportError:
    from backend_poster import BackendPoster
    from threshold_checker import ThresholdChecker
    try:
        from layer2_laptop.utils.debug import debug_log
    except ImportError:
        from utils.debug import debug_log

app = Flask(__name__)
latest = {}
checker = ThresholdChecker()
poster = BackendPoster()


@app.route("/sensor", methods=["POST"])
def sensor():
    global latest
    try:
        payload = request.get_json(silent=True) or {}
        latest = checker.enrich_payload(payload)
        debug_log("SENSOR", f"Gas={latest.get('gas', 0.0)}, Metal={latest.get('metal', 0.0)}")
        if "lat" in latest and "lon" in latest and latest.get("status") in ("YELLOW", "RED"):
            hazard_type = _hazard_type(float(latest.get("metal", 0.0)), float(latest.get("gas", 0.0)))
            debug_log(
                "HAZARD",
                (
                    f"{hazard_type} detected at "
                    f"({float(latest['lat']):.6f},{float(latest['lon']):.6f}) -> {latest.get('status')}"
                ),
            )
            poster.post(latest)
        return {"status": "ok", "latest": latest.get("status", "UNKNOWN")}
    except Exception as e:
        debug_log("ERROR", str(e))
        return {"status": "error", "message": "sensor processing failed"}, 500


@app.route("/sensor/latest", methods=["GET"])
def sensor_latest():
    return latest or {"status": "empty"}


def _hazard_type(metal: float, gas: float) -> str:
    metal_alert = metal >= checker.metal_low
    gas_alert = gas >= checker.gas_low
    if metal_alert and gas_alert:
        return "COMBINED"
    if gas_alert:
        return "GAS"
    if metal_alert:
        return "METAL"
    return "UNKNOWN"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
