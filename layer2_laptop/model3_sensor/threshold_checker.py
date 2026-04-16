"""Sensor hazard classification."""

from __future__ import annotations

from typing import Dict


class ThresholdChecker:
    def __init__(self, metal_low: float = 400, metal_high: float = 700, gas_low: float = 400, gas_high: float = 700) -> None:
        self.metal_low = metal_low
        self.metal_high = metal_high
        self.gas_low = gas_low
        self.gas_high = gas_high

    def check(self, metal: float, gas: float) -> str:
        if metal < 0 or gas < 0:
            return "GREEN"
        if metal >= self.metal_high or gas >= self.gas_high:
            return "RED"
        if metal >= self.metal_low or gas >= self.gas_low:
            return "YELLOW"
        return "GREEN"

    def enrich_payload(self, payload: Dict[str, float]) -> Dict[str, float]:
        metal = self._safe_float(payload.get("metal", 0.0))
        gas = self._safe_float(payload.get("gas", 0.0))
        out = dict(payload)
        out["metal"] = metal
        out["gas"] = gas
        out["status"] = self.check(metal, gas)
        return out

    @staticmethod
    def _safe_float(value) -> float:
        try:
            number = float(value)
            if number != number:  # NaN
                return 0.0
            return number
        except Exception:
            return 0.0
