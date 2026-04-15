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
        if metal >= self.metal_high or gas >= self.gas_high:
            return "RED"
        if metal >= self.metal_low or gas >= self.gas_low:
            return "YELLOW"
        return "GREEN"

    def enrich_payload(self, payload: Dict[str, float]) -> Dict[str, float]:
        metal = float(payload.get("metal", 0.0))
        gas = float(payload.get("gas", 0.0))
        out = dict(payload)
        out["status"] = self.check(metal, gas)
        return out