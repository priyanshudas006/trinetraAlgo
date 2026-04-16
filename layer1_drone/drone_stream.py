"""Drone snapshot provider for TRINETRA."""

from __future__ import annotations

import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency for OCR
    pytesseract = None

if pytesseract is not None:
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files/Tesseract/tesseract.exe"),
    ):
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            break


@dataclass
class DroneTelemetry:
    lat: float
    lon: float
    altitude: float
    pitch: float
    roll: float
    yaw: float


class DroneStream:
    """Fetches a drone snapshot from webcam + overlay OCR (with simulation fallback)."""

    def __init__(
        self,
        url: Optional[str] = None,
        timeout_s: float = 3.0,
        simulation: bool = True,
        video_source: int | str = 0,
        ocr_interval_s: float = 0.6,
        ocr_every_n_frames: int = 10,
    ) -> None:
        self.url = url  # retained for backward compatibility; no longer used as primary source
        self.timeout_s = timeout_s
        self.simulation = simulation
        self.video_source = self._normalize_source(video_source)
        self.ocr_interval_s = max(0.2, ocr_interval_s)
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_ocr_ts = 0.0
        self._last_telemetry: Optional[Dict[str, float]] = None
        self._frame_index = 0
        self.ocr_every_n_frames = max(1, int(ocr_every_n_frames))

    def capture_snapshot(self) -> Dict[str, Any]:
        """Returns {lat, lon, altitude, pitch, roll, yaw, image}."""
        if not self.simulation:
            webcam = self._capture_from_webcam()
            if webcam is not None:
                return webcam
        return self._simulate_snapshot()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_from_webcam(self) -> Optional[Dict[str, Any]]:
        frame = self._read_frame()
        if frame is None:
            return None

        now = time.time()
        self._frame_index += 1
        telemetry = self._last_telemetry
        should_run_ocr = (self._frame_index % self.ocr_every_n_frames) == 0
        if telemetry is None:
            should_run_ocr = True
        if (now - self._last_ocr_ts) >= self.ocr_interval_s:
            should_run_ocr = True
        if should_run_ocr:
            parsed = self._extract_telemetry_from_frame(frame)
            if self._is_valid_telemetry(parsed):
                telemetry = parsed
                self._last_telemetry = parsed
            self._last_ocr_ts = now

        if telemetry is None:
            return None

        return {
            "lat": telemetry["lat"],
            "lon": telemetry["lon"],
            "altitude": telemetry["altitude"],
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0,
            "image": frame,
        }

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_source)
            if not self._cap.isOpened():
                self._cap.release()
                self._cap = None
                return None
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _extract_telemetry_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        if pytesseract is None:
            return self._extract_by_pattern_only(frame)

        h, w = frame.shape[:2]
        roi = frame[int(h * 0.75) : h, 0:w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.-:LATONGlatongmMALT "
        text = pytesseract.image_to_string(bw, config=config)
        parsed = self._parse_overlay_text(text)
        if parsed is not None:
            return parsed

        return self._extract_by_pattern_only(roi)

    def _extract_by_pattern_only(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        # No reliable fallback OCR without an OCR engine; keep last valid telemetry instead.
        return None

    def _parse_overlay_text(self, text: str) -> Optional[Dict[str, float]]:
        cleaned = " ".join(text.replace("\n", " ").split())
        lat = self._extract_value(cleaned, ("lat", "latitude"))
        lon = self._extract_value(cleaned, ("lon", "lng", "longitude"))
        alt = self._extract_value(cleaned, ("alt", "altitude", "h"))
        if lat is None or lon is None or alt is None:
            generic = re.findall(r"-?\d{1,3}\.\d+", cleaned)
            if len(generic) >= 3:
                lat = lat if lat is not None else float(generic[0])
                lon = lon if lon is not None else float(generic[1])
                alt = alt if alt is not None else float(generic[2])
        if lat is None or lon is None or alt is None:
            return None
        parsed = {"lat": lat, "lon": lon, "altitude": alt}
        return parsed if self._is_valid_telemetry(parsed) else None

    @staticmethod
    def _extract_value(text: str, labels: Tuple[str, ...]) -> Optional[float]:
        for label in labels:
            match = re.search(rf"{label}\s*[:=]?\s*(-?\d{{1,3}}\.\d+|-?\d+)", text, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except Exception:
                    continue
        return None

    @staticmethod
    def _normalize_source(source: int | str) -> int | str:
        if isinstance(source, int):
            return source
        s = str(source).strip()
        if s.isdigit():
            return int(s)
        return s

    @staticmethod
    def _is_valid_telemetry(payload: Optional[Dict[str, float]]) -> bool:
        if payload is None:
            return False
        try:
            lat = float(payload["lat"])
            lon = float(payload["lon"])
            alt = float(payload["altitude"])
        except Exception:
            return False
        return (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0) and (-500.0 <= alt <= 15000.0)

    def _simulate_snapshot(self) -> Dict[str, Any]:
        """Builds a deterministic synthetic terrain image for local development."""
        height, width = 1000, 1000
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Base traversable terrain.
        image[:] = (60, 110, 70)

        # Non-traversable regions.
        cv2.rectangle(image, (140, 180), (360, 520), (45, 45, 45), -1)
        cv2.rectangle(image, (590, 120), (870, 350), (45, 45, 45), -1)
        cv2.rectangle(image, (520, 580), (950, 860), (40, 40, 40), -1)

        # Partial terrain strip.
        cv2.rectangle(image, (100, 700), (500, 930), (70, 130, 90), -1)

        return {
            "lat": 29.9000,
            "lon": 78.1000,
            "altitude": 22.0,
            "pitch": 0.0,
            "roll": 0.0,
            "yaw": 0.0,
            "image": image,
        }


def get_latest_frame() -> Optional[np.ndarray]:
    """Compatibility helper for legacy integrations."""
    stream = DroneStream(simulation=True)
    return stream.capture_snapshot().get("image")
