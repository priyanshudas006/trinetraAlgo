"""Drone snapshot provider for TRINETRA."""

from __future__ import annotations

import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
try:
    from layer2_laptop.utils.debug import debug_log
except Exception:
    def debug_log(_section: str, _message: str) -> None:
        return None

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

LOGGER = logging.getLogger(__name__)


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
        fallback_lat: float = 29.9000,
        fallback_lon: float = 78.1000,
        fallback_altitude: float = 20.0,
        allow_source_fallback: bool = False,
        blocked_sources: Optional[List[int]] = None,
        strict_real_data: bool = True,
        emergency_fallback: bool = False,
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
        self._active_source: Optional[int | str] = None
        self.fallback_lat = float(fallback_lat)
        self.fallback_lon = float(fallback_lon)
        self.fallback_altitude = float(fallback_altitude)
        self.allow_source_fallback = bool(allow_source_fallback)
        self.blocked_sources = set(int(s) for s in (blocked_sources or []))
        self._last_ocr_warn_ts = 0.0
        self.strict_real_data = bool(strict_real_data)
        self.emergency_fallback = bool(emergency_fallback)
        if self.strict_real_data and self.simulation:
            raise RuntimeError("STRICT_REAL_DATA enabled: DRONE_SIMULATION must be false")

    def capture_snapshot(self) -> Dict[str, Any]:
        """Returns {lat, lon, altitude, pitch, roll, yaw, image}."""
        if not self.simulation:
            webcam = self._capture_from_webcam()
            if webcam is not None:
                return webcam
            raise RuntimeError("Drone webcam feed unavailable. Close other camera apps and verify DRONE_VIDEO_SOURCE.")
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
            elif self._last_telemetry is not None and (now - self._last_ocr_warn_ts) > 2.0:
                LOGGER.warning("Drone OCR failed; using last valid telemetry")
                self._last_ocr_warn_ts = now
            self._last_ocr_ts = now

        if telemetry is None:
            if self.emergency_fallback:
                debug_log("ERROR", "Drone OCR telemetry missing; using emergency fallback coordinates")
                telemetry = {
                    "lat": self.fallback_lat,
                    "lon": self.fallback_lon,
                    "altitude": self.fallback_altitude,
                }
            else:
                raise RuntimeError("Drone telemetry missing from OCR")

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
            self._cap, self._active_source = self._open_capture_with_fallback(self.video_source)
            if self._cap is None:
                return None
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Device might have glitched or switched ownership; reopen once.
            self.close()
            self._cap, self._active_source = self._open_capture_with_fallback(self.video_source)
            if self._cap is None:
                return None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
        return frame

    def _open_capture_with_fallback(self, source: int | str) -> tuple[Optional[cv2.VideoCapture], Optional[int | str]]:
        candidates: list[int | str] = [source]
        if self.allow_source_fallback and isinstance(source, int):
            candidates.extend([max(0, source + 1), max(0, source + 2)])

        for candidate in candidates:
            if isinstance(candidate, int) and candidate in self.blocked_sources:
                continue
            cap = self._open_capture(candidate)
            if cap is not None:
                return cap, candidate
        return None, None

    def get_active_source(self) -> Optional[int | str]:
        return self._active_source

    def set_video_source(self, source: int | str) -> None:
        self.video_source = self._normalize_source(source)
        self.close()

    def get_configured_source(self) -> int | str:
        return self.video_source

    def auto_select_source(self, max_sources: int = 6, osd_keywords: Optional[List[str]] = None) -> tuple[bool, str]:
        keywords = [k.strip().upper() for k in (osd_keywords or ["BATT", "HOR", "GPS", "ALT"]) if k.strip()]
        best_source: Optional[int] = None
        best_score: float = -1.0

        for idx in range(max(1, int(max_sources))):
            if idx in self.blocked_sources:
                continue
            cap = self._open_capture(idx)
            if cap is None:
                continue
            frames = []
            for _ in range(4):
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
            cap.release()
            if not frames:
                continue

            score = self._score_source(frames, keywords)
            if score > best_score:
                best_score = score
                best_source = idx

        if best_source is None:
            return False, "No camera source could be opened"

        self.set_video_source(best_source)
        return True, f"Auto-selected drone source index: {best_source} (score={best_score:.2f})"

    def _score_source(self, frames: List[np.ndarray], keywords: List[str]) -> float:
        score = 0.0
        if len(frames) >= 2:
            g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
            motion = float(np.mean(cv2.absdiff(g0, g1)))
            score += min(25.0, motion / 2.0)

        if pytesseract is not None and keywords:
            h, w = frames[-1].shape[:2]
            roi = frames[-1][int(h * 0.55) : h, 0:w]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                bw, config="--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:<-> "
            ).upper()
            hits = sum(1 for k in keywords if k in text)
            score += hits * 20.0

        # Penalize likely OBS "camera off" splash (dominant blue + static icon)
        hsv = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, np.array([95, 40, 30], dtype=np.uint8), np.array([135, 255, 255], dtype=np.uint8))
        blue_ratio = float(np.count_nonzero(blue)) / float(blue.size)
        if blue_ratio > 0.55:
            score -= 12.0

        return score

    @staticmethod
    def _open_capture(source: int | str) -> Optional[cv2.VideoCapture]:
        cap = None
        if isinstance(source, int):
            # CAP_DSHOW is more reliable on Windows webcam devices.
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        return cap

    def _extract_telemetry_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        if pytesseract is None:
            return self._extract_by_pattern_only(frame)

        h, w = frame.shape[:2]
        rois = [
            frame[0 : int(h * 0.42), int(w * 0.5) : w],   # top-right OSD
            frame[0 : int(h * 0.35), 0:w],                # top full strip
            frame[int(h * 0.7) : h, 0:w],                 # bottom telemetry strip
        ]
        config = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789.-:LATLONGlatlongALTMH"
        for roi in rois:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(bw, config=config)
            parsed = self._parse_overlay_text(text)
            if parsed is not None:
                return parsed
        return self._extract_by_pattern_only(frame)

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
