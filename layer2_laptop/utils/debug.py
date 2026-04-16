"""Shared debug logging helper for TRINETRA runtime."""

from __future__ import annotations


DEBUG = True


def debug_log(section: str, message: str) -> None:
    if DEBUG:
        print(f"[{section}] {message}")
