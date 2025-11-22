"""Application version helper.

This simple module centralizes the app version so other modules can
read it or override it via the `APP_VERSION` environment variable.
"""
from __future__ import annotations

import os

DEFAULT_VERSION = "v1.0.1"

def get_version() -> str:
    """Return the application version.

    Priority: `APP_VERSION` env var (if set) -> `DEFAULT_VERSION`.
    """
    return os.environ.get("APP_VERSION", DEFAULT_VERSION)


__all__ = ["get_version", "DEFAULT_VERSION"]
