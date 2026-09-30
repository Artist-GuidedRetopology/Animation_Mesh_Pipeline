"""Naming rules shared by Stage 1 output folders and splits.json keys."""

from __future__ import annotations

import re

ANIMATIONS_SUBDIR = "animations"


def sanitize_name(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "unnamed"
