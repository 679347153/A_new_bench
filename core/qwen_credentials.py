#!/usr/bin/env python3
from __future__ import annotations

"""Credential loading helpers that never print API key material."""

import os
import stat
from pathlib import Path


DEFAULT_DASHSCOPE_KEY_FILE = Path("/tmp/a_new_bench_dashscope_api_key")
PERSISTENT_DASHSCOPE_KEY_FILE = Path(__file__).resolve().parent.parent / ".secrets" / "dashscope_api_key"


def load_dashscope_api_key() -> str:
    """Load a DashScope key from the environment or a private file."""
    value = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if value:
        return value

    configured = os.environ.get("DASHSCOPE_API_KEY_FILE", "").strip()
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [PERSISTENT_DASHSCOPE_KEY_FILE, DEFAULT_DASHSCOPE_KEY_FILE]
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return ""
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"DashScope key file must be private (chmod 600): {path} mode={oct(mode)}"
        )
    return path.read_text(encoding="utf-8").strip()
