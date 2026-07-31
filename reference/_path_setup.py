"""Path bootstrap for reference scripts executed as files."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
for path in (PROJECT_ROOT, CORE_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
