"""Keep reference scripts importable after moving core modules into `core/`."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
for path in (PROJECT_ROOT, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
