"""Make `python core/<script>.py` work from the project root.

When Python executes a file under `core/`, `sys.path[0]` points at `core/`.
Adding the project root keeps package imports such as `benchmark.*` available
while preserving sibling imports such as `import project_paths`.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
