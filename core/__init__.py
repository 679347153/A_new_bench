"""Core executable modules for the dynamic household layout benchmark.

Most files in this directory are still executable scripts and historically use
sibling imports such as `from project_paths import ...`.  When imported as a
package (`import core.hm3d_paths`) those sibling modules are not on `sys.path`
by default, so we add the `core/` directory here.  `core/sitecustomize.py`
handles the symmetric case for `python core/<script>.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
