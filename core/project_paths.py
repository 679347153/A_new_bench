#!/usr/bin/env python3
from __future__ import annotations

"""Central path helpers for the benchmark workspace.

The project is being migrated from flat root-level data folders to a more
regular `data/` layout.  These helpers prefer the new layout but keep legacy
fallbacks so existing commands continue to work.
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("BENCH_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
RESULTS_ROOT = Path(os.environ.get("BENCH_RESULTS_ROOT", PROJECT_ROOT / "results")).expanduser()

SCENES_ROOT = DATA_ROOT / "scenes"
OBJECT_DATASETS_ROOT = DATA_ROOT / "object_datasets"
OBJECT_IMAGES_ROOT = DATA_ROOT / "object_images"
OBJECT_CATALOG_ROOT = DATA_ROOT / "object_catalog"
OBJECT_SETS_ROOT = DATA_ROOT / "object_sets"
OBJECT_PREVIEWS_ROOT = DATA_ROOT / "object_previews"
ARCHIVES_ROOT = DATA_ROOT / "archives"

HM3D_ROOT = SCENES_ROOT / "hm3d"
LEGACY_OBJECTS_DIR = OBJECT_DATASETS_ROOT / "legacy" / "objects"
LEGACY_OBJECT_IMAGES_DIR = OBJECT_IMAGES_ROOT / "legacy"
YCB_ROOT = OBJECT_DATASETS_ROOT / "ycb-v1.2"
HSSD_ROOT = OBJECT_DATASETS_ROOT / "hssd-hab-v0.2.3"

OBJECT_CATALOG_PATH = OBJECT_CATALOG_ROOT / "object_catalog.json"
OBJECT_TEXT_OVERRIDES_PATH = OBJECT_CATALOG_ROOT / "object_text_overrides.json"
HSSD_SEMANTIC_TEXT_PATH = OBJECT_CATALOG_ROOT / "hssd_semantic_text.jsonl"
MISSING_SEMANTIC_TEXT_PATH = OBJECT_CATALOG_ROOT / "missing_semantic_text.csv"


def _first_existing(candidates: Sequence[Path], default: Optional[Path] = None) -> Path:
    for path in candidates:
        if path.expanduser().exists():
            return path.expanduser()
    return default.expanduser() if default is not None else candidates[0].expanduser()


def resolve_hm3d_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return _first_existing([HM3D_ROOT, PROJECT_ROOT / "hm3d"], default=HM3D_ROOT)


def resolve_results_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return _first_existing([RESULTS_ROOT, PROJECT_ROOT / "results"], default=RESULTS_ROOT)


def resolve_legacy_objects_dir(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return _first_existing([LEGACY_OBJECTS_DIR, PROJECT_ROOT / "objects"], default=LEGACY_OBJECTS_DIR)


def resolve_legacy_images_dir(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return _first_existing([LEGACY_OBJECT_IMAGES_DIR, PROJECT_ROOT / "objects_images"], default=LEGACY_OBJECT_IMAGES_DIR)


def resolve_ycb_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return _first_existing([YCB_ROOT, PROJECT_ROOT / "ycb-v1.2"], default=YCB_ROOT)


def resolve_hssd_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return _first_existing([HSSD_ROOT, PROJECT_ROOT / "hssd-hab-v0.2.3"], default=HSSD_ROOT)


def split_path_list(value: Optional[str | Iterable[str | Path]]) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value)
        if not text:
            return []
        parts = text.split(os.pathsep)
    else:
        parts = [str(x) for x in value]
    return [Path(part).expanduser() for part in parts if str(part).strip()]


def default_object_config_dirs() -> List[Path]:
    candidates = [
        resolve_legacy_objects_dir(),
        resolve_ycb_root() / "configs",
        resolve_hssd_root() / "objects",
        resolve_hssd_root() / "objects" / "decomposed",
    ]
    out: List[Path] = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def default_object_config_dirs_str() -> str:
    return os.pathsep.join(str(path) for path in default_object_config_dirs())


def iter_object_config_dirs(value: Optional[str | Iterable[str | Path]] = None) -> List[Path]:
    """Return existing directories that directly contain object config files.

    `value` may be a single path, an `os.pathsep` separated path list, or an
    iterable of paths.  Directory roots are searched recursively so newly added
    object datasets with nested template layouts can be used without changing
    every script.
    """
    roots = split_path_list(value) if value else default_object_config_dirs()
    out: List[Path] = []
    seen = set()
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.name.endswith(".object_config.json"):
            candidates = [root.parent]
        elif root.is_dir():
            candidates = []
            if any(root.glob("*.object_config.json")):
                candidates.append(root)
            candidates.extend(path.parent for path in root.rglob("*.object_config.json"))
        else:
            candidates = []
        for candidate in candidates:
            try:
                key = str(candidate.resolve())
            except Exception:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
    return out


def find_object_config_path(model_id: str, objects_dir: Optional[str | Iterable[str | Path]] = None) -> Optional[Path]:
    """Find a matching `.object_config.json` in the configured object roots."""
    raw = str(model_id or "").strip()
    if not raw:
        return None
    stem = raw.replace(".object_config.json", "")
    aliases = [raw, stem, f"{stem}.object_config.json"]
    if stem.endswith("_4k"):
        aliases.append(stem[:-3])
        aliases.append(f"{stem[:-3]}.object_config.json")
    else:
        aliases.append(f"{stem}_4k")
        aliases.append(f"{stem}_4k.object_config.json")
    names = []
    for alias in aliases:
        name = Path(alias).name
        if not name.endswith(".object_config.json"):
            name = f"{name}.object_config.json"
        if name not in names:
            names.append(name)

    for config_dir in iter_object_config_dirs(objects_dir):
        for name in names:
            path = config_dir / name
            if path.is_file():
                return path
    return None


def ensure_data_layout_dirs() -> None:
    for path in (
        DATA_ROOT,
        SCENES_ROOT,
        OBJECT_DATASETS_ROOT,
        OBJECT_IMAGES_ROOT,
        OBJECT_CATALOG_ROOT,
        OBJECT_SETS_ROOT,
        OBJECT_PREVIEWS_ROOT,
        ARCHIVES_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def to_project_relative(path: str | Path) -> str:
    p = Path(path).expanduser()
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(p)
