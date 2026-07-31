#!/usr/bin/env python3
from __future__ import annotations

"""Prepare the regular project data directory layout.

By default this script performs moves inside the workspace.  Use `--dry-run` to
print the planned operations without changing files.
"""

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from project_paths import (
    ARCHIVES_ROOT,
    DATA_ROOT,
    HSSD_ROOT,
    HM3D_ROOT,
    LEGACY_OBJECT_IMAGES_DIR,
    LEGACY_OBJECTS_DIR,
    OBJECT_DATASETS_ROOT,
    OBJECT_IMAGES_ROOT,
    PROJECT_ROOT,
    SCENES_ROOT,
    YCB_ROOT,
    ensure_data_layout_dirs,
)


Move = Tuple[Path, Path, str]


def _within_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT.resolve())
        return True
    except Exception:
        return False


def _planned_moves() -> Iterable[Move]:
    yield PROJECT_ROOT / "hm3d", HM3D_ROOT, "HM3D scenes"
    yield PROJECT_ROOT / "objects", LEGACY_OBJECTS_DIR, "legacy object templates"
    yield PROJECT_ROOT / "objects_images", LEGACY_OBJECT_IMAGES_DIR, "legacy object images"
    yield PROJECT_ROOT / "ycb-v1.2", YCB_ROOT, "YCB object dataset"
    yield PROJECT_ROOT / "hssd-hab-v0.2.3", HSSD_ROOT, "HSSD object dataset"
    for archive in sorted(PROJECT_ROOT.glob("*.zip")):
        yield archive, ARCHIVES_ROOT / archive.name, "root archive"


def _move_path(src: Path, dst: Path, description: str, dry_run: bool) -> str:
    if not src.exists():
        return f"[Skip] {description}: source not found: {src}"
    if dst.exists():
        return f"[Skip] {description}: target already exists: {dst}"
    if not _within_root(src) or not _within_root(dst):
        raise RuntimeError(f"Refusing to move outside workspace: {src} -> {dst}")
    message = f"[Move] {description}: {src} -> {dst}"
    if dry_run:
        return message
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return message


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare regular data/ layout for this benchmark project.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned moves")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[DryRun] Would ensure data layout under: {DATA_ROOT}")
    else:
        ensure_data_layout_dirs()
        SCENES_ROOT.mkdir(parents=True, exist_ok=True)
        OBJECT_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
        OBJECT_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    for src, dst, description in _planned_moves():
        print(_move_path(src, dst, description, bool(args.dry_run)))

    print("[OK] Project structure preparation complete" if not args.dry_run else "[OK] Dry run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
