#!/usr/bin/env python3
"""Utilities for resolving HM3D scenes across val/minival splits.

The workspace now contains two HM3D splits under ./hm3d:
- ./hm3d/minival
- ./hm3d/val

Rules:
- Prefer val when the same scene exists in both splits.
- Only expose scenes that have semantic.txt by default.
- Keep split-specific dataset config files aligned with the data split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent
HM3D_ROOT = WORKSPACE_ROOT / "hm3d"
MINIVAL_ROOT = HM3D_ROOT / "minival"
VAL_ROOT = HM3D_ROOT / "val"

MINIVAL_CONFIG = HM3D_ROOT / "hm3d_annotated_basis.scene_dataset_config.json"
VAL_CONFIG = VAL_ROOT / "hm3d_annotated_val_basis.scene_dataset_config.json"

SPLIT_PRIORITY = ("val", "minival")


@dataclass(frozen=True)
class ScenePaths:
    scene_name: str
    scene_id: str
    split: str
    split_root: Path
    scene_dir: Path
    stage_glb: Path
    semantic_glb: Path
    semantic_txt: Path
    navmesh: Path
    scene_instance_json: Path
    dataset_config: Path


def scene_id_from_name(scene_name: str) -> str:
    return scene_name.split("-", 1)[1] if "-" in scene_name else scene_name


def split_root(split: str, root: Optional[Path] = None) -> Path:
    base = root if root is not None else HM3D_ROOT
    if split == "val":
        return base / "val"
    if split == "minival":
        return base / "minival"
    raise ValueError(f"Unknown HM3D split: {split}")


def dataset_config_for_split(split: str, root: Optional[Path] = None) -> Path:
    base = root if root is not None else HM3D_ROOT
    if split == "val":
        return base / "val" / "hm3d_annotated_val_basis.scene_dataset_config.json"
    if split == "minival":
        return base / "hm3d_annotated_basis.scene_dataset_config.json"
    raise ValueError(f"Unknown HM3D split: {split}")


def _candidate_scene_paths(scene_name: str, split: str, root: Optional[Path] = None) -> ScenePaths:
    split_root_dir = split_root(split, root)
    scene_dir = split_root_dir / scene_name
    scene_id = scene_id_from_name(scene_name)
    return ScenePaths(
        scene_name=scene_name,
        scene_id=scene_id,
        split=split,
        split_root=split_root_dir,
        scene_dir=scene_dir,
        stage_glb=scene_dir / f"{scene_id}.basis.glb",
        semantic_glb=scene_dir / f"{scene_id}.semantic.glb",
        semantic_txt=scene_dir / f"{scene_id}.semantic.txt",
        navmesh=scene_dir / f"{scene_id}.basis.navmesh",
        scene_instance_json=scene_dir / f"{scene_id}.basis.scene_instance.json",
        dataset_config=dataset_config_for_split(split, root),
    )


def resolve_scene_paths(scene_name: str, require_semantic: bool = True, root: Optional[Path] = None) -> Optional[ScenePaths]:
    """Resolve a scene name to actual files, preferring val over minival."""
    for split in SPLIT_PRIORITY:
        candidate = _candidate_scene_paths(scene_name, split, root=root)
        if not candidate.scene_dir.is_dir():
            continue
        if not candidate.stage_glb.is_file():
            continue
        if require_semantic and not candidate.semantic_txt.is_file():
            continue
        if not candidate.dataset_config.is_file():
            continue
        return candidate
    return None


def iter_available_scenes(require_semantic: bool = True, root: Optional[Path] = None) -> List[ScenePaths]:
    """Return merged scenes across val/minival, de-duplicated with val priority."""
    results: List[ScenePaths] = []
    seen = set()
    for split in SPLIT_PRIORITY:
        split_root_dir = split_root(split, root)
        if not split_root_dir.is_dir():
            continue
        for scene_dir in sorted([p for p in split_root_dir.iterdir() if p.is_dir()]):
            scene_name = scene_dir.name
            if scene_name in seen:
                continue
            candidate = _candidate_scene_paths(scene_name, split, root=root)
            if not candidate.stage_glb.is_file():
                continue
            if require_semantic and not candidate.semantic_txt.is_file():
                continue
            if not candidate.dataset_config.is_file():
                continue
            seen.add(scene_name)
            results.append(candidate)
    return results


def list_available_scenes(require_semantic: bool = True, root: Optional[Path] = None) -> List[str]:
    return [item.scene_name for item in iter_available_scenes(require_semantic=require_semantic, root=root)]


def scene_exists(scene_name: str, require_semantic: bool = True, root: Optional[Path] = None) -> bool:
    return resolve_scene_paths(scene_name, require_semantic=require_semantic, root=root) is not None
