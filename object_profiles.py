#!/usr/bin/env python3
from __future__ import annotations

"""
Object placement profile helpers.

The automatic placement pipeline needs approximate object geometry before the
object is instantiated in Habitat-Sim.  This module centralizes those estimates
and supports an optional project-level override file:

  object_profiles.json

Expected shape:

  {
    "alarm_clock_01_4k": {
      "radius": 0.16,
      "y_offset": 0.08,
      "footprint_x": 0.22,
      "footprint_z": 0.12,
      "height": 0.18,
      "placement_class": "small_tabletop"
    }
  }

When the file or an entry is missing, callers receive a deterministic keyword
fallback and a `profile_source` field that explains the fallback.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_OBJECT_PROFILE: Dict[str, Any] = {
    "radius": 0.35,
    "y_offset": 0.05,
    "footprint_x": 0.70,
    "footprint_z": 0.70,
    "height": 0.30,
    "placement_class": "tabletop_or_floor",
    "profile_source": "default_keyword_fallback",
}


PROFILE_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "wheelchair": {"radius": 0.70, "y_offset": 0.05, "footprint_x": 0.80, "footprint_z": 1.10, "height": 1.10, "placement_class": "floor_only"},
    "console": {"radius": 0.75, "y_offset": 0.05, "footprint_x": 1.20, "footprint_z": 0.45, "height": 0.80, "placement_class": "floor_only"},
    "table": {"radius": 0.60, "y_offset": 0.05, "footprint_x": 0.90, "footprint_z": 0.90, "height": 0.75, "placement_class": "floor_only"},
    "chair": {"radius": 0.40, "y_offset": 0.05, "footprint_x": 0.55, "footprint_z": 0.55, "height": 0.90, "placement_class": "floor_only"},
    "stool": {"radius": 0.35, "y_offset": 0.05, "footprint_x": 0.45, "footprint_z": 0.45, "height": 0.55, "placement_class": "floor_only"},
    "plant": {"radius": 0.35, "y_offset": 0.08, "footprint_x": 0.40, "footprint_z": 0.40, "height": 0.85, "placement_class": "floor_or_large_surface"},
    "statue": {"radius": 0.32, "y_offset": 0.08, "footprint_x": 0.35, "footprint_z": 0.35, "height": 0.55, "placement_class": "large_tabletop"},
    "vase": {"radius": 0.20, "y_offset": 0.10, "footprint_x": 0.22, "footprint_z": 0.22, "height": 0.38, "placement_class": "small_tabletop"},
    "pot": {"radius": 0.22, "y_offset": 0.08, "footprint_x": 0.25, "footprint_z": 0.25, "height": 0.25, "placement_class": "small_tabletop"},
    "bottle": {"radius": 0.14, "y_offset": 0.08, "footprint_x": 0.12, "footprint_z": 0.12, "height": 0.32, "placement_class": "small_tabletop"},
    "clock": {"radius": 0.16, "y_offset": 0.10, "footprint_x": 0.22, "footprint_z": 0.12, "height": 0.20, "placement_class": "small_tabletop"},
    "camera": {"radius": 0.18, "y_offset": 0.08, "footprint_x": 0.22, "footprint_z": 0.16, "height": 0.16, "placement_class": "small_tabletop"},
    "cake": {"radius": 0.20, "y_offset": 0.04, "footprint_x": 0.30, "footprint_z": 0.25, "height": 0.12, "placement_class": "small_tabletop"},
    "food": {"radius": 0.14, "y_offset": 0.04, "footprint_x": 0.18, "footprint_z": 0.18, "height": 0.12, "placement_class": "small_tabletop"},
    "apple": {"radius": 0.08, "y_offset": 0.05, "footprint_x": 0.10, "footprint_z": 0.10, "height": 0.10, "placement_class": "small_tabletop"},
    "pear": {"radius": 0.08, "y_offset": 0.05, "footprint_x": 0.10, "footprint_z": 0.10, "height": 0.12, "placement_class": "small_tabletop"},
    "chess": {"radius": 0.28, "y_offset": 0.04, "footprint_x": 0.45, "footprint_z": 0.45, "height": 0.08, "placement_class": "large_tabletop"},
    "tea": {"radius": 0.24, "y_offset": 0.04, "footprint_x": 0.40, "footprint_z": 0.30, "height": 0.16, "placement_class": "large_tabletop"},
    "pillow": {"radius": 0.28, "y_offset": 0.05, "footprint_x": 0.45, "footprint_z": 0.45, "height": 0.16, "placement_class": "soft_surface"},
}


def _aliases(model_id: str) -> list[str]:
    raw = str(model_id or "").strip()
    if not raw:
        return []
    stem = raw.replace(".object_config.json", "")
    aliases = [raw, stem]
    if stem.endswith("_4k"):
        aliases.append(stem[:-3])
    else:
        aliases.append(f"{stem}_4k")
    return list(dict.fromkeys([x for x in aliases if x]))


def _load_profile_overrides(objects_dir: str = "./objects", profile_path: Optional[str] = None) -> Dict[str, Any]:
    candidates = []
    if profile_path:
        candidates.append(Path(profile_path))
    candidates.append(Path("object_profiles.json"))
    candidates.append(Path(objects_dir).expanduser().parent / "object_profiles.json")
    candidates.append(Path(objects_dir).expanduser() / "object_profiles.json")

    for path in candidates:
        try:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
        except Exception:
            continue
    return {}


def _read_object_config(objects_dir: str, model_id: str) -> Dict[str, Any]:
    root = Path(objects_dir).expanduser()
    for alias in _aliases(model_id):
        name = alias if alias.endswith(".object_config.json") else f"{alias}.object_config.json"
        path = root / Path(name).name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


def _keyword_profile(model_id: str) -> Dict[str, Any]:
    key = str(model_id or "").lower()
    for keyword, profile in PROFILE_KEYWORDS.items():
        if keyword in key:
            out = dict(DEFAULT_OBJECT_PROFILE)
            out.update(profile)
            out["profile_source"] = f"keyword:{keyword}"
            return out
    return dict(DEFAULT_OBJECT_PROFILE)


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(DEFAULT_OBJECT_PROFILE)
    out.update(profile)
    radius = max(0.02, float(out.get("radius", DEFAULT_OBJECT_PROFILE["radius"])))
    fx = max(0.02, float(out.get("footprint_x", radius * 2.0)))
    fz = max(0.02, float(out.get("footprint_z", radius * 2.0)))
    out["radius"] = round(float(max(radius, math.sqrt(fx * fx + fz * fz) * 0.35)), 4)
    out["y_offset"] = round(max(0.0, float(out.get("y_offset", DEFAULT_OBJECT_PROFILE["y_offset"]))), 4)
    out["footprint_x"] = round(fx, 4)
    out["footprint_z"] = round(fz, 4)
    out["height"] = round(max(0.01, float(out.get("height", DEFAULT_OBJECT_PROFILE["height"]))), 4)
    out["placement_class"] = str(out.get("placement_class", "tabletop_or_floor"))
    out["profile_source"] = str(out.get("profile_source", "unknown"))
    return out


def get_object_profile(model_id: str, objects_dir: str = "./objects", profile_path: Optional[str] = None) -> Dict[str, Any]:
    profile = _keyword_profile(model_id)
    overrides = _load_profile_overrides(objects_dir=objects_dir, profile_path=profile_path)
    for alias in _aliases(model_id):
        override = overrides.get(alias)
        if isinstance(override, dict):
            profile.update(override)
            profile["profile_source"] = f"object_profiles.json:{alias}"
            break

    config = _read_object_config(objects_dir, model_id)
    if config:
        if "is_collidable" in config:
            profile["is_collidable"] = bool(config.get("is_collidable"))
        if "margin" in config:
            try:
                profile["collision_margin"] = float(config.get("margin"))
            except Exception:
                pass
        if "scale" in config and isinstance(config.get("scale"), list):
            profile["template_scale"] = config.get("scale")
    else:
        profile["missing_template_config"] = True

    return _normalize_profile(profile)


def is_floor_like_category(category: Any) -> bool:
    text = str(category or "").lower()
    return any(key in text for key in ("floor", "ground", "carpet", "rug", "room_floor"))


def _category_has_any(category: Any, keywords: Tuple[str, ...]) -> bool:
    text = str(category or "").lower()
    return any(key in text for key in keywords)


TABLETOP_CATEGORIES = (
    "table",
    "desk",
    "counter",
    "countertop",
    "shelf",
    "cabinet",
    "dresser",
    "nightstand",
    "stand",
    "stool",
    "bench",
)
LARGE_SURFACE_CATEGORIES = (
    "table",
    "desk",
    "counter",
    "countertop",
    "cabinet",
    "dresser",
    "bench",
    "floor",
    "ground",
    "room_floor",
)
SOFT_SURFACE_CATEGORIES = (
    "bed",
    "sofa",
    "couch",
    "chair",
    "armchair",
    "pillow",
    "floor",
    "carpet",
    "rug",
    "room_floor",
)


def surface_affordance_score(
    model_id: str,
    category: Any,
    profile: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, float, str]:
    """Return whether a support category is semantically valid for an object.

    The geometric filters answer "can it fit"; this helper answers "does this
    surface make sense".  The score delta is intentionally small enough to
    preserve LLM/geometry signals, but large enough to break ties away from
    implausible surfaces.
    """
    profile = profile or get_object_profile(model_id)
    placement_class = str(profile.get("placement_class", "tabletop_or_floor"))
    category_text = str(category or "").lower()
    floor_like = is_floor_like_category(category_text)

    if placement_class == "floor_only":
        if floor_like:
            return True, 0.35, "floor_only_object_on_floor"
        return False, -0.60, "floor_only_object_on_non_floor"

    if placement_class == "soft_surface":
        if _category_has_any(category_text, SOFT_SURFACE_CATEGORIES):
            return True, 0.28, "soft_object_on_soft_or_floor_surface"
        return False, -0.45, "soft_object_on_hard_surface"

    if placement_class == "small_tabletop":
        if _category_has_any(category_text, TABLETOP_CATEGORIES):
            return True, 0.28, "small_tabletop_object_on_tabletop_surface"
        if floor_like:
            return True, -0.18, "small_tabletop_object_on_floor_fallback"
        if _category_has_any(category_text, ("bed", "sofa", "couch", "chair")):
            return True, -0.10, "small_tabletop_object_on_soft_fallback"
        return True, -0.06, "small_tabletop_object_on_unknown_surface"

    if placement_class == "large_tabletop":
        if _category_has_any(category_text, LARGE_SURFACE_CATEGORIES):
            return True, 0.24, "large_tabletop_object_on_large_surface"
        if _category_has_any(category_text, ("shelf", "nightstand", "stand")):
            return True, -0.12, "large_tabletop_object_on_limited_surface"
        if _category_has_any(category_text, ("bed", "sofa", "couch", "chair")):
            return True, -0.20, "large_tabletop_object_on_soft_fallback"
        return True, -0.08, "large_tabletop_object_on_unknown_surface"

    if placement_class == "floor_or_large_surface":
        if floor_like:
            return True, 0.30, "floor_or_large_object_on_floor"
        if _category_has_any(category_text, LARGE_SURFACE_CATEGORIES):
            return True, 0.10, "floor_or_large_object_on_large_surface"
        return True, -0.22, "floor_or_large_object_on_small_surface"

    if _category_has_any(category_text, TABLETOP_CATEGORIES):
        return True, 0.16, "generic_object_on_tabletop_surface"
    if floor_like:
        return True, -0.06, "generic_object_on_floor_fallback"
    return True, 0.0, "generic_object_neutral_surface"


def surface_requirement(profile: Dict[str, Any], safety_factor: float = 1.15) -> Dict[str, float]:
    fx = float(profile.get("footprint_x", float(profile.get("radius", 0.2)) * 2.0))
    fz = float(profile.get("footprint_z", float(profile.get("radius", 0.2)) * 2.0))
    radius = float(profile.get("radius", max(fx, fz) / 2.0))
    return {
        "required_span_x": round(max(0.04, fx * safety_factor), 4),
        "required_span_z": round(max(0.04, fz * safety_factor), 4),
        "required_min_span": round(max(0.04, min(fx, fz) * safety_factor), 4),
        "required_area": round(max(0.0025, fx * fz * safety_factor * safety_factor), 4),
        "edge_margin": round(max(0.03, radius * 0.85), 4),
    }
