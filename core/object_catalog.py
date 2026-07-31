#!/usr/bin/env python3
from __future__ import annotations

"""Object catalog helpers.

The old pipeline treated `objects_images/*.webp` as the object list.  This
module decouples object enumeration from images so datasets such as YCB and
HSSD can participate using semantic text.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from project_paths import (
    HSSD_SEMANTIC_TEXT_PATH,
    MISSING_SEMANTIC_TEXT_PATH,
    OBJECT_CATALOG_PATH,
    OBJECT_TEXT_OVERRIDES_PATH,
    resolve_hssd_root,
    resolve_legacy_images_dir,
    resolve_legacy_objects_dir,
    resolve_ycb_root,
)


IMAGE_EXTENSIONS = (".webp", ".jpg", ".jpeg", ".png", ".bmp")


def object_name_to_text(name: str) -> str:
    stem = Path(str(name)).stem
    stem = re.sub(r"\.object_config$", "", stem)
    stem = re.sub(r"^[0-9]+[-_a-z]*_", "", stem)
    stem = stem.replace("_4k", "")
    words = [w for w in re.split(r"[_\-\s]+", stem) if w]
    return " ".join(words) if words else stem


def default_semantic_text(display_name: str, dataset: str = "object") -> str:
    name = display_name.strip() or "object"
    return (
        f"{name}. A household object from the {dataset} dataset. "
        "Choose plausible rooms and support surfaces based on the object category, "
        "typical household usage, size, and whether it is usually placed on floors, tables, shelves, counters, or soft furniture."
    )


def safe_output_name(text: str) -> str:
    value = str(text).strip().replace(".object_config.json", "")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "object"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_text_overrides(path: Path = OBJECT_TEXT_OVERRIDES_PATH) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            out[str(key)] = {"semantic_text": value, "semantic_source": "override"}
        elif isinstance(value, dict):
            out[str(key)] = dict(value)
    return out


def load_hssd_semantic_text(path: Path = HSSD_SEMANTIC_TEXT_PATH) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        key = str(item.get("object_key") or item.get("model_id") or item.get("object_name") or "").strip()
        if key:
            out[key] = item
    return out


def _apply_text_metadata(entry: Dict[str, Any], sources: Sequence[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    keys = [
        str(entry.get("object_key", "")),
        str(entry.get("object_name", "")),
        str(entry.get("model_id", "")),
    ]
    for source in sources:
        for key in keys:
            if key and key in source:
                meta = source[key]
                out = dict(entry)
                out.update({k: v for k, v in meta.items() if v not in (None, "")})
                out.setdefault("semantic_source", meta.get("semantic_source", "metadata"))
                return out
    return entry


def _read_object_config(path: Path) -> Dict[str, Any]:
    return _load_json(path)


def legacy_image_entries(images_dir: Optional[Path] = None, objects_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    image_root = images_dir or resolve_legacy_images_dir()
    object_root = objects_dir or resolve_legacy_objects_dir()
    entries: List[Dict[str, Any]] = []
    if not image_root.is_dir():
        return entries
    for image_path in sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        stem = image_path.stem
        config_path = object_root / f"{stem}.object_config.json"
        model_id = stem
        if not config_path.is_file() and (object_root / f"{stem}_4k.object_config.json").is_file():
            config_path = object_root / f"{stem}_4k.object_config.json"
            model_id = f"{stem}_4k"
        display = object_name_to_text(stem)
        entries.append(
            {
                "object_key": f"legacy:{stem}",
                "object_name": stem,
                "dataset": "legacy",
                "model_id": model_id,
                "display_name": display,
                "image_path": str(image_path),
                "template_config_path": str(config_path) if config_path.is_file() else "",
                "semantic_text": default_semantic_text(display, "legacy"),
                "semantic_source": "image_filename",
            }
        )
    return entries


def ycb_entries(ycb_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = ycb_root or resolve_ycb_root()
    config_root = root / "configs"
    entries: List[Dict[str, Any]] = []
    if not config_root.is_dir():
        return entries
    for config_path in sorted(config_root.glob("*.object_config.json")):
        model_id = config_path.name[: -len(".object_config.json")]
        cfg = _read_object_config(config_path)
        display = object_name_to_text(model_id)
        entries.append(
            {
                "object_key": f"ycb:{model_id}",
                "object_name": model_id,
                "dataset": "ycb",
                "model_id": model_id,
                "display_name": display,
                "image_path": "",
                "template_config_path": str(config_path),
                "render_asset": cfg.get("render_asset", ""),
                "collision_asset": cfg.get("collision_asset", ""),
                "semantic_text": default_semantic_text(display, "YCB"),
                "semantic_source": "filename_rule",
            }
        )
    return entries


def hssd_entries(
    hssd_root: Optional[Path] = None,
    *,
    text_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    generated_text: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    root = hssd_root or resolve_hssd_root()
    object_root = root / "objects"
    entries: List[Dict[str, Any]] = []
    if not object_root.is_dir():
        return entries
    text_sources = [text_overrides or load_text_overrides(), generated_text or load_hssd_semantic_text()]
    for config_path in sorted(object_root.rglob("*.object_config.json")):
        if ".cache" in config_path.parts:
            continue
        model_id = config_path.name[: -len(".object_config.json")]
        cfg = _read_object_config(config_path)
        semantic_id = cfg.get("semantic_id")
        base = {
            "object_key": f"hssd:{model_id}",
            "object_name": model_id,
            "dataset": "hssd",
            "model_id": model_id,
            "display_name": f"HSSD object {model_id[:8]}",
            "image_path": "",
            "template_config_path": str(config_path),
            "render_asset": cfg.get("render_asset", ""),
            "semantic_id": semantic_id,
            "semantic_text": "",
            "semantic_source": "missing",
        }
        entry = _apply_text_metadata(base, text_sources)
        if not entry.get("semantic_text"):
            entry["semantic_text"] = (
                f"An HSSD household object with asset id {model_id}. "
                f"semantic_id={semantic_id if semantic_id is not None else 'unknown'}. "
                "Semantic category text is missing; generate or provide a description before using LLM room recommendation."
            )
            entry["semantic_source"] = "missing_placeholder"
        entries.append(entry)
    return entries


def build_catalog(
    datasets: Iterable[str] = ("legacy", "ycb", "hssd"),
    *,
    legacy_images_dir: Optional[Path] = None,
    legacy_objects_dir: Optional[Path] = None,
    ycb_root: Optional[Path] = None,
    hssd_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    selected = {str(d).strip().lower() for d in datasets if str(d).strip()}
    entries: List[Dict[str, Any]] = []
    if "legacy" in selected:
        entries.extend(legacy_image_entries(legacy_images_dir, legacy_objects_dir))
    if "ycb" in selected:
        entries.extend(ycb_entries(ycb_root))
    if "hssd" in selected:
        entries.extend(hssd_entries(hssd_root))
    unique: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        key = str(entry.get("object_key") or entry.get("object_name") or entry.get("model_id"))
        unique[key] = entry
    return list(unique.values())


def load_catalog(path: Path = OBJECT_CATALOG_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("objects", [])
    else:
        items = payload
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def save_catalog(entries: Sequence[Dict[str, Any]], path: Path = OBJECT_CATALOG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "object_catalog.v1",
        "object_count": len(entries),
        "objects": list(entries),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_entries(
    entries: Sequence[Dict[str, Any]],
    *,
    datasets: Optional[Iterable[str]] = None,
    object_set: Optional[Sequence[str]] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    selected = list(entries)
    if datasets:
        allowed = {str(x).strip().lower() for x in datasets if str(x).strip()}
        selected = [e for e in selected if str(e.get("dataset", "")).lower() in allowed]
    if object_set:
        allowed_keys = {str(x).strip() for x in object_set if str(x).strip()}
        selected = [
            e for e in selected
            if str(e.get("object_key")) in allowed_keys
            or str(e.get("object_name")) in allowed_keys
            or str(e.get("model_id")) in allowed_keys
        ]
    if limit and limit > 0:
        selected = selected[: int(limit)]
    return selected


def write_missing_semantic_csv(entries: Sequence[Dict[str, Any]], path: Path = MISSING_SEMANTIC_TEXT_PATH) -> int:
    missing = [
        e for e in entries
        if str(e.get("semantic_source", "")).startswith("missing")
        or not str(e.get("semantic_text", "")).strip()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["object_key", "dataset", "model_id", "template_config_path", "render_asset", "semantic_id", "semantic_text", "failure_reason"],
        )
        writer.writeheader()
        for entry in missing:
            writer.writerow(
                {
                    "object_key": entry.get("object_key", ""),
                    "dataset": entry.get("dataset", ""),
                    "model_id": entry.get("model_id", ""),
                    "template_config_path": entry.get("template_config_path", ""),
                    "render_asset": entry.get("render_asset", ""),
                    "semantic_id": entry.get("semantic_id", ""),
                    "semantic_text": entry.get("semantic_text", ""),
                    "failure_reason": "missing_semantic_text",
                }
            )
    return len(missing)


def object_entries_from_args(
    *,
    catalog_path: Optional[str | Path] = None,
    datasets: Optional[Iterable[str]] = None,
    object_set_path: Optional[str | Path] = None,
    images_dir: Optional[str | Path] = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    catalog_file = Path(catalog_path).expanduser() if catalog_path else OBJECT_CATALOG_PATH
    entries = load_catalog(catalog_file)
    if not entries:
        entries = build_catalog(datasets or ("legacy",), legacy_images_dir=Path(images_dir).expanduser() if images_dir else None)

    object_set: Optional[List[str]] = None
    if object_set_path:
        payload = json.loads(Path(object_set_path).expanduser().read_text(encoding="utf-8"))
        if isinstance(payload, list):
            object_set = [str(x) for x in payload]
        elif isinstance(payload, dict):
            raw = payload.get("objects", payload.get("object_keys", []))
            if isinstance(raw, list):
                object_set = [str(x.get("object_key", x.get("object_name", ""))) if isinstance(x, dict) else str(x) for x in raw]

    return filter_entries(entries, datasets=datasets, object_set=object_set, limit=limit)
