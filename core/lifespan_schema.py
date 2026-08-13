#!/usr/bin/env python3
from __future__ import annotations

"""Shared JSON helpers and lightweight schema checks for lifespan generation."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


JsonDict = Dict[str, Any]


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(str(p))
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_object(payload: Any, *, label: str) -> JsonDict:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def require_list(payload: Any, *, label: str) -> List[Any]:
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON array")
    return payload


def validate_resident_personas(payload: Mapping[str, Any]) -> List[JsonDict]:
    profiles = payload.get("profiles", [])
    if not isinstance(profiles, list):
        raise ValueError("resident persona payload must contain a profiles array")
    required = {
        "profile_id",
        "name",
        "age",
        "gender",
        "occupation",
        "personality",
        "thoughts",
        "routine_preferences",
        "preferences",
    }
    out: List[JsonDict] = []
    missing: List[str] = []
    for idx, item in enumerate(profiles):
        if not isinstance(item, dict):
            missing.append(f"profiles[{idx}] is not an object")
            continue
        absent = [key for key in sorted(required) if key not in item]
        if absent:
            missing.append(f"{item.get('profile_id', idx)} missing {absent}")
        out.append(dict(item))
    if missing:
        raise ValueError("invalid resident personas: " + "; ".join(missing[:10]))
    return out


def validate_household_profile(payload: Mapping[str, Any]) -> None:
    required = ["scene", "resident_count", "selected_residents", "relationship_graph"]
    absent = [key for key in required if key not in payload]
    if absent:
        raise ValueError(f"household_profile missing fields: {absent}")
    if not isinstance(payload.get("selected_residents"), list):
        raise ValueError("household_profile.selected_residents must be a list")


def validate_daily_routines(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload.get("routines"), list):
        raise ValueError("resident_daily_routines must contain routines list")


def validate_daily_events(payload: Mapping[str, Any], *, duration_days: int) -> None:
    events = payload.get("daily_events")
    if not isinstance(events, list):
        raise ValueError("daily_important_events must contain daily_events list")
    days = {int(item.get("day_index", -1)) for item in events if isinstance(item, dict)}
    missing_days = [day for day in range(1, int(duration_days) + 1) if day not in days]
    if missing_days:
        raise ValueError(f"daily_important_events missing day_index values: {missing_days[:20]}")


def compact_for_prompt(value: Any, *, max_items: int = 80, max_chars: int = 12000) -> str:
    """Return compact JSON text suitable for LLM prompts."""
    if isinstance(value, list) and len(value) > max_items:
        value = value[:max_items]
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) > max_chars:
        return text[: max_chars - 20] + "...<truncated>"
    return text


def merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> JsonDict:
    out: JsonDict = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def normalize_time_label(day_index: int, time_text: str) -> str:
    return f"day_{int(day_index):02d}_{str(time_text).replace(':', '')}"


def list_to_lookup(items: Iterable[Mapping[str, Any]], key: str) -> Dict[str, JsonDict]:
    out: Dict[str, JsonDict] = {}
    for item in items:
        value = str(item.get(key, "")).strip()
        if value:
            out[value] = dict(item)
    return out
