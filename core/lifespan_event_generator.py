#!/usr/bin/env python3
from __future__ import annotations

"""Convert routines and month-level events into chronological object events."""

from typing import Any, Dict, List, Mapping, Sequence


JsonDict = Dict[str, Any]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if value == "household":
        return ["household"]
    if isinstance(value, list):
        return value
    return [value]


def _effect_from_hint(hint: Any, effect_type: str = "MOVE") -> JsonDict:
    return {
        "type": effect_type,
        "object_selector": {"category_keywords": [str(hint)]},
        "target": "activity_surface",
        "target_state": "active",
    }


def build_event_log(
    daily_routines: Mapping[str, Any],
    daily_events: Mapping[str, Any],
    *,
    duration_days: int,
) -> JsonDict:
    routines = daily_routines.get("routines", [])
    routine_events: List[JsonDict] = []
    for day in range(1, int(duration_days) + 1):
        is_weekend = day % 7 in (6, 0)
        routine_key = "weekend" if is_weekend else "weekday"
        for routine in routines if isinstance(routines, list) else []:
            if not isinstance(routine, dict):
                continue
            rid = str(routine.get("resident_id", "resident"))
            for item in routine.get(routine_key, []):
                if not isinstance(item, dict):
                    continue
                hints = item.get("object_effect_hints", [])
                effects = [_effect_from_hint(hint) for hint in _as_list(hints)[:5]]
                activity = str(item.get("activity", "activity"))
                time_text = str(item.get("time", "12:00"))
                routine_events.append(
                    {
                        "event_id": f"day_{day:02d}_{time_text.replace(':', '')}_{activity}_{rid}",
                        "event_type": "routine_activity",
                        "activity": activity,
                        "day_index": day,
                        "time": time_text,
                        "sort_key": f"{day:03d}-{time_text}",
                        "residents": [rid],
                        "rooms": item.get("rooms", []),
                        "with_residents": item.get("with_residents", []),
                        "effects": effects,
                        "parent_event_id": None,
                    }
                )

    special_events: List[JsonDict] = []
    for item in daily_events.get("daily_events", []) if isinstance(daily_events.get("daily_events"), list) else []:
        if not isinstance(item, dict):
            continue
        day = int(item.get("day_index", 1))
        effects: List[JsonDict] = []
        for eff in item.get("expected_object_effects", []):
            if not isinstance(eff, dict):
                continue
            effects.append(
                {
                    "type": eff.get("type", "MOVE"),
                    "object_selector": {"category_keywords": eff.get("category_keywords", ["all"])},
                    "target": eff.get("target", "event_surface"),
                    "target_state": eff.get("target_state", "active"),
                    "quantity": eff.get("quantity", 1),
                }
            )
        special_events.append(
            {
                "event_id": f"day_{day:02d}_1200_{item.get('event_type', 'important_event')}",
                "event_type": item.get("event_type", "important_event"),
                "activity": item.get("title", item.get("event_type", "important_event")),
                "day_index": day,
                "time": "12:00",
                "sort_key": f"{day:03d}-12:00",
                "residents": item.get("participants", []),
                "rooms": [],
                "phases": item.get("phases", ["main"]),
                "effects": effects,
                "parent_event_id": None,
            }
        )

    events = sorted(routine_events + special_events, key=lambda x: str(x.get("sort_key", "")))
    return {
        "schema_version": "1.0",
        "event_count": len(events),
        "events": events,
    }
