#!/usr/bin/env python3
from __future__ import annotations

"""Chronological object state propagation for lifespan generation."""

from copy import deepcopy
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple


JsonDict = Dict[str, Any]


def _keyword_match(profile: Mapping[str, Any], keywords: Sequence[Any]) -> bool:
    if not keywords:
        return False
    text = " ".join(
        str(profile.get(key, ""))
        for key in ("object_name", "model_id", "display_name", "semantic_text", "home_location_type", "mobility_class")
    ).lower()
    if any(str(k).lower() == "all" for k in keywords):
        return True
    return any(str(k).lower() in text for k in keywords)


def initialize_object_states(object_profiles: Mapping[str, Any]) -> List[JsonDict]:
    states: List[JsonDict] = []
    for profile in object_profiles.get("objects", []) if isinstance(object_profiles.get("objects"), list) else []:
        if not isinstance(profile, dict):
            continue
        inv = profile.get("inventory", {}) if isinstance(profile.get("inventory"), dict) else {}
        states.append(
            {
                "object_id": profile.get("object_id"),
                "model_id": profile.get("model_id"),
                "object_name": profile.get("object_name"),
                "exists": True,
                "quantity": int(inv.get("initial_quantity", 1)),
                "mobility_class": profile.get("mobility_class", "routine_movable"),
                "home_anchor": profile.get("home_anchor", {}),
                "location_state": "home",
                "semantic_room_type": (profile.get("home_anchor", {}) or {}).get("room_type", ""),
                "semantic_target": (profile.get("home_anchor", {}) or {}).get("receptacle_category", ""),
                "condition": "normal",
                "owner": "",
                "last_changed_at": "initial_state",
                "last_event_id": "",
            }
        )
    return states


def _apply_effect(states: List[JsonDict], effect: Mapping[str, Any], event: Mapping[str, Any]) -> Tuple[int, List[str]]:
    selector = effect.get("object_selector", {}) if isinstance(effect.get("object_selector"), dict) else {}
    keywords = selector.get("category_keywords", [])
    if not isinstance(keywords, list):
        keywords = [keywords]
    changed = 0
    changed_ids: List[str] = []
    effect_type = str(effect.get("type", "MOVE")).upper()
    for state in states:
        if not state.get("exists", True) and effect_type not in {"REPLENISH", "INTRODUCE"}:
            continue
        if not _keyword_match(state, keywords):
            continue
        if effect_type == "MOVE":
            state["location_state"] = str(effect.get("target_state", "active"))
            state["semantic_target"] = str(effect.get("target", "activity_surface"))
            event_rooms = event.get("rooms", [])
            if isinstance(event_rooms, list) and event_rooms:
                state["semantic_room_type"] = str(event_rooms[0])
        elif effect_type == "CLEANUP":
            state["location_state"] = "home"
            state["semantic_target"] = (state.get("home_anchor", {}) or {}).get("receptacle_category", "home_anchor")
        elif effect_type == "CONSUME":
            qty = int(state.get("quantity", 1))
            state["quantity"] = max(0, qty - int(effect.get("quantity", 1)))
            if state["quantity"] <= 0:
                state["exists"] = False
                state["location_state"] = "absent"
        elif effect_type == "REPLENISH":
            state["exists"] = True
            state["quantity"] = int(state.get("quantity", 0)) + int(effect.get("quantity", 1))
            state["location_state"] = "home"
        elif effect_type == "REMOVE":
            state["exists"] = False
            state["location_state"] = "absent"
        elif effect_type == "INTRODUCE":
            state["exists"] = True
            state["location_state"] = "home"
        elif effect_type == "DAMAGE":
            state["condition"] = str(effect.get("condition", "damaged"))
        elif effect_type == "REPAIR":
            state["condition"] = "normal"
        elif effect_type == "REPLACE":
            state["exists"] = True
            state["condition"] = "normal"
            state["location_state"] = "home"
            state["quantity"] = max(1, int(state.get("quantity", 1)))
        else:
            continue
        state["last_changed_at"] = f"day_{int(event.get('day_index', 0)):02d}_{event.get('time', '')}"
        state["last_event_id"] = event.get("event_id", "")
        changed += 1
        changed_ids.append(str(state.get("object_id")))
        # Keep routine changes sparse; important for temporal continuity.
        if effect_type in {"MOVE", "CLEANUP"} and changed >= 3:
            break
    return changed, changed_ids


def _time_minutes(value: Any) -> int:
    """Convert HH:MM, H:MM AM/PM, and time ranges to minutes.

    For a range such as ``9:00 AM - 5:00 PM`` the start time is used because
    that is when the activity begins affecting the scene.
    """
    text = str(value or "00:00").strip().upper()
    match = re.search(r"(?<!\d)(\d{1,2})(?::(\d{1,2}))?\s*(AM|PM)?", text)
    if not match:
        return 0
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem:
        hour %= 12
        if meridiem == "PM":
            hour += 12
    return max(0, min(23, hour)) * 60 + max(0, min(59, minute))


def propagate_states(
    object_profiles: Mapping[str, Any],
    event_log: Mapping[str, Any],
    *,
    duration_days: int,
    snapshots_per_day: Sequence[str],
) -> JsonDict:
    states = initialize_object_states(object_profiles)
    history: List[JsonDict] = []
    snapshot_requests: List[JsonDict] = []
    events = event_log.get("events", [])
    if not isinstance(events, list):
        events = []
    ordered_events = sorted(
        (event for event in events if isinstance(event, dict)),
        key=lambda event: (int(event.get("day_index", 1)), _time_minutes(event.get("time", ""))),
    )
    snapshot_points = sorted(
        (
            (day, _time_minutes(time_text), str(time_text))
            for day in range(1, int(duration_days) + 1)
            for time_text in snapshots_per_day
        ),
        key=lambda item: (item[0], item[1]),
    )

    event_index = 0
    changed_since_snapshot: List[str] = []
    events_since_snapshot: List[str] = []
    for snapshot_index, (day, snapshot_minute, time_text) in enumerate(snapshot_points):
        while event_index < len(ordered_events):
            event = ordered_events[event_index]
            event_key = (int(event.get("day_index", 1)), _time_minutes(event.get("time", "")))
            if event_key > (day, snapshot_minute):
                break
            event_changed_ids: List[str] = []
            for effect in event.get("effects", []) if isinstance(event.get("effects"), list) else []:
                if isinstance(effect, dict):
                    _, changed_ids = _apply_effect(states, effect, event)
                    event_changed_ids.extend(changed_ids)
            event_id = str(event.get("event_id", ""))
            events_since_snapshot.append(event_id)
            changed_since_snapshot.extend(event_changed_ids)
            history.append(
                {
                    "event_id": event_id,
                    "day_index": int(event.get("day_index", 1)),
                    "time": event.get("time", ""),
                    "changed_object_ids": sorted(set(event_changed_ids)),
                    "present_object_count": sum(1 for state in states if state.get("exists", True)),
                    "absent_object_count": sum(1 for state in states if not state.get("exists", True)),
                }
            )
            event_index += 1

        snapshot_requests.append(
            {
                "snapshot_index": snapshot_index,
                "day_index": day,
                "time": time_text,
                "time_label": f"day_{day:02d}_{time_text.replace(':', '')}",
                "event_ids": list(events_since_snapshot),
                "changed_object_ids": sorted(set(changed_since_snapshot)),
                "present_object_count": sum(1 for state in states if state.get("exists", True)),
                "absent_object_count": sum(1 for state in states if not state.get("exists", True)),
                "objects": deepcopy(states),
            }
        )
        changed_since_snapshot = []
        events_since_snapshot = []
    return {
        "schema_version": "1.0",
        "initial_object_count": len(states),
        "event_state_history": history,
        "snapshot_requests": snapshot_requests,
    }
