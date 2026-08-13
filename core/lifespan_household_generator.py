#!/usr/bin/env python3
from __future__ import annotations

"""High-level household planning for lifespan scenes.

This module implements the semantic planning stages from `doc/lifespan_exec.md`:
scene/persona-aware household selection, relationship-aware daily routines, and
Los Angeles monthly important-event planning.  Qwen/OpenAI-compatible calls are
optional; deterministic rule fallback returns the same schema.
"""

import calendar
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lifespan_schema import compact_for_prompt


JsonDict = Dict[str, Any]


ROOM_KEYWORDS = {
    "bedroom": ["bedroom", "bed room", "primary bedroom", "kids room", "nursery", "guest room"],
    "bathroom": ["bathroom", "toilet", "shower"],
    "kitchen": ["kitchen"],
    "living room": ["living", "family room", "lounge"],
    "dining room": ["dining"],
    "study": ["study", "office", "library"],
    "laundry room": ["laundry"],
    "entryway": ["entry", "foyer", "hallway"],
}


FURNITURE_HINTS = {
    "beds": ["bed", "crib", "bunk"],
    "work_surfaces": ["desk", "office chair", "computer", "monitor"],
    "child_hints": ["toy", "crib", "bunk", "children", "kid"],
    "elder_hints": ["walker", "wheelchair", "medicine", "grab bar"],
    "hosting_hints": ["dining table", "sofa", "couch", "serving", "bar"],
}


MONTH_EVENT_HINTS = {
    1: ["New Year reset", "winter cleaning", "school returns"],
    2: ["Valentine dinner", "Super Bowl gathering", "winter errands"],
    3: ["spring cleaning", "tax paperwork", "school project"],
    4: ["spring gathering", "Easter-related family meal", "garden prep"],
    5: ["Memorial Day gathering", "graduation visit", "spring hosting"],
    6: ["summer preparation", "vacation packing", "school break routine"],
    7: ["Independence Day gathering", "BBQ", "summer guest visit"],
    8: ["back-to-school preparation", "summer cleanup", "family visit"],
    9: ["Labor Day gathering", "routine transition", "school schedule"],
    10: ["Halloween decoration", "fall cleaning", "school event"],
    11: ["Thanksgiving preparation", "guest visit", "large grocery shopping"],
    12: ["holiday decoration", "gift/package arrival", "family gathering"],
}


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _category_from_room(room: Mapping[str, Any]) -> str:
    text = " ".join(
        str(room.get(key, ""))
        for key in ("category", "name", "room_name", "label", "semantic_label", "function")
    ).lower()
    for category, keywords in ROOM_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return text.strip() or "unknown"


def _iter_scene_objects(scene_info: Mapping[str, Any]) -> List[JsonDict]:
    objects = scene_info.get("objects", [])
    if isinstance(objects, list):
        return [dict(obj) for obj in objects if isinstance(obj, dict)]
    categories = scene_info.get("categories", [])
    if isinstance(categories, list):
        return [{"category": str(item)} for item in categories]
    return []


def summarize_scene(scene: str, scene_info: Mapping[str, Any]) -> JsonDict:
    rooms_raw = scene_info.get("rooms", [])
    rooms: List[JsonDict] = [dict(room) for room in rooms_raw if isinstance(room, dict)] if isinstance(rooms_raw, list) else []
    room_summaries: List[JsonDict] = []
    room_type_counts: Dict[str, int] = {}
    for idx, room in enumerate(rooms):
        category = _category_from_room(room)
        room_type_counts[category] = room_type_counts.get(category, 0) + 1
        room_summaries.append(
            {
                "room_id": room.get("id", room.get("room_id", room.get("region_id", idx))),
                "category": category,
                "name": room.get("name", room.get("label", category)),
                "raw_category": room.get("category", room.get("semantic_label", "")),
            }
        )

    objects = _iter_scene_objects(scene_info)
    object_categories: Dict[str, int] = {}
    for obj in objects:
        category = str(obj.get("category") or obj.get("name") or obj.get("label") or "unknown").lower()
        object_categories[category] = object_categories.get(category, 0) + 1
    category_text = " ".join(f"{k} " * min(v, 3) for k, v in object_categories.items()).lower()
    furniture_hints = {
        key: any(word in category_text for word in words)
        for key, words in FURNITURE_HINTS.items()
    }

    bedroom_count = room_type_counts.get("bedroom", 0)
    if bedroom_count <= 0:
        bedroom_count = sum(1 for key in object_categories if "bed" in key and "bedroom" not in key)
        bedroom_count = max(1, min(4, bedroom_count))

    return {
        "scene": scene,
        "room_count": len(rooms),
        "bedroom_count": int(max(1, bedroom_count)),
        "bathroom_count": int(room_type_counts.get("bathroom", 0)),
        "room_type_counts": room_type_counts,
        "rooms": room_summaries[:80],
        "top_object_categories": sorted(object_categories.items(), key=lambda x: (-x[1], x[0]))[:80],
        "furniture_hints": furniture_hints,
    }


def _extract_json_object(text: str) -> Optional[JsonDict]:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    return None


def _call_llm_json(client: Any, model: str, system: str, user: str, *, max_tokens: int = 4096) -> Optional[JsonDict]:
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        return _extract_json_object(text or "")
    except Exception as exc:
        print(f"[Warning] Lifespan LLM call failed, using rule fallback: {exc}")
        return None


def _persona_by_id(personas: Sequence[Mapping[str, Any]], profile_id: str) -> Optional[JsonDict]:
    for persona in personas:
        if str(persona.get("profile_id")) == profile_id:
            return dict(persona)
    return None


def _choose_fallback_personas(scene_summary: Mapping[str, Any], personas: Sequence[Mapping[str, Any]], rng: random.Random) -> List[JsonDict]:
    bedroom_count = int(scene_summary.get("bedroom_count", 1))
    hints = scene_summary.get("furniture_hints", {}) if isinstance(scene_summary.get("furniture_hints"), dict) else {}
    if bedroom_count <= 1:
        wanted = ["persona_011_single_professional"]
        if hints.get("work_surfaces"):
            wanted = ["persona_008_remote_worker_parent"]
    elif bedroom_count == 2:
        wanted = ["persona_012_young_couple_partner_a", "persona_013_young_couple_partner_b"]
        if hints.get("child_hints"):
            wanted.append("persona_004_elementary_student")
    elif bedroom_count == 3:
        wanted = ["persona_008_remote_worker_parent", "persona_009_commuter_parent", "persona_004_elementary_student"]
        if hints.get("elder_hints"):
            wanted.append("persona_016_grandparent_live_in")
    else:
        wanted = [
            "persona_050_large_family_coordinator",
            "persona_009_commuter_parent",
            "persona_006_high_school_student",
            "persona_004_elementary_student",
        ]
        if hints.get("elder_hints"):
            wanted.append("persona_015_elderly_assisted")
    selected: List[JsonDict] = []
    for profile_id in wanted:
        persona = _persona_by_id(personas, profile_id)
        if persona:
            selected.append(persona)
    if not selected and personas:
        selected.append(dict(personas[0]))
    return selected


def _shared_rooms(scene_summary: Mapping[str, Any]) -> List[str]:
    counts = scene_summary.get("room_type_counts", {}) if isinstance(scene_summary.get("room_type_counts"), dict) else {}
    out = [room for room in ("kitchen", "living room", "dining room", "study") if counts.get(room, 0)]
    return out or ["kitchen", "living room"]


def build_household_profile(
    scene: str,
    scene_summary: Mapping[str, Any],
    personas: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    client: Any = None,
    model: str = "",
) -> JsonDict:
    system = (
        "You generate structured household profiles for a long-horizon household dynamics simulator. "
        "Return only valid JSON. Choose residents from the provided persona profile_id values."
    )
    user = (
        "Read this scene summary and all candidate resident personas. Infer a plausible household for this specific home. "
        "Bedroom count should guide the number of long-term residents; furniture and room layout should affect age, care, work, and visitor choices.\n\n"
        f"Scene summary:\n{compact_for_prompt(scene_summary, max_chars=6000)}\n\n"
        f"Candidate personas:\n{compact_for_prompt(list(personas), max_items=60, max_chars=18000)}\n\n"
        "Return JSON with fields: scene_household_reasoning, scene, bedroom_count, resident_count, selected_residents, relationship_graph, household_habits. "
        "Each selected_residents item must include resident_id, source_profile_id, name, age, gender, relationship_role, assigned_private_room, primary_shared_rooms, personal_object_categories."
    )
    payload = _call_llm_json(client, model, system, user, max_tokens=4096)
    if payload and isinstance(payload.get("selected_residents"), list):
        payload.setdefault("scene", scene)
        payload.setdefault("generation_source", "llm")
        return payload

    rng = random.Random(int(config.get("seed", 42)))
    selected = _choose_fallback_personas(scene_summary, personas, rng)
    rooms = _shared_rooms(scene_summary)
    residents: List[JsonDict] = []
    for idx, persona in enumerate(selected):
        residents.append(
            {
                "resident_id": f"resident_{idx}",
                "source_profile_id": persona.get("profile_id"),
                "name": persona.get("name", persona.get("display_name", f"Resident {idx}")),
                "age": persona.get("age"),
                "gender": persona.get("gender"),
                "relationship_role": (persona.get("household_roles") or ["resident"])[0],
                "assigned_private_room": "bedroom" if idx < int(scene_summary.get("bedroom_count", 1)) else "shared bedroom",
                "primary_shared_rooms": rooms,
                "personal_object_categories": persona.get("owned_object_categories", []),
                "persona": persona,
            }
        )
    graph: List[JsonDict] = []
    if len(residents) >= 2:
        graph.append({"from": "resident_0", "to": "resident_1", "relation": "partner_or_co_resident"})
    for idx in range(2, len(residents)):
        graph.append({"from": "resident_0", "to": f"resident_{idx}", "relation": "family_or_household_member"})
    return {
        "scene_household_reasoning": "Rule fallback selected residents from bedroom count, furniture hints, and persona coverage.",
        "scene": scene,
        "generation_source": "rule_fallback",
        "bedroom_count": scene_summary.get("bedroom_count", 1),
        "resident_count": len(residents),
        "selected_residents": residents,
        "relationship_graph": graph,
        "household_habits": {
            "location": config.get("location", "Los Angeles, USA"),
            "shopping_day_preference": "Saturday",
            "cleaning_day_preference": "Sunday",
            "tidiness": "medium",
            "visitor_frequency": "occasional",
        },
    }


def generate_daily_routines(
    household_profile: Mapping[str, Any],
    scene_summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    client: Any = None,
    model: str = "",
) -> JsonDict:
    system = (
        "You create relationship-aware daily routines for a household simulator. "
        "Return only valid JSON; activities must reference resident_id values."
    )
    user = (
        "Create weekday and weekend daily routines for every long-term resident. "
        "Routines must match personality, age, occupation, room layout, and household relationships. "
        "Include collaborative activities between residents and object interaction hints.\n\n"
        f"Household profile:\n{compact_for_prompt(household_profile, max_chars=12000)}\n\n"
        f"Scene summary:\n{compact_for_prompt(scene_summary, max_chars=6000)}\n\n"
        "Return JSON with fields: routines, collaborative_activities. "
        "Each routine has resident_id, weekday, weekend. Each activity has time, activity, rooms, with_residents, object_effect_hints."
    )
    payload = _call_llm_json(client, model, system, user, max_tokens=4096)
    if payload and isinstance(payload.get("routines"), list):
        payload.setdefault("generation_source", "llm")
        return payload

    routines: List[JsonDict] = []
    residents = household_profile.get("selected_residents", [])
    if not isinstance(residents, list):
        residents = []
    for resident in residents:
        rid = str(resident.get("resident_id", "resident"))
        role = _lower(resident.get("relationship_role"))
        persona = resident.get("persona", {}) if isinstance(resident.get("persona"), dict) else {}
        prefs = persona.get("activity_preferences", []) if isinstance(persona, dict) else []
        is_child = "child" in role or any("child" in _lower(x) for x in persona.get("household_roles", [])) if isinstance(persona, dict) else False
        is_elder = "elder" in role or "retired" in _lower(persona.get("occupation_or_status", "")) if isinstance(persona, dict) else False
        weekday = [
            {"time": "07:00", "activity": "breakfast", "rooms": ["kitchen", "dining room"], "with_residents": "household", "object_effect_hints": ["mug", "plate", "food"]},
            {"time": "09:00", "activity": "school" if is_child else ("medication_and_reading" if is_elder else "work"), "rooms": ["study", "bedroom"], "with_residents": [], "object_effect_hints": ["laptop", "book", "medicine box"]},
            {"time": "18:30", "activity": "dinner", "rooms": ["kitchen", "dining room"], "with_residents": "household", "object_effect_hints": ["plate", "cup", "food"]},
            {"time": "20:00", "activity": "leisure", "rooms": ["living room", "bedroom"], "with_residents": [], "object_effect_hints": ["book", "remote", "tea", "pillow"]},
        ]
        weekend = [
            {"time": "08:30", "activity": "breakfast", "rooms": ["kitchen", "dining room"], "with_residents": "household", "object_effect_hints": ["mug", "plate", "food"]},
            {"time": "10:00", "activity": "cleaning", "rooms": ["living room", "bedroom", "kitchen"], "with_residents": "household", "object_effect_hints": ["all"]},
            {"time": "15:00", "activity": "leisure", "rooms": ["living room", "study"], "with_residents": [], "object_effect_hints": list(prefs)[:3]},
            {"time": "18:30", "activity": "dinner", "rooms": ["kitchen", "dining room"], "with_residents": "household", "object_effect_hints": ["plate", "cup", "food"]},
        ]
        routines.append({"resident_id": rid, "weekday": weekday, "weekend": weekend})
    return {
        "generation_source": "rule_fallback",
        "routines": routines,
        "collaborative_activities": [
            {
                "activity_id": "family_breakfast",
                "time": "07:00",
                "participants": [r.get("resident_id") for r in residents],
                "activity": "breakfast",
                "object_effect_hints": ["mug", "plate", "food"],
            },
            {
                "activity_id": "shared_dinner",
                "time": "18:30",
                "participants": [r.get("resident_id") for r in residents],
                "activity": "dinner",
                "object_effect_hints": ["plate", "cup", "food"],
            },
        ],
    }


def generate_monthly_events(
    household_profile: Mapping[str, Any],
    daily_routines: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    client: Any = None,
    model: str = "",
) -> JsonDict:
    rng = random.Random(int(config.get("seed", 42)) + 101)
    month_value = config.get("month", "random")
    month = rng.randint(1, 12) if str(month_value).lower() == "random" else int(month_value)
    duration_days = int(config.get("duration_days", calendar.monthrange(2026, month)[1]))
    duration_days = max(1, min(duration_days, calendar.monthrange(2026, month)[1]))
    location = str(config.get("location", "Los Angeles, USA"))
    system = (
        "You plan a month of important household events for Los Angeles, USA. "
        "Return only valid JSON."
    )
    user = (
        "Randomly choose or use the provided month, then create one important day-level event for every day. "
        "Events must fit Los Angeles, the month, household residents, routines, and home capacity. "
        "Include expected object effects and pre/main/post phases for complex events.\n\n"
        f"location={location}, selected_month={month}, duration_days={duration_days}\n"
        f"Household profile:\n{compact_for_prompt(household_profile, max_chars=10000)}\n\n"
        f"Daily routines:\n{compact_for_prompt(daily_routines, max_chars=9000)}\n\n"
        "Return JSON fields: location, selected_month, daily_events. daily_events length must equal duration_days. "
        "Each event has day_index, event_type, title, participants, importance, phases, expected_object_effects."
    )
    payload = _call_llm_json(client, model, system, user, max_tokens=8192)
    if payload and isinstance(payload.get("daily_events"), list):
        payload.setdefault("location", location)
        payload.setdefault("selected_month", month)
        payload.setdefault("generation_source", "llm")
        return payload

    hints = MONTH_EVENT_HINTS.get(month, ["normal routine"])
    events: List[JsonDict] = []
    residents = household_profile.get("selected_residents", [])
    resident_ids = [str(r.get("resident_id")) for r in residents if isinstance(r, dict)]
    for day in range(1, duration_days + 1):
        weekday = calendar.day_name[calendar.weekday(2026, month, day)]
        if weekday == "Saturday":
            event_type = "shopping"
            title = "weekly grocery shopping"
            effects = [{"type": "REPLENISH", "category_keywords": ["food", "fruit", "bottle"], "quantity": 3}]
        elif weekday == "Sunday":
            event_type = "cleaning"
            title = "weekly household cleaning"
            effects = [{"type": "CLEANUP", "category_keywords": ["all"]}]
        elif day in (1, 15):
            event_type = "special_event"
            title = rng.choice(hints)
            effects = [{"type": "MOVE", "category_keywords": ["plate", "cup", "decor", "gift"], "target_state": "active"}]
        else:
            event_type = "normal_routine_day"
            title = "normal routine day"
            effects = [{"type": "MOVE", "category_keywords": ["mug", "book", "phone"], "target_state": "active"}]
        events.append(
            {
                "day_index": day,
                "date_hint": f"2026-{month:02d}-{day:02d}",
                "weekday": weekday,
                "event_type": event_type,
                "title": title,
                "participants": resident_ids,
                "importance": "medium" if event_type != "normal_routine_day" else "low",
                "phases": ["main"] if event_type in ("normal_routine_day", "shopping", "cleaning") else ["pre", "main", "post"],
                "expected_object_effects": effects,
            }
        )
    return {
        "generation_source": "rule_fallback",
        "location": location,
        "selected_month": month,
        "daily_events": events,
    }
