#!/usr/bin/env python3
from __future__ import annotations

"""Object lifespan profile inference.

The first implementation is rule-based and intentionally conservative.  It
turns object catalog entries into mobility/lifecycle metadata consumed by the
event generator and state engine.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence


JsonDict = Dict[str, Any]


CONSUMABLE_WORDS = {"apple", "fruit", "food", "cake", "snack", "drink", "bottle", "paper", "tissue"}
FLOOR_WORDS = {"chair", "table", "stool", "wheelchair", "cart", "plant"}
ROUTINE_WORDS = {"mug", "cup", "plate", "bowl", "book", "remote", "phone", "laptop", "notebook", "camera", "toy", "pillow"}
TEMPORARY_WORDS = {"package", "gift", "bag", "box", "suitcase"}
REPLACEABLE_WORDS = {"lamp", "clock", "alarm", "vase"}


def _text(entry: Mapping[str, Any]) -> str:
    return " ".join(
        str(entry.get(key, ""))
        for key in ("object_name", "model_id", "display_name", "semantic_text", "category")
    ).lower()


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def infer_mobility_class(entry: Mapping[str, Any]) -> str:
    text = _text(entry)
    if _contains_any(text, CONSUMABLE_WORDS):
        return "consumable"
    if _contains_any(text, TEMPORARY_WORDS):
        return "temporary"
    if _contains_any(text, FLOOR_WORDS):
        return "semi_static"
    if _contains_any(text, REPLACEABLE_WORDS):
        return "replaceable"
    if _contains_any(text, ROUTINE_WORDS):
        return "routine_movable"
    return "frequently_movable"


def infer_home_location_type(entry: Mapping[str, Any], mobility_class: str) -> str:
    text = _text(entry)
    if mobility_class == "consumable":
        return "kitchen_storage"
    if "pillow" in text or "blanket" in text:
        return "soft_surface"
    if mobility_class == "semi_static":
        return "floor"
    if any(word in text for word in ("book", "laptop", "notebook", "camera", "clock")):
        return "private_room"
    if any(word in text for word in ("plate", "bowl", "cup", "mug", "tea")):
        return "kitchen_storage"
    return "shared_room"


def persistence_for_class(mobility_class: str) -> JsonDict:
    if mobility_class == "semi_static":
        return {"p_stay": 0.96, "p_move_on_activity": 0.04, "p_return_home": 0.7, "p_clean_up": 0.5, "p_disappear": 0.0}
    if mobility_class == "consumable":
        return {"p_stay": 0.75, "p_move_on_activity": 0.35, "p_return_home": 0.2, "p_clean_up": 0.2, "p_disappear": 0.25}
    if mobility_class == "temporary":
        return {"p_stay": 0.55, "p_move_on_activity": 0.4, "p_return_home": 0.1, "p_clean_up": 0.45, "p_disappear": 0.2}
    if mobility_class == "replaceable":
        return {"p_stay": 0.9, "p_move_on_activity": 0.12, "p_return_home": 0.65, "p_clean_up": 0.35, "p_disappear": 0.02}
    if mobility_class == "routine_movable":
        return {"p_stay": 0.78, "p_move_on_activity": 0.55, "p_return_home": 0.45, "p_clean_up": 0.45, "p_disappear": 0.0}
    return {"p_stay": 0.7, "p_move_on_activity": 0.45, "p_return_home": 0.35, "p_clean_up": 0.35, "p_disappear": 0.0}


def compatible_activities(entry: Mapping[str, Any], mobility_class: str) -> List[str]:
    text = _text(entry)
    activities: List[str] = []
    if any(word in text for word in ("mug", "cup", "plate", "bowl", "food", "apple", "tea", "bottle")):
        activities.extend(["breakfast", "dinner", "shopping"])
    if any(word in text for word in ("laptop", "book", "notebook", "camera", "clock")):
        activities.extend(["work", "leisure"])
    if any(word in text for word in ("toy", "chess", "remote", "pillow", "statue")):
        activities.extend(["leisure", "cleaning"])
    if mobility_class in {"semi_static", "temporary", "replaceable"}:
        activities.append("cleaning")
    return sorted(set(activities or ["leisure"]))


def build_object_lifespan_profiles(entries: Sequence[Mapping[str, Any]], *, limit: int = 0) -> JsonDict:
    selected = list(entries[: int(limit)]) if limit and limit > 0 else list(entries)
    objects: List[JsonDict] = []
    for idx, entry in enumerate(selected):
        mobility = infer_mobility_class(entry)
        home_type = infer_home_location_type(entry, mobility)
        model_id = str(entry.get("model_id") or entry.get("object_name") or f"object_{idx}")
        profile = {
            "object_id": f"object_{idx:04d}",
            "object_key": entry.get("object_key", model_id),
            "object_name": entry.get("object_name", model_id),
            "model_id": model_id,
            "dataset": entry.get("dataset", ""),
            "display_name": entry.get("display_name", model_id),
            "semantic_text": entry.get("semantic_text", ""),
            "mobility_class": mobility,
            "home_location_type": home_type,
            "persistence": persistence_for_class(mobility),
            "activities": compatible_activities(entry, mobility),
            "inventory": {
                "is_consumable": mobility == "consumable",
                "initial_quantity": 3 if mobility == "consumable" else 1,
                "replenish_quantity": 3 if mobility == "consumable" else 0,
            },
            "home_anchor": {
                "room_type": "kitchen" if home_type == "kitchen_storage" else ("bedroom" if home_type == "private_room" else "living room"),
                "receptacle_category": "storage" if home_type.endswith("storage") else ("floor" if home_type == "floor" else "table_or_shelf"),
            },
        }
        objects.append(profile)
    return {
        "schema_version": "1.0",
        "object_count": len(objects),
        "objects": objects,
    }
