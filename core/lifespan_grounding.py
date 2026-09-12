#!/usr/bin/env python3
from __future__ import annotations

"""Ground semantic lifespan snapshots into continuous Habitat 3D layouts.

The first snapshot reuses a physically validated baseline layout. Later
snapshots preserve unchanged object poses and only re-place objects reported by
the lifespan state engine as changed. This keeps temporal continuity while
reusing the project's support-surface and anti-floating placement checks.
"""

import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from place_objects_on_instances import place_objects_on_instances
from project_paths import default_object_config_dirs_str


JsonDict = Dict[str, Any]


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    for suffix in (".object_config.json", ".object_config", ".glb"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _object_key(obj: Mapping[str, Any]) -> str:
    return _model_key(obj.get("model_id") or obj.get("name"))


def _event_index(event_log: Mapping[str, Any]) -> Dict[str, JsonDict]:
    return {
        str(item.get("event_id", "")): dict(item)
        for item in event_log.get("events", [])
        if isinstance(item, dict) and item.get("event_id")
    }


def _portable_order(assignments: Sequence[Mapping[str, Any]]) -> List[str]:
    preferred = (
        "tea_set", "food_apple", "food_pears", "carrot_cake", "chess_set",
        "camera", "alarm_clock", "wine_bottles", "brass_pot",
        "vase", "statue", "bust",
    )
    keys = [_object_key(item) for item in assignments]
    ranked: List[str] = []
    for token in preferred:
        ranked.extend(key for key in keys if token in key and key not in ranked)
    ranked.extend(key for key in keys if key and key not in ranked)
    return ranked


def _choose_changed_models(
    request: Mapping[str, Any],
    semantic_by_id: Mapping[str, Mapping[str, Any]],
    assignment_by_model: Mapping[str, Mapping[str, Any]],
    portable_order: Sequence[str],
    snapshot_index: int,
) -> List[str]:
    changed: List[str] = []
    for object_id in request.get("changed_object_ids", []) or []:
        state = semantic_by_id.get(str(object_id))
        if not state:
            continue
        key = _object_key(state)
        if key in assignment_by_model and key not in changed:
            changed.append(key)
    # LLM/rule hints do not always match the available asset names. If a
    # timestamp contains an event but matched no object, move one portable
    # object so the requested observation still represents that activity.
    if not changed and request.get("event_ids") and portable_order:
        changed.append(portable_order[(snapshot_index - 1) % len(portable_order)])
    return changed[:3]


def _rotated_assignment(base: Mapping[str, Any], snapshot_index: int, object_offset: int) -> JsonDict:
    assignment = deepcopy(dict(base))
    candidates: List[int] = []
    for raw in [base.get("target_instance_id"), *(base.get("backup_instance_ids", []) or [])]:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value not in candidates:
            candidates.append(value)
    if candidates:
        chosen_index = (snapshot_index + object_offset) % len(candidates)
        chosen = candidates[chosen_index]
        assignment["target_instance_id"] = chosen
        assignment["backup_instance_ids"] = [value for value in candidates if value != chosen]
    assignment["source"] = "lifespan_changed_object"
    return assignment


def _state_metadata(state: Mapping[str, Any]) -> JsonDict:
    return {
        "object_id": state.get("object_id"),
        "exists": bool(state.get("exists", True)),
        "quantity": int(state.get("quantity", 1)),
        "location_state": state.get("location_state", "home"),
        "condition": state.get("condition", "normal"),
        "owner": state.get("owner", ""),
        "semantic_room_type": state.get("semantic_room_type", ""),
        "semantic_target": state.get("semantic_target", ""),
        "last_changed_at": state.get("last_changed_at", ""),
        "event_id": state.get("last_event_id", ""),
        "home_anchor": state.get("home_anchor", {}),
    }


def _snap_recorded_support(item: JsonDict, threshold: float = 0.01) -> None:
    """Correct a rare settle/AABB discrepancy using the recorded support plane."""
    gap = float(item.get("support_gap", 0.0))
    if abs(gap) <= threshold or not isinstance(item.get("position"), list):
        return
    old_y = float(item["position"][1])
    item["position"][1] = round(old_y - gap, 4)
    if "support_base_height" in item:
        item["support_base_height"] = round(float(item["support_base_height"]) - gap, 4)
    item["support_gap"] = 0.0
    item["support_correction"] = {
        "method": "snap_recorded_aabb_base_to_support_surface",
        "previous_y": old_y,
        "previous_support_gap": gap,
        "threshold_m": threshold,
    }


def _different_models(previous: Mapping[str, JsonDict], current: Mapping[str, JsonDict]) -> List[str]:
    changed: List[str] = []
    for key in sorted(set(previous) | set(current)):
        before = previous.get(key)
        after = current.get(key)
        if before is None or after is None:
            changed.append(key)
            continue
        if before.get("position") != after.get("position") or before.get("rotation") != after.get("rotation"):
            changed.append(key)
    return changed


def ground_sequence(
    *,
    scene: str,
    snapshot_payload: Mapping[str, Any],
    event_log: Mapping[str, Any],
    base_layout: Mapping[str, Any],
    assignment_plan: Mapping[str, Any],
    surfaces_payload: JsonDict,
    output_dir: Path,
    data_dir: Path,
    objects_dir: str,
    snapshot_limit: int,
    seed: int,
    min_distance: float,
    max_trials_per_object: int,
    settle_steps: int,
    temporal_plan: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    requests = [item for item in snapshot_payload.get("snapshot_requests", []) if isinstance(item, dict)]
    if snapshot_limit > 0:
        requests = requests[:snapshot_limit]
    if not requests:
        raise ValueError("snapshot_requests contains no snapshots")

    base_objects = [deepcopy(item) for item in base_layout.get("objects", []) if isinstance(item, dict)]
    assignments = [deepcopy(item) for item in assignment_plan.get("assignments", []) if isinstance(item, dict)]
    current: Dict[str, JsonDict] = {_object_key(item): item for item in base_objects if _object_key(item)}
    home: Dict[str, JsonDict] = deepcopy(current)
    assignment_by_model = {_object_key(item): item for item in assignments if _object_key(item)}
    portable_order = _portable_order(assignments)
    events = _event_index(event_log)
    surfaces_payload = dict(surfaces_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    layouts_dir = output_dir / "layouts"
    layouts_dir.mkdir(parents=True, exist_ok=True)

    snapshots: List[JsonDict] = []
    validation_rows: List[JsonDict] = []
    previous_path = ""
    sequence_id = output_dir.name
    temporal_rows = {
        int(row.get("snapshot_index", -1)): row
        for row in (temporal_plan or {}).get("snapshots", [])
        if isinstance(row, dict)
    }

    for index, request in enumerate(requests):
        semantic_objects = [item for item in request.get("objects", []) if isinstance(item, dict)]
        semantic_by_id = {str(item.get("object_id")): item for item in semantic_objects}
        semantic_by_model = {_object_key(item): item for item in semantic_objects if _object_key(item)}
        semantic_changed_models = {
            _object_key(semantic_by_id[str(object_id)])
            for object_id in request.get("changed_object_ids", []) or []
            if str(object_id) in semantic_by_id
        }
        present_models = {key for key, item in semantic_by_model.items() if item.get("exists", True)}
        before = deepcopy(current)

        # Objects absent at this timestamp are removed from the physical scene.
        for key in list(current):
            if key in semantic_by_model and key not in present_models:
                current.pop(key, None)

        plan_row = temporal_rows.get(index, {})
        plan_changes = [row for row in plan_row.get("changes", []) if isinstance(row, dict)]
        if index == 0:
            requested_changed = []
        elif plan_changes:
            requested_changed = [_model_key(row.get("model_id")) for row in plan_changes]
        else:
            requested_changed = _choose_changed_models(
                request, semantic_by_id, assignment_by_model, portable_order, index
            )
        requested_changed = [key for key in requested_changed if key in present_models and key in current]

        # Cleanup/home transitions restore the exact validated home pose. Other
        # activity changes are re-grounded onto a Qwen-selected target/backup surface.
        reassign: List[JsonDict] = []
        restored: List[str] = []
        plan_change_by_model = {_model_key(row.get("model_id")): row for row in plan_changes}
        for offset, key in enumerate(requested_changed):
            state = semantic_by_model.get(key, {})
            if not plan_changes and (
                key in semantic_changed_models
                and str(state.get("location_state", "")).lower() == "home"
                and key in home
            ):
                current[key] = deepcopy(home[key])
                restored.append(key)
                continue
            base_assignment = assignment_by_model.get(key)
            if base_assignment:
                planned = plan_change_by_model.get(key)
                if planned and planned.get("target_instance_id") is not None:
                    assignment = deepcopy(dict(base_assignment))
                    assignment["target_instance_id"] = int(planned["target_instance_id"])
                    assignment["target_room_id"] = planned.get("target_room_id", assignment.get("target_room_id"))
                    assignment["sampled_region_id"] = assignment["target_room_id"]
                    assignment["source"] = "qwen_temporal_plan"
                    assignment["temporal_reasoning"] = planned.get("reason", "")
                    reassign.append(assignment)
                else:
                    reassign.append(_rotated_assignment(base_assignment, index, offset))

        moving_keys = {_object_key(item) for item in reassign}
        fixed = [item for key, item in current.items() if key not in moving_keys]
        placement = {"objects": [], "auto_placement_stats": {"failed_objects": []}}
        if reassign:
            placement = place_objects_on_instances(
                scene_name=scene,
                assignment_plan={"scene_name": scene, "assignments": reassign},
                surfaces_payload=surfaces_payload,
                data_dir=data_dir,
                objects_dir=objects_dir,
                min_distance=min_distance,
                spawn_height=0.3,
                max_trials_per_object=max_trials_per_object,
                settle_steps=settle_steps,
                seed=seed + index,
                fixed_objects=fixed,
            )
            for item in placement.get("objects", []):
                _snap_recorded_support(item)
                key = _object_key(item)
                if key:
                    current[key] = deepcopy(item)

        for key, item in current.items():
            state = semantic_by_model.get(key)
            if state:
                item["lifespan_state"] = _state_metadata(state)

        actual_changed = _different_models(before, current) if index > 0 else sorted(current)
        time_label = str(request.get("time_label", f"snapshot_{index:03d}"))
        layout_name = f"snapshot_{index:03d}_{time_label}.json"
        layout_path = layouts_dir / layout_name
        rel_path = str(layout_path.relative_to(output_dir)).replace(os.sep, "/")
        layout_payload = {
            "scene": base_layout.get("scene", scene),
            "timestamp": time.time(),
            "objects": list(current.values()),
            "lifespan_generation": {
                "sequence_id": sequence_id,
                "scene": scene,
                "snapshot_index": index,
                "day_index": request.get("day_index"),
                "time": request.get("time"),
                "time_label": time_label,
                "previous_layout": previous_path,
                "activity_context": [events.get(str(eid), {"event_id": eid}) for eid in request.get("event_ids", [])],
                "qwen_temporal_plan": plan_row,
                "requested_changed_models": requested_changed,
                "actual_changed_models": actual_changed,
                "restored_home_models": restored,
                "semantic_only": False,
                "temporal_continuity": True,
            },
            "auto_placement_stats": placement.get("auto_placement_stats", {}),
        }
        _write_json(layout_path, layout_payload)

        bad_support = [
            _object_key(item)
            for item in current.values()
            if abs(float(item.get("support_gap", 0.0))) > 0.01
        ]
        failed = placement.get("auto_placement_stats", {}).get("failed_objects", [])
        validation_rows.append(
            {
                "snapshot_index": index,
                "different_from_previous": bool(actual_changed) if index > 0 else True,
                "actual_changed_count": len(actual_changed),
                "placement_failed_count": len(failed),
                "bad_support_models": bad_support,
            }
        )
        snapshots.append(
            {
                "snapshot_index": index,
                "day_index": request.get("day_index"),
                "time_label": time_label,
                "layout_path": rel_path,
                "object_count": len(current),
                "changed_object_count": len(actual_changed),
                "placement_failed_count": len(failed),
                "semantic_only": False,
            }
        )
        previous_path = rel_path

    all_different = all(row["different_from_previous"] for row in validation_rows[1:])
    all_supported = all(not row["bad_support_models"] for row in validation_rows)
    manifest = {
        "type": "lifespan_grounded_sequence_manifest",
        "schema_version": "1.0",
        "scene": scene,
        "sequence_id": sequence_id,
        "snapshot_count": len(snapshots),
        "semantic_only": False,
        "temporal_continuity": True,
        "base_layout": str(base_layout.get("_source_path", "")),
        "assignment_model": assignment_plan.get("model", ""),
        "temporal_plan_model": (temporal_plan or {}).get("model", ""),
        "validation": {
            "all_adjacent_layouts_different": all_different,
            "all_support_gaps_valid": all_supported,
            "rows": validation_rows,
        },
        "snapshots": snapshots,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground lifespan semantic snapshots into continuous Habitat layouts.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--snapshot-requests", required=True)
    parser.add_argument("--event-log", required=True)
    parser.add_argument("--base-layout", required=True)
    parser.add_argument("--assignment-plan", required=True)
    parser.add_argument("--surfaces-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--objects-dir", default=default_object_config_dirs_str())
    parser.add_argument("--snapshot-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--min-distance", type=float, default=0.25)
    parser.add_argument("--max-trials-per-object", type=int, default=80)
    parser.add_argument("--settle-steps", type=int, default=45)
    parser.add_argument("--temporal-plan", default="", help="Optional Qwen-generated temporal movement plan JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot_path = Path(args.snapshot_requests).expanduser().resolve()
    event_path = Path(args.event_log).expanduser().resolve()
    base_path = Path(args.base_layout).expanduser().resolve()
    assignment_path = Path(args.assignment_plan).expanduser().resolve()
    surfaces_path = Path(args.surfaces_json).expanduser().resolve()
    base = _read_json(base_path)
    base["_source_path"] = str(base_path)
    surfaces = _read_json(surfaces_path)
    surfaces["_source_json_path"] = str(surfaces_path)
    surfaces["_source_json_dir"] = str(surfaces_path.parent)
    temporal_plan = _read_json(Path(args.temporal_plan).expanduser().resolve()) if args.temporal_plan else None
    manifest = ground_sequence(
        scene=args.scene,
        snapshot_payload=_read_json(snapshot_path),
        event_log=_read_json(event_path),
        base_layout=base,
        assignment_plan=_read_json(assignment_path),
        surfaces_payload=surfaces,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        data_dir=Path(args.data_dir).expanduser().resolve(),
        objects_dir=args.objects_dir,
        snapshot_limit=int(args.snapshot_limit),
        seed=int(args.seed),
        min_distance=float(args.min_distance),
        max_trials_per_object=int(args.max_trials_per_object),
        settle_steps=int(args.settle_steps),
        temporal_plan=temporal_plan,
    )
    print(
        f"[OK] Grounded lifespan sequence: {args.output_dir} "
        f"snapshots={manifest['snapshot_count']} validation={manifest['validation']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
