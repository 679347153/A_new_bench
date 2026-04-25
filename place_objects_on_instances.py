#!/usr/bin/env python3
from __future__ import annotations

"""
Physics-aware placement on assigned receptacle instances (File2).

Overview
--------
This script receives object->instance assignments and executes automatic placement:
1) Build target surface index from scene-level receptacle output.
2) For each object, sample candidate points on target instance top surface.
3) Spawn object at `surface_y + spawn_height` (default 0.3m).
4) Enforce pairwise minimum distance constraint.
5) If habitat-sim is available:
   - instantiate rigid body from object template
   - step physics for settling
   - reject candidate on contact collision with already placed objects
6) Export final layout json.

Collision Policy
----------------
- Geometric pre-check: minimum XZ distance between object centers.
- Physics contact check: reject candidate if new object contacts existing objects.
- Retry policy: iterate multiple surface points per object (`max_trials_per_object`).

Execution Guide
---------------
1) Direct invocation with prepared assignment plan:
   python place_objects_on_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --assignment-plan results/object_instance_assignments/00824-Dd4bFSTQ8gi/00824-Dd4bFSTQ8gi_object_instance_plan.json \
     --surfaces-json results/receptacle_queries/00824-Dd4bFSTQ8gi/00824-Dd4bFSTQ8gi_receptacle_surfaces_all_rooms.json

2) Stricter spacing + more retries:
   python place_objects_on_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --assignment-plan <plan_json> \
     --surfaces-json <surfaces_json> \
     --min-distance 0.3 \
     --max-trials-per-object 60 \
     --settle-steps 80

3) Keep requested spawn policy explicit:
   python place_objects_on_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --assignment-plan <plan_json> \
     --surfaces-json <surfaces_json> \
     --spawn-height 0.3
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from extract_room_instances import DEFAULT_DATA_DIR
from hm3d_paths import resolve_scene_paths

try:
    import habitat_sim  # type: ignore[import-not-found]
except ImportError:
    habitat_sim = None

try:
    from sample_and_place_objects import infer_object_profile
except Exception:
    infer_object_profile = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion with deterministic fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _get_profile(model_id: str) -> Dict[str, float]:
    """
    Resolve object collision profile from existing project heuristics.

    Falls back to conservative defaults when no profile is found.
    """
    if infer_object_profile is not None:
        try:
            profile = infer_object_profile(model_id)
            if isinstance(profile, dict):
                return {
                    "radius": float(profile.get("radius", 0.2)),
                    "y_offset": float(profile.get("y_offset", 0.05)),
                }
        except Exception:
            pass
    return {"radius": 0.2, "y_offset": 0.05}


def _resolve_template_handle(template_mgr: Any, model_id: str) -> Optional[str]:
    """
    Resolve habitat object template handle from flexible model id aliases.

    Supports direct id, `.object_config.json`, and `_4k` variants.
    """
    try:
        candidates = template_mgr.get_template_handles(model_id)
        if candidates:
            return candidates[0]
    except Exception:
        pass
    try:
        candidates = template_mgr.get_template_handles(f"{model_id}.object_config.json")
        if candidates:
            return candidates[0]
    except Exception:
        pass
    if not model_id.endswith("_4k"):
        for key in (f"{model_id}_4k", f"{model_id}_4k.object_config.json"):
            try:
                candidates = template_mgr.get_template_handles(key)
                if candidates:
                    return candidates[0]
            except Exception:
                pass
    try:
        all_handles = template_mgr.get_template_handles()
        needle = model_id.lower().replace(".object_config.json", "")
        for handle in all_handles:
            name = str(handle).lower().replace(".object_config.json", "")
            if name == needle or name == f"{needle}_4k":
                return handle
    except Exception:
        pass
    return None


def _remove_object_safe(rom: Any, obj: Any) -> None:
    """Remove a temporary/failed object instance without raising."""
    if obj is None:
        return
    object_id = getattr(obj, "object_id", None)
    handle = getattr(obj, "handle", None)
    if handle:
        try:
            rom.remove_object_by_handle(handle)
            return
        except Exception:
            pass
    if object_id is not None:
        try:
            rom.remove_object_by_id(object_id)
        except Exception:
            pass


def _step_physics(sim: Any, steps: int) -> None:
    """Step simulator physics multiple frames, compatible with different APIs."""
    for _ in range(max(0, int(steps))):
        try:
            sim.step_physics(1.0 / 60.0)
        except TypeError:
            sim.step_physics()
        except Exception:
            break


def _contact_with_existing(sim: Any, candidate_object_id: int, existing_ids: Sequence[int]) -> bool:
    """
    Detect whether candidate object physically contacts any already-placed object.

    Returns True if a collision/contact is found.
    """
    try:
        if hasattr(sim, "perform_discrete_collision_detection"):
            sim.perform_discrete_collision_detection()
    except Exception:
        pass

    if not hasattr(sim, "get_physics_contact_points"):
        return False
    try:
        contacts = sim.get_physics_contact_points()
    except Exception:
        return False
    existing_set = set(int(x) for x in existing_ids)
    for cp in contacts:
        a = getattr(cp, "object_id_a", getattr(cp, "obj_id_a", None))
        b = getattr(cp, "object_id_b", getattr(cp, "obj_id_b", None))
        if a is None or b is None:
            continue
        try:
            ai = int(a)
            bi = int(b)
        except Exception:
            continue
        if ai == int(candidate_object_id) and bi in existing_set:
            return True
        if bi == int(candidate_object_id) and ai in existing_set:
            return True
    return False


def _distance_ok(
    pos: Sequence[float],
    radius: float,
    placed: Sequence[Dict[str, Any]],
    min_distance: float,
) -> bool:
    """Check pairwise minimum distance constraint on XZ plane."""
    x = _safe_float(pos[0])
    z = _safe_float(pos[2])
    for item in placed:
        p = item.get("position", [0.0, 0.0, 0.0])
        px = _safe_float(p[0])
        pz = _safe_float(p[2])
        other_r = _safe_float(item.get("_radius", 0.2), 0.2)
        dx = x - px
        dz = z - pz
        required = max(float(min_distance), float(radius + other_r))
        if (dx * dx + dz * dz) < (required * required):
            return False
    return True


def _build_surface_index(surfaces_payload: Dict[str, Any]) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Build fast lookup maps for receptacle surfaces by `(room_id, instance_id)` and `instance_id`."""
    by_room_instance: Dict[Tuple[int, int], Dict[str, Any]] = {}
    by_instance: Dict[int, Dict[str, Any]] = {}
    for room in surfaces_payload.get("rooms", []) or []:
        room_id = int(room.get("room_id", -1))
        for item in room.get("receptacle_instances", []) or []:
            try:
                instance_id = int(item.get("instance_id"))
            except Exception:
                continue
            by_room_instance[(room_id, instance_id)] = item
            by_instance[instance_id] = item
    return by_room_instance, by_instance


def _choose_surface_item(
    assignment: Dict[str, Any],
    by_room_instance: Dict[Tuple[int, int], Dict[str, Any]],
    by_instance: Dict[int, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Select matched surface entry for one assignment (room-aware first, global fallback)."""
    try:
        target_instance_id = int(assignment.get("target_instance_id"))
    except Exception:
        return None
    room_id = assignment.get("target_room_id", assignment.get("sampled_region_id", None))
    if room_id is not None:
        try:
            key = (int(room_id), target_instance_id)
            if key in by_room_instance:
                return by_room_instance[key]
        except Exception:
            pass
    return by_instance.get(target_instance_id)


def _sample_surface_points(points: List[List[float]], max_trials: int, rng: random.Random) -> List[List[float]]:
    """Shuffle/downsample surface points so each object tries bounded candidates."""
    clean = [p for p in points if isinstance(p, list) and len(p) >= 3]
    if not clean:
        return []
    if len(clean) <= max_trials:
        rng.shuffle(clean)
        return clean
    idx = list(range(len(clean)))
    rng.shuffle(idx)
    return [clean[i] for i in idx[:max_trials]]


def _make_simulator(scene_name: str, data_dir: Path, enable_physics: bool = True) -> Optional[Any]:
    """Create lightweight habitat-sim simulator for placement and contact checks."""
    if habitat_sim is None:
        return None
    scene_paths = resolve_scene_paths(scene_name, require_semantic=False, root=data_dir)
    if scene_paths is None:
        return None
    try:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
        sim_cfg.scene_id = str(scene_paths.stage_glb)
        sim_cfg.enable_physics = bool(enable_physics)
        sim_cfg.gpu_device_id = 0

        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "color"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [32, 32]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor]

        return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    except Exception:
        return None


def _load_templates(sim: Any, objects_dir: str) -> None:
    """Load object template configs into simulator from local `objects/` directory."""
    if sim is None:
        return
    try:
        template_mgr = sim.get_object_template_manager()
    except Exception:
        return
    abs_dir = os.path.abspath(objects_dir)
    if not os.path.isdir(abs_dir):
        return
    try:
        if hasattr(template_mgr, "load_configs"):
            template_mgr.load_configs(abs_dir)
        elif hasattr(template_mgr, "add_template_search_path"):
            template_mgr.add_template_search_path(abs_dir)
        elif hasattr(template_mgr, "load_object_configs"):
            template_mgr.load_object_configs(abs_dir)
    except Exception:
        return


def place_objects_on_instances(
    scene_name: str,
    assignment_plan: Dict[str, Any],
    surfaces_payload: Dict[str, Any],
    data_dir: Path = DEFAULT_DATA_DIR,
    objects_dir: str = "./objects",
    min_distance: float = 0.25,
    spawn_height: float = 0.3,
    max_trials_per_object: int = 30,
    settle_steps: int = 45,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Core placement routine used by file1.

    Input contract:
    - `assignment_plan["assignments"]` must include `target_instance_id`.
    - `surfaces_payload` must include `top_surface.points` for each instance.

    Placement loop:
    1) pick candidate surface points for one object
    2) spawn at `point_y + spawn_height`
    3) enforce minimum distance
    4) optional habitat-sim contact validation
    5) accept first valid candidate or mark as failed
    """
    rng = random.Random(int(seed))
    np.random.seed(int(seed))

    scene_paths = resolve_scene_paths(scene_name, require_semantic=False, root=data_dir)
    scene_path = str(scene_paths.stage_glb) if scene_paths is not None else scene_name
    by_room_instance, by_instance = _build_surface_index(surfaces_payload)

    sim = _make_simulator(scene_name, data_dir=data_dir, enable_physics=True)
    if sim is not None:
        _load_templates(sim, objects_dir)

    assignments = assignment_plan.get("assignments", []) if isinstance(assignment_plan, dict) else []
    placed_layout_objects: List[Dict[str, Any]] = []
    placed_internal: List[Dict[str, Any]] = []
    failed_objects: List[Dict[str, Any]] = []

    rom = None
    template_mgr = None
    if sim is not None:
        try:
            rom = sim.get_rigid_object_manager()
            template_mgr = sim.get_object_template_manager()
        except Exception:
            rom = None
            template_mgr = None

    for idx, assignment in enumerate(assignments):
        model_id = str(assignment.get("model_id", "")).strip()
        name = str(assignment.get("name", model_id or f"obj_{idx}"))
        object_id = assignment.get("object_id", idx)
        target_instance_id = assignment.get("target_instance_id")
        room_id = assignment.get("target_room_id", assignment.get("sampled_region_id", -1))
        surface_item = _choose_surface_item(assignment, by_room_instance, by_instance)
        if surface_item is None:
            failed_objects.append(
                {
                    "object_id": object_id,
                    "model_id": model_id,
                    "target_instance_id": target_instance_id,
                    "reason": "missing_surface_instance",
                }
            )
            continue

        surface_points = surface_item.get("top_surface", {}).get("points", [])
        candidates = _sample_surface_points(surface_points, max_trials=max_trials_per_object, rng=rng)
        if not candidates:
            failed_objects.append(
                {
                    "object_id": object_id,
                    "model_id": model_id,
                    "target_instance_id": target_instance_id,
                    "reason": "empty_surface_points",
                }
            )
            continue

        profile = _get_profile(model_id)
        radius = float(profile.get("radius", 0.2))
        placed = False
        failure_reason = "no_valid_candidate"
        for pt in candidates:
            spawn_pos = [
                _safe_float(pt[0]),
                _safe_float(pt[1]) + float(spawn_height),
                _safe_float(pt[2]),
            ]
            if not _distance_ok(spawn_pos, radius, placed_internal, min_distance=min_distance):
                failure_reason = "min_distance_rejected"
                continue

            yaw = float(rng.uniform(0.0, 360.0))
            final_pos = list(spawn_pos)
            sim_object_id = None
            sim_handle = None

            if sim is not None and rom is not None and template_mgr is not None and model_id:
                template_handle = _resolve_template_handle(template_mgr, model_id)
                if template_handle is None:
                    failure_reason = "template_not_found"
                    continue
                try:
                    obj = rom.add_object_by_template_handle(template_handle)
                    if obj is None:
                        failure_reason = "failed_to_add_object"
                        continue
                    sim_object_id = int(getattr(obj, "object_id", -1))
                    sim_handle = getattr(obj, "handle", None)
                    obj.translation = np.array(spawn_pos, dtype=np.float32)
                    if hasattr(obj, "motion_type") and hasattr(habitat_sim, "physics"):
                        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                    _step_physics(sim, steps=settle_steps)
                    pos = getattr(obj, "translation", np.array(spawn_pos, dtype=np.float32))
                    final_pos = [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)]
                    existing_ids = [x.get("_sim_object_id") for x in placed_internal if x.get("_sim_object_id") is not None]
                    if not _distance_ok(final_pos, radius, placed_internal, min_distance=min_distance):
                        _remove_object_safe(rom, obj)
                        failure_reason = "min_distance_after_settle"
                        continue
                    if sim_object_id is not None and _contact_with_existing(sim, sim_object_id, existing_ids):
                        _remove_object_safe(rom, obj)
                        failure_reason = "habitat_contact_collision"
                        continue
                    if hasattr(obj, "motion_type") and hasattr(habitat_sim, "physics"):
                        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                except Exception:
                    failure_reason = "habitat_sim_runtime_error"
                    continue

            placed = True
            layout_obj = {
                "id": int(idx),
                "name": name,
                "model_id": model_id,
                "position": [round(float(final_pos[0]), 4), round(float(final_pos[1]), 4), round(float(final_pos[2]), 4)],
                "rotation": [0.0, round(float(yaw), 4), 0.0],
                "sampled_region_id": int(room_id) if room_id is not None else -1,
                "target_instance_id": int(target_instance_id) if target_instance_id is not None else -1,
                "source": "assigned_instance_surface",
            }
            placed_layout_objects.append(layout_obj)
            placed_internal.append(
                {
                    "object_id": object_id,
                    "position": layout_obj["position"],
                    "_radius": radius,
                    "_sim_object_id": sim_object_id,
                    "_sim_handle": sim_handle,
                }
            )
            break

        if not placed:
            failed_objects.append(
                {
                    "object_id": object_id,
                    "model_id": model_id,
                    "target_instance_id": target_instance_id,
                    "reason": failure_reason,
                }
            )

    if sim is not None:
        try:
            sim.close()
        except Exception:
            pass

    return {
        "scene": scene_path,
        "timestamp": time.time(),
        "objects": placed_layout_objects,
        "placement_scope": {
            "mode": "assigned_instance_surface",
            "min_distance": float(min_distance),
            "spawn_height": float(spawn_height),
            "max_trials_per_object": int(max_trials_per_object),
            "settle_steps": int(settle_steps),
        },
        "auto_placement_stats": {
            "total_objects": int(len(assignments)),
            "placed_count": int(len(placed_layout_objects)),
            "failed_count": int(len(failed_objects)),
            "habitat_sim_used": bool(sim is not None),
            "failed_objects": failed_objects,
        },
    }


def parse_args() -> argparse.Namespace:
    """Define CLI for standalone file2 execution."""
    parser = argparse.ArgumentParser(description="Place objects on assigned instance top surfaces and write layout JSON.")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--assignment-plan", required=True, help="Assignment plan JSON from file1")
    parser.add_argument("--surfaces-json", required=True, help="Receptacle surfaces JSON from query_room_receptacle_objects.py")
    parser.add_argument("--output-layout", type=str, default=None, help="Output layout JSON path")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="HM3D root directory")
    parser.add_argument("--objects-dir", type=str, default="./objects", help="Object template configs directory")
    parser.add_argument("--min-distance", type=float, default=0.25, help="Minimum pairwise object distance on XZ")
    parser.add_argument("--spawn-height", type=float, default=0.3, help="Spawn height above target surface point")
    parser.add_argument("--max-trials-per-object", type=int, default=30, help="Max candidate points per object")
    parser.add_argument("--settle-steps", type=int, default=45, help="Physics settle steps after spawn")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint: load inputs, run placement, write layout JSON."""
    args = parse_args()
    try:
        assignment_plan = json.loads(Path(args.assignment_plan).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Error] Failed to load assignment plan: {exc}", file=sys.stderr)
        return 1
    try:
        surfaces_payload = json.loads(Path(args.surfaces_json).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Error] Failed to load surfaces json: {exc}", file=sys.stderr)
        return 1

    layout = place_objects_on_instances(
        scene_name=args.scene,
        assignment_plan=assignment_plan,
        surfaces_payload=surfaces_payload,
        data_dir=Path(args.data_dir),
        objects_dir=args.objects_dir,
        min_distance=float(args.min_distance),
        spawn_height=float(args.spawn_height),
        max_trials_per_object=int(args.max_trials_per_object),
        settle_steps=int(args.settle_steps),
        seed=int(args.seed),
    )

    if args.output_layout:
        output_path = Path(args.output_layout)
    else:
        out_dir = Path("./results/layouts") / args.scene
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"assigned_instance_layout_{int(time.time())}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Layout saved: {output_path}")
    stats = layout.get("auto_placement_stats", {})
    print(
        "[OK] Placement stats: placed={}/{} failed={} habitat_sim_used={}".format(
            int(stats.get("placed_count", 0)),
            int(stats.get("total_objects", 0)),
            int(stats.get("failed_count", 0)),
            bool(stats.get("habitat_sim_used", False)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
