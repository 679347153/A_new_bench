#!/usr/bin/env python3
from __future__ import annotations

"""Generate lifespan household plans and semantic layout snapshots.

MVP scope:
1. Read scene information and candidate resident personas.
2. Build a scene-specific household profile and relationship graph.
3. Generate relationship-aware daily routines.
4. Select a Los Angeles month and create daily important events.
5. Infer object lifespan profiles.
6. Expand routines/events into chronological object effects.
7. Propagate object states and write semantic snapshot layouts.

The generated snapshot JSONs are compatible in spirit with downstream layout
metadata, but the first version is semantic-only.  Habitat grounding can be
added on top of `snapshot_requests.json` in the next iteration.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lifespan_event_generator import build_event_log
from lifespan_household_generator import (
    build_household_profile,
    generate_daily_routines,
    generate_monthly_events,
    summarize_scene,
)
from lifespan_profiles import build_object_lifespan_profiles
from lifespan_schema import (
    merge_dict,
    read_json,
    validate_daily_events,
    validate_daily_routines,
    validate_household_profile,
    validate_resident_personas,
    write_json,
)
from lifespan_state_engine import propagate_states
from object_catalog import build_catalog, filter_entries, load_catalog
from project_paths import OBJECT_CATALOG_PATH, PROJECT_ROOT, resolve_results_root
from query_rooms_for_objects import (
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_KEY,
    DEFAULT_SSH_PASSWORD,
    DEFAULT_SSH_PORT,
    DEFAULT_SSH_USER,
    OpenAI,
    SSHTunnel,
)


JsonDict = Dict[str, Any]
DEFAULT_LIFESPAN_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_config(path: Path, overrides: Mapping[str, Any]) -> JsonDict:
    base = read_json(path, default={}) if path.is_file() else {}
    if not isinstance(base, dict):
        base = {}
    return merge_dict(base, {k: v for k, v in overrides.items() if v is not None})


def _find_scene_info(scene: str, explicit: Optional[str], results_root: Path) -> Tuple[JsonDict, str]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(_resolve_project_path(explicit))
    candidates.extend(
        [
            results_root / "scene_info" / scene / f"{scene}_scene_info.json",
            results_root / "scene_info" / "temp_export" / f"{scene}_scene_info.json",
            PROJECT_ROOT / "scene_info_export" / f"{scene}_scene_info.json",
            PROJECT_ROOT / "hm3d" / "scene_info_export" / f"{scene}_scene_info.json",
        ]
    )
    for path in candidates:
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict):
                return payload, str(path)
    print(f"[Warning] scene_info not found for {scene}; using minimal fallback scene summary.")
    return {"scene_info": {"scene": scene}, "rooms": [], "objects": [], "categories": []}, "minimal_fallback"


def _scene_info_root(payload: Mapping[str, Any]) -> JsonDict:
    if isinstance(payload.get("scene_info"), dict):
        root = dict(payload)
        if "rooms" not in root and isinstance(payload.get("scene_info", {}).get("rooms"), list):
            root["rooms"] = payload["scene_info"]["rooms"]
        return root
    return dict(payload)


def _load_personas(path: Path) -> List[JsonDict]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"resident persona file must be object: {path}")
    return validate_resident_personas(payload)


def _load_objects(args: argparse.Namespace) -> List[JsonDict]:
    catalog_path = _resolve_project_path(args.object_catalog)
    entries = load_catalog(catalog_path)
    if not entries:
        datasets = [part.strip() for part in str(args.object_datasets).split(",") if part.strip()]
        entries = build_catalog(datasets or ("legacy",))
    datasets = [part.strip() for part in str(args.object_datasets).split(",") if part.strip()]
    return filter_entries(entries, datasets=datasets or None, limit=int(args.object_limit))


def _start_llm_client(args: argparse.Namespace) -> Tuple[Optional[SSHTunnel], Optional[Any]]:
    if args.disable_lifespan_llm:
        print("[Info] Lifespan LLM disabled; using rule fallback.")
        return None, None
    if OpenAI is None:
        print("[Warning] openai package unavailable; using rule fallback.")
        return None, None
    if not (args.ssh_host and args.ssh_user and (args.ssh_password or args.ssh_key)):
        print("[Warning] SSH args incomplete; using rule fallback.")
        return None, None
    tunnel = SSHTunnel(
        ssh_host=str(args.ssh_host),
        ssh_port=int(args.ssh_port),
        ssh_user=str(args.ssh_user),
        ssh_password=args.ssh_password,
        ssh_key=args.ssh_key,
        remote_host=str(args.vllm_host),
        remote_port=int(args.vllm_port),
        local_port=int(args.local_port),
    )
    if not tunnel.start(timeout_s=30):
        print("[Warning] Lifespan LLM tunnel failed; using rule fallback.")
        return None, None
    client = OpenAI(api_key="EMPTY", base_url=tunnel.base_url, timeout=args.timeout)
    print(f"[Info] Lifespan LLM tunnel ready: {tunnel.base_url}")
    return tunnel, client


def _write_semantic_layouts(
    out_dir: Path,
    scene: str,
    sequence_id: str,
    selected_month: int,
    state_payload: Mapping[str, Any],
) -> List[JsonDict]:
    layouts_dir = out_dir / "layouts"
    layouts_dir.mkdir(parents=True, exist_ok=True)
    snapshots: List[JsonDict] = []
    previous_path = ""
    for request in state_payload.get("snapshot_requests", []) if isinstance(state_payload.get("snapshot_requests"), list) else []:
        if not isinstance(request, dict):
            continue
        idx = int(request.get("snapshot_index", len(snapshots)))
        time_label = str(request.get("time_label", f"snapshot_{idx:03d}"))
        objects: List[JsonDict] = []
        for obj in request.get("objects", []) if isinstance(request.get("objects"), list) else []:
            if not isinstance(obj, dict) or not obj.get("exists", True):
                continue
            objects.append(
                {
                    "id": obj.get("object_id"),
                    "object_id": obj.get("object_id"),
                    "model_id": obj.get("model_id"),
                    "name": obj.get("object_name"),
                    "semantic_room_type": obj.get("semantic_room_type", ""),
                    "semantic_target": obj.get("semantic_target", ""),
                    "position": None,
                    "rotation": None,
                    "lifespan_state": {
                        "exists": obj.get("exists", True),
                        "quantity": obj.get("quantity", 1),
                        "location_state": obj.get("location_state", "home"),
                        "condition": obj.get("condition", "normal"),
                        "owner": obj.get("owner", ""),
                        "last_changed_at": obj.get("last_changed_at", ""),
                        "event_id": obj.get("last_event_id", ""),
                        "home_anchor": obj.get("home_anchor", {}),
                    },
                }
            )
        payload = {
            "schema_version": "lifespan_semantic_layout.v1",
            "scene": scene,
            "objects": objects,
            "lifespan_generation": {
                "sequence_id": sequence_id,
                "scene": scene,
                "snapshot_index": idx,
                "day_index": request.get("day_index"),
                "time_label": time_label,
                "selected_month": selected_month,
                "previous_layout": previous_path,
                "activity_context": request.get("event_ids", []),
                "changed_object_count": len(request.get("changed_object_ids", [])),
                "present_object_count": request.get("present_object_count", len(objects)),
                "absent_object_count": request.get("absent_object_count", 0),
                "semantic_only": True,
            },
        }
        layout_path = layouts_dir / f"snapshot_{idx:03d}_{time_label}.json"
        write_json(layout_path, payload)
        rel_layout_path = layout_path.relative_to(out_dir)
        snapshots.append(
            {
                "snapshot_index": idx,
                "layout_path": str(rel_layout_path).replace(os.sep, "/"),
                "day_index": request.get("day_index"),
                "time_label": time_label,
                "event_ids": request.get("event_ids", []),
                "changed_object_count": len(request.get("changed_object_ids", [])),
                "placed_count": 0,
                "failed_count": 0,
                "semantic_only": True,
            }
        )
        previous_path = str(rel_layout_path).replace(os.sep, "/")
    return snapshots


def run_scene(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_dir)
    config_path = _resolve_project_path(args.config)
    overrides = {
        "duration_days": args.duration_days,
        "month": args.month,
        "seed": args.seed,
    }
    config = _load_config(config_path, overrides)
    if args.snapshots_per_day:
        config["snapshots_per_day"] = [part.strip() for part in args.snapshots_per_day.split(",") if part.strip()]
    duration_days = int(config.get("duration_days", 30))
    snapshots_per_day = list(config.get("snapshots_per_day", ["07:00", "12:00", "18:00", "22:00"]))
    personas_path = _resolve_project_path(str(config.get("resident_persona_pool", args.personas)))

    scene_info_payload, scene_info_path = _find_scene_info(args.scene, args.scene_info, results_root)
    scene_info = _scene_info_root(scene_info_payload)
    scene_summary = summarize_scene(args.scene, scene_info)
    personas = _load_personas(personas_path)
    object_entries = _load_objects(args)
    if not object_entries:
        print("[Warning] No object catalog entries found; state history will contain zero objects.")

    sequence_id = args.sequence_id or f"lifespan_{_timestamp()}"
    out_dir = results_root / "lifespan" / args.scene / sequence_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 0/6] Prepared inputs scene={args.scene} personas={len(personas)} objects={len(object_entries)}")
    print(f"[Info] Output directory: {out_dir}")

    tunnel: Optional[SSHTunnel] = None
    client: Optional[Any] = None
    if not args.disable_lifespan_llm:
        tunnel, client = _start_llm_client(args)

    try:
        print("[Stage 1/6] Household selection")
        household = build_household_profile(args.scene, scene_summary, personas, config, client=client, model=args.model)
        validate_household_profile(household)
        write_json(out_dir / "scene_summary.json", scene_summary)
        write_json(out_dir / "household_profile.json", household)
        write_json(out_dir / "household_relationship_graph.json", {"relationship_graph": household.get("relationship_graph", [])})

        print("[Stage 2/6] Object lifespan profiles")
        object_profiles = build_object_lifespan_profiles(object_entries, limit=int(args.object_limit))
        write_json(out_dir / "object_lifespan_profiles.json", object_profiles)

        print("[Stage 3/6] Relationship-aware daily routines")
        routines = generate_daily_routines(household, scene_summary, config, client=client, model=args.model)
        validate_daily_routines(routines)
        write_json(out_dir / "resident_daily_routines.json", routines)
        write_json(out_dir / "collaborative_activity_templates.json", {"collaborative_activities": routines.get("collaborative_activities", [])})

        print("[Stage 4/6] Los Angeles monthly important events")
        monthly_events = generate_monthly_events(household, routines, config, client=client, model=args.model)
        validate_daily_events(monthly_events, duration_days=duration_days)
        selected_month = int(monthly_events.get("selected_month", 1))
        write_json(out_dir / "monthly_calendar.json", {"location": monthly_events.get("location"), "selected_month": selected_month})
        write_json(out_dir / "daily_important_events.json", monthly_events)

        print("[Stage 5/6] Chronological state propagation")
        event_log = build_event_log(routines, monthly_events, duration_days=duration_days)
        state_payload = propagate_states(object_profiles, event_log, duration_days=duration_days, snapshots_per_day=snapshots_per_day)
        write_json(out_dir / "event_log.json", event_log)
        write_json(out_dir / "state_history.json", {"event_state_history": state_payload.get("event_state_history", [])})
        write_json(out_dir / "snapshot_requests.json", {"snapshot_requests": state_payload.get("snapshot_requests", [])})

        print("[Stage 6/6] Writing semantic snapshot layouts and manifest")
        snapshots = _write_semantic_layouts(out_dir, args.scene, sequence_id, selected_month, state_payload)
        validation = {
            "schema_version": "1.0",
            "status": "ok",
            "semantic_only": True,
            "checks": {
                "resident_count": int(household.get("resident_count", 0)),
                "daily_event_count": len(monthly_events.get("daily_events", [])),
                "event_count": int(event_log.get("event_count", 0)),
                "snapshot_count": len(snapshots),
                "object_count": int(object_profiles.get("object_count", 0)),
            },
            "notes": [
                "MVP output is semantic-only; 3D Habitat grounding should consume snapshot_requests.json in the next stage.",
            ],
        }
        manifest = {
            "type": "lifespan_sequence_manifest",
            "schema_version": "1.0",
            "scene": args.scene,
            "sequence_id": sequence_id,
            "duration_days": duration_days,
            "snapshot_count": len(snapshots),
            "location": monthly_events.get("location", config.get("location", "Los Angeles, USA")),
            "selected_month": selected_month,
            "resident_count_mode": config.get("resident_count_mode", "infer_from_bedrooms"),
            "bedroom_count": scene_summary.get("bedroom_count", 1),
            "resident_count": household.get("resident_count", 0),
            "semantic_only": True,
            "cache_paths": {
                "scene_info": scene_info_path,
                "resident_persona_pool": str(personas_path),
                "object_catalog": str(_resolve_project_path(args.object_catalog)),
            },
            "statistics": {
                "event_count": int(event_log.get("event_count", 0)),
                "object_count": int(object_profiles.get("object_count", 0)),
                "daily_event_count": len(monthly_events.get("daily_events", [])),
            },
            "snapshots": snapshots,
        }
        write_json(out_dir / "manifest.json", manifest)
        write_json(out_dir / "validation_report.json", validation)
        write_json(out_dir / "config_resolved.json", config)
        print(f"[OK] Lifespan semantic sequence saved: {out_dir}")
        print(f"[OK] snapshots={len(snapshots)} events={event_log.get('event_count', 0)} residents={household.get('resident_count', 0)}")
    finally:
        if tunnel:
            tunnel.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate lifespan household dynamics plans and semantic snapshots.")
    parser.add_argument("--scene", required=True, help="Scene id, e.g. 00808-y9hTuugGdiq")
    parser.add_argument("--scene-info", default="", help="Optional explicit scene_info JSON path")
    parser.add_argument("--config", default="data/lifespan/default_lifespan_config.json")
    parser.add_argument("--personas", default="data/lifespan/resident_persona_profiles.json")
    parser.add_argument("--object-catalog", default=str(OBJECT_CATALOG_PATH))
    parser.add_argument("--object-datasets", default="legacy,ycb,hssd")
    parser.add_argument("--object-limit", type=int, default=40, help="Limit objects for MVP state simulation; 0 means all")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--sequence-id", default="")
    parser.add_argument("--duration-days", type=int, default=None)
    parser.add_argument("--snapshots-per-day", default="", help="Comma-separated times, e.g. 07:00,12:00,18:00,22:00")
    parser.add_argument("--month", default=None, help="'random' or 1-12")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run-household-plan", action="store_true", help="Compatibility flag; MVP still writes all semantic plan files")
    parser.add_argument("--dry-run-state-only", action="store_true", help="Compatibility flag; MVP writes state files and semantic snapshots")
    parser.add_argument("--disable-lifespan-llm", action="store_true", help="Use deterministic rule fallback for household/routine/month planning")
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--ssh-password", default=DEFAULT_SSH_PASSWORD)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--local-port", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_LIFESPAN_MODEL)
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_scene(args)


if __name__ == "__main__":
    raise SystemExit(main())
