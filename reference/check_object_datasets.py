#!/usr/bin/env python3
from __future__ import annotations
import _path_setup  # noqa: F401

"""Check whether YCB/HSSD object datasets are wired into the benchmark flow.

This script has two layers:
1. Always available: validate catalog entries, template config files, and
   render/collision asset references.
2. Optional `--habitat-load`: in a Habitat environment, load template configs
   into Habitat-Sim and optionally instantiate a few sampled objects.

Example:
  python check_object_datasets.py --datasets ycb,hssd

Habitat-Sim load check:
  python check_object_datasets.py \
    --datasets ycb,hssd \
    --habitat-load \
    --scene 00808-y9hTuugGdiq \
    --limit-load 5
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.hm3d_paths import resolve_scene_paths
from core.object_catalog import object_entries_from_args
from core.project_paths import (
    OBJECT_CATALOG_PATH,
    default_object_config_dirs_str,
    iter_object_config_dirs,
    resolve_hm3d_root,
)
from core.sample_and_place_objects import build_object_template_index, resolve_model_id_for_template


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _asset_path(config_path: Path, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def validate_catalog_assets(entries: Iterable[Dict[str, Any]]) -> Counter:
    stats: Counter = Counter()
    missing: List[Dict[str, Any]] = []
    for entry in entries:
        dataset = str(entry.get("dataset", "unknown"))
        config_path = Path(str(entry.get("template_config_path", "")))
        if not config_path.is_file():
            stats[(dataset, "missing_config")] += 1
            missing.append({"dataset": dataset, "object": entry.get("object_name"), "kind": "config", "path": str(config_path)})
            continue
        stats[(dataset, "config_ok")] += 1
        config = _load_config(config_path)
        if not config:
            stats[(dataset, "config_json_unreadable")] += 1
            missing.append({"dataset": dataset, "object": entry.get("object_name"), "kind": "config_json", "path": str(config_path)})
            continue
        for key in ("render_asset", "collision_asset"):
            path = _asset_path(config_path, config.get(key))
            if path is None:
                stats[(dataset, f"{key}_empty")] += 1
            elif path.is_file():
                stats[(dataset, f"{key}_ok")] += 1
            else:
                stats[(dataset, f"{key}_missing")] += 1
                missing.append({"dataset": dataset, "object": entry.get("object_name"), "kind": key, "path": str(path)})
    for (dataset, key), value in sorted(stats.items()):
        print(f"[Check] {dataset}.{key}: {value}")
    if missing:
        print(f"[Error] Missing/unreadable asset count: {len(missing)}")
        for item in missing[:12]:
            print(f"  - {item}")
    else:
        print("[OK] All checked config/render/collision asset references are valid.")
    return stats


def habitat_load_check(args: argparse.Namespace, entries: List[Dict[str, Any]]) -> int:
    try:
        import habitat_sim  # type: ignore[import-not-found]
        import numpy as np
    except Exception as exc:
        print(f"[Skip] habitat_sim unavailable in this environment: {exc}")
        return 2

    scene_paths = resolve_scene_paths(args.scene, require_semantic=False, root=Path(args.data_dir))
    if scene_paths is None:
        print(f"[Error] Scene not found for Habitat load check: {args.scene}")
        return 1

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
    sim_cfg.scene_id = str(scene_paths.stage_glb)
    sim_cfg.enable_physics = True
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [16, 16]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        template_mgr = sim.get_object_template_manager()
        for config_dir in iter_object_config_dirs(args.objects_dir):
            if hasattr(template_mgr, "load_configs"):
                template_mgr.load_configs(str(config_dir.resolve()))
            elif hasattr(template_mgr, "add_template_search_path"):
                template_mgr.add_template_search_path(str(config_dir.resolve()))
            elif hasattr(template_mgr, "load_object_configs"):
                template_mgr.load_object_configs(str(config_dir.resolve()))

        template_index = build_object_template_index(args.objects_dir)
        rom = sim.get_rigid_object_manager()
        ok = 0
        failed: List[Dict[str, str]] = []
        for entry in entries[: max(1, int(args.limit_load))]:
            model_id = resolve_model_id_for_template(str(entry.get("model_id", "")), template_index)
            handles = template_mgr.get_template_handles(model_id)
            if not handles:
                handles = template_mgr.get_template_handles(f"{model_id}.object_config.json")
            if not handles:
                failed.append({"model_id": model_id, "reason": "template_handle_not_found"})
                continue
            try:
                obj = rom.add_object_by_template_handle(handles[0])
                if obj is None:
                    failed.append({"model_id": model_id, "reason": "add_object_returned_none"})
                    continue
                obj.translation = np.array([0.0, 1.0 + ok * 0.05, 0.0], dtype=np.float32)
                ok += 1
            except Exception as exc:
                failed.append({"model_id": model_id, "reason": str(exc)})
        print(f"[OK] Habitat template instantiate check: loaded={ok}/{min(len(entries), int(args.limit_load))}")
        if failed:
            print(f"[Error] Habitat load failures: {len(failed)}")
            for item in failed[:10]:
                print(f"  - {item}")
            return 1
        return 0
    finally:
        sim.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate object datasets and optional Habitat-Sim loading.")
    parser.add_argument("--datasets", default="ycb,hssd", help="Comma-separated datasets: legacy,ycb,hssd")
    parser.add_argument("--object-catalog", default=str(OBJECT_CATALOG_PATH))
    parser.add_argument("--objects-dir", default=default_object_config_dirs_str())
    parser.add_argument("--limit", type=int, default=0, help="Limit catalog validation entries; 0 means all")
    parser.add_argument("--habitat-load", action="store_true", help="Also load templates and instantiate samples in Habitat-Sim")
    parser.add_argument("--scene", default="00808-y9hTuugGdiq", help="Scene used for --habitat-load")
    parser.add_argument("--data-dir", default=str(resolve_hm3d_root()))
    parser.add_argument("--limit-load", type=int, default=5, help="How many objects to instantiate per Habitat load check")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = _split_csv(args.datasets)
    entries = object_entries_from_args(
        catalog_path=args.object_catalog,
        datasets=datasets,
        limit=int(args.limit),
    )
    print(f"[Info] Entries selected: {len(entries)} datasets={datasets}")
    if not entries:
        print("[Error] No entries selected.")
        return 1
    stats = validate_catalog_assets(entries)
    template_index = build_object_template_index(args.objects_dir)
    print(f"[Info] Template index size: {len(template_index)}")
    missing_template = [
        str(e.get("model_id", ""))
        for e in entries
        if resolve_model_id_for_template(str(e.get("model_id", "")), template_index).lower() not in template_index
    ]
    if missing_template:
        print(f"[Error] Template index misses: {len(missing_template)} preview={missing_template[:10]}")
        return 1
    print("[OK] Catalog entries can be resolved to local template ids.")

    bad_counts = [value for (dataset, key), value in stats.items() if key.endswith("_missing") or key == "missing_config" or key == "config_json_unreadable"]
    if any(bad_counts):
        return 1
    if args.habitat_load:
        return habitat_load_check(args, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
