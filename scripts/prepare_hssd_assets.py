#!/usr/bin/env python3
from __future__ import annotations

"""Assemble Habitat-ready HSSD object configs around an existing GLB download."""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def read_csv_by_id(path: Path, id_column: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {str(row.get(id_column, "")).strip(): row for row in rows if str(row.get(id_column, "")).strip()}


def condensed_categories(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        fieldnames = rows.fieldnames or []
        category_column = next((name for name in fieldnames if "CONDENSED" in name), "")
        return {
            str(row.get("Object Hash", "")).strip(): str(row.get(category_column, "")).strip()
            for row in rows
            if str(row.get("Object Hash", "")).strip()
        }


def semantic_record(model_id: str, row: dict[str, str], category: str, config: dict[str, Any]) -> dict[str, Any]:
    display_name = str(row.get("name", "")).strip() or f"HSSD object {model_id[:8]}"
    category = category or str(row.get("main_category", "")).strip() or "household object"
    rooms = [item.strip() for item in str(row.get("foundIn", "")).split(",") if item.strip()]
    support = str(row.get("support", "")).strip()
    room_text = f" It is commonly found in {', '.join(rooms)}." if rooms else ""
    support_text = f" Its support orientation is {support}." if support else ""
    return {
        "object_key": f"hssd:{model_id}",
        "model_id": model_id,
        "object_name": model_id,
        "display_name": display_name,
        "category": category,
        "semantic_id": config.get("semantic_id"),
        "semantic_text": f"{display_name}. HSSD category: {category}.{room_text}{support_text}",
        "likely_rooms": rooms,
        "semantic_source": "hssd_official_metadata",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-source", required=True, type=Path)
    parser.add_argument("--mesh-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--semantic-output", required=True, type=Path)
    args = parser.parse_args()

    config_source = args.config_source.expanduser().resolve()
    mesh_source = args.mesh_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    semantic_output = args.semantic_output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty output directory: {output}")

    object_rows = read_csv_by_id(config_source / "semantics" / "objects.csv", "id")
    categories = condensed_categories(config_source / "metadata" / "hssd_obj_semantics_condensed.csv")
    records: list[dict[str, Any]] = []
    skipped_missing_render = 0
    collision_fallbacks = 0

    for config_path in sorted((config_source / "objects").rglob("*.object_config.json")):
        model_id = config_path.name.removesuffix(".object_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        render_name = str(config.get("render_asset", f"{model_id}.glb"))
        source_dir = mesh_source / config_path.parent.name
        render_source = source_dir / render_name
        if not render_source.is_file():
            skipped_missing_render += 1
            continue

        destination = output / "objects" / config_path.parent.name
        destination.mkdir(parents=True, exist_ok=True)
        render_link = destination / render_name
        render_link.symlink_to(render_source)

        collision_name = str(config.get("collision_asset", ""))
        collision_source = source_dir / collision_name if collision_name else Path()
        if collision_name and collision_source.is_file():
            if collision_name != render_name:
                (destination / collision_name).symlink_to(collision_source)
            config["is_collidable"] = True
        else:
            config["collision_asset"] = render_name
            config["is_collidable"] = False
            config["collision_asset_fallback"] = "render_asset"
            collision_fallbacks += 1

        (destination / config_path.name).write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(semantic_record(model_id, object_rows.get(model_id, {}), categories.get(model_id, ""), config))

    for name in ("metadata", "semantics"):
        source = config_source / name
        if source.is_dir():
            shutil.copytree(source, output / name, dirs_exist_ok=True)
    for source in config_source.glob("*.scene_dataset_config.json"):
        shutil.copy2(source, output / source.name)

    semantic_output.parent.mkdir(parents=True, exist_ok=True)
    semantic_output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    print(json.dumps({
        "usable_objects": len(records),
        "skipped_missing_render": skipped_missing_render,
        "collision_fallbacks": collision_fallbacks,
        "output": str(output),
        "semantic_output": str(semantic_output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
