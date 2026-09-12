#!/usr/bin/env python3
"""Convert a placed-layout JSON into a Habitat scene-instance bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from project_paths import find_object_config_path


def _quaternion_from_yaw(yaw_degrees: float) -> list[float]:
    half_angle = math.radians(yaw_degrees) / 2.0
    return [0.0, math.sin(half_angle), 0.0, math.cos(half_angle)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--motion-type",
        choices=("static", "kinematic", "dynamic"),
        default="kinematic",
        help="Kinematic keeps the authored poses while still providing rigid-body collision.",
    )
    args = parser.parse_args()

    layout_path = args.layout.resolve()
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    stage_path = Path(payload["scene"]).resolve()
    if not stage_path.is_file():
        raise FileNotFoundError(f"Stage asset does not exist: {stage_path}")

    output_dir = args.output_dir.resolve()
    object_dir = output_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)

    instances = []
    exported = []
    for item in payload.get("objects", []):
        model_id = str(item.get("model_id", "")).strip()
        source_config = find_object_config_path(model_id)
        if source_config is None:
            raise FileNotFoundError(f"Object config not found: {model_id}")

        object_config = json.loads(source_config.read_text(encoding="utf-8"))
        render_asset = Path(object_config["render_asset"])
        if not render_asset.is_absolute():
            render_asset = (source_config.parent / render_asset).resolve()
        object_config["render_asset"] = str(render_asset)
        object_config["collision_asset"] = str(render_asset)
        object_config["is_collidable"] = True
        object_config["join_collision_meshes"] = True

        exported_config = object_dir / f"{model_id}.object_config.json"
        exported_config.write_text(
            json.dumps(object_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        rotation = item.get("rotation", [0.0, 0.0, 0.0])
        yaw = float(rotation[1]) if isinstance(rotation, list) and len(rotation) > 1 else 0.0
        instances.append(
            {
                "template_name": model_id,
                "translation_origin": "asset_local",
                "translation": [float(value) for value in item["position"][:3]],
                "rotation": _quaternion_from_yaw(yaw),
                "motion_type": args.motion_type,
            }
        )
        exported.append(model_id)

    scene_instance = {
        "translation_origin": "asset_local",
        "stage_instance": {"template_name": str(stage_path)},
        "object_instances": instances,
        "default_lighting": "no_lights",
        "user_defined": {
            "source_layout": str(layout_path),
            "exported_objects": exported,
        },
    }
    output_path = output_dir / f"{layout_path.stem}.scene_instance.json"
    output_path.write_text(
        json.dumps(scene_instance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(object_dir)
    print(f"exported_objects={len(instances)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
