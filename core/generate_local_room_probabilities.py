#!/usr/bin/env python3
"""Generate deterministic offline room probabilities from scene furniture semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFERENCES = {
    "alarm": ("bedroom",),
    "cake": ("kitchen", "living"),
    "coffee": ("kitchen", "living"),
    "chair": ("living", "kitchen", "bedroom"),
    "console": ("living", "storage"),
    "camera": ("living", "storage", "bedroom"),
    "megaphone": ("living", "storage"),
    "pot": ("kitchen", "living"),
    "vase": ("living", "bedroom", "kitchen"),
}

ROOM_MARKERS = {
    "bedroom": ("bed", "nightstand", "pillow"),
    "kitchen": ("kitchen", "refrigerator", "freezer", "oven"),
    "living": ("sofa", "coffee table", "tv", "armchair"),
    "bathroom": ("toilet", "bath", "shower"),
    "storage": ("shelving", "storage", "washer-dryer"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-info", type=Path, required=True)
    parser.add_argument("--object-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scene = json.loads(args.scene_info.read_text(encoding="utf-8"))
    objects = json.loads(args.object_set.read_text(encoding="utf-8"))
    scene_name = str(scene.get("scene_name") or scene.get("scene") or args.scene_info.parent.name)
    rooms = [r for r in scene.get("rooms", []) if int(r.get("region_id", -1)) >= 0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    room_types = {}
    for room in rooms:
        categories = " ".join(str(k).lower() for k in (room.get("categories") or {}))
        room_types[int(room["region_id"])] = {
            kind: sum(marker in categories for marker in markers)
            for kind, markers in ROOM_MARKERS.items()
        }

    for object_name in objects:
        needle = str(object_name).lower()
        preferred = next((value for key, value in PREFERENCES.items() if key in needle), ("living", "bedroom", "kitchen"))
        ranked = []
        for room in rooms:
            region_id = int(room["region_id"])
            semantic_score = sum((len(preferred) - idx) * room_types[region_id][kind] for idx, kind in enumerate(preferred))
            ranked.append((max(float(semantic_score), 0.1), room))
        ranked.sort(key=lambda item: (-item[0], int(item[1]["region_id"])))
        chosen = ranked[:5]
        total = sum(score for score, _ in chosen)
        payload = {
            "object_name": object_name,
            "scene_name": scene_name,
            "source": "local_semantic_heuristic",
            "probabilities": [
                {
                    "rank": idx,
                    "region_id": int(room["region_id"]),
                    "room_center": room.get("room_center", [0.0, 0.0, 0.0]),
                    "room_aabb": room.get("bounding_box", {}),
                    "probability": score / total,
                }
                for idx, (score, room) in enumerate(chosen, start=1)
            ],
        }
        path = args.output_dir / f"{object_name}_probs.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
