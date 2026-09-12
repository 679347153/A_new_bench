#!/usr/bin/env python3
"""Re-ground Qwen-flagged placements that also fail semantic support checks."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from place_objects_on_instances import place_objects_on_instances  # noqa: E402
from project_paths import default_object_config_dirs_str, resolve_hm3d_root  # noqa: E402


CONFIG = {
    "00401-H8rQCnvBgo6": {
        "data": resolve_hm3d_root(),
        "surfaces": ROOT / "results/receptacle_queries/00401-H8rQCnvBgo6/00401-H8rQCnvBgo6_receptacle_surfaces_coordinate_fixed.json",
        "targets": {"chess_set_4k": (20, [37]), "throw_pillows_01_4k": (23, [121, 253])},
    },
    "00808-y9hTuugGdiq": {
        "data": resolve_hm3d_root(),
        "surfaces": ROOT / "results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_coordinate_fixed.json",
        "targets": {
            "carrot_cake_4k": (649, [584]),
            "chess_set_4k": (584, [573, 572]),
            "throw_pillows_01_4k": (577, [586, 228]),
        },
    },
}


def key(obj):
    return obj.get("model_id"), tuple(obj.get("position", [])), obj.get("target_instance_id")


def main() -> int:
    report = []
    for scene, cfg in CONFIG.items():
        layout_dir = ROOT / "results/lifespan" / scene / "lifespan_qwen_10/grounded/layouts"
        paths = sorted(layout_dir.glob("snapshot_*.json"))
        payloads = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in paths]
        surfaces = json.loads(Path(cfg["surfaces"]).read_text(encoding="utf-8"))
        replacements = {}
        for model_id, (target, backups) in cfg["targets"].items():
            representatives = {}
            for _, payload in payloads:
                for obj in payload.get("objects", []):
                    if obj.get("model_id") == model_id:
                        representatives.setdefault(key(obj), (obj, payload.get("objects", [])))
            for ordinal, (old_key, (obj, all_objects)) in enumerate(representatives.items()):
                assignment = {
                    "object_id": obj.get("object_id", obj.get("id")), "model_id": model_id,
                    "name": obj.get("name", model_id), "target_instance_id": target,
                    "backup_instance_ids": backups, "target_room_id": -1,
                    "orientation_mode": obj.get("orientation_mode", "free"),
                    "yaw_offset_deg": obj.get("yaw_offset_deg", 0.0),
                    "source": "qwen_visual_audit_geometry_verified",
                }
                fixed = [deepcopy(x) for x in all_objects if x.get("model_id") != model_id]
                placed = place_objects_on_instances(
                    scene_name=scene, assignment_plan={"scene_name": scene, "assignments": [assignment]},
                    surfaces_payload=surfaces, data_dir=Path(cfg["data"]),
                    objects_dir=default_object_config_dirs_str(), min_distance=0.12,
                    spawn_height=0.3, max_trials_per_object=100, settle_steps=120,
                    seed=9200 + ordinal, fixed_objects=fixed,
                )
                new = placed.get("objects", [])
                replacements[old_key] = deepcopy(new[0]) if new else None
                report.append({"scene":scene,"model_id":model_id,"old":old_key,"action":"replaced" if new else "deleted","new":new[0].get("position") if new else None,"target":new[0].get("target_instance_id") if new else None})
        for path, payload in payloads:
            revised=[]
            for obj in payload.get("objects", []):
                replacement=replacements.get(key(obj), obj)
                if replacement is not None:
                    if replacement is not obj and obj.get("lifespan_state"):
                        replacement["lifespan_state"]=deepcopy(obj["lifespan_state"])
                    revised.append(deepcopy(replacement))
            payload["objects"]=revised
            payload["qwen_visual_audit_repair"]={"audit_model":"qwen3-vl-plus","geometry_verified":True}
            path.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        manifest_path=layout_dir.parent/"manifest.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        for row,(path,payload) in zip(manifest.get("snapshots",[]),payloads):
            row["object_count"]=len(payload["objects"])
        manifest["qwen_visual_audit"]={"model":"qwen3-vl-plus","report":"results/qwen_visual_audit/lifespan_20/summary.json","repairs_applied":True}
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    out=ROOT/"results/qwen_visual_audit/lifespan_20/repair_report.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
