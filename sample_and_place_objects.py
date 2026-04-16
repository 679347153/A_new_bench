#!/usr/bin/env python3
"""
根据概率信息采样物体位置，并通过交互式编辑器进行微调。

逻辑流程：
1. 选择模式：--mode load（读取已有概率）或 generate（随机生成概率）
2. 加载/生成概率文件：对每个物体，前5个房间各分配一个概率（Σ=1）
3. 采样物体位置：每个物体根据概率分布选择一个房间，使用房间几何中心作为位置
4. 生成中间布局JSON
5. 启动 test_layout.py 进行人工微调
6. 保存微调结果，循环重复或退出

用法：
  # 首次运行：生成概率文件
  python sample_and_place_objects.py \
    --scene 00808-y9hTuugGdiq \
    --mode generate \
    --images-dir ./objects_images \
    --rooms-info-dir ./results/scene_info \
    --probabilities-dir ./results/probabilities \
    --layouts-dir ./results/layouts

  # 后续运行：读取已有概率文件
  python sample_and_place_objects.py \
    --scene 00808-y9hTuugGdiq \
    --mode load \
    --probabilities-dir ./results/probabilities \
    --layouts-dir ./results/layouts
"""

import argparse
import json
import os
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from hm3d_paths import list_available_scenes, resolve_scene_paths

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Install with: pip install numpy")
    sys.exit(1)


# ===== 常量 =====
DEFAULT_ROOMS_INFO_DIR = "./results/scene_info"
DEFAULT_PROBABILITIES_DIR = "./results/probabilities"
DEFAULT_LAYOUTS_DIR = "./results/layouts"
DEFAULT_IMAGES_DIR = "./objects_images"

AVAILABLE_SCENES = list_available_scenes(require_semantic=True)

PROFILE_KEYWORDS: Dict[str, Dict[str, float]] = {
    "table": {"radius": 0.60, "y_offset": 0.05},
    "desk": {"radius": 0.58, "y_offset": 0.05},
    "sofa": {"radius": 0.70, "y_offset": 0.05},
    "chair": {"radius": 0.40, "y_offset": 0.05},
    "bed": {"radius": 0.80, "y_offset": 0.05},
    "cabinet": {"radius": 0.55, "y_offset": 0.05},
    "shelf": {"radius": 0.50, "y_offset": 0.05},
    "statue": {"radius": 0.42, "y_offset": 0.08},
    "vase": {"radius": 0.28, "y_offset": 0.10},
    "bottle": {"radius": 0.24, "y_offset": 0.08},
    "clock": {"radius": 0.20, "y_offset": 0.10},
    "camera": {"radius": 0.20, "y_offset": 0.08},
}

DEFAULT_OBJECT_PROFILE = {"radius": 0.35, "y_offset": 0.05}


# ===== 模板映射 =====

def build_object_template_index(objects_dir: str = "./objects") -> Dict[str, str]:
    """Build case-insensitive map: object stem -> template model_id."""
    index: Dict[str, str] = {}
    if not os.path.isdir(objects_dir):
        return index

    for filename in os.listdir(objects_dir):
        if not filename.endswith(".object_config.json"):
            continue
        model_id = filename[:-len(".object_config.json")]
        lower_model = model_id.lower()
        index[lower_model] = model_id

        # Add non-_4k alias if template uses _4k suffix.
        if lower_model.endswith("_4k"):
            alias = lower_model[:-3]
            if alias and alias not in index:
                index[alias] = model_id
    return index


def resolve_model_id_for_template(object_name: str, template_index: Dict[str, str]) -> str:
    """Resolve object image stem to an existing template model_id."""
    key = object_name.lower()
    if key in template_index:
        return template_index[key]

    key_4k = f"{key}_4k"
    if key_4k in template_index:
        return template_index[key_4k]

    # Fallback to original name, editor will try flexible matching.
    return object_name


def infer_object_profile(model_id: str) -> Dict[str, float]:
    """Infer placement profile from model keywords."""
    key = (model_id or "").lower()
    for keyword, profile in PROFILE_KEYWORDS.items():
        if keyword in key:
            return dict(profile)
    return dict(DEFAULT_OBJECT_PROFILE)


def load_sd_ovon_layout(
    scene_name: str, sd_ovon_layouts_dir: str
) -> Optional[Dict[str, Any]]:
    """Load layout from SD-OVON output."""
    if not sd_ovon_layouts_dir:
        return None
    
    # Search for SD-OVON layout file
    candidates = []
    if os.path.isdir(sd_ovon_layouts_dir):
        for f in os.listdir(sd_ovon_layouts_dir):
            if scene_name in f and f.endswith(".json"):
                candidates.append(os.path.join(sd_ovon_layouts_dir, f))
    
    if not candidates:
        print(f"[Warning] No SD-OVON layout found in {sd_ovon_layouts_dir}")
        return None
    
    # Use most recent
    layout_file = max(candidates, key=os.path.getmtime)
    
    try:
        with open(layout_file, "r", encoding="utf-8") as f:
            layout = json.load(f)
        print(f"[SD-OVON] Loaded layout from {layout_file}")
        return layout
    except Exception as e:
        print(f"[Error] Failed to load SD-OVON layout: {e}")
        return None


def _room_center_from_room(room: Dict[str, Any]) -> List[float]:
    """Extract a room center with a safe zero fallback."""
    center = room.get("room_center", [0.0, 0.0, 0.0])
    if not isinstance(center, list) or len(center) < 3:
        return [0.0, 0.0, 0.0]
    try:
        return [float(center[0]), float(center[1]), float(center[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def sd_ovon_to_sampling_layout(
    sd_ovon_layout: Dict[str, Any], template_index: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Convert SD-OVON layout to sampling layout format."""
    objects = []
    for obj in sd_ovon_layout.get("objects", []):
        sampling_obj = {
            "id": obj.get("id", f"obj_{len(objects)}"),
            "category": obj.get("category", "object"),
            "position": obj.get("position", [0, 0, 0]),
            "rotation": obj.get("rotation", [0, 0, 0]),
            "confidence": obj.get("confidence", 1.0),
            "source": "sd_ovon",
        }
        objects.append(sampling_obj)
    
    return objects


def run_sd_ovon_roomwise_bridge(scene_name: str, layout_json: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge HM3D sampled room groups into the SD-OVON room-wise pipeline."""
    from orchestrate_sd_ovon_complete import SDOVONPipelineOrchestrator

    orchestrator = SDOVONPipelineOrchestrator(config_level="mock")
    return orchestrator.run_roomwise_layout_pipeline(layout_json, scene_name)



def load_scene_rooms(scene_name: str, rooms_info_dir: str) -> List[Dict[str, Any]]:
    """Load scene room metadata from exported scene_info files."""
    candidates = [
        os.path.join(rooms_info_dir, scene_name, f"{scene_name}_scene_info.json"),
        os.path.join(rooms_info_dir, "temp_export", f"{scene_name}_scene_info.json"),
    ]

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rooms = data.get("rooms", [])
            if isinstance(rooms, list):
                return rooms
        except Exception:
            continue
    return []


def _normalize_bbox(raw_bbox: Any) -> Optional[Tuple[List[float], List[float]]]:
    """Normalize bbox to ([min_x, min_y, min_z], [max_x, max_y, max_z])."""
    if not isinstance(raw_bbox, dict):
        return None
    min_pt = raw_bbox.get("min")
    max_pt = raw_bbox.get("max")
    if not isinstance(min_pt, list) or not isinstance(max_pt, list):
        return None
    if len(min_pt) < 3 or len(max_pt) < 3:
        return None

    try:
        min_pt = [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])]
        max_pt = [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])]
    except Exception:
        return None

    if any((not math.isfinite(v)) for v in min_pt + max_pt):
        return None
    if min_pt[0] > max_pt[0] or min_pt[1] > max_pt[1] or min_pt[2] > max_pt[2]:
        return None
    return min_pt, max_pt


def _random_point_in_bbox(
    min_pt: List[float],
    max_pt: List[float],
    y_offset: float,
    safety_margin: float = 0.12,
    preferred_y: Optional[float] = None,
    global_y_lift: float = 0.0,
) -> List[float]:
    """Sample a random point in bbox with margin and robust Y anchor."""
    def _sample_1d(lo: float, hi: float) -> float:
        if hi - lo <= 1e-6:
            return lo
        inner_lo = lo + safety_margin
        inner_hi = hi - safety_margin
        if inner_hi <= inner_lo:
            return float((lo + hi) / 2.0)
        return float(np.random.uniform(inner_lo, inner_hi))

    x = _sample_1d(min_pt[0], max_pt[0])
    z = _sample_1d(min_pt[2], max_pt[2])
    base_y = float(min_pt[1])
    if preferred_y is not None and math.isfinite(float(preferred_y)):
        base_y = float(preferred_y)
    y = float(base_y + max(y_offset, 0.02) + max(float(global_y_lift), 0.0))
    y = min(max(y, float(min_pt[1]) + 0.01), float(max_pt[1]))
    return [x, y, z]


def _random_point_near_center(
    center: List[float],
    y_offset: float,
    jitter: float = 0.45,
    global_y_lift: float = 0.0,
) -> List[float]:
    """Fallback point sampler when room bbox is unavailable."""
    if not isinstance(center, list) or len(center) < 3:
        center = [0.0, 0.0, 0.0]
    x = float(center[0] + np.random.uniform(-jitter, jitter))
    z = float(center[2] + np.random.uniform(-jitter, jitter))
    y = float(center[1] + max(y_offset, 0.02) + max(float(global_y_lift), 0.0))
    return [x, y, z]


def _collides_xz(candidate: List[float], candidate_radius: float, placed: List[Dict[str, Any]]) -> bool:
    """Simple XZ collision check with stored placed radii."""
    for obj in placed:
        other_pos = obj.get("position", [0.0, 0.0, 0.0])
        other_radius = float(obj.get("_placement_radius", DEFAULT_OBJECT_PROFILE["radius"]))
        dx = float(candidate[0]) - float(other_pos[0])
        dz = float(candidate[2]) - float(other_pos[2])
        min_dist = float(candidate_radius + other_radius)
        if (dx * dx + dz * dz) < (min_dist * min_dist):
            return True
    return False


def _distance_to_bbox_edge(point: List[float], min_pt: List[float], max_pt: List[float]) -> float:
    """Approximate distance from point to bbox inner edge on XZ plane."""
    dx = min(float(point[0]) - float(min_pt[0]), float(max_pt[0]) - float(point[0]))
    dz = min(float(point[2]) - float(min_pt[2]), float(max_pt[2]) - float(point[2]))
    return max(0.0, min(dx, dz))


def _score_candidate_rule(
    candidate: List[float],
    center: List[float],
    bbox_pair: Optional[Tuple[List[float], List[float]]],
) -> float:
    """Rule-based score: prefer near-center and away from boundary."""
    center_dist = math.sqrt((float(candidate[0]) - float(center[0])) ** 2 + (float(candidate[2]) - float(center[2])) ** 2)
    score = -center_dist
    if bbox_pair is not None:
        edge_margin = _distance_to_bbox_edge(candidate, bbox_pair[0], bbox_pair[1])
        score += 0.35 * edge_margin
    return float(score)


def _score_candidate_by_backend(
    backend: str,
    candidate: List[float],
    center: List[float],
    bbox_pair: Optional[Tuple[List[float], List[float]]],
) -> float:
    """
    Unified candidate scoring hook for future model integration.

    backend=rerank/policy currently falls back to rule scoring.
    Replace here to plug in external reranker/policy models.
    """
    if backend in ("rerank", "policy"):
        return _score_candidate_rule(candidate, center, bbox_pair)
    return _score_candidate_rule(candidate, center, bbox_pair)


def auto_place_objects(
    layout_json: Dict[str, Any],
    rooms: List[Dict[str, Any]],
    max_attempts: int,
    placement_backend: str = "rule",
    collision_radius_override: Optional[float] = None,
    global_y_lift: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Automatically place objects room-by-room using each object's sampled room."""
    start_time = time.time()
    room_bbox_map: Dict[int, Tuple[List[float], List[float]]] = {}
    room_center_map: Dict[int, List[float]] = {}
    room_order: List[int] = []
    for room in rooms:
        rid = room.get("region_id")
        if rid is None:
            continue
        rid_int = int(rid)
        room_order.append(rid_int)
        room_center_map[rid_int] = _room_center_from_room(room)
        normalized = _normalize_bbox(room.get("bounding_box", {}))
        if normalized is not None:
            room_bbox_map[rid_int] = normalized

    source_objects = layout_json.get("objects", [])
    placed_objects: List[Dict[str, Any]] = []
    failed_objects: List[Dict[str, Any]] = []
    failed_ids: List[int] = []
    fallback_to_center_count = 0
    per_room_stats: Dict[str, Dict[str, int]] = {}

    room_grouped_indices: Dict[int, List[int]] = {}
    unknown_indices: List[int] = []
    for idx, src in enumerate(source_objects):
        region_id = src.get("sampled_region_id")
        if isinstance(region_id, int) and region_id in room_center_map:
            room_grouped_indices.setdefault(region_id, []).append(idx)
        else:
            unknown_indices.append(idx)

    processing_order_indices: List[int] = []
    for rid in room_order:
        processing_order_indices.extend(room_grouped_indices.get(rid, []))
    processing_order_indices.extend(unknown_indices)
    object_processing_sequence: List[Any] = [source_objects[idx].get("id", idx) for idx in processing_order_indices]

    def _place_single_object(src: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal fallback_to_center_count

        obj = dict(src)
        profile = infer_object_profile(obj.get("model_id", ""))
        radius = float(collision_radius_override) if collision_radius_override else float(profile["radius"])
        y_offset = float(profile["y_offset"])

        region_id = obj.get("sampled_region_id")
        room_key = str(region_id) if isinstance(region_id, int) else "unknown"
        if room_key not in per_room_stats:
            per_room_stats[room_key] = {"total": 0, "placed": 0, "failed": 0}
        per_room_stats[room_key]["total"] += 1

        bbox_pair: Optional[Tuple[List[float], List[float]]] = None
        room_center = obj.get("position", [0.0, 0.0, 0.0])
        if isinstance(region_id, int) and region_id in room_bbox_map:
            bbox_pair = room_bbox_map[region_id]
            room_center = room_center_map.get(region_id, room_center)
        elif isinstance(region_id, int) and region_id in room_center_map:
            room_center = room_center_map[region_id]

        if bbox_pair is None:
            bbox_pair = _normalize_bbox(obj.get("room_aabb", {}))
        if bbox_pair is None and isinstance(region_id, int) and region_id in room_center_map:
            fallback_to_center_count += 1

        placed = False
        best_candidate = obj.get("position", [0.0, 0.0, 0.0])
        best_score = float("-inf")
        last_candidate = obj.get("position", [0.0, 0.0, 0.0])
        collision_hits = 0
        center = room_center
        if bbox_pair is not None:
            center = [
                (float(bbox_pair[0][0]) + float(bbox_pair[1][0])) / 2.0,
                (float(bbox_pair[0][1]) + float(bbox_pair[1][1])) / 2.0,
                (float(bbox_pair[0][2]) + float(bbox_pair[1][2])) / 2.0,
            ]

        for _ in range(max(1, int(max_attempts))):
            if bbox_pair is not None:
                candidate = _random_point_in_bbox(
                    bbox_pair[0],
                    bbox_pair[1],
                    y_offset,
                    preferred_y=float(room_center[1]),
                    global_y_lift=global_y_lift,
                )
            else:
                candidate = _random_point_near_center(
                    room_center,
                    y_offset,
                    global_y_lift=global_y_lift,
                )

            last_candidate = candidate
            if _collides_xz(candidate, radius, placed_objects):
                collision_hits += 1
                continue

            score = _score_candidate_by_backend(placement_backend, candidate, center, bbox_pair)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_score > float("-inf"):
            candidate = best_candidate
            obj["position"] = [round(float(candidate[0]), 4), round(float(candidate[1]), 4), round(float(candidate[2]), 4)]
            obj["rotation"] = [0.0, float(np.random.uniform(0.0, 360.0)), 0.0]
            obj["source"] = "auto_placement"
            obj["_placement_radius"] = radius
            obj["edit_priority"] = "normal"
            obj["edit_status"] = "auto_placed"
            placed = True

        if placed:
            per_room_stats[room_key]["placed"] += 1
            placed_objects.append(obj)
        else:
            per_room_stats[room_key]["failed"] += 1
            obj["source"] = "auto_placement_failed"
            obj["_placement_radius"] = radius
            obj["edit_priority"] = "high"
            obj["edit_status"] = "needs_manual_fix"
            if bbox_pair is None:
                fail_reason = "no_valid_bbox_fallback_collision"
            else:
                fail_reason = "collision_after_retries"
            obj["edit_note"] = f"AUTO_FAILED: {fail_reason}"
            failed_objects.append(
                {
                    "id": obj.get("id"),
                    "model_id": obj.get("model_id"),
                    "sampled_region_id": obj.get("sampled_region_id", -1),
                    "reason": fail_reason,
                    "room_center": [round(float(room_center[0]), 4), round(float(room_center[1]), 4), round(float(room_center[2]), 4)],
                    "target_center": [round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)],
                    "last_candidate": [round(float(last_candidate[0]), 4), round(float(last_candidate[1]), 4), round(float(last_candidate[2]), 4)],
                    "attempts": int(max(1, int(max_attempts))),
                    "collision_hits": int(collision_hits),
                }
            )
            if isinstance(obj.get("id"), int):
                failed_ids.append(int(obj["id"]))
            placed_objects.append(obj)

        return obj

    for idx in processing_order_indices:
        _place_single_object(source_objects[idx])

    placed_objects.sort(key=lambda x: 0 if x.get("edit_status") == "needs_manual_fix" else 1)

    for obj in placed_objects:
        obj.pop("_placement_radius", None)

    duration = round(float(time.time() - start_time), 4)
    failed_by_reason: Dict[str, int] = {}
    for item in failed_objects:
        reason = str(item.get("reason", "unknown"))
        failed_by_reason[reason] = failed_by_reason.get(reason, 0) + 1

    stats = {
        "total_objects": len(source_objects),
        "placed_count": len(source_objects) - len(failed_objects),
        "failed_count": len(failed_objects),
        "processing_time_sec": duration,
        "placement_backend": placement_backend,
        "global_y_lift": float(global_y_lift),
        "center_fallback_count": fallback_to_center_count,
        "room_processing_order": room_order,
        "object_processing_sequence": object_processing_sequence,
        "room_processing": per_room_stats,
        "failed_by_reason": failed_by_reason,
        "failed_ids": failed_ids,
        "failed_objects": failed_objects,
    }

    new_layout = dict(layout_json)
    new_layout["objects"] = placed_objects
    new_layout["auto_placement_stats"] = stats
    new_layout["editor_focus_failed_ids"] = failed_ids
    new_layout["editor_review_order"] = "failed_first"
    return new_layout, stats


# ===== 概率文件管理 =====

def generate_probabilities(
    object_name: str,
    scene_name: str,
    rooms_info_dir: str,
    probabilities_dir: str,
):
    """
    Load room recommendations from query result and generate random probabilities.
    """
    # Load room recommendations
    rooms_info_path = os.path.join(rooms_info_dir, scene_name, f"{object_name}_rooms.json")
    if not os.path.isfile(rooms_info_path):
        print(f"  [Warning] Room info not found: {rooms_info_path}")
        return None
    
    try:
        with open(rooms_info_path, "r", encoding="utf-8") as f:
            rooms_info = json.load(f)
    except Exception as e:
        print(f"  [Error] Failed to load room info: {e}")
        return None
    
    recommended_rooms = rooms_info.get("recommended_rooms", [])
    if not recommended_rooms:
        print(f"  [Warning] No recommended rooms found for {object_name}")
        return None

    # 允许推荐房间少于5，但至少需要2个。
    if len(recommended_rooms) < 2:
        print(f"  [Warning] Recommended rooms < 2 for {object_name}, skipping")
        return None
    
    # Generate random probabilities for top-N rooms, where N in [2, 5]
    n_rooms = min(5, len(recommended_rooms))
    raw_probs = np.random.rand(n_rooms)
    normalized_probs = raw_probs / raw_probs.sum()
    
    probabilities_data = {
        "object_name": object_name,
        "scene_name": scene_name,
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probabilities": [
            {
                "rank": room["rank"],
                "region_id": room.get("region_id", -1),
                "room_center": room.get("room_center", [0, 0, 0]),
                "room_aabb": room.get("room_aabb", room.get("bounding_box", {})),
                "probability": float(normalized_probs[i]),
            }
            for i, room in enumerate(recommended_rooms[:n_rooms])
        ],
    }
    
    # Ensure directory exists
    prob_scene_dir = os.path.join(probabilities_dir, scene_name)
    os.makedirs(prob_scene_dir, exist_ok=True)
    
    # Save probabilities
    prob_path = os.path.join(prob_scene_dir, f"{object_name}_probs.json")
    try:
        with open(prob_path, "w", encoding="utf-8") as f:
            json.dump(probabilities_data, f, ensure_ascii=False, indent=2)
        print(f"    ✓ Generated probabilities: {prob_path}")
        return probabilities_data
    except Exception as e:
        print(f"  [Error] Failed to save probabilities: {e}")
        return None


def load_probabilities(object_name: str, scene_name: str, probabilities_dir: str) -> Optional[Dict]:
    """Load pre-generated probabilities."""
    prob_path = os.path.join(probabilities_dir, scene_name, f"{object_name}_probs.json")
    if not os.path.isfile(prob_path):
        print(f"  [Warning] Probability file not found: {prob_path}")
        return None
    
    try:
        with open(prob_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            probs = data.get("probabilities", [])
            if len(probs) < 2:
                print(f"  [Warning] Probability entries < 2: {prob_path}")
                return None
            return data
    except Exception as e:
        print(f"  [Error] Failed to load probabilities: {e}")
        return None


def get_or_create_probabilities(
    object_name: str,
    scene_name: str,
    mode: str,
    rooms_info_dir: str,
    probabilities_dir: str,
) -> Optional[Dict]:
    """Get probabilities: load if exists and mode=load, or generate if mode=generate."""
    if mode == "load":
        return load_probabilities(object_name, scene_name, probabilities_dir)
    elif mode == "generate":
        return generate_probabilities(
            object_name,
            scene_name,
            rooms_info_dir,
            probabilities_dir,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


# ===== 采样与布局生成 =====

def sample_object_positions(
    scene_name: str,
    images_dir: str,
    mode: str,
    rooms_info_dir: str,
    probabilities_dir: str,
) -> Optional[Dict]:
    """
    Sample an object position for each image based on probabilities.
    
    Returns:
        {
            "scene": "...",
            "timestamp": "...",
            "objects": [
                {
                    "id": int,
                    "model_id": str,
                    "name": str,
                    "position": [x, y, z],
                    "rotation": [0, 0, 0],
                    "confidence": float,
                    "source": "probability_sampling"
                },
                ...
            ]
        }
    """
    
    # Scan images
    image_files = []
    if os.path.isdir(images_dir):
        for ext in ["*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(Path(images_dir).glob(ext))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"[Warning] No image files found in {images_dir}")
        return None

    template_index = build_object_template_index("./objects")
    
    sampled_objects = []
    for obj_idx, image_path in enumerate(image_files):
        object_name = image_path.stem
        
        # Get/create probabilities
        probs_data = get_or_create_probabilities(
            object_name,
            scene_name,
            mode,
            rooms_info_dir,
            probabilities_dir,
        )
        if not probs_data:
            print(f"  [Warning] Cannot sample for {object_name}, skipping")
            continue
        
        probabilities = probs_data.get("probabilities", [])
        if not probabilities:
            print(f"  [Warning] No probabilities for {object_name}")
            continue
        if len(probabilities) < 2:
            print(f"  [Warning] Probabilities less than 2 for {object_name}, skipping")
            continue
        
        # Extract room centers and probability weights
        room_centers = [p.get("room_center", [0, 0, 0]) for p in probabilities]
        prob_weights = np.array([p.get("probability", 0) for p in probabilities])
        
        # Normalize if needed
        if prob_weights.sum() > 0:
            prob_weights = prob_weights / prob_weights.sum()
        else:
            prob_weights = np.ones(len(probabilities)) / len(probabilities)
        
        # Sample one room
        sampled_idx = np.random.choice(len(probabilities), p=prob_weights)
        sampled_room = probabilities[sampled_idx]
        sampled_center = sampled_room.get("room_center", [0, 0, 0])
        sampled_confidence = sampled_room.get("probability", 0.5)
        sampled_region_id = int(sampled_room.get("region_id", -1))
        room_aabb = sampled_room.get("room_aabb", {})

        resolved_model_id = resolve_model_id_for_template(object_name, template_index)
        
        # Create object entry
        obj_entry = {
            "id": obj_idx,
            "model_id": resolved_model_id,
            "name": _prettify_model_name(resolved_model_id),
            "position": sampled_center,
            "rotation": [0.0, 0.0, 0.0],
            "confidence": float(sampled_confidence),
            "sampled_region_id": sampled_region_id,
            "room_aabb": room_aabb,
            "source": "probability_sampling",
        }
        sampled_objects.append(obj_entry)
        print(
            f"  ✓ Sampled {object_name} -> {resolved_model_id}: "
            f"room={sampled_room.get('region_id')}, center={sampled_center}"
        )
    
    if not sampled_objects:
        print("[Error] No objects sampled")
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True)
    if scene_paths is None:
        print(f"[Error] Scene not found in merged hm3d splits or missing semantic.txt: {scene_name}")
        return None
    
    # Build layout JSON
    layout_json = {
        "scene": str(scene_paths.stage_glb),
        "timestamp": time.time(),
        "objects": sampled_objects,
        "room_groups": _build_room_groups(sampled_objects),
        "placement_scope": {
            "mode": "sampled_region_per_object",
        },
    }
    
    return layout_json


def _prettify_model_name(model_id: str) -> str:
    """Convert snake_case or camelCase to Title Case."""
    return model_id.replace("_", " ").title()


def _build_room_groups(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group sampled objects by sampled_region_id for downstream room-wise placement."""
    grouped: Dict[int, Dict[str, Any]] = {}
    unknown_group: Optional[Dict[str, Any]] = None

    for obj in objects:
        region_id = obj.get("sampled_region_id")
        if isinstance(region_id, int) and region_id >= 0:
            group = grouped.setdefault(
                region_id,
                {
                    "region_id": region_id,
                    "objects": [],
                    "room_center": obj.get("room_center", obj.get("position", [0.0, 0.0, 0.0])),
                    "room_aabb": obj.get("room_aabb", {}),
                },
            )
            group["objects"].append(obj)
            if obj.get("room_aabb"):
                group["room_aabb"] = obj.get("room_aabb", {})
        else:
            if unknown_group is None:
                unknown_group = {
                    "region_id": -1,
                    "objects": [],
                    "room_center": [0.0, 0.0, 0.0],
                    "room_aabb": {},
                }
            unknown_group["objects"].append(obj)

    room_groups = [grouped[key] for key in sorted(grouped.keys())]
    if unknown_group is not None and unknown_group["objects"]:
        room_groups.append(unknown_group)
    return room_groups


def _get_scene_id(scene_name: str) -> str:
    """Extract scene ID from scene name (e.g., 00800-TEEsavR23oF -> TEEsavR23oF)."""
    parts = scene_name.split("-", 1)
    return parts[1] if len(parts) > 1 else scene_name


# ===== 编辑器集成 =====

def launch_editor(scene_name: str, layout_json_path: str, ui_lang: str = "zh") -> bool:
    """
    Launch test_layout.py with the sampled layout.
    
    Returns:
        True if editor exited successfully, False otherwise.
    """
    # Determine scene ID from scene_name
    scene_id = _get_scene_id(scene_name)
    
    layout_json_path = os.path.abspath(layout_json_path)

    cmd = [
        sys.executable,
        "test_layout.py",
        scene_name,
        "--layout", layout_json_path,
        "--ui-lang", ui_lang,
    ]
    
    print(f"\n[Info] Launching editor: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"[Error] Failed to launch editor: {e}")
        return False


# ===== 主流程 =====

def interactive_sampling_loop(
    scene_name: str,
    images_dir: str,
    mode: str,
    rooms_info_dir: str,
    probabilities_dir: str,
    layouts_dir: str,
    placement: str,
    placement_backend: str,
    placement_attempts: int,
    collision_radius_override: Optional[float],
    global_y_lift: float,
    ui_lang: str = "zh",
):
    """
    Main loop: sample -> edit -> save -> repeat.
    """
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        
        # Step 1: Sample object positions
        print(f"\n[Step 1] Sampling object positions from probabilities...")
        layout_json = sample_object_positions(
            scene_name,
            images_dir,
            mode,
            rooms_info_dir,
            probabilities_dir,
        )
        if not layout_json:
            print("[Error] Failed to sample layout")
            return

        if placement == "auto":
            print("[Info] Running auto placement (quality-first)...")
            rooms = load_scene_rooms(scene_name, rooms_info_dir)
            layout_json, auto_stats = auto_place_objects(
                layout_json,
                rooms,
                max_attempts=placement_attempts,
                placement_backend=placement_backend,
                collision_radius_override=collision_radius_override,
                global_y_lift=global_y_lift,
            )
            print(
                "[Info] Auto placement stats: "
                f"placed={auto_stats['placed_count']}/{auto_stats['total_objects']}, "
                f"failed={auto_stats['failed_count']}, "
                f"backend={auto_stats['placement_backend']}, "
                f"time={auto_stats['processing_time_sec']}s"
            )

            placed_positions = [obj.get("position", [0.0, 0.0, 0.0]) for obj in layout_json.get("objects", [])]
            y_values = []
            for pos in placed_positions:
                if isinstance(pos, list) and len(pos) >= 3:
                    try:
                        y_values.append(float(pos[1]))
                    except Exception:
                        continue
            if y_values:
                y_min = min(y_values)
                y_max = max(y_values)
                y_avg = sum(y_values) / len(y_values)
                below_zero = sum(1 for y in y_values if y < 0.0)
                print(
                    "[Diag] Placement Y summary: "
                    f"min={y_min:.4f}, max={y_max:.4f}, avg={y_avg:.4f}, below_0={below_zero}/{len(y_values)}"
                )
                if below_zero > (len(y_values) // 2) and global_y_lift <= 0.0:
                    print(
                        "[Diag] Many objects are below y=0. "
                        "Try adding --global-y-lift 0.4 (or 0.6) to raise all auto placements."
                    )

            if auto_stats["failed_count"] > 0:
                failed_models = [x.get("model_id", "?") for x in auto_stats.get("failed_objects", [])]
                print("[Info] Failed objects to focus in editor:", ", ".join(failed_models))
                print("[Info] Failed object IDs:", auto_stats.get("failed_ids", []))
                print("[Diag] Failed by reason:", auto_stats.get("failed_by_reason", {}))
                for idx, fo in enumerate(auto_stats.get("failed_objects", []), start=1):
                    print(
                        "[Diag] Failed#{idx}: id={id} model={model} room={room} reason={reason} "
                        "target_center={target} last_candidate={last} collision_hits={hits}/{attempts}".format(
                            idx=idx,
                            id=fo.get("id"),
                            model=fo.get("model_id"),
                            room=fo.get("sampled_region_id"),
                            reason=fo.get("reason"),
                            target=fo.get("target_center"),
                            last=fo.get("last_candidate"),
                            hits=fo.get("collision_hits"),
                            attempts=fo.get("attempts"),
                        )
                    )
        
        # Step 2: Write temporary layout file
        prefix = "temp_auto" if placement == "auto" else "temp_sampled"
        temp_layout_path = os.path.join(layouts_dir, scene_name, f"{prefix}_{int(time.time())}.json")
        os.makedirs(os.path.dirname(temp_layout_path), exist_ok=True)
        
        try:
            with open(temp_layout_path, "w", encoding="utf-8") as f:
                json.dump(layout_json, f, ensure_ascii=False, indent=2)
            print(f"[Info] Saved sampled layout to: {temp_layout_path}")
        except Exception as e:
            print(f"[Error] Failed to save layout: {e}")
            return
        
        # Step 3: Launch editor
        print(f"\n[Step 2] Launching interactive editor...")
        print(f"  You can now adjust object positions manually.")
        print(f"  Press [M] in the editor to save, then close the window.")
        
        success = launch_editor(scene_name, temp_layout_path, ui_lang)
        if not success:
            print("[Warning] Editor exited with error or was closed")
        
        # Step 4: User decision
        print(f"\n[Step 3] Save final layout?")
        while True:
            user_input = input("Save to final layout? (y/n/q): ").strip().lower()
            if user_input in ("y", "yes"):
                # Find the latest saved layout (should be overwritten by editor)
                scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
                scene_configs_dir = os.path.join(str(scene_paths.scene_dir), "configs") if scene_paths else os.path.join("hm3d", "minival", scene_name, "configs")
                if os.path.isdir(scene_configs_dir):
                    # Get the most recently modified JSON
                    json_files = list(Path(scene_configs_dir).glob("*.json"))
                    if json_files:
                        latest_layout = max(json_files, key=lambda p: p.stat().st_mtime)
                        
                        # Copy to final layouts directory
                        final_layout_dir = os.path.join(layouts_dir, scene_name)
                        os.makedirs(final_layout_dir, exist_ok=True)
                        
                        final_name = f"final_{int(time.time())}.json"
                        final_path = os.path.join(final_layout_dir, final_name)
                        
                        try:
                            with open(latest_layout, "r") as src:
                                data = json.load(src)
                            with open(final_path, "w") as dst:
                                json.dump(data, dst, ensure_ascii=False, indent=2)
                            print(f"✓ Saved final layout to: {final_path}")
                        except Exception as e:
                            print(f"[Error] Failed to save final layout: {e}")
                
                break
            elif user_input in ("n", "no"):
                break
            elif user_input in ("q", "quit"):
                print("Exiting...")
                return
            else:
                print("Please enter y, n, or q")
        
        # Step 5: Ask to continue
        print(f"\n[Step 4] Continue to next iteration?")
        while True:
            user_input = input("Continue sampling and editing? (y/n): ").strip().lower()
            if user_input in ("y", "yes"):
                continue  # Go to next iteration
            elif user_input in ("n", "no"):
                print("Done!")
                return
            else:
                print("Please enter y or n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample object positions based on probabilities and edit in interactive layout editor"
    )
    
    parser.add_argument("--scene", required=True, help="Scene name (e.g., 00800-TEEsavR23oF)")
    parser.add_argument(
        "--mode",
        choices=["load", "generate"],
        default="load",
        help="load: use existing probabilities; generate: create new random probabilities",
    )
    
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Object images directory")
    parser.add_argument("--rooms-info-dir", default=DEFAULT_ROOMS_INFO_DIR, help="Room query results directory")
    parser.add_argument("--probabilities-dir", default=DEFAULT_PROBABILITIES_DIR, help="Probabilities directory")
    parser.add_argument("--layouts-dir", default=DEFAULT_LAYOUTS_DIR, help="Layouts output directory")
    parser.add_argument(
        "--placement",
        choices=["manual", "auto"],
        default="manual",
        help="manual: sample then edit; auto: collision-aware auto placement then edit failed objects",
    )
    parser.add_argument(
        "--placement-backend",
        choices=["rule", "rerank", "policy"],
        default="rule",
        help="Auto placement backend. rerank/policy currently use rule fallback hook",
    )
    parser.add_argument(
        "--placement-attempts",
        type=int,
        default=24,
        help="Max retry attempts per object in auto placement",
    )
    parser.add_argument(
        "--collision-radius-override",
        type=float,
        default=None,
        help="Optional fixed collision radius for all objects in auto placement",
    )
    parser.add_argument(
        "--global-y-lift",
        type=float,
        default=0.0,
        help="Global Y lift added to all auto placement candidates to avoid sinking",
    )
    parser.add_argument(
        "--placement-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and auto placement",
    )
    
    parser.add_argument(
        "--backend",
        choices=["default", "sd_ovon"],
        default="default",
        help="Sampling backend: default (rule-based) or sd_ovon (from SD-OVON pipeline)",
    )
    parser.add_argument(
        "--sd-ovon-layouts-dir",
        default=None,
        help="SD-OVON layouts directory (if --backend sd_ovon)",
    )
    
    parser.add_argument("--ui-lang", choices=["en", "zh"], default="zh", help="Editor UI language")
    
    args = parser.parse_args()
    np.random.seed(args.placement_seed)
    
    # Validate scene
    if args.scene not in AVAILABLE_SCENES:
        print(f"[Error] Scene {args.scene} not in merged available list")
        sys.exit(1)
    
    # Handle SD-OVON backend
    if args.backend == "sd_ovon":
        print(f"[Info] Using SD-OVON backend")
        sd_ovon_layout = load_sd_ovon_layout(args.scene, args.sd_ovon_layouts_dir)
        if sd_ovon_layout is None:
            print(f"[Error] Failed to load SD-OVON layout for {args.scene}")
            sys.exit(1)

        roomwise_report = run_sd_ovon_roomwise_bridge(args.scene, sd_ovon_layout)
        print(
            "[Info] SD-OVON roomwise bridge completed: "
            f"rooms={roomwise_report.get('room_count', 0)}, "
            f"objects={roomwise_report.get('object_count', 0)}, "
            f"status={roomwise_report.get('pipeline_status')}"
        )
        
        # Save roomwise report as intermediate layout
        layouts_dir = args.layouts_dir
        layout_json = {
            "scene": str(args.scene),
            "timestamp": time.time(),
            "objects": roomwise_report.get("placements", []),
            "room_groups": roomwise_report.get("room_groups", []),
            "placement_scope": {
                "mode": "roomwise_from_sd_ovon",
                "bridge_status": roomwise_report.get("pipeline_status"),
            },
        }
        
        os.makedirs(os.path.join(layouts_dir, args.scene), exist_ok=True)
        temp_layout_path = os.path.join(layouts_dir, args.scene, f"roomwise_sd_ovon_{int(time.time())}.json")
        try:
            with open(temp_layout_path, "w", encoding="utf-8") as f:
                json.dump(layout_json, f, ensure_ascii=False, indent=2)
            print(f"[Info] Saved roomwise layout to: {temp_layout_path}")
        except Exception as e:
            print(f"[Error] Failed to save roomwise layout: {e}")
            sys.exit(1)
        
        print(f"[Info] SD-OVON roomwise backend complete. Layout saved.")
        return
    
    # Default backend validation
    if args.mode == "load":
        # Check if at least one probability file exists
        prob_dir = os.path.join(args.probabilities_dir, args.scene)
        if not os.path.isdir(prob_dir):
            print(f"[Error] Probability directory not found: {prob_dir}")
            print(f"        Use --mode generate to create probabilities first")
            sys.exit(1)
    elif args.mode == "generate":
        # Check if room info files exist
        rooms_dir = os.path.join(args.rooms_info_dir, args.scene)
        if not os.path.isdir(rooms_dir):
            print(f"[Error] Room info directory not found: {rooms_dir}")
            print(f"        Run query_rooms_for_objects.py first")
            sys.exit(1)
    
    # Run interactive loop
    try:
        interactive_sampling_loop(
            args.scene,
            args.images_dir,
            args.mode,
            args.rooms_info_dir,
            args.probabilities_dir,
            args.layouts_dir,
            args.placement,
            args.placement_backend,
            args.placement_attempts,
            args.collision_radius_override,
            args.global_y_lift,
            args.ui_lang,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)



if __name__ == "__main__":
    main()
