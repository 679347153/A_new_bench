#!/usr/bin/env python3
from __future__ import annotations

"""
基于物理约束的实例上表面放置（文件2）。

概述
----
本脚本读取“物体 -> 实例”分配结果并执行自动放置：
1) 从场景级上表面结果建立目标实例索引。
2) 对每个物体在目标实例上表面采样候选点。
3) 对可碰撞模板以 `surface_y + spawn_height + y_offset` 生成物理下落初始位置。
   对不可碰撞模板直接以 `surface_y + y_offset` 生成最终位置，避免被重力带到承载面下方。
4) 执行物体间最小距离约束。
5) 若 habitat-sim 可用：
   - 通过模板实例化刚体
   - 进行若干步物理稳定
   - 若与已放置物体发生接触则拒绝该候选
6) 导出最终布局 JSON。

碰撞策略
--------
- 几何预检：XZ 平面最小中心距约束。
- 物理接触检验：新物体与已放置物体接触则回退重试。
- 重试机制：每个物体最多尝试多个表面采样点（`max_trials_per_object`）。

执行指引
--------
1) 使用现成分配计划直接执行：
   python place_objects_on_instances.py \
     --scene 00808-y9hTuugGdiq \
     --assignment-plan results/object_instance_assignments/00808-y9hTuugGdiq/00808-y9hTuugGdiq_object_instance_plan.json \
     --surfaces-json results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json

2) 更严格间距 + 更多重试：
   python place_objects_on_instances.py \
     --scene 00808-y9hTuugGdiq \
     --assignment-plan <plan_json> \
     --surfaces-json <surfaces_json> \
     --min-distance 0.3 \
     --max-trials-per-object 60 \
     --settle-steps 80

3) 显式指定你要求的生成高度：
   python place_objects_on_instances.py \
     --scene 00808-y9hTuugGdiq \
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
from object_profiles import get_object_profile, surface_requirement
from project_paths import default_object_config_dirs_str, find_object_config_path, iter_object_config_dirs

try:
    import habitat_sim  # type: ignore[import-not-found]
except ImportError:
    habitat_sim = None

def _safe_float(value: Any, default: float = 0.0) -> float:
    """尽力转换为 float，失败时返回确定性的默认值。"""
    try:
        return float(value)
    except Exception:
        return default


def _get_profile(model_id: str, objects_dir: str = "") -> Dict[str, Any]:
    """
    获取物体几何/放置 profile。

    优先读取 `object_profiles.json` 手工覆盖，其次读取模板配置中的碰撞字段，
    最后回退到关键词估计。
    """
    return get_object_profile(model_id, objects_dir=objects_dir)


def _resolve_template_handle(template_mgr: Any, model_id: str) -> Optional[str]:
    """
    通过多个别名规则解析 habitat 物体模板句柄。

    支持直接 id、`.object_config.json`、`_4k` 等变体。
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
    """安全移除临时或失败对象，不向上抛异常。"""
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


def _vec3_to_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        pass
    try:
        return [float(value.x), float(value.y), float(value.z)]
    except Exception:
        return None


def _bbox_min_max(bbox: Any) -> Optional[Tuple[List[float], List[float]]]:
    if bbox is None:
        return None
    min_vec = None
    max_vec = None
    for key in ("min", "min_", "back_bottom_left"):
        try:
            attr = getattr(bbox, key)
            attr = attr() if callable(attr) else attr
            min_vec = _vec3_to_list(attr)
            if min_vec is not None:
                break
        except Exception:
            continue
    for key in ("max", "max_", "front_top_right"):
        try:
            attr = getattr(bbox, key)
            attr = attr() if callable(attr) else attr
            max_vec = _vec3_to_list(attr)
            if max_vec is not None:
                break
        except Exception:
            continue
    if min_vec is None or max_vec is None:
        return None
    if any((not np.isfinite(v)) for v in min_vec + max_vec):
        return None
    if min_vec[0] > max_vec[0] or min_vec[1] > max_vec[1] or min_vec[2] > max_vec[2]:
        return None
    return min_vec, max_vec


def _object_bbox_min_max(obj: Any) -> Optional[Tuple[List[float], List[float]]]:
    for attr_path in (
        ("aabb",),
        ("root_scene_node", "cumulative_bb"),
        ("visual_scene_node", "cumulative_bb"),
    ):
        try:
            cur = obj
            for attr in attr_path:
                cur = getattr(cur, attr)
                cur = cur() if callable(cur) else cur
            pair = _bbox_min_max(cur)
            if pair is not None:
                return pair
        except Exception:
            continue
    return None


def _profile_from_sim_object(obj: Any, base_profile: Dict[str, Any]) -> Dict[str, Any]:
    pair = _object_bbox_min_max(obj)
    if pair is None:
        return dict(base_profile)
    bmin, bmax = pair
    pos = _vec3_to_list(getattr(obj, "translation", None)) or [0.0, 0.0, 0.0]
    sx = max(0.0, float(bmax[0]) - float(bmin[0]))
    sy = max(0.0, float(bmax[1]) - float(bmin[1]))
    sz = max(0.0, float(bmax[2]) - float(bmin[2]))
    if sx <= 1e-4 or sy <= 1e-4 or sz <= 1e-4:
        return dict(base_profile)
    out = dict(base_profile)
    out["radius"] = round(max(float(out.get("radius", 0.2)), float(np.sqrt(sx * sx + sz * sz) * 0.5)), 4)
    out["footprint_x"] = round(sx, 4)
    out["footprint_z"] = round(sz, 4)
    out["height"] = round(sy, 4)
    out["y_offset"] = round(max(0.0, float(pos[1]) - float(bmin[1])), 4)
    out["profile_source"] = "habitat_runtime_aabb"
    return out


def _runtime_template_profile(
    rom: Any,
    template_handle: str,
    base_profile: Dict[str, Any],
) -> Dict[str, Any]:
    obj = None
    try:
        obj = rom.add_object_by_template_handle(template_handle)
        if obj is None:
            return dict(base_profile)
        try:
            obj.translation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        except Exception:
            pass
        return _profile_from_sim_object(obj, base_profile)
    except Exception:
        return dict(base_profile)
    finally:
        _remove_object_safe(rom, obj)


def _step_physics(sim: Any, steps: int) -> None:
    """多帧推进物理模拟，兼容不同版本 API。"""
    for _ in range(max(0, int(steps))):
        try:
            sim.step_physics(1.0 / 60.0)
        except TypeError:
            sim.step_physics()
        except Exception:
            break


def _contact_with_existing(sim: Any, candidate_object_id: int, existing_ids: Sequence[int]) -> bool:
    """
    检查候选对象是否与已放置对象发生物理接触。

    返回 True 表示检测到碰撞/接触。
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
    target_instance_id: Optional[int] = None,
    surface_height: Optional[float] = None,
    height_threshold: float = 0.25,
) -> bool:
    """检查 XZ 平面上的两两最小距离约束，允许不同高度层适度放宽。"""
    x = _safe_float(pos[0])
    z = _safe_float(pos[2])
    for item in placed:
        other_target = item.get("_target_instance_id")
        other_height = item.get("_surface_height")
        if target_instance_id is not None and other_target != target_instance_id:
            if surface_height is not None and other_height is not None:
                if abs(float(surface_height) - float(other_height)) > float(height_threshold):
                    continue
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
    """构建上表面快速索引：`(room_id, instance_id)` 与 `instance_id` 两级映射。"""
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
    """为单条分配记录选择匹配的上表面条目（先房间内匹配，再全局回退）。"""
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


def _candidate_surface_items(
    assignment: Dict[str, Any],
    by_room_instance: Dict[Tuple[int, int], Dict[str, Any]],
    by_instance: Dict[int, Dict[str, Any]],
) -> List[Tuple[int, Dict[str, Any], str]]:
    """Return target surface followed by backup surfaces, de-duplicated."""
    ids: List[Tuple[int, str]] = []
    try:
        ids.append((int(assignment.get("target_instance_id")), "target"))
    except Exception:
        pass
    backup_raw = assignment.get("backup_instance_ids", [])
    if isinstance(backup_raw, list):
        for item in backup_raw:
            try:
                ids.append((int(item), "backup"))
            except Exception:
                continue

    room_id = assignment.get("target_room_id", assignment.get("sampled_region_id", None))
    out: List[Tuple[int, Dict[str, Any], str]] = []
    seen = set()
    for instance_id, source in ids:
        if instance_id in seen:
            continue
        seen.add(instance_id)
        item = None
        if room_id is not None:
            try:
                item = by_room_instance.get((int(room_id), instance_id))
            except Exception:
                item = None
        if item is None:
            item = by_instance.get(instance_id)
        if item is not None:
            out.append((instance_id, item, source))
    return out


def _surface_bounds(surface_item: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    top = surface_item.get("top_surface", {}) if isinstance(surface_item, dict) else {}
    bounds = top.get("bounds", {}) if isinstance(top, dict) else {}
    bmin = bounds.get("min", [0.0, 0.0, 0.0]) if isinstance(bounds, dict) else [0.0, 0.0, 0.0]
    bmax = bounds.get("max", [0.0, 0.0, 0.0]) if isinstance(bounds, dict) else [0.0, 0.0, 0.0]
    if not isinstance(bmin, list) or not isinstance(bmax, list) or len(bmin) < 3 or len(bmax) < 3:
        return None
    return bmin[:3], bmax[:3]


def _surface_height(surface_item: Dict[str, Any]) -> float:
    top = surface_item.get("top_surface", {}) if isinstance(surface_item, dict) else {}
    if isinstance(top, dict):
        return _safe_float(top.get("plane_height"), 0.0)
    return 0.0


def _surface_fits_profile(surface_item: Dict[str, Any], profile: Dict[str, Any]) -> Tuple[bool, str]:
    bounds = _surface_bounds(surface_item)
    if bounds is None:
        return False, "invalid_surface_bounds"
    bmin, bmax = bounds
    span_x = max(0.0, _safe_float(bmax[0]) - _safe_float(bmin[0]))
    span_z = max(0.0, _safe_float(bmax[2]) - _safe_float(bmin[2]))
    area = span_x * span_z
    req = surface_requirement(profile)
    if area < float(req["required_area"]):
        return False, "surface_area_smaller_than_object"
    if span_x < float(req["required_min_span"]) or span_z < float(req["required_min_span"]):
        return False, "surface_span_smaller_than_object"
    return True, ""


def _sample_surface_points(
    points: List[List[float]],
    max_trials: int,
    rng: random.Random,
    edge_margin: float = 0.0,
) -> List[List[float]]:
    """Downsample surface points, preferring points away from surface edges."""
    clean = [p for p in points if isinstance(p, list) and len(p) >= 3]
    if not clean:
        return []
    margin = max(0.0, float(edge_margin))
    arr = np.asarray(clean, dtype=np.float32)[:, :3]
    bmin = arr.min(axis=0)
    bmax = arr.max(axis=0)
    eligible: List[Tuple[float, List[float]]] = []
    fallback: List[Tuple[float, List[float]]] = []
    for p in clean:
        x = _safe_float(p[0])
        z = _safe_float(p[2])
        edge_score = min(x - float(bmin[0]), float(bmax[0]) - x, z - float(bmin[2]), float(bmax[2]) - z)
        row = (edge_score + rng.random() * 1e-4, p)
        fallback.append(row)
        if x >= float(bmin[0]) + margin and x <= float(bmax[0]) - margin and z >= float(bmin[2]) + margin and z <= float(bmax[2]) - margin:
            eligible.append(row)
    pool = eligible if eligible else fallback
    pool.sort(key=lambda item: item[0], reverse=True)
    selected = [p for _, p in pool[: max(1, int(max_trials))]]
    rng.shuffle(selected)
    return selected


def _load_point_cloud_file(path: Path) -> np.ndarray:
    """Load point cloud from `.ply` (ascii) or `.xyz` file."""
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float32)
    suffix = path.suffix.lower()

    if suffix == ".xyz":
        pts: List[List[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except Exception:
                    continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return arr[:, :3]

    if suffix == ".ply":
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines or not lines[0].strip().lower().startswith("ply"):
            return np.zeros((0, 3), dtype=np.float32)
        vertex_count = 0
        header_end = -1
        for i, line in enumerate(lines):
            text = line.strip().lower()
            if text.startswith("element vertex"):
                parts = text.split()
                if len(parts) >= 3:
                    try:
                        vertex_count = int(parts[2])
                    except Exception:
                        vertex_count = 0
            if text == "end_header":
                header_end = i
                break
        if header_end < 0 or vertex_count <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        pts: List[List[float]] = []
        for line in lines[header_end + 1 : header_end + 1 + vertex_count]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except Exception:
                continue
        arr = np.asarray(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return arr[:, :3]

    return np.zeros((0, 3), dtype=np.float32)


def _template_collidable_from_config(objects_dir: str, model_id: str) -> Optional[bool]:
    """Read is_collidable from a local object config when available."""
    raw = str(model_id).strip()
    if not raw:
        return None
    path = find_object_config_path(raw, objects_dir or default_object_config_dirs_str())
    if path is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(payload, dict) and "is_collidable" in payload:
            return bool(payload.get("is_collidable"))
    return None


def _load_surface_points(surface_item: Dict[str, Any], base_dir: Optional[Path] = None) -> List[List[float]]:
    """
    Load surface points from `top_surface.point_cloud_file`.

    Backward-compatible fallback:
    if file path is missing, use legacy `top_surface.points` in JSON.
    """
    top_surface = surface_item.get("top_surface", {}) if isinstance(surface_item, dict) else {}
    if not isinstance(top_surface, dict):
        return []

    file_path = top_surface.get("point_cloud_file")
    if isinstance(file_path, str) and file_path.strip():
        p = Path(file_path).expanduser()
        if not p.is_absolute() and base_dir is not None:
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        arr = _load_point_cloud_file(p)
        if arr.size > 0:
            return np.round(arr[:, :3], 4).tolist()

    points = top_surface.get("points", [])
    if isinstance(points, list):
        return points
    return []



def _make_simulator(scene_name: str, data_dir: Path, enable_physics: bool = True) -> Optional[Any]:
    """创建轻量 habitat-sim 模拟器，用于放置与接触检测。"""
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
    """Load object template configs from all configured dataset roots."""
    if sim is None:
        return
    try:
        template_mgr = sim.get_object_template_manager()
    except Exception:
        return
    for config_dir in iter_object_config_dirs(objects_dir or default_object_config_dirs_str()):
        abs_dir = str(config_dir.expanduser().resolve())
        try:
            if hasattr(template_mgr, "load_configs"):
                template_mgr.load_configs(abs_dir)
            elif hasattr(template_mgr, "add_template_search_path"):
                template_mgr.add_template_search_path(abs_dir)
            elif hasattr(template_mgr, "load_object_configs"):
                template_mgr.load_object_configs(abs_dir)
        except Exception:
            continue


def place_objects_on_instances(
    scene_name: str,
    assignment_plan: Dict[str, Any],
    surfaces_payload: Dict[str, Any],
    data_dir: Path = DEFAULT_DATA_DIR,
    objects_dir: str = "",
    min_distance: float = 0.25,
    spawn_height: float = 0.3,
    max_trials_per_object: int = 30,
    settle_steps: int = 45,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    文件1调用的核心放置函数。

    输入约定：
    - `assignment_plan["assignments"]` 必须包含 `target_instance_id`
    - `surfaces_payload` 必须包含实例的 `top_surface.point_cloud_file`

    放置循环：
    1) 取一个候选表面点
    2) 以 `point_y + y_offset` 生成目标位置；仅可碰撞模板额外上抬 `spawn_height` 后执行物理稳定
    3) 执行最小距离约束
    4) 按需执行 habitat-sim 接触检测
    5) 首个合法候选即接受，否则记为失败
    """
    rng = random.Random(int(seed))
    np.random.seed(int(seed))

    scene_paths = resolve_scene_paths(scene_name, require_semantic=False, root=data_dir)
    scene_path = str(scene_paths.stage_glb) if scene_paths is not None else scene_name
    by_room_instance, by_instance = _build_surface_index(surfaces_payload)
    surfaces_base_dir: Optional[Path] = None
    raw_source_dir = surfaces_payload.get("_source_json_dir") if isinstance(surfaces_payload, dict) else None
    raw_source_path = surfaces_payload.get("_source_json_path") if isinstance(surfaces_payload, dict) else None
    if isinstance(raw_source_dir, str) and raw_source_dir.strip():
        surfaces_base_dir = Path(raw_source_dir).expanduser().resolve()
    elif isinstance(raw_source_path, str) and raw_source_path.strip():
        surfaces_base_dir = Path(raw_source_path).expanduser().resolve().parent

    sim = _make_simulator(scene_name, data_dir=data_dir, enable_physics=True)
    if sim is not None:
        _load_templates(sim, objects_dir)

    assignments = assignment_plan.get("assignments", []) if isinstance(assignment_plan, dict) else []
    placed_layout_objects: List[Dict[str, Any]] = []
    placed_internal: List[Dict[str, Any]] = []
    failed_objects: List[Dict[str, Any]] = []
    profile_diagnostics: List[Dict[str, Any]] = []
    profile_diag_seen = set()

    rom = None
    template_mgr = None
    runtime_profile_cache: Dict[str, Dict[str, Any]] = {}
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
        surface_attempts = _candidate_surface_items(assignment, by_room_instance, by_instance)
        if not surface_attempts:
            failed_objects.append(
                {
                    "object_id": object_id,
                    "model_id": model_id,
                    "target_instance_id": target_instance_id,
                    "reason": "missing_surface_instance",
                    "backup_instance_ids": assignment.get("backup_instance_ids", []),
                }
            )
            continue

        profile = _get_profile(model_id, objects_dir=objects_dir)
        template_collidable = _template_collidable_from_config(objects_dir, model_id)
        template_handle = None
        if sim is not None and rom is not None and template_mgr is not None and model_id:
            template_handle = _resolve_template_handle(template_mgr, model_id)
            if template_handle and template_handle not in runtime_profile_cache:
                runtime_profile_cache[template_handle] = _runtime_template_profile(rom, template_handle, profile)
            if template_handle and template_handle in runtime_profile_cache:
                profile = dict(runtime_profile_cache[template_handle])
        radius = float(profile.get("radius", 0.2))
        y_offset = max(float(profile.get("y_offset", 0.05)), 0.0)
        edge_margin = float(surface_requirement(profile)["edge_margin"])
        profile_source = str(profile.get("profile_source", "unknown"))
        if profile_source.startswith("keyword:") or profile_source.startswith("default_") or profile.get("missing_template_config"):
            diag_key = (model_id, profile_source, bool(profile.get("missing_template_config")))
            if diag_key not in profile_diag_seen:
                profile_diag_seen.add(diag_key)
                profile_diagnostics.append(
                    {
                        "model_id": model_id,
                        "profile_source": profile_source,
                        "missing_template_config": bool(profile.get("missing_template_config", False)),
                        "message": "Using estimated object profile; add object_profiles.json entry or valid template geometry for higher accuracy.",
                    }
                )
        use_physics_settle = template_collidable is not False and int(settle_steps) > 0
        placed = False
        failure_reason = "no_valid_candidate"
        placement_attempts: List[Dict[str, Any]] = []
        chosen_instance_id = target_instance_id
        chosen_surface_source = "target"

        for candidate_instance_id, surface_item, surface_source in surface_attempts:
            surface_height = _surface_height(surface_item)
            surface_ok, surface_fit_reason = _surface_fits_profile(surface_item, profile)
            if not surface_ok:
                failure_reason = surface_fit_reason
                placement_attempts.append(
                    {
                        "target_instance_id": int(candidate_instance_id),
                        "source": surface_source,
                        "reason": surface_fit_reason,
                    }
                )
                continue

            surface_points = _load_surface_points(surface_item, base_dir=surfaces_base_dir)
            candidates = _sample_surface_points(
                surface_points,
                max_trials=max_trials_per_object,
                rng=rng,
                edge_margin=edge_margin,
            )
            if not candidates:
                failure_reason = "empty_surface_points"
                placement_attempts.append(
                    {
                        "target_instance_id": int(candidate_instance_id),
                        "source": surface_source,
                        "reason": "empty_surface_points",
                    }
                )
                continue

            for pt in candidates:
                target_pos = [
                    _safe_float(pt[0]),
                    _safe_float(pt[1]) + y_offset,
                    _safe_float(pt[2]),
                ]
                spawn_pos = [
                    target_pos[0],
                    target_pos[1] + (float(spawn_height) if use_physics_settle else 0.0),
                    target_pos[2],
                ]
                if not _distance_ok(
                    target_pos,
                    radius,
                    placed_internal,
                    min_distance=min_distance,
                    target_instance_id=int(candidate_instance_id),
                    surface_height=surface_height,
                ):
                    failure_reason = "min_distance_rejected"
                    continue

                yaw = float(rng.uniform(0.0, 360.0))
                final_pos = list(target_pos)
                sim_object_id = None
                sim_handle = None

                if sim is not None and rom is not None and template_mgr is not None and model_id:
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
                        if use_physics_settle and hasattr(obj, "motion_type") and hasattr(habitat_sim, "physics"):
                            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                        elif hasattr(obj, "motion_type") and hasattr(habitat_sim, "physics"):
                            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                        if use_physics_settle:
                            _step_physics(sim, steps=settle_steps)
                        pos = getattr(obj, "translation", np.array(spawn_pos, dtype=np.float32))
                        final_pos = [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)]
                        existing_ids = [x.get("_sim_object_id") for x in placed_internal if x.get("_sim_object_id") is not None]
                        if not _distance_ok(
                            final_pos,
                            radius,
                            placed_internal,
                            min_distance=min_distance,
                            target_instance_id=int(candidate_instance_id),
                            surface_height=surface_height,
                        ):
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
                chosen_instance_id = candidate_instance_id
                chosen_surface_source = surface_source
                layout_obj = {
                    "id": int(idx),
                    "name": name,
                    "model_id": model_id,
                    "position": [round(float(final_pos[0]), 4), round(float(final_pos[1]), 4), round(float(final_pos[2]), 4)],
                    "rotation": [0.0, round(float(yaw), 4), 0.0],
                    "sampled_region_id": int(room_id) if room_id is not None else -1,
                    "target_instance_id": int(chosen_instance_id) if chosen_instance_id is not None else -1,
                    "assigned_target_instance_id": int(target_instance_id) if target_instance_id is not None else -1,
                    "placement_target_source": chosen_surface_source,
                    "source": "assigned_instance_surface",
                    "placement_y_offset": round(float(y_offset), 4),
                    "placement_radius": round(float(radius), 4),
                    "object_profile": {
                        "profile_source": profile.get("profile_source", "unknown"),
                        "placement_class": profile.get("placement_class", "tabletop_or_floor"),
                        "footprint_x": round(float(profile.get("footprint_x", 0.0)), 4),
                        "footprint_z": round(float(profile.get("footprint_z", 0.0)), 4),
                        "height": round(float(profile.get("height", 0.0)), 4),
                    },
                    "template_collidable": template_collidable,
                    "physics_settle": bool(use_physics_settle),
                }
                placed_layout_objects.append(layout_obj)
                placed_internal.append(
                    {
                        "object_id": object_id,
                        "position": layout_obj["position"],
                        "_radius": radius,
                        "_sim_object_id": sim_object_id,
                        "_sim_handle": sim_handle,
                        "_target_instance_id": int(chosen_instance_id),
                        "_surface_height": surface_height,
                    }
                )
                break

            if placed:
                break
            placement_attempts.append(
                {
                    "target_instance_id": int(candidate_instance_id),
                    "source": surface_source,
                    "reason": failure_reason,
                }
            )

        if not placed:
            failed_objects.append(
                {
                    "object_id": object_id,
                    "model_id": model_id,
                    "target_instance_id": target_instance_id,
                    "reason": failure_reason,
                    "backup_instance_ids": assignment.get("backup_instance_ids", []),
                    "placement_attempts": placement_attempts,
                    "object_profile": {
                        "profile_source": profile.get("profile_source", "unknown"),
                        "placement_class": profile.get("placement_class", "tabletop_or_floor"),
                        "radius": round(float(profile.get("radius", 0.0)), 4),
                        "y_offset": round(float(profile.get("y_offset", 0.0)), 4),
                    },
                }
            )

    if sim is not None:
        try:
            sim.close()
        except Exception:
            pass

    failed_by_reason: Dict[str, int] = {}
    for item in failed_objects:
        reason = str(item.get("reason", "unknown"))
        failed_by_reason[reason] = failed_by_reason.get(reason, 0) + 1

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
            "failed_by_reason": failed_by_reason,
            "failed_objects": failed_objects,
            "profile_diagnostics": profile_diagnostics,
        },
    }


def parse_args() -> argparse.Namespace:
    """定义文件2独立执行时的命令行参数。"""
    parser = argparse.ArgumentParser(description="Place objects on assigned instance top surfaces and write layout JSON.")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--assignment-plan", required=True, help="Assignment plan JSON from file1")
    parser.add_argument("--surfaces-json", required=True, help="Receptacle surfaces JSON from query_room_receptacle_objects.py")
    parser.add_argument("--output-layout", type=str, default=None, help="Output layout JSON path")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="HM3D root directory")
    parser.add_argument("--objects-dir", type=str, default=default_object_config_dirs_str(), help="Object template config directory or os.pathsep-separated directories")
    parser.add_argument("--min-distance", type=float, default=0.25, help="Minimum pairwise object distance on XZ")
    parser.add_argument("--spawn-height", type=float, default=0.3, help="Spawn height above target surface point")
    parser.add_argument("--max-trials-per-object", type=int, default=30, help="Max candidate points per object")
    parser.add_argument("--settle-steps", type=int, default=45, help="Physics settle steps after spawn")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    """命令行入口：加载输入、执行放置、写出布局 JSON。"""
    args = parse_args()
    try:
        assignment_plan = json.loads(Path(args.assignment_plan).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Error] Failed to load assignment plan: {exc}", file=sys.stderr)
        return 1
    try:
        surfaces_payload = json.loads(Path(args.surfaces_json).read_text(encoding="utf-8"))
        if isinstance(surfaces_payload, dict):
            src_path = Path(args.surfaces_json).expanduser().resolve()
            surfaces_payload["_source_json_path"] = str(src_path)
            surfaces_payload["_source_json_dir"] = str(src_path.parent)
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
    profile_diags = stats.get("profile_diagnostics", [])
    if isinstance(profile_diags, list) and profile_diags:
        print(f"[Warning] Estimated object profiles used: {len(profile_diags)}")
        for item in profile_diags[:8]:
            if isinstance(item, dict):
                print(
                    "  [Profile] model={model} source={source} missing_template_config={missing}".format(
                        model=item.get("model_id", "?"),
                        source=item.get("profile_source", "?"),
                        missing=bool(item.get("missing_template_config", False)),
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
