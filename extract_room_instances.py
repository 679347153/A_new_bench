#!/usr/bin/env python3
"""HM3D 房间实例提取与单实例点云导出工具。

这个脚本有两类用途：
1) 房间级查询：输入 scene + room_id，返回该房间内全部 instance 元信息。
2) 单实例点云：再输入 instance_id，返回该实例点云摘要并导出 ply/xyz 文件。

点云获取链路（按优先级）
------------------------
1. semantic.glb 颜色匹配（推荐优先）
    从 HM3D semantic.glb 中按 semantic.txt 对应 color_hex 匹配面片，
    再按面面积采样真实表面点。
2. habitat-sim 语义对象直读
    尝试从 semantic object 的 point_cloud / vertices 等字段直接读取。
3. stage mesh 颜色匹配（次级补充）
    在 stage.basis.glb 中按颜色近邻匹配面片并采样。
4. AABB/OBB 采样兜底
    若前三条都失败，才退回包围盒近似采样。

为什么会看到“habitat-sim 像是启动了多次”
--------------------------------------
这是当前设计下的正常现象，不一定是错误：
1. 当 scene_info 缺少 objects 时，会先启动一次 Simulator 重建对象清单。
2. 进入单实例点云查询时，可能再启动一次 Simulator 读取 semantic_scene。
3. 如果你后续再运行可视化脚本，那个脚本也可能单独启动 Simulator。

每次创建 Simulator 都会触发 OpenGL/Renderer 初始化日志，所以终端里会看到
类似 "Renderer: ..." 的重复输出。

输出形式
--------
- 仅房间：返回 room 元信息 + instance 列表。
- 指定 instance：返回 room 元信息 + instance + point_cloud + generation trace。

推荐用法
--------
python extract_room_instances.py --scene 00824-Dd4bFSTQ8gi --room-id 0
python extract_room_instances.py --scene 00824-Dd4bFSTQ8gi --room-id 0 --instance-id 1

说明：默认优先读取已导出的 scene_info JSON；若未找到，可用 --scene-info-path 显式指定。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hm3d_paths import resolve_scene_paths

try:
    import habitat_sim  # type: ignore[import-not-found]
except ImportError:
    habitat_sim = None

try:
    import trimesh  # type: ignore[import-not-found]
except ImportError:
    trimesh = None


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "hm3d"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "room_instances"


def _warn(message: str) -> None:
    """向终端标准错误输出警告信息。"""
    print(f"[Warning] {message}", file=sys.stderr)


def _as_vec3(value: Any) -> Optional[List[float]]:
    """把不同形态的向量对象统一转成 [x, y, z] 列表。

    这个函数专门用来兼容 habitat-sim / magnum 不同版本的返回值类型，
    避免直接假设 value 一定能下标访问或一定有 x/y/z 属性。
    """
    if value is None:
        return None
    if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
        return [round(float(value.x), 4), round(float(value.y), 4), round(float(value.z), 4)]
    try:
        return [round(float(value[0]), 4), round(float(value[1]), 4), round(float(value[2]), 4)]
    except Exception:
        pass
    try:
        data = list(value)
        if len(data) >= 3:
            return [round(float(data[0]), 4), round(float(data[1]), 4), round(float(data[2]), 4)]
    except Exception:
        pass
    return None


def _get_attr_or_call(obj: Any, name: str) -> Any:
    """读取对象属性；如果该属性本身是可调用对象，则自动调用一次。

    这么做是为了兼容 habitat-sim 里既有属性字段、又有方法字段的情况，
    例如某些绑定版本里 center 可能是属性，也可能是方法。
    """
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    try:
        return value() if callable(value) else value
    except Exception:
        return None


def _extract_bbox_info(bbox: Any) -> Dict[str, List[float]]:
    """提取包围盒的 min/max/center/size，兼容多版本字段名。

    这一层是兼容适配器：不直接依赖单一字段名，而是按多个候选字段依次尝试。
    只要能拿到 min/max，就会补出 center 和 size。
    """
    min_vec = None
    max_vec = None

    min_candidates = ["min", "min_corner", "back_bottom_left", "back_bottom_left_corner"]
    max_candidates = ["max", "max_corner", "front_top_right", "front_top_right_corner"]

    for name in min_candidates:
        min_vec = _as_vec3(_get_attr_or_call(bbox, name))
        if min_vec is not None:
            break

    for name in max_candidates:
        max_vec = _as_vec3(_get_attr_or_call(bbox, name))
        if max_vec is not None:
            break

    center_vec = _as_vec3(_get_attr_or_call(bbox, "center"))
    size_vec = _as_vec3(_get_attr_or_call(bbox, "size"))

    if center_vec is None and min_vec is not None and max_vec is not None:
        center_vec = [
            round((min_vec[0] + max_vec[0]) / 2.0, 4),
            round((min_vec[1] + max_vec[1]) / 2.0, 4),
            round((min_vec[2] + max_vec[2]) / 2.0, 4),
        ]

    if size_vec is None and min_vec is not None and max_vec is not None:
        size_vec = [
            round(max_vec[0] - min_vec[0], 4),
            round(max_vec[1] - min_vec[1], 4),
            round(max_vec[2] - min_vec[2], 4),
        ]

    # habitat-sim 0.2.5 某些绑定下可能给出 center/size，但 min/max 读不到或退化为零。
    # 这里补一层反推，避免把有效几何误判为零包围盒。
    if (min_vec is None or max_vec is None) and center_vec is not None and size_vec is not None:
        half = [round(float(size_vec[i]) / 2.0, 4) for i in range(3)]
        min_vec = [round(float(center_vec[i]) - half[i], 4) for i in range(3)]
        max_vec = [round(float(center_vec[i]) + half[i], 4) for i in range(3)]

    if min_vec is not None and max_vec is not None and center_vec is not None and size_vec is not None:
        minmax_near_zero = all(abs(float(v)) < 1e-8 for v in [*min_vec, *max_vec])
        size_has_extent = any(abs(float(v)) > 1e-6 for v in size_vec)
        center_nonzero = any(abs(float(v)) > 1e-6 for v in center_vec)
        if minmax_near_zero and size_has_extent:
            half = [round(float(size_vec[i]) / 2.0, 4) for i in range(3)]
            min_vec = [round(float(center_vec[i]) - half[i], 4) for i in range(3)]
            max_vec = [round(float(center_vec[i]) + half[i], 4) for i in range(3)]
        elif minmax_near_zero and center_nonzero and size_has_extent:
            half = [round(float(size_vec[i]) / 2.0, 4) for i in range(3)]
            min_vec = [round(float(center_vec[i]) - half[i], 4) for i in range(3)]
            max_vec = [round(float(center_vec[i]) + half[i], 4) for i in range(3)]

    return {
        "min": min_vec if min_vec is not None else [0.0, 0.0, 0.0],
        "max": max_vec if max_vec is not None else [0.0, 0.0, 0.0],
        "center": center_vec if center_vec is not None else [0.0, 0.0, 0.0],
        "size": size_vec if size_vec is not None else [0.0, 0.0, 0.0],
    }


def _bbox_is_zero(bbox_info: Dict[str, List[float]]) -> bool:
    """判断一个 bbox 是否是全零或近似全零。

    全零包围盒通常意味着当前对象几何信息不可用，后续点云回退逻辑会依赖这个判断。
    """
    min_pt = bbox_info.get("min", [0.0, 0.0, 0.0])
    max_pt = bbox_info.get("max", [0.0, 0.0, 0.0])
    center_pt = bbox_info.get("center", [0.0, 0.0, 0.0])
    size_pt = bbox_info.get("size", [0.0, 0.0, 0.0])

    minmax_zero = all(abs(float(v)) < 1e-8 for v in [*min_pt, *max_pt])
    size_zero = all(abs(float(v)) < 1e-8 for v in size_pt)
    center_zero = all(abs(float(v)) < 1e-8 for v in center_pt)

    # 只要 size 有非零尺寸，就不应被视为零包围盒。
    if not size_zero:
        return False

    # min/max 全零但 center 非零且 size 为零，通常仍表示无有效体积，按零包围盒处理。
    return minmax_zero and size_zero and center_zero


def _obb_to_aabb_info(obb_center: Optional[List[float]], obb_half_extents: Optional[List[float]]) -> Optional[Dict[str, List[float]]]:
    """用 OBB center/half_extents 近似生成 AABB 信息。"""
    if not obb_center or not obb_half_extents:
        return None
    if len(obb_center) < 3 or len(obb_half_extents) < 3:
        return None

    try:
        min_pt = [round(float(obb_center[i]) - float(obb_half_extents[i]), 4) for i in range(3)]
        max_pt = [round(float(obb_center[i]) + float(obb_half_extents[i]), 4) for i in range(3)]
    except Exception:
        return None

    return {
        "min": min_pt,
        "max": max_pt,
        "center": [round((min_pt[i] + max_pt[i]) / 2.0, 4) for i in range(3)],
        "size": [round(max_pt[i] - min_pt[i], 4) for i in range(3)],
    }


def _load_scene_info(scene_name: str, data_dir: Path, scene_info_path: Optional[Path] = None) -> Dict[str, Any]:
    """加载场景导出的 scene_info JSON。

    搜索顺序：
    1. 用户显式传入的 --scene-info-path
    2. 当前工作目录下的常见结果文件
    3. data_dir 下面的常见导出目录

    如果都没有找到，就直接抛错，让调用者知道需要先导出 scene_info。
    """
    candidates: List[Path] = []
    if scene_info_path is not None:
        candidates.append(scene_info_path)

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    scene_info_name = f"{scene_name}_scene_info.json"

    # Common export layouts:
    # 1) <base>/scene_info_export/<scene>_scene_info.json
    # 2) <base>/scene_info_export/<scene>/<scene>_scene_info.json
    # 3) <base>/results/scene_info/<scene>/<scene>_scene_info.json
    # 4) <base>/<scene>_scene_info.json
    bases = [
        cwd,
        script_dir,
        data_dir,
        data_dir.parent,
        cwd / "hm3d",
        script_dir / "hm3d",
    ]

    for base in bases:
        candidates.extend(
            [
                base / scene_info_name,
                base / "scene_info_export" / scene_info_name,
                base / "scene_info_export" / scene_name / scene_info_name,
                base / "results" / "scene_info" / scene_name / scene_info_name,
                base / scene_name / scene_info_name,
            ]
        )

    # Keep backward-compatible explicit paths.
    candidates.extend([
        cwd / scene_info_name,
        cwd / "results" / "scene_info" / scene_name / scene_info_name,
        data_dir / "scene_info_export" / scene_info_name,
        data_dir / "results" / "scene_info" / scene_name / scene_info_name,
        data_dir / scene_name / scene_info_name,
    ])

    # De-duplicate while preserving order.
    seen = set()
    dedup_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        dedup_candidates.append(candidate)

    for candidate in dedup_candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

    # Last fallback: recursive search in likely roots.
    search_roots = [cwd, script_dir, data_dir, data_dir.parent]
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for found in root.rglob(scene_info_name):
            if found.is_file():
                with open(found, "r", encoding="utf-8") as f:
                    return json.load(f)

    tried = "\n".join(f"  - {p}" for p in dedup_candidates)
    raise FileNotFoundError(
        "找不到 scene_info JSON: "
        f"{scene_name}.\n"
        "已尝试以下路径:\n"
        f"{tried}\n"
        "你可以通过 --scene-info-path 显式指定，或先运行 export_scene_info.py。"
    )


def _load_sim(scene_name: str, data_dir: Path) -> Optional[Any]:
    """创建一个最小化 habitat_sim Simulator，用于尝试读取语义对象。

    这个 simulator 不做渲染和物理，只是为了访问 semantic_scene 以及对象几何。
    如果 habitat_sim 不可用，函数返回 None，调用方会自动走纯 JSON 逻辑。

    注意：每调用一次本函数，habitat-sim 都会初始化一次图形上下文，
    终端通常会打印 Renderer/OpenGL 信息。若在一次任务里多处调用本函数，
    看起来就像“启动了多次”。
    """
    if habitat_sim is None:
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None:
        return None

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
    sim_cfg.scene_id = str(scene_paths.stage_glb)
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.load_semantic_mesh = True

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [32, 32]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def _parse_semantic_txt(semantic_txt_path: Path) -> Dict[int, Dict[str, Any]]:
    """解析 semantic.txt，建立 semantic_id 到类别/房间映射。"""
    entries: Dict[int, Dict[str, Any]] = {}
    if not semantic_txt_path.is_file():
        return entries

    with open(semantic_txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line_no == 0 or not line:
                continue
            reader = csv.reader(io.StringIO(line))
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    obj_id = int(row[0])
                    entries[obj_id] = {
                        "category": row[2].strip().strip('"'),
                        "region_id": int(row[3]),
                        "color_hex": row[1].strip(),
                    }
                except Exception:
                    continue
    return entries


def _build_scene_objects_from_sim(scene_name: str, data_dir: Path) -> List[Dict[str, Any]]:
    """当 scene_info 缺少 objects 时，从 habitat-sim 重建对象明细。

    这是“对象清单补全”步骤，可能触发一次独立 Simulator 初始化。
    """
    if habitat_sim is None:
        return []

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None:
        return []

    txt_entries = _parse_semantic_txt(scene_paths.semantic_txt)
    sim = _load_sim(scene_name, data_dir)
    if sim is None:
        return []

    objects: List[Dict[str, Any]] = []
    try:
        sem_scene = sim.semantic_scene
        for obj in getattr(sem_scene, "objects", []) or []:
            if obj is None:
                continue
            sid = int(getattr(obj, "semantic_id", -1))
            txt_info = txt_entries.get(sid, {})
            category = txt_info.get("category", obj.category.name() if getattr(obj, "category", None) else "unknown")
            region_id = int(txt_info.get("region_id", -1))
            color_hex = txt_info.get("color_hex", "")

            aabb_info = _extract_bbox_info(getattr(obj, "aabb", None))
            obb = getattr(obj, "obb", None)
            obb_center = _as_vec3(_get_attr_or_call(obb, "center")) or [0.0, 0.0, 0.0]
            obb_half_extents = _as_vec3(_get_attr_or_call(obb, "half_extents"))
            if obb_half_extents is None:
                obb_half_extents = _as_vec3(_get_attr_or_call(obb, "halfExtents")) or [0.0, 0.0, 0.0]

            bbox_source = "aabb"
            if _bbox_is_zero(aabb_info):
                obb_fallback = _obb_to_aabb_info(obb_center, obb_half_extents)
                if obb_fallback is not None and not _bbox_is_zero(obb_fallback):
                    aabb_info = obb_fallback
                    bbox_source = "obb_fallback"
                else:
                    bbox_source = "zero"

            objects.append(
                {
                    "id": sid,
                    "category": category,
                    "region_id": region_id,
                    "color_hex": color_hex,
                    "bbox_source": bbox_source,
                    "aabb": aabb_info,
                    "obb": {
                        "center": obb_center,
                        "half_extents": obb_half_extents,
                        "rotation": None,
                    },
                }
            )
    finally:
        sim.close()

    return objects


def _find_scene_object(sim: Any, instance_id: int) -> Optional[Any]:
    """在 semantic_scene.objects 中找到指定 instance_id 对应的对象。"""
    sem_scene = getattr(sim, "semantic_scene", None)
    if sem_scene is None:
        return None

    for obj in getattr(sem_scene, "objects", []) or []:
        if obj is None:
            continue
        semantic_id = getattr(obj, "semantic_id", None)
        if semantic_id == instance_id:
            return obj
        obj_id = getattr(obj, "id", None)
        if obj_id == instance_id:
            return obj
    return None


def _extract_direct_point_cloud(obj: Any) -> Optional[np.ndarray]:
    """尽可能从 habitat-sim 对象上直接读取点云或顶点数据。

    这个函数是“真点云优先”的入口。由于不同 habitat-sim 版本暴露的字段名不完全一样，
    这里会按多个候选字段尝试；只要找到一个二维点数组就直接返回。
    """
    candidate_names = [
        "point_cloud",
        "points",
        "vertices",
        "vertex_positions",
        "mesh_vertices",
        "mesh_points",
        "world_vertices",
    ]

    for name in candidate_names:
        value = _get_attr_or_call(obj, name)
        if value is None:
            continue
        try:
            points = np.asarray(value, dtype=np.float32)
        except Exception:
            continue
        if points.ndim == 2 and points.shape[1] >= 3 and len(points) > 0:
            return points[:, :3]

    mesh = _get_attr_or_call(obj, "mesh")
    if mesh is not None:
        mesh_candidates = ["vertices", "vertex_positions", "points"]
        for name in mesh_candidates:
            value = _get_attr_or_call(mesh, name)
            if value is None:
                continue
            try:
                points = np.asarray(value, dtype=np.float32)
            except Exception:
                continue
            if points.ndim == 2 and points.shape[1] >= 3 and len(points) > 0:
                return points[:, :3]

    return None


def _parse_color_hex_to_rgb(color_hex: Any) -> Optional[Tuple[int, int, int]]:
    """把 semantic.txt 里的十六进制颜色解析成 RGB 三元组。"""
    if color_hex is None:
        return None
    text = str(color_hex).strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[i:i + 2], 16) for i in range(0, 6, 2))
    except ValueError:
        return None


def _as_rgb_uint8(arr: Any) -> Optional[np.ndarray]:
    """把颜色数组统一转成 uint8 RGB。"""
    try:
        rgb = np.asarray(arr)
    except Exception:
        return None
    if rgb.ndim != 2 or rgb.shape[1] < 3:
        return None
    rgb = rgb[:, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        max_val = float(np.max(rgb)) if rgb.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            rgb = np.clip(np.round(rgb * 255.0), 0.0, 255.0)
        else:
            rgb = np.clip(np.round(rgb), 0.0, 255.0)
    else:
        rgb = np.clip(rgb, 0, 255)
    return rgb.astype(np.uint8, copy=False)


def _extract_mesh_face_colors(geom: Any) -> Optional[np.ndarray]:
    """尽量把 mesh 的颜色统一到 per-face RGB，便于按 semantic color 筛面片。"""
    visual = getattr(geom, "visual", None)
    if visual is None:
        return None

    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None:
        arr = _as_rgb_uint8(face_colors)
        if arr is not None and arr.ndim == 2 and arr.shape[0] == len(getattr(geom, "faces", [])) and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.uint8, copy=False)

    vertex_colors = getattr(visual, "vertex_colors", None)
    vertices = np.asarray(getattr(geom, "vertices", []), dtype=np.float32)
    faces = np.asarray(getattr(geom, "faces", []), dtype=np.int64)
    if vertex_colors is None or len(vertices) == 0 or len(faces) == 0:
        return None

    arr = _as_rgb_uint8(vertex_colors)
    if arr is None or arr.ndim != 2 or arr.shape[0] != len(vertices) or arr.shape[1] < 3:
        return None

    rgb = arr[:, :3].astype(np.int32, copy=False)
    face_rgb = rgb[faces]
    return np.round(face_rgb.mean(axis=1)).astype(np.uint8)


def _sample_points_from_triangles(vertices: np.ndarray, faces: np.ndarray, num_points: int) -> np.ndarray:
    """按三角面面积加权采样真实表面点。"""
    verts = np.asarray(vertices, dtype=np.float32)
    tris = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] < 3 or tris.ndim != 2 or tris.shape[1] < 3 or len(tris) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    tri_verts = verts[tris[:, :3]]
    cross = np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid_mask = areas > 1e-12
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32)

    tri_verts = tri_verts[valid_mask]
    areas = areas[valid_mask]
    probs = areas / areas.sum()

    count = max(int(num_points), 1)
    rng = np.random.default_rng(42)
    tri_ids = rng.choice(len(tri_verts), size=count, p=probs)
    chosen = tri_verts[tri_ids]

    r1 = rng.random((count, 1), dtype=np.float32)
    r2 = rng.random((count, 1), dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary_a = 1.0 - sqrt_r1
    bary_b = sqrt_r1 * (1.0 - r2)
    bary_c = sqrt_r1 * r2
    points = bary_a * chosen[:, 0] + bary_b * chosen[:, 1] + bary_c * chosen[:, 2]
    return np.round(points.astype(np.float32), 4)


def _populate_instance_color_from_semantic_txt(
    scene_name: str,
    instance: Dict[str, Any],
    data_dir: Path,
) -> bool:
    """若 instance 缺少 color_hex，尝试从 semantic.txt 按 instance id 补全。"""
    color_hex = str(instance.get("color_hex", "") or "").strip()
    if color_hex:
        return True

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None:
        return False
    entries = _parse_semantic_txt(scene_paths.semantic_txt)
    sid = int(instance.get("id", -1))
    info = entries.get(sid)
    if not info:
        return False

    ch = str(info.get("color_hex", "") or "").strip()
    if not ch:
        return False
    instance["color_hex"] = ch
    return True


def _extract_point_cloud_from_semantic_mesh(
    scene_name: str,
    instance: Dict[str, Any],
    data_dir: Path,
    num_points: int,
) -> Optional[np.ndarray]:
    """从 HM3D 的 semantic.glb 中按 instance 颜色恢复真实表面点云。"""
    if trimesh is None:
        _warn("trimesh 不可用，无法从 semantic.glb 提取实例真实面片；将继续尝试其他点云来源。")
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None or not scene_paths.semantic_glb.is_file():
        return None

    target_rgb = _parse_color_hex_to_rgb(instance.get("color_hex"))
    if target_rgb is None:
        _warn(
            f"instance_id={int(instance.get('id', -1))} 缺少可用 color_hex，"
            "semantic.glb 颜色匹配无法执行。"
        )
        return None

    try:
        loaded = trimesh.load(scene_paths.semantic_glb, force="scene")
    except Exception as exc:
        _warn(f"加载 semantic.glb 失败: {exc}")
        return None

    geometries = getattr(loaded, "geometry", None)
    if isinstance(loaded, trimesh.Trimesh):
        geometries = {"semantic_mesh": loaded}
    if not geometries:
        return None

    collected_vertices: List[np.ndarray] = []
    collected_faces: List[np.ndarray] = []
    vertex_offset = 0

    for geom in geometries.values():
        vertices = np.asarray(getattr(geom, "vertices", []), dtype=np.float32)
        faces = np.asarray(getattr(geom, "faces", []), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] < 3 or len(faces) == 0:
            continue

        per_face_rgb = _extract_mesh_face_colors(geom)
        if per_face_rgb is None or len(per_face_rgb) != len(faces):
            continue

        target_arr = np.asarray(target_rgb, dtype=np.int16)
        per_face_int = per_face_rgb.astype(np.int16)
        exact_mask = np.all(per_face_int == target_arr[None, :], axis=1)

        if np.any(exact_mask):
            match_mask = exact_mask
        else:
            # 兼容少量颜色量化误差：允许与目标色在 RGB 欧氏距离 <= 6 的面片作为近似匹配。
            diff = per_face_int - target_arr[None, :]
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            near_mask = dist <= 6.0
            if np.any(near_mask):
                match_mask = near_mask
            else:
                continue

        matched_faces = faces[match_mask][:, :3]
        unique_vids, inverse = np.unique(matched_faces.reshape(-1), return_inverse=True)
        local_vertices = vertices[unique_vids][:, :3]
        local_faces = inverse.reshape(-1, 3).astype(np.int64) + vertex_offset

        collected_vertices.append(local_vertices)
        collected_faces.append(local_faces)
        vertex_offset += len(local_vertices)

    if not collected_vertices or not collected_faces:
        _warn(
            f"semantic.glb 颜色匹配失败：instance_id={int(instance.get('id', -1))}, "
            f"target_color={instance.get('color_hex', 'N/A')}。"
        )
        return None

    merged_vertices = np.concatenate(collected_vertices, axis=0)
    merged_faces = np.concatenate(collected_faces, axis=0)
    _warn(
        f"semantic.glb 颜色匹配成功：instance_id={int(instance.get('id', -1))}, "
        f"matched_faces={int(len(merged_faces))}, source=semantic_mesh_by_color。"
    )
    return _sample_points_from_triangles(merged_vertices, merged_faces, num_points)


def _sample_points_on_aabb(min_pt: List[float], max_pt: List[float], num_points: int) -> np.ndarray:
    """基于 AABB 表面采样一个近似点云。

    这是点云回退策略：当无法从底层语义对象直接提取点云时，脚本会用该房间/instance
    的包围盒表面生成一组采样点，用来提供可读、可下游处理的几何近似结果。
    """
    min_arr = np.asarray(min_pt, dtype=np.float32)
    max_arr = np.asarray(max_pt, dtype=np.float32)
    extent = np.maximum(max_arr - min_arr, 1e-6)

    face_areas = np.array([
        extent[1] * extent[2],
        extent[1] * extent[2],
        extent[0] * extent[2],
        extent[0] * extent[2],
        extent[0] * extent[1],
        extent[0] * extent[1],
    ], dtype=np.float64)
    face_probs = face_areas / face_areas.sum()

    rng = np.random.default_rng(42)
    face_ids = rng.choice(6, size=max(int(num_points), 1), p=face_probs)
    points = np.empty((len(face_ids), 3), dtype=np.float32)

    for idx, face_id in enumerate(face_ids):
        u = rng.random()
        v = rng.random()

        if face_id == 0:
            points[idx] = [min_arr[0], min_arr[1] + u * extent[1], min_arr[2] + v * extent[2]]
        elif face_id == 1:
            points[idx] = [max_arr[0], min_arr[1] + u * extent[1], min_arr[2] + v * extent[2]]
        elif face_id == 2:
            points[idx] = [min_arr[0] + u * extent[0], min_arr[1], min_arr[2] + v * extent[2]]
        elif face_id == 3:
            points[idx] = [min_arr[0] + u * extent[0], max_arr[1], min_arr[2] + v * extent[2]]
        elif face_id == 4:
            points[idx] = [min_arr[0] + u * extent[0], min_arr[1] + v * extent[1], min_arr[2]]
        else:
            points[idx] = [min_arr[0] + u * extent[0], min_arr[1] + v * extent[1], max_arr[2]]

    return np.round(points, 4)


def _hex_to_rgb01(color_hex: str) -> Optional[np.ndarray]:
    """将 '#RRGGBB' 或 'RRGGBB' 转为 [0,1] 的 RGB。"""
    if not color_hex:
        return None
    text = str(color_hex).strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        rgb = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        return np.asarray(rgb, dtype=np.float32)
    except Exception:
        return None


def _extract_instance_mesh_points_by_semantic_color(
    scene_name: str,
    instance: Dict[str, Any],
    data_dir: Path,
    num_points: int,
) -> Optional[np.ndarray]:
    """从 stage mesh 中按 semantic 颜色提取实例表面点云。

    适用场景：habitat-sim 未暴露对象直接顶点/点云字段时，
    通过 semantic.txt 中的 color_hex 与 mesh 面颜色匹配，提取更接近实体表面的点。
    """
    if trimesh is None:
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None or not scene_paths.stage_glb.is_file():
        return None

    color_hex = str(instance.get("color_hex", "") or "").strip()
    target_rgb = _hex_to_rgb01(color_hex)
    if target_rgb is None:
        return None

    try:
        loaded = trimesh.load(scene_paths.stage_glb, force="scene")
    except Exception:
        return None

    geometries = getattr(loaded, "geometry", None)
    if not geometries:
        return None

    matched_meshes: List[Any] = []
    face_counts: List[int] = []

    for geom in geometries.values():
        faces = np.asarray(getattr(geom, "faces", []), dtype=np.int64)
        vertices = np.asarray(getattr(geom, "vertices", []), dtype=np.float32)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or len(vertices) == 0:
            continue

        visual = getattr(geom, "visual", None)
        face_colors = None
        if visual is not None and hasattr(visual, "face_colors"):
            try:
                face_colors = np.asarray(visual.face_colors, dtype=np.float32)
            except Exception:
                face_colors = None

        if face_colors is None or face_colors.ndim != 2 or face_colors.shape[1] < 3:
            continue

        rgb = face_colors[:, :3] / 255.0
        dist = np.linalg.norm(rgb - target_rgb[None, :], axis=1)
        # 语义颜色通常是离散色，阈值设小一点，避免混入其他物体。
        mask = dist <= 0.03
        if not np.any(mask):
            continue

        selected_faces = faces[mask]
        if len(selected_faces) == 0:
            continue

        try:
            sub = trimesh.Trimesh(vertices=vertices, faces=selected_faces, process=False)
        except Exception:
            continue
        if len(sub.faces) == 0 or len(sub.vertices) == 0:
            continue

        matched_meshes.append(sub)
        face_counts.append(int(len(sub.faces)))

    if not matched_meshes:
        return None

    total_faces = max(int(sum(face_counts)), 1)
    sampled_chunks: List[np.ndarray] = []

    for idx, sub in enumerate(matched_meshes):
        ratio = float(face_counts[idx]) / float(total_faces)
        k = max(int(round(ratio * float(max(num_points, 1)))), 32)
        try:
            pts = np.asarray(sub.sample(k), dtype=np.float32)
        except Exception:
            try:
                pts = np.asarray(sub.vertices, dtype=np.float32)
            except Exception:
                pts = np.zeros((0, 3), dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
            sampled_chunks.append(pts[:, :3])

    if not sampled_chunks:
        return None

    points = np.concatenate(sampled_chunks, axis=0)
    if len(points) > max(int(num_points), 1):
        rng = np.random.default_rng(42)
        keep = rng.choice(len(points), size=int(num_points), replace=False)
        points = points[keep]
    return np.round(points, 4)


def _summarize_point_cloud(points: np.ndarray, source: str) -> Dict[str, Any]:
    """把点云压缩成一个便于落盘和调试的摘要结构。"""
    if points.size == 0:
        return {
            "source": source,
            "point_count": 0,
            "centroid": [0.0, 0.0, 0.0],
            "bounds": {
                "min": [0.0, 0.0, 0.0],
                "max": [0.0, 0.0, 0.0],
            },
            "points": [],
        }

    pts = np.asarray(points, dtype=np.float32)[:, :3]
    centroid = np.round(pts.mean(axis=0), 4).tolist()
    bounds_min = np.round(pts.min(axis=0), 4).tolist()
    bounds_max = np.round(pts.max(axis=0), 4).tolist()

    return {
        "source": source,
        "point_count": int(len(pts)),
        "centroid": centroid,
        "bounds": {
            "min": bounds_min,
            "max": bounds_max,
        },
        "points": np.round(pts, 4).tolist(),
    }


def _write_point_cloud_file(points: List[List[float]], output_path: Path, fmt: str = "ply") -> Path:
    """将点云导出为常见格式文件（ply/xyz）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fmt = (fmt or "ply").lower()

    if safe_fmt not in {"ply", "xyz"}:
        raise ValueError(f"Unsupported point cloud format: {fmt}")

    pts = points or []
    if safe_fmt == "xyz":
        if output_path.suffix.lower() != ".xyz":
            output_path = output_path.with_suffix(".xyz")
        with open(output_path, "w", encoding="utf-8") as f:
            for p in pts:
                if len(p) >= 3:
                    f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
        return output_path

    if output_path.suffix.lower() != ".ply":
        output_path = output_path.with_suffix(".ply")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            if len(p) >= 3:
                f.write(f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f}\n")
    return output_path


def get_room_instances(
    scene_info: Dict[str, Any],
    room_id: int,
    scene_name: Optional[str] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> Dict[str, Any]:
    """根据房间号从 scene_info 中提取该房间内的所有 instance。

    优先使用 room 里的 object_ids 做精确关联；如果没有 object_ids，则退化为按
    region_id 过滤 objects。这样可以兼容不同版本的 scene_info 导出结构。
    """
    rooms = scene_info.get("rooms", []) or []
    objects = scene_info.get("objects", []) or []

    # 某些导出版本只包含 rooms 的数量统计，没有 objects 明细；这里自动补全。
    if not objects and scene_name:
        _warn(
            "scene_info 中缺少 objects 明细，正在尝试从 habitat-sim 重建实例列表，"
            "用于输出每个物体的编号与几何信息。"
        )
        objects = _build_scene_objects_from_sim(scene_name=scene_name, data_dir=data_dir)
    room = next((item for item in rooms if int(item.get("region_id", -999)) == int(room_id)), None)
    if room is None:
        raise KeyError(f"找不到房间 region_id={room_id}")

    object_ids = room.get("object_ids")
    if isinstance(object_ids, list) and object_ids:
        wanted_ids = {int(x) for x in object_ids}
        room_objects = [obj for obj in objects if int(obj.get("id", -1)) in wanted_ids]
    else:
        room_objects = [obj for obj in objects if int(obj.get("region_id", -999)) == int(room_id)]

    return {
        "room": room,
        "instances": room_objects,
        "instance_count": len(room_objects),
        "instance_ids": [int(obj.get("id", -1)) for obj in room_objects if int(obj.get("id", -1)) >= 0],
    }


def get_instance_point_cloud(
    scene_name: str,
    instance_id: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    num_points: int = 2048,
    instance_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """获取单个 instance 的点云信息。

    处理顺序：
    1. 从 scene_info 中定位 instance 的 AABB 和基础元数据。
    2. 先尝试 semantic.glb 颜色匹配，恢复实例真实表面点。
    3. 再尝试 habitat-sim semantic object 直读。
    4. 再尝试 stage mesh 颜色匹配。
    5. 若仍失败，则用 AABB/OBB 表面采样兜底。

    注意：步骤 3 会创建 Simulator，因此会出现一次 Renderer/OpenGL 初始化日志。
    """
    scene_info = _load_scene_info(scene_name, data_dir)
    generation_trace: List[str] = [
        "loaded_scene_info",
        f"target_instance_id={int(instance_id)}",
    ]
    objects = scene_info.get("objects", []) or []
    instance = next((obj for obj in objects if int(obj.get("id", -1)) == int(instance_id)), None)

    # 某些 scene_info 导出版本不包含 objects，或 objects 不完整：尝试从 habitat-sim 重建后再匹配。
    if instance is None:
        rebuilt_objects = _build_scene_objects_from_sim(scene_name=scene_name, data_dir=data_dir)
        if rebuilt_objects:
            generation_trace.append("rebuilt_objects_from_sim")
            instance = next((obj for obj in rebuilt_objects if int(obj.get("id", -1)) == int(instance_id)), None)
            if instance is not None:
                objects = rebuilt_objects

    # 如果上面仍未命中，但调用方已经在 room 内匹配到了该 instance，就直接使用该提示对象。
    if instance is None and instance_hint is not None:
        try:
            if int(instance_hint.get("id", -1)) == int(instance_id):
                instance = instance_hint
                generation_trace.append("used_instance_hint")
        except Exception:
            pass

    if instance is None:
        available_ids = sorted({int(obj.get("id", -1)) for obj in objects if int(obj.get("id", -1)) >= 0})
        preview = available_ids[:30]
        suffix = "..." if len(available_ids) > 30 else ""
        raise KeyError(
            "找不到 instance_id="
            f"{instance_id}. "
            f"当前可用 instance_id 示例: {preview}{suffix}"
        )

    # scene_info 可能缺少 color_hex，先尝试从 semantic.txt 补齐，以提高 semantic mesh 提取命中率。
    if not str(instance.get("color_hex", "") or "").strip():
        if _populate_instance_color_from_semantic_txt(scene_name, instance, data_dir):
            generation_trace.append("instance_color_hex_filled_from_semantic_txt")
        else:
            generation_trace.append("instance_color_hex_missing")

    semantic_mesh_points = _extract_point_cloud_from_semantic_mesh(
        scene_name=scene_name,
        instance=instance,
        data_dir=data_dir,
        num_points=num_points,
    )
    if semantic_mesh_points is not None and len(semantic_mesh_points) > 0:
        generation_trace.append("point_cloud_source=semantic_mesh_by_color")
        return {
            "scene_name": scene_name,
            "instance": instance,
            "point_cloud": _summarize_point_cloud(semantic_mesh_points, "semantic_mesh_by_color"),
            "point_cloud_generation": {
                "method": "semantic_mesh_by_color",
                "details": "sampled from matched triangles in HM3D semantic.glb using semantic.txt color mapping",
                "trace": generation_trace,
            },
        }
    generation_trace.append("semantic_mesh_by_color_unavailable")
    _warn(
        "semantic.glb 颜色匹配未命中，继续尝试 habitat-sim 直读与其他回退策略。"
    )

    # 这一段会触发 habitat-sim 初始化日志（Renderer/OpenGL）。
    sim = _load_sim(scene_name, data_dir)
    try:
        if sim is not None:
            generation_trace.append("simulator_ready")
            obj = _find_scene_object(sim, int(instance_id))
            if obj is not None:
                generation_trace.append("semantic_object_found")
                direct_points = _extract_direct_point_cloud(obj)
                if direct_points is not None:
                    # 直接提取到真实几何点云时，优先返回这一路径。
                    generation_trace.append("point_cloud_source=direct_attribute")
                    return {
                        "scene_name": scene_name,
                        "instance": instance,
                        "point_cloud": _summarize_point_cloud(direct_points, "direct_attribute"),
                        "point_cloud_generation": {
                            "method": "direct_attribute",
                            "details": "point cloud/vertices directly extracted from habitat-sim semantic object",
                            "trace": generation_trace,
                        },
                    }
                _warn(
                    "habitat-sim 已定位到目标 instance，但未暴露可直接读取的点云/顶点字段，"
                    "将先尝试语义 mesh 提取，再回退到包围盒采样。"
                )
                generation_trace.append("direct_point_cloud_unavailable")
            else:
                _warn(
                    "habitat-sim 已初始化，但 semantic_scene.objects 中未找到目标 instance，"
                    "将先尝试语义 mesh 提取，再回退到包围盒采样。"
                )
                generation_trace.append("semantic_object_not_found")
        else:
            _warn(
                "habitat-sim 不可用或场景无法初始化，无法直接读取实例几何；"
                "将先尝试语义 mesh 提取，再回退到包围盒采样。"
            )
            generation_trace.append("simulator_unavailable")

        mesh_points = _extract_instance_mesh_points_by_semantic_color(
            scene_name=scene_name,
            instance=instance,
            data_dir=data_dir,
            num_points=num_points,
        )
        if mesh_points is not None and len(mesh_points) > 0:
            generation_trace.append("point_cloud_source=semantic_mesh_color")
            _warn(
                f"direct 点云不可用，已使用语义 mesh 颜色匹配提取实例表面点云，点数={int(len(mesh_points))}，"
                "source=semantic_mesh_color。"
            )
            return {
                "scene_name": scene_name,
                "instance": instance,
                "point_cloud": _summarize_point_cloud(mesh_points, "semantic_mesh_color"),
                "point_cloud_generation": {
                    "method": "semantic_mesh_color",
                    "details": "instance surface points sampled from stage mesh faces matched by semantic color",
                    "trace": generation_trace,
                },
            }

        generation_trace.append("semantic_mesh_color_unavailable")

        bbox = instance.get("aabb") or {}
        min_pt = bbox.get("min", [0.0, 0.0, 0.0])
        max_pt = bbox.get("max", [0.0, 0.0, 0.0])
        if _bbox_is_zero({"min": min_pt, "max": max_pt}):
            # AABB 不可用时，尝试用 OBB center+half_extents 构造近似 AABB。
            obb_info = instance.get("obb") or {}
            obb_center = obb_info.get("center")
            obb_half_extents = obb_info.get("half_extents")
            obb_fallback = _obb_to_aabb_info(obb_center, obb_half_extents)

            if obb_fallback is not None and not _bbox_is_zero(obb_fallback):
                min_pt = obb_fallback.get("min", min_pt)
                max_pt = obb_fallback.get("max", max_pt)
                points = _sample_points_on_aabb(min_pt, max_pt, num_points)
                _warn(
                    f"instance 的 AABB 无效，已回退为 OBB->AABB 近似采样，采样点数={int(num_points)}，"
                    "source=sampled_obb_fallback。"
                )
                generation_trace.append(f"fallback=sampled_obb_fallback,num_points={int(num_points)}")
            else:
                # 如果连 OBB 也不可用，就返回空点云，避免伪造几何。
                _warn(
                    "instance 的 AABB 为全零或缺失，且 OBB 也不可用于回退采样，最终输出为空点云。"
                )
                points = np.zeros((0, 3), dtype=np.float32)
                generation_trace.append("fallback=empty_point_cloud_due_to_invalid_bbox_and_obb")
        else:
            # 回退到包围盒表面采样，生成一个可用的近似点云。
            points = _sample_points_on_aabb(min_pt, max_pt, num_points)
            _warn(
                f"最终输出由 AABB 表面采样生成，采样点数={int(num_points)}，"
                "source=sampled_aabb_surface。"
            )
            generation_trace.append(f"fallback=sampled_aabb_surface,num_points={int(num_points)}")

        return {
            "scene_name": scene_name,
            "instance": instance,
            "point_cloud": _summarize_point_cloud(
                points,
                (
                    "sampled_obb_fallback"
                    if any("sampled_obb_fallback" in t for t in generation_trace)
                    else "sampled_aabb_surface"
                ),
            ),
            "point_cloud_generation": {
                "method": (
                    "sampled_obb_fallback"
                    if any("sampled_obb_fallback" in t for t in generation_trace)
                    else "sampled_aabb_surface"
                ),
                "details": (
                    "fallback sampling from OBB-derived AABB because instance AABB is invalid"
                    if any("sampled_obb_fallback" in t for t in generation_trace)
                    else "fallback sampling from instance AABB surfaces because direct semantic point cloud is unavailable"
                ),
                "trace": generation_trace,
            },
        }
    finally:
        if sim is not None:
            sim.close()


def extract_room_instances(
    scene_name: str,
    room_id: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    scene_info_path: Optional[Path] = None,
    instance_id: Optional[int] = None,
    num_points: int = 2048,
) -> Dict[str, Any]:
    """对外暴露的主查询接口。

    当只传 room_id 时，返回该房间的所有 instance；当再传 instance_id 时，
    则返回该 instance 的点云信息，并附带所属房间。
    """
    scene_info = _load_scene_info(scene_name, data_dir, scene_info_path=scene_info_path)
    room_report = get_room_instances(
        scene_info,
        room_id,
        scene_name=scene_name,
        data_dir=data_dir,
    )

    if instance_id is not None:
        matched = next((obj for obj in room_report["instances"] if int(obj.get("id", -1)) == int(instance_id)), None)
        if matched is None:
            raise KeyError(f"instance_id={instance_id} 不属于 room_id={room_id}")

        point_cloud_report = get_instance_point_cloud(
            scene_name=scene_name,
            instance_id=instance_id,
            data_dir=data_dir,
            num_points=num_points,
            instance_hint=matched,
        )
        point_cloud_report["room"] = room_report["room"]
        return point_cloud_report

    return {
        "scene_name": scene_name,
        "room": room_report["room"],
        "instance_count": room_report["instance_count"],
        "instances": room_report["instances"],
    }


def main() -> None:
    """命令行入口。

    这个入口只负责参数解析、调用查询函数、打印 JSON、以及把结果保存到本地。
    实际业务逻辑都在上面的函数里，方便后续被别的脚本直接 import 使用。
    """
    parser = argparse.ArgumentParser(description="提取 HM3D 场景中某个房间的所有 instance，并可返回单 instance 点云")
    parser.add_argument("--scene", required=True, help="场景名，例如 00824-Dd4bFSTQ8gi")
    parser.add_argument("--room-id", type=int, required=True, help="房间 region_id")
    parser.add_argument("--instance-id", type=int, default=None, help="可选：只查询某个 instance")
    parser.add_argument("--scene-info-path", type=str, default=None, help="可选：scene_info JSON 路径")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="HM3D 数据根目录")
    parser.add_argument("--num-points", type=int, default=2048, help="点云采样数量（仅在没有直接点云时使用）")
    parser.add_argument("--output", type=str, default=None, help="可选：输出 JSON 文件路径")
    parser.add_argument("--pointcloud-format", choices=["ply", "xyz"], default="ply", help="instance 查询时点云文件格式")
    parser.add_argument("--pointcloud-output", type=str, default=None, help="可选：instance 查询时点云输出路径")
    parser.add_argument("--print-json", action="store_true", help="可选：将 JSON 结果打印到终端（默认不打印）")
    args = parser.parse_args()

    result = extract_room_instances(
        scene_name=args.scene,
        room_id=args.room_id,
        data_dir=Path(args.data_dir),
        scene_info_path=Path(args.scene_info_path) if args.scene_info_path else None,
        instance_id=args.instance_id,
        num_points=args.num_points,
    )

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.print_json:
        print(payload)

    if args.output:
        output_path = Path(args.output)
    else:
        # 默认保存到 results/room_instances/<scene>/ 下，便于按场景整理输出。
        output_dir = DEFAULT_OUTPUT_DIR / args.scene
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.instance_id is None:
            output_path = output_dir / f"room_{args.room_id}_instances.json"
        else:
            output_path = output_dir / f"room_{args.room_id}_instance_{args.instance_id}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(payload)

    point_cloud_saved_path = None
    if args.instance_id is not None:
        point_cloud = result.get("point_cloud", {}) if isinstance(result, dict) else {}
        points = point_cloud.get("points", []) if isinstance(point_cloud, dict) else []
        if args.pointcloud_output:
            pc_path = Path(args.pointcloud_output)
        else:
            pc_ext = ".ply" if args.pointcloud_format == "ply" else ".xyz"
            pc_path = output_path.with_suffix(pc_ext)
        point_cloud_saved_path = _write_point_cloud_file(points, pc_path, fmt=args.pointcloud_format)

    print(f"[OK] JSON 结果已保存到: {output_path}")
    if point_cloud_saved_path is not None:
        print(f"[OK] 点云文件已保存到: {point_cloud_saved_path}")


if __name__ == "__main__":
    main()
