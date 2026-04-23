#!/usr/bin/env python3
"""用 viser 可视化 extract_room_instances.py 导出的实例点云。

功能：
1. 可视化单个 instance 的点云。
2. 叠加显示该 instance 的包围盒、所属房间包围盒。
3. 可选加载 HM3D 场景网格，辅助查看点云在场景中的位置。

使用方式示例：
  python visualize_instance_pointcloud_viser.py --input results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.json
  python visualize_instance_pointcloud_viser.py --input results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.json --show-scene-mesh

  python visualize_instance_pointcloud_viser.py \
  --input results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.json \
  --pointcloud-file results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.ply \
  --show-scene-mesh \
  --port 8080
  
如果输入的 JSON 是 room 级别的（包含多个实例），则会显示房间包围盒，并在房间内用小球标记每个实例的中心位置。
如果输入的是 room 级 JSON，而不是 instance 级 JSON，也会尽量把房间内的实例中心和边框显示出来。
"""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hm3d_paths import resolve_scene_paths

try:
    import viser  # type: ignore[import-not-found]
except ImportError as exc:
    raise SystemExit(
        "viser package not found. Install it with: pip install viser"
    ) from exc

try:
    import trimesh  # type: ignore[import-not-found]
except ImportError:
    trimesh = None

try:
    import habitat_sim  # type: ignore[import-not-found]
except ImportError:
    habitat_sim = None


def _as_list3(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    try:
        data = list(value)
        if len(data) >= 3:
            return [float(data[0]), float(data[1]), float(data[2])]
    except Exception:
        pass
    return None


def _as_np3(value: Any) -> np.ndarray:
    """将输入转为 shape=(3,) 的 float32 numpy 向量。"""
    vec = _as_list3(value) or [0.0, 0.0, 0.0]
    return np.asarray(vec, dtype=np.float32).reshape(3)


def _get_attr_or_call(obj: Any, name: str) -> Any:
    if obj is None or not hasattr(obj, name):
        return None
    val = getattr(obj, name)
    try:
        return val() if callable(val) else val
    except Exception:
        return None


def _as_matrix4(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
    except Exception:
        pass
    try:
        rows = []
        for i in range(4):
            row = [float(value[i][j]) for j in range(4)]
            rows.append(row)
        arr = np.asarray(rows, dtype=np.float32)
        if arr.shape == (4, 4):
            return arr
    except Exception:
        pass
    return None


def _bbox_to_corners(bbox: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    min_pt = np.asarray(_as_list3(bbox.get("min")) or [0.0, 0.0, 0.0], dtype=np.float32)
    max_pt = np.asarray(_as_list3(bbox.get("max")) or [0.0, 0.0, 0.0], dtype=np.float32)
    return min_pt, max_pt


def _load_export_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_point_cloud_file(path: Path) -> np.ndarray:
    """从 ply/xyz 文件加载点云。"""
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float32)
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        points: List[List[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except Exception:
                    continue
        arr = np.asarray(points, dtype=np.float32)
        return arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else np.zeros((0, 3), dtype=np.float32)

    if trimesh is not None:
        try:
            loaded = trimesh.load(path, force="scene")
            if hasattr(loaded, "vertices"):
                arr = np.asarray(loaded.vertices, dtype=np.float32)
                return arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else np.zeros((0, 3), dtype=np.float32)
            if hasattr(loaded, "geometry"):
                vertices = []
                for geom in loaded.geometry.values():
                    if hasattr(geom, "vertices"):
                        vertices.append(np.asarray(geom.vertices, dtype=np.float32))
                if vertices:
                    arr = np.concatenate(vertices, axis=0)
                    return arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else np.zeros((0, 3), dtype=np.float32)
        except Exception:
            pass

    return np.zeros((0, 3), dtype=np.float32)


def _extract_points(data: Dict[str, Any]) -> np.ndarray:
    point_cloud = data.get("point_cloud", {})
    points = point_cloud.get("points", []) if isinstance(point_cloud, dict) else []
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return arr[:, :3]


def _make_box_edges(min_pt: np.ndarray, max_pt: np.ndarray) -> np.ndarray:
    x0, y0, z0 = min_pt.tolist()
    x1, y1, z1 = max_pt.tolist()
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    return corners, edges


def _add_bbox_lines(server: viser.ViserServer, name: str, bbox: Dict[str, Any], color: Tuple[int, int, int]) -> None:
    min_pt, max_pt = _bbox_to_corners(bbox)
    if np.allclose(min_pt, max_pt):
        return
    corners, edges = _make_box_edges(min_pt, max_pt)
    # viser expects line segments with shape (N, 2, 3).
    seg_points = corners[edges]
    seg_colors_vertex = np.tile(np.asarray(color, dtype=np.uint8)[None, None, :], (len(edges), 2, 1))
    try:
        server.scene.add_line_segments(
            name=name,
            points=seg_points,
            color=color,
            line_width=2.0,
        )
        return
    except (TypeError, ValueError, AssertionError):
        pass
    try:
        server.scene.add_line_segments(
            name=name,
            points=seg_points,
            colors=seg_colors_vertex,
            line_width=2.0,
        )
        return
    except (TypeError, ValueError, AssertionError):
        pass
    # Last fallback: no color options.
    server.scene.add_line_segments(name=name, points=seg_points)


def _add_axis_lines(server: viser.ViserServer, name: str, length: float = 1.5) -> None:
    """显式绘制 HM3D 基坐标轴线（X=红, Y=绿, Z=蓝）。"""
    origin = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    x_end = np.asarray([length, 0.0, 0.0], dtype=np.float32)
    y_end = np.asarray([0.0, length, 0.0], dtype=np.float32)
    z_end = np.asarray([0.0, 0.0, length], dtype=np.float32)

    segments = np.asarray([
        [origin, x_end],
        [origin, y_end],
        [origin, z_end],
    ], dtype=np.float32)
    colors = np.asarray([
        [[255, 0, 0], [255, 0, 0]],
        [[0, 255, 0], [0, 255, 0]],
        [[0, 0, 255], [0, 0, 255]],
    ], dtype=np.uint8)

    try:
        server.scene.add_line_segments(name=name, points=segments, colors=colors, line_width=3.0)
        return
    except Exception:
        pass
    try:
        server.scene.add_line_segments(name=name, points=segments, colors=colors)
        return
    except Exception:
        pass
    server.scene.add_line_segments(name=name, points=segments)


def _add_axes(server: viser.ViserServer) -> None:
    server.scene.add_frame(
        "world_frame",
        wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        show_axes=True,
    )
    # 明确标注 HM3D 基坐标：默认与 world_frame 重合。
    server.scene.add_frame(
        "hm3d_base_frame",
        wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        show_axes=True,
    )
    _add_axis_lines(server, name="hm3d_base_axes", length=1.8)


def _add_point_cloud(server: viser.ViserServer, name: str, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    if points.size == 0:
        return
    rgb = np.zeros((len(points), 3), dtype=np.uint8)
    rgb[:, 0] = color[0]
    rgb[:, 1] = color[1]
    rgb[:, 2] = color[2]
    try:
        server.scene.add_point_cloud(
            name=name,
            points=points,
            colors=rgb,
            point_size=0.02,
        )
        return
    except TypeError:
        pass
    try:
        server.scene.add_point_cloud(
            name=name,
            points=points,
            colors=rgb,
        )
        return
    except TypeError:
        pass
    server.scene.add_point_cloud(name=name, points=points)


def _load_scene_mesh_if_available(server: viser.ViserServer, scene_name: str, data_dir: Path, alpha: float = 0.25) -> bool:
    if trimesh is None:
        print("[Info] trimesh not available, skip scene mesh visualization.")
        return False

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None:
        print(f"[Info] Cannot resolve scene paths for {scene_name}, skip scene mesh.")
        return False

    mesh_path = scene_paths.stage_glb
    if not mesh_path.is_file():
        print(f"[Info] Scene mesh not found: {mesh_path}")
        return False

    stage_transform = _get_habitat_stage_transform(scene_name=scene_name, data_dir=data_dir)

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh is None:
            return False
        if stage_transform is not None:
            try:
                mesh = mesh.copy()
                mesh.apply_transform(stage_transform)
                print("[Info] Applied habitat stage transform to scene mesh for coordinate alignment.")
            except Exception as exc:
                print(f"[Warning] Failed to apply stage transform: {exc}")
        try:
            server.scene.add_mesh_trimesh(
                name="scene_mesh",
                mesh=mesh,
                color=(180, 180, 180),
                opacity=alpha,
            )
        except TypeError:
            try:
                server.scene.add_mesh_trimesh(
                    name="scene_mesh",
                    mesh=mesh,
                    opacity=alpha,
                )
            except TypeError:
                server.scene.add_mesh_trimesh(
                    name="scene_mesh",
                    mesh=mesh,
                )
        return True
    except Exception as exc:
        print(f"[Warning] Failed to load scene mesh: {exc}")
        return False


def _get_scene_mesh_vertices(scene_name: str, data_dir: Path) -> np.ndarray:
    """加载用于可视化的场景网格顶点（含 stage transform）。"""
    if trimesh is None:
        return np.zeros((0, 3), dtype=np.float32)
    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None or not scene_paths.stage_glb.is_file():
        return np.zeros((0, 3), dtype=np.float32)

    try:
        mesh = trimesh.load(scene_paths.stage_glb, force="mesh")
        if mesh is None or not hasattr(mesh, "vertices"):
            return np.zeros((0, 3), dtype=np.float32)
        stage_transform = _get_habitat_stage_transform(scene_name=scene_name, data_dir=data_dir)
        if stage_transform is not None:
            mesh = mesh.copy()
            mesh.apply_transform(stage_transform)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)
        return vertices[:, :3]
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)


def _get_habitat_stage_transform(scene_name: str, data_dir: Path) -> Optional[np.ndarray]:
    """尝试从 habitat-sim 读取场景舞台变换矩阵（若可用）。"""
    if habitat_sim is None:
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True, root=data_dir)
    if scene_paths is None:
        return None

    sim = None
    try:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
        sim_cfg.scene_id = str(scene_paths.stage_glb)
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        sim_cfg.load_semantic_mesh = True

        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = "color"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [16, 16]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor]

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)

        # 常见路径：scene graph root node transformation。
        graph = _get_attr_or_call(sim, "get_active_scene_graph")
        if graph is not None:
            root_node = _get_attr_or_call(graph, "get_root_node")
            if root_node is None:
                root_node = _get_attr_or_call(graph, "root_node")
            transform = _get_attr_or_call(root_node, "transformation")
            mat = _as_matrix4(transform)
            if mat is not None:
                return mat

        # 如果拿不到，返回单位阵（表示不额外变换）。
        return np.eye(4, dtype=np.float32)
    except Exception as exc:
        print(f"[Info] Cannot query habitat stage transform: {exc}")
        return None
    finally:
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass


def _pick_scene_name(data: Dict[str, Any], input_path: Path) -> Optional[str]:
    if "scene_name" in data:
        return str(data["scene_name"])
    if "scene_info" in data and isinstance(data["scene_info"], dict):
        scene_name = data["scene_info"].get("scene_name")
        if scene_name:
            return str(scene_name)
    name = input_path.name
    if "_instance_" in name:
        return name.split("_instance_", 1)[0]
    if "_instances" in name:
        return name.split("_instances", 1)[0]
    return None


def _maybe_add_room_and_instance_boxes(server: viser.ViserServer, data: Dict[str, Any]) -> None:
    room = data.get("room")
    if isinstance(room, dict):
        bbox = room.get("bounding_box")
        if isinstance(bbox, dict):
            _add_bbox_lines(server, "room_bbox", bbox, (255, 180, 0))

    instance = data.get("instance")
    if isinstance(instance, dict):
        bbox = instance.get("aabb") or instance.get("bbox")
        if isinstance(bbox, dict):
            _add_bbox_lines(server, "instance_bbox", bbox, (0, 200, 255))


def _find_companion_point_cloud_file(input_path: Path, explicit_path: Optional[Path] = None) -> Optional[Path]:
    """查找与 JSON 同名的 ply/xyz 点云文件。"""
    if explicit_path is not None:
        return explicit_path if explicit_path.is_file() else None

    candidates = [
        input_path.with_suffix(".ply"),
        input_path.with_suffix(".xyz"),
        input_path.parent / f"{input_path.stem}.ply",
        input_path.parent / f"{input_path.stem}.xyz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _add_room_instances_overview(server: viser.ViserServer, data: Dict[str, Any]) -> None:
    instances = data.get("instances", [])
    if not isinstance(instances, list) or not instances:
        return

    centers = []
    colors = []
    for idx, inst in enumerate(instances):
        if not isinstance(inst, dict):
            continue
        bbox = inst.get("aabb") or {}
        if not isinstance(bbox, dict):
            continue
        min_pt, max_pt = _bbox_to_corners(bbox)
        if np.allclose(min_pt, max_pt):
            continue
        centers.append(((min_pt + max_pt) / 2.0).tolist())
        hue = (idx * 37) % 255
        colors.append((hue, 255 - hue // 2, 120))

    for i, center in enumerate(centers):
        center_np = _as_np3(center)
        try:
            server.scene.add_sphere(
                name=f"room_instance_center_{i}",
                radius=0.03,
                center=center_np,
                color=colors[i],
            )
        except TypeError:
            try:
                server.scene.add_sphere(
                    name=f"room_instance_center_{i}",
                    radius=0.03,
                    position=center_np,
                    color=colors[i],
                )
            except TypeError:
                server.scene.add_sphere(
                    name=f"room_instance_center_{i}",
                    radius=0.03,
                    center=center_np,
                )


def _rotation_y(yaw_rad: float) -> np.ndarray:
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _mean_nn_distance_sq(src: np.ndarray, dst: np.ndarray, chunk: int = 256) -> float:
    if src.size == 0 or dst.size == 0:
        return float("inf")
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    d2_sum = 0.0
    n = len(src)
    for i in range(0, n, chunk):
        block = src[i:i + chunk]
        diff = block[:, None, :] - dst[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        d2_min = np.min(d2, axis=1)
        d2_sum += float(np.sum(d2_min))
    return d2_sum / float(max(n, 1))


def _transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return (points @ rot.T) + trans[None, :]


def _bbox_corners(bbox: Dict[str, Any]) -> np.ndarray:
    min_pt, max_pt = _bbox_to_corners(bbox)
    x0, y0, z0 = min_pt.tolist()
    x1, y1, z1 = max_pt.tolist()
    return np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float32,
    )


def _transform_bbox_aabb(bbox: Dict[str, Any], rot: np.ndarray, trans: np.ndarray) -> Dict[str, Any]:
    corners = _bbox_corners(bbox)
    tc = _transform_points(corners, rot, trans)
    min_pt = np.min(tc, axis=0)
    max_pt = np.max(tc, axis=0)
    center = (min_pt + max_pt) / 2.0
    size = max_pt - min_pt
    return {
        "min": [round(float(v), 4) for v in min_pt],
        "max": [round(float(v), 4) for v in max_pt],
        "center": [round(float(v), 4) for v in center],
        "size": [round(float(v), 4) for v in size],
    }


def _estimate_transform_by_grid(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    rounds: int = 3,
    yaw_range_deg: float = 180.0,
    yaw_step_deg: float = 30.0,
    trans_range: float = 4.0,
    trans_step: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """遍历法估计 src->dst 的刚体变换（绕Y旋转 + XYZ平移）。"""
    if src_points.size == 0 or dst_points.size == 0:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0.0, float("inf")

    rng = np.random.default_rng(0)
    src = src_points
    dst = dst_points
    if len(src) > 1800:
        src = src[rng.choice(len(src), size=1800, replace=False)]
    if len(dst) > 12000:
        dst = dst[rng.choice(len(dst), size=12000, replace=False)]

    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)

    best_yaw_deg = 0.0
    best_trans = (dst_center - src_center).astype(np.float32)
    best_score = float("inf")

    cur_yaw_range = float(yaw_range_deg)
    cur_yaw_step = float(yaw_step_deg)
    cur_trans_range = float(trans_range)
    cur_trans_step = float(trans_step)

    for _ in range(max(rounds, 1)):
        yaw_values = np.arange(best_yaw_deg - cur_yaw_range, best_yaw_deg + cur_yaw_range + 1e-6, cur_yaw_step)
        tx_values = np.arange(best_trans[0] - cur_trans_range, best_trans[0] + cur_trans_range + 1e-6, cur_trans_step)
        ty_values = np.arange(best_trans[1] - cur_trans_range, best_trans[1] + cur_trans_range + 1e-6, cur_trans_step)
        tz_values = np.arange(best_trans[2] - cur_trans_range, best_trans[2] + cur_trans_range + 1e-6, cur_trans_step)

        for yaw_deg in yaw_values:
            rot = _rotation_y(np.deg2rad(yaw_deg))
            src_rot = src @ rot.T
            for tx in tx_values:
                for ty in ty_values:
                    for tz in tz_values:
                        trans = np.asarray([tx, ty, tz], dtype=np.float32)
                        moved = src_rot + trans[None, :]
                        score = _mean_nn_distance_sq(moved, dst)
                        if score < best_score:
                            best_score = score
                            best_yaw_deg = float(yaw_deg)
                            best_trans = trans

        cur_yaw_range = max(cur_yaw_range * 0.45, 2.0)
        cur_yaw_step = max(cur_yaw_step * 0.5, 1.0)
        cur_trans_range = max(cur_trans_range * 0.45, 0.2)
        cur_trans_step = max(cur_trans_step * 0.5, 0.05)

    best_rot = _rotation_y(np.deg2rad(best_yaw_deg))
    return best_rot, best_trans, best_yaw_deg, best_score


def build_visualization(
    server: viser.ViserServer,
    data: Dict[str, Any],
    show_scene_mesh: bool,
    data_dir: Path,
    scene_name: Optional[str],
    point_cloud_path: Optional[Path] = None,
    auto_align_grid: bool = False,
) -> None:
    _add_axes(server)

    if show_scene_mesh and scene_name:
        _load_scene_mesh_if_available(server, scene_name, data_dir=data_dir)

    points = _load_point_cloud_file(point_cloud_path) if point_cloud_path is not None else np.zeros((0, 3), dtype=np.float32)
    point_source = None
    if point_cloud_path is not None and points.size > 0:
        point_source = f"file:{point_cloud_path.suffix.lower()}"
    if points.size == 0:
        points = _extract_points(data)
        if points.size > 0:
            point_source = "json_embedded"
    if points.size > 0:
        if auto_align_grid and scene_name:
            mesh_vertices = _get_scene_mesh_vertices(scene_name=scene_name, data_dir=data_dir)
            if mesh_vertices.size > 0:
                print("[Info] Running traversal search for coordinate transform (yaw+translation)...")
                rot, trans, yaw_deg, score = _estimate_transform_by_grid(
                    src_points=points,
                    dst_points=mesh_vertices,
                )
                points = _transform_points(points, rot, trans)
                print(
                    "[Info] Estimated transform: "
                    f"yaw_deg={yaw_deg:.3f}, "
                    f"translation=({trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}), "
                    f"mean_nn_dist={np.sqrt(max(score, 0.0)):.4f}"
                )

                # 同步变换包围盒，确保点云与框体仍一致。
                room = data.get("room")
                if isinstance(room, dict) and isinstance(room.get("bounding_box"), dict):
                    room["bounding_box"] = _transform_bbox_aabb(room["bounding_box"], rot, trans)
                instance = data.get("instance")
                if isinstance(instance, dict) and isinstance(instance.get("aabb"), dict):
                    instance["aabb"] = _transform_bbox_aabb(instance["aabb"], rot, trans)
            else:
                print("[Info] Auto-align skipped: scene mesh vertices unavailable.")

        # 在 auto align 可能更新了 bbox 后再绘制包围盒。
        _maybe_add_room_and_instance_boxes(server, data)

        _add_point_cloud(server, "instance_point_cloud", points, (30, 180, 255))
        server.scene.add_frame(
            "point_cloud_origin",
            position=np.asarray(points.mean(axis=0), dtype=np.float32).reshape(3),
            wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            show_axes=True,
        )
        if point_source:
            print(f"[Info] Loaded point cloud from: {point_source}")
    else:
        _maybe_add_room_and_instance_boxes(server, data)
        _add_room_instances_overview(server, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize exported HM3D instance point clouds with viser")
    parser.add_argument("--input", required=True, help="导出的 instance/room JSON 文件路径")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "hm3d"), help="HM3D 数据根目录")
    parser.add_argument("--show-scene-mesh", action="store_true", help="同时显示场景网格")
    parser.add_argument(
        "--auto-align-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用遍历法自动估计点云到场景网格的变换关系（默认开启）",
    )
    parser.add_argument("--pointcloud-file", type=str, default=None, help="可选：显式指定点云文件（.ply/.xyz）")
    parser.add_argument("--port", type=int, default=8080, help="viser 服务端口，0 表示自动分配")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    data = _load_export_json(input_path)
    scene_name = _pick_scene_name(data, input_path)
    explicit_pc = Path(args.pointcloud_file).expanduser().resolve() if args.pointcloud_file else None
    companion_pc = _find_companion_point_cloud_file(input_path, explicit_pc)
    if companion_pc is not None:
        print(f"[Info] Point cloud file detected: {companion_pc}")

    try:
        server = viser.ViserServer(port=args.port if args.port >= 0 else 8080)
    except Exception as exc:
        print(f"[Warning] Failed to bind requested port {args.port}: {exc}. Using auto port instead.")
        server = viser.ViserServer(port=0)

    build_visualization(
        server=server,
        data=data,
        show_scene_mesh=args.show_scene_mesh,
        data_dir=Path(args.data_dir),
        scene_name=scene_name,
        point_cloud_path=companion_pc,
        auto_align_grid=args.auto_align_grid,
    )

    print("[OK] viser server started.")
    print("[OK] Open the printed URL in your browser to inspect the scene.")
    if "point_cloud_generation" in data:
        print(f"[Info] Point cloud source: {data['point_cloud_generation'].get('method', 'unknown')}")
    print("[Info] Press Ctrl+C to stop the server.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[OK] Server stopped.")


if __name__ == "__main__":
    main()