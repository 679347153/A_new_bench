#!/usr/bin/env python3
"""用 viser 可视化 extract_room_instances.py 导出的实例点云，并支持终端交互式调参。

本文件解决的问题
----------------
1) 把 instance 点云、instance 包围盒、room 包围盒放在同一 3D 视图中查看。
2) 可选叠加 HM3D 场景网格，直观检查点云与场景是否对齐。
3) 支持在终端中反复输入旋转参数，实时重绘 instance 相关元素，直到满意为止。

输入数据约定
-----------
1) --input 指向 extract_room_instances.py 导出的 JSON。
2) 点云优先级：
    - 若指定 --pointcloud-file，则优先读取该 .ply/.xyz。
    - 否则自动查找与 JSON 同名的 .ply/.xyz。
    - 再否则回退到 JSON 内嵌 point_cloud.points。
3) 当输入为 room 级 JSON（包含 instances 列表）时，会显示房间框，并尽量显示实例中心标记。

核心运行流程
-----------
1) 解析 JSON 与可选点云文件。
2) 启动 viser 服务器并绘制坐标轴。
3) 可选加载场景网格（--show-scene-mesh）。
4) 对点云应用变换后重绘：
    - 自动模式：90 度离散遍历，估计最优朝向。
    - 手动模式：使用终端输入的角度直接旋转。
5) 进入交互循环，你可以持续 set/step/reload/auto，观察浏览器视图变化。

坐标变换说明
-----------
1) 本脚本当前只做“旋转朝向”变换，不做平移偏置。
2) 自动对齐只遍历 rx/ry/rz in {0, 90, 180, 270}，共 64 组组合。
3) 打分前会对点云和场景点去中心，以减少平移对朝向估计的干扰。

交互命令
--------
启动后进入 transform> 提示符，可用命令如下：
1) set RX RY RZ
    直接设置绝对角度（度），例如：set 0 0 180
2) step AXIS DIR
    按 90 度步进，AXIS 为 x|y|z，DIR 为 +|-，例如：step z +
3) reload
    按当前角度重新加载并重绘 instance 相关元素
4) auto
    重新执行 90 度离散遍历，自动估计最佳朝向
5) show / help / quit

快速示例
--------
1) 最小运行：
    python visualize_instance_pointcloud_viser.py \
      --input results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.json

2) 带场景网格：
    python visualize_instance_pointcloud_viser.py \
      --input results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.json \
      --show-scene-mesh

3) 指定点云文件与端口：
    python visualize_instance_pointcloud_viser.py \
      --input results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.json \
      --pointcloud-file results/room_instances/00808-y9hTuugGdiq/room_0_instance_1.ply \
      --show-scene-mesh \
      --port 8080

依赖
----
必需：viser, numpy
可选：trimesh（加载场景网格/ply 时更稳定），habitat_sim（读取 stage transform）
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


def _rotation_xyz_deg(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """按 XYZ 欧拉角（度）构造旋转矩阵。"""
    rx = float(np.deg2rad(rx_deg))
    ry = float(np.deg2rad(ry_deg))
    rz = float(np.deg2rad(rz_deg))

    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))

    rot_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32
    )
    rot_y = np.asarray(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32
    )
    rot_z = np.asarray(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )

    # 约定按 X->Y->Z 顺序作用（右乘向量时等价于 R = Rz * Ry * Rx）。
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


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


def _transform_points(points: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return points @ rot.T


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


def _transform_bbox_aabb(bbox: Dict[str, Any], rot: np.ndarray) -> Dict[str, Any]:
    corners = _bbox_corners(bbox)
    tc = _transform_points(corners, rot)
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


def _estimate_rotation_by_90deg(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int], float]:
    """遍历法估计 src->dst 的朝向变换（仅旋转，不平移）。

    遍历集合：rx, ry, rz in {0, 90, 180, 270}，共 64 组。
    评分时先对 src/dst 去中心，避免平移影响朝向匹配。
    """
    if src_points.size == 0 or dst_points.size == 0:
        return np.eye(3, dtype=np.float32), (0, 0, 0), float("inf")

    rng = np.random.default_rng(0)
    src = src_points
    dst = dst_points
    if len(src) > 1800:
        src = src[rng.choice(len(src), size=1800, replace=False)]
    if len(dst) > 12000:
        dst = dst[rng.choice(len(dst), size=12000, replace=False)]

    src_centered = src - np.mean(src, axis=0, keepdims=True)
    dst_centered = dst - np.mean(dst, axis=0, keepdims=True)

    best_rot = np.eye(3, dtype=np.float32)
    best_angles = (0, 0, 0)
    best_score = float("inf")

    for rx in (0, 90, 180, 270):
        for ry in (0, 90, 180, 270):
            for rz in (0, 90, 180, 270):
                rot = _rotation_xyz_deg(rx, ry, rz)
                moved = _transform_points(src_centered, rot)
                score = _mean_nn_distance_sq(moved, dst_centered)
                if score < best_score:
                    best_score = score
                    best_rot = rot
                    best_angles = (rx, ry, rz)

    return best_rot, best_angles, best_score


def _safe_scene_reset(server: viser.ViserServer) -> None:
    """尽量清空场景，便于同一进程内重复重绘。"""
    try:
        server.scene.reset()
    except Exception as exc:
        print(f"[Warning] Scene reset not supported in this viser version: {exc}")


def _normalize_angles_deg(rx: float, ry: float, rz: float) -> Tuple[float, float, float]:
    return (float(rx) % 360.0, float(ry) % 360.0, float(rz) % 360.0)


def build_visualization(
    server: viser.ViserServer,
    data: Dict[str, Any],
    show_scene_mesh: bool,
    data_dir: Path,
    scene_name: Optional[str],
    point_cloud_path: Optional[Path] = None,
    auto_align_grid: bool = False,
    manual_angles_deg: Optional[Tuple[float, float, float]] = None,
    reset_scene: bool = False,
) -> Dict[str, Any]:
    if reset_scene:
        _safe_scene_reset(server)

    render_data = deepcopy(data)

    _add_axes(server)

    if show_scene_mesh and scene_name:
        _load_scene_mesh_if_available(server, scene_name, data_dir=data_dir)

    points = _load_point_cloud_file(point_cloud_path) if point_cloud_path is not None else np.zeros((0, 3), dtype=np.float32)
    point_source = None
    if point_cloud_path is not None and points.size > 0:
        point_source = f"file:{point_cloud_path.suffix.lower()}"
    if points.size == 0:
        points = _extract_points(render_data)
        if points.size > 0:
            point_source = "json_embedded"

    used_angles = (0.0, 0.0, 0.0)
    used_score = None
    align_mode = "identity"

    if points.size > 0:
        if manual_angles_deg is not None:
            used_angles = _normalize_angles_deg(*manual_angles_deg)
            rot = _rotation_xyz_deg(used_angles[0], used_angles[1], used_angles[2])
            points = _transform_points(points, rot)
            align_mode = "manual"
            print(
                "[Info] Manual orientation applied: "
                f"rx={used_angles[0]:.1f} deg, ry={used_angles[1]:.1f} deg, rz={used_angles[2]:.1f} deg"
            )

            room = render_data.get("room")
            if isinstance(room, dict) and isinstance(room.get("bounding_box"), dict):
                room["bounding_box"] = _transform_bbox_aabb(room["bounding_box"], rot)
            instance = render_data.get("instance")
            if isinstance(instance, dict) and isinstance(instance.get("aabb"), dict):
                instance["aabb"] = _transform_bbox_aabb(instance["aabb"], rot)
        elif auto_align_grid and scene_name:
            mesh_vertices = _get_scene_mesh_vertices(scene_name=scene_name, data_dir=data_dir)
            if mesh_vertices.size > 0:
                print("[Info] Running orientation-only traversal (Rx/Ry/Rz in 90-degree steps)...")
                rot, angles, score = _estimate_rotation_by_90deg(
                    src_points=points,
                    dst_points=mesh_vertices,
                )
                points = _transform_points(points, rot)
                used_angles = (float(angles[0]), float(angles[1]), float(angles[2]))
                used_score = float(score)
                align_mode = "auto_90deg"
                print(
                    "[Info] Estimated orientation: "
                    f"rx={angles[0]} deg, ry={angles[1]} deg, rz={angles[2]} deg, "
                    f"mean_nn_dist={np.sqrt(max(score, 0.0)):.4f}"
                )

                # 同步旋转包围盒，确保点云与框体仍一致。
                room = render_data.get("room")
                if isinstance(room, dict) and isinstance(room.get("bounding_box"), dict):
                    room["bounding_box"] = _transform_bbox_aabb(room["bounding_box"], rot)
                instance = render_data.get("instance")
                if isinstance(instance, dict) and isinstance(instance.get("aabb"), dict):
                    instance["aabb"] = _transform_bbox_aabb(instance["aabb"], rot)
            else:
                print("[Info] Auto-align skipped: scene mesh vertices unavailable.")

        # 在 auto align 可能更新了 bbox 后再绘制包围盒。
        _maybe_add_room_and_instance_boxes(server, render_data)

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
        _maybe_add_room_and_instance_boxes(server, render_data)
        _add_room_instances_overview(server, render_data)

    return {
        "used_angles_deg": used_angles,
        "align_mode": align_mode,
        "score": used_score,
        "point_source": point_source,
    }


def _print_interactive_help() -> None:
    print("[Interactive] Commands:")
    print("  set RX RY RZ   -> 设置绝对旋转角度（单位：度）")
    print("  step AXIS DIR  -> 按 90 度步进，AXIS:x|y|z, DIR:+|-  (例如: step z +)")
    print("  auto           -> 重新执行 90 度遍历自动对齐")
    print("  reload         -> 按当前角度重新加载 instance 元素")
    print("  show           -> 显示当前角度")
    print("  help           -> 显示帮助")
    print("  quit           -> 退出")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize HM3D instance point clouds with viser and interactive terminal transform tuning.",
        epilog=(
            "Interactive commands after startup:\n"
            "  set RX RY RZ   set absolute rotation in degrees\n"
            "  step AXIS DIR  rotate by 90 deg, AXIS in x|y|z, DIR in +|-\n"
            "  auto           run 90-degree discrete orientation search\n"
            "  reload         redraw with current angles\n"
            "  show/help/quit"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
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

    result = build_visualization(
        server=server,
        data=data,
        show_scene_mesh=args.show_scene_mesh,
        data_dir=Path(args.data_dir),
        scene_name=scene_name,
        point_cloud_path=companion_pc,
        auto_align_grid=args.auto_align_grid,
        manual_angles_deg=None,
        reset_scene=True,
    )

    current_angles = (
        float(result.get("used_angles_deg", (0.0, 0.0, 0.0))[0]),
        float(result.get("used_angles_deg", (0.0, 0.0, 0.0))[1]),
        float(result.get("used_angles_deg", (0.0, 0.0, 0.0))[2]),
    )

    print("[OK] viser server started.")
    print("[OK] Open the printed URL in your browser to inspect the scene.")
    if "point_cloud_generation" in data:
        print(f"[Info] Point cloud source: {data['point_cloud_generation'].get('method', 'unknown')}")
    print("[Info] Enter interactive mode to iteratively adjust transform.")
    print(f"[Info] Current orientation: rx={current_angles[0]:.1f}, ry={current_angles[1]:.1f}, rz={current_angles[2]:.1f}")
    _print_interactive_help()

    try:
        while True:
            command = input("transform> ").strip()
            if not command:
                continue

            low = command.lower()
            if low in {"quit", "q", "exit"}:
                break
            if low in {"help", "h", "?"}:
                _print_interactive_help()
                continue
            if low == "show":
                print(
                    f"[Info] Current orientation: "
                    f"rx={current_angles[0]:.1f}, ry={current_angles[1]:.1f}, rz={current_angles[2]:.1f}"
                )
                continue
            if low == "auto":
                result = build_visualization(
                    server=server,
                    data=data,
                    show_scene_mesh=args.show_scene_mesh,
                    data_dir=Path(args.data_dir),
                    scene_name=scene_name,
                    point_cloud_path=companion_pc,
                    auto_align_grid=True,
                    manual_angles_deg=None,
                    reset_scene=True,
                )
                used = result.get("used_angles_deg", current_angles)
                current_angles = _normalize_angles_deg(float(used[0]), float(used[1]), float(used[2]))
                continue
            if low == "reload":
                build_visualization(
                    server=server,
                    data=data,
                    show_scene_mesh=args.show_scene_mesh,
                    data_dir=Path(args.data_dir),
                    scene_name=scene_name,
                    point_cloud_path=companion_pc,
                    auto_align_grid=False,
                    manual_angles_deg=current_angles,
                    reset_scene=True,
                )
                continue

            parts = command.split()
            if len(parts) == 4 and parts[0].lower() == "set":
                try:
                    rx = float(parts[1])
                    ry = float(parts[2])
                    rz = float(parts[3])
                except ValueError:
                    print("[Error] Invalid numbers. Usage: set RX RY RZ")
                    continue
                current_angles = _normalize_angles_deg(rx, ry, rz)
                build_visualization(
                    server=server,
                    data=data,
                    show_scene_mesh=args.show_scene_mesh,
                    data_dir=Path(args.data_dir),
                    scene_name=scene_name,
                    point_cloud_path=companion_pc,
                    auto_align_grid=False,
                    manual_angles_deg=current_angles,
                    reset_scene=True,
                )
                continue

            if len(parts) == 3 and parts[0].lower() == "step":
                axis = parts[1].lower()
                direction = parts[2]
                if axis not in {"x", "y", "z"} or direction not in {"+", "-"}:
                    print("[Error] Usage: step AXIS DIR, e.g. step z +")
                    continue
                delta = 90.0 if direction == "+" else -90.0
                rx, ry, rz = current_angles
                if axis == "x":
                    rx += delta
                elif axis == "y":
                    ry += delta
                else:
                    rz += delta
                current_angles = _normalize_angles_deg(rx, ry, rz)
                build_visualization(
                    server=server,
                    data=data,
                    show_scene_mesh=args.show_scene_mesh,
                    data_dir=Path(args.data_dir),
                    scene_name=scene_name,
                    point_cloud_path=companion_pc,
                    auto_align_grid=False,
                    manual_angles_deg=current_angles,
                    reset_scene=True,
                )
                continue

            print("[Error] Unknown command. Type 'help' to see supported commands.")
    except KeyboardInterrupt:
        pass

    print("[OK] Server stopped.")


if __name__ == "__main__":
    main()