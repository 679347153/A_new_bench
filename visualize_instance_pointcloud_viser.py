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
    seg_colors_segment = np.tile(np.asarray(color, dtype=np.uint8)[None, :], (len(edges), 1))
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


def _add_axes(server: viser.ViserServer) -> None:
    server.scene.add_frame(
        "world_frame",
        wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        show_axes=True,
    )


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

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh is None:
            return False
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


def build_visualization(
    server: viser.ViserServer,
    data: Dict[str, Any],
    show_scene_mesh: bool,
    data_dir: Path,
    scene_name: Optional[str],
    point_cloud_path: Optional[Path] = None,
) -> None:
    _add_axes(server)

    if show_scene_mesh and scene_name:
        _load_scene_mesh_if_available(server, scene_name, data_dir=data_dir)

    _maybe_add_room_and_instance_boxes(server, data)

    points = _load_point_cloud_file(point_cloud_path) if point_cloud_path is not None else np.zeros((0, 3), dtype=np.float32)
    point_source = None
    if point_cloud_path is not None and points.size > 0:
        point_source = f"file:{point_cloud_path.suffix.lower()}"
    if points.size == 0:
        points = _extract_points(data)
        if points.size > 0:
            point_source = "json_embedded"
    if points.size > 0:
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
        _add_room_instances_overview(server, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize exported HM3D instance point clouds with viser")
    parser.add_argument("--input", required=True, help="导出的 instance/room JSON 文件路径")
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "hm3d"), help="HM3D 数据根目录")
    parser.add_argument("--show-scene-mesh", action="store_true", help="同时显示场景网格")
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