#!/usr/bin/env python3
"""HM3D 房间实例提取工具。

这个脚本的目标很单一：给定一个 HM3D 场景和一个房间编号，把这个房间里
的所有 instance 取出来；如果再给一个具体 instance_id，则进一步返回该
instance 的点云信息。

运行逻辑分成 4 步：
1. 读取 scene_info JSON，拿到 rooms / objects 的结构化信息。
2. 按 region_id 过滤出目标房间，再按 object_ids 或 region_id 关联该房间内的 instance。
3. 如果用户要求单个 instance 的点云，则优先尝试从 habitat-sim 的语义对象上直接
     读取点云/顶点信息。
4. 如果当前版本的 habitat-sim 没有暴露可直接读取的数据，则退化为基于 AABB 的
     表面采样点云，并在结果里标明 source，方便你区分“真实点云”和“几何近似点云”。

输出形式：
    - 仅房间：返回 room 元信息 + 该房间内的 instance 列表。
    - 指定 instance：返回 room 元信息 + instance 信息 + point_cloud 摘要。

推荐用法：
    python extract_room_instances.py --scene 00824-Dd4bFSTQ8gi --room-id 0
    python extract_room_instances.py --scene 00824-Dd4bFSTQ8gi --room-id 0 --instance-id 1

默认会优先读取已导出的 scene_info JSON；若未找到，可通过 --scene-info-path 显式指定。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hm3d_paths import resolve_scene_paths

try:
    import habitat_sim  # type: ignore[import-not-found]
except ImportError:
    habitat_sim = None


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
    return all(abs(float(v)) < 1e-8 for v in [*min_pt, *max_pt])


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


def get_room_instances(scene_info: Dict[str, Any], room_id: int) -> Dict[str, Any]:
    """根据房间号从 scene_info 中提取该房间内的所有 instance。

    优先使用 room 里的 object_ids 做精确关联；如果没有 object_ids，则退化为按
    region_id 过滤 objects。这样可以兼容不同版本的 scene_info 导出结构。
    """
    rooms = scene_info.get("rooms", []) or []
    objects = scene_info.get("objects", []) or []
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
    }


def get_instance_point_cloud(
    scene_name: str,
    instance_id: int,
    data_dir: Path = DEFAULT_DATA_DIR,
    num_points: int = 2048,
) -> Dict[str, Any]:
    """获取单个 instance 的点云信息。

    处理顺序：
    1. 从 scene_info 中定位 instance 的 AABB 和基础元数据。
    2. 如果 habitat-sim 可用，尝试从语义对象直接读取点云或顶点。
    3. 如果直接读取失败，则用 AABB 表面采样生成近似点云。
    """
    scene_info = _load_scene_info(scene_name, data_dir)
    generation_trace: List[str] = [
        "loaded_scene_info",
        f"target_instance_id={int(instance_id)}",
    ]
    objects = scene_info.get("objects", []) or []
    instance = next((obj for obj in objects if int(obj.get("id", -1)) == int(instance_id)), None)
    if instance is None:
        raise KeyError(f"找不到 instance_id={instance_id}")

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
                    "将回退为 AABB 表面采样点云。"
                )
                generation_trace.append("direct_point_cloud_unavailable")
            else:
                _warn(
                    "habitat-sim 已初始化，但 semantic_scene.objects 中未找到目标 instance，"
                    "将回退为 AABB 表面采样点云。"
                )
                generation_trace.append("semantic_object_not_found")
        else:
            _warn(
                "habitat-sim 不可用或场景无法初始化，无法直接读取实例几何；"
                "将回退为 AABB 表面采样点云。"
            )
            generation_trace.append("simulator_unavailable")

        bbox = instance.get("aabb") or {}
        min_pt = bbox.get("min", [0.0, 0.0, 0.0])
        max_pt = bbox.get("max", [0.0, 0.0, 0.0])
        if _bbox_is_zero({"min": min_pt, "max": max_pt}):
            # 如果连 AABB 都不可用，就返回空点云，避免伪造几何。
            _warn(
                "instance 的 AABB 为全零或缺失，无法进行几何采样，最终输出为空点云。"
            )
            points = np.zeros((0, 3), dtype=np.float32)
            generation_trace.append("fallback=empty_point_cloud_due_to_zero_bbox")
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
            "point_cloud": _summarize_point_cloud(points, "sampled_aabb_surface"),
            "point_cloud_generation": {
                "method": "sampled_aabb_surface",
                "details": "fallback sampling from instance AABB surfaces because direct semantic point cloud is unavailable",
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
    room_report = get_room_instances(scene_info, room_id)

    if instance_id is not None:
        matched = next((obj for obj in room_report["instances"] if int(obj.get("id", -1)) == int(instance_id)), None)
        if matched is None:
            raise KeyError(f"instance_id={instance_id} 不属于 room_id={room_id}")

        point_cloud_report = get_instance_point_cloud(
            scene_name=scene_name,
            instance_id=instance_id,
            data_dir=data_dir,
            num_points=num_points,
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
    args = parser.parse_args()

    result = extract_room_instances(
        scene_name=args.scene,
        room_id=args.room_id,
        data_dir=Path(args.data_dir),
        scene_info_path=Path(args.scene_info_path) if args.scene_info_path else None,
        instance_id=args.instance_id,
        num_points=args.num_points,
    )

    # 标准输出便于管道处理或临时查看。
    payload = json.dumps(result, ensure_ascii=False, indent=2)
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

    print(f"\n[OK] 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()