#!/usr/bin/env python3
"""
导出 HM3D 场景的完整语义信息为 JSON。

输出内容：
  - scene_info: 场景级元数据（名称、AABB、可导航面积）
  - categories: 所有物体类别汇总
  - rooms: 每个房间及其包含的物体列表
  - objects: 所有物体的详细信息（类别、3D中心、包围盒、所属房间）

用法：
  # 导出单个场景
  python export_scene_info.py --scene 00824-Dd4bFSTQ8gi

  # 导出 seap_test/data 下所有场景
  python export_scene_info.py --all

  # 指定数据目录和输出目录
  python export_scene_info.py --scene 00824-Dd4bFSTQ8gi --data-dir /path/to/data --output-dir /path/to/output

实现思路概览：
    1) 读取 semantic.txt，建立 semantic_id -> 类别/房间映射
    2) 通过 habitat_sim 读取语义场景对象及其 3D 包围盒
    3) 将对象按 region_id 聚合为房间，并统计类别分布
    4) 生成统一 JSON（scene_info / categories / rooms / objects）
"""

import argparse
import csv
import json
import os
import sys
import io
import glob

import numpy as np

try:
    import habitat_sim
except ImportError:
    print("Error: habitat_sim not found. 请在 agentrag 环境中运行。")
    sys.exit(1)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, "hm3d", "minival")
DEFAULT_DATASET_CONFIG = os.path.join(SCRIPT_DIR, "hm3d", "hm3d_annotated_basis.scene_dataset_config.json")


def parse_semantic_txt(semantic_txt_path):
    """
    解析 HM3D 的 semantic.txt。

    参数:
        semantic_txt_path: semantic.txt 文件路径。

    返回:
        dict，键为 semantic_id(int)，值为:
        {
            "category": str,
            "region_id": int,
            "color_hex": str
        }

    说明:
        - semantic.txt 通常首行为表头，因此 line_no == 0 时跳过。
        - 单行格式近似为: id,color_hex,"category",region_id
        - 使用 csv.reader 处理，避免类别名中潜在逗号导致的分割错误。
    """
    entries = {}
    with open(semantic_txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line_no == 0 or not line:
                continue
            # 格式: id,color_hex,"category",region_id
            reader = csv.reader(io.StringIO(line))
            for row in reader:
                # 最小字段数校验，避免脏数据导致索引越界。
                if len(row) < 4:
                    continue
                obj_id = int(row[0])
                color_hex = row[1].strip()
                category = row[2].strip().strip('"')
                region_id = int(row[3])
                entries[obj_id] = {
                    "category": category,
                    "region_id": region_id,
                    "color_hex": color_hex,
                }
    return entries


def make_sim(scene_glb_path, dataset_config):
    """
    创建一个最小化 habitat_sim Simulator，仅用于读取语义信息。

    参数:
        scene_glb_path: 场景 mesh 文件路径（通常是 *.basis.glb）。
        dataset_config: scene_dataset_config.json 路径。

    返回:
        habitat_sim.Simulator 实例。

    说明:
        - 此处不做物理仿真，只读取语义图与包围盒，所以 enable_physics=False。
        - scene_id 采用文件名去后缀的方式推导，并兼容 *.basis.glb -> scene_id。
        - 传感器配置是最小可运行配置，脚本并不依赖图像输出。
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.abspath(dataset_config)
    # 直接使用场景文件绝对路径，避免不同版本对 scene_id 解析差异。
    sim_cfg.scene_id = os.path.abspath(scene_glb_path)
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    # 语义网格必须开启，否则许多对象/房间几何会退化为默认值。
    sim_cfg.load_semantic_mesh = True

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [480, 640]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def vec3_to_list(v):
    """将 habitat/magnum 的 vec3 转为保留 4 位小数的 Python 列表。"""
    return [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]


def _as_vec3(v):
    """将不同类型的向量对象统一转成 [x, y, z]，兼容旧版绑定对象。"""
    if v is None:
        return None
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return [round(float(v.x), 4), round(float(v.y), 4), round(float(v.z), 4)]
    try:
        return [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
    except Exception:
        pass
    try:
        vv = list(v)
        if len(vv) >= 3:
            return [round(float(vv[0]), 4), round(float(vv[1]), 4), round(float(vv[2]), 4)]
    except Exception:
        pass
    return None


def _get_attr_or_call(obj, name):
    if not hasattr(obj, name):
        return None
    value = getattr(obj, name)
    try:
        return value() if callable(value) else value
    except Exception:
        return None


def _extract_bbox_info(bbox):
    """兼容 habitat-sim 不同版本的 BBox 接口。"""
    min_vec = None
    max_vec = None

    min_candidates = ["min", "min_corner", "back_bottom_left", "back_bottom_left_corner"]
    max_candidates = ["max", "max_corner", "front_top_right", "front_top_right_corner"]

    for key in min_candidates:
        min_vec = _as_vec3(_get_attr_or_call(bbox, key))
        if min_vec is not None:
            break

    for key in max_candidates:
        max_vec = _as_vec3(_get_attr_or_call(bbox, key))
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


def _extract_obb_rotation(obb):
    """尽可能提取 3x3 旋转矩阵；不同版本失败时返回 None。"""
    rot = _get_attr_or_call(obb, "rotation")
    if rot is None:
        return None
    try:
        return [
            [round(float(rot[i][j]), 6) for j in range(3)]
            for i in range(3)
        ]
    except Exception:
        pass

    for method_name in ["to_matrix", "toMatrix"]:
        if hasattr(rot, method_name):
            try:
                mat = getattr(rot, method_name)()
                return [
                    [round(float(mat[i][j]), 6) for j in range(3)]
                    for i in range(3)
                ]
            except Exception:
                continue
    return None


def _bbox_is_zero(bbox_info):
    min_pt = bbox_info.get("min", [0.0, 0.0, 0.0])
    max_pt = bbox_info.get("max", [0.0, 0.0, 0.0])
    return all(abs(float(v)) < 1e-8 for v in [*min_pt, *max_pt])


def export_scene(scene_name, data_dir, dataset_config, output_dir):
    """
    导出单个场景信息并写入 JSON。

    参数:
        scene_name: 场景目录名，例如 00824-Dd4bFSTQ8gi。
        data_dir: 场景根目录。
        dataset_config: scene_dataset_config.json 路径。
        output_dir: 输出目录。

    返回:
        成功时返回 result(dict)，失败/跳过时返回 None。

    产出 JSON 结构:
        - scene_info: 场景级统计和场景 AABB
        - categories: 类别计数列表（按数量降序）
        - rooms: 按 region_id 聚合的房间统计
        - objects: 每个对象的几何和语义信息
    """
    scene_dir = os.path.join(data_dir, scene_name)
    if not os.path.isdir(scene_dir):
        print(f"  [跳过] 目录不存在: {scene_dir}")
        return None

    parts = scene_name.split("-", 1)
    scene_id = parts[1] if len(parts) > 1 else scene_name

    # 目录名常见格式是 "00824-Dd4bFSTQ8gi"，scene_id 取 '-' 后半段。
    # 如果没有 '-'，则直接退化为 scene_name。

    # 检查必需文件。
    # 注意：navmesh 可选（缺失时仅无法输出 navigable_area_m2）。
    basis_glb = os.path.join(scene_dir, f"{scene_id}.basis.glb")
    semantic_txt = os.path.join(scene_dir, f"{scene_id}.semantic.txt")
    navmesh = os.path.join(scene_dir, f"{scene_id}.basis.navmesh")

    if not os.path.isfile(basis_glb):
        print(f"  [跳过] 缺少 basis.glb: {basis_glb}")
        return None
    if not os.path.isfile(semantic_txt):
        print(f"  [跳过] 缺少 semantic.txt: {semantic_txt}")
        return None

    # 1) 解析 semantic.txt，得到语义 ID 到类别/房间的映射。
    txt_entries = parse_semantic_txt(semantic_txt)

    # 2) 通过 habitat_sim 读取语义场景与对象几何信息。
    sim = make_sim(basis_glb, dataset_config)
    sem_scene = sim.semantic_scene

    # 场景 AABB
    scene_aabb = sem_scene.aabb
    scene_aabb_info = _extract_bbox_info(scene_aabb)

    # 可导航面积（单位 m^2）。
    # pathfinder 依赖可用 navmesh，若读取失败则保持 None。
    nav_area = None
    if os.path.isfile(navmesh):
        try:
            nav_area = round(float(sim.pathfinder.navigable_area), 4)
        except Exception:
            pass

    # 3) 构建物体列表，并同时做类别计数和房间聚合。
    objects_list = []
    category_counter = {}
    room_objects = {}  # region_id -> [obj_info, ...]

    # 预取 region 几何，作为对象 AABB 异常时的回退。
    region_bbox_map = {}
    for region in getattr(sem_scene, "regions", []) or []:
        if region is None:
            continue
        rid = getattr(region, "id", None)
        if rid is None:
            continue
        rbbox = _extract_bbox_info(getattr(region, "aabb", None))
        if not _bbox_is_zero(rbbox):
            region_bbox_map[int(rid)] = rbbox

    sem_objects = sem_scene.objects
    for obj in sem_objects:
        sid = obj.semantic_id
        # 以 semantic.txt 为主数据源；若缺失，则退化到 sim 内建类别信息。
        txt_info = txt_entries.get(sid, {})
        cat = txt_info.get("category", obj.category.name() if obj.category else "unknown")
        region_id = txt_info.get("region_id", -1)
        color_hex = txt_info.get("color_hex", "")

        # 3D AABB
        aabb = obj.aabb
        aabb_info = _extract_bbox_info(aabb)

        # OBB (有向包围盒)
        obb = obj.obb
        obb_center = _as_vec3(_get_attr_or_call(obb, "center"))
        if obb_center is None:
            obb_center = [0.0, 0.0, 0.0]

        obb_half_extents = _as_vec3(_get_attr_or_call(obb, "half_extents"))
        if obb_half_extents is None:
            obb_half_extents = _as_vec3(_get_attr_or_call(obb, "halfExtents"))
        if obb_half_extents is None:
            obb_half_extents = [0.0, 0.0, 0.0]
        # 输出 3x3 旋转矩阵，旧版本绑定若不支持则返回 None。
        obb_rotation = _extract_obb_rotation(obb)

        obj_info = {
            "id": sid,
            "category": cat,
            "region_id": region_id,
            "color_hex": color_hex,
            "aabb": {
                "min": aabb_info["min"],
                "max": aabb_info["max"],
                "center": aabb_info["center"],
                "size": aabb_info["size"],
            },
            "obb": {
                "center": obb_center,
                "half_extents": obb_half_extents,
                "rotation": obb_rotation,
            },
        }
        objects_list.append(obj_info)

        # 统计
        category_counter[cat] = category_counter.get(cat, 0) + 1
        room_objects.setdefault(region_id, []).append(obj_info)

    # 4) 构建房间列表。
    # 房间 bounding_box 采用“该房间所有对象 AABB 的并集”近似得到。
    rooms_list = []
    for rid in sorted(room_objects.keys()):
        room_objs = room_objects[rid]

        # 从房间内所有“非零AABB”物体估算范围，避免全零几何污染。
        valid_boxes = [o["aabb"] for o in room_objs if not _bbox_is_zero(o["aabb"])]
        if valid_boxes:
            all_mins = np.array([b["min"] for b in valid_boxes])
            all_maxs = np.array([b["max"] for b in valid_boxes])
            room_min = all_mins.min(axis=0).tolist()
            room_max = all_maxs.max(axis=0).tolist()
        elif rid in region_bbox_map:
            room_min = region_bbox_map[rid]["min"]
            room_max = region_bbox_map[rid]["max"]
        else:
            room_min = [0.0, 0.0, 0.0]
            room_max = [0.0, 0.0, 0.0]

        # 统计房间内的物体类别
        room_cats = {}
        for o in room_objs:
            room_cats[o["category"]] = room_cats.get(o["category"], 0) + 1

        room_center = [
            round((float(room_min[0]) + float(room_max[0])) / 2.0, 4),
            round((float(room_min[1]) + float(room_max[1])) / 2.0, 4),
            round((float(room_min[2]) + float(room_max[2])) / 2.0, 4),
        ]

        rooms_list.append({
            "region_id": rid,
            "object_count": len(room_objs),
            "categories": room_cats,
            "room_center": room_center,
            "bounding_box": {
                "min": [round(x, 4) for x in room_min],
                "max": [round(x, 4) for x in room_max],
            },
        })

    # 5) 类别统计按数量降序，便于快速观察场景主导类别。
    categories_sorted = sorted(category_counter.items(), key=lambda x: -x[1])

    # 6) 组装最终 JSON。
    result = {
        "scene_info": {
            "scene_name": scene_name,
            "scene_id": scene_id,
            "total_objects": len(objects_list),
            "total_categories": len(category_counter),
            "total_rooms": len(rooms_list),
            "navigable_area_m2": nav_area,
            "scene_aabb": scene_aabb_info,
        },
        "categories": [
            {"name": name, "count": count} for name, count in categories_sorted
        ],
        "rooms": rooms_list,
    }

    # 显式关闭 Simulator，避免批量导出时资源占用累积。
    sim.close()

    # 保存输出文件。
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scene_name}_scene_info.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  [OK] {scene_name}: {len(objects_list)} objects, {len(rooms_list)} rooms, {len(category_counter)} categories -> {out_path}")
    return result


def find_scenes(data_dir):
    """
    自动发现 data_dir 下所有看起来像 HM3D 的场景目录。

    当前规则较宽松：
        - 是目录
        - 前两位是数字
        - 名称中含 '-'
    """
    scenes = []
    for entry in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, entry)
        if not os.path.isdir(full):
            continue
        # 匹配 00xxx-xxxx 格式
        if len(entry) > 6 and entry[0:2].isdigit() and "-" in entry:
            scenes.append(entry)
    return scenes


def main():
    """
    命令行入口。

    逻辑：
        - 参数解析与互斥校验（--scene 与 --all 至少其一）
        - 默认路径推导（dataset_config / output_dir）
        - 单场景或全量批处理导出
    """
    parser = argparse.ArgumentParser(description="导出 HM3D 场景的完整语义信息为 JSON")
    parser.add_argument("--scene", type=str, help="场景名, 例如 00824-Dd4bFSTQ8gi")
    parser.add_argument("--all", action="store_true", help="导出 data_dir 下所有场景")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="场景数据根目录 (默认: 脚本目录下 hm3d/minival)")
    parser.add_argument("--dataset-config", type=str, default=None,
                        help="scene_dataset_config.json 路径 (默认: data_dir 下的 hm3d_annotated_basis.scene_dataset_config.json)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: data_dir/scene_info_export)")
    args = parser.parse_args()

    if not args.scene and not args.all:
        parser.error("请指定 --scene 或 --all")

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"数据目录不存在: {data_dir}")
        sys.exit(1)

    dataset_config = args.dataset_config
    if dataset_config is None:
        dataset_config = DEFAULT_DATASET_CONFIG
    if not os.path.isfile(dataset_config):
        print(f"dataset config 不存在: {dataset_config}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(data_dir, "scene_info_export")

    if args.all:
        scenes = find_scenes(data_dir)
        print(f"发现 {len(scenes)} 个场景，开始导出...\n")
        for s in scenes:
            export_scene(s, data_dir, dataset_config, output_dir)
    else:
        export_scene(args.scene, data_dir, dataset_config, output_dir)

    print(f"\n导出完成! 文件保存在: {output_dir}")


if __name__ == "__main__":
    main()
