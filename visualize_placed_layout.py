#!/usr/bin/env python3
from __future__ import annotations

"""
用 Habitat-Sim 可视化已经放置好的 layout JSON。

用途
----
读取 `assign_objects_to_receptacle_instances.py` 或 `place_objects_on_instances.py`
生成的 layout JSON，将场景与已放置物体加载到 Habitat-Sim 中，提供一个轻量查看器
用于检查物体是否落在合理承载面上、是否穿模、是否集中到错误房间。

它不会重新执行分配、采样或物理放置；它只复现 layout JSON 中的
`objects[*].position / rotation / model_id`，因此很适合定位“生成结果本身”
是否有高度偏移、模板缺失、房间错误等问题。

基础示例
--------
python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \
  --scene 00808-y9hTuugGdiq

同一场景多 layout 切换示例
----------------------
如果使用 `batch_generate_layouts.py` 生成了一组 layout，它们通常位于同一个 batch 目录：

  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/

直接打开其中任意一个 layout 后，按 `[` / `]` 就可以在该目录内切换上一个/下一个
layout JSON，脚本会卸载旧物体、加载新 layout、重置当前选中物体与相机视角：
python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq

默认只扫描当前 layout 所在目录。如果想跨多个 batch 目录比较，使用：
python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --layout-scan-dir results/layouts/00808-y9hTuugGdiq \
  --recursive-layout-scan

手动微调高度示例
----------------
当物体看起来在桌面/架子/地面下方或上方时，打开调试模式：

python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \
  --scene 00808-y9hTuugGdiq \
  --debug-offset \
  --offset-step 0.02

默认情况下，所有物体加载后会先应用 `--initial-y-offset 2.5`，也就是
整体上移 2.5m。这个默认值用于快速验证“当前 layout 是否整体偏低”。
如果需要严格复现原始 layout，请显式传入：

python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \
  --scene 00808-y9hTuugGdiq \
  --initial-y-offset 0

推荐流程：
1. 用 `[/]` 在同一场景的不同 layout 之间切换。
2. 用 `,/.` 或 `9/0` 切到异常物体。
3. 用 `F` 聚焦当前物体。
4. 默认只调整当前选中物体；按 `B` 可在 `selected/all` 之间切换调试作用域。
5. 用 `U` 将调试对象上移，用 `O` 将调试对象下移；每次移动 `--offset-step` 米。
6. HUD 中的 `scope` 会显示当前作用域，`offset=(x,y,z)` 会显示选中物体的累计偏移。
7. 调到合适位置后按 `M` 保存。默认输出到同目录：
   `<原文件名>_offset_debug.json`
8. 若希望指定保存路径，传入：
   `--output-layout results/layouts/.../manual_height_fixed.json`

保存后的 JSON：
- `objects[*].position` 会被更新为调试后的最终可视位置。
- `objects[*].debug_base_position` 记录调试前的原始位置。
- `objects[*].debug_visual_offset` 记录相对原始位置的手动偏移。
- 顶层 `visual_debug_adjustment` 记录保存来源和时间。

无窗口截图示例
--------------
python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \
  --scene 00808-y9hTuugGdiq \
  --headless --headless-max-focus 20

按键
----
W/S A/D E/C    相机前后、左右、上下移动
I/K J/L        相机俯仰、左右转向
R              视角重置到布局中心附近
[/]            切换同目录中的上一个/下一个 layout
,/. 或 9/0     切换当前查看对象
F              相机聚焦当前对象
--debug-offset 启用物体高度偏移调试
U/O            调试模式下上/下调整 Y 偏移
B              调试模式下切换 selected/all 调整作用域
M              调试模式下保存调整后的 layout JSON
H              显示/隐藏帮助
P              保存当前窗口截图
ESC/Q          退出

注意
----
- 如果 OpenCV HighGUI 不可用，脚本会自动尝试 pygame 后端。
- 由于本脚本的目标是检查最终 layout，加载物体时会设置为 KINEMATIC，
  避免可视化过程中的物理模拟再次改变位置。
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hm3d_paths import resolve_scene_paths

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import pygame  # type: ignore[import-not-found]
except ImportError:
    pygame = None  # type: ignore[assignment]

try:
    import habitat_sim  # type: ignore[import-not-found]
    import habitat_sim.utils.common as utils  # type: ignore[import-not-found]
    import magnum as mn  # type: ignore[import-not-found]
except ImportError:
    habitat_sim = None  # type: ignore[assignment]
    utils = None  # type: ignore[assignment]
    mn = None  # type: ignore[assignment]


DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CAMERA_HEIGHT = 1.35
CAMERA_MOVE_SPEED = 0.18
ROTATE_SPEED = 2.5
PITCH_LIMIT = 85.0
DEFAULT_INITIAL_Y_OFFSET = 2.5


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_yaw_deg(rotation_value: Any) -> float:
    if isinstance(rotation_value, list) and len(rotation_value) >= 3:
        return _safe_float(rotation_value[1])
    return 0.0


def _yaw_to_magnum_quat(yaw_deg: float) -> mn.Quaternion:
    return mn.Quaternion.rotation(mn.Rad(math.radians(float(yaw_deg))), mn.Vector3(0.0, 1.0, 0.0))


def _camera_rotation(yaw_deg: float, pitch_deg: float) -> Any:
    qy = utils.quat_from_angle_axis(math.radians(float(yaw_deg)), np.array([0.0, 1.0, 0.0]))
    qx = utils.quat_from_angle_axis(math.radians(float(pitch_deg)), np.array([1.0, 0.0, 0.0]))
    return qy * qx


def _camera_vectors(yaw_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    yaw_rad = math.radians(float(yaw_deg))
    forward = np.array([-math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype=np.float32)
    right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)], dtype=np.float32)
    return forward, right


def _look_at_yaw_pitch(camera_pos: Sequence[float], target: Sequence[float]) -> Tuple[float, float]:
    dx = _safe_float(target[0]) - _safe_float(camera_pos[0])
    dy = _safe_float(target[1]) - _safe_float(camera_pos[1])
    dz = _safe_float(target[2]) - _safe_float(camera_pos[2])
    yaw = math.degrees(math.atan2(-dx, -dz))
    horiz = max(math.sqrt(dx * dx + dz * dz), 1e-6)
    pitch = math.degrees(math.atan2(dy, horiz))
    return yaw, max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch))


def _normalize_key(raw_key: int) -> int:
    if raw_key < 0:
        return raw_key
    key = raw_key & 0xFF
    if ord("A") <= key <= ord("Z"):
        return key + 32
    return key


def _normalize_pygame_key(key: int) -> int:
    if pygame is None:
        return -1
    mapping = {
        pygame.K_ESCAPE: 27,
        pygame.K_q: ord("q"),
        pygame.K_h: ord("h"),
        pygame.K_r: ord("r"),
        pygame.K_f: ord("f"),
        pygame.K_p: ord("p"),
        pygame.K_w: ord("w"),
        pygame.K_s: ord("s"),
        pygame.K_a: ord("a"),
        pygame.K_d: ord("d"),
        pygame.K_e: ord("e"),
        pygame.K_c: ord("c"),
        pygame.K_i: ord("i"),
        pygame.K_k: ord("k"),
        pygame.K_j: ord("j"),
        pygame.K_l: ord("l"),
        pygame.K_u: ord("u"),
        pygame.K_o: ord("o"),
        pygame.K_m: ord("m"),
        pygame.K_b: ord("b"),
        pygame.K_LEFTBRACKET: ord("["),
        pygame.K_RIGHTBRACKET: ord("]"),
        pygame.K_COMMA: ord(","),
        pygame.K_PERIOD: ord("."),
        pygame.K_9: ord("9"),
        pygame.K_0: ord("0"),
    }
    return mapping.get(key, -1)


def _layout_objects(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = payload.get("objects", [])
    return objects if isinstance(objects, list) else []


def _object_position(obj: Dict[str, Any]) -> Optional[List[float]]:
    pos = obj.get("position", obj.get("translation"))
    if not isinstance(pos, list) or len(pos) < 3:
        return None
    return [_safe_float(pos[0]), _safe_float(pos[1]), _safe_float(pos[2])]


def _layout_center(objects: Sequence[Dict[str, Any]]) -> np.ndarray:
    points = [_object_position(obj) for obj in objects]
    points = [p for p in points if p is not None]
    if not points:
        return np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    center = arr.mean(axis=0)
    center[1] = max(float(center[1]), 0.0)
    return center


def _infer_scene_from_layout(layout_path: Path, payload: Dict[str, Any]) -> Optional[str]:
    scene_value = payload.get("scene")
    if isinstance(scene_value, str) and scene_value.strip():
        parts = Path(scene_value).parts
        for part in reversed(parts):
            if "-" in part and part[:5].isdigit():
                return part
        stem = Path(scene_value).stem
        if "-" in stem and stem[:5].isdigit():
            return stem

    for part in reversed(layout_path.parts):
        if "-" in part and part[:5].isdigit():
            return part
    return None


def _make_simulator(scene_name: str, data_dir: Path, width: int, height: int) -> habitat_sim.Simulator:
    scene_paths = resolve_scene_paths(scene_name, require_semantic=False, root=data_dir)
    if scene_paths is None:
        raise FileNotFoundError(f"找不到场景或 stage glb: {scene_name}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
    sim_cfg.scene_id = str(scene_paths.stage_glb)
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [int(height), int(width)]
    sensor.hfov = 90
    sensor.position = [0.0, CAMERA_HEIGHT, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.height = CAMERA_HEIGHT
    agent_cfg.radius = 0.18

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _load_templates(sim: habitat_sim.Simulator, objects_dir: Path) -> None:
    template_mgr = sim.get_object_template_manager()
    abs_dir = str(objects_dir.expanduser().resolve())
    if not os.path.isdir(abs_dir):
        print(f"[Warning] objects dir not found: {abs_dir}")
        return
    try:
        if hasattr(template_mgr, "load_configs"):
            template_mgr.load_configs(abs_dir)
        elif hasattr(template_mgr, "add_template_search_path"):
            template_mgr.add_template_search_path(abs_dir)
        elif hasattr(template_mgr, "load_object_configs"):
            template_mgr.load_object_configs(abs_dir)
    except Exception as exc:
        print(f"[Warning] failed to load object templates: {exc}")


def _resolve_template_handle(template_mgr: Any, model_id: str) -> Optional[str]:
    candidates_to_try = [model_id]
    if not model_id.endswith(".object_config.json"):
        candidates_to_try.append(f"{model_id}.object_config.json")
    if not model_id.endswith("_4k"):
        candidates_to_try.extend([f"{model_id}_4k", f"{model_id}_4k.object_config.json"])

    for key in candidates_to_try:
        try:
            handles = template_mgr.get_template_handles(key)
            if handles:
                return handles[0]
        except Exception:
            continue

    try:
        needle = model_id.lower().replace(".object_config.json", "")
        for handle in template_mgr.get_template_handles():
            name = os.path.basename(str(handle)).lower().replace(".object_config.json", "")
            if name == needle or name == f"{needle}_4k":
                return handle
    except Exception:
        pass
    return None


def _load_layout_objects(
    sim: habitat_sim.Simulator,
    objects: Sequence[Dict[str, Any]],
    initial_y_offset: float = DEFAULT_INITIAL_Y_OFFSET,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    将 layout JSON 中的 objects 加载为 Habitat-Sim 刚体。

    这里故意不做任何“重新放置”逻辑，只按 JSON 的 position/rotation 复现结果：
    - `position/base_position`：layout 原始坐标，作为调试偏移的零点。
    - `debug_offset`：运行时手动调试的累计偏移，初始 Y 值来自
      `--initial-y-offset`，当前默认是 +2.5m。
    - `index`：原始 objects 数组下标，保存调整结果时用它回写对应条目。

    物体统一设置为 KINEMATIC，是为了让可视化器成为稳定的检查工具；
    否则 DYNAMIC 物体可能因为模板碰撞属性/重力再次移动，掩盖 layout 本身的问题。
    """
    rom = sim.get_rigid_object_manager()
    template_mgr = sim.get_object_template_manager()
    loaded: List[Dict[str, Any]] = []
    skipped = 0

    for idx, cfg in enumerate(objects):
        model_id = str(cfg.get("model_id", cfg.get("template_name", ""))).strip()
        pos = _object_position(cfg)
        if not model_id or pos is None:
            skipped += 1
            continue

        handle = _resolve_template_handle(template_mgr, model_id)
        if handle is None:
            print(f"[skip] object#{idx} model={model_id}: template not found")
            skipped += 1
            continue

        try:
            obj = rom.add_object_by_template_handle(handle)
            if obj is None:
                raise RuntimeError("add_object_by_template_handle returned None")
            yaw = _extract_yaw_deg(cfg.get("rotation"))
            debug_offset = [0.0, float(initial_y_offset), 0.0]
            obj.translation = np.asarray(pos, dtype=np.float32) + np.asarray(debug_offset, dtype=np.float32)
            obj.rotation = _yaw_to_magnum_quat(yaw)
            if hasattr(obj, "motion_type") and hasattr(habitat_sim, "physics"):
                obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            loaded.append(
                {
                    "object": obj,
                    "object_id": getattr(obj, "object_id", None),
                    "handle": getattr(obj, "handle", None),
                    "layout": cfg,
                    "index": idx,
                    "model_id": model_id,
                    "name": str(cfg.get("name", model_id)),
                    "position": pos,
                    "base_position": list(pos),
                    "debug_offset": debug_offset,
                    "target_instance_id": cfg.get("target_instance_id", "?"),
                    "sampled_region_id": cfg.get("sampled_region_id", "?"),
                }
            )
        except Exception as exc:
            print(f"[skip] object#{idx} model={model_id}: {exc}")
            skipped += 1
    return loaded, skipped


def _layout_json_valid(path: Path) -> bool:
    """轻量判断一个 JSON 是否像本脚本可加载的 layout。"""
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("objects"), list)


def _list_layout_files(current_layout_path: Path, scan_dir: Optional[Path], recursive: bool = False) -> List[Path]:
    """
    列出同一场景下可切换的 layout 文件。

    默认只扫描当前 layout 所在目录；这正好适配 batch_generate_layouts.py 的输出：
    `results/layouts/<scene>/batch_xxx/layout_000_seed_42.json` 等文件都在同一目录下。
    如果用户想跨多个 batch 目录比较，可以传 `--layout-scan-dir results/layouts/<scene>`
    并开启 `--recursive-layout-scan`。
    """
    root = scan_dir if scan_dir is not None else current_layout_path.parent
    candidates: List[Path] = []
    if root.is_dir():
        pattern = "**/*.json" if recursive else "*.json"
        candidates.extend(path for path in root.glob(pattern) if _layout_json_valid(path))
    elif _layout_json_valid(root):
        candidates.append(root)

    if _layout_json_valid(current_layout_path):
        candidates.append(current_layout_path)

    unique: Dict[str, Path] = {}
    for path in candidates:
        key = os.path.normcase(str(path.resolve()))
        unique[key] = path
    return sorted(unique.values(), key=lambda p: str(p.parent).lower() + "/" + p.name.lower())


def _find_layout_index(layout_files: Sequence[Path], target_path: Path) -> int:
    target = os.path.normcase(str(target_path.resolve()))
    for idx, path in enumerate(layout_files):
        if os.path.normcase(str(path.resolve())) == target:
            return idx
    return 0


def _remove_loaded_item(rom: Any, item: Dict[str, Any]) -> None:
    """参考 test_layout.py，尽量通过 handle/id 移除已加载刚体。"""
    handle = item.get("handle")
    if handle:
        try:
            rom.remove_object_by_handle(handle)
            return
        except Exception:
            pass
    object_id = item.get("object_id")
    if object_id is not None:
        try:
            rom.remove_object_by_id(object_id)
        except Exception:
            pass


def _clear_loaded_objects(sim: habitat_sim.Simulator, loaded_items: List[Dict[str, Any]]) -> None:
    rom = sim.get_rigid_object_manager()
    for item in list(loaded_items):
        _remove_loaded_item(rom, item)
    loaded_items.clear()


def _load_layout_path_into_viewer(
    sim: habitat_sim.Simulator,
    layout_path: Path,
    loaded_items: List[Dict[str, Any]],
    initial_y_offset: float,
) -> Tuple[Dict[str, Any], int, int]:
    """卸载当前物体，并把指定 layout 文件重新加载到同一个 simulator 中。"""
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    objects = _layout_objects(payload)
    _clear_loaded_objects(sim, loaded_items)
    new_items, skipped = _load_layout_objects(sim, objects, initial_y_offset=initial_y_offset)
    loaded_items.extend(new_items)
    return payload, len(objects), skipped


def _switch_layout(
    sim: habitat_sim.Simulator,
    state: Dict[str, Any],
    loaded_items: List[Dict[str, Any]],
    direction: int,
) -> None:
    """在同一场景中切换到上一个/下一个 layout 文件，并重置选择与相机。"""
    current_path = Path(str(state.get("layout_path", ".")))
    scan_dir_value = state.get("layout_scan_dir")
    scan_dir = Path(str(scan_dir_value)) if scan_dir_value else current_path.parent
    layout_files = _list_layout_files(current_path, scan_dir, bool(state.get("recursive_layout_scan", False)))
    if not layout_files:
        print(f"[Warning] No switchable layout JSON found in: {scan_dir}")
        return

    current_idx = _find_layout_index(layout_files, current_path)
    next_idx = (current_idx + int(direction)) % len(layout_files)
    next_path = layout_files[next_idx]
    if os.path.normcase(str(next_path.resolve())) == os.path.normcase(str(current_path.resolve())) and len(layout_files) == 1:
        print(f"[Info] Only one layout available: {next_path.name}")
        return

    payload, object_count, skipped = _load_layout_path_into_viewer(
        sim=sim,
        layout_path=next_path,
        loaded_items=loaded_items,
        initial_y_offset=float(state.get("initial_y_offset", DEFAULT_INITIAL_Y_OFFSET)),
    )
    state["layout_payload"] = payload
    state["layout_path"] = str(next_path)
    state["layout_index"] = next_idx
    state["layout_count"] = len(layout_files)
    state["object_count"] = object_count
    state["skipped"] = skipped
    state["selected_idx"] = 0
    state["offset_scope"] = "selected"
    camera_pos, yaw, pitch = _reset_camera(loaded_items)
    state["camera_pos"] = camera_pos
    state["yaw"] = yaw
    state["pitch"] = pitch
    print(f"[OK] Switched layout {next_idx + 1}/{len(layout_files)} -> {next_path.name} loaded={len(loaded_items)}/{object_count} skipped={skipped}")


def _set_camera(sim: habitat_sim.Simulator, camera_pos: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = camera_pos.astype(np.float32)
    state.rotation = _camera_rotation(yaw, pitch)
    agent.set_state(state, reset_sensors=False)
    obs = sim.get_sensor_observations()
    return obs["color"][:, :, :3]


def _draw_text(frame: np.ndarray, lines: Sequence[str], x: int, y: int, color: Tuple[int, int, int]) -> np.ndarray:
    for i, line in enumerate(lines):
        yy = y + i * 22
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def _selected_label(items: Sequence[Dict[str, Any]], selected_idx: int) -> str:
    """构造 HUD 中当前选中物体的简短状态行。"""
    if not items:
        return "none"
    item = items[selected_idx % len(items)]
    obj = item["object"]
    pos = obj.translation
    offset = item.get("debug_offset", [0.0, 0.0, 0.0])
    return (
        f"{selected_idx + 1}/{len(items)} {item['model_id']} "
        f"pos=({float(pos[0]):.2f},{float(pos[1]):.2f},{float(pos[2]):.2f}) "
        f"offset=({float(offset[0]):+.2f},{float(offset[1]):+.2f},{float(offset[2]):+.2f}) "
        f"room={item.get('sampled_region_id')} target={item.get('target_instance_id')}"
    )


def _reset_camera(objects: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, float, float]:
    """把相机放到能看到整体布局的位置。"""
    center = _layout_center([item.get("layout", {}) for item in objects])
    if objects:
        positions = np.asarray([item["object"].translation for item in objects], dtype=np.float32)
        span = positions.max(axis=0) - positions.min(axis=0)
        distance = max(float(np.linalg.norm(span[[0, 2]])) * 0.55, 3.0)
    else:
        distance = 4.0
    camera_pos = np.array([center[0], center[1] + 1.4, center[2] + distance], dtype=np.float32)
    yaw, pitch = _look_at_yaw_pitch(camera_pos, center)
    return camera_pos, yaw, pitch


def _focus_object(item: Dict[str, Any]) -> Tuple[np.ndarray, float, float]:
    """把相机移动到当前物体前方，用于逐个检查高度和穿模。"""
    target = np.asarray(item["object"].translation, dtype=np.float32)
    camera_pos = target + np.array([0.0, 1.0, 2.4], dtype=np.float32)
    yaw, pitch = _look_at_yaw_pitch(camera_pos, target)
    return camera_pos, yaw, pitch


def _build_hud(
    scene_name: str,
    layout_path: Path,
    loaded_items: Sequence[Dict[str, Any]],
    object_count: int,
    skipped: int,
    selected_idx: int,
    camera_pos: np.ndarray,
    yaw: float,
    pitch: float,
    debug_offset: bool = False,
    offset_step: float = 0.02,
    offset_scope: str = "selected",
) -> List[str]:
    """生成左上角状态文本；调试模式下额外显示步长和保存提示。"""
    lines = [
        f"Scene: {scene_name}",
        f"Layout: {layout_path.name}",
        f"Objects loaded: {len(loaded_items)}/{object_count} skipped={skipped}",
        f"Selected: {_selected_label(loaded_items, selected_idx)}",
        f"Camera: ({camera_pos[0]:.2f},{camera_pos[1]:.2f},{camera_pos[2]:.2f}) yaw={yaw:.1f} pitch={pitch:.1f}",
    ]
    if debug_offset:
        lines.append(f"Offset debug: ON  scope={offset_scope}  step={offset_step:.3f}m  B=scope  U/O=y +/-  M=save")
    return lines


def _render_frame(
    sim: habitat_sim.Simulator,
    scene_name: str,
    layout_path: Path,
    loaded_items: Sequence[Dict[str, Any]],
    object_count: int,
    skipped: int,
    selected_idx: int,
    camera_pos: np.ndarray,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
    show_help: bool,
    help_lines: Sequence[str],
    debug_offset: bool = False,
    offset_step: float = 0.02,
    offset_scope: str = "selected",
) -> np.ndarray:
    try:
        rgb = _set_camera(sim, camera_pos, yaw, pitch)
        frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        _draw_text(frame, [f"Render error: {exc}"], 20, 60, (80, 80, 255))

    hud = _build_hud(
        scene_name=scene_name,
        layout_path=layout_path,
        loaded_items=loaded_items,
        object_count=object_count,
        skipped=skipped,
        selected_idx=selected_idx,
        camera_pos=camera_pos,
        yaw=yaw,
        pitch=pitch,
        debug_offset=debug_offset,
        offset_step=offset_step,
        offset_scope=offset_scope,
    )
    _draw_text(frame, hud, 10, 24, (80, 255, 255))
    if show_help:
        y0 = int(height) - len(help_lines) * 22 - 16
        _draw_text(frame, help_lines, 10, max(24, y0), (80, 255, 80))
    return frame


def _adjust_selected_object_y(loaded_items: Sequence[Dict[str, Any]], selected_idx: int, delta_y: float) -> Optional[Dict[str, Any]]:
    """
    在调试模式下沿 Y 轴调整当前物体。

    `debug_offset` 始终是“相对 base_position 的累计偏移”，而不是相对上一次
    obj.translation 的增量保存。这样反复按 U/O 后，保存出来的
    `debug_base_position + debug_visual_offset == position`，后续排查更直观。
    """
    if not loaded_items:
        return None
    item = loaded_items[selected_idx % len(loaded_items)]
    offset = list(item.get("debug_offset", [0.0, 0.0, 0.0]))
    while len(offset) < 3:
        offset.append(0.0)
    offset[1] = float(offset[1]) + float(delta_y)
    base = np.asarray(item.get("base_position", item.get("position", [0.0, 0.0, 0.0])), dtype=np.float32)
    obj = item["object"]
    obj.translation = base + np.asarray(offset[:3], dtype=np.float32)
    item["debug_offset"] = offset[:3]
    return item


def _adjust_all_objects_y(loaded_items: Sequence[Dict[str, Any]], delta_y: float) -> int:
    """
    批量沿 Y 轴调整所有已加载物体。

    这个功能用于判断“整个 layout 是否整体偏低/偏高”。例如所有物体都埋在
    承载面下方时，切到 all 作用域后按 U，可以立即在同一帧看到整体抬升效果。
    """
    moved = 0
    for idx in range(len(loaded_items)):
        if _adjust_selected_object_y(loaded_items, idx, delta_y) is not None:
            moved += 1
    return moved


def _save_adjusted_layout(
    payload: Dict[str, Any],
    loaded_items: Sequence[Dict[str, Any]],
    input_layout_path: Path,
    output_layout_path: Optional[Path],
) -> Path:
    """
    保存手动微调后的 layout。

    保存策略：
    - 不破坏原始 payload 的内存对象，先深拷贝一份再写文件。
    - 只回写已成功加载的物体；模板缺失/无法加载的物体保持原 JSON。
    - `position` 直接写成当前可视化位置，便于后续脚本直接消费。
    - 额外保留 `debug_base_position/debug_visual_offset`，方便知道改了多少。
    """
    out_path = output_layout_path
    if out_path is None:
        out_path = input_layout_path.with_name(f"{input_layout_path.stem}_offset_debug{input_layout_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    adjusted_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    objects = adjusted_payload.get("objects", [])
    if not isinstance(objects, list):
        objects = []
        adjusted_payload["objects"] = objects

    for item in loaded_items:
        idx = int(item.get("index", -1))
        if idx < 0 or idx >= len(objects) or not isinstance(objects[idx], dict):
            continue
        obj = item["object"]
        pos = [round(float(obj.translation[0]), 4), round(float(obj.translation[1]), 4), round(float(obj.translation[2]), 4)]
        offset = [round(float(v), 4) for v in item.get("debug_offset", [0.0, 0.0, 0.0])[:3]]
        objects[idx]["position"] = pos
        objects[idx]["debug_base_position"] = [round(float(v), 4) for v in item.get("base_position", pos)[:3]]
        objects[idx]["debug_visual_offset"] = offset

    adjusted_payload["visual_debug_adjustment"] = {
        "tool": "visualize_placed_layout.py",
        "saved_at_unix": int(time.time()),
        "source_layout": str(input_layout_path),
        "note": "position fields include manual visual debug offsets",
    }
    out_path.write_text(json.dumps(adjusted_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _apply_viewer_key(
    key: int,
    state: Dict[str, Any],
    sim: habitat_sim.Simulator,
    loaded_items: List[Dict[str, Any]],
    screenshot_dir: Path,
    scene_name: str,
) -> bool:
    """
    处理 cv2/pygame 统一后的按键。

    调试模式只绑定三个键：
    - B：在 selected/all 作用域之间切换
    - U：当前作用域上移 `offset_step`
    - O：当前作用域下移 `offset_step`
    - M：保存当前所有已加载物体的位置
    """
    if key < 0:
        return False
    if key in (27, ord("q")):
        state["quit"] = True
        return True
    if key == ord("h"):
        state["show_help"] = not bool(state.get("show_help", True))
    if key == ord("r"):
        camera_pos, yaw, pitch = _reset_camera(loaded_items)
        state["camera_pos"] = camera_pos
        state["yaw"] = yaw
        state["pitch"] = pitch
    if key == ord("["):
        _switch_layout(sim, state, loaded_items, -1)
        return False
    if key == ord("]"):
        _switch_layout(sim, state, loaded_items, 1)
        return False
    if key in (ord(","), ord("<"), ord("9")) and loaded_items:
        state["selected_idx"] = (int(state.get("selected_idx", 0)) - 1) % len(loaded_items)
    if key in (ord("."), ord(">"), ord("0")) and loaded_items:
        state["selected_idx"] = (int(state.get("selected_idx", 0)) + 1) % len(loaded_items)
    if key == ord("f") and loaded_items:
        camera_pos, yaw, pitch = _focus_object(loaded_items[int(state.get("selected_idx", 0))])
        state["camera_pos"] = camera_pos
        state["yaw"] = yaw
        state["pitch"] = pitch
    if key == ord("p") and state.get("last_frame") is not None:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        out_path = screenshot_dir / f"{scene_name}_layout_view_{int(time.time())}.png"
        cv2.imwrite(str(out_path), state["last_frame"])
        print(f"[OK] Screenshot saved: {out_path}")
    if bool(state.get("debug_offset", False)) and loaded_items:
        selected_idx = int(state.get("selected_idx", 0))
        step = float(state.get("offset_step", 0.02))
        scope = str(state.get("offset_scope", "selected"))
        if key == ord("b"):
            state["offset_scope"] = "all" if scope != "all" else "selected"
            print(f"[Debug] offset scope -> {state['offset_scope']}")
        if key == ord("u"):
            if scope == "all":
                moved = _adjust_all_objects_y(loaded_items, step)
                print(f"[Debug] all objects y_offset += {step:+.4f} ({moved} moved)")
            else:
                item = _adjust_selected_object_y(loaded_items, selected_idx, step)
                if item is not None:
                    print(f"[Debug] {item['model_id']} y_offset={item['debug_offset'][1]:+.4f}")
        if key == ord("o"):
            if scope == "all":
                moved = _adjust_all_objects_y(loaded_items, -step)
                print(f"[Debug] all objects y_offset += {-step:+.4f} ({moved} moved)")
            else:
                item = _adjust_selected_object_y(loaded_items, selected_idx, -step)
                if item is not None:
                    print(f"[Debug] {item['model_id']} y_offset={item['debug_offset'][1]:+.4f}")
        if key == ord("m"):
            out_path = _save_adjusted_layout(
                payload=state.get("layout_payload", {}),
                loaded_items=loaded_items,
                input_layout_path=Path(state.get("layout_path", ".")),
                output_layout_path=state.get("output_layout_path"),
            )
            print(f"[OK] Adjusted layout saved: {out_path}")

    yaw = float(state.get("yaw", 0.0))
    pitch = float(state.get("pitch", 0.0))
    camera_pos = np.asarray(state.get("camera_pos", np.zeros(3)), dtype=np.float32)
    forward, right = _camera_vectors(yaw)

    if key == ord("j"):
        yaw += ROTATE_SPEED
    if key == ord("l"):
        yaw -= ROTATE_SPEED
    if key == ord("i"):
        pitch = min(pitch + ROTATE_SPEED, PITCH_LIMIT)
    if key == ord("k"):
        pitch = max(pitch - ROTATE_SPEED, -PITCH_LIMIT)
    if key == ord("w"):
        camera_pos += forward * CAMERA_MOVE_SPEED
    if key == ord("s"):
        camera_pos -= forward * CAMERA_MOVE_SPEED
    if key == ord("a"):
        camera_pos -= right * CAMERA_MOVE_SPEED
    if key == ord("d"):
        camera_pos += right * CAMERA_MOVE_SPEED
    if key == ord("e"):
        camera_pos[1] += CAMERA_MOVE_SPEED
    if key == ord("c"):
        camera_pos[1] -= CAMERA_MOVE_SPEED

    state["camera_pos"] = camera_pos
    state["yaw"] = yaw
    state["pitch"] = pitch
    return False


def _render_current_state(
    sim: habitat_sim.Simulator,
    scene_name: str,
    layout_path: Path,
    loaded_items: Sequence[Dict[str, Any]],
    object_count: int,
    skipped: int,
    state: Dict[str, Any],
    width: int,
    height: int,
    help_lines: Sequence[str],
) -> np.ndarray:
    active_layout_path = Path(str(state.get("layout_path", layout_path)))
    active_object_count = int(state.get("object_count", object_count))
    active_skipped = int(state.get("skipped", skipped))
    return _render_frame(
        sim=sim,
        scene_name=scene_name,
        layout_path=active_layout_path,
        loaded_items=loaded_items,
        object_count=active_object_count,
        skipped=active_skipped,
        selected_idx=int(state.get("selected_idx", 0)),
        camera_pos=np.asarray(state.get("camera_pos", np.zeros(3)), dtype=np.float32),
        yaw=float(state.get("yaw", 0.0)),
        pitch=float(state.get("pitch", 0.0)),
        width=width,
        height=height,
        show_help=bool(state.get("show_help", True)),
        help_lines=help_lines,
        debug_offset=bool(state.get("debug_offset", False)),
        offset_step=float(state.get("offset_step", 0.02)),
        offset_scope=str(state.get("offset_scope", "selected")),
    )


def _opencv_gui_available() -> bool:
    if cv2 is None:
        return False
    try:
        build_info = cv2.getBuildInformation()
    except Exception:
        return True
    for raw_line in build_info.splitlines():
        text = raw_line.strip()
        if text.startswith("GUI:"):
            return not text.endswith("NONE")
    return True


def _run_pygame_viewer(
    sim: habitat_sim.Simulator,
    scene_name: str,
    layout_path: Path,
    loaded_items: List[Dict[str, Any]],
    object_count: int,
    skipped: int,
    state: Dict[str, Any],
    width: int,
    height: int,
    screenshot_dir: Path,
    help_lines: Sequence[str],
) -> int:
    if pygame is None:
        print(
            "[Error] cv2 HighGUI is unavailable and pygame is not installed. "
            "Install pygame or use a GUI-enabled OpenCV build.",
            file=sys.stderr,
        )
        return 1

    pygame.init()
    screen = pygame.display.set_mode((int(width), int(height)))
    pygame.display.set_caption("Placed Layout Viewer  [H]Help  [P]Screenshot  [ESC/Q]Quit")
    clock = pygame.time.Clock()
    print("[Info] Using pygame window backend because OpenCV HighGUI is unavailable.")

    try:
        while not bool(state.get("quit", False)):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    state["quit"] = True
                elif event.type == pygame.KEYDOWN:
                    _apply_viewer_key(
                        _normalize_pygame_key(event.key),
                        state,
                        sim,
                        loaded_items,
                        screenshot_dir,
                        scene_name,
                    )

            frame_bgr = _render_current_state(
                sim=sim,
                scene_name=scene_name,
                layout_path=layout_path,
                loaded_items=loaded_items,
                object_count=object_count,
                skipped=skipped,
                state=state,
                width=width,
                height=height,
                help_lines=help_lines,
            )
            state["last_frame"] = frame_bgr
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()
    return 0


def _run_cv2_viewer(
    sim: habitat_sim.Simulator,
    scene_name: str,
    layout_path: Path,
    loaded_items: List[Dict[str, Any]],
    object_count: int,
    skipped: int,
    state: Dict[str, Any],
    width: int,
    height: int,
    screenshot_dir: Path,
    help_lines: Sequence[str],
) -> int:
    window = "Placed Layout Viewer  [H]Help  [P]Screenshot  [ESC/Q]Quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, int(width), int(height))

    try:
        while not bool(state.get("quit", False)):
            raw_key = cv2.waitKeyEx(30)
            key = _normalize_key(raw_key)
            _apply_viewer_key(key, state, sim, loaded_items, screenshot_dir, scene_name)
            frame = _render_current_state(
                sim=sim,
                scene_name=scene_name,
                layout_path=layout_path,
                loaded_items=loaded_items,
                object_count=object_count,
                skipped=skipped,
                state=state,
                width=width,
                height=height,
                help_lines=help_lines,
            )
            state["last_frame"] = frame
            cv2.imshow(window, frame)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
    return 0


def _save_headless_snapshots(
    sim: habitat_sim.Simulator,
    scene_name: str,
    layout_path: Path,
    loaded_items: Sequence[Dict[str, Any]],
    object_count: int,
    skipped: int,
    screenshot_dir: Path,
    width: int,
    height: int,
    max_focus: int,
    help_lines: Sequence[str],
) -> List[Path]:
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    saved: List[Path] = []

    camera_pos, yaw, pitch = _reset_camera(loaded_items)
    overview = _render_frame(
        sim=sim,
        scene_name=scene_name,
        layout_path=layout_path,
        loaded_items=loaded_items,
        object_count=object_count,
        skipped=skipped,
        selected_idx=0,
        camera_pos=camera_pos,
        yaw=yaw,
        pitch=pitch,
        width=width,
        height=height,
        show_help=False,
        help_lines=help_lines,
    )
    overview_path = screenshot_dir / f"{scene_name}_overview_{timestamp}.png"
    cv2.imwrite(str(overview_path), overview)
    saved.append(overview_path)

    for idx, item in enumerate(loaded_items[: max(0, int(max_focus))]):
        camera_pos, yaw, pitch = _focus_object(item)
        frame = _render_frame(
            sim=sim,
            scene_name=scene_name,
            layout_path=layout_path,
            loaded_items=loaded_items,
            object_count=object_count,
            skipped=skipped,
            selected_idx=idx,
            camera_pos=camera_pos,
            yaw=yaw,
            pitch=pitch,
            width=width,
            height=height,
            show_help=False,
            help_lines=help_lines,
        )
        safe_model = str(item.get("model_id", f"object_{idx}")).replace("/", "_")
        out_path = screenshot_dir / f"{scene_name}_focus_{idx + 1:02d}_{safe_model}_{timestamp}.png"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    return saved


def _opencv_gui_diagnostics(exc: Exception) -> List[str]:
    lines = [
        "[Error] OpenCV HighGUI window creation failed.",
        f"[Error] cv2 exception: {exc}",
    ]
    if cv2 is not None:
        lines.append(f"[Diag] cv2 version: {getattr(cv2, '__version__', 'unknown')}")
        lines.append(f"[Diag] cv2 module: {getattr(cv2, '__file__', 'unknown')}")
        try:
            build_info = cv2.getBuildInformation()
        except Exception as build_exc:
            build_info = ""
            lines.append(f"[Diag] cv2.getBuildInformation() failed: {build_exc}")
        if build_info:
            gui_lines = []
            capture = False
            for raw_line in build_info.splitlines():
                text = raw_line.strip()
                if text.startswith("GUI:"):
                    capture = True
                    gui_lines.append(text)
                    continue
                if capture:
                    if not text:
                        break
                    if (
                        text.startswith("QT:")
                        or text.startswith("GTK")
                        or text.startswith("Win32 UI:")
                        or text.startswith("Cocoa:")
                        or text.startswith("VTK")
                    ):
                        gui_lines.append(text)
            if gui_lines:
                lines.append("[Diag] OpenCV GUI build info:")
                lines.extend(f"  {line}" for line in gui_lines)
    lines.append(f"[Diag] DISPLAY={os.environ.get('DISPLAY', '')!r}")
    lines.append(f"[Diag] WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY', '')!r}")
    lines.extend(
        [
            "[Hint] This is not a Habitat scene-loading failure; the scene and objects may have loaded before HighGUI failed.",
            "[Hint] If GUI build info says GUI: NONE, your environment likely uses opencv-python-headless or a no-GUI OpenCV build.",
            "[Hint] Fix options: install a GUI-enabled OpenCV build in the habitat env, or run with --headless explicitly to save snapshots.",
            "[Hint] For conda, try: conda install -c conda-forge opencv pyqt",
            "[Hint] For pip, try removing opencv-python-headless and installing opencv-python, then ensure a display/X forwarding is available.",
        ]
    )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a placed Habitat layout JSON.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Basic viewer:\n"
            "    python visualize_placed_layout.py \\\n"
            "      results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \\\n"
            "      --scene 00808-y9hTuugGdiq\n\n"
            "  Manual height debugging:\n"
            "    python visualize_placed_layout.py \\\n"
            "      results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \\\n"
            "      --scene 00808-y9hTuugGdiq --debug-offset --offset-step 0.02\n\n"
            "  Strictly reproduce original layout without the default +2.5m lift:\n"
            "    python visualize_placed_layout.py \\\n"
            "      results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \\\n"
            "      --scene 00808-y9hTuugGdiq --initial-y-offset 0\n\n"
            "  Manual height debugging with explicit output:\n"
            "    python visualize_placed_layout.py \\\n"
            "      results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \\\n"
            "      --scene 00808-y9hTuugGdiq --debug-offset \\\n"
            "      --output-layout results/layouts/00808-y9hTuugGdiq/height_fixed.json\n\n"
            "  Switch layouts in the same batch directory with [ and ]:\n"
            "    python visualize_placed_layout.py \\\n"
            "      results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json \\\n"
            "      --scene 00808-y9hTuugGdiq\n\n"
            "  Switch layouts recursively across multiple batch directories:\n"
            "    python visualize_placed_layout.py \\\n"
            "      results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json \\\n"
            "      --scene 00808-y9hTuugGdiq \\\n"
            "      --layout-scan-dir results/layouts/00808-y9hTuugGdiq --recursive-layout-scan\n\n"
            "Debug keys:\n"
            "  [/]        : switch previous / next layout JSON\n"
            "  ,/. or 9/0 : select previous / next object\n"
            "  F          : focus selected object\n"
            "  B          : toggle offset scope between selected and all objects\n"
            "  U / O      : move current scope up / down by --offset-step meters\n"
            "  M          : save adjusted layout JSON\n"
        ),
    )
    parser.add_argument("layout", help="已放置 layout JSON 路径")
    parser.add_argument("--scene", default=None, help="场景名；不填时尝试从 layout 或路径推断")
    parser.add_argument("--data-dir", default="hm3d", help="HM3D 数据根目录")
    parser.add_argument("--objects-dir", default="./objects", help="物体模板目录")
    parser.add_argument(
        "--layout-scan-dir",
        default=None,
        help="同场景 layout 切换目录；默认扫描当前 layout 所在目录",
    )
    parser.add_argument(
        "--recursive-layout-scan",
        action="store_true",
        help="递归扫描 --layout-scan-dir 下的 JSON layout，便于跨 batch 目录比较",
    )
    parser.add_argument("--width", type=int, default=DISPLAY_WIDTH, help="窗口宽度")
    parser.add_argument("--height", type=int, default=DISPLAY_HEIGHT, help="窗口高度")
    parser.add_argument("--screenshot-dir", default="./results/visual_checks", help="截图输出目录")
    parser.add_argument(
        "--window-backend",
        choices=["auto", "cv2", "pygame"],
        default="auto",
        help="窗口后端；auto 会优先使用 cv2，cv2 无 GUI 时改用 pygame",
    )
    parser.add_argument("--headless", action="store_true", help="不打开窗口，直接保存总览和物体聚焦截图")
    parser.add_argument("--headless-max-focus", type=int, default=12, help="headless 模式最多保存多少张物体聚焦图")
    parser.add_argument(
        "--debug-offset",
        action="store_true",
        help=(
            "启用可视化高度调试。\n"
            "按 B 切换 selected/all 作用域，按 U/O 上下移动当前作用域，按 M 保存调整后的 layout。"
        ),
    )
    parser.add_argument(
        "--offset-step",
        type=float,
        default=0.02,
        help="debug-offset 每次调整的高度步长，单位米；0.02 表示每次 2cm",
    )
    parser.add_argument(
        "--initial-y-offset",
        type=float,
        default=DEFAULT_INITIAL_Y_OFFSET,
        help="所有物体加载时默认应用的 Y 偏移，单位米；默认 2.5，设为 0 可严格复现原始 layout",
    )
    parser.add_argument(
        "--output-layout",
        default=None,
        help="debug-offset 保存路径；不填则写到原 layout 同目录的 *_offset_debug.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if cv2 is None:
        print("Error: cv2 not found. 请在带 opencv-python 的 Habitat 环境中运行此脚本。", file=sys.stderr)
        return 1
    if habitat_sim is None or utils is None or mn is None:
        print("Error: habitat_sim not found. 请在 Habitat 环境中运行此脚本。", file=sys.stderr)
        return 1

    layout_path = Path(args.layout)
    if not layout_path.is_file():
        print(f"[Error] layout not found: {layout_path}", file=sys.stderr)
        return 1

    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    scene_name = args.scene or _infer_scene_from_layout(layout_path, payload)
    if not scene_name:
        print("[Error] cannot infer scene name; please pass --scene", file=sys.stderr)
        return 1

    objects = _layout_objects(payload)
    sim = _make_simulator(scene_name, Path(args.data_dir), int(args.width), int(args.height))
    _load_templates(sim, Path(args.objects_dir))
    loaded_items, skipped = _load_layout_objects(sim, objects, initial_y_offset=float(args.initial_y_offset))
    layout_scan_dir = Path(args.layout_scan_dir) if args.layout_scan_dir else layout_path.parent
    layout_files = _list_layout_files(layout_path, layout_scan_dir, bool(args.recursive_layout_scan))
    layout_index = _find_layout_index(layout_files, layout_path) if layout_files else 0

    print(f"[OK] Loaded scene: {scene_name}")
    print(f"[OK] Loaded objects: {len(loaded_items)}/{len(objects)} skipped={skipped}")
    print(
        f"[Info] Layout switch catalog: {len(layout_files)} file(s) "
        f"from {layout_scan_dir} recursive={bool(args.recursive_layout_scan)}"
    )
    print(f"[Info] Initial visual y_offset applied to all loaded objects: {float(args.initial_y_offset):+.4f}")
    stats = payload.get("auto_placement_stats", {})
    if isinstance(stats, dict):
        print(
            "[Info] Layout stats: placed={}/{} failed={}".format(
                stats.get("placed_count", len(objects)),
                stats.get("total_objects", len(objects)),
                stats.get("failed_count", 0),
            )
        )
        if stats.get("failed_objects"):
            print(f"[Info] Failed objects recorded in layout: {stats.get('failed_objects')}")

    camera_pos, yaw, pitch = _reset_camera(loaded_items)
    state: Dict[str, Any] = {
        "camera_pos": camera_pos,
        "yaw": yaw,
        "pitch": pitch,
        "selected_idx": 0,
        "show_help": True,
        "last_frame": None,
        "quit": False,
        "debug_offset": bool(args.debug_offset),
        "offset_step": float(args.offset_step),
        "offset_scope": "selected",
        "layout_payload": payload,
        "layout_path": str(layout_path),
        "layout_scan_dir": str(layout_scan_dir),
        "recursive_layout_scan": bool(args.recursive_layout_scan),
        "layout_index": layout_index,
        "layout_count": len(layout_files),
        "object_count": len(objects),
        "skipped": skipped,
        "initial_y_offset": float(args.initial_y_offset),
        "output_layout_path": Path(args.output_layout) if args.output_layout else None,
    }

    help_lines = [
        "W/S A/D E/C: move camera",
        "I/K J/L: pitch / yaw",
        "R: reset view   F: focus selected",
        "[/]: previous / next layout",
        ",/. or 9/0: previous / next object",
        "Debug offset: B toggles selected/all, U/O move y +/- step, M save",
        "H: help   P: screenshot   ESC/Q: quit",
    ]

    try:
        if args.headless:
            saved = _save_headless_snapshots(
                sim=sim,
                scene_name=scene_name,
                layout_path=layout_path,
                loaded_items=loaded_items,
                object_count=len(objects),
                skipped=skipped,
                screenshot_dir=Path(args.screenshot_dir),
                width=int(args.width),
                height=int(args.height),
                max_focus=int(args.headless_max_focus),
                help_lines=help_lines,
            )
            print(f"[OK] Headless snapshots saved: {len(saved)}")
            for path in saved:
                print(f"  - {path}")
            return 0

        backend = str(args.window_backend)
        if backend == "auto":
            backend = "cv2" if _opencv_gui_available() else "pygame"

        if backend == "cv2":
            try:
                return _run_cv2_viewer(
                    sim=sim,
                    scene_name=scene_name,
                    layout_path=layout_path,
                    loaded_items=loaded_items,
                    object_count=len(objects),
                    skipped=skipped,
                    state=state,
                    width=int(args.width),
                    height=int(args.height),
                    screenshot_dir=Path(args.screenshot_dir),
                    help_lines=help_lines,
                )
            except Exception as exc:
                if args.window_backend == "cv2":
                    for line in _opencv_gui_diagnostics(exc):
                        print(line, file=sys.stderr)
                    return 1
                print("[Warning] OpenCV window backend failed; trying pygame instead.", file=sys.stderr)
                for line in _opencv_gui_diagnostics(exc):
                    print(line, file=sys.stderr)
                backend = "pygame"

        if backend == "pygame":
            return _run_pygame_viewer(
                sim=sim,
                scene_name=scene_name,
                layout_path=layout_path,
                loaded_items=loaded_items,
                object_count=len(objects),
                skipped=skipped,
                state=state,
                width=int(args.width),
                height=int(args.height),
                screenshot_dir=Path(args.screenshot_dir),
                help_lines=help_lines,
            )

        print(f"[Error] Unknown window backend: {backend}", file=sys.stderr)
        return 1
    finally:
        sim.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
