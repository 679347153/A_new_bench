#!/usr/bin/env python3
"""
Interactive layout editor.

用途:
1. 在固定场景中可视化添加物体。
2. 选中物体后微调位置和朝向。
3. 将当前布局保存为 configs 下的 JSON 文件。

详细使用指南:
1. 先移动相机到合适视角。
	- W/S: 前进/后退
	- A/D: 左移/右移
	- E/C: 上升/下降
	- J/L: 左右转向
	- I/K: 上下俯仰
2. 选择准备添加的物体模板。
	- Z: 上一个模板
	- X: 下一个模板
	- 左上角 HUD 的 Template/模板 一行会显示当前模板名。
3. 添加物体。
	- V: 将当前模板添加到相机前方固定距离处。
	- 新增物体会自动成为当前选中物体，方便立刻微调。
4. 选中已有物体。
	- , / < / 9 : 选中上一个物体
	- . / > / 0 : 选中下一个物体
	- 左上角 HUD 的 Selected/选中 一行会显示当前被选中的物体。
	- 如果当前没有任何物体，就无法选中，需要先按 V 添加一个。
5. 调整当前选中物体。
	- t/g: 沿相机前后方向移动（字母键建议不按 Shift）
	- f/y: 沿相机左右方向移动（字母键建议不按 Shift）
	- u/o: 上下移动（字母键建议不按 Shift）
	- 1/2（或 !/@）: 左转/右转
6. 删除或保存。
	- B: 删除当前选中物体
	- M: 保存当前布局到 JSON 文件
7. 切换布局。
	- [ / ]: 切换当前场景 configs 目录下的布局文件
	- 切换布局后会重载该布局中的物体，并自动选中第一个物体

实现说明:
- 当前版本没有鼠标点选，使用“循环切换 selected_idx”的方式选中物体。
- 物体添加位置由当前相机位置和朝向决定，默认生成在相机前方 SPAWN_DISTANCE 米处。
- HUD 文本是叠加在图像上的 2D 文本，不是 Habitat 场景中的 3D 文本。

示例:
	python test_layout.py 00800-TEEsavR23oF --layout scene_objects_v2.json
	python test_layout.py 00800-TEEsavR23oF --ui-lang zh --font-path C:/Windows/Fonts/msyh.ttc
	python test_layout.py  00808-y9hTuugGdiq
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from hm3d_paths import list_available_scenes, resolve_scene_paths

try:
	from PIL import Image, ImageDraw, ImageFont
except ImportError:
	Image = None
	ImageDraw = None
	ImageFont = None

try:
	import magnum as mn
	import habitat_sim
	import habitat_sim.utils.common as utils
except ImportError:
	print("Error: habitat_sim not found. Run this in the correct environment.")
	sys.exit(1)


SCENES_DIR = "hm3d"
OBJECTS_DIR = "./objects"

AVAILABLE_SCENES = list_available_scenes(require_semantic=True)

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CAMERA_MOVE_SPEED = 0.15
OBJECT_MOVE_SPEED = 0.08
ROTATE_SPEED = 2.5
OBJECT_ROTATE_SPEED = 5.0
PITCH_LIMIT = 85.0
SPAWN_DISTANCE = 1.5
CAMERA_HEIGHT = 0.88

UI_TEXT = {
	"en": {
		"help": [
			"=== Camera ===",
			"W/S A/D E/C    Move forward/back left/right up/down",
			"I/K J/L        Pitch / turn camera",
			"N/P            Next / previous scene",
			"R              Reset camera to navigable point",
			"=== Layout ===",
			"[/]            Switch layout file",
			"Z/X            Switch spawn template",
			"V              Add current template in front of camera",
			",/. </> 9/0    Switch selected object",
			";/'            Switch selected object in current room",
			"T/G            Move selected object forward/back",
			"F/Y            Move selected object left/right",
			"U/O            Move selected object up/down",
			"1/2 (!/@)      Rotate selected object left/right",
			"B              Delete selected object",
			"M              Save current layout",
			"H              Toggle help",
			"ESC/Q          Quit",
		],
		"run_from_root": "Please run this script from the project root.",
		"window_title": "Layout Editor  [H]Help  [M]Save  [ESC/Q]Quit",
		"loading_scene": "\\n>>> Loading scene: {scene}",
		"layout_loaded": "    Layout: {name}  (loaded {loaded} / skipped {skipped})",
		"switch_layout": "    Switched layout -> {name} (loaded {loaded} / skipped {skipped})",
		"add_object": "    Added object -> {model}",
		"add_failed": "    Add failed -> {model}: {error}",
		"delete_object": "    Deleted object -> {model}",
		"saved": "    Saved -> {path}",
		"render_error": "Render error: {error}",
		"scene": "Scene: {scene}  ({idx}/{total})",
		"layout": "Layout: {name}{dirty}",
		"template": "Template: {name}  ({idx}/{total})",
		"object_count": "Objects: {count}  Selected: {selected}",
		"room_info": "Room: {room_id}  In-room objects: {count}",
		"room_objects": "Room objects: {items}",
		"camera": "Camera: ({x:.2f}, {y:.2f}, {z:.2f})  yaw={yaw:.1f} pitch={pitch:.1f}",
		"help_hint": "[H] Help  [M] Save",
		"selected_none": "None",
		"selected_format": "{idx}: {model} @ ({x:.2f}, {y:.2f}, {z:.2f}) yaw={yaw:.1f}",
		"font_warning": "Warning: Chinese UI needs Pillow and a CJK font. Falling back to ASCII-safe rendering.",
	},
	"zh": {
		"help": [
			"=== 相机 ===",
			"W/S A/D E/C    前后 左右 上下移动",
			"I/K J/L        俯仰 / 左右旋转视角",
			"N/P            切换场景",
			"R              相机复位到可导航点",
			"=== 布局 ===",
			"[/]            切换布局文件",
			"Z/X            切换待添加模板",
			"V              添加当前模板到相机前方",
			",/. </> 9/0    切换选中物体",
			";/'            在当前房间内切换物体",
			"T/G            选中物体前后移动",
			"F/Y            选中物体左右移动",
			"U/O            选中物体上下移动",
			"1/2 (!/@)      选中物体左/右旋转",
			"B              删除选中物体",
			"M              保存当前布局",
			"H              显示/隐藏帮助",
			"ESC/Q          退出",
		],
		"run_from_root": "请在项目根目录下运行此脚本",
		"window_title": "布局编辑器  [H]帮助  [M]保存  [ESC/Q]退出",
		"loading_scene": "\\n>>> 加载场景: {scene}",
		"layout_loaded": "    布局: {name}  (加载 {loaded} / 跳过 {skipped})",
		"switch_layout": "    切换布局 -> {name} (加载 {loaded} / 跳过 {skipped})",
		"add_object": "    添加物体 -> {model}",
		"add_failed": "    添加失败 -> {model}: {error}",
		"delete_object": "    删除物体 -> {model}",
		"saved": "    已保存 -> {path}",
		"render_error": "渲染错误: {error}",
		"scene": "场景: {scene}  ({idx}/{total})",
		"layout": "布局: {name}{dirty}",
		"template": "模板: {name}  ({idx}/{total})",
		"object_count": "物体数: {count}  选中: {selected}",
		"room_info": "房间: {room_id}  房间内物体: {count}",
		"room_objects": "房间物体: {items}",
		"camera": "相机: ({x:.2f}, {y:.2f}, {z:.2f})  yaw={yaw:.1f} pitch={pitch:.1f}",
		"help_hint": "[H] 帮助  [M] 保存",
		"selected_none": "无",
		"selected_format": "{idx}: {model} @ ({x:.2f}, {y:.2f}, {z:.2f}) yaw={yaw:.1f}",
		"font_warning": "警告: 中文界面需要 Pillow 和中文字体，当前退回到 ASCII 安全渲染。",
	},
}

DEFAULT_FONT_CANDIDATES = [
	r"C:/Windows/Fonts/msyh.ttc",
	r"C:/Windows/Fonts/msyhbd.ttc",
	r"C:/Windows/Fonts/simhei.ttf",
	r"C:/Windows/Fonts/simsun.ttc",
	"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
	"/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
	"/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
]


def contains_non_ascii(text):
	return any(ord(ch) > 127 for ch in text)


def find_default_font_path():
	for path in DEFAULT_FONT_CANDIDATES:
		if os.path.isfile(path):
			return path
	return None


def build_text_renderer(ui_lang, font_path=None, font_size=20):
	font = None
	resolved_font_path = font_path or find_default_font_path()
	if ui_lang == "zh" and ImageFont is not None and resolved_font_path:
		try:
			font = ImageFont.truetype(resolved_font_path, font_size)
		except Exception:
			font = None

	def draw_text(img, lines, x, y, color=(255, 255, 80), scale=0.52, thickness=1):
		line_height = 21
		if font is not None and any(contains_non_ascii(line) for line in lines):
			rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			pil_img = Image.fromarray(rgb)
			draw = ImageDraw.Draw(pil_img)
			for line in lines:
				draw.text((x + 1, y + 1), line, font=font, fill=(0, 0, 0))
				draw.text((x, y), line, font=font, fill=(int(color[2]), int(color[1]), int(color[0])))
				y += line_height
			return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

		for line in lines:
			safe_line = line if not contains_non_ascii(line) else line.encode("ascii", errors="replace").decode("ascii")
			cv2.putText(img, safe_line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
						scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
			cv2.putText(img, safe_line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
						scale, color, thickness, cv2.LINE_AA)
			y += line_height
		return img

	return draw_text, font is not None


def normalize_key(raw_key):
	"""Normalize key code from waitKeyEx to make letter hotkeys case-insensitive."""
	if raw_key < 0:
		return -1
	key = raw_key & 0xFF
	if ord('A') <= key <= ord('Z'):
		return key + 32
	return key


def get_scene_dir(scene_name):
	scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
	if scene_paths is None:
		raise FileNotFoundError(f"场景目录不存在: {scene_name}")
	return str(scene_paths.scene_dir)


def get_scene_id(scene_name):
	scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
	if scene_paths is None:
		raise FileNotFoundError(f"场景目录不存在: {scene_name}")
	return scene_paths.scene_id


def get_stage_glb_path(scene_name):
	scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
	if scene_paths is None:
		raise FileNotFoundError(f"场景目录不存在: {scene_name}")
	return str(scene_paths.stage_glb)


def get_configs_dir(scene_name):
	return os.path.join(get_scene_dir(scene_name), "configs")


def ensure_configs_dir(scene_name):
	os.makedirs(get_configs_dir(scene_name), exist_ok=True)


def list_layout_files(scene_name):
	config_dir = get_configs_dir(scene_name)
	if not os.path.isdir(config_dir):
		return []

	files = []
	for name in sorted(os.listdir(config_dir)):
		if name.lower().endswith(".json"):
			files.append(os.path.join(config_dir, name))

	files.sort(key=lambda p: (0 if os.path.basename(p) == "scene_objects.json" else 1, os.path.basename(p)))
	return files


def get_default_layout_path(scene_name, layout_name):
	ensure_configs_dir(scene_name)
	if not layout_name.lower().endswith(".json"):
		layout_name = f"{layout_name}.json"
	return os.path.join(get_configs_dir(scene_name), layout_name)


def load_scene_rooms(scene_name):
	"""Load room metadata from scene_info json for room-aware selection."""
	candidates = [
		os.path.join("results", "scene_info", scene_name, f"{scene_name}_scene_info.json"),
		os.path.join("results", "scene_info", "temp_export", f"{scene_name}_scene_info.json"),
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


def infer_room_id(position, rooms):
	if position is None or not rooms:
		return -1
	try:
		x, y, z = float(position[0]), float(position[1]), float(position[2])
	except Exception:
		return -1

	for room in rooms:
		bbox = room.get("bounding_box", {})
		min_pt = bbox.get("min", [])
		max_pt = bbox.get("max", [])
		if len(min_pt) < 3 or len(max_pt) < 3:
			continue
		if min_pt[0] <= x <= max_pt[0] and min_pt[1] <= y <= max_pt[1] and min_pt[2] <= z <= max_pt[2]:
			return int(room.get("region_id", -1))
	return -1


def get_room_object_indices(editor_items, room_id, rooms):
	indices = []
	if room_id < 0:
		return indices
	for idx, item in enumerate(editor_items):
		obj = item.get("object")
		if obj is None:
			continue
		if infer_room_id(obj.translation, rooms) == room_id:
			indices.append(idx)
	return indices


def format_room_object_preview(editor_items, indices, max_items=4):
	if not indices:
		return "-"
	labels = []
	for idx in indices[:max_items]:
		labels.append(f"{idx}:{editor_items[idx].get('model_id', '?')}")
	if len(indices) > max_items:
		labels.append(f"+{len(indices) - max_items}")
	return " | ".join(labels)


def scan_object_templates():
	if not os.path.isdir(OBJECTS_DIR):
		raise FileNotFoundError(f"物体模板目录不存在: {OBJECTS_DIR}")

	templates = []
	for name in sorted(os.listdir(OBJECTS_DIR)):
		if not name.endswith(".object_config.json"):
			continue
		templates.append(name[:-len(".object_config.json")])

	if not templates:
		raise RuntimeError("未在 objects 下发现任何 .object_config.json 模板")
	return templates


def make_sim_cfg(scene_name):
	scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
	if scene_paths is None:
		raise FileNotFoundError(f"场景目录不存在: {scene_name}")

	sim_cfg = habitat_sim.SimulatorConfiguration()
	sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
	sim_cfg.scene_id = str(scene_paths.stage_glb)
	sim_cfg.enable_physics = False
	sim_cfg.gpu_device_id = 0

	sensor = habitat_sim.CameraSensorSpec()
	sensor.uuid = "color"
	sensor.sensor_type = habitat_sim.SensorType.COLOR
	sensor.resolution = [DISPLAY_HEIGHT, DISPLAY_WIDTH]
	sensor.hfov = 90
	sensor.position = [0.0, CAMERA_HEIGHT, 0.0]

	agent_cfg = habitat_sim.agent.AgentConfiguration()
	agent_cfg.sensor_specifications = [sensor]
	agent_cfg.height = CAMERA_HEIGHT
	agent_cfg.radius = 0.18

	return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def rotation_from_euler(yaw_deg, pitch_deg):
	qy = utils.quat_from_angle_axis(math.radians(yaw_deg), np.array([0, 1, 0]))
	qx = utils.quat_from_angle_axis(math.radians(pitch_deg), np.array([1, 0, 0]))
	return qy * qx


def yaw_to_quat(yaw_deg):
	return utils.quat_from_angle_axis(math.radians(yaw_deg), np.array([0, 1, 0]))


def yaw_to_magnum_quat(yaw_deg):
	return mn.Quaternion.rotation(mn.Rad(math.radians(yaw_deg)), mn.Vector3(0.0, 1.0, 0.0))


def prettify_model_name(model_id):
	return model_id.replace("_", " ").title()


def resolve_template_handle(template_mgr, model_id):
	# 1) exact
	candidates = template_mgr.get_template_handles(model_id)
	if candidates:
		return candidates[0]

	# 2) exact with suffix
	if not model_id.endswith(".object_config.json"):
		candidates = template_mgr.get_template_handles(f"{model_id}.object_config.json")
		if candidates:
			return candidates[0]

	# 3) auto *_4k variant
	if not model_id.endswith("_4k") and not model_id.endswith("_4k.object_config.json"):
		model_4k = f"{model_id}_4k"
		candidates = template_mgr.get_template_handles(model_4k)
		if candidates:
			return candidates[0]
		candidates = template_mgr.get_template_handles(f"{model_4k}.object_config.json")
		if candidates:
			return candidates[0]

	# 4) case-insensitive fallback
	try:
		needle = model_id.lower().replace(".object_config.json", "")
		all_handles = template_mgr.get_template_handles()
		for handle in all_handles:
			name = os.path.basename(handle).lower().replace(".object_config.json", "")
			if name == needle or name == f"{needle}_4k":
				return handle
	except Exception:
		pass

	return None


def debug_list_templates(template_mgr):
	"""List all loaded object templates for debugging."""
	try:
		all_handles = template_mgr.get_template_handles()
		return all_handles if all_handles else ["(no templates loaded)"]
	except Exception:
		return []


def safe_remove_object(rom, item):
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


def clear_editor_objects(sim, editor_items):
	rom = sim.get_rigid_object_manager()
	for item in list(editor_items):
		safe_remove_object(rom, item)
	editor_items.clear()


def create_editor_item(sim, model_id, position, yaw_deg=0.0):
	rom = sim.get_rigid_object_manager()
	template_mgr = sim.get_object_template_manager()
	template_handle = resolve_template_handle(template_mgr, model_id)
	if template_handle is None:
		raise RuntimeError(f"找不到模板: {model_id}")

	obj = rom.add_object_by_template_handle(template_handle)
	if obj is None:
		raise RuntimeError(f"无法实例化模板: {template_handle}")

	# 这里必须给刚体对象使用 Magnum Quaternion。
	# 你看到的 semantic warnings 只是提示当前场景没有语义标注，不会阻止添加物体。
	# 真正导致按 V 失败的是: obj.rotation 不能接收 utils.quat_from_angle_axis 生成的 quaternion 类型。
	obj.translation = np.array(position, dtype=np.float32)
	obj.rotation = yaw_to_magnum_quat(yaw_deg)
	# 在编辑器中使用 KINEMATIC，才能在按键微调时稳定地实时更新位姿。
	obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

	return {
		"object": obj,
		"object_id": getattr(obj, "object_id", None),
		"handle": getattr(obj, "handle", None),
		"model_id": model_id,
		"name": prettify_model_name(model_id),
		"yaw_deg": float(yaw_deg),
	}


def extract_yaw_deg(rotation_value):
	if isinstance(rotation_value, list) and len(rotation_value) == 3:
		return float(rotation_value[1])
	return 0.0


def load_layout_into_editor(sim, layout_path, editor_items):
	clear_editor_objects(sim, editor_items)

	if not os.path.isfile(layout_path):
		return 0, 0

	with open(layout_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	loaded = 0
	skipped = 0
	for obj_cfg in data.get("objects", []):
		model_id = obj_cfg.get("model_id") or obj_cfg.get("template_name")
		position = obj_cfg.get("position") or obj_cfg.get("translation")
		if not model_id or not isinstance(position, list) or len(position) != 3:
			skipped += 1
			continue

		yaw_deg = extract_yaw_deg(obj_cfg.get("rotation"))
		try:
			item = create_editor_item(sim, model_id, position, yaw_deg)
			editor_items.append(item)
			loaded += 1
		except Exception as exc:
			print(f"      [skip] {model_id}: {exc}")
			skipped += 1

	return loaded, skipped


def save_layout(scene_name, layout_path, editor_items):
	ensure_configs_dir(scene_name)
	payload = {
		"scene": get_stage_glb_path(scene_name),
		"timestamp": time.time(),
		"objects": [],
	}

	for index, item in enumerate(editor_items):
		pos = item["object"].translation
		payload["objects"].append({
			"id": index,
			"name": item["name"],
			"model_id": item["model_id"],
			"position": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
			"rotation": [0.0, round(float(item["yaw_deg"]), 4), 0.0],
		})

	with open(layout_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)


def get_camera_vectors(yaw_deg):
	yaw_rad = math.radians(yaw_deg)
	forward = np.array([-math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype=np.float32)
	right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)], dtype=np.float32)
	return forward, right


def pick_spawn_position(camera_pos, yaw_deg):
	# 新物体默认生成在相机前方固定距离处，并尽量贴近当前楼层高度。
	forward, _ = get_camera_vectors(yaw_deg)
	spawn = camera_pos.copy() + forward * SPAWN_DISTANCE
	spawn[1] = max(0.0, float(camera_pos[1] - CAMERA_HEIGHT + 0.1))
	return spawn


def cycle_index(current_idx, size, step):
	if size <= 0:
		return 0
	return (current_idx + step) % size


def find_layout_index(layout_files, target_path):
	for idx, path in enumerate(layout_files):
		if os.path.normcase(path) == os.path.normcase(target_path):
			return idx
	return 0


def refresh_layout_files(scene_name, current_layout_path):
	files = list_layout_files(scene_name)
	if current_layout_path and os.path.isfile(current_layout_path):
		normalized = {os.path.normcase(path) for path in files}
		if os.path.normcase(current_layout_path) not in normalized:
			files.append(current_layout_path)
			files.sort(key=lambda p: os.path.basename(p))
	return files


def main():
	parser = argparse.ArgumentParser(description="Interactive layout editor")
	parser.add_argument("scene", nargs="?", default="00800-TEEsavR23oF", help="Scene name, for example 00800-TEEsavR23oF")
	parser.add_argument("--layout", default="scene_objects.json", help="Layout filename or absolute path")
	parser.add_argument("--ui-lang", choices=["en", "zh"], default="en", help="UI language")
	parser.add_argument("--font-path", default="", help="Path to a CJK font, for example C:/Windows/Fonts/msyh.ttc")
	args = parser.parse_args()

	ui_lang = args.ui_lang
	overlay_text, unicode_ready = build_text_renderer(ui_lang, args.font_path or None)
	if ui_lang == "zh" and not unicode_ready:
		print("Warning: CJK font not found, fallback to English UI.")
		ui_lang = "en"
		overlay_text, unicode_ready = build_text_renderer(ui_lang, args.font_path or None)

	ui = UI_TEXT[ui_lang]

	if not os.path.isdir("hm3d") or not os.path.isdir("objects"):
		print(ui["run_from_root"])
		sys.exit(1)

	if ui_lang == "zh" and not unicode_ready:
		print(ui["font_warning"])

	templates = scan_object_templates()

	scene_idx = 0
	for i, scene_name in enumerate(AVAILABLE_SCENES):
		if args.scene in scene_name:
			scene_idx = i
			break

	# Keep user-provided layout path if it is absolute or points to an existing file.
	if os.path.isabs(args.layout) or os.path.isfile(args.layout):
		current_layout_path = os.path.abspath(args.layout)
	else:
		current_layout_path = get_default_layout_path(AVAILABLE_SCENES[scene_idx], args.layout)

	win = ui["window_title"]
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(win, DISPLAY_WIDTH, DISPLAY_HEIGHT)

	sim = None
	scene_layout_files = []
	layout_idx = 0
	editor_items = []
	selected_idx = -1
	template_idx = 0
	show_help = True
	dirty = False
	status_text = ""
	scene_rooms = []
	camera_pos = np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
	yaw = 0.0
	pitch = 0.0

	def set_status(message):
		nonlocal status_text
		status_text = message
		print(message)

	def select_after_load():
		nonlocal selected_idx
		selected_idx = 0 if editor_items else -1

	def update_layout_catalog(scene_name):
		nonlocal scene_layout_files, layout_idx
		scene_layout_files = refresh_layout_files(scene_name, current_layout_path)
		layout_idx = find_layout_index(scene_layout_files, current_layout_path) if scene_layout_files else 0

	def load_scene(idx):
		nonlocal sim, camera_pos, yaw, pitch, dirty, current_layout_path, scene_rooms
		scene_name = AVAILABLE_SCENES[idx]
		set_status(ui["loading_scene"].format(scene=scene_name))
		cfg = make_sim_cfg(scene_name)

		if sim is not None:
			sim.close()
		sim = habitat_sim.Simulator(cfg)

		# 显式加载物体配置目录
		try:
			template_mgr = sim.get_object_template_manager()
			abs_objects_dir = os.path.abspath(OBJECTS_DIR)
			if os.path.isdir(abs_objects_dir):
				# 尝试多种可能的加载方法
				if hasattr(template_mgr, "load_configs"):
					template_mgr.load_configs(abs_objects_dir)
					print(f"[Info] Loaded object configs from: {abs_objects_dir}")
				elif hasattr(template_mgr, "add_template_search_path"):
					template_mgr.add_template_search_path(abs_objects_dir)
					print(f"[Info] Added template search path: {abs_objects_dir}")
				elif hasattr(template_mgr, "load_object_configs"):
					template_mgr.load_object_configs(abs_objects_dir)
					print(f"[Info] Loaded object configs (method 3): {abs_objects_dir}")
			# Debug output
			loaded_templates = debug_list_templates(template_mgr)
			print(f"[Debug] Loaded templates: {len(loaded_templates)} template(s)")
			if loaded_templates and loaded_templates[0] != "(no templates loaded)":
				print(f"        Examples: {loaded_templates[:3]}")
		except Exception as exc:
			print(f"[Warning] 无法加载物体配置: {exc}")

		# Only remap to scene configs when --layout is just a filename token.
		layout_is_named_config = os.path.dirname(args.layout) == "" and not os.path.isfile(args.layout)
		if not os.path.isabs(args.layout) and layout_is_named_config:
			current_layout_path = get_default_layout_path(scene_name, os.path.basename(current_layout_path))
		update_layout_catalog(scene_name)

		loaded, skipped = load_layout_into_editor(sim, current_layout_path, editor_items)
		select_after_load()
		scene_rooms = load_scene_rooms(scene_name)
		camera_pos = np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
		yaw = 0.0
		pitch = 0.0
		dirty = False
		set_status(ui["layout_loaded"].format(name=os.path.basename(current_layout_path), loaded=loaded, skipped=skipped))

	load_scene(scene_idx)

	while True:
		raw_key = cv2.waitKeyEx(30)
		key = normalize_key(raw_key)
		if key < 0:
			continue
		scene_name = AVAILABLE_SCENES[scene_idx]

		if key in (27, ord('q')):
			break

		if key == ord('n'):
			scene_idx = (scene_idx + 1) % len(AVAILABLE_SCENES)
			load_scene(scene_idx)
			continue
		if key == ord('p'):
			scene_idx = (scene_idx - 1) % len(AVAILABLE_SCENES)
			load_scene(scene_idx)
			continue
		if key == ord(']'):
			update_layout_catalog(scene_name)
			if scene_layout_files:
				layout_idx = cycle_index(layout_idx, len(scene_layout_files), 1)
				current_layout_path = scene_layout_files[layout_idx]
				loaded, skipped = load_layout_into_editor(sim, current_layout_path, editor_items)
				select_after_load()
				dirty = False
				set_status(ui["switch_layout"].format(name=os.path.basename(current_layout_path), loaded=loaded, skipped=skipped))
			continue
		if key == ord('['):
			update_layout_catalog(scene_name)
			if scene_layout_files:
				layout_idx = cycle_index(layout_idx, len(scene_layout_files), -1)
				current_layout_path = scene_layout_files[layout_idx]
				loaded, skipped = load_layout_into_editor(sim, current_layout_path, editor_items)
				select_after_load()
				dirty = False
				set_status(ui["switch_layout"].format(name=os.path.basename(current_layout_path), loaded=loaded, skipped=skipped))
			continue

		if key == ord('h'):
			show_help = not show_help

		if key == ord('r'):
			camera_pos = np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
			yaw = 0.0
			pitch = 0.0
			try:
				nav_pt = sim.pathfinder.get_random_navigable_point()
				camera_pos = np.array([nav_pt[0], nav_pt[1] + CAMERA_HEIGHT, nav_pt[2]], dtype=np.float32)
			except Exception:
				pass

		if key == ord('z'):
			template_idx = cycle_index(template_idx, len(templates), -1)
		if key == ord('x'):
			template_idx = cycle_index(template_idx, len(templates), 1)

		if key == ord('v'):
			# 添加物体流程:
			# 1. 读取当前模板 template_idx
			# 2. 根据相机位置和朝向计算生成点
			# 3. 实例化物体
			# 4. 自动选中新物体，便于继续微调
			spawn_pos = pick_spawn_position(camera_pos, yaw)
			model_id = templates[template_idx]
			try:
				item = create_editor_item(sim, model_id, spawn_pos, yaw)
				editor_items.append(item)
				selected_idx = len(editor_items) - 1
				dirty = True
				set_status(ui["add_object"].format(model=model_id))
			except Exception as exc:
				set_status(ui["add_failed"].format(model=model_id, error=exc))

		# 当前版本不支持鼠标点选，因此通过循环 selected_idx 来切换选中物体。
		# 支持标点和数字两套热键，适配不同键盘布局和输入法。
		if key in (ord(','), ord('<'), ord('9')) and editor_items:
			selected_idx = cycle_index(selected_idx, len(editor_items), -1)
		if key in (ord('.'), ord('>'), ord('0')) and editor_items:
			selected_idx = cycle_index(selected_idx, len(editor_items), 1)

		# 在当前房间内切换选中物体。
		if key in (ord(';'), ord(':')) and editor_items and scene_rooms:
			selected_item = editor_items[selected_idx] if 0 <= selected_idx < len(editor_items) else None
			if selected_item is not None:
				room_id = infer_room_id(selected_item["object"].translation, scene_rooms)
				same_room_indices = get_room_object_indices(editor_items, room_id, scene_rooms)
				if selected_idx in same_room_indices and len(same_room_indices) > 1:
					pos = same_room_indices.index(selected_idx)
					selected_idx = same_room_indices[(pos - 1) % len(same_room_indices)]

		if key == ord("'") and editor_items and scene_rooms:
			selected_item = editor_items[selected_idx] if 0 <= selected_idx < len(editor_items) else None
			if selected_item is not None:
				room_id = infer_room_id(selected_item["object"].translation, scene_rooms)
				same_room_indices = get_room_object_indices(editor_items, room_id, scene_rooms)
				if selected_idx in same_room_indices and len(same_room_indices) > 1:
					pos = same_room_indices.index(selected_idx)
					selected_idx = same_room_indices[(pos + 1) % len(same_room_indices)]

		if key == ord('b') and 0 <= selected_idx < len(editor_items):
			# 删除后尽量保持 selected_idx 仍指向一个有效对象，避免越界。
			item = editor_items.pop(selected_idx)
			safe_remove_object(sim.get_rigid_object_manager(), item)
			selected_idx = min(selected_idx, len(editor_items) - 1)
			dirty = True
			set_status(ui["delete_object"].format(model=item["model_id"]))

		forward, right = get_camera_vectors(yaw)
		selected_item = editor_items[selected_idx] if 0 <= selected_idx < len(editor_items) else None
		if selected_item is not None:
			moved = False
			obj = selected_item["object"]
			pos = obj.translation
			if key == ord('t'):
				obj.translation = pos + forward * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('g'):
				obj.translation = pos - forward * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('f'):
				obj.translation = pos - right * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('y'):
				obj.translation = pos + right * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('u'):
				obj.translation = pos + np.array([0.0, OBJECT_MOVE_SPEED, 0.0], dtype=np.float32)
				moved = True
			if key == ord('o'):
				obj.translation = pos - np.array([0.0, OBJECT_MOVE_SPEED, 0.0], dtype=np.float32)
				moved = True
			if key in (ord('1'), ord('!')):
				selected_item["yaw_deg"] -= OBJECT_ROTATE_SPEED
				obj.rotation = yaw_to_magnum_quat(selected_item["yaw_deg"])
				moved = True
			if key in (ord('2'), ord('@')):
				selected_item["yaw_deg"] += OBJECT_ROTATE_SPEED
				obj.rotation = yaw_to_magnum_quat(selected_item["yaw_deg"])
				moved = True
			if moved:
				dirty = True

		if key == ord('m'):
			save_layout(scene_name, current_layout_path, editor_items)
			update_layout_catalog(scene_name)
			dirty = False
			set_status(ui["saved"].format(path=current_layout_path))

		if key == ord('j'):
			yaw += ROTATE_SPEED
		if key == ord('l'):
			yaw -= ROTATE_SPEED
		if key == ord('i'):
			pitch = min(pitch + ROTATE_SPEED, PITCH_LIMIT)
		if key == ord('k'):
			pitch = max(pitch - ROTATE_SPEED, -PITCH_LIMIT)

		if key == ord('w'):
			camera_pos += forward * CAMERA_MOVE_SPEED
		if key == ord('s'):
			camera_pos -= forward * CAMERA_MOVE_SPEED
		if key == ord('a'):
			camera_pos -= right * CAMERA_MOVE_SPEED
		if key == ord('d'):
			camera_pos += right * CAMERA_MOVE_SPEED
		if key == ord('e'):
			camera_pos[1] += CAMERA_MOVE_SPEED
		if key == ord('c'):
			camera_pos[1] -= CAMERA_MOVE_SPEED

		try:
			agent = sim.get_agent(0)
			state = agent.get_state()
			state.position = camera_pos.copy()
			state.rotation = rotation_from_euler(yaw, pitch)
			agent.set_state(state, reset_sensors=False)
			obs = sim.get_sensor_observations()
			rgb = obs["color"][:, :, :3]
			frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
		except Exception as exc:
			frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
			frame = overlay_text(frame, [ui["render_error"].format(error=exc)], 20, 60, color=(80, 80, 255))

		selected_label = ui["selected_none"]
		room_id = -1
		room_count = 0
		room_preview = "-"
		if 0 <= selected_idx < len(editor_items):
			item = editor_items[selected_idx]
			pos = item["object"].translation
			room_id = infer_room_id(pos, scene_rooms)
			room_indices = get_room_object_indices(editor_items, room_id, scene_rooms)
			room_count = len(room_indices)
			room_preview = format_room_object_preview(editor_items, room_indices)
			selected_label = ui["selected_format"].format(
				idx=selected_idx,
				model=item["model_id"],
				x=pos[0],
				y=pos[1],
				z=pos[2],
				yaw=item["yaw_deg"],
			)

		hud = [
			ui["scene"].format(scene=scene_name, idx=scene_idx + 1, total=len(AVAILABLE_SCENES)),
			ui["layout"].format(name=os.path.basename(current_layout_path), dirty=" *" if dirty else ""),
			ui["template"].format(name=templates[template_idx], idx=template_idx + 1, total=len(templates)),
			ui["object_count"].format(count=len(editor_items), selected=selected_label),
			ui["room_info"].format(room_id=room_id, count=room_count),
			ui["room_objects"].format(items=room_preview),
			ui["camera"].format(x=camera_pos[0], y=camera_pos[1], z=camera_pos[2], yaw=yaw, pitch=pitch),
			ui["help_hint"],
		]
		frame = overlay_text(frame, hud, 10, 22, color=(80, 255, 255))

		if status_text:
			frame = overlay_text(frame, [status_text], 10, DISPLAY_HEIGHT - 16, color=(255, 220, 120))

		if show_help:
			help_lines = ui["help"]
			y0 = DISPLAY_HEIGHT - len(help_lines) * 21 - 40
			frame = overlay_text(frame, help_lines, 10, y0, color=(80, 255, 80))

		cv2.imshow(win, frame)

	if sim is not None:
		clear_editor_objects(sim, editor_items)
		sim.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
