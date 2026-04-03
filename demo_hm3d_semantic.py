#!/usr/bin/env python3
"""
HM3DSem 语义数据读取演示脚本
用途：用 Habitat 加载 HM3D 场景，渲染 RGB 图 + 彩色语义分割图，并打印语义物体列表。

运行方式（在 agentrag 环境下）：
    conda run -n agentrag python AgenticRAG/scripts/demo_hm3d_semantic.py
    或指定场景：
    conda run -n agentrag python AgenticRAG/scripts/demo_hm3d_semantic.py --scene 00824-Dd4bFSTQ8gi
    保存输出图：
    conda run -n agentrag python AgenticRAG/scripts/demo_hm3d_semantic.py --save-dir /tmp/sem_demo
"""

import argparse
import os
import sys

import numpy as np

from hm3d_paths import list_available_scenes, resolve_scene_paths

try:
    import habitat_sim
except ImportError:
    print("Error: habitat_sim not found. Please activate the agentrag conda environment.")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None

# ─────────────────────── 默认路径配置 ───────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENE  = "00800-TEEsavR23oF"
AVAILABLE_SCENES = list_available_scenes(require_semantic=True)

# 调色板：为每个 category index 分配固定颜色（HSV 均匀分布，转 BGR）
def _build_palette(n: int) -> np.ndarray:
    """返回 shape=(n,3) uint8 BGR 调色板，颜色均匀分布于色轮。"""
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        hue = int(179 * i / max(n - 1, 1))
        hsv = np.uint8([[[hue, 220, 200]]])
        if cv2 is not None:
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        else:
            # 简单 fallback：无 cv2 时用随机颜色
            rng = np.random.default_rng(i)
            bgr = rng.integers(80, 255, 3, dtype=np.uint8)
        palette[i] = bgr
    palette[0] = [50, 50, 50]  # index 0 = Unknown → 深灰
    return palette


def make_simulator(scene_name: str) -> habitat_sim.Simulator:
    scene_paths = resolve_scene_paths(scene_name, require_semantic=True)
    if scene_paths is None:
        raise FileNotFoundError(f"场景文件不存在或缺少 semantic.txt: {scene_name}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_paths.stage_glb)
    sim_cfg.scene_dataset_config_file = str(scene_paths.dataset_config)
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = True

    def _sensor(uuid, sensor_type, h=720, w=1280, y=1.5):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [h, w]
        spec.position = [0.0, y, 0.0]
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _sensor("color_sensor",    habitat_sim.SensorType.COLOR),
        _sensor("depth_sensor",    habitat_sim.SensorType.DEPTH),
        _sensor("semantic_sensor", habitat_sim.SensorType.SEMANTIC),
    ]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def render_semantic_colormap(
    sem_obs: np.ndarray,
    id_to_cat_idx: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """
    将语义观测（object instance id → category index → 颜色）转为 BGR 彩图。
    sem_obs: H×W uint32，每像素是 object instance id
    id_to_cat_idx: 大小为 (max_id+1,) 的映射数组
    palette: (num_cats, 3) BGR uint8
    """
    h, w = sem_obs.shape
    sem_clipped = np.clip(sem_obs, 0, len(id_to_cat_idx) - 1)
    cat_idx_map  = id_to_cat_idx[sem_clipped]          # H×W
    cat_clipped  = np.clip(cat_idx_map, 0, len(palette) - 1)
    color_img    = palette[cat_clipped]                 # H×W×3 BGR
    return color_img


def print_semantic_summary(sim: habitat_sim.Simulator) -> tuple:
    """打印并返回 (id_to_cat_idx, category_names)。"""
    sem_scene = sim.semantic_scene
    objects    = [o for o in sem_scene.objects     if o is not None]
    categories = [c for c in sem_scene.categories  if c is not None]

    print(f"\n{'─'*60}")
    print(f"  场景语义信息")
    print(f"{'─'*60}")
    print(f"  语义物体数量: {len(objects)}")
    print(f"  语义类别数量: {len(categories)}")
    print(f"\n  类别列表 (前 20):")
    for i, cat in enumerate(categories[:20]):
        print(f"    [{i:3d}] {cat.name()}")
    if len(categories) > 20:
        print(f"    ... 共 {len(categories)} 个类别")

    print(f"\n  物体样本 (前 15):")
    for obj in objects[:15]:
        cat_name = obj.category.name() if obj.category else "Unknown"
        print(f"    {obj.id:<20s}  category={cat_name}")

    # 构建 instance_id → category_index 的映射数组
    max_id = max((int(o.id.split("_")[-1]) for o in objects if "_" in o.id), default=0)
    id_to_cat_idx = np.zeros(max_id + 2, dtype=np.int32)
    cat_name_list = [c.name() for c in categories]

    for obj in objects:
        try:
            inst_id = int(obj.id.split("_")[-1])
        except (ValueError, IndexError):
            continue
        if obj.category is None:
            continue
        cat_name = obj.category.name()
        if cat_name in cat_name_list:
            id_to_cat_idx[inst_id] = cat_name_list.index(cat_name)

    return id_to_cat_idx, cat_name_list


def overlay_legend(img: np.ndarray, palette: np.ndarray, cat_names: list,
                   visible_cats: set) -> np.ndarray:
    """在图右侧叠加可见类别的颜色图例（仅显示当前帧出现的类别，最多20条）。"""
    if cv2 is None:
        return img
    visible = [c for c in sorted(visible_cats) if 0 <= c < len(cat_names)][:20]
    if not visible:
        return img
    out = img.copy()
    x0, y0 = img.shape[1] - 200, 10
    for i, cat_idx in enumerate(visible):
        color = tuple(int(c) for c in palette[cat_idx])
        y = y0 + i * 22
        cv2.rectangle(out, (x0, y), (x0 + 16, y + 16), color, -1)
        cv2.putText(out, cat_names[cat_idx], (x0 + 20, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, cat_names[cat_idx], (x0 + 20, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, cat_names[cat_idx], (x0 + 20, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out

from typing import Optional

def run(scene_name: str, save_dir: Optional[str], n_frames: int = 6):
    print(f"\n加载场景: {scene_name} ...")
    os.environ["MAGNUM_LOG"]     = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"

    sim = make_simulator(scene_name)
    print("Simulator 已创建")

    id_to_cat_idx, cat_names = print_semantic_summary(sim)
    palette = _build_palette(max(len(cat_names), 80))

    agent  = sim.initialize_agent(0)
    nav_pt = sim.pathfinder.get_random_navigable_point()
    state  = habitat_sim.AgentState()
    state.position = nav_pt
    agent.set_state(state)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"\n渲染 {n_frames} 帧 RGB + Semantic 图像...")
    frames_rgb = []
    frames_sem = []

    for i in range(n_frames):
        # 每帧右转 60°
        sim.step("turn_right")
        obs = sim.get_sensor_observations()

        rgb = obs["color_sensor"][..., :3]          # H×W×3 RGB uint8
        sem = obs["semantic_sensor"].astype(np.int32)  # H×W uint32 instance id

        sem_color = render_semantic_colormap(sem, id_to_cat_idx, palette)

        # 统计当前帧出现的类别
        unique_ids  = np.unique(sem)
        clipped     = np.clip(unique_ids, 0, len(id_to_cat_idx) - 1)
        visible_cats = set(id_to_cat_idx[clipped].tolist())

        sem_legend = overlay_legend(sem_color, palette, cat_names, visible_cats)

        if cv2 is not None:
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frames_rgb.append(rgb_bgr)
            frames_sem.append(sem_legend)

            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"frame{i:02d}_rgb.png"), rgb_bgr)
                cv2.imwrite(os.path.join(save_dir, f"frame{i:02d}_sem.png"), sem_legend)
                print(f"  帧 {i}: 已保存 → {save_dir}/frame{i:02d}_{{rgb,sem}}.png")
            else:
                print(f"  帧 {i}: 渲染完成（unique instance ids: {len(unique_ids)}）")
        else:
            print(f"  帧 {i}: cv2 不可用，跳过保存")

    # 合并展示：RGB 和 Semantic 上下拼接成对比图
    if cv2 is not None and frames_rgb and save_dir:
        for i, (rgb_bgr, sem_bgr) in enumerate(zip(frames_rgb, frames_sem)):
            combined = np.vstack([rgb_bgr, sem_bgr])
            out_path = os.path.join(save_dir, f"frame{i:02d}_compare.png")
            cv2.imwrite(out_path, combined)
        print(f"\n对比图已保存到: {save_dir}")

    sim.close()
    print("\nDONE")


def main():
    parser = argparse.ArgumentParser(description="HM3DSem 语义加载演示")
    parser.add_argument("--scene",    default=DEFAULT_SCENE,
                        help="场景名称，例如 00824-Dd4bFSTQ8gi")
    parser.add_argument("--save-dir", default=None,
                        help="图片输出目录，不指定则只打印不保存")
    parser.add_argument("--frames",   type=int, default=6,
                        help="渲染帧数（每帧旋转 60°）")
    args = parser.parse_args()
    run(args.scene, args.save_dir, args.frames)


if __name__ == "__main__":
    main()
