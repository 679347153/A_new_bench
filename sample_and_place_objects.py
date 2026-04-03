#!/usr/bin/env python3
"""
根据概率信息采样物体位置，并通过交互式编辑器进行微调。

逻辑流程：
1. 选择模式：--mode load（读取已有概率）或 generate（随机生成概率）
2. 加载/生成概率文件：对每个物体，前5个房间各分配一个概率（Σ=1）
3. 采样物体位置：每个物体根据概率分布选择一个房间，使用房间几何中心作为位置
4. 生成中间布局JSON
5. 启动 test_layout.py 进行人工微调
6. 保存微调结果，循环重复或退出

用法：
  # 首次运行：生成概率文件
  python sample_and_place_objects.py \
    --scene 00808-y9hTuugGdiq \
    --mode generate \
    --images-dir ./objects_images \
    --rooms-info-dir ./results/scene_info \
    --probabilities-dir ./results/probabilities \
    --layouts-dir ./results/layouts

  # 后续运行：读取已有概率文件
  python sample_and_place_objects.py \
    --scene 00808-y9hTuugGdiq \
    --mode load \
    --probabilities-dir ./results/probabilities \
    --layouts-dir ./results/layouts
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from hm3d_paths import list_available_scenes, resolve_scene_paths

try:
    import numpy as np
except ImportError:
    print("Error: numpy not found. Install with: pip install numpy")
    sys.exit(1)


# ===== 常量 =====
DEFAULT_ROOMS_INFO_DIR = "./results/scene_info"
DEFAULT_PROBABILITIES_DIR = "./results/probabilities"
DEFAULT_LAYOUTS_DIR = "./results/layouts"
DEFAULT_IMAGES_DIR = "./objects_images"

AVAILABLE_SCENES = list_available_scenes(require_semantic=True)


# ===== 模板映射 =====

def build_object_template_index(objects_dir: str = "./objects") -> Dict[str, str]:
    """Build case-insensitive map: object stem -> template model_id."""
    index: Dict[str, str] = {}
    if not os.path.isdir(objects_dir):
        return index

    for filename in os.listdir(objects_dir):
        if not filename.endswith(".object_config.json"):
            continue
        model_id = filename[:-len(".object_config.json")]
        lower_model = model_id.lower()
        index[lower_model] = model_id

        # Add non-_4k alias if template uses _4k suffix.
        if lower_model.endswith("_4k"):
            alias = lower_model[:-3]
            if alias and alias not in index:
                index[alias] = model_id
    return index


def resolve_model_id_for_template(object_name: str, template_index: Dict[str, str]) -> str:
    """Resolve object image stem to an existing template model_id."""
    key = object_name.lower()
    if key in template_index:
        return template_index[key]

    key_4k = f"{key}_4k"
    if key_4k in template_index:
        return template_index[key_4k]

    # Fallback to original name, editor will try flexible matching.
    return object_name


# ===== 概率文件管理 =====

def generate_probabilities(
    object_name: str,
    scene_name: str,
    rooms_info_dir: str,
    probabilities_dir: str,
):
    """
    Load room recommendations from query result and generate random probabilities.
    """
    # Load room recommendations
    rooms_info_path = os.path.join(rooms_info_dir, scene_name, f"{object_name}_rooms.json")
    if not os.path.isfile(rooms_info_path):
        print(f"  [Warning] Room info not found: {rooms_info_path}")
        return None
    
    try:
        with open(rooms_info_path, "r", encoding="utf-8") as f:
            rooms_info = json.load(f)
    except Exception as e:
        print(f"  [Error] Failed to load room info: {e}")
        return None
    
    recommended_rooms = rooms_info.get("recommended_rooms", [])
    if not recommended_rooms:
        print(f"  [Warning] No recommended rooms found for {object_name}")
        return None

    # 允许推荐房间少于5，但至少需要2个。
    if len(recommended_rooms) < 2:
        print(f"  [Warning] Recommended rooms < 2 for {object_name}, skipping")
        return None
    
    # Generate random probabilities for top-N rooms, where N in [2, 5]
    n_rooms = min(5, len(recommended_rooms))
    raw_probs = np.random.rand(n_rooms)
    normalized_probs = raw_probs / raw_probs.sum()
    
    probabilities_data = {
        "object_name": object_name,
        "scene_name": scene_name,
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probabilities": [
            {
                "rank": room["rank"],
                "region_id": room.get("region_id", -1),
                "room_center": room.get("room_center", [0, 0, 0]),
                "probability": float(normalized_probs[i]),
            }
            for i, room in enumerate(recommended_rooms[:n_rooms])
        ],
    }
    
    # Ensure directory exists
    prob_scene_dir = os.path.join(probabilities_dir, scene_name)
    os.makedirs(prob_scene_dir, exist_ok=True)
    
    # Save probabilities
    prob_path = os.path.join(prob_scene_dir, f"{object_name}_probs.json")
    try:
        with open(prob_path, "w", encoding="utf-8") as f:
            json.dump(probabilities_data, f, ensure_ascii=False, indent=2)
        print(f"    ✓ Generated probabilities: {prob_path}")
        return probabilities_data
    except Exception as e:
        print(f"  [Error] Failed to save probabilities: {e}")
        return None


def load_probabilities(object_name: str, scene_name: str, probabilities_dir: str) -> Optional[Dict]:
    """Load pre-generated probabilities."""
    prob_path = os.path.join(probabilities_dir, scene_name, f"{object_name}_probs.json")
    if not os.path.isfile(prob_path):
        print(f"  [Warning] Probability file not found: {prob_path}")
        return None
    
    try:
        with open(prob_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            probs = data.get("probabilities", [])
            if len(probs) < 2:
                print(f"  [Warning] Probability entries < 2: {prob_path}")
                return None
            return data
    except Exception as e:
        print(f"  [Error] Failed to load probabilities: {e}")
        return None


def get_or_create_probabilities(
    object_name: str,
    scene_name: str,
    mode: str,
    rooms_info_dir: str,
    probabilities_dir: str,
) -> Optional[Dict]:
    """Get probabilities: load if exists and mode=load, or generate if mode=generate."""
    if mode == "load":
        return load_probabilities(object_name, scene_name, probabilities_dir)
    elif mode == "generate":
        return generate_probabilities(object_name, scene_name, rooms_info_dir, probabilities_dir)
    else:
        raise ValueError(f"Invalid mode: {mode}")


# ===== 采样与布局生成 =====

def sample_object_positions(
    scene_name: str,
    images_dir: str,
    mode: str,
    rooms_info_dir: str,
    probabilities_dir: str,
) -> Optional[Dict]:
    """
    Sample an object position for each image based on probabilities.
    
    Returns:
        {
            "scene": "...",
            "timestamp": "...",
            "objects": [
                {
                    "id": int,
                    "model_id": str,
                    "name": str,
                    "position": [x, y, z],
                    "rotation": [0, 0, 0],
                    "confidence": float,
                    "source": "probability_sampling"
                },
                ...
            ]
        }
    """
    
    # Scan images
    image_files = []
    if os.path.isdir(images_dir):
        for ext in ["*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(Path(images_dir).glob(ext))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"[Warning] No image files found in {images_dir}")
        return None

    template_index = build_object_template_index("./objects")
    
    sampled_objects = []
    for obj_idx, image_path in enumerate(image_files):
        object_name = image_path.stem
        
        # Get/create probabilities
        probs_data = get_or_create_probabilities(
            object_name, scene_name, mode, rooms_info_dir, probabilities_dir
        )
        if not probs_data:
            print(f"  [Warning] Cannot sample for {object_name}, skipping")
            continue
        
        probabilities = probs_data.get("probabilities", [])
        if not probabilities:
            print(f"  [Warning] No probabilities for {object_name}")
            continue
        if len(probabilities) < 2:
            print(f"  [Warning] Probabilities less than 2 for {object_name}, skipping")
            continue
        
        # Extract room centers and probability weights
        room_centers = [p.get("room_center", [0, 0, 0]) for p in probabilities]
        prob_weights = np.array([p.get("probability", 0) for p in probabilities])
        
        # Normalize if needed
        if prob_weights.sum() > 0:
            prob_weights = prob_weights / prob_weights.sum()
        else:
            prob_weights = np.ones(len(probabilities)) / len(probabilities)
        
        # Sample one room
        sampled_idx = np.random.choice(len(probabilities), p=prob_weights)
        sampled_room = probabilities[sampled_idx]
        sampled_center = sampled_room.get("room_center", [0, 0, 0])
        sampled_confidence = sampled_room.get("probability", 0.5)

        resolved_model_id = resolve_model_id_for_template(object_name, template_index)
        
        # Create object entry
        obj_entry = {
            "id": obj_idx,
            "model_id": resolved_model_id,
            "name": _prettify_model_name(resolved_model_id),
            "position": sampled_center,
            "rotation": [0.0, 0.0, 0.0],
            "confidence": float(sampled_confidence),
            "source": "probability_sampling",
        }
        sampled_objects.append(obj_entry)
        print(
            f"  ✓ Sampled {object_name} -> {resolved_model_id}: "
            f"room={sampled_room.get('region_id')}, center={sampled_center}"
        )
    
    if not sampled_objects:
        print("[Error] No objects sampled")
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True)
    if scene_paths is None:
        print(f"[Error] Scene not found in merged hm3d splits or missing semantic.txt: {scene_name}")
        return None
    
    # Build layout JSON
    layout_json = {
        "scene": str(scene_paths.stage_glb),
        "timestamp": time.time(),
        "objects": sampled_objects,
    }
    
    return layout_json


def _prettify_model_name(model_id: str) -> str:
    """Convert snake_case or camelCase to Title Case."""
    return model_id.replace("_", " ").title()


def _get_scene_id(scene_name: str) -> str:
    """Extract scene ID from scene name (e.g., 00800-TEEsavR23oF -> TEEsavR23oF)."""
    parts = scene_name.split("-", 1)
    return parts[1] if len(parts) > 1 else scene_name


# ===== 编辑器集成 =====

def launch_editor(scene_name: str, layout_json_path: str, ui_lang: str = "zh") -> bool:
    """
    Launch test_layout.py with the sampled layout.
    
    Returns:
        True if editor exited successfully, False otherwise.
    """
    # Determine scene ID from scene_name
    scene_id = _get_scene_id(scene_name)
    
    layout_json_path = os.path.abspath(layout_json_path)

    cmd = [
        sys.executable,
        "test_layout.py",
        scene_name,
        "--layout", layout_json_path,
        "--ui-lang", ui_lang,
    ]
    
    print(f"\n[Info] Launching editor: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"[Error] Failed to launch editor: {e}")
        return False


# ===== 主流程 =====

def interactive_sampling_loop(
    scene_name: str,
    images_dir: str,
    mode: str,
    rooms_info_dir: str,
    probabilities_dir: str,
    layouts_dir: str,
    ui_lang: str = "zh",
):
    """
    Main loop: sample -> edit -> save -> repeat.
    """
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"{'='*60}")
        
        # Step 1: Sample object positions
        print(f"\n[Step 1] Sampling object positions from probabilities...")
        layout_json = sample_object_positions(
            scene_name, images_dir, mode, rooms_info_dir, probabilities_dir
        )
        if not layout_json:
            print("[Error] Failed to sample layout")
            return
        
        # Step 2: Write temporary layout file
        temp_layout_path = os.path.join(layouts_dir, scene_name, f"temp_sampled_{int(time.time())}.json")
        os.makedirs(os.path.dirname(temp_layout_path), exist_ok=True)
        
        try:
            with open(temp_layout_path, "w", encoding="utf-8") as f:
                json.dump(layout_json, f, ensure_ascii=False, indent=2)
            print(f"[Info] Saved sampled layout to: {temp_layout_path}")
        except Exception as e:
            print(f"[Error] Failed to save layout: {e}")
            return
        
        # Step 3: Launch editor
        print(f"\n[Step 2] Launching interactive editor...")
        print(f"  You can now adjust object positions manually.")
        print(f"  Press [M] in the editor to save, then close the window.")
        
        success = launch_editor(scene_name, temp_layout_path, ui_lang)
        if not success:
            print("[Warning] Editor exited with error or was closed")
        
        # Step 4: User decision
        print(f"\n[Step 3] Save final layout?")
        while True:
            user_input = input("Save to final layout? (y/n/q): ").strip().lower()
            if user_input in ("y", "yes"):
                # Find the latest saved layout (should be overwritten by editor)
                scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
                scene_configs_dir = os.path.join(str(scene_paths.scene_dir), "configs") if scene_paths else os.path.join("hm3d", "minival", scene_name, "configs")
                if os.path.isdir(scene_configs_dir):
                    # Get the most recently modified JSON
                    json_files = list(Path(scene_configs_dir).glob("*.json"))
                    if json_files:
                        latest_layout = max(json_files, key=lambda p: p.stat().st_mtime)
                        
                        # Copy to final layouts directory
                        final_layout_dir = os.path.join(layouts_dir, scene_name)
                        os.makedirs(final_layout_dir, exist_ok=True)
                        
                        final_name = f"final_{int(time.time())}.json"
                        final_path = os.path.join(final_layout_dir, final_name)
                        
                        try:
                            with open(latest_layout, "r") as src:
                                data = json.load(src)
                            with open(final_path, "w") as dst:
                                json.dump(data, dst, ensure_ascii=False, indent=2)
                            print(f"✓ Saved final layout to: {final_path}")
                        except Exception as e:
                            print(f"[Error] Failed to save final layout: {e}")
                
                break
            elif user_input in ("n", "no"):
                break
            elif user_input in ("q", "quit"):
                print("Exiting...")
                return
            else:
                print("Please enter y, n, or q")
        
        # Step 5: Ask to continue
        print(f"\n[Step 4] Continue to next iteration?")
        while True:
            user_input = input("Continue sampling and editing? (y/n): ").strip().lower()
            if user_input in ("y", "yes"):
                continue  # Go to next iteration
            elif user_input in ("n", "no"):
                print("Done!")
                return
            else:
                print("Please enter y or n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample object positions based on probabilities and edit in interactive layout editor"
    )
    
    parser.add_argument("--scene", required=True, help="Scene name (e.g., 00800-TEEsavR23oF)")
    parser.add_argument(
        "--mode",
        choices=["load", "generate"],
        default="load",
        help="load: use existing probabilities; generate: create new random probabilities",
    )
    
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Object images directory")
    parser.add_argument("--rooms-info-dir", default=DEFAULT_ROOMS_INFO_DIR, help="Room query results directory")
    parser.add_argument("--probabilities-dir", default=DEFAULT_PROBABILITIES_DIR, help="Probabilities directory")
    parser.add_argument("--layouts-dir", default=DEFAULT_LAYOUTS_DIR, help="Layouts output directory")
    
    parser.add_argument("--ui-lang", choices=["en", "zh"], default="zh", help="Editor UI language")
    
    args = parser.parse_args()
    
    # Validate scene
    if args.scene not in AVAILABLE_SCENES:
        print(f"[Error] Scene {args.scene} not in merged available list")
        sys.exit(1)
    
    # Validate mode
    if args.mode == "load":
        # Check if at least one probability file exists
        prob_dir = os.path.join(args.probabilities_dir, args.scene)
        if not os.path.isdir(prob_dir):
            print(f"[Error] Probability directory not found: {prob_dir}")
            print(f"        Use --mode generate to create probabilities first")
            sys.exit(1)
    elif args.mode == "generate":
        # Check if room info files exist
        rooms_dir = os.path.join(args.rooms_info_dir, args.scene)
        if not os.path.isdir(rooms_dir):
            print(f"[Error] Room info directory not found: {rooms_dir}")
            print(f"        Run query_rooms_for_objects.py first")
            sys.exit(1)
    
    # Run interactive loop
    try:
        interactive_sampling_loop(
            args.scene,
            args.images_dir,
            args.mode,
            args.rooms_info_dir,
            args.probabilities_dir,
            args.layouts_dir,
            args.ui_lang,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
