#!/usr/bin/env python3
"""Render one annotated inspection sheet for every grounded lifespan layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


SCENES = ("00401-H8rQCnvBgo6", "00808-y9hTuugGdiq")


def text(img, value, xy, scale=0.55, color=(235, 235, 235), thickness=1):
    cv2.putText(img, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def make_sheet(overview: np.ndarray, layout_path: Path, scene: str, index: int) -> np.ndarray:
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    canvas = np.full((1440, 2560, 3), (25, 27, 31), dtype=np.uint8)

    # Habitat view, letterboxed into the left panel.
    target_w, target_h = 1780, 1120
    ratio = min(target_w / overview.shape[1], target_h / overview.shape[0])
    view = cv2.resize(overview, (int(overview.shape[1] * ratio), int(overview.shape[0] * ratio)))
    x0, y0 = 30, 135
    canvas[y0:y0 + view.shape[0], x0:x0 + view.shape[1]] = view
    cv2.rectangle(canvas, (x0, y0), (x0 + target_w, y0 + target_h), (85, 90, 100), 2)

    label = payload.get("time_label") or layout_path.stem.replace(f"snapshot_{index:03d}_", "")
    text(canvas, f"HM3D {scene}  |  Layout {index:02d}  |  {label}", (32, 58), 1.05, (255, 255, 255), 2)
    text(canvas, "Habitat overview (left) / numbered top-down object positions (right)", (32, 100), 0.65, (175, 205, 255), 1)

    positions = []
    for obj in objects:
        p = obj.get("position", [0, 0, 0])
        if len(p) >= 3:
            positions.append((float(p[0]), float(p[1]), float(p[2])))
    xs = [p[0] for p in positions] or [0, 1]
    zs = [p[2] for p in positions] or [0, 1]
    xmin, xmax, zmin, zmax = min(xs), max(xs), min(zs), max(zs)
    padx = max((xmax - xmin) * .08, .5)
    padz = max((zmax - zmin) * .08, .5)
    xmin, xmax, zmin, zmax = xmin - padx, xmax + padx, zmin - padz, zmax + padz

    map_x, map_y, map_w, map_h = 1840, 135, 680, 650
    cv2.rectangle(canvas, (map_x, map_y), (map_x + map_w, map_y + map_h), (50, 54, 62), -1)
    cv2.rectangle(canvas, (map_x, map_y), (map_x + map_w, map_y + map_h), (130, 140, 155), 2)
    text(canvas, "TOP-DOWN POSITION MAP (X / Z)", (map_x + 18, map_y + 32), .62, (255, 255, 255), 1)
    colors = [(80,200,255),(120,230,130),(255,170,80),(215,120,255),(90,150,255)]
    for i, (obj, p) in enumerate(zip(objects, positions), 1):
        px = map_x + 35 + int((p[0]-xmin) / max(xmax-xmin, 1e-6) * (map_w-70))
        py = map_y + map_h - 35 - int((p[2]-zmin) / max(zmax-zmin, 1e-6) * (map_h-85))
        color = colors[(i-1) % len(colors)]
        cv2.circle(canvas, (px, py), 13, (20,20,20), -1)
        cv2.circle(canvas, (px, py), 12, color, -1)
        text(canvas, i, (px-8 if i > 9 else px-5, py+5), .43, (15,15,15), 1)
    text(canvas, f"X: {xmin:.2f} .. {xmax:.2f} m", (map_x+18, map_y+map_h-12), .48, (190,195,205), 1)

    list_y = 825
    text(canvas, "OBJECT INDEX AND WORLD POSITION (meters)", (1840, list_y), .58, (255,255,255), 1)
    for i, (obj, p) in enumerate(zip(objects, positions), 1):
        model = str(obj.get("model_id") or obj.get("name") or f"object_{i}")
        if len(model) > 27:
            model = model[:26] + "~"
        line = f"{i:02d} {model:<28} ({p[0]:6.2f}, {p[1]:5.2f}, {p[2]:6.2f})"
        text(canvas, line, (1840, list_y + 28 + (i-1)*29), .40, (220,225,232), 1)

    changed = payload.get("changed_object_ids") or payload.get("changed_objects") or []
    text(canvas, f"Objects: {len(objects)}   Changed metadata entries: {len(changed)}", (32, 1315), .62, (210,215,225), 1)
    text(canvas, f"Source: {layout_path.name}", (32, 1360), .55, (155,165,180), 1)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data-dir", type=Path, help="Optional HM3D root; otherwise use the project path resolver")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    viewer = root / "core" / "visualize_placed_layout.py"

    with tempfile.TemporaryDirectory(prefix="lifespan_render_") as temp_name:
        temp = Path(temp_name)
        for short, scene in (("00401", SCENES[0]), ("00808", SCENES[1])):
            scene_out = output / short
            scene_out.mkdir(parents=True, exist_ok=True)
            layout_dir = root / "results" / "lifespan" / scene / "lifespan_qwen_10" / "grounded" / "layouts"
            for index, layout in enumerate(sorted(layout_dir.glob("snapshot_*.json"))):
                destination = scene_out / f"{short}_layout_{index:02d}_{layout.stem.split('_', 3)[-1]}.png"
                if destination.is_file():
                    print(f"[SKIP existing] {destination}", flush=True)
                    continue
                render_dir = temp / short / f"{index:02d}"
                cmd = [args.python, str(viewer), str(layout), "--scene", scene, "--headless",
                       "--headless-max-focus", "0", "--initial-y-offset", "0", "--width", "1800",
                       "--height", "1120", "--screenshot-dir", str(render_dir)]
                if args.data_dir:
                    cmd += ["--data-dir", str(args.data_dir)]
                subprocess.run(cmd, cwd=root, check=True)
                overview_path = next(render_dir.glob("*_overview_*.png"))
                overview = cv2.imread(str(overview_path), cv2.IMREAD_COLOR)
                sheet = make_sheet(overview, layout, scene, index)
                cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                print(f"[OK] {destination}", flush=True)

    readme = output / "README.txt"
    readme.write_text(
        "场景物体摆放情况\n\n00401、00808 文件夹各含 10 张 2560x1440 检查图。\n"
        "左侧为 Habitat-Sim 场景总览；右侧为物体 X/Z 俯视位置编号及世界坐标 (x,y,z)，单位为米。\n"
        "渲染严格使用布局原始高度（initial-y-offset=0），没有额外抬高物体。\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
