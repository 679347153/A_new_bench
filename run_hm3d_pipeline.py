#!/usr/bin/env python3
"""
One-click HM3D minival pipeline:
1) Build standardized layout under ./hm3d/minival
2) Export scene info JSON for all scenes
3) Render semantic demo frames for scenes with semantic annotations
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_HAB = SCRIPT_DIR / "hm3d-minival-habitat-v0.2"
SRC_SEM = SCRIPT_DIR / "hm3d-minival-semantic-annots-v0.2"
SRC_CFG = SCRIPT_DIR / "hm3d-minival-semantic-configs-v0.2" / "hm3d_annotated_basis.scene_dataset_config.json"

DST_ROOT = SCRIPT_DIR / "hm3d"
DST_MINIVAL = DST_ROOT / "minival"
DST_CFG = DST_ROOT / "hm3d_annotated_basis.scene_dataset_config.json"

OUT_JSON = SCRIPT_DIR / "scene_info_export"
OUT_SEM = SCRIPT_DIR / "semantic_demo_output"
REPORT_PATH = SCRIPT_DIR / "hm3d_pipeline_report.json"


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def build_standard_layout() -> tuple[list[str], list[str], list[str]]:
    if not SRC_HAB.exists():
        raise FileNotFoundError(f"Missing source folder: {SRC_HAB}")
    if not SRC_CFG.exists():
        raise FileNotFoundError(f"Missing dataset config: {SRC_CFG}")

    DST_MINIVAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_CFG, DST_CFG)

    all_scenes: list[str] = []
    semantic_ready_scenes: list[str] = []
    semantic_missing_scenes: list[str] = []

    for scene_dir in sorted([p for p in SRC_HAB.iterdir() if p.is_dir()]):
        scene_name = scene_dir.name
        all_scenes.append(scene_name)

        if "-" in scene_name:
            stem = scene_name.split("-", 1)[1]
        else:
            stem = scene_name

        dst_scene = DST_MINIVAL / scene_name
        dst_scene.mkdir(parents=True, exist_ok=True)

        copy_if_exists(scene_dir / f"{stem}.basis.glb", dst_scene / f"{stem}.basis.glb")
        copy_if_exists(scene_dir / f"{stem}.basis.navmesh", dst_scene / f"{stem}.basis.navmesh")

        sem_dir = SRC_SEM / scene_name
        sem_txt_ok = copy_if_exists(sem_dir / f"{stem}.semantic.txt", dst_scene / f"{stem}.semantic.txt")
        copy_if_exists(sem_dir / f"{stem}.semantic.glb", dst_scene / f"{stem}.semantic.glb")

        if sem_txt_ok:
            semantic_ready_scenes.append(scene_name)
        else:
            semantic_missing_scenes.append(scene_name)

    return all_scenes, semantic_ready_scenes, semantic_missing_scenes


def run_command(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=SCRIPT_DIR, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(cmd)}")


def export_all_scene_info() -> None:
    OUT_JSON.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "export_scene_info.py",
            "--all",
            "--data-dir",
            str(DST_MINIVAL),
            "--dataset-config",
            str(DST_CFG),
            "--output-dir",
            str(OUT_JSON),
        ]
    )


def render_semantic_demos(semantic_scenes: list[str], frames: int = 6) -> dict[str, str]:
    OUT_SEM.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}

    for scene in semantic_scenes:
        scene_out = OUT_SEM / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "demo_hm3d_semantic.py",
            "--scene",
            scene,
            "--save-dir",
            str(scene_out),
            "--frames",
            str(frames),
        ]
        try:
            run_command(cmd)
            status[scene] = "ok"
        except Exception as exc:  # pylint: disable=broad-except
            status[scene] = f"failed: {exc}"

    return status


def write_report(
    all_scenes: list[str],
    semantic_ready_scenes: list[str],
    semantic_missing_scenes: list[str],
    semantic_render_status: dict[str, str],
) -> None:
    report = {
        "dataset_root": str(DST_ROOT),
        "minival_root": str(DST_MINIVAL),
        "dataset_config": str(DST_CFG),
        "all_scene_count": len(all_scenes),
        "all_scenes": all_scenes,
        "semantic_ready_count": len(semantic_ready_scenes),
        "semantic_ready_scenes": semantic_ready_scenes,
        "semantic_missing_count": len(semantic_missing_scenes),
        "semantic_missing_scenes": semantic_missing_scenes,
        "scene_info_output_dir": str(OUT_JSON),
        "semantic_demo_output_dir": str(OUT_SEM),
        "semantic_render_status": semantic_render_status,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Report written: {REPORT_PATH}")


def main() -> None:
    print("[STEP] Build standardized hm3d/minival layout")
    all_scenes, semantic_ready_scenes, semantic_missing_scenes = build_standard_layout()

    print("[STEP] Export scene info for all scenes")
    export_all_scene_info()

    print("[STEP] Render semantic demos for semantic-ready scenes")
    semantic_render_status = render_semantic_demos(semantic_ready_scenes)

    write_report(
        all_scenes,
        semantic_ready_scenes,
        semantic_missing_scenes,
        semantic_render_status,
    )

    print("[DONE] Pipeline completed")


if __name__ == "__main__":
    main()
