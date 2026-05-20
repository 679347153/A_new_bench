#!/usr/bin/env python3
from __future__ import annotations

"""
批量生成同一场景下的多个最终物体布局。

用途
----
本脚本面向“同一场景 + 同一批物体图片”反复生成多个不同 layout 的需求。
它不是重新实现整条放置链路，而是作为一个编排层，尽量复用现有文件和函数：

1. `export_scene_info.py`
   准备或复用场景语义信息 `scene_info`。
2. `query_rooms_for_objects.py`
   准备或复用每个物体的候选房间推荐。
3. `sample_and_place_objects.py`
   复用已生成的概率分布，只通过不同随机种子重新 sample 物体房间。
4. `query_room_receptacle_objects.py`
   准备或复用全场景可放置 instance 的上表面结果。
5. `assign_objects_to_receptacle_instances.py`
   复用其中的 LLM/启发式分配函数，把每个物体分配到同房间内的承载 instance。
6. `place_objects_on_instances.py`
   调用最终放置函数，在目标 instance 的 top surface 上生成最终 layout。

核心原则
--------
- 对同一个 scene，`scene_info / room query / probabilities / receptacle surfaces`
  都应尽量只生成一次，后续批量 layout 直接复用。
- 每个 layout 的差异主要来自 `base_seed + layout_index`：
  重新 sample 房间、重新执行 instance assignment、重新在承载面点云上采样落点。
- 默认生成“最终可视化 layout”，不是只生成中间采样布局。
- 默认 instance assignment 使用 LLM；若远端模型不可用，可用
  `--disable-assignment-llm` 切换为启发式分配。

默认输出
--------
脚本会创建一个批次目录：

  results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/

其中包含：

  layout_000_seed_42.json
  layout_001_seed_43.json
  ...
  manifest.json

`manifest.json` 会记录本批次的 scene、seed、复用/生成的缓存路径、
每个 layout 的输出路径、采样数量、分配数量、放置成功/失败数量和失败原因摘要。

常用示例
--------
1. 默认 LLM 分配，生成 10 个最终 layout：

  python batch_generate_layouts.py \
    --scene 00808-y9hTuugGdiq \
    --num-layouts 10 \
    --ssh-key /home/yuhang/Desktop/zw_B200.txt

2. 快速启发式 smoke test，不依赖远端 LLM：

  python batch_generate_layouts.py \
    --scene 00808-y9hTuugGdiq \
    --num-layouts 2 \
    --disable-assignment-llm \
    --disable-surface-llm

3. 指定起始随机种子，便于复现实验：

  python batch_generate_layouts.py \
    --scene 00808-y9hTuugGdiq \
    --num-layouts 5 \
    --base-seed 100

4. 保存每次循环的中间 sampled layout 和 assignment plan：

  python batch_generate_layouts.py \
    --scene 00808-y9hTuugGdiq \
    --num-layouts 3 \
    --keep-intermediates

5. 强制重新生成概率或承载面：

  python batch_generate_layouts.py \
    --scene 00808-y9hTuugGdiq \
    --regenerate-probabilities \
    --regenerate-surfaces

注意
----
- 如果概率文件已经齐全，脚本不会强制要求 room recommendation JSON 存在；
  因为批量循环只需要读取概率并重新 sample。
- 如果概率缺失，脚本会先确保 room recommendation JSON 存在，再调用
  `generate_probabilities(...)` 补齐概率文件。
- 如果未提供 `--surfaces-json`，脚本会优先复用默认路径：
  `results/receptacle_queries/<scene>/<scene>_receptacle_surfaces_all_rooms.json`。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from assign_objects_to_receptacle_instances import (
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_KEY,
    DEFAULT_SSH_PORT,
    DEFAULT_SSH_USER,
    OpenAI,
    SSHTunnel,
    _build_surface_candidates_for_room,
    _find_image_for_object,
    _normalize_assignment_response,
    _normalize_object_id,
    _query_assignment_for_object,
    _safe_int,
)
from extract_room_instances import DEFAULT_DATA_DIR
from place_objects_on_instances import place_objects_on_instances
from sample_and_place_objects import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_LAYOUTS_DIR,
    DEFAULT_PROBABILITIES_DIR,
    DEFAULT_ROOMS_INFO_DIR,
    generate_probabilities,
    sample_object_positions,
)


DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"
IMAGE_EXTENSIONS = ("*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp")


def _image_files(images_dir: str) -> List[Path]:
    root = Path(images_dir)
    files: List[Path] = []
    if root.is_dir():
        for ext in IMAGE_EXTENSIONS:
            files.extend(root.glob(ext))
    return sorted(files)


def _run_command(cmd: Sequence[str], description: str) -> None:
    print(f"[Info] {description}: {' '.join(str(x) for x in cmd)}")
    completed = subprocess.run(list(cmd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def _scene_info_path(scene: str, rooms_info_dir: str) -> Path:
    return Path(rooms_info_dir) / scene / f"{scene}_scene_info.json"


def _ensure_scene_info(args: argparse.Namespace) -> Path:
    path = _scene_info_path(args.scene, args.rooms_info_dir)
    if path.is_file():
        print(f"[Info] Reusing scene_info: {path}")
        return path

    out_dir = Path(args.rooms_info_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "export_scene_info.py",
        "--scene",
        args.scene,
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(out_dir),
    ]
    _run_command(cmd, "Exporting scene_info")
    if not path.is_file():
        raise FileNotFoundError(f"scene_info was not created: {path}")
    return path


def _room_query_path(scene: str, object_name: str, rooms_info_dir: str) -> Path:
    return Path(rooms_info_dir) / scene / f"{object_name}_rooms.json"


def _missing_room_queries(scene: str, images_dir: str, rooms_info_dir: str) -> List[str]:
    missing = []
    for image_path in _image_files(images_dir):
        if not _room_query_path(scene, image_path.stem, rooms_info_dir).is_file():
            missing.append(image_path.stem)
    return missing


def _append_ssh_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(
        [
            "--ssh-host",
            str(args.ssh_host),
            "--ssh-port",
            str(args.ssh_port),
            "--ssh-user",
            str(args.ssh_user),
            "--vllm-host",
            str(args.vllm_host),
            "--vllm-port",
            str(args.vllm_port),
            "--local-port",
            str(args.local_port),
            "--model",
            str(args.model),
            "--timeout",
            str(args.timeout),
        ]
    )
    if args.ssh_key:
        cmd.extend(["--ssh-key", str(args.ssh_key)])
    if args.ssh_password:
        cmd.extend(["--ssh-password", str(args.ssh_password)])


def _ensure_room_queries(args: argparse.Namespace) -> None:
    missing = _missing_room_queries(args.scene, args.images_dir, args.rooms_info_dir)
    if missing and not args.regenerate_room_queries:
        print(f"[Info] Missing room query files: {len(missing)}; generating room recommendations once.")
    elif args.regenerate_room_queries:
        print("[Info] Regenerating room recommendations by request.")
    else:
        print("[Info] Reusing all room recommendation files.")
        return

    cmd = [
        sys.executable,
        "query_rooms_for_objects.py",
        "--scene",
        args.scene,
        "--images-dir",
        args.images_dir,
        "--output-dir",
        args.rooms_info_dir,
        "--max-tokens",
        str(args.room_query_max_tokens),
    ]
    _append_ssh_args(cmd, args)
    if args.skip_api_health_check:
        cmd.append("--skip-api-health-check")
    _run_command(cmd, "Querying object room recommendations")

    still_missing = _missing_room_queries(args.scene, args.images_dir, args.rooms_info_dir)
    if still_missing:
        preview = ", ".join(still_missing[:8])
        raise RuntimeError(f"room query files still missing: {len(still_missing)} preview=[{preview}]")


def _probability_path(scene: str, object_name: str, probabilities_dir: str) -> Path:
    return Path(probabilities_dir) / scene / f"{object_name}_probs.json"


def _ensure_probabilities(args: argparse.Namespace) -> List[Path]:
    ensured: List[Path] = []
    for image_path in _image_files(args.images_dir):
        object_name = image_path.stem
        prob_path = _probability_path(args.scene, object_name, args.probabilities_dir)
        if prob_path.is_file() and not args.regenerate_probabilities:
            ensured.append(prob_path)
            continue
        data = generate_probabilities(
            object_name=object_name,
            scene_name=args.scene,
            rooms_info_dir=args.rooms_info_dir,
            probabilities_dir=args.probabilities_dir,
        )
        if not data or not prob_path.is_file():
            raise RuntimeError(f"failed to generate probability file for {object_name}: {prob_path}")
        ensured.append(prob_path)
    print(f"[Info] Probability files ready: {len(ensured)}")
    return ensured


def _missing_probabilities(scene: str, images_dir: str, probabilities_dir: str) -> List[str]:
    missing = []
    for image_path in _image_files(images_dir):
        if not _probability_path(scene, image_path.stem, probabilities_dir).is_file():
            missing.append(image_path.stem)
    return missing


def _default_surfaces_path(scene: str) -> Path:
    return Path("results") / "receptacle_queries" / scene / f"{scene}_receptacle_surfaces_all_rooms.json"


def _ensure_surfaces(args: argparse.Namespace) -> Path:
    if args.surfaces_json:
        path = Path(args.surfaces_json)
        if not path.is_file():
            raise FileNotFoundError(f"--surfaces-json not found: {path}")
        print(f"[Info] Using provided surfaces json: {path}")
        return path

    path = _default_surfaces_path(args.scene)
    if path.is_file() and not args.regenerate_surfaces:
        print(f"[Info] Reusing surfaces json: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "query_room_receptacle_objects.py",
        "--scene",
        args.scene,
        "--data-dir",
        str(args.data_dir),
        "--scene-info-path",
        str(_scene_info_path(args.scene, args.rooms_info_dir)),
        "--output",
        str(path),
        "--max-results",
        str(args.surface_max_results),
        "--surface-points-per-instance",
        str(args.surface_points_per_instance),
        "--surface-min-points",
        str(args.surface_min_points),
        "--instance-pointcloud-points",
        str(args.surface_instance_pointcloud_points),
    ]
    if args.disable_surface_llm:
        cmd.append("--disable-llm")
    else:
        _append_ssh_args(cmd, args)
        cmd.extend(["--max-tokens", str(args.surface_max_tokens)])
    _run_command(cmd, "Generating receptacle surfaces")
    if not path.is_file():
        raise FileNotFoundError(f"surface query did not create output: {path}")
    return path


def _load_surfaces_payload(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"surfaces json root is not object: {path}")
    resolved = path.expanduser().resolve()
    payload["_source_json_path"] = str(resolved)
    payload["_source_json_dir"] = str(resolved.parent)
    return payload


def _surface_room_map(surfaces_payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for room in surfaces_payload.get("rooms", []) or []:
        try:
            out[int(room.get("room_id"))] = room
        except Exception:
            continue
    return out


def _start_assignment_client(args: argparse.Namespace) -> Tuple[bool, Optional[SSHTunnel], Optional[Any]]:
    if args.disable_assignment_llm:
        print("[Info] Assignment LLM disabled; using heuristic assignment.")
        return False, None, None
    if OpenAI is None:
        print("[Warning] openai package unavailable; using heuristic assignment.", file=sys.stderr)
        return False, None, None
    if not (args.ssh_host and args.ssh_user and (args.ssh_key or args.ssh_password)):
        print("[Warning] SSH args incomplete; using heuristic assignment.", file=sys.stderr)
        return False, None, None

    tunnel = SSHTunnel(
        ssh_host=str(args.ssh_host),
        ssh_port=int(args.ssh_port),
        ssh_user=str(args.ssh_user),
        ssh_password=args.ssh_password,
        ssh_key=args.ssh_key,
        remote_host=args.vllm_host,
        remote_port=int(args.vllm_port),
        local_port=int(args.local_port),
    )
    if not tunnel.start():
        print("[Warning] Assignment tunnel failed; using heuristic assignment.", file=sys.stderr)
        return False, None, None
    client = OpenAI(api_key="EMPTY", base_url=tunnel.base_url, timeout=args.timeout)
    print(f"[Info] Assignment LLM tunnel ready: {tunnel.base_url}")
    return True, tunnel, client


def _assign_objects(
    *,
    args: argparse.Namespace,
    sampled_objects: List[Dict[str, Any]],
    room_map: Dict[int, Dict[str, Any]],
    use_llm: bool,
    client: Optional[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    debug: List[Dict[str, Any]] = []

    for idx, obj in enumerate(sampled_objects):
        room_id = _safe_int(obj.get("sampled_region_id", -1), -1)
        room_entry = room_map.get(room_id)
        if room_entry is None:
            debug.append({"object_id": obj.get("id", idx), "status": "missing_room_surface", "room_id": room_id})
            continue

        candidates = _build_surface_candidates_for_room(room_entry)
        if not candidates:
            debug.append({"object_id": obj.get("id", idx), "status": "empty_candidates", "room_id": room_id})
            continue

        model_id = str(obj.get("model_id", ""))
        name = str(obj.get("name", model_id or f"obj_{idx}"))
        image_path = _find_image_for_object(args.images_dir, model_id=model_id, name=name)
        raw_output = ""
        cleaned_output = ""
        parsed_output: Optional[Dict[str, Any]] = None
        source = "heuristic_assignment"
        status = "heuristic"
        llm_error = ""

        if use_llm and client is not None:
            try:
                raw_output, cleaned_output, parsed_output = _query_assignment_for_object(
                    client=client,
                    model=args.model,
                    scene_name=args.scene,
                    room_id=room_id,
                    object_entry=obj,
                    image_path=image_path,
                    candidates=candidates,
                    max_tokens=int(args.max_tokens),
                )
                source = "llm_assignment"
                status = "llm"
            except Exception as exc:
                llm_error = str(exc)
                status = "llm_error_fallback_heuristic"

        decision = _normalize_assignment_response(parsed_output, candidates, model_id=model_id)
        if int(decision.get("target_instance_id", -1)) < 0:
            debug.append({"object_id": obj.get("id", idx), "status": "invalid_decision", "room_id": room_id})
            continue

        object_id = _normalize_object_id(obj.get("id", idx), fallback=idx)
        assignments.append(
            {
                "object_id": object_id,
                "model_id": model_id,
                "name": name,
                "image_path": image_path,
                "sampled_region_id": room_id,
                "target_room_id": room_id,
                "target_instance_id": int(decision["target_instance_id"]),
                "backup_instance_ids": decision.get("backup_instance_ids", []),
                "confidence_score": float(decision.get("confidence_score", 0.5)),
                "reasoning": str(decision.get("reasoning", "")),
                "source": source if not llm_error else "heuristic_assignment",
            }
        )
        debug.append(
            {
                "object_id": obj.get("id", idx),
                "room_id": room_id,
                "status": status,
                "llm_error": llm_error,
                "candidate_count": len(candidates),
                "target_instance_id": int(decision["target_instance_id"]),
                "raw_output": raw_output,
                "cleaned_output": cleaned_output,
            }
        )

    plan_payload = {
        "scene_name": args.scene,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model if use_llm else "heuristic_only",
        "input_object_count": len(sampled_objects),
        "assignment_count": len(assignments),
        "assignments": assignments,
        "debug": debug,
    }
    summary = {
        "sampled_object_count": len(sampled_objects),
        "assignment_count": len(assignments),
        "llm_error_count": sum(1 for item in debug if item.get("status") == "llm_error_fallback_heuristic"),
        "missing_room_surface_count": sum(1 for item in debug if item.get("status") == "missing_room_surface"),
        "empty_candidates_count": sum(1 for item in debug if item.get("status") == "empty_candidates"),
    }
    return plan_payload, summary


def _failure_summary(stats: Dict[str, Any]) -> str:
    failed_by_reason = stats.get("failed_by_reason", {})
    if isinstance(failed_by_reason, dict) and failed_by_reason:
        return ", ".join(f"{k}={v}" for k, v in sorted(failed_by_reason.items()))
    failed = stats.get("failed_objects", [])
    if not isinstance(failed, list) or not failed:
        return ""
    counts: Dict[str, int] = {}
    for item in failed:
        if isinstance(item, dict):
            reason = str(item.get("reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_batch_dir(args: argparse.Namespace) -> Tuple[str, Path]:
    batch_id = time.strftime("batch_%Y%m%d_%H%M%S")
    out_dir = Path(args.layouts_dir) / args.scene / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return batch_id, out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate multiple final layouts for one scene.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scene", required=True, help="Scene name, e.g. 00808-y9hTuugGdiq")
    parser.add_argument("--num-layouts", type=int, default=10, help="Number of final layouts to generate")
    parser.add_argument("--base-seed", type=int, default=42, help="Seed for layout_000; later layouts use base_seed + index")

    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR, help="Object image directory")
    parser.add_argument("--rooms-info-dir", default=DEFAULT_ROOMS_INFO_DIR, help="Scene info / room query output root")
    parser.add_argument("--probabilities-dir", default=DEFAULT_PROBABILITIES_DIR, help="Probability files root")
    parser.add_argument("--layouts-dir", default=DEFAULT_LAYOUTS_DIR, help="Layout output root")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="HM3D data root")
    parser.add_argument("--objects-dir", default="./objects", help="Object template config directory")
    parser.add_argument("--surfaces-json", default=None, help="Existing receptacle surfaces JSON")

    parser.add_argument("--regenerate-room-queries", action="store_true", help="Regenerate room recommendation files")
    parser.add_argument("--regenerate-probabilities", action="store_true", help="Regenerate probability files")
    parser.add_argument("--regenerate-surfaces", action="store_true", help="Regenerate receptacle surfaces JSON")
    parser.add_argument("--disable-assignment-llm", action="store_true", help="Use heuristic-only object-to-instance assignment")
    parser.add_argument("--disable-surface-llm", action="store_true", help="Use heuristic-only receptacle surface query when surfaces need generation")
    parser.add_argument("--keep-intermediates", action="store_true", help="Write sampled layout and assignment plan for each final layout")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failed layout")
    parser.add_argument("--skip-api-health-check", action="store_true", help="Pass through to query_rooms_for_objects.py")

    parser.add_argument("--min-distance", type=float, default=0.25, help="Minimum pairwise object distance in placement")
    parser.add_argument("--spawn-height", type=float, default=0.3, help="Spawn height above target surface for collidable objects")
    parser.add_argument("--max-trials-per-object", type=int, default=30, help="Max surface candidates tried per object")
    parser.add_argument("--settle-steps", type=int, default=45, help="Habitat physics settle steps")

    parser.add_argument("--surface-max-results", type=int, default=10, help="Max receptacle instances per room")
    parser.add_argument("--surface-instance-pointcloud-points", type=int, default=2048, help="Point count for instance pointcloud extraction")
    parser.add_argument("--surface-points-per-instance", type=int, default=256, help="Saved top-surface points per instance")
    parser.add_argument("--surface-min-points", type=int, default=48, help="Minimum valid top-surface point count")
    parser.add_argument("--surface-max-tokens", type=int, default=2048, help="Max tokens for surface LLM query")
    parser.add_argument("--room-query-max-tokens", type=int, default=2048, help="Max tokens for room recommendation query")

    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST, help="SSH server host")
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT, help="SSH server port")
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER, help="SSH user")
    parser.add_argument("--ssh-password", default=None, help="Legacy SSH password fallback; prefer --ssh-key")
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="SSH private key path")
    parser.add_argument("--vllm-host", default="127.0.0.1", help="Remote vLLM host")
    parser.add_argument("--vllm-port", type=int, default=8000, help="Remote vLLM OpenAI API port")
    parser.add_argument("--local-port", type=int, default=0, help="Local forwarded port; 0 means auto")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI-compatible model name")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for assignment LLM query")
    parser.add_argument("--timeout", type=int, default=3600, help="OpenAI-compatible request timeout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.num_layouts) <= 0:
        print("[Error] --num-layouts must be positive", file=sys.stderr)
        return 1
    if not _image_files(args.images_dir):
        print(f"[Error] No object images found in {args.images_dir}", file=sys.stderr)
        return 1

    batch_id, batch_dir = _make_batch_dir(args)
    manifest_path = batch_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "scene": args.scene,
        "batch_id": batch_id,
        "num_layouts": int(args.num_layouts),
        "base_seed": int(args.base_seed),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {},
        "layouts": [],
    }

    try:
        scene_info_path = _ensure_scene_info(args)
        missing_probs = _missing_probabilities(args.scene, args.images_dir, args.probabilities_dir)
        if missing_probs or args.regenerate_probabilities:
            _ensure_room_queries(args)
        else:
            print("[Info] Reusing probability files; room recommendation files are not needed for this batch.")
        probability_paths = _ensure_probabilities(args)
        surfaces_path = _ensure_surfaces(args)
        surfaces_payload = _load_surfaces_payload(surfaces_path)
        room_map = _surface_room_map(surfaces_payload)

        manifest["paths"] = {
            "scene_info": str(scene_info_path),
            "probabilities_dir": str(Path(args.probabilities_dir) / args.scene),
            "probability_file_count": len(probability_paths),
            "surfaces_json": str(surfaces_path),
            "batch_dir": str(batch_dir),
        }
    except Exception as exc:
        manifest["fatal_error"] = str(exc)
        _write_json(manifest_path, manifest)
        print(f"[Error] Preparation failed: {exc}", file=sys.stderr)
        return 1

    use_llm, tunnel, client = _start_assignment_client(args)
    exit_code = 0
    try:
        for layout_idx in range(int(args.num_layouts)):
            seed = int(args.base_seed) + layout_idx
            print(f"\n[Batch] layout_index={layout_idx} seed={seed}")
            entry: Dict[str, Any] = {
                "layout_index": layout_idx,
                "seed": seed,
                "status": "started",
            }
            try:
                np.random.seed(seed)
                sampled_layout = sample_object_positions(
                    scene_name=args.scene,
                    images_dir=args.images_dir,
                    mode="load",
                    rooms_info_dir=args.rooms_info_dir,
                    probabilities_dir=args.probabilities_dir,
                )
                if not sampled_layout or not isinstance(sampled_layout.get("objects"), list):
                    raise RuntimeError("sampling produced no layout objects")
                sampled_objects = sampled_layout["objects"]

                plan_payload, assignment_summary = _assign_objects(
                    args=args,
                    sampled_objects=sampled_objects,
                    room_map=room_map,
                    use_llm=use_llm,
                    client=client,
                )
                if not plan_payload.get("assignments"):
                    raise RuntimeError("no assignments generated")

                layout_payload = place_objects_on_instances(
                    scene_name=args.scene,
                    assignment_plan=plan_payload,
                    surfaces_payload=surfaces_payload,
                    data_dir=Path(args.data_dir),
                    objects_dir=args.objects_dir,
                    min_distance=float(args.min_distance),
                    spawn_height=float(args.spawn_height),
                    max_trials_per_object=int(args.max_trials_per_object),
                    settle_steps=int(args.settle_steps),
                    seed=seed,
                )
                layout_payload["batch_generation"] = {
                    "batch_id": batch_id,
                    "layout_index": layout_idx,
                    "seed": seed,
                    "sampled_object_count": len(sampled_objects),
                    "assignment_count": int(plan_payload.get("assignment_count", 0)),
                }

                layout_path = batch_dir / f"layout_{layout_idx:03d}_seed_{seed}.json"
                _write_json(layout_path, layout_payload)
                if args.keep_intermediates:
                    _write_json(batch_dir / f"sampled_{layout_idx:03d}_seed_{seed}.json", sampled_layout)
                    _write_json(batch_dir / f"assignment_{layout_idx:03d}_seed_{seed}.json", plan_payload)

                stats = layout_payload.get("auto_placement_stats", {})
                if not isinstance(stats, dict):
                    stats = {}
                entry.update(
                    {
                        "status": "ok",
                        "layout_path": str(layout_path),
                        "sampled_object_count": len(sampled_objects),
                        "assignment_count": int(plan_payload.get("assignment_count", 0)),
                        "placed_count": int(stats.get("placed_count", 0)),
                        "failed_count": int(stats.get("failed_count", 0)),
                        "failed_by_reason": stats.get("failed_by_reason", {}),
                        "failure_summary": _failure_summary(stats),
                        "assignment_summary": assignment_summary,
                    }
                )
                if int(stats.get("placed_count", 0)) <= 0:
                    entry["status"] = "failed"
                    entry["error"] = "placement produced zero objects"
                    exit_code = 1
                    if args.fail_fast:
                        manifest["layouts"].append(entry)
                        break
                print(
                    "[OK] layout={path} placed={placed}/{total} failed={failed}".format(
                        path=layout_path,
                        placed=int(stats.get("placed_count", 0)),
                        total=int(stats.get("total_objects", 0)),
                        failed=int(stats.get("failed_count", 0)),
                    )
                )
            except Exception as exc:
                entry.update({"status": "failed", "error": str(exc)})
                print(f"[Error] layout_index={layout_idx} failed: {exc}", file=sys.stderr)
                exit_code = 1
                if args.fail_fast:
                    manifest["layouts"].append(entry)
                    break
            manifest["layouts"].append(entry)
            _write_json(manifest_path, manifest)
    finally:
        if tunnel is not None:
            tunnel.close()
            print("[Info] Assignment SSH tunnel closed")

    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["success_count"] = sum(1 for item in manifest["layouts"] if item.get("status") == "ok")
    manifest["failed_count"] = sum(1 for item in manifest["layouts"] if item.get("status") != "ok")
    _write_json(manifest_path, manifest)
    print(f"\n[OK] Manifest saved: {manifest_path}")
    print(f"[OK] Batch summary: success={manifest['success_count']} failed={manifest['failed_count']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
