#!/usr/bin/env python3
from __future__ import annotations

"""
Query receptacle instances for one scene (all rooms by default) and extract top-surface points.

Main flow:
1. Read scene rooms and instances via extract_room_instances.py.
2. For each room, ask LLM (or heuristic fallback) which instances are suitable for placing small objects.
3. For each selected instance, extract instance point cloud via extract_room_instances.get_instance_point_cloud.
4. Derive top-surface point cloud and save scene-level JSON for downstream placement.
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from extract_room_instances import (
    DEFAULT_DATA_DIR,
    _load_scene_info,
    extract_room_instances,
    get_instance_point_cloud,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


DEFAULT_OUTPUT_DIR = Path("./results/receptacle_queries")

SYSTEM_PROMPT = (
    "You are an indoor affordance assistant. "
    "Given room instances, identify which objects can stably support placing smaller objects on top."
)

USER_PROMPT_TEMPLATE = """
Task:
Select objects that are suitable as receptacles/surfaces for placing smaller items (e.g., cup, book, phone, decoration).

Hard constraints:
1) You must ONLY choose instance_id values from the provided candidate list.
2) Return ONLY JSON, no markdown and no extra text.
3) confidence_score must be a float in [0, 1], sorted descending.
4) Return between 1 and {max_results} objects.
5) Reasoning must be one short sentence.

Output JSON schema:
{{
  "receptacle_candidates": [
    {{
      "rank": 1,
      "instance_id": 123,
      "confidence_score": 0.91,
      "reasoning": "short sentence"
    }}
  ],
  "overall_notes": "optional short note"
}}

Scene: {scene_name}
Room ID: {room_id}
Room object count: {room_object_count}

Candidate instances (JSON):
{candidates_json}
""".strip()


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tunnel_ready(host: str, port: int, timeout_s: float = 10.0) -> bool:
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _clean_model_output(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class SSHTunnel:
    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        ssh_user: str,
        ssh_password: Optional[str] = None,
        ssh_key: Optional[str] = None,
        remote_host: str = "127.0.0.1",
        remote_port: int = 8000,
        local_port: int = 0,
    ):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.ssh_password = ssh_password
        self.ssh_key = ssh_key
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port if local_port > 0 else _pick_free_local_port()
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{self.local_port}/v1"

    def start(self, timeout_s: float = 30.0) -> bool:
        tunnel_cmd = [
            "ssh",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-p",
            str(self.ssh_port),
            "-N",
            "-L",
            f"127.0.0.1:{self.local_port}:{self.remote_host}:{self.remote_port}",
            f"{self.ssh_user}@{self.ssh_host}",
        ]
        cmd_for_run = tunnel_cmd
        env = os.environ.copy()

        if self.ssh_password:
            if shutil.which("sshpass") is None:
                print("[Error] sshpass not found. Install with: sudo apt-get install -y sshpass", file=sys.stderr)
                return False
            tunnel_cmd[1:1] = [
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "PreferredAuthentications=password,keyboard-interactive",
                "-o",
                "NumberOfPasswordPrompts=1",
            ]
            env["SSHPASS"] = self.ssh_password
            cmd_for_run = ["sshpass", "-e", *tunnel_cmd]
        elif self.ssh_key:
            ssh_key_path = os.path.expanduser(self.ssh_key)
            if not os.path.exists(ssh_key_path):
                print(f"[Error] SSH key not found: {ssh_key_path}", file=sys.stderr)
                return False
            tunnel_cmd[1:1] = ["-i", ssh_key_path]
        else:
            print("[Error] Must provide either --ssh-password or --ssh-key", file=sys.stderr)
            return False

        self.proc = subprocess.Popen(
            cmd_for_run,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        time.sleep(0.2)
        if self.proc.poll() is not None:
            print("[Error] SSH tunnel process exited immediately", file=sys.stderr)
            return False
        if not _wait_tunnel_ready("127.0.0.1", self.local_port, timeout_s):
            print(f"[Error] SSH tunnel did not become ready within {timeout_s}s", file=sys.stderr)
            self.close()
            return False
        return True

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _instance_size_features(instance: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    aabb = instance.get("aabb") or {}
    size = aabb.get("size") or [0.0, 0.0, 0.0]
    if not isinstance(size, list) or len(size) < 3:
        size = [0.0, 0.0, 0.0]
    sx, sy, sz = _safe_float(size[0]), _safe_float(size[1]), _safe_float(size[2])
    top_area = max(0.0, sx * sz)
    volume = max(0.0, sx * sy * sz)
    return sx, sy, sz, top_area, volume


def _build_candidates(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for ins in instances:
        ins_id = int(ins.get("id", -1))
        if ins_id < 0:
            continue
        category = str(ins.get("category", "unknown"))
        sx, sy, sz, top_area, volume = _instance_size_features(ins)
        candidates.append(
            {
                "instance_id": ins_id,
                "category": category,
                "aabb_size": [round(sx, 4), round(sy, 4), round(sz, 4)],
                "top_area_est": round(top_area, 4),
                "volume_est": round(volume, 4),
            }
        )
    return candidates


def _heuristic_fallback(candidates: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    keyword_score = {
        "table": 0.95,
        "desk": 0.92,
        "counter": 0.9,
        "nightstand": 0.88,
        "dresser": 0.87,
        "shelf": 0.86,
        "cabinet": 0.85,
        "bench": 0.82,
        "bed": 0.8,
        "sofa": 0.72,
        "chair": 0.6,
    }
    scored = []
    for c in candidates:
        category = str(c.get("category", "")).lower()
        score = 0.35
        for key, val in keyword_score.items():
            if key in category:
                score = max(score, val)
        top_area = _safe_float(c.get("top_area_est"), 0.0)
        if top_area > 0.8:
            score += 0.1
        elif top_area > 0.3:
            score += 0.05
        score = min(0.99, score)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = scored[: max(1, max_results)]
    output = []
    for i, (score, c) in enumerate(chosen, start=1):
        output.append(
            {
                "rank": i,
                "instance_id": int(c["instance_id"]),
                "category": str(c.get("category", "unknown")),
                "confidence_score": round(score, 4),
                "reasoning": "Heuristic fallback based on category priors and estimated top area.",
            }
        )
    return output


def _normalize_candidates(
    parsed: Optional[Dict[str, Any]],
    source_candidates: List[Dict[str, Any]],
    max_results: int,
) -> Tuple[List[Dict[str, Any]], str]:
    src_map = {int(c["instance_id"]): c for c in source_candidates}
    if parsed is None:
        return _heuristic_fallback(source_candidates, max_results), "model_output_not_json_use_heuristic"

    raw_list = parsed.get("receptacle_candidates", [])
    if not isinstance(raw_list, list):
        return _heuristic_fallback(source_candidates, max_results), "missing_receptacle_candidates_use_heuristic"

    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            ins_id = int(item.get("instance_id"))
        except Exception:
            continue
        if ins_id in seen or ins_id not in src_map:
            continue
        seen.add(ins_id)
        src = src_map[ins_id]
        conf = max(0.0, min(1.0, _safe_float(item.get("confidence_score"), 0.5)))
        reason = str(item.get("reasoning", "")).strip() or "Likely has usable top surface for small objects."
        cleaned.append(
            {
                "rank": len(cleaned) + 1,
                "instance_id": ins_id,
                "category": src.get("category", "unknown"),
                "confidence_score": round(conf, 4),
                "reasoning": reason,
            }
        )
        if len(cleaned) >= max_results:
            break

    if not cleaned:
        return _heuristic_fallback(source_candidates, max_results), "empty_or_invalid_model_output_use_heuristic"

    cleaned.sort(key=lambda x: float(x.get("confidence_score", 0.0)), reverse=True)
    for i, row in enumerate(cleaned, start=1):
        row["rank"] = i
    return cleaned, str(parsed.get("overall_notes", "")).strip()


def query_receptacles_for_room(
    client: OpenAI,
    model: str,
    scene_name: str,
    room_id: int,
    candidates: List[Dict[str, Any]],
    max_results: int,
    max_tokens: int,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    candidates_json = json.dumps(candidates, ensure_ascii=False, indent=2)
    prompt = USER_PROMPT_TEMPLATE.format(
        scene_name=scene_name,
        room_id=room_id,
        room_object_count=len(candidates),
        max_results=max_results,
        candidates_json=candidates_json,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    raw_output = response.choices[0].message.content or ""
    cleaned_output = _clean_model_output(raw_output)
    parsed = _extract_json_block(cleaned_output)
    return raw_output, cleaned_output, parsed


def _build_surface_from_aabb(instance: Dict[str, Any], target_points: int) -> np.ndarray:
    aabb = instance.get("aabb") or {}
    min_pt = aabb.get("min", [0.0, 0.0, 0.0])
    max_pt = aabb.get("max", [0.0, 0.0, 0.0])
    if not isinstance(min_pt, list) or len(min_pt) < 3:
        min_pt = [0.0, 0.0, 0.0]
    if not isinstance(max_pt, list) or len(max_pt) < 3:
        max_pt = [0.0, 0.0, 0.0]
    rng = np.random.default_rng(42)
    n = max(1, int(target_points))
    xs = rng.uniform(float(min_pt[0]), float(max_pt[0]), size=n)
    ys = np.full((n,), float(max_pt[1]), dtype=np.float32)
    zs = rng.uniform(float(min_pt[2]), float(max_pt[2]), size=n)
    pts = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    return np.round(pts, 4)


def _estimate_surface_normal(points: np.ndarray) -> List[float]:
    if len(points) < 3:
        return [0.0, 1.0, 0.0]
    centered = points - points.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1].astype(np.float32)
        if n[1] < 0:
            n = -n
        norm = float(np.linalg.norm(n))
        if norm < 1e-8 or abs(float(n[1])) < 0.5:
            return [0.0, 1.0, 0.0]
        n = n / norm
        return [round(float(n[0]), 4), round(float(n[1]), 4), round(float(n[2]), 4)]
    except Exception:
        return [0.0, 1.0, 0.0]


def _extract_top_surface(
    raw_points: List[List[float]],
    instance: Dict[str, Any],
    target_points: int,
    min_points: int,
) -> Dict[str, Any]:
    points = np.asarray(raw_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
        points = _build_surface_from_aabb(instance, target_points)
        source = "aabb_top_fallback"
    else:
        points = points[:, :3]
        y = points[:, 1]
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        y_range = max(y_max - y_min, 1e-6)
        band = max(0.02, y_range * 0.08)
        top = points[y >= (y_max - band)]
        if len(top) < max(8, int(min_points)):
            wider_band = max(0.03, y_range * 0.18)
            top = points[y >= (y_max - wider_band)]
        if len(top) < max(8, int(min_points)):
            top = _build_surface_from_aabb(instance, max(target_points, min_points))
            source = "aabb_top_fallback"
        else:
            top = np.round(top, 4)
            source = "point_cloud_top_band"
        points = np.asarray(top, dtype=np.float32)

    if len(points) > max(1, int(target_points)):
        rng = np.random.default_rng(42)
        keep = rng.choice(len(points), size=int(target_points), replace=False)
        points = points[keep]

    centroid = np.round(points.mean(axis=0), 4).tolist() if len(points) else [0.0, 0.0, 0.0]
    bounds_min = np.round(points.min(axis=0), 4).tolist() if len(points) else [0.0, 0.0, 0.0]
    bounds_max = np.round(points.max(axis=0), 4).tolist() if len(points) else [0.0, 0.0, 0.0]
    return {
        "source": source,
        "point_count": int(len(points)),
        "centroid": centroid,
        "bounds": {"min": bounds_min, "max": bounds_max},
        "plane_height": round(float(bounds_max[1]), 4) if len(points) else 0.0,
        "normal": _estimate_surface_normal(points) if len(points) else [0.0, 1.0, 0.0],
        "points": np.round(points, 4).tolist(),
    }


def _resolve_room_ids(
    scene_info: Dict[str, Any],
    explicit_room_ids: Optional[List[int]],
    include_room_minus_one: bool,
) -> List[int]:
    if explicit_room_ids:
        out = sorted({int(x) for x in explicit_room_ids})
        if not include_room_minus_one:
            out = [rid for rid in out if rid != -1]
        return out

    room_ids = []
    for room in scene_info.get("rooms", []) or []:
        try:
            room_ids.append(int(room.get("region_id")))
        except Exception:
            continue
    room_ids = sorted(set(room_ids))
    if not include_room_minus_one:
        room_ids = [rid for rid in room_ids if rid != -1]
    return room_ids


def _validate_ssh_args(args: argparse.Namespace) -> bool:
    if args.ssh_host and args.ssh_user and (args.ssh_password or args.ssh_key):
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query receptacle instances for all rooms in a scene and extract top-surface point clouds."
    )
    parser.add_argument("--scene", required=True, help="Scene name, e.g. 00824-Dd4bFSTQ8gi")
    parser.add_argument("--room-id", type=int, action="append", default=None, help="Optional room region_id (repeatable)")
    parser.add_argument("--include-room-minus-one", action="store_true", help="Include region_id=-1 room")
    parser.add_argument("--scene-info-path", type=str, default=None, help="Optional explicit scene_info JSON path")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="HM3D data root")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")

    parser.add_argument("--max-results", type=int, default=10, help="Max receptacle instances per room")
    parser.add_argument("--instance-pointcloud-points", type=int, default=2048, help="Point count for per-instance extraction")
    parser.add_argument("--surface-points-per-instance", type=int, default=256, help="Saved top-surface points per instance")
    parser.add_argument("--surface-min-points", type=int, default=48, help="Minimum points required for a valid top surface")
    parser.add_argument("--debug-mesh-crop", action="store_true", help="Pass-through debug flag to point cloud extraction")
    parser.add_argument("--disable-llm", action="store_true", help="Use heuristic-only mode")

    parser.add_argument("--ssh-host", default=None, help="SSH server host")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH server port")
    parser.add_argument("--ssh-user", default=None, help="SSH username")
    parser.add_argument("--ssh-password", default=None, help="SSH password")
    parser.add_argument("--ssh-key", default=None, help="SSH private key path")
    parser.add_argument("--vllm-host", default="127.0.0.1", help="vLLM host in remote namespace")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM OpenAI-compatible API port")
    parser.add_argument("--local-port", type=int, default=0, help="Local forwarded port, 0 = auto")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking", help="LLM model name")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max response tokens")
    parser.add_argument("--timeout", type=int, default=3600, help="API timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    scene_info_path = Path(args.scene_info_path) if args.scene_info_path else None

    try:
        scene_info = _load_scene_info(args.scene, data_dir, scene_info_path=scene_info_path)
    except Exception as exc:
        print(f"[Error] Failed to load scene_info: {exc}", file=sys.stderr)
        return 1

    room_ids = _resolve_room_ids(scene_info, args.room_id, args.include_room_minus_one)
    if not room_ids:
        print("[Error] No room ids to process.", file=sys.stderr)
        return 1

    client: Optional[OpenAI] = None
    tunnel: Optional[SSHTunnel] = None
    use_llm = (not args.disable_llm) and _validate_ssh_args(args) and (OpenAI is not None)
    if not args.disable_llm and not use_llm:
        if OpenAI is None:
            print("[Warning] openai package not found, auto switch to heuristic-only mode.", file=sys.stderr)
        else:
            print("[Warning] SSH args incomplete, auto switch to heuristic-only mode.", file=sys.stderr)

    if use_llm:
        tunnel = SSHTunnel(
            ssh_host=str(args.ssh_host),
            ssh_port=int(args.ssh_port),
            ssh_user=str(args.ssh_user),
            ssh_password=args.ssh_password,
            ssh_key=args.ssh_key,
            remote_host=args.vllm_host,
            remote_port=args.vllm_port,
            local_port=args.local_port,
        )
        if not tunnel.start():
            print("[Warning] Failed to start tunnel, fallback to heuristic-only mode.", file=sys.stderr)
            use_llm = False
            tunnel = None
        else:
            client = OpenAI(api_key="EMPTY", base_url=tunnel.base_url, timeout=args.timeout)

    scene_results: List[Dict[str, Any]] = []
    processed_rooms = 0
    for room_id in room_ids:
        print(f"[Info] Processing room_id={room_id}")
        try:
            room_report = extract_room_instances(
                scene_name=args.scene,
                room_id=int(room_id),
                data_dir=data_dir,
                scene_info_path=scene_info_path,
                instance_id=None,
            )
        except Exception as exc:
            print(f"[Warning] Failed room {room_id}: {exc}", file=sys.stderr)
            continue

        instances = room_report.get("instances", []) if isinstance(room_report, dict) else []
        if not isinstance(instances, list) or not instances:
            scene_results.append(
                {
                    "room_id": int(room_id),
                    "room": room_report.get("room", {}) if isinstance(room_report, dict) else {},
                    "room_object_count": 0,
                    "receptacle_instances": [],
                    "overall_notes": "no_instances",
                    "debug": {},
                }
            )
            processed_rooms += 1
            continue

        source_candidates = _build_candidates(instances)
        raw_output = ""
        cleaned_output = ""
        parsed_output: Optional[Dict[str, Any]] = None

        if use_llm and client is not None:
            try:
                raw_output, cleaned_output, parsed_output = query_receptacles_for_room(
                    client=client,
                    model=args.model,
                    scene_name=args.scene,
                    room_id=int(room_id),
                    candidates=source_candidates,
                    max_results=max(1, int(args.max_results)),
                    max_tokens=int(args.max_tokens),
                )
            except Exception as exc:
                print(f"[Warning] LLM query failed in room {room_id}, fallback heuristic: {exc}", file=sys.stderr)

        final_candidates, notes = _normalize_candidates(
            parsed=parsed_output,
            source_candidates=source_candidates,
            max_results=max(1, int(args.max_results)),
        )

        instance_map = {}
        for ins in instances:
            try:
                instance_map[int(ins.get("id", -1))] = ins
            except Exception:
                continue

        receptacle_entries: List[Dict[str, Any]] = []
        for row in final_candidates:
            instance_id = int(row["instance_id"])
            ins_hint = instance_map.get(instance_id)
            if ins_hint is None:
                continue
            point_cloud_report = {}
            try:
                point_cloud_report = get_instance_point_cloud(
                    scene_name=args.scene,
                    instance_id=instance_id,
                    data_dir=data_dir,
                    num_points=max(int(args.instance_pointcloud_points), int(args.surface_points_per_instance)),
                    instance_hint=ins_hint,
                    debug_mesh_crop=args.debug_mesh_crop,
                )
            except Exception as exc:
                print(f"[Warning] Point cloud extraction failed instance={instance_id}: {exc}", file=sys.stderr)

            raw_points = []
            if isinstance(point_cloud_report, dict):
                pc = point_cloud_report.get("point_cloud", {})
                if isinstance(pc, dict):
                    raw_points = pc.get("points", []) or []

            top_surface = _extract_top_surface(
                raw_points=raw_points,
                instance=ins_hint,
                target_points=max(1, int(args.surface_points_per_instance)),
                min_points=max(4, int(args.surface_min_points)),
            )
            if int(top_surface.get("point_count", 0)) < max(1, int(args.surface_min_points)):
                continue

            receptacle_entries.append(
                {
                    "rank": int(row.get("rank", len(receptacle_entries) + 1)),
                    "instance_id": instance_id,
                    "category": str(row.get("category", ins_hint.get("category", "unknown"))),
                    "confidence_score": float(row.get("confidence_score", 0.5)),
                    "reasoning": str(row.get("reasoning", "")),
                    "instance": {
                        "id": instance_id,
                        "category": str(ins_hint.get("category", "unknown")),
                        "region_id": int(ins_hint.get("region_id", room_id)),
                        "aabb": ins_hint.get("aabb", {}),
                        "obb": ins_hint.get("obb", {}),
                    },
                    "point_cloud_generation": (
                        point_cloud_report.get("point_cloud_generation", {}) if isinstance(point_cloud_report, dict) else {}
                    ),
                    "top_surface": top_surface,
                }
            )

        receptacle_entries.sort(key=lambda x: float(x.get("confidence_score", 0.0)), reverse=True)
        for idx, item in enumerate(receptacle_entries, start=1):
            item["rank"] = idx

        scene_results.append(
            {
                "room_id": int(room_id),
                "room": room_report.get("room", {}),
                "room_object_count": int(len(instances)),
                "receptacle_instance_count": int(len(receptacle_entries)),
                "receptacle_instances": receptacle_entries,
                "overall_notes": notes,
                "debug": {
                    "raw_output": raw_output,
                    "cleaned_output": cleaned_output,
                    "candidate_pool_size": len(source_candidates),
                },
            }
        )
        processed_rooms += 1

    if tunnel is not None:
        tunnel.close()

    total_receptacles = sum(int(r.get("receptacle_instance_count", 0)) for r in scene_results)
    payload = {
        "scene_name": args.scene,
        "query_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query_model": args.model if use_llm else "heuristic_only",
        "surface_points_per_instance": int(args.surface_points_per_instance),
        "surface_min_points": int(args.surface_min_points),
        "rooms_processed": processed_rooms,
        "rooms_requested": len(room_ids),
        "total_receptacle_instances": total_receptacles,
        "rooms": scene_results,
    }

    if args.output:
        output_path = Path(args.output)
    else:
        out_dir = DEFAULT_OUTPUT_DIR / args.scene
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "all_rooms" if args.room_id is None else "selected_rooms"
        output_path = out_dir / f"{args.scene}_receptacle_surfaces_{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved result to: {output_path}")
    print(f"[OK] Rooms processed: {processed_rooms}/{len(room_ids)}")
    print(f"[OK] Total receptacle instances: {total_receptacles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
