#!/usr/bin/env python3
from __future__ import annotations

"""
Room-constrained object-to-instance assignment (File1) and auto placement trigger.

Overview
--------
This script bridges object sampling and final placement:
1) Load sampled objects (either from an existing layout json or on-the-fly sampling).
2) For each object, collect receptacle instance candidates from the same room only.
3) Ask LLM (optionally with object image) to choose one target instance.
4) Validate model output strictly against in-room candidate ids.
5) Save assignment plan json.
6) Call `place_objects_on_instances.py` (File2) to execute physics-aware placement.

Inputs
------
- `--surfaces-json`: output of `query_room_receptacle_objects.py`
- sampled objects: either
  - `--object-layout <layout.json>`, or
  - generated with `sample_object_positions(...)`

Outputs
-------
- assignment plan json (object -> target_instance_id)
- optional final layout json (if `--skip-placement` is not set)

Execution Guide
---------------
1) Standard pipeline (LLM assignment + auto placement):
   python assign_objects_to_receptacle_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --surfaces-json results/receptacle_queries/00824-Dd4bFSTQ8gi/00824-Dd4bFSTQ8gi_receptacle_surfaces_all_rooms.json \
     --ssh-host 7.216.187.6 --ssh-port 31822 --ssh-user root --ssh-password 666666

2) Heuristic-only assignment (no LLM), still run placement:
   python assign_objects_to_receptacle_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --surfaces-json results/receptacle_queries/00824-Dd4bFSTQ8gi/00824-Dd4bFSTQ8gi_receptacle_surfaces_all_rooms.json \
     --disable-llm

3) Only generate assignment plan (do not place):
   python assign_objects_to_receptacle_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --surfaces-json results/receptacle_queries/00824-Dd4bFSTQ8gi/00824-Dd4bFSTQ8gi_receptacle_surfaces_all_rooms.json \
     --skip-placement --disable-llm

4) Use pre-sampled object layout:
   python assign_objects_to_receptacle_instances.py \
     --scene 00824-Dd4bFSTQ8gi \
     --surfaces-json results/receptacle_queries/00824-Dd4bFSTQ8gi/00824-Dd4bFSTQ8gi_receptacle_surfaces_all_rooms.json \
     --object-layout results/layouts/00824-Dd4bFSTQ8gi/some_layout.json
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from extract_room_instances import DEFAULT_DATA_DIR
from place_objects_on_instances import place_objects_on_instances
from sample_and_place_objects import (
    DEFAULT_IMAGES_DIR,
    DEFAULT_PROBABILITIES_DIR,
    DEFAULT_ROOMS_INFO_DIR,
    sample_object_positions,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


SYSTEM_PROMPT = (
    "You are an indoor object-placement assistant. "
    "Choose the best target instance in the same room for placing the given object."
)

USER_PROMPT_TEMPLATE = """
Task:
Given one object (name + optional image) and candidate support instances in the SAME room, choose the best target instance.

Hard constraints:
1) target_instance_id must be selected only from provided candidate instance_id list.
2) Return ONLY JSON, no markdown, no extra text.
3) confidence_score must be float in [0, 1].
4) backup_instance_ids must be from candidate list and must not include target_instance_id.

Output JSON schema:
{{
  "target_instance_id": 123,
  "confidence_score": 0.91,
  "reasoning": "short sentence",
  "backup_instance_ids": [456, 789]
}}

Scene: {scene_name}
Room ID: {room_id}
Object:
{object_json}

Candidate instances (JSON):
{candidates_json}
""".strip()


def _pick_free_local_port() -> int:
    """Ask OS for a free local TCP port for SSH forwarding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tunnel_ready(host: str, port: int, timeout_s: float = 10.0) -> bool:
    """Wait until local forwarded port accepts TCP connection."""
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _clean_model_output(text: Optional[str]) -> str:
    """Strip `<think>` traces and normalize line breaks before JSON parse."""
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
    """Parse full text json first, fallback to largest `{...}` block."""
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
    """Manage SSH tunnel lifecycle for remote model endpoint."""

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
        """Start SSH tunnel and block until local base_url is reachable."""
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
            print(f"[Error] SSH tunnel not ready within {timeout_s}s", file=sys.stderr)
            self.close()
            return False
        return True

    def close(self) -> None:
        """Stop SSH tunnel process."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _build_image_url(image_path: str) -> str:
    """Convert local image file to data URL for OpenAI-compatible image input."""
    p = Path(image_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image path not found: {p}")
    mime_type, _ = mimetypes.guess_type(str(p))
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort cast to float with explicit fallback."""
    try:
        return float(value)
    except Exception:
        return default


def _find_image_for_object(images_dir: str, model_id: str, name: str) -> Optional[str]:
    """
    Resolve object image by matching filename stem to model_id/name aliases.

    This allows visual assignment prompts without requiring strict naming format.
    """
    root = Path(images_dir)
    if not root.is_dir():
        return None
    candidates = []
    keys = []
    if model_id:
        keys.append(model_id)
        if model_id.endswith("_4k"):
            keys.append(model_id[:-3])
    if name:
        keys.append(name)
        keys.append(name.lower().replace(" ", "_"))
    unique_keys = []
    seen = set()
    for key in keys:
        k = str(key).strip()
        if not k:
            continue
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    for ext in ("*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp"):
        candidates.extend(root.glob(ext))
    for image_path in sorted(candidates):
        stem = image_path.stem
        for key in unique_keys:
            if stem == key or stem.lower() == key.lower():
                return str(image_path.resolve())
    return None


def _build_surface_candidates_for_room(room_entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build compact per-room candidate list from top-surface extraction output."""
    out = []
    for item in room_entry.get("receptacle_instances", []) or []:
        top = item.get("top_surface", {}) if isinstance(item, dict) else {}
        bounds = top.get("bounds", {}) if isinstance(top, dict) else {}
        bmin = bounds.get("min", [0.0, 0.0, 0.0]) if isinstance(bounds, dict) else [0.0, 0.0, 0.0]
        bmax = bounds.get("max", [0.0, 0.0, 0.0]) if isinstance(bounds, dict) else [0.0, 0.0, 0.0]
        area = max(0.0, (_safe_float(bmax[0]) - _safe_float(bmin[0])) * (_safe_float(bmax[2]) - _safe_float(bmin[2])))
        out.append(
            {
                "instance_id": int(item.get("instance_id", -1)),
                "category": str(item.get("category", "unknown")),
                "receptacle_confidence": float(item.get("confidence_score", 0.5)),
                "surface_point_count": int(top.get("point_count", 0)),
                "surface_height": float(top.get("plane_height", 0.0)),
                "surface_area_est": round(area, 4),
            }
        )
    out = [x for x in out if int(x.get("instance_id", -1)) >= 0]
    return out


def _heuristic_choose_instance(candidates: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    """
    Heuristic selector when LLM is unavailable.

    Uses candidate confidence + surface area + light category priors.
    """
    model_key = (model_id or "").lower()
    scored = []
    for c in candidates:
        score = float(c.get("receptacle_confidence", 0.5))
        score += min(0.2, float(c.get("surface_area_est", 0.0)) * 0.05)
        category = str(c.get("category", "")).lower()
        if "table" in category or "desk" in category:
            score += 0.1
        if "bed" in category and ("lamp" in model_key or "book" in model_key):
            score += 0.05
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    target = scored[0][1] if scored else {"instance_id": -1}
    backups = [int(x[1].get("instance_id", -1)) for x in scored[1:4] if int(x[1].get("instance_id", -1)) >= 0]
    return {
        "target_instance_id": int(target.get("instance_id", -1)),
        "confidence_score": round(float(scored[0][0]) if scored else 0.0, 4),
        "reasoning": "Heuristic fallback from receptacle score and surface area.",
        "backup_instance_ids": backups,
    }


def _normalize_assignment_response(parsed: Optional[Dict[str, Any]], candidates: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    """Validate assignment response against candidate whitelist and normalize fields."""
    candidate_ids = {int(c["instance_id"]) for c in candidates if int(c.get("instance_id", -1)) >= 0}
    if parsed is None:
        return _heuristic_choose_instance(candidates, model_id)

    try:
        target_id = int(parsed.get("target_instance_id"))
    except Exception:
        target_id = -1
    if target_id not in candidate_ids:
        return _heuristic_choose_instance(candidates, model_id)

    confidence = max(0.0, min(1.0, _safe_float(parsed.get("confidence_score"), 0.5)))
    reasoning = str(parsed.get("reasoning", "")).strip() or "Likely best support in-room."
    backup_raw = parsed.get("backup_instance_ids", [])
    backups = []
    if isinstance(backup_raw, list):
        for item in backup_raw:
            try:
                iid = int(item)
            except Exception:
                continue
            if iid == target_id or iid not in candidate_ids or iid in backups:
                continue
            backups.append(iid)
            if len(backups) >= 3:
                break
    return {
        "target_instance_id": target_id,
        "confidence_score": round(confidence, 4),
        "reasoning": reasoning,
        "backup_instance_ids": backups,
    }


def _validate_ssh_args(args: argparse.Namespace) -> bool:
    """Check whether SSH credentials are sufficient for remote LLM mode."""
    return bool(args.ssh_host and args.ssh_user and (args.ssh_password or args.ssh_key))


def _load_or_sample_objects(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """
    Load sampled objects from layout json, or sample on the fly.

    Optional filter:
    `--only-manual-fix` keeps only objects previously flagged for manual repair.
    """
    if args.object_layout:
        payload = json.loads(Path(args.object_layout).read_text(encoding="utf-8"))
    else:
        payload = sample_object_positions(
            scene_name=args.scene,
            images_dir=args.images_dir,
            mode=args.sampling_mode,
            rooms_info_dir=args.rooms_info_dir,
            probabilities_dir=args.probabilities_dir,
        )
        if payload is None:
            return []
    objects = payload.get("objects", []) if isinstance(payload, dict) else []
    if not isinstance(objects, list):
        return []
    if args.only_manual_fix:
        filtered = [o for o in objects if str(o.get("edit_status", "")) == "needs_manual_fix"]
        return filtered
    return objects


def _query_assignment_for_object(
    client: OpenAI,
    model: str,
    scene_name: str,
    room_id: int,
    object_entry: Dict[str, Any],
    image_path: Optional[str],
    candidates: List[Dict[str, Any]],
    max_tokens: int,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Query model for one object's target instance within one room.

    If image exists, send multimodal input (image + text constraints).
    """
    obj_payload = {
        "object_id": object_entry.get("id"),
        "model_id": object_entry.get("model_id"),
        "name": object_entry.get("name"),
        "confidence": object_entry.get("confidence", 0.5),
        "image_path": image_path or "",
    }
    prompt = USER_PROMPT_TEMPLATE.format(
        scene_name=scene_name,
        room_id=room_id,
        object_json=json.dumps(obj_payload, ensure_ascii=False, indent=2),
        candidates_json=json.dumps(candidates, ensure_ascii=False, indent=2),
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_path:
        try:
            image_url = _build_image_url(image_path)
            content.insert(0, {"type": "image_url", "image_url": {"url": image_url}})
        except Exception:
            pass

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
    )
    raw_text = response.choices[0].message.content or ""
    cleaned = _clean_model_output(raw_text)
    parsed = _extract_json_block(cleaned)
    return raw_text, cleaned, parsed


def parse_args() -> argparse.Namespace:
    """Define CLI for assignment generation and optional placement execution."""
    parser = argparse.ArgumentParser(description="Assign sampled objects to in-room receptacle instances, then place automatically.")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--surfaces-json", required=True, help="Output json from query_room_receptacle_objects.py")
    parser.add_argument("--object-layout", type=str, default=None, help="Optional sampled layout json with objects + sampled_region_id")
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR, help="Object images directory")
    parser.add_argument("--sampling-mode", choices=["load", "generate"], default="load", help="Sampling mode when --object-layout is absent")
    parser.add_argument("--rooms-info-dir", type=str, default=DEFAULT_ROOMS_INFO_DIR, help="Room query results dir for sampling")
    parser.add_argument("--probabilities-dir", type=str, default=DEFAULT_PROBABILITIES_DIR, help="Probability files dir for sampling")
    parser.add_argument("--only-manual-fix", action="store_true", help="Only process objects with edit_status=needs_manual_fix")

    parser.add_argument("--output-plan", type=str, default=None, help="Output assignment plan JSON")
    parser.add_argument("--output-layout", type=str, default=None, help="Output final layout JSON")
    parser.add_argument("--skip-placement", action="store_true", help="Only output assignment plan and skip calling file2")

    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="HM3D root dir")
    parser.add_argument("--objects-dir", type=str, default="./objects", help="Object template configs directory for file2")
    parser.add_argument("--min-distance", type=float, default=0.25, help="Minimum object pair distance in placement")
    parser.add_argument("--spawn-height", type=float, default=0.3, help="Spawn height above target surface in placement")
    parser.add_argument("--max-trials-per-object", type=int, default=30, help="Max surface points tried per object in placement")
    parser.add_argument("--settle-steps", type=int, default=45, help="Physics settle steps in placement")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--disable-llm", action="store_true", help="Use heuristic-only assignment")
    parser.add_argument("--ssh-host", default=None, help="SSH server host")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH server port")
    parser.add_argument("--ssh-user", default=None, help="SSH username")
    parser.add_argument("--ssh-password", default=None, help="SSH password")
    parser.add_argument("--ssh-key", default=None, help="SSH private key")
    parser.add_argument("--vllm-host", default="127.0.0.1", help="vLLM host")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM OpenAI API port")
    parser.add_argument("--local-port", type=int, default=0, help="Local forwarded port")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking", help="LLM model")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max response tokens")
    parser.add_argument("--timeout", type=int, default=3600, help="API timeout")
    return parser.parse_args()


def main() -> int:
    """
    Entrypoint for full assignment workflow.

    Stages:
    1) load surfaces + sampled objects
    2) optional LLM tunnel setup
    3) per-object in-room assignment
    4) write assignment plan
    5) optionally call file2 for final placement/layout
    """
    args = parse_args()
    np.random.seed(int(args.seed))

    try:
        surfaces_payload = json.loads(Path(args.surfaces_json).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Error] Failed to load surfaces json: {exc}", file=sys.stderr)
        return 1

    objects = _load_or_sample_objects(args)
    if not objects:
        print("[Error] No objects available for assignment.", file=sys.stderr)
        return 1

    room_map: Dict[int, Dict[str, Any]] = {}
    for room in surfaces_payload.get("rooms", []) or []:
        try:
            room_map[int(room.get("room_id"))] = room
        except Exception:
            continue

    use_llm = (not args.disable_llm) and _validate_ssh_args(args) and (OpenAI is not None)
    if not args.disable_llm and not use_llm:
        if OpenAI is None:
            print("[Warning] openai package not found, using heuristic-only assignment.", file=sys.stderr)
        else:
            print("[Warning] SSH args incomplete, using heuristic-only assignment.", file=sys.stderr)

    tunnel: Optional[SSHTunnel] = None
    client: Optional[OpenAI] = None
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
            print("[Warning] Tunnel failed, fallback to heuristic-only assignment.", file=sys.stderr)
            use_llm = False
            tunnel = None
        else:
            client = OpenAI(api_key="EMPTY", base_url=tunnel.base_url, timeout=args.timeout)

    assignments: List[Dict[str, Any]] = []
    llm_debug: List[Dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        room_id = int(obj.get("sampled_region_id", -1))
        room_entry = room_map.get(room_id)
        if room_entry is None:
            llm_debug.append({"object_id": obj.get("id", idx), "status": "missing_room_surface"})
            continue

        candidates = _build_surface_candidates_for_room(room_entry)
        if not candidates:
            llm_debug.append({"object_id": obj.get("id", idx), "status": "empty_candidates"})
            continue

        model_id = str(obj.get("model_id", ""))
        name = str(obj.get("name", model_id or f"obj_{idx}"))
        image_path = _find_image_for_object(args.images_dir, model_id=model_id, name=name)

        raw_output = ""
        cleaned_output = ""
        parsed_output: Optional[Dict[str, Any]] = None
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
            except Exception as exc:
                llm_debug.append(
                    {"object_id": obj.get("id", idx), "status": "llm_error", "error": str(exc)}
                )

        decision = _normalize_assignment_response(parsed_output, candidates, model_id=model_id)
        if int(decision.get("target_instance_id", -1)) < 0:
            llm_debug.append({"object_id": obj.get("id", idx), "status": "invalid_decision"})
            continue

        assignments.append(
            {
                "object_id": int(obj.get("id", idx)),
                "model_id": model_id,
                "name": name,
                "image_path": image_path,
                "sampled_region_id": room_id,
                "target_room_id": room_id,
                "target_instance_id": int(decision["target_instance_id"]),
                "backup_instance_ids": decision.get("backup_instance_ids", []),
                "confidence_score": float(decision.get("confidence_score", 0.5)),
                "reasoning": str(decision.get("reasoning", "")),
                "source": "llm_assignment" if use_llm else "heuristic_assignment",
            }
        )
        llm_debug.append(
            {
                "object_id": obj.get("id", idx),
                "room_id": room_id,
                "raw_output": raw_output,
                "cleaned_output": cleaned_output,
                "candidate_count": len(candidates),
            }
        )

    if tunnel is not None:
        tunnel.close()

    plan_payload = {
        "scene_name": args.scene,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model if use_llm else "heuristic_only",
        "input_object_count": len(objects),
        "assignment_count": len(assignments),
        "assignments": assignments,
        "debug": llm_debug,
    }

    if args.output_plan:
        plan_path = Path(args.output_plan)
    else:
        out_dir = Path("./results/object_instance_assignments") / args.scene
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / f"{args.scene}_object_instance_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Assignment plan saved: {plan_path}")
    print(f"[OK] Assigned objects: {len(assignments)}/{len(objects)}")

    if args.skip_placement:
        return 0

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
        seed=int(args.seed),
    )

    if args.output_layout:
        layout_path = Path(args.output_layout)
    else:
        out_dir = Path("./results/layouts") / args.scene
        out_dir.mkdir(parents=True, exist_ok=True)
        layout_path = out_dir / f"{args.scene}_assigned_instance_layout.json"
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(json.dumps(layout_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Final layout saved: {layout_path}")
    stats = layout_payload.get("auto_placement_stats", {})
    print(
        "[OK] Placement stats: placed={}/{} failed={}".format(
            int(stats.get("placed_count", 0)),
            int(stats.get("total_objects", 0)),
            int(stats.get("failed_count", 0)),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
