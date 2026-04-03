#!/usr/bin/env python3
from __future__ import annotations

"""
批量查询物体在各场景中最有可能出现的房间位置。

逻辑流程：
1. 读取场景导出的scene_info JSON（或实时导出）
2. 遍历 objects_images/ 目录中的所有图片
3. 对每个(场景, 物体图片)对，通过SSH隧道连接远程Qwen3-VL
4. 询问："该物体最有可能出现在房间的哪些地方？前5个房间+3D中心"
5. 解析回复，提取房间推荐列表
6. 生成JSON：场景信息 + 查询内容 + Qwen原始/清洗后回复 + 前5房间推荐 + 元数据

用法：
  python query_rooms_for_objects.py \
    --ssh-host 7.216.187.6 --ssh-port 31822 --ssh-user root --ssh-password 666666 \
    --vllm-host 127.0.0.1 --vllm-port 8000 \
    --images-dir ./objects_images \
    --scenes all \
    --output-dir ./results/scene_info/

  # 或单个场景

  python query_rooms_for_objects.py \
    --ssh-host 7.216.187.6 --ssh-port 31822 --ssh-user root --ssh-password 666666 \
    --vllm-host 127.0.0.1 --vllm-port 8000 \
    --images-dir ./objects_images \
    --scene 00808-y9hTuugGdiq \
    --output-dir ./results/scene_info/
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from hm3d_paths import list_available_scenes, resolve_scene_paths

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not found. Install with: pip install openai")
    sys.exit(1)

try:
    import habitat_sim
except ImportError:
    print("Warning: habitat_sim not found; will rely on pre-exported scene_info files")
    habitat_sim = None


# ===== 常量 =====
DEFAULT_OUTPUT_DIR = "./results/scene_info"
DEFAULT_IMAGES_DIR = "./objects_images"
AVAILABLE_SCENES = list_available_scenes(require_semantic=True)

QWEN_SYSTEM_TEMPLATE = (
    "你是室内布局推荐助手。你必须基于用户给的候选房间列表作答，"
    "禁止编造不存在的region_id和坐标。"
)

QWEN_QUERY_TEMPLATE = (
    "任务：根据图片中的物体，选择该物体最可能出现的房间区域。\n"
    "\n"
    "约束（必须遵守）：\n"
    "1) 只能从下方候选房间中选择 region_id。\n"
    "2) room_center 必须与候选房间中的 center 完全一致，不得改写。\n"
    "3) 输出房间数量必须在 2 到 5 之间（可以少于5，但至少2个）。\n"
    "4) confidence_score 必须是 0 到 1 的浮点数，并按从高到低排序。\n"
    "5) reasoning 必须是一句话，简短且具体。\n"
    "6) 只输出 JSON，不要输出解释、推理过程、Markdown、前后缀文本。\n"
    "\n"
    "请按如下 JSON Schema 输出（字段名必须一致）：\n"
    "{{\n"
    "  \"recommended_rooms\": [\n"
    "    {{\n"
    "      \"rank\": 1,\n"
    "      \"region_id\": 0,\n"
    "      \"room_center\": [0.0, 0.0, 0.0],\n"
    "      \"confidence_score\": 0.9,\n"
    "      \"reasoning\": \"一句话理由\"\n"
    "    }}\n"
    "  ]\n"
    "}}\n"
    "\n"
    "候选房间列表（仅可从中选5个）：\n"
    "{room_candidates_json}\n"
)


# ===== 从 qwen3_vl_connect 复用的工具函数 =====

def _pick_free_local_port() -> int:
    """Pick a free local TCP port by asking the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_tunnel_ready(host: str, port: int, timeout_s: float = 10.0) -> bool:
    """Wait until local forwarded port accepts TCP connections."""
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _build_image_url(image_input: str) -> str:
    """Return either original HTTP(S) URL or data URL for local image file."""
    if image_input.startswith(("http://", "https://")):
        return image_input

    image_path = Path(image_input).expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _clean_model_output(text: str | None) -> str:
    """Remove common thinking traces and return user-facing answer text."""
    if not text:
        return ""

    cleaned = text
    # Remove complete <think>...</think> blocks.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    # If reasoning leaked before a stray closing tag, keep content after the last closing tag.
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        parts = re.split(r"</think>", cleaned, flags=re.IGNORECASE)
        cleaned = parts[-1]
    # Remove stray open/close think tags.
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    # Normalize extra blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from model output."""
    if not text:
        return None

    # Fast path: whole text is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: find first {...} block and parse
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


# ===== SSH隧道管理 =====

class SSHTunnel:
    """Manages SSH tunnel lifecycle for Qwen remote connection."""
    
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
        
        self.proc = None
        self.base_url = f"http://127.0.0.1:{self.local_port}/v1"
    
    def start(self, timeout_s: float = 30.0) -> bool:
        """Start SSH tunnel and wait for readiness."""
        tunnel_cmd = [
            "ssh",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(self.ssh_port),
            "-N",
            "-L", f"127.0.0.1:{self.local_port}:{self.remote_host}:{self.remote_port}",
            f"{self.ssh_user}@{self.ssh_host}",
        ]
        
        cmd_for_run = tunnel_cmd
        proc_env = os.environ.copy()
        
        if self.ssh_password:
            if shutil.which("sshpass") is None:
                print("[Error] sshpass not found. Install with: sudo apt-get install -y sshpass")
                return False
            tunnel_cmd[1:1] = [
                "-o", "PubkeyAuthentication=no",
                "-o", "PreferredAuthentications=password,keyboard-interactive",
                "-o", "NumberOfPasswordPrompts=1",
            ]
            proc_env["SSHPASS"] = self.ssh_password
            cmd_for_run = ["sshpass", "-e", *tunnel_cmd]
        elif self.ssh_key:
            ssh_key_path = os.path.expanduser(self.ssh_key)
            if not os.path.exists(ssh_key_path):
                print(f"[Error] SSH key not found: {ssh_key_path}")
                return False
            tunnel_cmd[1:1] = ["-i", ssh_key_path]
        else:
            print("[Error] Must provide either --ssh-password or --ssh-key")
            return False
        
        print(f"[Info] Opening SSH tunnel to {self.ssh_user}@{self.ssh_host}:{self.ssh_port}")
        try:
            self.proc = subprocess.Popen(
                cmd_for_run,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=proc_env,
            )
        except Exception as e:
            print(f"[Error] Failed to start SSH tunnel: {e}")
            return False
        
        # Check if process started successfully
        time.sleep(0.2)
        if self.proc.poll() is not None:
            print("[Error] SSH tunnel process exited immediately")
            return False
        
        # Wait for tunnel to be ready
        if not _wait_tunnel_ready("127.0.0.1", self.local_port, timeout_s):
            print(f"[Error] SSH tunnel did not become ready within {timeout_s}s")
            self.close()
            return False
        
        print(f"[Info] SSH tunnel ready at {self.base_url}")
        return True
    
    def close(self):
        """Close SSH tunnel gracefully."""
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            print("[Info] SSH tunnel closed")


# ===== Qwen 调用 =====

def query_qwen_for_rooms(
    client: OpenAI,
    image_path: str,
    scene_info: Dict[str, Any],
    model: str = "Qwen/Qwen3-VL-235B-A22B-Thinking",
    max_tokens: int = 2048,
) -> Tuple[str, str]:
    """
    Query Qwen3-VL about rooms where object should be placed.
    
    Returns:
        (raw_output, cleaned_output)
    """
    try:
        image_url = _build_image_url(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to build image URL: {e}")
    
    room_candidates = []
    for room in scene_info.get("rooms", []):
        rid = room.get("region_id")
        bbox = room.get("bounding_box", {})
        min_pt = bbox.get("min", [0.0, 0.0, 0.0])
        max_pt = bbox.get("max", [0.0, 0.0, 0.0])
        if rid is None or len(min_pt) < 3 or len(max_pt) < 3:
            continue
        center = [
            round((float(min_pt[0]) + float(max_pt[0])) / 2.0, 4),
            round((float(min_pt[1]) + float(max_pt[1])) / 2.0, 4),
            round((float(min_pt[2]) + float(max_pt[2])) / 2.0, 4),
        ]
        room_candidates.append(
            {
                "region_id": rid,
                "center": center,
                "object_count": room.get("object_count", 0),
                "top_categories": list(room.get("categories", {}).keys())[:5],
            }
        )

    room_candidates_json = json.dumps(room_candidates, ensure_ascii=False, indent=2)
    query_text = QWEN_QUERY_TEMPLATE.format(room_candidates_json=room_candidates_json)

    messages = [
        {"role": "system", "content": QWEN_SYSTEM_TEMPLATE},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": query_text},
            ],
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        raw_output = response.choices[0].message.content
        cleaned_output = _clean_model_output(raw_output)
        return raw_output, cleaned_output
    except Exception as e:
        raise RuntimeError(f"Qwen API call failed: {e}")


# ===== 房间推荐解析 =====

def parse_room_recommendations(
    cleaned_output: str,
    scene_info: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse Qwen's cleaned output to extract top-5 room recommendations.
    
    Expected format (from QWEN_QUERY_TEMPLATE):
      房间1: region_id=0, 中心=(-2.5, 1.2, 3.8), 置信度=0.95, 理由：...
      房间2: region_id=2, 中心=(-1.0, 0.9, 5.2), 置信度=0.80, 理由：...
      ...
    
    Returns:
        List of dicts with {rank, region_id, room_center, confidence_score, reasoning, ...}
    """
    recommendations = []

    def _room_center_from_room(room: Dict[str, Any]) -> List[float]:
        if "room_center" in room and isinstance(room["room_center"], list) and len(room["room_center"]) >= 3:
            return [round(float(room["room_center"][0]), 4), round(float(room["room_center"][1]), 4), round(float(room["room_center"][2]), 4)]
        bbox = room.get("bounding_box", {})
        min_pt = bbox.get("min", [0.0, 0.0, 0.0])
        max_pt = bbox.get("max", [0.0, 0.0, 0.0])
        return [
            round((float(min_pt[0]) + float(max_pt[0])) / 2.0, 4),
            round((float(min_pt[1]) + float(max_pt[1])) / 2.0, 4),
            round((float(min_pt[2]) + float(max_pt[2])) / 2.0, 4),
        ]

    def _normalize_and_fill(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup: List[Dict[str, Any]] = []
        seen = set()

        for rec in recs:
            rid = rec.get("region_id")
            if rid is None:
                continue
            try:
                rid = int(rid)
            except Exception:
                continue
            if rid in seen or rid not in room_map:
                continue
            seen.add(rid)

            confidence = float(rec.get("confidence_score", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(rec.get("reasoning", "")).strip() or "基于房间物体分布与空间语义的综合判断"

            room = room_map[rid]
            clean_rec = {
                "rank": len(dedup) + 1,
                "region_id": rid,
                "confidence_score": round(confidence, 4),
                "reasoning": reason,
                "room_center": _room_center_from_room(room),
            }
            if "bounding_box" in room:
                clean_rec["room_aabb"] = room["bounding_box"]
            dedup.append(clean_rec)

        if len(dedup) < 2:
            fallback_rooms = sorted(
                room_map.values(),
                key=lambda r: int(r.get("object_count", 0)),
                reverse=True,
            )
            for room in fallback_rooms:
                rid = int(room.get("region_id", -1))
                if rid < 0 or rid in seen:
                    continue
                seen.add(rid)
                confidence = 0.51 if len(dedup) == 0 else 0.5
                fallback = {
                    "rank": len(dedup) + 1,
                    "region_id": rid,
                    "confidence_score": confidence,
                    "reasoning": "模型输出候选不足，按场景房间信息回退补足",
                    "room_center": _room_center_from_room(room),
                }
                if "bounding_box" in room:
                    fallback["room_aabb"] = room["bounding_box"]
                dedup.append(fallback)
                if len(dedup) >= 2:
                    break

        dedup = sorted(dedup, key=lambda x: float(x.get("confidence_score", 0.0)), reverse=True)[:5]
        for i, rec in enumerate(dedup, start=1):
            rec["rank"] = i
        return dedup
    
    # Build map of region_id -> room_info from scene_info
    room_map = {}
    if "rooms" in scene_info:
        for room in scene_info["rooms"]:
            rid = room.get("region_id")
            if rid is not None:
                room_map[rid] = room
    
    # Try strict JSON parsing first
    parsed_json = _extract_json_block(cleaned_output)
    if parsed_json and isinstance(parsed_json.get("recommended_rooms"), list):
        for idx, item in enumerate(parsed_json["recommended_rooms"][:5], start=1):
            if not isinstance(item, dict):
                continue
            rid = item.get("region_id")
            if rid is None:
                continue
            record = {
                "rank": int(item.get("rank", idx)),
                "region_id": int(rid),
                "confidence_score": float(item.get("confidence_score", 0.5)),
                "reasoning": str(item.get("reasoning", "")).strip(),
            }
            # Always trust scene_info center for consistency
            if int(rid) in room_map and "bounding_box" in room_map[int(rid)]:
                bbox = room_map[int(rid)]["bounding_box"]
                min_pt = bbox.get("min", [0, 0, 0])
                max_pt = bbox.get("max", [0, 0, 0])
                record["room_center"] = [
                    round((min_pt[0] + max_pt[0]) / 2, 4),
                    round((min_pt[1] + max_pt[1]) / 2, 4),
                    round((min_pt[2] + max_pt[2]) / 2, 4),
                ]
                record["room_aabb"] = bbox
            recommendations.append(record)

        return _normalize_and_fill(recommendations)

    # Fallback: Parse each line looking for "房间N:" pattern
    lines = cleaned_output.split("\n")
    rank = 0
    for line in lines:
        line = line.strip()
        if not line or "房间" not in line:
            continue
        
        rank += 1
        if rank > 5:
            break
        
        record = {"rank": rank}
        
        # Extract region_id
        m = re.search(r"region_id[=：]+(\d+)", line)
        if m:
            rid = int(m.group(1))
            record["region_id"] = rid
            
            # Get room_center from scene_info
            if rid in room_map:
                room = room_map[rid]
                if "bounding_box" in room:
                    bbox = room["bounding_box"]
                    min_pt = bbox.get("min", [0, 0, 0])
                    max_pt = bbox.get("max", [0, 0, 0])
                    center = [
                        round((min_pt[0] + max_pt[0]) / 2, 4),
                        round((min_pt[1] + max_pt[1]) / 2, 4),
                        round((min_pt[2] + max_pt[2]) / 2, 4),
                    ]
                    record["room_center"] = center
                    record["room_aabb"] = bbox
        
        # Extract confidence score
        m = re.search(r"置信度[=：]+([0-9.]+)", line)
        if m:
            record["confidence_score"] = float(m.group(1))
        else:
            record["confidence_score"] = 0.5  # default
        
        # Extract reasoning (after 理由)
        m = re.search(r"理由[=：]+(.+?)(?:,|$)", line)
        if m:
            record["reasoning"] = m.group(1).strip()
        else:
            record["reasoning"] = line
        
        recommendations.append(record)
    
    return _normalize_and_fill(recommendations)


# ===== 主логика =====

def load_or_export_scene_info(scene_name: str, output_dir: str) -> Optional[Dict]:
    """
    Load scene_info from pre-exported JSON, or export on-the-fly if available.
    """
    # Try to load from file first
    scene_info_path = os.path.join(output_dir, f"{scene_name}", f"{scene_name}_scene_info.json")
    if os.path.isfile(scene_info_path):
        try:
            with open(scene_info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"  [Warning] Failed to load scene_info from {scene_info_path}: {e}")
    
    # Try to export on-the-fly if habitat_sim available
    if habitat_sim is None:
        print(f"  [Error] habitat_sim not available and scene_info not pre-exported")
        return None

    scene_paths = resolve_scene_paths(scene_name, require_semantic=True)
    if scene_paths is None:
        print(f"  [Error] Scene not found in merged hm3d splits or missing semantic.txt: {scene_name}")
        return None
    
    try:
        from export_scene_info import export_scene
        print(f"  [Info] Exporting scene_info on-the-fly...")
        # 导出到临时目录
        temp_export_dir = os.path.join(output_dir, "temp_export")
        result = export_scene(
            scene_name,
            str(scene_paths.split_root.parent),
            str(scene_paths.dataset_config),
            temp_export_dir,
        )
        return result
    except Exception as e:
        print(f"  [Error] Failed to export scene_info: {e}")
        return None


def process_scene(
    scene_name: str,
    images_dir: str,
    output_dir: str,
    tunnel: SSHTunnel,
    client: OpenAI,
    model: str,
) -> Dict[str, Any]:
    """
    Process a single scene: query Qwen for each object image.
    
    Returns:
        {success_count, fail_count, results}
    """
    print(f"\n>>> Processing scene: {scene_name}")
    
    # Ensure scene output directory exists
    scene_output_dir = os.path.join(output_dir, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # Load/export scene_info
    scene_info = load_or_export_scene_info(scene_name, output_dir)
    if not scene_info:
        print(f"  [Error] Cannot proceed without scene_info")
        return {"success_count": 0, "fail_count": 0, "results": []}
    
    # Scan image directory
    if not os.path.isdir(images_dir):
        print(f"  [Error] Images directory not found: {images_dir}")
        return {"success_count": 0, "fail_count": 0, "results": []}
    
    image_files = []
    for ext in ["*.webp", "*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(Path(images_dir).glob(ext))
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"  [Warning] No image files found in {images_dir}")
        return {"success_count": 0, "fail_count": 0, "results": []}
    
    success_count = 0
    fail_count = 0
    results = []
    
    for image_path in image_files:
        object_name = image_path.stem  # filename without extension
        print(f"  Processing image: {image_path.name} (object: {object_name})")
        
        try:
            # Query Qwen
            raw_output, cleaned_output = query_qwen_for_rooms(client, str(image_path), scene_info, model)
            
            # Parse recommendations
            recommendations = parse_room_recommendations(cleaned_output, scene_info)
            
            # Build output JSON
            result_json = {
                "scene_info": {
                    "scene_name": scene_name,
                    "image": image_path.name,
                    "object_name": object_name,
                    "image_path": str(image_path.resolve()),
                    "query_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model,
                },
                "query": "严格JSON模板，包含候选房间约束（见raw_output输入上下文）",
                "raw_output": raw_output,
                "cleaned_output": cleaned_output,
                "recommended_rooms": recommendations,
                "metadata": {
                    "total_objects_in_scene": scene_info.get("scene_info", {}).get(
                        "total_objects", len(scene_info.get("objects", []))
                    ),
                    "total_rooms_in_scene": len(scene_info.get("rooms", [])),
                    "top_5_found": len(recommendations),
                },
            }
            
            # Save to file
            output_path = os.path.join(scene_output_dir, f"{object_name}_rooms.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=2)
            
            print(f"    ✓ Saved to {output_path}")
            results.append(result_json)
            success_count += 1
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            fail_count += 1
    
    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Query Qwen3-VL for object room recommendations in HM3D scenes"
    )
    
    # Scene selection
    parser.add_argument("--scene", type=str, help="Single scene name (e.g., 00800-TEEsavR23oF)")
    parser.add_argument("--scenes", choices=["all"], default=None, help="Process all available scenes")
    
    # Paths
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR, help="Directory containing object images")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    
    # SSH tunnel
    parser.add_argument("--ssh-host", required=True, help="SSH server host")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH server port")
    parser.add_argument("--ssh-user", required=True, help="SSH username")
    parser.add_argument("--ssh-password", default=None, help="SSH password (non-interactive mode)")
    parser.add_argument("--ssh-key", default=None, help="SSH private key path")
    
    # Qwen server
    parser.add_argument("--vllm-host", default="127.0.0.1", help="vLLM API host (default: 127.0.0.1)")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM API port (default: 8000)")
    parser.add_argument("--local-port", type=int, default=0, help="Local forwarding port (0 = auto-assign)")
    
    # Inference
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help="Model name",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens in response")
    parser.add_argument("--timeout", type=int, default=3600, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Validate scene selection
    if args.scene and args.scenes is not None:
        print("[Error] Cannot specify both --scene and --scenes all")
        sys.exit(1)
    if args.scene and args.scene not in AVAILABLE_SCENES:
        print(f"[Error] Scene not found in merged valid scenes: {args.scene}")
        sys.exit(1)

    scenes_to_process = [args.scene] if args.scene else AVAILABLE_SCENES
    tunnel = SSHTunnel(
        ssh_host=args.ssh_host,
        ssh_port=args.ssh_port,
        ssh_user=args.ssh_user,
        ssh_password=args.ssh_password,
        ssh_key=args.ssh_key,
        remote_host=args.vllm_host,
        remote_port=args.vllm_port,
        local_port=args.local_port,
    )
    
    if not tunnel.start():
        print("[Error] Failed to start SSH tunnel")
        sys.exit(1)
    
    # Create OpenAI client
    client = OpenAI(api_key="EMPTY", base_url=tunnel.base_url, timeout=args.timeout)
    
    # Process scenes
    try:
        total_success = 0
        total_fail = 0
        
        for scene_name in scenes_to_process:
            result = process_scene(
                scene_name,
                args.images_dir,
                args.output_dir,
                tunnel,
                client,
                args.model,
            )
            total_success += result["success_count"]
            total_fail += result["fail_count"]
        
        print(f"\n=== Summary ===")
        print(f"Total successful: {total_success}")
        print(f"Total failed: {total_fail}")
        print(f"Results saved to: {args.output_dir}")
        
    finally:
        tunnel.close()


if __name__ == "__main__":
    main()
