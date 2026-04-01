#!/usr/bin/env python3
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
  python query_rooms_for_objects.py \\
    --ssh-host 7.216.187.6 --ssh-port 31822 --ssh-user root --ssh-password 666666 \\
    --vllm-host 127.0.0.1 --vllm-port 8000 \\
    --images-dir ./objects_images \\
    --scenes all \\
    --output-dir ./results/scene_info/

  # 或单个场景

  python query_rooms_for_objects.py \\
    --ssh-host 7.216.187.6 --ssh-port 31822 --ssh-user root --ssh-password 666666 \\
    --vllm-host 127.0.0.1 --vllm-port 8000 \\
    --images-dir ./objects_images \\
    --scene 00808-y9hTuugGdiq \\
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
SCENES_DIR = "hm3d"
SCENE_DATASET_CONFIG = "./hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEFAULT_OUTPUT_DIR = "./results/scene_info"
DEFAULT_IMAGES_DIR = "./objects_images"

AVAILABLE_SCENES = [
    "00800-TEEsavR23oF",
    "00801-HaxA7YrQdEC",
    "00802-wcojb4TFT35",
    "00803-k1cupFYWXJ6",
    "00804-BHXhpBwSMLh",
    "00805-SUHsP6z2gcJ",
    "00806-tQ5s4ShP627",
    "00807-rsggHU7g7dh",
    "00808-y9hTuugGdiq",
    "00809-Qpor2mEya8F",
]

QWEN_QUERY_TEMPLATE = (
    "看图片中的物体，它最有可能出现在房间的哪些地方？\n"
    "请给出：\n"
    "1. 前5个最可能的房间区域\n"
    "2. 每个房间在3D空间中的几何中心点坐标 (x, y, z)\n"
    "3. 置信分数（0-1）\n"
    "4. 简短的理由（一句话）\n"
    "\n"
    "格式示例：\n"
    "房间1: region_id=0, 中心=(-2.5, 1.2, 3.8), 置信度=0.95, 理由：这是客厅中央的茶几\n"
    "房间2: region_id=2, 中心=(-1.0, 0.9, 5.2), 置信度=0.80, 理由：厨房台面\n"
    "..."
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
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": QWEN_QUERY_TEMPLATE},
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
    
    # Build map of region_id -> room_info from scene_info
    room_map = {}
    if "rooms" in scene_info:
        for room in scene_info["rooms"]:
            rid = room.get("region_id")
            if rid is not None:
                room_map[rid] = room
    
    # Parse each line looking for "房间N:" pattern
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
    
    return recommendations[:5]  # Top 5 only


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
    
    try:
        from export_scene_info import export_scene
        print(f"  [Info] Exporting scene_info on-the-fly...")
        # 导出到临时目录
        temp_export_dir = os.path.join(output_dir, "temp_export")
        result = export_scene(
            scene_name,
            os.path.join(SCENES_DIR, "minival"),
            SCENE_DATASET_CONFIG,
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
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
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
            raw_output, cleaned_output = query_qwen_for_rooms(client, str(image_path), model)
            
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
                "query": QWEN_QUERY_TEMPLATE,
                "raw_output": raw_output,
                "cleaned_output": cleaned_output,
                "recommended_rooms": recommendations,
                "metadata": {
                    "total_objects_in_scene": len(scene_info.get("objects", [])),
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
    parser.add_argument("--scenes", choices=["all"], default="all", help="Process all available scenes")
    
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
    if args.scene and args.scenes == "all":
        print("[Error] Cannot specify both --scene and --scenes all")
        sys.exit(1)
    
    scenes_to_process = [args.scene] if args.scene else AVAILABLE_SCENES
    
    # Setup SSH tunnel
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
