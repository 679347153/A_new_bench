"""Connect to a remote vLLM Qwen3-VL endpoint through SSH private-key tunnel.

Usage example:
python qwen3_vl_connect.py \
  --ssh-host 7.216.187.6 \
  --ssh-user ubuntu \
  --remote-port 31822\
  --ssh-key /home/yuhang/zw_ws/qwen/zw_B200.txt \
  --image /home/yuhang/zw_ws/qwen/receipt.jpg \
  --prompt "Read all the text in the image."
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Call remote vLLM Qwen3-VL via SSH private-key tunnel."
	)

	parser.add_argument("--ssh-host", required=True, help="SSH server host or IP")
	parser.add_argument("--ssh-port", type=int, default=22, help="SSH server port")
	parser.add_argument("--ssh-user", required=True, help="SSH username")
	parser.add_argument("--ssh-key", required=True, help="Path to private key file")
	parser.add_argument(
		"--ssh-key-passphrase",
		default=None,
		help="Private key passphrase (if needed)",
	)

	parser.add_argument(
		"--remote-host",
		default="127.0.0.1",
		help="Remote host where vLLM API listens (usually 127.0.0.1)",
	)
	parser.add_argument(
		"--remote-port",
		type=int,
		default=8000,
		help="Remote vLLM API port",
	)
	parser.add_argument(
		"--local-port",
		type=int,
		default=0,
		help="Local forwarding port (0 means auto-assign)",
	)

	parser.add_argument(
		"--model",
		default="Qwen/Qwen3-VL-235B-A22B-Thinking",
		help="Model name served by vLLM",
	)
	parser.add_argument(
		"--image",
		required=True,
		help="Image URL or local image path",
	)
	parser.add_argument(
		"--prompt",
		default="Read all the text in the image.",
		help="Text prompt",
	)
	parser.add_argument("--max-tokens", type=int, default=2048)
	parser.add_argument("--timeout", type=int, default=3600)
	parser.add_argument(
		"--tunnel-ready-timeout",
		type=float,
		default=12.0,
		help="Seconds to wait for SSH tunnel to become ready",
	)

	return parser.parse_args()


def main() -> int:
	args = parse_args()

	ssh_key_path = os.path.expanduser(args.ssh_key)
	if not os.path.exists(ssh_key_path):
		print(f"[Error] SSH private key not found: {ssh_key_path}", file=sys.stderr)
		return 1

	try:
		image_url = _build_image_url(args.image)
	except Exception as exc:
		print(f"[Error] Invalid image input: {exc}", file=sys.stderr)
		return 1

	print(
		f"[Info] Opening SSH tunnel {args.ssh_user}@{args.ssh_host}:{args.ssh_port} "
		f"-> {args.remote_host}:{args.remote_port}"
	)

	local_port = args.local_port if args.local_port > 0 else _pick_free_local_port()
	tunnel_cmd = [
		"ssh",
		"-o",
		"ExitOnForwardFailure=yes",
		"-o",
		"ServerAliveInterval=30",
		"-o",
		"ServerAliveCountMax=3",
		"-i",
		ssh_key_path,
		"-p",
		str(args.ssh_port),
		"-N",
		"-L",
		f"127.0.0.1:{local_port}:{args.remote_host}:{args.remote_port}",
		f"{args.ssh_user}@{args.ssh_host}",
	]

	if args.ssh_key_passphrase:
		print(
			"[Warn] --ssh-key-passphrase is not used by native ssh command. "
			"Use ssh-agent or an unencrypted key, or enter passphrase when prompted."
		)

	print("[Info] SSH command:")
	print(" ".join(shlex.quote(part) for part in tunnel_cmd))
	tunnel_proc = subprocess.Popen(
		tunnel_cmd,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
	)

	try:
		if tunnel_proc.poll() is not None:
			stderr_text = tunnel_proc.stderr.read() if tunnel_proc.stderr else ""
			print(
				f"[Error] SSH tunnel process exited early. Details:\n{stderr_text}",
				file=sys.stderr,
			)
			return 1

		if not _wait_tunnel_ready("127.0.0.1", local_port, args.tunnel_ready_timeout):
			stderr_text = ""
			if tunnel_proc.poll() is not None and tunnel_proc.stderr:
				stderr_text = tunnel_proc.stderr.read()
			print(
				"[Error] SSH tunnel was not ready in time. "
				+ (f"Details:\n{stderr_text}" if stderr_text else ""),
				file=sys.stderr,
			)
			return 1

		base_url = f"http://127.0.0.1:{local_port}/v1"
		print(f"[Info] Tunnel ready at {base_url}")

		client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=args.timeout)

		messages = [
			{
				"role": "user",
				"content": [
					{"type": "image_url", "image_url": {"url": image_url}},
					{"type": "text", "text": args.prompt},
				],
			}
		]

		start = time.time()
		response = client.chat.completions.create(
			model=args.model,
			messages=messages,
			max_tokens=args.max_tokens,
		)
		elapsed = time.time() - start

		print(f"Response costs: {elapsed:.2f}s")
		print("Generated text:")
		print(response.choices[0].message.content)

	except Exception as exc:
		print(f"[Error] Request failed: {exc}", file=sys.stderr)
		return 1
	finally:
		if tunnel_proc.poll() is None:
			tunnel_proc.terminate()
			try:
				tunnel_proc.wait(timeout=5)
			except subprocess.TimeoutExpired:
				tunnel_proc.kill()
		print("[Info] SSH tunnel closed")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
