"""Connect to a remote vLLM Qwen3-VL endpoint through SSH private-key tunnel.

Usage example:
python qwen3_vl_connect.py \
  --ssh-host 7.216.187.6 \
  --ssh-user ubuntu \
  --remote-port 31822\
  --ssh-key /home/yuhang/zw_ws/qwen/zw_B200.txt \
  --image /home/yuhang/zw_ws/qwen/receipt.png \
  --prompt "Read all the text in the image."
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

from openai import OpenAI
from sshtunnel import SSHTunnelForwarder


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

	tunnel = SSHTunnelForwarder(
		ssh_address_or_host=(args.ssh_host, args.ssh_port),
		ssh_username=args.ssh_user,
		ssh_pkey=ssh_key_path,
		ssh_private_key_password=args.ssh_key_passphrase,
		remote_bind_address=(args.remote_host, args.remote_port),
		local_bind_address=("127.0.0.1", args.local_port),
	)

	try:
		tunnel.start()
		local_port = tunnel.local_bind_port
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
		if tunnel.is_active:
			tunnel.stop()
			print("[Info] SSH tunnel closed")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
