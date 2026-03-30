"""Server-side helper to launch vLLM for Qwen3-VL.

Run this script on the server where GPUs are available.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM server for Qwen3-VL")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-235B-A22B-Thinking",
        help="Model name/path",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--mm-encoder-tp-mode", default="data")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--disable-async-scheduling", action="store_true")
    parser.add_argument("--disable-enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cmd = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--mm-encoder-tp-mode",
        args.mm_encoder_tp_mode,
    ]

    if not args.disable_async_scheduling:
        cmd.append("--async-scheduling")
    if not args.disable_enforce_eager:
        cmd.append("--enforce-eager")

    print("Launching command:")
    print(" ".join(shlex.quote(part) for part in cmd))

    proc = subprocess.run(cmd, check=False)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
