#!/usr/bin/env python3
from __future__ import annotations

"""Generate semantic text for object catalog entries without images.

This script is primarily for HSSD, whose object ids are hashes and whose local
Habitat object configs often only contain `semantic_id`.  It calls the same
remote Qwen endpoint through the existing SSH tunnel helpers and appends JSONL
records to `data/object_catalog/hssd_semantic_text.jsonl`.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from object_catalog import build_catalog, load_hssd_semantic_text
from project_paths import HSSD_SEMANTIC_TEXT_PATH, OBJECT_CATALOG_PATH
from query_rooms_for_objects import (
    DEFAULT_SSH_HOST,
    DEFAULT_SSH_KEY,
    DEFAULT_SSH_PASSWORD,
    DEFAULT_SSH_PORT,
    DEFAULT_SSH_USER,
    OpenAI,
    SSHTunnel,
    _clean_model_output,
    _extract_json_block,
)


SYSTEM_PROMPT = (
    "You label 3D household object assets for embodied AI benchmarks. "
    "Return compact JSON only."
)

USER_PROMPT_TEMPLATE = """Object metadata:
{object_json}

The object has no photo. Infer a useful household semantic label from metadata.
If the id is only a hash and category is uncertain, say so but still provide a
best-effort description.

Return JSON:
{{
  "category": "short noun category",
  "display_name": "human readable name",
  "semantic_text": "1-2 sentences for deciding likely rooms and support surfaces",
  "likely_rooms": ["kitchen", "living room"],
  "placement_class": "small_tabletop|large_tabletop|floor_only|floor_or_large_surface|soft_surface|tabletop_or_floor",
  "confidence": 0.0
}}
"""


def _load_catalog_or_hssd() -> List[Dict[str, Any]]:
    if OBJECT_CATALOG_PATH.is_file():
        payload = json.loads(OBJECT_CATALOG_PATH.read_text(encoding="utf-8"))
        items = payload.get("objects", []) if isinstance(payload, dict) else payload
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict) and item.get("dataset") == "hssd"]
    return build_catalog(["hssd"])


def _needs_text(entry: Dict[str, Any], existing: Dict[str, Dict[str, Any]]) -> bool:
    key = str(entry.get("object_key", ""))
    model_id = str(entry.get("model_id", ""))
    if key in existing or model_id in existing:
        return False
    source = str(entry.get("semantic_source", ""))
    return source.startswith("missing") or not str(entry.get("semantic_text", "")).strip()


def _query_text(client: OpenAI, model: str, entry: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    payload = {
        "object_key": entry.get("object_key"),
        "model_id": entry.get("model_id"),
        "template_config_path": entry.get("template_config_path"),
        "render_asset": entry.get("render_asset"),
        "semantic_id": entry.get("semantic_id"),
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(object_json=json.dumps(payload, ensure_ascii=False, indent=2)),
            },
        ],
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    cleaned = _clean_model_output(raw)
    parsed = _extract_json_block(cleaned) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    parsed["raw_output"] = raw
    parsed["cleaned_output"] = cleaned
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HSSD semantic text JSONL through Qwen.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum missing entries to process")
    parser.add_argument("--output", default=str(HSSD_SEMANTIC_TEXT_PATH), help="Output JSONL")
    parser.add_argument("--dry-run", action="store_true", help="List missing entries without calling Qwen")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--ssh-password", default=DEFAULT_SSH_PASSWORD)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--local-port", type=int, default=0)
    args = parser.parse_args()

    entries = _load_catalog_or_hssd()
    existing = load_hssd_semantic_text(Path(args.output))
    missing = [entry for entry in entries if _needs_text(entry, existing)]
    if args.limit and args.limit > 0:
        missing = missing[: int(args.limit)]

    print(f"[Info] HSSD entries needing semantic text: {len(missing)}")
    for item in missing[:10]:
        print(f"  - {item.get('object_key')} semantic_id={item.get('semantic_id')}")
    if args.dry_run or not missing:
        return 0

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
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key="EMPTY", base_url=tunnel.base_url, timeout=args.timeout)
    try:
        with out_path.open("a", encoding="utf-8") as f:
            for idx, entry in enumerate(missing, start=1):
                print(f"[Progress] semantic_text {idx}/{len(missing)} {entry.get('object_key')}")
                record: Dict[str, Any] = {
                    "object_key": entry.get("object_key"),
                    "model_id": entry.get("model_id"),
                    "semantic_id": entry.get("semantic_id"),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "semantic_source": "qwen_text_from_asset_metadata",
                }
                try:
                    record.update(_query_text(client, args.model, entry, int(args.max_tokens)))
                except Exception as exc:
                    record["semantic_source"] = "qwen_error"
                    record["error"] = str(exc)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        tunnel.close()

    print(f"[OK] HSSD semantic text appended: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
