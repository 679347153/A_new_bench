#!/usr/bin/env python3
"""Ask Qwen-VL to visually audit the 20 annotated lifespan layout sheets."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from qwen_credentials import load_dashscope_api_key  # noqa: E402


SYSTEM = """You are a meticulous 3D indoor-scene placement auditor. Inspect only visible evidence.
Return strict JSON, no markdown. Do not invent exact coordinates or claim an issue when occlusion prevents judgment.
Flag: floating above a support, penetrating furniture, object implausibly embedded/oversized, wrong support category,
or clearly wrong upright orientation. Negative world Y alone is NOT an error because HM3D buildings have multiple levels."""


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def clean_json(text: str):
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(value)
    except Exception:
        return {"parse_error": True, "raw": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / "results/visual_checks/场景物体摆放情况")
    ap.add_argument("--output", type=Path, default=ROOT / "results/qwen_visual_audit/lifespan_20")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    args = ap.parse_args()
    key = load_dashscope_api_key()
    if not key:
        raise SystemExit("DashScope credential not found")
    client = OpenAI(api_key=key, base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), timeout=180)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for image in sorted(args.images.glob("*/*.png")):
        prompt = """Audit this single annotated HM3D layout sheet. The left side is a Habitat render; the right side maps
numbered object origins and lists names/world positions. Identify only credible placement problems.
Return exactly this JSON shape:
{"verdict":"pass|questionable|fail","confidence":0.0,"issues":[{"object_index":1,"model_id":"...","issue_type":"floating|penetrating|wrong_support|wrong_orientation|scale_or_embedding","evidence":"short visible reason","recommended_action":"keep|snap_to_support|reassign_support|rotate|remove","severity":"low|medium|high"}],"notes":"short"}
If an object is too occluded or too small to assess, do not flag it. Negative Y coordinates are valid and must not be flagged by themselves."""
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":[
                {"type":"image_url","image_url":{"url":data_url(image)}},
                {"type":"text","text":prompt},
            ]}],
            max_tokens=1400,
        )
        raw = response.choices[0].message.content or ""
        parsed = clean_json(raw)
        record = {"image":str(image),"model":args.model,"audit":parsed}
        (args.output / f"{image.stem}.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        records.append(record)
        print(f"[QWEN] {image.name}: {parsed.get('verdict','parse_error')} issues={len(parsed.get('issues',[]))}",flush=True)
    summary={"created_at":datetime.now(timezone.utc).isoformat(),"model":args.model,"count":len(records),"records":records}
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
