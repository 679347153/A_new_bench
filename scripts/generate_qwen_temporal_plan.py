#!/usr/bin/env python3
"""Generate conservative, everyday-plausible 2-3 object changes per snapshot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from qwen_credentials import load_dashscope_api_key  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def surface_index(payload):
    result = {}
    for room in payload.get("rooms", []):
        room_id = room.get("room_id")
        room_categories = room.get("room", {}).get("categories", {})
        room_hint = ", ".join(sorted(room_categories, key=room_categories.get, reverse=True)[:8])
        for receptacle in room.get("receptacle_instances", []):
            instance_id = receptacle.get("instance_id", receptacle.get("instance", {}).get("id"))
            result[int(instance_id)] = {
                "instance_id": int(instance_id),
                "room_id": room_id,
                "category": receptacle.get("category", receptacle.get("instance", {}).get("category", "")),
                "room_contents_hint": room_hint,
            }
    return result


def clean_json(text: str):
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(value)


def compatible_support(model_id: str, support: dict) -> bool:
    """Hard everyday-semantics gate; Qwen cannot override these exclusions."""
    model = model_id.lower()
    category = str(support.get("category", "")).lower()
    room_hint = str(support.get("room_contents_hint", "")).lower()
    if "bathroom" in room_hint or "toilet" in room_hint:
        return False
    allowed = {
        "camera": ("table", "counter", "cabinet", "drawers"),
        "alarm_clock": ("nightstand", "table", "drawers", "cabinet"),
        "brass_pot": ("counter", "cabinet", "table"),
        "carrot_cake": ("table", "counter"),
        "food_apple": ("table", "counter", "cabinet"),
        "food_pears": ("table", "counter", "cabinet"),
        "tea_set": ("table", "counter", "cabinet"),
        "wine_bottles": ("table", "counter", "cabinet"),
    }
    for token, categories in allowed.items():
        if token in model:
            return any(value in category for value in categories) and "table tennis" not in category
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--assignment-plan", type=Path, required=True)
    ap.add_argument("--surfaces-json", type=Path, required=True)
    ap.add_argument("--snapshot-requests", type=Path, required=True)
    ap.add_argument("--event-log", type=Path, required=True)
    ap.add_argument("--base-layout", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="qwen3-vl-plus")
    args = ap.parse_args()
    key = load_dashscope_api_key()
    if not key:
        raise SystemExit("DashScope credential not found")

    assignments = read(args.assignment_plan)
    surfaces = surface_index(read(args.surfaces_json))
    base_models = {str(o.get("model_id")) for o in read(args.base_layout).get("objects", [])}
    catalog = []
    for item in assignments.get("assignments", []):
        if item.get("model_id") not in base_models:
            continue
        ids = []
        for raw in [item.get("target_instance_id"), *(item.get("backup_instance_ids", []) or [])]:
            try: iid = int(raw)
            except Exception: continue
            if iid not in ids and iid in surfaces: ids.append(iid)
        candidates = [surfaces[i] for i in ids if compatible_support(str(item.get("model_id", "")), surfaces[i])]
        if not candidates:
            continue
        catalog.append({
            "model_id": item.get("model_id"),
            "name": item.get("name"),
            "current_support_id": item.get("target_instance_id"),
            "candidate_supports": candidates,
        })
    requests = read(args.snapshot_requests).get("snapshot_requests", [])[:10]
    events = {str(e.get("event_id")): e for e in read(args.event_log).get("events", [])}
    timeline = []
    for request in requests:
        timeline.append({
            "snapshot_index": request.get("snapshot_index"),
            "day_index": request.get("day_index"),
            "time": request.get("time"),
            "events": [events.get(str(eid), {"event_id": eid}) for eid in request.get("event_ids", [])],
        })

    prompt = f"""Create a two-day temporal object-placement plan for HM3D scene {args.scene}.
For every snapshot index 1 through 9, select exactly 2 or 3 DISTINCT objects that a resident would plausibly move since the previous timestamp. Choose target_instance_id ONLY from that object's candidate_supports and choose a DIFFERENT support from its current support at that point in the timeline. Preserve temporal continuity and use everyday activities: morning routine, meals, work/leisure and cleanup. The catalog has already removed decorations, large furniture, bathroom/toilet contexts and incompatible supports; do not invent any omitted object or support. Do not repeatedly move an object without a clear reason. Changes must represent a meaningful support/position change, not rotation-only.

Available objects and supports:
{json.dumps(catalog, ensure_ascii=False)}

Timeline context:
{json.dumps(timeline, ensure_ascii=False)}

Return strict JSON only:
{{"scene":"{args.scene}","snapshots":[{{"snapshot_index":1,"activity":"short description","changes":[{{"model_id":"exact catalog model_id","target_instance_id":123,"target_room_id":4,"reason":"short everyday reason"}}]}}]}}
Include exactly snapshots 1..9; each changes array length must be 2 or 3."""
    client = OpenAI(api_key=key, base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"), timeout=300)
    by_model = {str(row["model_id"]).lower(): row for row in catalog}
    initial_support = {str(row["model_id"]).lower(): int(row["current_support_id"]) for row in catalog}
    messages = [
        {"role": "system", "content": "You are a conservative household activity and indoor-placement planner. Obey the supplied candidate constraints exactly."},
        {"role": "user", "content": prompt},
    ]
    plan = None
    rows = []
    last_error = ""
    for attempt in range(1, 5):
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=5000,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or ""
        try:
            candidate = clean_json(raw)
            candidate["model"] = args.model
            candidate_rows = candidate.get("snapshots", [])
            current_support = dict(initial_support)
            if [row.get("snapshot_index") for row in candidate_rows] != list(range(1, 10)):
                raise ValueError("must return exactly snapshot indices 1..9")
            for row in candidate_rows:
                changes = row.get("changes", [])
                if len(changes) not in (2, 3) or len({str(c.get("model_id", "")).lower() for c in changes}) != len(changes):
                    raise ValueError(f"invalid change count/duplicates at snapshot {row.get('snapshot_index')}")
                for change in changes:
                    model = str(change.get("model_id", ""))
                    catalog_row = by_model.get(model.lower(), {})
                    if not catalog_row:
                        raise ValueError(f"unknown model {model}")
                    change["model_id"] = catalog_row["model_id"]
                    allowed = {s["instance_id"]: s for s in catalog_row.get("candidate_supports", [])}
                    iid = int(change.get("target_instance_id", -1))
                    if iid not in allowed:
                        raise ValueError(f"support {iid} is not allowed for {model}; allowed={sorted(allowed)}")
                    if int(change.get("target_room_id")) != int(allowed[iid]["room_id"]):
                        raise ValueError(f"room/support mismatch for {model}")
                    key = str(catalog_row["model_id"]).lower()
                    if iid == current_support[key]:
                        raise ValueError(f"{model} must move away from current support {iid}")
                    current_support[key] = iid
            plan, rows = candidate, candidate_rows
            break
        except Exception as exc:
            last_error = str(exc)
            print(f"[RETRY {attempt}/4] Qwen plan rejected: {last_error}", flush=True)
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"The plan failed strict validation: {last_error}. Return a complete corrected JSON plan. Use only the exact model IDs and per-object candidate support IDs from the original catalog."},
            ])
    if plan is None:
        raise ValueError(f"Qwen failed strict validation after 4 attempts: {last_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Qwen temporal plan: {args.output} snapshots={len(rows)}")


if __name__ == "__main__":
    main()
