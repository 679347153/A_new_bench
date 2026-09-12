#!/usr/bin/env python3
"""Snap visibly separated/sunk grounded objects to their recorded support surface."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()
    changed = []
    for path in sorted(args.root.glob("*/lifespan_qwen_10/grounded/layouts/snapshot_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_changes = []
        for obj in payload.get("objects", []):
            gap = float(obj.get("support_gap", 0.0))
            if abs(gap) <= args.threshold:
                continue
            old_y = float(obj["position"][1])
            obj["position"][1] = round(old_y - gap, 4)
            if "support_base_height" in obj:
                obj["support_base_height"] = round(float(obj["support_base_height"]) - gap, 4)
            obj["support_gap"] = 0.0
            obj["support_correction"] = {
                "method": "snap_recorded_aabb_base_to_support_surface",
                "previous_y": old_y,
                "previous_support_gap": gap,
                "threshold_m": args.threshold,
            }
            file_changes.append(obj.get("model_id", obj.get("name", "unknown")))
        if file_changes:
            payload["support_correction"] = {
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "threshold_m": args.threshold,
                "objects": file_changes,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            changed.append((str(path), file_changes))
    print(json.dumps({"changed_files": len(changed), "changes": changed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
