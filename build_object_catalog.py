#!/usr/bin/env python3
from __future__ import annotations

"""Build the unified object catalog used by room-query and layout generation."""

import argparse
from pathlib import Path

from object_catalog import build_catalog, save_catalog, write_missing_semantic_csv
from project_paths import MISSING_SEMANTIC_TEXT_PATH, OBJECT_CATALOG_PATH


def _parse_datasets(text: str) -> list[str]:
    return [x.strip().lower() for x in str(text).split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build object_catalog.json from legacy/YCB/HSSD object sources.")
    parser.add_argument("--datasets", default="legacy,ycb,hssd", help="Comma-separated datasets: legacy,ycb,hssd")
    parser.add_argument("--output", default=str(OBJECT_CATALOG_PATH), help="Output catalog JSON")
    parser.add_argument("--missing-output", default=str(MISSING_SEMANTIC_TEXT_PATH), help="CSV for objects missing semantic text")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files")
    parser.add_argument("--write-missing", action="store_true", help="Write missing semantic text CSV")
    args = parser.parse_args()

    datasets = _parse_datasets(args.datasets)
    entries = build_catalog(datasets)
    by_dataset: dict[str, int] = {}
    missing = 0
    for entry in entries:
        dataset = str(entry.get("dataset", "unknown"))
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
        if str(entry.get("semantic_source", "")).startswith("missing") or not str(entry.get("semantic_text", "")).strip():
            missing += 1

    print(f"[Info] Objects discovered: {len(entries)}")
    for dataset, count in sorted(by_dataset.items()):
        print(f"  - {dataset}: {count}")
    print(f"[Info] Missing semantic text: {missing}")

    if args.dry_run:
        print("[OK] Dry run complete")
        return 0

    out_path = Path(args.output)
    save_catalog(entries, out_path)
    print(f"[OK] Catalog saved: {out_path}")

    if args.write_missing or missing:
        count = write_missing_semantic_csv(entries, Path(args.missing_output))
        print(f"[OK] Missing semantic CSV saved: {args.missing_output} ({count} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
