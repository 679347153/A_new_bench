#!/usr/bin/env python3
"""
Lightweight workflow verification for the HM3D room/object pipeline.

Usage:
  python verify_workflow.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_FILES: List[str] = [
    "export_scene_info.py",
    "query_rooms_for_objects.py",
    "sample_and_place_objects.py",
    "test_layout.py",
    "extract_room_instances.py",
    "query_room_receptacle_objects.py",
    "assign_objects_to_receptacle_instances.py",
    "place_objects_on_instances.py",
    "visualize_instance_pointcloud_viser.py",
    "hm3d_paths.py",
]

REQUIRED_DIRS: List[str] = [
    "results",
    "results/scene_info",
    "results/probabilities",
    "results/layouts",
]


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERR] {msg}")


def verify_file_existence() -> bool:
    print("[Verify] File existence")
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        _err(f"Missing files: {missing}")
        return False
    _ok("All required files exist")
    return True


def verify_directory_structure() -> bool:
    print("[Verify] Directory structure")
    ok = True
    for d in REQUIRED_DIRS:
        if Path(d).is_dir():
            _ok(d)
        else:
            _err(f"Missing directory: {d}")
            ok = False
    return ok


def _parse_file(path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def verify_python_syntax() -> bool:
    print("[Verify] Python syntax")
    files = [Path(p) for p in REQUIRED_FILES if p.endswith(".py")]
    ok = True
    for p in files:
        good, msg = _parse_file(p)
        if good:
            _ok(str(p))
        else:
            _err(f"{p}: {msg}")
            ok = False
    return ok


def _collect_functions(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            yield node.name


def verify_function_signatures() -> bool:
    print("[Verify] Function signatures")
    required: Dict[str, List[str]] = {
        "query_room_receptacle_objects.py": [
            "_resolve_room_ids",
            "_extract_top_surface",
            "_write_ply_points",
            "query_receptacles_for_room",
            "main",
        ],
        "assign_objects_to_receptacle_instances.py": [
            "_load_or_sample_objects",
            "_query_assignment_for_object",
            "_resolve_or_generate_surfaces_json",
            "main",
        ],
        "place_objects_on_instances.py": [
            "_build_surface_index",
            "_load_surface_points",
            "place_objects_on_instances",
            "main",
        ],
    }
    ok = True
    for file_name, funcs in required.items():
        path = Path(file_name)
        if not path.is_file():
            _err(f"{file_name} not found")
            ok = False
            continue
        defined = set(_collect_functions(path))
        missing = [f for f in funcs if f not in defined]
        if missing:
            _err(f"{file_name} missing functions: {missing}")
            ok = False
        else:
            _ok(f"{file_name} signatures look good")
    return ok


def verify_artifact_schema_if_present() -> bool:
    """
    Optional schema checks:
    - If a receptacle surfaces json exists, ensure top_surface uses point_cloud_file.
    """
    print("[Verify] Optional artifact schema")
    surface_candidates = list(Path("results/receptacle_queries").glob("*/*_receptacle_surfaces_*.json"))
    if not surface_candidates:
        _warn("No receptacle surfaces artifacts found; skip schema check.")
        return True

    target = max(surface_candidates, key=lambda p: p.stat().st_mtime)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _err(f"Failed to parse {target}: {exc}")
        return False

    rooms = payload.get("rooms", []) if isinstance(payload, dict) else []
    for room in rooms:
        for rec in room.get("receptacle_instances", []) or []:
            top = rec.get("top_surface", {}) if isinstance(rec, dict) else {}
            if not isinstance(top, dict):
                continue
            if "point_cloud_file" not in top:
                _err(f"{target} has top_surface without point_cloud_file")
                return False
            if "points" in top:
                _err(f"{target} still stores top_surface.points")
                return False
    _ok(f"Artifact schema looks good: {target}")
    return True


def main() -> int:
    checks = [
        ("File Existence", verify_file_existence),
        ("Directory Structure", verify_directory_structure),
        ("Python Syntax", verify_python_syntax),
        ("Function Signatures", verify_function_signatures),
        ("Artifact Schema", verify_artifact_schema_if_present),
    ]

    print("=" * 60)
    print("Workflow Verification")
    print("=" * 60)

    results: Dict[str, bool] = {}
    for name, fn in checks:
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            _err(f"{name} raised exception: {exc}")
            results[name] = False
        print("")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nTotal: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
