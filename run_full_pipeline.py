#!/usr/bin/env python3
from __future__ import annotations

"""
One-click project pipeline runner.

This script is intentionally placed in the project root.  It orchestrates the
existing executable modules instead of re-implementing their logic:

1. Optionally prepare the regular data/ directory layout.
2. Build the unified object catalog.
3. Generate one or more final object layouts with core/batch_generate_layouts.py.
4. Build benchmark episodes from the generated layout manifest(s).
5. Optionally run oracle/noop smoke evaluation.

By default every subprocess is wrapped with `core/log_filter.py --run ...`, so
the terminal output and `pipeline.log` both use the same Habitat/HM3D noise
filtering rules as the rest of the project.  Use `--no-log-filter` only when you
need the raw unfiltered logs for low-level debugging.

Filtered subprocess output is streamed to the terminal and to:

  results/pipeline_runs/<run_id>/pipeline.log

A compact bug/result report is continuously refreshed at:

  results/pipeline_runs/<run_id>/bugs_and_results.txt

Machine-readable stage metadata is written to:

  results/pipeline_runs/<run_id>/run_summary.json

Examples:

  python run_full_pipeline.py

  python run_full_pipeline.py --config run_full_pipeline_config.json

  python run_full_pipeline.py --config run_full_pipeline_config.json --scene 00808-y9hTuugGdiq --num-layouts 5
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "results" / "pipeline_runs"
DEFAULT_OBJECT_CATALOG = PROJECT_ROOT / "data" / "object_catalog" / "object_catalog.json"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "object_images" / "legacy"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "scenes" / "hm3d"
DEFAULT_ROOMS_INFO_DIR = PROJECT_ROOT / "results" / "scene_info"
DEFAULT_PROBABILITIES_DIR = PROJECT_ROOT / "results" / "probabilities"
DEFAULT_LAYOUTS_DIR = PROJECT_ROOT / "results" / "layouts"
DEFAULT_EPISODES_ROOT = PROJECT_ROOT / "benchmark" / "episodes"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "run_full_pipeline_config.json"

BUG_LINE_RE = re.compile(
    r"(traceback|exception|error|failed|fatal|not found|cannot|timeout|refusing|warning)",
    flags=re.IGNORECASE,
)
RESULT_LINE_RE = re.compile(
    r"(\[OK\]|\[Info\]|summary|saved|output|manifest|episodes|layouts|success|failed_count|placed=)",
    flags=re.IGNORECASE,
)

CONFIG_GROUPS = (
    "pipeline",
    "paths",
    "objects",
    "qwen",
    "generation",
    "placement",
    "benchmark",
    "evaluation",
)


def _now_id(prefix: str = "run") -> str:
    return time.strftime(f"{prefix}_%Y%m%d_%H%M%S")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def _shell_join(cmd: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in cmd])
    return shlex.join([str(x) for x in cmd])


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_config_payload() -> Dict[str, Any]:
    return {
        "_comment": (
            "Default parameters for run_full_pipeline.py. Command-line options "
            "override values in this file."
        ),
        "pipeline": {
            "scene": "00808-y9hTuugGdiq",
            "plan_json": None,
            "run_id": None,
            "dry_run": False,
            "smoke": False,
            "no_log_filter": False,
            "keep_going": False,
            "fail_fast": False,
        },
        "paths": {
            "pipeline_output_dir": "results/pipeline_runs",
            "images_dir": "data/object_images/legacy",
            "data_dir": "data/scenes/hm3d",
            "rooms_info_dir": "results/scene_info",
            "probabilities_dir": "results/probabilities",
            "layouts_dir": "results/layouts",
            "object_catalog": "data/object_catalog/object_catalog.json",
            "episodes_output_root": "benchmark/episodes",
            "episodes_images_dir": "data/object_images/legacy",
            "objects_dir": None,
            "surfaces_json": None,
            "layout_manifest": [],
        },
        "objects": {
            "object_datasets": "legacy,ycb,hssd",
            "object_set": None,
            "limit_objects": 50,
        },
        "qwen": {
            "ssh_host": "7.216.187.6",
            "ssh_port": 30180,
            "ssh_user": "root",
            "ssh_password": "666666",
            "ssh_key": None,
            "vllm_host": "127.0.0.1",
            "vllm_port": 8000,
            "local_port": 0,
            "model": "Qwen/Qwen3-VL-235B-A22B-Thinking",
            "max_tokens": 1024,
            "timeout": 3600,
            "disable_llm": False,
            "disable_assignment_llm": False,
            "disable_surface_llm": False,
            "skip_api_health_check": False,
        },
        "generation": {
            "num_layouts": 10,
            "base_seed": 42,
            "prepare_structure": False,
            "skip_catalog": False,
            "catalog_dry_run": False,
            "skip_layouts": False,
            "skip_episodes": False,
            "regenerate_room_queries": False,
            "regenerate_probabilities": False,
            "regenerate_surfaces": False,
            "keep_intermediates": False,
            "no_progress": False,
        },
        "placement": {
            "min_distance": 0.25,
            "max_trials_per_object": 30,
            "settle_steps": 45,
            "disable_placement_retry": False,
        },
        "benchmark": {
            "benchmark_version": None,
            "split": "val",
            "episodes_per_layout": 3,
            "min_subtasks": 5,
            "max_subtasks": 10,
            "success_radius": 1.2,
            "max_steps": 500,
            "min_objects": 1,
            "episode_seed": 42,
        },
        "evaluation": {
            "run_oracle": False,
            "run_noop": False,
            "sample_start_pose": False,
            "load_layout_objects": False,
            "no_euclidean_fallback": False,
        },
    }


def _normalize_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _flatten_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten grouped config JSON into argparse destination names."""
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        norm_key = _normalize_key(key)
        if norm_key.startswith("_"):
            continue
        if norm_key in CONFIG_GROUPS and isinstance(value, dict):
            for child_key, child_value in value.items():
                child_norm = _normalize_key(child_key)
                if not child_norm.startswith("_"):
                    out[child_norm] = child_value
        else:
            out[norm_key] = value
    return out


def _load_config_defaults(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a JSON object: {path}")
    defaults = _flatten_config(payload)
    defaults["config"] = str(path)
    return defaults


def _resolve_project_path_value(value: Any) -> Any:
    if value in (None, ""):
        return value
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _resolve_path_args(args: argparse.Namespace) -> None:
    for key in (
        "pipeline_output_dir",
        "object_catalog",
        "images_dir",
        "data_dir",
        "rooms_info_dir",
        "probabilities_dir",
        "layouts_dir",
        "surfaces_json",
        "object_set",
        "episodes_output_root",
        "episodes_images_dir",
        "plan_json",
    ):
        setattr(args, key, _resolve_project_path_value(getattr(args, key, None)))
    args.layout_manifest = [
        _resolve_project_path_value(item) for item in (getattr(args, "layout_manifest", []) or [])
    ]
    if getattr(args, "objects_dir", None) and os.pathsep not in str(args.objects_dir):
        args.objects_dir = _resolve_project_path_value(args.objects_dir)


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: Sequence[str]) -> List[str]:
    option_to_dest: Dict[str, str] = {}
    for action in parser._actions:  # argparse has no public iterator for this mapping.
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    out: List[str] = []
    for item in argv:
        if not str(item).startswith("--"):
            continue
        option = str(item).split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest and dest not in out:
            out.append(dest)
    return out


class TeeLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", errors="replace")

    def write(self, text: str = "") -> None:
        print(text, end="" if text.endswith("\n") else "\n")
        self.handle.write(text if text.endswith("\n") else text + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _append_arg(cmd: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and value == "":
        return
    cmd.extend([flag, str(value)])


def _append_flag(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def _pipeline_command(args: argparse.Namespace, cmd: List[str]) -> List[str]:
    if args.no_log_filter:
        return cmd
    return [sys.executable, "core/log_filter.py", "--run", _shell_join(cmd)]


def _stage_record(name: str, cmd: Sequence[str]) -> Dict[str, Any]:
    return {
        "name": name,
        "command": [str(x) for x in cmd],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_code": None,
        "duration_sec": None,
        "bug_lines": [],
        "result_lines": [],
        "log_filter_applied": None,
    }


def _refresh_report(summary: Dict[str, Any], report_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Pipeline Bugs And Results")
    lines.append("")
    lines.append(f"run_id: {summary.get('run_id')}")
    lines.append(f"status: {summary.get('status')}")
    lines.append(f"log_file: {summary.get('log_file')}")
    lines.append(f"summary_file: {summary.get('summary_file')}")
    lines.append(f"log_filter: {summary.get('log_filter')}")
    lines.append("")
    lines.append("## Outputs")
    outputs = summary.get("outputs", {})
    if isinstance(outputs, dict) and outputs:
        for key in sorted(outputs):
            value = outputs[key]
            if isinstance(value, list):
                lines.append(f"- {key}:")
                for item in value:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- none yet")
    lines.append("")
    lines.append("## Stage Results")
    for stage in summary.get("stages", []):
        lines.append(
            "- {name}: exit_code={code} duration={duration}".format(
                name=stage.get("name"),
                code=stage.get("exit_code"),
                duration=stage.get("duration_sec"),
            )
        )
        lines.append(f"  log_filter_applied: {stage.get('log_filter_applied')}")
        result_lines = stage.get("result_lines") or []
        for item in result_lines[-8:]:
            lines.append(f"  result: {item}")
    lines.append("")
    lines.append("## Bugs / Warnings / Failures")
    any_bug = False
    top_level_bugs = summary.get("bugs", [])
    if isinstance(top_level_bugs, list) and top_level_bugs:
        any_bug = True
        lines.append("### pipeline")
        for item in top_level_bugs[-30:]:
            lines.append(f"- {item}")
    for stage in summary.get("stages", []):
        bug_lines = stage.get("bug_lines") or []
        if not bug_lines:
            continue
        any_bug = True
        lines.append(f"### {stage.get('name')}")
        for item in bug_lines[-30:]:
            lines.append(f"- {item}")
    if not any_bug:
        lines.append("- none captured yet")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_summary(summary: Dict[str, Any], summary_path: Path, report_path: Path) -> None:
    _write_json(summary_path, summary)
    _refresh_report(summary, report_path)


def run_command(
    *,
    name: str,
    cmd: List[str],
    args: argparse.Namespace,
    log: TeeLog,
    summary: Dict[str, Any],
    summary_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    record = _stage_record(name, cmd)
    summary.setdefault("stages", []).append(record)
    _save_summary(summary, summary_path, report_path)

    actual_cmd = _pipeline_command(args, cmd)
    record["log_filter_applied"] = actual_cmd != cmd
    log.write("\n" + "=" * 88)
    log.write(f"[Stage] {name}")
    log.write(f"[Command] {_shell_join(cmd)}")
    if actual_cmd != cmd:
        log.write(f"[Wrapped] {_shell_join(actual_cmd)}")
        log.write("[LogFilter] enabled: using core/log_filter.py default Habitat/HM3D suppression rules")
    else:
        log.write("[LogFilter] disabled: raw subprocess output will be recorded")
    log.write("=" * 88)

    if args.dry_run:
        record["exit_code"] = 0
        record["duration_sec"] = 0.0
        record["result_lines"].append("dry-run: command not executed")
        log.write(f"[StageDone] {name} exit_code=0 duration=0.0s dry-run")
        _save_summary(summary, summary_path, report_path)
        return record

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    start = time.time()
    try:
        proc = subprocess.Popen(
            actual_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        record["exit_code"] = 1
        record["duration_sec"] = round(time.time() - start, 3)
        record["bug_lines"].append(f"failed to start command: {exc}")
        log.write(f"[Error] Failed to start command: {exc}")
        _save_summary(summary, summary_path, report_path)
        return record

    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        log.write(line)
        stripped = line.strip()
        if stripped and BUG_LINE_RE.search(stripped) and len(record["bug_lines"]) < 300:
            record["bug_lines"].append(stripped)
            _save_summary(summary, summary_path, report_path)
        elif stripped and RESULT_LINE_RE.search(stripped) and len(record["result_lines"]) < 300:
            record["result_lines"].append(stripped)

    proc.wait()
    record["exit_code"] = int(proc.returncode or 0)
    record["duration_sec"] = round(time.time() - start, 3)
    record["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if record["exit_code"] != 0:
        record["bug_lines"].append(f"stage exited with non-zero code: {record['exit_code']}")
    log.write(f"[StageDone] {name} exit_code={record['exit_code']} duration={record['duration_sec']}s")
    _save_summary(summary, summary_path, report_path)
    return record


def _new_layout_manifests(layouts_dir: Path, start_time: float) -> List[Path]:
    if not layouts_dir.exists():
        return []
    candidates = []
    for path in layouts_dir.rglob("manifest.json"):
        try:
            if path.stat().st_mtime >= start_time - 2.0:
                candidates.append(path)
        except OSError:
            continue
    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def _layout_manifest_success_count(path: Path) -> int:
    payload = _read_json(path) or {}
    try:
        return int(payload.get("success_count", 0))
    except Exception:
        return 0


def _dry_run_manifest_placeholders(args: argparse.Namespace) -> List[Path]:
    if args.plan_json:
        return [Path(args.layouts_dir) / "DRY_RUN_SCENE" / "batch_DRY_RUN" / "manifest.json"]
    return [Path(args.layouts_dir) / str(args.scene) / "batch_DRY_RUN" / "manifest.json"]


def _catalog_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "core/build_object_catalog.py",
        "--datasets",
        args.object_datasets,
        "--output",
        args.object_catalog,
        "--write-missing",
    ]
    if args.catalog_dry_run:
        cmd.append("--dry-run")
    return cmd


def _prepare_command() -> List[str]:
    return [sys.executable, "core/prepare_project_structure.py"]


def _batch_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "core/batch_generate_layouts.py",
        "--num-layouts",
        str(args.num_layouts),
        "--base-seed",
        str(args.base_seed),
        "--images-dir",
        args.images_dir,
        "--object-catalog",
        args.object_catalog,
        "--object-datasets",
        args.object_datasets,
        "--rooms-info-dir",
        args.rooms_info_dir,
        "--probabilities-dir",
        args.probabilities_dir,
        "--layouts-dir",
        args.layouts_dir,
        "--data-dir",
        args.data_dir,
        "--ssh-host",
        args.ssh_host,
        "--ssh-port",
        str(args.ssh_port),
        "--ssh-user",
        args.ssh_user,
        "--vllm-host",
        args.vllm_host,
        "--vllm-port",
        str(args.vllm_port),
        "--local-port",
        str(args.local_port),
        "--model",
        args.model,
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--min-distance",
        str(args.min_distance),
        "--max-trials-per-object",
        str(args.max_trials_per_object),
        "--settle-steps",
        str(args.settle_steps),
    ]
    if args.plan_json:
        cmd.extend(["--plan-json", args.plan_json])
    else:
        cmd.extend(["--scene", args.scene])
    if int(args.limit_objects) > 0:
        cmd.extend(["--limit-objects", str(args.limit_objects)])
    _append_arg(cmd, "--object-set", args.object_set)
    _append_arg(cmd, "--objects-dir", args.objects_dir)
    _append_arg(cmd, "--surfaces-json", args.surfaces_json)
    _append_arg(cmd, "--ssh-password", args.ssh_password)
    _append_arg(cmd, "--ssh-key", args.ssh_key)
    _append_flag(cmd, "--regenerate-room-queries", args.regenerate_room_queries)
    _append_flag(cmd, "--regenerate-probabilities", args.regenerate_probabilities)
    _append_flag(cmd, "--regenerate-surfaces", args.regenerate_surfaces)
    _append_flag(cmd, "--disable-assignment-llm", args.disable_assignment_llm or args.disable_llm)
    _append_flag(cmd, "--disable-surface-llm", args.disable_surface_llm or args.disable_llm)
    _append_flag(cmd, "--keep-intermediates", args.keep_intermediates)
    _append_flag(cmd, "--fail-fast", args.fail_fast)
    _append_flag(cmd, "--skip-api-health-check", args.skip_api_health_check)
    _append_flag(cmd, "--no-progress", args.no_progress)
    _append_flag(cmd, "--disable-placement-retry", args.disable_placement_retry)
    return cmd


def _build_episodes_command(args: argparse.Namespace, manifests: Sequence[Path], version: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.build_episodes",
        "--layout-manifest",
        *[str(path) for path in manifests],
        "--version",
        version,
        "--split",
        args.split,
        "--output-root",
        args.episodes_output_root,
        "--images-dir",
        args.episodes_images_dir,
        "--episodes-per-layout",
        str(args.episodes_per_layout),
        "--min-subtasks",
        str(args.min_subtasks),
        "--max-subtasks",
        str(args.max_subtasks),
        "--success-radius",
        str(args.success_radius),
        "--max-steps",
        str(args.max_steps),
        "--min-objects",
        str(args.min_objects),
        "--seed",
        str(args.episode_seed),
    ]
    return cmd


def _runner_command(args: argparse.Namespace, episodes_dir: Path, output_path: Path, mode: str) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.runner",
        "--episodes",
        str(episodes_dir),
        "--output",
        str(output_path),
        "--mode",
        mode,
        "--agent-id",
        mode,
        "--data-dir",
        args.data_dir,
    ]
    _append_arg(cmd, "--objects-dir", args.objects_dir)
    _append_flag(cmd, "--sample-start-pose", args.sample_start_pose)
    _append_flag(cmd, "--load-layout-objects", args.load_layout_objects)
    if args.no_euclidean_fallback:
        cmd.append("--no-euclidean-fallback")
    else:
        cmd.append("--euclidean-fallback")
    return cmd


def _evaluate_command(args: argparse.Namespace, episodes_dir: Path, trajectories: Path, output_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.evaluate",
        "--episodes",
        str(episodes_dir),
        "--trajectories",
        str(trajectories),
        "--output-dir",
        str(output_dir),
        "--data-dir",
        args.data_dir,
    ]
    _append_arg(cmd, "--objects-dir", args.objects_dir)
    if args.no_euclidean_fallback:
        cmd.append("--no-euclidean-fallback")
    else:
        cmd.append("--euclidean-fallback")
    return cmd


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_parser.add_argument("--no-config", action="store_true")
    pre_parser.add_argument("--write-default-config", nargs="?", const=str(DEFAULT_CONFIG_PATH), default=None)
    pre_args, _ = pre_parser.parse_known_args(raw_argv)

    if pre_args.write_default_config is not None:
        config_path = Path(pre_args.write_default_config).expanduser()
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        _write_json(config_path, _default_config_payload())
        print(f"[OK] Default pipeline config written: {config_path}")
        raise SystemExit(0)

    config_defaults: Dict[str, Any] = {}
    explicit_config = any(item == "--config" or item.startswith("--config=") for item in raw_argv)
    if not pre_args.no_config:
        config_path = Path(pre_args.config).expanduser()
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if config_path.is_file() or explicit_config:
            config_defaults = _load_config_defaults(str(config_path))

    parser = argparse.ArgumentParser(
        description="One-click runner for layout generation, task generation, and optional smoke evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="JSON parameter file loaded before parsing CLI overrides")
    parser.add_argument("--no-config", action="store_true", help="Ignore the default/configured parameter file")
    parser.add_argument(
        "--write-default-config",
        nargs="?",
        const=str(DEFAULT_CONFIG_PATH),
        default=None,
        help="Write a default JSON parameter file and exit; optional path may be provided",
    )
    parser.add_argument("--scene", default="00808-y9hTuugGdiq", help="Single scene name. Ignored when --plan-json is set.")
    parser.add_argument("--plan-json", default=None, help="Optional multi-scene plan JSON for batch_generate_layouts.py")
    parser.add_argument("--run-id", default=None, help="Run id used under results/pipeline_runs")
    parser.add_argument("--pipeline-output-dir", default=str(DEFAULT_RUNS_DIR), help="Directory for pipeline logs and reports")
    parser.add_argument("--dry-run", action="store_true", help="Print and record commands without executing them")
    parser.add_argument("--smoke", action="store_true", help="Fast local smoke mode: 2 layouts, 10 objects, heuristic LLM-disabled path")
    parser.add_argument("--no-log-filter", action="store_true", help="Do not wrap subprocesses with core/log_filter.py")
    parser.add_argument("--keep-going", action="store_true", help="Continue independent later stages after a non-zero stage when possible")
    parser.add_argument("--fail-fast", action="store_true", help="Pass fail-fast to batch generation")

    parser.add_argument("--prepare-structure", action="store_true", help="Run core/prepare_project_structure.py before other stages")
    parser.add_argument("--skip-catalog", action="store_true", help="Skip object catalog build stage")
    parser.add_argument("--catalog-dry-run", action="store_true", help="Run catalog stage in dry-run mode")
    parser.add_argument("--skip-layouts", action="store_true", help="Skip batch layout generation and use --layout-manifest")
    parser.add_argument("--layout-manifest", nargs="*", default=[], help="Existing manifest(s) for episode generation")
    parser.add_argument("--skip-episodes", action="store_true", help="Skip benchmark episode generation")
    parser.add_argument("--run-oracle", action="store_true", help="Run oracle smoke trajectory and evaluation")
    parser.add_argument("--run-noop", action="store_true", help="Run noop smoke trajectory and evaluation")

    parser.add_argument("--num-layouts", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--object-datasets", default="legacy,ycb,hssd")
    parser.add_argument("--object-catalog", default=str(DEFAULT_OBJECT_CATALOG))
    parser.add_argument("--object-set", default=None)
    parser.add_argument("--limit-objects", type=int, default=50)
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--rooms-info-dir", default=str(DEFAULT_ROOMS_INFO_DIR))
    parser.add_argument("--probabilities-dir", default=str(DEFAULT_PROBABILITIES_DIR))
    parser.add_argument("--layouts-dir", default=str(DEFAULT_LAYOUTS_DIR))
    parser.add_argument("--objects-dir", default=None)
    parser.add_argument("--surfaces-json", default=None)

    parser.add_argument("--disable-llm", action="store_true", help="Disable both assignment and surface LLM paths")
    parser.add_argument("--disable-assignment-llm", action="store_true")
    parser.add_argument("--disable-surface-llm", action="store_true")
    parser.add_argument("--regenerate-room-queries", action="store_true")
    parser.add_argument("--regenerate-probabilities", action="store_true")
    parser.add_argument("--regenerate-surfaces", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--skip-api-health-check", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--disable-placement-retry", action="store_true")

    parser.add_argument("--min-distance", type=float, default=0.25)
    parser.add_argument("--max-trials-per-object", type=int, default=30)
    parser.add_argument("--settle-steps", type=int, default=45)

    parser.add_argument("--ssh-host", default="7.216.187.6")
    parser.add_argument("--ssh-port", type=int, default=30180)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--ssh-password", default="666666")
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--local-port", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-235B-A22B-Thinking")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=3600)

    parser.add_argument("--benchmark-version", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes-output-root", default=str(DEFAULT_EPISODES_ROOT))
    parser.add_argument("--episodes-images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--episodes-per-layout", type=int, default=3)
    parser.add_argument("--min-subtasks", type=int, default=5)
    parser.add_argument("--max-subtasks", type=int, default=10)
    parser.add_argument("--success-radius", type=float, default=1.2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--min-objects", type=int, default=1)
    parser.add_argument("--episode-seed", type=int, default=42)

    parser.add_argument("--sample-start-pose", action="store_true")
    parser.add_argument("--load-layout-objects", action="store_true")
    parser.add_argument("--no-euclidean-fallback", action="store_true")
    parser.set_defaults(**config_defaults)
    explicit_cli_keys = _explicit_cli_dests(parser, raw_argv)
    args = parser.parse_args(raw_argv)
    args._explicit_cli_keys = explicit_cli_keys
    _resolve_path_args(args)
    return args


def _apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    explicit = set(getattr(args, "_explicit_cli_keys", []) or [])
    if "num_layouts" not in explicit:
        args.num_layouts = 2
    if "limit_objects" not in explicit:
        args.limit_objects = 10
    if "episodes_per_layout" not in explicit:
        args.episodes_per_layout = 1
    if "disable_llm" not in explicit:
        args.disable_llm = True
    if "disable_assignment_llm" not in explicit:
        args.disable_assignment_llm = True
    if "disable_surface_llm" not in explicit:
        args.disable_surface_llm = True
    if "skip_api_health_check" not in explicit:
        args.skip_api_health_check = True
    if "no_progress" not in explicit:
        args.no_progress = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _apply_smoke_defaults(args)

    run_id = args.run_id or _now_id()
    run_dir = Path(args.pipeline_output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    summary_path = run_dir / "run_summary.json"
    report_path = run_dir / "bugs_and_results.txt"
    version = args.benchmark_version or run_id

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "log_file": str(log_path),
        "summary_file": str(summary_path),
        "report_file": str(report_path),
        "config_file": str(getattr(args, "config", "")),
        "log_filter": {
            "enabled": not bool(args.no_log_filter),
            "wrapper": "core/log_filter.py --run",
            "note": "Subprocess output is filtered with the same default Habitat/HM3D suppression rules as core/log_filter.py unless --no-log-filter is set.",
        },
        "args": vars(args),
        "stages": [],
        "outputs": {
            "run_dir": str(run_dir),
        },
    }

    log = TeeLog(log_path)
    exit_code = 0
    layout_start_time = time.time()
    manifest_paths = [Path(x) for x in (args.layout_manifest or [])]

    try:
        log.write(f"[Pipeline] run_id={run_id}")
        log.write(f"[Pipeline] project_root={PROJECT_ROOT}")
        log.write(f"[Pipeline] log={log_path}")
        log.write(f"[Pipeline] report={report_path}")
        _save_summary(summary, summary_path, report_path)

        if args.prepare_structure:
            record = run_command(
                name="prepare_project_structure",
                cmd=_prepare_command(),
                args=args,
                log=log,
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
            )
            if record["exit_code"] != 0 and not args.keep_going:
                return_code = int(record["exit_code"])
                summary["status"] = "failed"
                summary["exit_code"] = return_code
                _save_summary(summary, summary_path, report_path)
                return return_code

        if not args.skip_catalog:
            record = run_command(
                name="build_object_catalog",
                cmd=_catalog_command(args),
                args=args,
                log=log,
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
            )
            if record["exit_code"] != 0:
                exit_code = 1
                if not args.keep_going:
                    summary["status"] = "failed"
                    summary["exit_code"] = exit_code
                    _save_summary(summary, summary_path, report_path)
                    return exit_code

        if not args.skip_layouts:
            layout_start_time = time.time()
            record = run_command(
                name="batch_generate_layouts",
                cmd=_batch_command(args),
                args=args,
                log=log,
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
            )
            discovered = _new_layout_manifests(Path(args.layouts_dir), layout_start_time)
            if discovered:
                manifest_paths = discovered
            elif args.dry_run and not manifest_paths:
                manifest_paths = _dry_run_manifest_placeholders(args)
            summary["outputs"]["layout_manifests"] = [str(path) for path in manifest_paths]
            summary["outputs"]["layout_success_count"] = sum(_layout_manifest_success_count(path) for path in manifest_paths)
            _save_summary(summary, summary_path, report_path)
            if record["exit_code"] != 0:
                exit_code = 1
                if not args.keep_going:
                    summary["status"] = "failed"
                    summary["exit_code"] = exit_code
                    _save_summary(summary, summary_path, report_path)
                    return exit_code

        if not manifest_paths and not args.skip_episodes:
            msg = "No layout manifest available for episode generation."
            log.write(f"[Error] {msg}")
            summary.setdefault("bugs", []).append(msg)
            exit_code = 1
            if not args.keep_going:
                summary["status"] = "failed"
                summary["exit_code"] = exit_code
                _save_summary(summary, summary_path, report_path)
                return exit_code

        episodes_dir = Path(args.episodes_output_root) / version / args.split
        if not args.skip_episodes:
            record = run_command(
                name="build_benchmark_episodes",
                cmd=_build_episodes_command(args, manifest_paths, version),
                args=args,
                log=log,
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
            )
            summary["outputs"]["benchmark_version"] = version
            summary["outputs"]["episodes_dir"] = str(episodes_dir)
            _save_summary(summary, summary_path, report_path)
            if record["exit_code"] != 0:
                exit_code = 1
                if not args.keep_going:
                    summary["status"] = "failed"
                    summary["exit_code"] = exit_code
                    _save_summary(summary, summary_path, report_path)
                    return exit_code

        for mode, enabled in (("oracle", args.run_oracle), ("noop", args.run_noop)):
            if not enabled:
                continue
            traj_path = run_dir / f"{mode}_trajectories.jsonl"
            eval_dir = run_dir / f"{mode}_eval"
            record = run_command(
                name=f"run_{mode}_agent",
                cmd=_runner_command(args, episodes_dir, traj_path, mode),
                args=args,
                log=log,
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
            )
            summary["outputs"][f"{mode}_trajectories"] = str(traj_path)
            _save_summary(summary, summary_path, report_path)
            if record["exit_code"] != 0:
                exit_code = 1
                if not args.keep_going:
                    break
                continue
            eval_record = run_command(
                name=f"evaluate_{mode}_agent",
                cmd=_evaluate_command(args, episodes_dir, traj_path, eval_dir),
                args=args,
                log=log,
                summary=summary,
                summary_path=summary_path,
                report_path=report_path,
            )
            summary["outputs"][f"{mode}_eval_dir"] = str(eval_dir)
            _save_summary(summary, summary_path, report_path)
            if eval_record["exit_code"] != 0:
                exit_code = 1
                if not args.keep_going:
                    break

        summary["status"] = "ok" if exit_code == 0 else "failed"
        summary["exit_code"] = exit_code
        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_summary(summary, summary_path, report_path)
        log.write("\n[PipelineDone] status={status} exit_code={code}".format(status=summary["status"], code=exit_code))
        log.write(f"[PipelineDone] report={report_path}")
        return exit_code
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
