from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class EpisodeResult:
    success: bool
    path_length: float
    shortest_path_length: float
    elapsed_seconds: float
    dynamic_memory_correct: int
    dynamic_memory_total: int
    fixed_memory_correct: int
    fixed_memory_total: int
    steps: int


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def spl(success: bool, path_length: float, shortest_path_length: float) -> float:
    if not success:
        return 0.0
    if shortest_path_length <= 0:
        return 0.0
    return shortest_path_length / max(path_length, shortest_path_length)


def soft_spl(path_length: float, shortest_path_length: float, goal_distance: float, threshold: float = 1.0) -> float:
    # A lightweight SoftSPL approximation: success is relaxed by distance to goal.
    soft_success = max(0.0, 1.0 - safe_div(goal_distance, threshold))
    if shortest_path_length <= 0:
        return 0.0
    efficiency = shortest_path_length / max(path_length, shortest_path_length)
    return soft_success * efficiency


def aggregate_metrics(results: Sequence[EpisodeResult]) -> dict:
    n = len(results)
    if n == 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "spl": 0.0,
            "avg_steps": 0.0,
            "avg_elapsed_seconds": 0.0,
            "dynamic_memory_accuracy": 0.0,
            "fixed_memory_accuracy": 0.0,
        }

    success_rate = sum(1 for r in results if r.success) / n
    spl_avg = sum(spl(r.success, r.path_length, r.shortest_path_length) for r in results) / n
    avg_steps = sum(r.steps for r in results) / n
    avg_elapsed_seconds = sum(r.elapsed_seconds for r in results) / n

    dyn_correct = sum(r.dynamic_memory_correct for r in results)
    dyn_total = sum(r.dynamic_memory_total for r in results)
    fixed_correct = sum(r.fixed_memory_correct for r in results)
    fixed_total = sum(r.fixed_memory_total for r in results)

    return {
        "episodes": n,
        "success_rate": success_rate,
        "spl": spl_avg,
        "avg_steps": avg_steps,
        "avg_elapsed_seconds": avg_elapsed_seconds,
        "dynamic_memory_accuracy": safe_div(dyn_correct, dyn_total),
        "fixed_memory_accuracy": safe_div(fixed_correct, fixed_total),
    }
