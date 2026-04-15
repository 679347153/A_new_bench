from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
import json

TaskType = Literal["open_vocab", "image_goal", "language_goal"]


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class Subtask:
    subtask_id: str
    task_type: TaskType
    target_object: str
    prompt: str
    image_path: Optional[str] = None
    success_radius: float = 1.2


@dataclass
class SceneState:
    state_id: str
    time_index: int
    fixed_objects: List[str]
    movable_objects: List[str]
    transition_hint: Dict[str, object]


@dataclass
class Episode:
    episode_id: str
    split: Literal["train", "val"]
    scene_name: str
    scene_state: SceneState
    seed: int
    start_pose: Pose
    max_steps: int
    subtasks: List[Subtask]
    metadata: Dict[str, object]


@dataclass
class TrajectoryStep:
    t: int
    position: Pose
    action: str
    completed_subtask_ids: List[str]


@dataclass
class Trajectory:
    episode_id: str
    steps: List[TrajectoryStep]
    finished: bool
    finish_reason: str
    elapsed_seconds: float


@dataclass
class SplitManifest:
    version: str
    seed: int
    train_scenes: List[str]
    val_scenes: List[str]


def _assert_subtask_balance(subtasks: List[Subtask]) -> None:
    counts = {
        "open_vocab": 0,
        "image_goal": 0,
        "language_goal": 0,
    }
    for item in subtasks:
        counts[item.task_type] += 1

    # "Balanced" here means no type exceeds another by more than 1.
    spread = max(counts.values()) - min(counts.values())
    if spread > 1:
        raise ValueError(f"Subtask type distribution is unbalanced: {counts}")


def validate_episode(ep: Episode) -> None:
    if not (5 <= len(ep.subtasks) <= 10):
        raise ValueError("Each episode must contain 5 to 10 subtasks")
    if ep.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    _assert_subtask_balance(ep.subtasks)


def to_json_dict(obj: object) -> Dict[str, object]:
    return asdict(obj)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_json_dict(obj), f, ensure_ascii=False, indent=2)


def read_split_manifest(path: Path) -> SplitManifest:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SplitManifest(
        version=str(data["version"]),
        seed=int(data["seed"]),
        train_scenes=list(data["train_scenes"]),
        val_scenes=list(data["val_scenes"]),
    )
