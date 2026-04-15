"""
Semantics-Aware Placer (Phase 3.4)

在 receptacle 平面上执行语义感知放置。
支持最小距离约束、碰撞检测、失败回退。
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import os
import numpy as np
import math
from sd_ovon_config import SDOVONConfig
from semantics_relation_model import SemanticsRelationModel
from pipeline_schema import SchemaValidator


class SemanticsAwarePlacer:
    """语义感知放置器"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.semantic_model = SemanticsRelationModel(config)
        self.placements = []
        self.failed_objects = []

    def place_objects_semantically(
        self,
        scene_name: str,
        objects: List[Dict[str, Any]],
        receptacles: List[Dict[str, Any]],
        region_label: str = "unknown",
    ) -> Dict[str, Any]:
        """
        主入口：基于语义关系在 receptacle 平面上放置对象

        Args:
            scene_name: 场景名称
            objects: 待放置的对象列表 [{"id": int, "model_id": str, "position": [...]}]
            receptacles: receptacle 列表
            region_label: 当前区域标签

        Returns:
            {
                "scene_name": str,
                "total_objects": int,
                "placed_count": int,
                "failed_count": int,
                "placements": [
                    {
                        "object_id": int,
                        "model_id": str,
                        "position": [x,y,z],
                        "rotation": [0,0,yaw],
                        "receptacle_id": str,
                        "semantic_confidence": float,
                        "source": "semantics_aware",
                        ...
                    }
                ],
                "failed_objects": [...]
            }
        """
        cfg = self.config.SEMANTIC_PLACEMENT
        placements = []
        failed_objects = []

        for obj in objects:
            obj_id = obj.get("id", 0)
            model_id = obj.get("model_id", "unknown")

            # Step 1: 获取语义排名
            ranked_receptacles = self.semantic_model.rank_placement_candidates(model_id, region_label, receptacles)

            # Step 2: 尝试在排名最高的 receptacle 上放置
            placed = False
            for rec_idx, confidence in ranked_receptacles:
                if rec_idx >= len(receptacles):
                    continue

                receptacle = receptacles[rec_idx]
                placement = self._place_on_receptacle(
                    obj,
                    receptacle,
                    confidence,
                    placements,
                    cfg,
                )

                if placement is not None:
                    placements.append(placement)
                    placed = True
                    break

            if not placed:
                # 失败：标记为需要人工修正
                failed_obj = {
                    "object_id": obj_id,
                    "model_id": model_id,
                    "reason": "no_suitable_receptacle",
                    "attempted_receptacles": len(ranked_receptacles),
                }
                failed_objects.append(failed_obj)

        report = {
            "scene_name": scene_name,
            "total_objects": len(objects),
            "placed_count": len(placements),
            "failed_count": len(failed_objects),
            "placements": placements,
            "failed_objects": failed_objects,
            "statistics": {
                "success_rate": len(placements) / max(len(objects), 1),
            },
        }
        return report

    def _place_on_receptacle(
        self,
        obj: Dict[str, Any],
        receptacle: Dict[str, Any],
        semantic_confidence: float,
        existing_placements: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        尝试在特定 receptacle 上放置对象

        Returns:
            placement dict 或 None（失败）
        """
        try:
            plans = receptacle.get("placement_planes", [])
            if not plans:
                return None

            # 选择第一个平面（顶面）
            plane = plans[0]
            plane_center = plane.get("center", [0, 0, 0])

            # 在平面上采样候选点
            candidates = self._sample_placement_points_on_plane(
                plane,
                cfg["num_candidate_points_per_receptacle"],
            )

            # 筛选无碰撞的候选
            min_distance = cfg["min_distance_between_objects"]
            obj_profile = self._get_object_profile(obj.get("model_id", ""))

            for candidate in candidates:
                # 碰撞检查
                if not self._check_collision(candidate, obj_profile["radius"], existing_placements):
                    # 成功：返回放置结果
                    return {
                        "object_id": obj.get("id", 0),
                        "model_id": obj.get("model_id", "unknown"),
                        "position": [float(c) for c in candidate],
                        "rotation": [0.0, float(np.random.uniform(0, 360)), 0.0],
                        "receptacle_id": receptacle.get("receptacle_id", "unknown"),
                        "plane_id": plane.get("plane_id", -1),
                        "relation_type": "on",
                        "semantic_confidence": float(semantic_confidence),
                        "stability_score": 0.8,  # mock
                        "is_stable": True,
                        "source": "semantics_aware",
                    }

            # 所有候选均失败（碰撞）
            return None

        except Exception as e:
            print(f"[Warning] Failed to place object: {e}")
            return None

    def _sample_placement_points_on_plane(
        self,
        plane: Dict[str, Any],
        num_points: int,
    ) -> List[Tuple[float, float, float]]:
        """在平面上均匀采样放置候选点"""
        candidates = []
        vertices = plane.get("polygon_vertices", [])

        if not vertices or len(vertices) < 3:
            # fallback：从平面中心采样
            center = plane.get("center", [0, 0, 0])
            candidates.append(tuple(center))
            return candidates

        # 简化：从平面内随机采样
        # 使用简单的 bbox 采样（可改进为对 convex hull 采样）
        vertices_arr = np.array(vertices)
        min_pt = vertices_arr.min(axis=0)
        max_pt = vertices_arr.max(axis=0)

        for _ in range(num_points):
            x = np.random.uniform(min_pt[0], max_pt[0])
            y = np.random.uniform(min_pt[1], max_pt[1])
            z = np.random.uniform(min_pt[2], max_pt[2])
            candidates.append((x, y, z))

        return candidates

    def _check_collision(
        self,
        candidate_pos: Tuple[float, float, float],
        obj_radius: float,
        existing_placements: List[Dict[str, Any]],
    ) -> bool:
        """
        检查是否与已放置对象碰撞

        Returns:
            True 如果有碰撞，False 无碰撞
        """
        for existing in existing_placements:
            existing_pos = existing.get("position", [0, 0, 0])
            existing_profile = self._get_object_profile(existing.get("model_id", ""))

            # XZ 平面距离
            dx = float(candidate_pos[0]) - float(existing_pos[0])
            dz = float(candidate_pos[2]) - float(existing_pos[2])
            dist_xz = math.sqrt(dx ** 2 + dz ** 2)

            min_distance = obj_radius + existing_profile["radius"]

            if dist_xz < min_distance:
                return True  # 碰撞

        return False

    def _get_object_profile(self, model_id: str) -> Dict[str, float]:
        """获取对象画像"""
        keywords = self.config.OBJECT_PROFILE_KEYWORDS
        key = model_id.lower()

        for keyword, profile in keywords.items():
            if keyword in key:
                return dict(profile)

        return dict(self.config.DEFAULT_OBJECT_PROFILE)


def place_objects_semantically(
    scene_name: str,
    objects: List[Dict[str, Any]],
    receptacles: List[Dict[str, Any]],
    region_label: str = "unknown",
    output_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：执行语义感知放置并保存结果

    Args:
        scene_name: 场景名称
        objects: 对象列表
        receptacles: receptacle 列表
        region_label: 区域标签
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    placer = SemanticsAwarePlacer(config)

    report = placer.place_objects_semantically(scene_name, objects, receptacles, region_label)

    if output_dir is None:
        output_dir = config.get_output_dir("placements")

    output_file = os.path.join(output_dir, f"{scene_name}_placements_semantic.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[SemanticsAwarePlacer] Placed {report['placed_count']}/{report['total_objects']} objects")
    print(f"  Success rate: {report['statistics']['success_rate']:.1%}")

    return output_file
