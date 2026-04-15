"""
Physics Stabilizer Heuristic (Phase 3.4 补强)

启发式物理稳定性检查。
预留接口用于接入真实 habitat-sim 物理模拟。
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig


class PhysicsStabilizerHeuristic:
    """启发式物理稳定性检查器"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.checks = []

    def check_stability(
        self,
        placements: List[Dict[str, Any]],
        receptacles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        对放置结果进行物理稳定性检查

        Args:
            placements: 放置列表 [{"object_id": ..., "position": [...], "receptacle_id": ...}]
            receptacles: receptacle 列表

        Returns:
            {
                "total_checked": int,
                "stable_count": int,
                "unstable_count": int,
                "stability_checks": [
                    {
                        "object_id": int,
                        "is_stable": bool,
                        "stability_score": float,
                        "contact_area": float,
                        "center_of_mass_offset": float,
                        "adjustment_needed": bool
                    }
                ]
            }
        """
        cfg = self.config.PHYSICS_STABILIZATION
        checks = []
        stable_count = 0

        for placement in placements:
            obj_id = placement.get("object_id", 0)
            rec_id = placement.get("receptacle_id", "unknown")

            # 获取 receptacle 信息
            receptacle = self._find_receptacle(rec_id, receptacles)
            if not receptacle:
                # fallback
                check = self._check_stability_heuristic(placement, None, cfg)
            else:
                check = self._check_stability_heuristic(placement, receptacle, cfg)

            check["object_id"] = obj_id
            checks.append(check)

            if check["is_stable"]:
                stable_count += 1

        report = {
            "total_checked": len(placements),
            "stable_count": stable_count,
            "unstable_count": len(placements) - stable_count,
            "stability_checks": checks,
            "statistics": {
                "stability_rate": stable_count / max(len(placements), 1),
                "average_stability_score": np.mean([c["stability_score"] for c in checks]),
            },
        }
        return report

    def _check_stability_heuristic(
        self,
        placement: Dict[str, Any],
        receptacle: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        基于启发式规则进行稳定性评估

        评估维度：
        - 接触面积（越大越稳定）
        - 重心偏移（越小越稳定）
        - 悬空比例（越小越稳定）
        """
        pos = placement.get("position", [0, 0, 0])

        # 1. 接触面积启发式
        contact_area = np.random.uniform(0.01, 0.5)  # mock

        # 2. 重心偏移启发式
        center_of_mass_offset = np.random.uniform(0, 0.15)  # mock

        # 3. 悬空比例启发式
        overhang_ratio = np.random.uniform(0, 0.3)  # mock

        # 组合稳定性分数
        stability_score = 1.0
        stability_score -= 0.2 * min(1.0, center_of_mass_offset / cfg["center_of_mass_offset_max"])
        stability_score -= 0.2 * min(1.0, overhang_ratio / cfg.get("max_overhang", 0.15))

        if contact_area < cfg["contact_area_threshold"]:
            stability_score -= 0.3

        stability_score = max(0.0, min(1.0, stability_score))

        # 判断是否稳定
        is_stable = (
            stability_score >= cfg["stability_score_threshold"]
            and center_of_mass_offset <= cfg["center_of_mass_offset_max"]
        )

        return {
            "is_stable": is_stable,
            "stability_score": float(stability_score),
            "contact_area": float(contact_area),
            "center_of_mass_offset": float(center_of_mass_offset),
            "overhang_ratio": float(overhang_ratio),
            "adjustment_needed": not is_stable,
        }

    def _find_receptacle(
        self,
        receptacle_id: str,
        receptacles: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """查找 receptacle"""
        for rec in receptacles:
            if rec.get("receptacle_id") == receptacle_id:
                return rec
        return None

    def suggest_adjustments(
        self,
        unstable_placements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        为不稳定的放置建议调整

        Returns:
            {object_id, adjustment_vector: [dx, dy, dz]}
        """
        suggestions = []

        for placement in unstable_placements:
            # 简化：向上移动一点
            adjustment = {
                "object_id": placement.get("object_id", 0),
                "adjustment_vector": [0.0, 0.1, 0.0],  # mock 向上调整
                "confidence": 0.5,
            }
            suggestions.append(adjustment)

        return suggestions


def check_placement_stability(
    scene_name: str,
    placements: List[Dict[str, Any]],
    receptacles: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：检查放置稳定性并保存报告

    Args:
        scene_name: 场景名称
        placements: 放置列表
        receptacles: receptacle 列表（可选）
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    stabilizer = PhysicsStabilizerHeuristic(config)

    if receptacles is None:
        receptacles = []

    report = stabilizer.check_stability(placements, receptacles)

    if output_dir is None:
        output_dir = config.get_output_dir("stability")

    output_file = os.path.join(output_dir, f"{scene_name}_stability_check.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[PhysicsStabilizerHeuristic] Stability check completed")
    print(f"  Stable: {report['stable_count']}/{report['total_checked']}")
    print(f"  Average score: {report['statistics']['average_stability_score']:.3f}")

    return output_file
