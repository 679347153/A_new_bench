"""
Validate SD-OVON Outputs

自动验证 SD-OVON 流程的输出质量。
"""

from typing import Dict, List, Any, Optional
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig


class OutputValidator:
    """输出验证工具"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.validations = []

    def validate_layout(
        self,
        layout: Dict[str, Any],
    ) -> Dict[str, Any]:
        """验证生成的布局"""
        issues = []
        warnings = []

        objects = layout.get("objects", [])
        stats = layout.get("sd_ovon_stats", {})

        # 1. 检查对象数量
        total = stats.get("total_objects", len(objects))
        placed = stats.get("placed_count", 0)
        failed = stats.get("failed_count", 0)

        if placed + failed != total:
            issues.append(f"Stats mismatch: {placed} + {failed} != {total}")

        # 2. 检查位置有效性
        invalid_positions = 0
        for obj in objects:
            pos = obj.get("position", [])
            if not isinstance(pos, list) or len(pos) != 3:
                invalid_positions += 1
            else:
                for coord in pos:
                    if not isinstance(coord, (int, float)) or np.isnan(coord):
                        invalid_positions += 1
                        break

        if invalid_positions > 0:
            warnings.append(f"{invalid_positions} objects have invalid positions")

        # 3. 检查碰撞
        collision_count = self._detect_collisions(objects)
        if collision_count > 0:
            warnings.append(f"{collision_count} potential collisions detected")

        # 4. 失败率检查
        if total > 0 and failed / total > 0.35:
            warnings.append(f"High failure rate: {failed}/{total} ({100*failed/total:.1f}%)")

        success_rate = placed / max(total, 1)

        report = {
            "is_valid": len(issues) == 0,
            "total_issues": len(issues),
            "total_warnings": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "metrics": {
                "total_objects": total,
                "placed_count": placed,
                "failed_count": failed,
                "success_rate": float(success_rate),
                "collision_count": collision_count,
                "invalid_positions": invalid_positions,
            },
        }
        return report

    def _detect_collisions(self, objects: List[Dict[str, Any]]) -> int:
        """简单的碰撞检测"""
        collision_count = 0
        cfg = self.config.SEMANTIC_PLACEMENT

        for i, obj1 in enumerate(objects):
            pos1 = obj1.get("position", [0, 0, 0])
            for obj2 in objects[i+1:]:
                pos2 = obj2.get("position", [0, 0, 0])

                dx = float(pos1[0]) - float(pos2[0])
                dz = float(pos1[2]) - float(pos2[2])
                dist = np.sqrt(dx ** 2 + dz ** 2)

                if dist < cfg["min_distance_between_objects"]:
                    collision_count += 1

        return collision_count


def validate_sd_ovon_outputs(
    scene_name: str,
    layout_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    便捷函数：验证 SD-OVON 输出

    Args:
        scene_name: 场景名称
        layout_file: 布局文件路径（若无则自动查找）
        output_dir: 输出目录

    Returns:
        验证报告
    """
    config = SDOVONConfig()
    validator = OutputValidator(config)

    # 加载布局
    if layout_file is None:
        layouts_dir = config.get_output_dir("layouts")
        candidates = [f for f in os.listdir(layouts_dir) if scene_name in f and f.endswith(".json")]
        if candidates:
            layout_file = os.path.join(layouts_dir, sorted(candidates)[-1])

    if layout_file is None:
        return {"valid": False, "error": "No layout file found"}

    with open(layout_file, "r") as f:
        layout = json.load(f)

    # 验证
    report = validator.validate_layout(layout)

    # 保存报告
    if output_dir is None:
        output_dir = config.get_output_dir("reports")

    report_file = os.path.join(output_dir, f"{scene_name}_validation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OutputValidator] Validation report saved to: {report_file}")
    print(f"  Valid: {report['is_valid']}")
    print(f"  Issues: {report['total_issues']}, Warnings: {report['total_warnings']}")

    return report
