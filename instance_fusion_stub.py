"""
Instance Fusion Stub (Phase 3.2)

实现 mock 实例提取、融合与误差修正接口。
当前使用 mock 实现；后续可替换为真实 G-SAM + ConceptGraphs。
"""

from typing import Dict, List, Any, Optional
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig
from pipeline_schema import SchemaValidator


class InstanceFusionStub:
    """Mock 实例融合引擎"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.instances = {}
        self.fusion_history = []

    def extract_and_fuse_instances(
        self,
        scene_name: str,
        observations: List[Dict[str, Any]],
        objects_from_scene: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        从多观测点提取并融合实例

        Args:
            scene_name: 场景名称
            observations: 观测列表
            objects_from_scene: 场景导出的已知对象列表（用于 mock）

        Returns:
            {
                "scene_name": str,
                "total_instances": int,
                "instances": [
                    {
                        "instance_id": str,
                        "semantic_label": str,
                        "confidence": float,
                        "center_3d": [x,y,z],
                        "bounding_box": {"min": [...], "max": [...]},
                        "point_count": int,
                        "is_receptacle": bool,
                        "viewpoints_observed": [int, ...]
                    }
                ],
                "statistics": {...}
            }
        """
        instances = {}

        # Step 1: Mock 实例提取（如果提供了场景对象，则使用它们作为 mock 提取结果）
        if objects_from_scene:
            for obj in objects_from_scene:
                instance_id = f"inst_{obj.get('id', 0):03d}"
                obj_center = obj.get("position", [0, 0, 0])
                obj_bbox = obj.get("bbox", {"min": [0, 0, 0], "max": [0, 0, 0]})

                instances[instance_id] = {
                    "instance_id": instance_id,
                    "semantic_label": obj.get("category", "unknown"),
                    "confidence": 0.8,  # mock 置信度
                    "center_3d": [float(x) for x in obj_center],
                    "bounding_box": {
                        "min": [float(x) for x in obj_bbox.get("min", [0, 0, 0])],
                        "max": [float(x) for x in obj_bbox.get("max", [0, 0, 0])],
                    },
                    "point_count": 100,  # mock 点数
                    "is_receptacle": False,
                    "viewpoints_observed": list(range(len(observations))),  # 全部观测视图可见
                }
        else:
            # 生成虚拟实例（当无输入对象时）
            for i in range(5):  # mock 5 个实例
                instance_id = f"inst_virtual_{i:03d}"
                instances[instance_id] = {
                    "instance_id": instance_id,
                    "semantic_label": self._random_label(),
                    "confidence": np.random.uniform(0.7, 0.95),
                    "center_3d": [
                        float(np.random.uniform(-2, 2)),
                        float(np.random.uniform(0, 2)),
                        float(np.random.uniform(-2, 2)),
                    ],
                    "bounding_box": {
                        "min": [float(np.random.uniform(-3, -1)), 0, float(np.random.uniform(-3, -1))],
                        "max": [float(np.random.uniform(1, 3)), float(np.random.uniform(1, 3)), float(np.random.uniform(1, 3))],
                    },
                    "point_count": int(np.random.randint(50, 500)),
                    "is_receptacle": i < 2,  # 前 2 个视为容纳容器
                    "viewpoints_observed": list(range(min(3, len(observations)))),
                }

        # Step 2: Mock 融合（这里简化为直接返回）
        instances_list = list(instances.values())

        # Step 3: Mock 纠错（检测重叠的实例）
        corrected_instances = self._apply_error_correction(instances_list)

        report = {
            "scene_name": scene_name,
            "total_instances": len(corrected_instances),
            "instances": corrected_instances,
            "statistics": {
                "receptacle_count": sum(1 for inst in corrected_instances if inst["is_receptacle"]),
                "non_receptacle_count": sum(1 for inst in corrected_instances if not inst["is_receptacle"]),
                "average_confidence": np.mean([inst["confidence"] for inst in corrected_instances]),
            },
        }
        return report

    def _random_label(self) -> str:
        """生成随机标签（用于 mock）"""
        labels = ["table", "chair", "shelf", "vase", "statue", "camera", "cup", "book"]
        return np.random.choice(labels)

    def _apply_error_correction(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        应用纠错机制：识别并处理重叠实例

        当两个实例有显著空间重叠时，保留置信度更高的实例。
        """
        corrected = []
        skip_ids = set()

        for i, inst_i in enumerate(instances):
            if i in skip_ids:
                continue

            for j, inst_j in enumerate(instances):
                if i >= j or j in skip_ids:
                    continue

                # 检查空间重叠（IoU）
                iou = self._compute_bbox_iou(inst_i["bounding_box"], inst_j["bounding_box"])
                if iou > 0.5:  # 显著重叠阈值
                    # 保留置信度高的，移除置信度低的
                    if inst_i["confidence"] >= inst_j["confidence"]:
                        skip_ids.add(j)
                    else:
                        skip_ids.add(i)
                        break

            if i not in skip_ids:
                corrected.append(inst_i)

        return corrected

    def _compute_bbox_iou(self, bbox1: Dict[str, Any], bbox2: Dict[str, Any]) -> float:
        """计算两个 BBOX 的 IoU"""
        try:
            min1, max1 = bbox1["min"], bbox1["max"]
            min2, max2 = bbox2["min"], bbox2["max"]

            # 交集
            inter_min = [max(min1[i], min2[i]) for i in range(3)]
            inter_max = [min(max1[i], max2[i]) for i in range(3)]

            if any(inter_max[i] < inter_min[i] for i in range(3)):
                return 0.0

            inter_vol = np.prod([inter_max[i] - inter_min[i] for i in range(3)])

            # 并集
            union_vol = (
                np.prod([max1[i] - min1[i] for i in range(3)])
                + np.prod([max2[i] - min2[i] for i in range(3)])
                - inter_vol
            )

            return inter_vol / union_vol if union_vol > 0 else 0.0
        except Exception:
            return 0.0


def extract_and_fuse_instances(
    scene_name: str,
    observations: List[Dict[str, Any]],
    objects_from_scene: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：提取、融合并保存实例

    Args:
        scene_name: 场景名称
        observations: 观测列表
        objects_from_scene: 场景对象列表（用于 mock）
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    fusion = InstanceFusionStub(config)

    report = fusion.extract_and_fuse_instances(scene_name, observations, objects_from_scene)

    if output_dir is None:
        output_dir = config.get_output_dir("instances")

    output_file = os.path.join(output_dir, f"{scene_name}_instances.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[InstanceFusionStub] Extracted and fused {report['total_instances']} instances")
    print(f"  Receptacles: {report['statistics']['receptacle_count']}")
    print(f"  Avg confidence: {report['statistics']['average_confidence']:.3f}")

    return output_file
