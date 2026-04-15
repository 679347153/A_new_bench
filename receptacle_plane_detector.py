"""
Receptacle Plane Detector (Phase 3.3)

从容纳实例及场景 mesh 中检测可放置平面。
当前实现基于 bbox 的简化版本；后续可接入真实 EM 平面拟合。
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig
from pipeline_schema import SchemaValidator


class ReceptaclePlaneDetector:
    """Receptacle 平面检测器"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.receptacles = {}

    def detect_receptacles(
        self,
        scene_name: str,
        instances: List[Dict[str, Any]],
        room_bboxes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        检测容纳表面（receptacles）及其可放置平面

        Args:
            scene_name: 场景名称
            instances: 融合后的实例列表
            room_bboxes: 房间 bbox 列表（可选，用于提升容纳面检测）

        Returns:
            {
                "scene_name": str,
                "total_receptacles": int,
                "receptacles": [
                    {
                        "receptacle_id": str,
                        "receptacle_type": str,
                        "center": [x,y,z],
                        "normal": [nx,ny,nz],
                        "area": float,
                        "height": float,
                        "placement_planes": [
                            {
                                "plane_id": int,
                                "center": [x,y,z],
                                "normal": [nx,ny,nz],
                                "polygon_vertices": [[x,y,z], ...],
                                "area": float
                            }
                        ]
                    }
                ],
                "object_receptacle_map": {object_id -> [receptacle_ids]}
            }
        """
        cfg = self.config.RECEPTACLE_DETECTION
        receptacles = []
        receptacle_id_counter = 0

        # Step 1: 从实例中识别 receptacle 类型的对象
        receptacle_candidates = self._identify_receptacle_candidates(instances, cfg["supported_types"])

        # Step 2: 为每个候选生成可放置平面
        for candidate in receptacle_candidates:
            planes = self._generate_placement_planes(candidate, cfg)

            if planes:  # 只保存有有效平面的 receptacle
                receptacle = {
                    "receptacle_id": f"rec_{receptacle_id_counter:03d}",
                    "receptacle_type": candidate.get("semantic_label", "unknown"),
                    "instance_id": candidate.get("instance_id"),
                    "region_id": -1,  # 后续可从房间关联
                    "center": candidate.get("center_3d"),
                    "normal": [0.0, 1.0, 0.0],  # 默认向上
                    "area": candidate.get("bounding_box", {}).get("area", 1.0),
                    "height": self._estimate_height(candidate),
                    "bounding_box": candidate.get("bounding_box", {}),
                    "placement_planes": planes,
                }

                # 校验
                is_valid, msg = SchemaValidator.validate_receptacle(receptacle)
                if is_valid:
                    receptacles.append(receptacle)
                    receptacle_id_counter += 1
                else:
                    print(f"[Warning] Invalid receptacle: {msg}")

        # Step 3: 生成后续使用的对象-receptacle 映射
        object_receptacle_map = self._build_object_receptacle_map(receptacles, instances)

        report = {
            "scene_name": scene_name,
            "total_receptacles": len(receptacles),
            "receptacles": receptacles,
            "object_receptacle_map": object_receptacle_map,
            "statistics": {
                "receptacles_by_type": self._count_by_type(receptacles),
                "total_placement_planes": sum(len(r.get("placement_planes", [])) for r in receptacles),
            },
        }
        return report

    def _identify_receptacle_candidates(self, instances: List[Dict[str, Any]], supported_types: List[str]) -> List[Dict[str, Any]]:
        """从实例列表中识别 receptacle 类型的对象"""
        candidates = []

        for inst in instances:
            label = inst.get("semantic_label", "").lower()

            # 匹配支持的类型
            for receptacle_type in supported_types:
                if receptacle_type.lower() in label:
                    candidates.append(inst)
                    break

            # 也检查标注的 is_receptacle 字段
            if inst.get("is_receptacle", False):
                candidates.append(inst)

        return candidates

    def _generate_placement_planes(self, receptacle: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """为 receptacle 生成可放置平面"""
        planes = []

        try:
            bbox = receptacle.get("bounding_box", {})
            min_pt = bbox.get("min", [0, 0, 0])
            max_pt = bbox.get("max", [0, 0, 0])

            # 简化：生成顶部和底部平面（表面法线向上或向下）
            center_x = (min_pt[0] + max_pt[0]) / 2.0
            center_z = (min_pt[2] + max_pt[2]) / 2.0

            # 顶面
            top_y = max_pt[1]
            top_plane = {
                "plane_id": 0,
                "center": [center_x, top_y, center_z],
                "normal": [0.0, 1.0, 0.0],  # 向上
                "polygon_vertices": [
                    [min_pt[0], top_y, min_pt[2]],
                    [max_pt[0], top_y, min_pt[2]],
                    [max_pt[0], top_y, max_pt[2]],
                    [min_pt[0], top_y, max_pt[2]],
                ],
                "area": abs((max_pt[0] - min_pt[0]) * (max_pt[2] - min_pt[2])),
            }

            if top_plane["area"] >= cfg["min_surface_area"]:
                planes.append(top_plane)

            # 底面（如果 receptacle 类型是 floor 等）
            if "floor" in receptacle.get("semantic_label", "").lower():
                bottom_y = min_pt[1]
                bottom_plane = {
                    "plane_id": 1,
                    "center": [center_x, bottom_y, center_z],
                    "normal": [0.0, -1.0, 0.0],  # 向下
                    "polygon_vertices": [
                        [min_pt[0], bottom_y, min_pt[2]],
                        [min_pt[0], bottom_y, max_pt[2]],
                        [max_pt[0], bottom_y, max_pt[2]],
                        [max_pt[0], bottom_y, min_pt[2]],
                    ],
                    "area": top_plane["area"],
                }
                if bottom_plane["area"] >= cfg["min_surface_area"]:
                    planes.append(bottom_plane)

        except Exception as e:
            print(f"[Warning] Failed to generate planes: {e}")

        return planes

    def _estimate_height(self, receptacle: Dict[str, Any]) -> Optional[float]:
        """估计容纳表面的高度"""
        try:
            bbox = receptacle.get("bounding_box", {})
            max_y = bbox.get("max", [0, 0, 0])[1]
            return float(max_y)
        except Exception:
            return None

    def _build_object_receptacle_map(
        self,
        receptacles: List[Dict[str, Any]],
        instances: List[Dict[str, Any]],
    ) -> Dict[int, List[int]]:
        """
        建立对象到 receptacle 的映射

        返回：{instance_index -> [receptacle_indices]}
        """
        obj_rec_map = {}

        for obj_idx, inst in enumerate(instances):
            compatible_receptacles = []

            # 规则：相邻的 receptacle 与该对象兼容
            obj_center = np.array(inst.get("center_3d", [0, 0, 0]))

            for rec_idx, rec in enumerate(receptacles):
                rec_center = np.array(rec.get("center", [0, 0, 0]))
                distance = np.linalg.norm(obj_center - rec_center)

                # 简化：距离小于 1 米的均视为兼容（启发式）
                if distance < 1.0:
                    compatible_receptacles.append(rec_idx)

            if compatible_receptacles:
                obj_rec_map[obj_idx] = compatible_receptacles

        return obj_rec_map

    def _count_by_type(self, receptacles: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计各类型 receptacle 的数量"""
        count = {}
        for rec in receptacles:
            rec_type = rec.get("receptacle_type", "unknown")
            count[rec_type] = count.get(rec_type, 0) + 1
        return count


def detect_receptacles(
    scene_name: str,
    instances: List[Dict[str, Any]],
    room_bboxes: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：检测 receptacle 并保存结果

    Args:
        scene_name: 场景名称
        instances: 融合后的实例列表
        room_bboxes: 房间 bbox （可选）
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    detector = ReceptaclePlaneDetector(config)

    report = detector.detect_receptacles(scene_name, instances, room_bboxes)

    if output_dir is None:
        output_dir = config.get_output_dir("receptacles")

    output_file = os.path.join(output_dir, f"{scene_name}_receptacles.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[ReceptaclePlaneDetector] Detected {report['total_receptacles']} receptacles")
    print(f"  Types: {report['statistics']['receptacles_by_type']}")
    print(f"  Total placement planes: {report['statistics']['total_placement_planes']}")

    return output_file
