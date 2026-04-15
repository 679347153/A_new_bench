"""
Mock Room Recommender

当真实推荐查询缺失时，为每个对象生成虚拟推荐房间。
保证 pipeline 不被阻塞，并支持优雅降级。
"""

from typing import Dict, List, Any, Optional
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig


class MockRoomRecommender:
    """虚拟房间推荐生成器"""

    def __init__(self, config: SDOVONConfig):
        self.config = config

    def generate_mock_recommendations(
        self,
        scene_name: str,
        objects: List[Dict[str, Any]],
        available_rooms: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        为对象生成虚拟房间推荐

        Args:
            scene_name: 场景名称
            objects: 对象列表
            available_rooms: 可用房间列表（如无则生成虚拟）

        Returns:
            {
                "object_name": {
                    "recommended_rooms": [
                        {"rank": 0, "region_id": 2, "room_center": [...], "room_aabb": {...}, "confidence": 0.8}
                    ]
                }
            }
        """
        if available_rooms is None:
            available_rooms = self._generate_virtual_rooms()

        recommendations = {}

        for obj in objects:
            obj_name = obj.get("model_id", f"obj_{obj.get('id', 0)}")

            # 为每个对象选择适合的房间
            recommended = self._select_rooms_for_object(obj_name, available_rooms)
            recommendations[obj_name] = {"recommended_rooms": recommended}

        return recommendations

    def _generate_virtual_rooms(self, num_rooms: int = 5) -> List[Dict[str, Any]]:
        """生成虚拟房间信息"""
        rooms = []

        room_types = ["bedroom", "living_room", "kitchen", "office", "hallway"]

        for i in range(num_rooms):
            room = {
                "region_id": i,
                "room_center": [
                    float(np.random.uniform(-5, 5)),
                    float(np.random.uniform(0, 3)),
                    float(np.random.uniform(-5, 5)),
                ],
                "bounding_box": {
                    "min": [float(np.random.uniform(-8, -2)), 0, float(np.random.uniform(-8, -2))],
                    "max": [float(np.random.uniform(2, 8)), float(np.random.uniform(2, 4)), float(np.random.uniform(2, 8))],
                },
                "room_type": room_types[i % len(room_types)],
            }
            rooms.append(room)

        return rooms

    def _select_rooms_for_object(
        self,
        object_name: str,
        available_rooms: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """为对象选择适合的房间"""
        # 启发式规则：某些对象倾向于特定房间类型
        preferences = {
            "clock": ["bedroom", "office", "living_room"],
            "book": ["office", "bedroom", "living_room"],
            "vase": ["living_room", "hallway", "bedroom"],
            "camera": ["office", "bedroom"],
            "statue": ["living_room", "office"],
            "bottle": ["kitchen", "dining_room"],
        }

        obj_key = object_name.lower()
        preferred_types = None

        for keyword, types in preferences.items():
            if keyword in obj_key:
                preferred_types = types
                break

        # 排序房间：优先选择偏好类型的房间
        ranked_rooms = []

        for rank, room in enumerate(available_rooms):
            room_type = room.get("room_type", "unknown")
            confidence = 0.5

            if preferred_types and room_type in preferred_types:
                confidence = 0.8 - rank * 0.1

            ranked_rooms.append({
                "rank": rank,
                "region_id": room.get("region_id", -1),
                "room_center": room.get("room_center", [0, 0, 0]),
                "room_aabb": room.get("bounding_box", {}),
                "confidence": float(confidence),
            })

        # 按置信度降序排列，返回Top 3
        ranked_rooms.sort(key=lambda x: -x["confidence"])
        return ranked_rooms[:3]


def generate_mock_recommendations(
    scene_name: str,
    objects: List[Dict[str, Any]],
    recommendations_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：生成虚拟推荐并为每个对象保存 {object}_rooms.json

    Args:
        scene_name: 场景名称
        objects: 对象列表
        recommendations_dir: 推荐目录（与 query_rooms_for_objects 兼容）

    Returns:
        生成的推荐数量
    """
    config = SDOVONConfig()
    recommender = MockRoomRecommender(config)

    all_recommendations = recommender.generate_mock_recommendations(scene_name, objects)

    if recommendations_dir is None:
        recommendations_dir = os.path.join("results", "scene_info", scene_name)

    os.makedirs(recommendations_dir, exist_ok=True)

    count = 0
    for obj_name, recommendations_data in all_recommendations.items():
        output_file = os.path.join(recommendations_dir, f"{obj_name}_rooms.json")
        with open(output_file, "w") as f:
            json.dump(recommendations_data, f, indent=2)
        count += 1
        print(f"  Generated: {output_file}")

    print(f"[MockRoomRecommender] Generated {count} recommendation files for scene {scene_name}")

    return count
