"""
Semantics Relation Model (Phase 3.4)

计算对象与区域、对象与容纳表面的语义关联概率。
支持 rule-based 和 mock LLM 两个后端。
"""

from typing import Dict, List, Any, Tuple, Optional
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig


class SemanticsRelationModel:
    """语义关系模型"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.relations = config.SEMANTIC_RELATIONS
        self.default_relations = self._build_default_relations()

    def compute_object_receptacle_probability(
        self,
        object_label: str,
        receptacle_label: str,
    ) -> Tuple[str, float]:
        """
        计算 P(object | receptacle) - 对象在特定容纳表面上的合理性

        Returns:
            (relation_type, probability)
            relation_type: "on", "near", "inside"
            probability: [0, 1]
        """
        obj_key = object_label.lower()

        # 查表
        if obj_key in self.relations:
            for rec_label, rel_type, prob in self.relations[obj_key]:
                if rec_label.lower() == receptacle_label.lower():
                    return rel_type, prob

        # 默认关系
        return self._get_default_relation(object_label, receptacle_label)

    def compute_object_region_probability(
        self,
        object_label: str,
        region_label: str,
    ) -> float:
        """
        计算 P(object | region) - 对象在特定区域的合理性

        Args:
            object_label: 对象类别（如 "clock"）
            region_label: 区域类别（如 "bedroom"）

        Returns:
            probability [0, 1]
        """
        # 简化规则：某些对象更倾向于特定区域
        #例如：clock 倾向于 bedroom/office；book 倾向于 library/office

        obj_key = object_label.lower()
        reg_key = region_label.lower()

        preferences = {
            "clock": {"bedroom": 0.8, "living_room": 0.6, "office": 0.7, "kitchen": 0.3},
            "book": {"office": 0.9, "library": 0.95, "bedroom": 0.6, "living_room": 0.4},
            "vase": {"living_room": 0.8, "entryway": 0.7, "bedroom": 0.4},
            "statue": {"living_room": 0.85, "bedroom": 0.5, "office": 0.4},
            "camera": {"office": 0.8, "bedroom": 0.5},
            "bottle": {"kitchen": 0.9, "dining_room": 0.7, "office": 0.5},
        }

        if obj_key in preferences:
            return preferences[obj_key].get(reg_key, 0.3)

        # 默认值
        return 0.4

    def compute_joint_probability(
        self,
        object_label: str,
        region_label: str,
        receptacle_label: str,
    ) -> float:
        """
        计算 P(object | region, receptacle) = P(object | region) * P(object | receptacle_rel)

        联合概率用于最终的放置决策排名。

        Args:
            object_label: 对象类别
            region_label: 区域类别
            receptacle_label: 容纳表面类别

        Returns:
            联合概率 [0, 1]
        """
        p_region = self.compute_object_region_probability(object_label, region_label)
        rel_type, p_receptacle = self.compute_object_receptacle_probability(object_label, receptacle_label)

        # 简单乘法（可加权）
        joint_prob = p_region * p_receptacle
        return float(joint_prob)

    def _get_default_relation(
        self,
        object_label: str,
        receptacle_label: str,
    ) -> Tuple[str, float]:
        """获取默认关系（当无表项时）"""
        # 启发式规则
        obj_key = object_label.lower()
        rec_key = receptacle_label.lower()

        # 小对象倾向于放在表面上
        small_objects = ["clock", "camera", "vase", "bottle", "statue"]
        large_receptacles = ["table", "shelf", "bed", "desk"]

        if obj_key in small_objects and rec_key in large_receptacles:
            return ("on", 0.6)

        # 对象靠近地面
        if rec_key == "floor":
            return ("near", 0.3)

        # 默认
        return ("on", 0.5)

    def _build_default_relations(self) -> Dict[str, List[Tuple[str, str, float]]]:
        """构建默认关系表"""
        defaults = {
            "default": [
                ("table", "on", 0.6),
                ("shelf", "on", 0.5),
                ("floor", "near", 0.3),
            ]
        }
        return defaults

    def rank_placement_candidates(
        self,
        object_label: str,
        region_label: str,
        candidate_receptacles: List[Dict[str, Any]],
    ) -> List[Tuple[int, float]]:
        """
        对候选 receptacle 列表按语义合理性排名

        Args:
            object_label: 对象类别
            region_label: 区域类别
            candidate_receptacles: receptacle 候选列表 [{"receptacle_id": ..., "receptacle_type": ...}]

        Returns:
            [(receptacle_index, score), ...] - 按分数降序排列
        """
        scores = []

        for idx, rec in enumerate(candidate_receptacles):
            rec_label = rec.get("receptacle_type", "unknown")
            score = self.compute_joint_probability(object_label, region_label, rec_label)
            scores.append((idx, score))

        # 按分数降序排列
        scores.sort(key=lambda x: -x[1])
        return scores


class SemanticRelationLLMBackend:
    """
    可选的 LLM 后端，调用 LLM 查询语义关系
    当前为 stub；后续可接入 Qwen/GPT
    """

    def query_semantic_relations(self, object_label: str) -> List[Tuple[str, str, float]]:
        """
        使用 LLM 查询对象的容纳表面与区域偏好

        Returns:
            [(receptacle_type, relation, probability), ...]
        """
        # stub 实现：返回空或使用规则默认值
        return []


def compute_semantic_scores(
    objects: List[Dict[str, Any]],
    receptacles: List[Dict[str, Any]],
    region_label: str = "unknown",
    output_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：计算所有对象-receptacle 对的语义分数并保存

    Args:
        objects: 对象列表
        receptacles: receptacle 候选列表
        region_label: 区域标签
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    model = SemanticsRelationModel(config)

    scores = {}
    for obj_idx, obj in enumerate(objects):
        obj_label = obj.get("model_id", "unknown")
        ranked_receptacles = model.rank_placement_candidates(obj_label, region_label, receptacles)
        scores[f"obj_{obj_idx}"] = ranked_receptacles

    if output_dir is None:
        output_dir = config.get_output_dir("semantics")

    output_file = os.path.join(output_dir, "semantic_scores.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"[SemanticsRelationModel] Computed scores for {len(objects)} objects")

    return output_file
