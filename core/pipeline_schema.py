"""
Pipeline Schema Definition and Validation

定义 SD-OVON 流程中的所有 JSON 数据结构规范，作为跨模块唯一契约。
- Observation: 多视角观察数据
- Instance: 融合后的语义实例
- Receptacle: 容纳表面及其可放置平面
- Placement: 放置决策与结果
- PipelineReport: 全流程执行报告
"""

from typing import Dict, List, Any, Optional, Tuple
import json


class PipelineSchema:
    """中央 schema 库，包含所有主要数据结构的字段定义与验证规则。"""

    @staticmethod
    def observation_schema() -> Dict[str, Any]:
        """
        Observation 结构：单个观察点的多角度观测
        """
        return {
            "type": "object",
            "properties": {
                "viewpoint_id": {"type": "integer"},
                "camera_pos": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "camera_lookat": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "camera_up": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "fov_degrees": {"type": "number", "minimum": 30, "maximum": 120},
                "depth_image": {"type": ["string", "null"]},  # 相对路径或 None
                "rgb_image": {"type": ["string", "null"]},
                "instance_mask": {"type": ["string", "null"]},
                "extracted_instances": {"type": "array", "items": {"type": "string"}},  # instance_ids
                "coverage_score": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["viewpoint_id", "camera_pos", "camera_lookat", "extracted_instances"],
        }

    @staticmethod
    def instance_schema() -> Dict[str, Any]:
        """
        Instance 结构：融合后的语义物体实例
        """
        return {
            "type": "object",
            "properties": {
                "instance_id": {"type": "string"},
                "semantic_label": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "center_3d": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "bounding_box": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                        "max": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                    },
                },
                "point_count": {"type": "integer"},
                "is_receptacle": {"type": "boolean"},
                "viewpoints_observed": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["instance_id", "semantic_label", "center_3d"],
        }

    @staticmethod
    def receptacle_schema() -> Dict[str, Any]:
        """
        Receptacle 结构：容纳物体的表面（如桌面、架子、地面等）
        """
        return {
            "type": "object",
            "properties": {
                "receptacle_id": {"type": "string"},
                "receptacle_type": {"type": "string"},  # "table", "shelf", "floor", "wall", etc.
                "instance_id": {"type": ["string", "null"]},  # 对应的 instance，若无则为 null
                "region_id": {"type": "integer"},
                "center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "area": {"type": "number", "minimum": 0.01},
                "height": {"type": ["number", "null"]},  # 对象高度（若已知）
                "bounding_box": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                        "max": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                    },
                },
                "placement_planes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "plane_id": {"type": "integer"},
                            "center": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                            "normal": {"type": "array", "items": {"type": "number"}, "minItems": 3},
                            "polygon_vertices": {"type": "array"},  # [[x,y,z], ...]
                            "area": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["receptacle_id", "receptacle_type", "center", "normal"],
        }

    @staticmethod
    def placement_schema() -> Dict[str, Any]:
        """
        Placement 结构：单个对象的放置决策
        """
        return {
            "type": "object",
            "properties": {
                "object_id": {"type": "integer"},
                "model_id": {"type": "string"},
                "position": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "rotation": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "receptacle_id": {"type": ["string", "null"]},
                "plane_id": {"type": ["integer", "null"]},
                "relation_type": {"type": "string"},  # "on", "near", "inside"
                "semantic_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "stability_score": {"type": "number", "minimum": 0, "maximum": 1},
                "conflicts": {"type": "array", "items": {"type": "integer"}},  # object_ids 冲突列表
                "is_stable": {"type": "boolean"},
                "adjustment_needed": {"type": "boolean"},
                "adjustment_vector": {"type": ["array", "null"]},
                "source": {"type": "string"},  # "semantics_aware", "auto_place", "manual", etc.
            },
            "required": ["object_id", "model_id", "position", "rotation", "source"],
        }

    @staticmethod
    def layout_schema() -> Dict[str, Any]:
        """
        Layout 结构：完整的场景布局（扩展自 sample_and_place_objects 格式）
        """
        return {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "scene_name": {"type": "string"},
                "timestamp": {"type": "number"},
                "objects": {"type": "array", "items": PipelineSchema.placement_schema()},
                "sd_ovon_stats": {
                    "type": "object",
                    "properties": {
                        "total_objects": {"type": "integer"},
                        "placed_count": {"type": "integer"},
                        "failed_count": {"type": "integer"},
                        "stable_count": {"type": "integer"},
                        "unstable_count": {"type": "integer"},
                        "processing_time_sec": {"type": "number"},
                        "coverage_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "editor_focus_failed_ids": {"type": "array", "items": {"type": "integer"}},
                "editor_review_order": {"type": "string"},  # "failed_first"
            },
            "required": ["scene", "timestamp", "objects"],
        }

    @staticmethod
    def pipeline_report_schema() -> Dict[str, Any]:
        """
        Pipeline Report 结构：全流程执行报告
        """
        return {
            "type": "object",
            "properties": {
                "scene_name": {"type": "string"},
                "start_time": {"type": "string"},  # ISO 8601
                "end_time": {"type": "string"},
                "total_duration_sec": {"type": "number"},
                "status": {"type": "string", "enum": ["success", "success_with_warnings", "failed"]},
                "phases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phase_name": {"type": "string"},
                            "status": {"type": "string"},
                            "duration_sec": {"type": "number"},
                            "output_file": {"type": ["string", "null"]},
                            "error_message": {"type": ["string", "null"]},
                            "warnings": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "statistics": {
                    "type": "object",
                    "properties": {
                        "viewpoints_sampled": {"type": "integer"},
                        "coverage_ratio": {"type": "number"},
                        "instances_extracted": {"type": "integer"},
                        "receptacles_detected": {"type": "integer"},
                        "objects_placed": {"type": "integer"},
                        "placement_success_rate": {"type": "number"},
                        "stability_score": {"type": "number"},
                    },
                },
                "failed_objects": {"type": "array"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["scene_name", "start_time", "status", "phases"],
        }


class SchemaValidator:
    """JSON schema 校验工具"""

    @staticmethod
    def validate_observation(obs: Dict[str, Any]) -> Tuple[bool, str]:
        """验证 Observation JSON"""
        try:
            required = ["viewpoint_id", "camera_pos", "camera_lookat", "extracted_instances"]
            for field in required:
                if field not in obs:
                    return False, f"Missing required field: {field}"
            if not isinstance(obs["camera_pos"], list) or len(obs["camera_pos"]) != 3:
                return False, "camera_pos must be 3D array"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_instance(inst: Dict[str, Any]) -> Tuple[bool, str]:
        """验证 Instance JSON"""
        try:
            required = ["instance_id", "semantic_label", "center_3d"]
            for field in required:
                if field not in inst:
                    return False, f"Missing required field: {field}"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_receptacle(rec: Dict[str, Any]) -> Tuple[bool, str]:
        """验证 Receptacle JSON"""
        try:
            required = ["receptacle_id", "receptacle_type", "center", "normal"]
            for field in required:
                if field not in rec:
                    return False, f"Missing required field: {field}"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_placement(place: Dict[str, Any]) -> Tuple[bool, str]:
        """验证 Placement JSON"""
        try:
            required = ["object_id", "model_id", "position", "rotation", "source"]
            for field in required:
                if field not in place:
                    return False, f"Missing required field: {field}"
            return True, "OK"
        except Exception as e:
            return False, str(e)

    @staticmethod
    def validate_layout(layout: Dict[str, Any]) -> Tuple[bool, str]:
        """验证 Layout JSON"""
        try:
            required = ["scene", "timestamp", "objects"]
            for field in required:
                if field not in layout:
                    return False, f"Missing required field: {field}"
            if not isinstance(layout["objects"], list):
                return False, "objects must be array"
            return True, "OK"
        except Exception as e:
            return False, str(e)


# 便捷导出函数
def get_schema(schema_type: str) -> Dict[str, Any]:
    """获取指定 schema"""
    schemas = {
        "observation": PipelineSchema.observation_schema,
        "instance": PipelineSchema.instance_schema,
        "receptacle": PipelineSchema.receptacle_schema,
        "placement": PipelineSchema.placement_schema,
        "layout": PipelineSchema.layout_schema,
        "report": PipelineSchema.pipeline_report_schema,
    }
    if schema_type not in schemas:
        raise ValueError(f"Unknown schema type: {schema_type}")
    return schemas[schema_type]()


def validate(schema_type: str, data: Dict[str, Any]) -> Tuple[bool, str]:
    """快捷验证函数"""
    validators = {
        "observation": SchemaValidator.validate_observation,
        "instance": SchemaValidator.validate_instance,
        "receptacle": SchemaValidator.validate_receptacle,
        "placement": SchemaValidator.validate_placement,
        "layout": SchemaValidator.validate_layout,
    }
    if schema_type not in validators:
        return False, f"Unknown validator type: {schema_type}"
    return validators[schema_type](data)
