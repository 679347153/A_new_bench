"""
SD-OVON 配置管理 (增强版 - 完整实现)

支持在 mock/stub 和完整实现之间切换。
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path


class SDOVONConfig:
    """SD-OVON 流程配置"""

    def __init__(self, implementation_level: str = "production"):
        """
        初始化配置

        Args:
            implementation_level: "mock" (快速原型) | "production" (完整实现)
        """
        self.IMPLEMENTATION_LEVEL = implementation_level

        # ==================== Phase 1: Semantic Router ====================
        self.SEMANTIC_ROUTER = {
            "mode": "full",  # "full" | "lite"
            "device": "cuda",
        }

        # ==================== Phase 2: Feature Preprocessing ====================
        self.FEATURE_PREPROCESSING = {
            "extract_color_histogram": True,
            "extract_geometric_features": True,
            "extract_semantic_features": True,
        }

        # ==================== Phase 3: Object Generation ====================
        self.DIFFUSION_MODEL = {
            "model_name": "stable-diffusion-v1-5",
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "seed": 42,
        }

        # ==================== Phase 3.2: Instance Fusion ====================
        if implementation_level == "production":
            self.INSTANCE_FUSION = {
                "backend": "gsam_3d",  # "gsam_3d" | "mock"
                "gsam_model_name": "grounding-sam-v1",
                "device": "cuda",
                "fusion_config": {
                    "max_fusion_distance": 0.5,
                    "overlap_threshold": 0.3,
                    "quality_threshold": 0.7,
                },
                "error_correction": {
                    "enable_deduplication": True,
                    "enable_smoothing": True,
                    "sparse_point_filter": 5,  # 最小点数
                },
            }
        else:
            self.INSTANCE_FUSION = {
                "backend": "mock",
                "num_mock_instances": 5,
            }

        # ==================== Phase 4: Deduplication ====================
        self.DEDUPLICATION = {
            "iou_threshold": 0.5,
            "confidence_threshold": 0.5,
            "max_duplicates_per_category": 3,
        }

        # ==================== Phase 5: Semantic Placement ====================
        self.SEMANTIC_PLACEMENT = {
            "backend": "rule_based",  # "rule_based" | "llm_based"
            "min_distance_between_objects": 0.3,
            "height_threshold": 0.1,
            "collision_check": True,
            "max_placement_attempts": 100,
        }

        # ==================== Phase 3.4: Physics Stabilization ====================
        if implementation_level == "production":
            self.PHYSICS_STABILIZATION = {
                "backend": "habitat_sim",  # "habitat_sim" | "heuristic"
                "enable_simulation": True,
                "gravity": [0, -9.8, 0],
                "friction": 0.5,
                "timestep": 0.01,
                "settle_time": 3.0,
                "max_iterations": 5,
                "contact_threshold": 0.05,
                "stability_threshold": 0.6,
            }
        else:
            self.PHYSICS_STABILIZATION = {
                "backend": "heuristic",
                "enable_simulation": False,
                "contact_threshold": 0.05,
                "stability_threshold": 0.6,
            }

        # ==================== Receptacle Detection ====================
        self.RECEPTACLE_DETECTION = {
            "mode": "bbox_based",  # "bbox_based" | "mesh_based" | "mock"
            "min_surface_area": 0.2,
            "normal_threshold": 0.7,
        }

        # ==================== Semantic Relations ====================
        self.SEMANTIC_RELATIONS = {
            "backend": "rule_based",  # "rule_based" | "llm_based" | "mock"
            "relations_db": {
                "clock": [["wall", "on"], ["shelf", "on"]],
                "vase": [["table", "on"], ["shelf", "on"], ["floor", "near"]],
                "chair": [["floor", "on"], ["table", "near"]],
                "lamp": [["table", "on"], ["shelf", "on"], ["floor", "near"]],
                "book": [["shelf", "on"], ["table", "on"]],
                "bottle": [["table", "on"], ["shelf", "on"], ["floor", "near"]],
            },
        }

        # ==================== Output Configuration ====================
        self.OUTPUT_DIRS = {
            "layouts": "./results/layouts",
            "reports": "./results/reports",
            "visualizations": "./results/visualizations",
            "intermediate": "./results/intermediate",
        }

        # ==================== Logging ====================
        self.LOGGING = {
            "level": "INFO",
            "log_file": "./logs/sd_ovon.log",
        }

    def get_output_dir(self, dir_type: str) -> str:
        """获取输出目录"""
        return self.OUTPUT_DIRS.get(dir_type, "./results")

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        return {
            "implementation_level": self.IMPLEMENTATION_LEVEL,
            "semantic_router": self.SEMANTIC_ROUTER,
            "feature_preprocessing": self.FEATURE_PREPROCESSING,
            "diffusion_model": self.DIFFUSION_MODEL,
            "instance_fusion": self.INSTANCE_FUSION,
            "deduplication": self.DEDUPLICATION,
            "semantic_placement": self.SEMANTIC_PLACEMENT,
            "physics_stabilization": self.PHYSICS_STABILIZATION,
            "receptacle_detection": self.RECEPTACLE_DETECTION,
            "semantic_relations": self.SEMANTIC_RELATIONS,
            "output_dirs": self.OUTPUT_DIRS,
        }

    def save_to_file(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_from_file(filepath: str) -> "SDOVONConfig":
        """从文件加载配置"""
        with open(filepath, "r") as f:
            conf_dict = json.load(f)

        impl_level = conf_dict.get("implementation_level", "production")
        config = SDOVONConfig(impl_level)

        # 应用自定义配置
        for key in conf_dict:
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), conf_dict[key])

        return config
