"""
SD-OVON Pipeline Configuration

集中管理所有参数，包括覆盖采样、检测、放置、物理等阶段的配置。
"""

from typing import Dict, Any
import os


class SDOVONConfig:
    """SD-OVON 流程配置类"""

    # ===== 全局配置 =====
    DEBUG_MODE = False
    LOG_LEVEL = "INFO"
    OUTPUT_BASE_DIR = "./results/sd_ovon_pipeline"
    CHECKPOINT_DIR = "./results/sd_ovon_pipeline/checkpoints"
    
    # ===== Phase 3.1: 覆盖采样配置 =====
    COVERAGE_SAMPLING = {
        "enable": True,
        "nav_map_height": 0.3,  # 采样点距离地面高度
        "min_distance_to_obstacle": 0.5,  # 采样点距离障碍物的最小距离
        "gaussian_radius_min": 1.5,  # Gaussian 覆盖半径下界
        "gaussian_radius_max": 4.0,  # 上界
        "coverage_threshold": 0.35,  # 覆盖率目标（30.5%）
        "max_iterations": 100,  # 最大采样迭代数
        "convergence_tolerance": 0.95,  # 收敛容差（百分比）
        "sample_points_per_iteration": 5,  # 每次采样的候选点数
    }

    # ===== Phase 3.2: 观测与视角 =====
    OBSERVATION_GENERATION = {
        "enable": True,
        "yaw_angles": [0, 90, 180, 270],  # 水平旋转角度（度）
        "pitch_angles": [-15, 0, 15],  # 竖直俯仰角度（度）
        "camera_height": 1.5,  # 相机高度（从地面算起）
        "image_resolution": [512, 512],
        "fov_degrees": 90,
        "store_depth": False,
        "store_rgb": False,
        "store_instance_mask": False,
    }

    # ===== Phase 3.3: Receptacle 平面检测 =====
    RECEPTACLE_DETECTION = {
        "enable": True,
        "mode": "bbox_based",  # "bbox_based" | "mesh_based" | "mock"
        "supported_types": ["table", "shelf", "bed", "floor", "wall", "cabinet"],
        "min_surface_area": 0.15,  # 最小表面积（m²）
        "normal_threshold": 0.7,  # 法向量相似度阈值（点积）
        "plane_thickness": 0.05,  # 平面厚度（m）
        "min_plane_points": 10,  # 最小点数
        "convex_hull_simplification": True,
    }

    # ===== Phase 3.4: 语义感知放置 =====
    SEMANTIC_PLACEMENT = {
        "enable": True,
        "mode": "rule_based",  # "rule_based" | "llm_based" | "mock"
        "min_distance_between_objects": 0.25,  # 物体间最小距离（m）
        "min_distance_to_edge": 0.08,  # 物体到表面边缘最小距离（m）
        "max_overhang": 0.15,  # 最大悬空比例
        "num_candidate_points_per_receptacle": 8,  # 每个 receptacle 采样点数
        "max_placement_attempts": 30,  # 单对象最大重试次数
        "confidence_threshold": 0.5,  # 语义置信度阈值
        "enable_collision_check": True,
        "enable_stability_check": True,
    }

    # ===== 对象画像与约束 =====
    OBJECT_PROFILE_KEYWORDS = {
        "table": {"radius": 0.60, "y_offset": 0.05, "support_ratio": 0.5},
        "desk": {"radius": 0.58, "y_offset": 0.05, "support_ratio": 0.5},
        "sofa": {"radius": 0.70, "y_offset": 0.05, "support_ratio": 0.6},
        "chair": {"radius": 0.40, "y_offset": 0.05, "support_ratio": 0.4},
        "bed": {"radius": 0.80, "y_offset": 0.05, "support_ratio": 0.7},
        "cabinet": {"radius": 0.55, "y_offset": 0.05, "support_ratio": 0.5},
        "shelf": {"radius": 0.50, "y_offset": 0.05, "support_ratio": 0.5},
        "statue": {"radius": 0.42, "y_offset": 0.08, "support_ratio": 0.3},
        "vase": {"radius": 0.28, "y_offset": 0.10, "support_ratio": 0.2},
        "bottle": {"radius": 0.24, "y_offset": 0.08, "support_ratio": 0.2},
        "clock": {"radius": 0.20, "y_offset": 0.10, "support_ratio": 0.15},
        "camera": {"radius": 0.20, "y_offset": 0.08, "support_ratio": 0.15},
    }

    DEFAULT_OBJECT_PROFILE = {
        "radius": 0.35,
        "y_offset": 0.05,
        "support_ratio": 0.3,
    }

    # ===== 物理稳定性检查 =====
    PHYSICS_STABILIZATION = {
        "enable": True,
        "mode": "heuristic",  # "heuristic" | "simulation"
        "enable_physics_simulation": False,  # 接入 habitat-sim 物理模拟
        "gravity_enabled": True,
        "friction_coefficient": 0.5,
        "contact_area_threshold": 0.02,  # 最小接触面积（m²）
        "center_of_mass_offset_max": 0.08,  # 重心偏移容限（m）
        "stability_score_threshold": 0.6,
    }

    # ===== 语义关系参数 =====
    SEMANTIC_RELATIONS = {
        "clock": [["wall", "on", 0.9], ["shelf", "on", 0.6], ["table", "on", 0.3]],
        "vase": [["table", "on", 0.85], ["shelf", "on", 0.7], ["floor", "near", 0.4]],
        "camera": [["shelf", "on", 0.8], ["table", "on", 0.5], ["desk", "on", 0.6]],
        "statue": [["shelf", "on", 0.75], ["table", "on", 0.6], ["floor", "near", 0.3]],
        "bottle": [["table", "on", 0.8], ["shelf", "on", 0.7], ["floor", "near", 0.2]],
        "book": [["shelf", "on", 0.9], ["table", "on", 0.7], ["desk", "on", 0.8]],
    }

    # ===== Fallback 与异常处理 =====
    FALLBACK_STRATEGY = {
        "missing_receptacles": "use_room_center",  # "use_room_center" | "skip_object"
        "missing_semantic_relations": "use_default_rule",
        "placement_failure": "mark_manual_fix",  # "mark_manual_fix" | "retry_aggressive"
        "physics_failure": "warn_but_place",  # "warn_but_place" | "reject"
    }

    # ===== 输出与报告 =====
    OUTPUT_CONFIG = {
        "save_observations": False,
        "save_instances": True,
        "save_receptacles": True,
        "save_placements": True,
        "save_layout": True,
        "save_report": True,
        "save_visualization": False,
        "include_failed_objects": True,
    }

    # ===== 目录结构 =====
    DIRECTORIES = {
        "observations": ".../observations",
        "instances": ".../instances",
        "receptacles": ".../receptacles",
        "placements": ".../placements",
        "layouts": ".../layouts",
        "reports": ".../reports",
        "visualizations": ".../visualizations",
    }

    @classmethod
    def get_output_dir(cls, subdir: str) -> str:
        """获取输出目录"""
        path = os.path.join(cls.OUTPUT_BASE_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_checkpoint_dir(cls) -> str:
        """获取 checkpoint 目录"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        return cls.CHECKPOINT_DIR

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "coverage_sampling": cls.COVERAGE_SAMPLING,
            "observation_generation": cls.OBSERVATION_GENERATION,
            "receptacle_detection": cls.RECEPTACLE_DETECTION,
            "semantic_placement": cls.SEMANTIC_PLACEMENT,
            "object_profile_keywords": cls.OBJECT_PROFILE_KEYWORDS,
            "physics_stabilization": cls.PHYSICS_STABILIZATION,
            "semantic_relations": cls.SEMANTIC_RELATIONS,
            "fallback_strategy": cls.FALLBACK_STRATEGY,
            "output_config": cls.OUTPUT_CONFIG,
        }


# 便捷函数
def get_config() -> SDOVONConfig:
    """获取配置实例"""
    return SDOVONConfig()


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    从 YAML 加载配置（可选增强）
    """
    try:
        import yaml
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[Warning] Failed to load config from YAML: {e}. Using defaults.")
        return SDOVONConfig.to_dict()


def create_sd_ovon_config_yaml() -> str:
    """
    生成示例 YAML 配置文件
    """
    yaml_content = """
# SD-OVON Pipeline Configuration

coverage_sampling:
  enable: true
  coverage_threshold: 0.35
  max_iterations: 100

observation_generation:
  enable: true
  yaw_angles: [0, 90, 180, 270]
  pitch_angles: [-15, 0, 15]

receptacle_detection:
  enable: true
  mode: bbox_based
  supported_types: [table, shelf, bed, floor, wall, cabinet]

semantic_placement:
  enable: true
  mode: rule_based
  min_distance_between_objects: 0.25

physics_stabilization:
  enable: true
  mode: heuristic
  enable_physics_simulation: false

fallback_strategy:
  missing_receptacles: use_room_center
  missing_semantic_relations: use_default_rule
  placement_failure: mark_manual_fix
"""
    return yaml_content
