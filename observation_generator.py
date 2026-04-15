"""
Observation Generator (Phase 3.1 扩展)

从采样点生成多角度观察位姿（yaw/pitch 网格组合），作为实例提取的输入。
"""

from typing import Dict, List, Any, Optional
import json
import os
import numpy as np
from sd_ovon_config import SDOVONConfig


class ObservationGenerator:
    """观测位姿生成器"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.observations = []

    def generate_observations_from_viewpoints(
        self,
        scene_name: str,
        viewpoints: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        从采样点生成多角度观测

        Args:
            scene_name: 场景名称
            viewpoints: 采样点列表 [{"viewpoint_id": int, "position": [x,y,z]}]

        Returns:
            {
                "scene_name": str,
                "total_observations": int,
                "observations": [
                    {
                        "observation_id": int,
                        "viewpoint_id": int,
                        "camera_pos": [x,y,z],
                        "camera_lookat": [x,y,z],
                        "camera_up": [0,1,0],
                        "fov_degrees": 90,
                        "yaw_angle": 0,
                        "pitch_angle": 0
                    }
                ]
            }
        """
        cfg = self.config.OBSERVATION_GENERATION
        observations = []
        observation_id = 0

        for vp in viewpoints:
            vp_id = vp.get("viewpoint_id", 0)
            pos = vp.get("position", [0, 0, 0])

            for yaw in cfg["yaw_angles"]:
                for pitch in cfg["pitch_angles"]:
                    # 计算相机位置与朝向
                    camera_pos = self._compute_camera_pos(pos, cfg["camera_height"])
                    camera_lookat = self._compute_lookat(camera_pos, yaw, pitch)
                    camera_up = [0.0, 1.0, 0.0]

                    obs = {
                        "observation_id": observation_id,
                        "viewpoint_id": vp_id,
                        "camera_pos": [float(x) for x in camera_pos],
                        "camera_lookat": [float(x) for x in camera_lookat],
                        "camera_up": camera_up,
                        "fov_degrees": cfg["fov_degrees"],
                        "yaw_angle": float(yaw),
                        "pitch_angle": float(pitch),
                        "extracted_instances": [],  # 後來會由檢測器填充
                    }

                    # 简单校验
                    if self._is_valid_observation(obs):
                        observations.append(obs)
                        observation_id += 1

        report = {
            "scene_name": scene_name,
            "total_observations": len(observations),
            "observations": observations,
        }
        return report

    def _compute_camera_pos(self, center: List[float], height: float) -> List[float]:
        """计算相机位置（以第一个采样点为中心，提升至指定高度）"""
        return [float(center[0]), float(height), float(center[2])]

    def _compute_lookat(self, camera_pos: List[float], yaw: float, pitch: float) -> List[float]:
        """计算相机看向的目标点（基于 yaw 和 pitch）"""
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        distance = 2.0  # 看向距离

        dx = distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        dy = distance * np.sin(pitch_rad)
        dz = distance * np.cos(pitch_rad) * np.cos(yaw_rad)

        lookat = [
            float(camera_pos[0] + dx),
            float(camera_pos[1] + dy),
            float(camera_pos[2] + dz),
        ]
        return lookat

    def _is_valid_observation(self, obs: Dict[str, Any]) -> bool:
        """校验观测有效性"""
        try:
            pos = obs.get("camera_pos", [])
            if not isinstance(pos, list) or len(pos) != 3:
                return False
            lookat = obs.get("camera_lookat", [])
            if not isinstance(lookat, list) or len(lookat) != 3:
                return False
            return True
        except Exception:
            return False


def generate_observations(
    scene_name: str,
    viewpoints: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> str:
    """
    便捷函数：生成并保存观测

    Args:
        scene_name: 场景名称
        viewpoints: 采样点列表
        output_dir: 输出目录

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    gen = ObservationGenerator(config)

    report = gen.generate_observations_from_viewpoints(scene_name, viewpoints)

    if output_dir is None:
        output_dir = config.get_output_dir("observations")

    output_file = os.path.join(output_dir, f"{scene_name}_observations_detailed.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[ObservationGenerator] Saved {report['total_observations']} observations to: {output_file}")

    return output_file
