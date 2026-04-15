"""
Coverage Random Observations Sampling (Phase 3.1)

根据可导航区域进行随机采样，使用 Gaussian 覆盖图更新机制，确保采样点覆盖率。
参考 SD-OVON section 3.1。
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import json
import os
from sd_ovon_config import SDOVONConfig
from pipeline_schema import SchemaValidator


class CoverageSampler:
    """覆盖采样引擎"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.nav_map = None
        self.coverage_map = None
        self.valid_viewpoints = []

    def sample_viewpoints_for_scene(
        self,
        scene_name: str,
        nav_map: Optional[np.ndarray] = None,
        objects: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        主入口：从导航

区域采样观测点，返回有效采样点列表。

        Args:
            scene_name: 场景名称
            nav_map: 可导航区域掩码 (H x W binary map) 或 None（则生成虚拟）
            objects: 场景中的对象列表

        Returns:
            {
                "scene_name": str,
                "viewpoints": [
                    {"viewpoint_id": int, "position": [x,y,z], "quality_score": float}
                ],
                "statistics": {
                    "total_sampled": int,
                    "valid_viewpoints": int,
                    "coverage_ratio": float,
                    "iterations": int
                }
            }
        """
        cfg = self.config.COVERAGE_SAMPLING

        # Step 1: 初始化或生成虚拟 nav map
        if nav_map is None:
            nav_map = self._generate_virtual_nav_map(scene_name)

        self.nav_map = nav_map
        h, w = nav_map.shape

        # Step 2: 初始化覆盖概率图
        self.coverage_map = np.zeros((h, w), dtype=np.float32)

        # Step 3: Gaussian 覆盖循环采样
        iteration = 0
        total_area = np.sum(nav_map > 0)
        viewpoints = []

        while iteration < cfg["max_iterations"]:
            iteration += 1

            # 根据 (1 - coverage_map) 采样
            prob_map = (1.0 - self.coverage_map) * (nav_map > 0).astype(np.float32)
            if np.sum(prob_map) < 1e-6:
                break

            prob_map = prob_map / np.sum(prob_map)

            # 采样候选点
            candidates = self._sample_candidates_from_map(prob_map, cfg["sample_points_per_iteration"], nav_map)

            # 筛选有效候选（距离障碍物足够远）
            valid_candidates = [
                c for c in candidates
                if self._is_valid_viewpoint(c, nav_map, cfg["min_distance_to_obstacle"])
            ]

            if not valid_candidates:
                continue

            # 对每个有效候选，生成多角度观测并更新覆盖图
            for candidate in valid_candidates:
                viewpoint_id = len(viewpoints)
                quality = self._compute_quality_score(candidate, nav_map, viewpoints)

                # 生成 Gaussian 覆盖核
                r_min = cfg["gaussian_radius_min"]
                r_max = cfg["gaussian_radius_max"]
                gaussian_kernel = self._create_gaussian_kernel(candidate, h, w, r_min, r_max)

                # 更新覆盖图
                self.coverage_map = np.maximum(self.coverage_map, gaussian_kernel)

                viewpoints.append({
                    "viewpoint_id": viewpoint_id,
                    "position": [float(candidate[0]), float(candidate[1]), 0.0],  # 2D -> 3D
                    "quality_score": quality,
                })

            # 检查是否达到覆盖目标
            coverage_ratio = np.sum(self.coverage_map >= 0.5) / max(total_area, 1)
            if coverage_ratio >= cfg["coverage_threshold"]:
                break

        # 统计
        final_coverage_ratio = np.sum(self.coverage_map >= 0.5) / max(total_area, 1)
        report = {
            "scene_name": scene_name,
            "viewpoints": viewpoints,
            "statistics": {
                "total_sampled": len(viewpoints),
                "valid_viewpoints": len(viewpoints),
                "coverage_ratio": float(final_coverage_ratio),
                "iterations": iteration,
                "target_coverage": cfg["coverage_threshold"],
            },
        }
        return report

    def _generate_virtual_nav_map(self, scene_name: str, h: int = 512, w: int = 512) -> np.ndarray:
        """生成虚拟可导航区域（用于 mock 测试）"""
        # 虚拟：中心矩形可导航
        nav_map = np.zeros((h, w), dtype=np.uint8)
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        nav_map[y1:y2, x1:x2] = 1
        return nav_map

    def _sample_candidates_from_map(
        self,
        prob_map: np.ndarray,
        num_samples: int,
        nav_map: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """从概率图中采样候选点"""
        h, w = prob_map.shape
        candidates = []

        valid_indices = np.where(nav_map > 0)
        if len(valid_indices[0]) == 0:
            return candidates

        for _ in range(num_samples):
            idx = np.random.choice(len(valid_indices[0]), p=prob_map[valid_indices])
            y, x = valid_indices[0][idx], valid_indices[1][idx]
            candidates.append((float(x), float(y)))

        return candidates

    def _is_valid_viewpoint(
        self,
        candidate: Tuple[float, float],
        nav_map: np.ndarray,
        min_distance: float,
    ) -> bool:
        """判断采样点是否有效（距离障碍物足够远）"""
        # 简化：检查局部区域内是否有足够多的可导航像素
        h, w = nav_map.shape
        x, y = int(candidate[0]), int(candidate[1])
        kernel_size = int(min_distance * 10 + 1)  # mock 距离转像素

        y_min = max(0, y - kernel_size)
        y_max = min(h, y + kernel_size)
        x_min = max(0, x - kernel_size)
        x_max = min(w, x + kernel_size)

        local_nav = nav_map[y_min:y_max, x_min:x_max]
        return np.sum(local_nav > 0) > (kernel_size * kernel_size * 0.3)

    def _compute_quality_score(
        self,
        candidate: Tuple[float, float],
        nav_map: np.ndarray,
        existing_viewpoints: List[Dict[str, Any]],
    ) -> float:
        """计算采样点的质量分数（覆盖新区域的程度）"""
        h, w = nav_map.shape
        x, y = int(candidate[0]), int(candidate[1])

        # 距离图边缘的远近（越近中心越好）
        dist_to_edge = min(x, y, w - x, h - y) / max(h, w)

        # 与现有采样点的距离
        min_dist_to_existing = 1.0
        if existing_viewpoints:
            for vp in existing_viewpoints:
                pos = vp["position"]
                dist = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                min_dist_to_existing = min(min_dist_to_existing, dist / (h + w))

        quality = 0.5 * dist_to_edge + 0.5 * min_dist_to_existing
        return float(quality)

    def _create_gaussian_kernel(
        self,
        center: Tuple[float, float],
        h: int,
        w: int,
        r_min: float,
        r_max: float,
    ) -> np.ndarray:
        """生成 Gaussian 覆盖核"""
        kernel_size = int(r_max * 20 + 1)  # mock 距离转像素

        y = int(center[1])
        x = int(center[0])

        kernel = np.zeros((h, w), dtype=np.float32)
        for dy in range(-kernel_size, kernel_size + 1):
            for dx in range(-kernel_size, kernel_size + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    dist = np.sqrt(dx ** 2 + dy ** 2) / (20 * r_max)  # 归一化距离
                    if dist <= 1.0:
                        value = np.exp(-(dist ** 2) / (2 * 0.3 ** 2))
                        kernel[ny, nx] = value

        return kernel


def run_coverage_sampler(
    scene_name: str,
    output_dir: Optional[str] = None,
) -> str:
    """
    运行覆盖采样，输出 observations.json

    Args:
        scene_name: 场景名称
        output_dir: 输出目录（不指定则用默认）

    Returns:
        输出文件路径
    """
    config = SDOVONConfig()
    sampler = CoverageSampler(config)

    report = sampler.sample_viewpoints_for_scene(scene_name)

    if output_dir is None:
        output_dir = config.get_output_dir("observations")

    output_file = os.path.join(output_dir, f"{scene_name}_observations.json")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[CoverageSampler] Saved observations to: {output_file}")
    print(f"  Viewpoints sampled: {report['statistics']['valid_viewpoints']}")
    print(f"  Coverage ratio: {report['statistics']['coverage_ratio']:.2%}")

    return output_file
