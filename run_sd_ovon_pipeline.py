"""
Run SD-OVON Pipeline

端到端编排脚本，串联 3.1 到 3.4 流程和编辑器交互。
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# 导入各阶段模块
from sd_ovon_config import SDOVONConfig
from coverage_sampler import CoverageSampler, run_coverage_sampler
from observation_generator import ObservationGenerator, generate_observations
from instance_fusion_stub import InstanceFusionStub, extract_and_fuse_instances
from receptacle_plane_detector import ReceptaclePlaneDetector, detect_receptacles
from semantics_relation_model import SemanticsRelationModel
from semantics_aware_placer import SemanticsAwarePlacer, place_objects_semantically
from physics_stabilizer_heuristic import PhysicsStabilizerHeuristic, check_placement_stability
from mock_room_recommender import MockRoomRecommender, generate_mock_recommendations
from hm3d_paths import resolve_scene_paths


class SDOVONPipeline:
    """SD-OVON 流程编排器"""

    def __init__(self, config: SDOVONConfig):
        self.config = config
        self.report = {
            "pipeline_status": "initialized",
            "phases": [],
            "warnings": [],
            "errors": [],
        }

    def run_full_pipeline(
        self,
        scene_name: str,
        objects: Optional[List[Dict[str, Any]]] = None,
        rooms_info_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整 SD-OVON 流程

        Args:
            scene_name: 场景名称
            objects: 待放置的对象列表（可选，若无则从场景导出）
            rooms_info_dir: 房间信息目录

        Returns:
            完整的流程报告
        """
        start_time = datetime.now()
        self.report["start_time"] = start_time.isoformat()

        try:
            # Phase 0: 准备
            print(f"\n{'='*60}")
            print(f"[Phase 0] Pipeline Initialization")
            print(f"{'='*60}")

            if objects is None:
                objects = self._load_objects_from_scene(scene_name)

            if not objects:
                raise RuntimeError(f"No objects found for scene {scene_name}")

            print(f"  Loaded {len(objects)} objects")

            # Phase 3.1: 覆盖采样
            if self.config.COVERAGE_SAMPLING["enable"]:
                phase_name = "coverage_sampling"
                print(f"\n{'='*60}")
                print(f"[Phase 3.1] Coverage Random Observations Sampling")
                print(f"{'='*60}")

                sampler = CoverageSampler(self.config)
                obs_report = sampler.sample_viewpoints_for_scene(scene_name)

                viewpoints = obs_report.get("viewpoints", [])
                print(f"  Sampled {len(viewpoints)} viewpoints")

                phase_status = "success" if len(viewpoints) > 0 else "warning"
                self._log_phase(phase_name, phase_status, obs_report)
            else:
                viewpoints = []
                self._log_phase("coverage_sampling", "skipped", {})

            # Phase 3.1 ext: 观测生成
            observations = []
            if len(viewpoints) > 0:
                print(f"\n[Phase 3.1 Ext] Observation Generation")
                gen = ObservationGenerator(self.config)
                obs_gen_report = gen.generate_observations_from_viewpoints(scene_name, viewpoints)
                observations = obs_gen_report.get("observations", [])
                print(f"  Generated {len(observations)} observations")

            # Phase 3.2: 实例提取与融合
            if self.config.RECEPTACLE_DETECTION["enable"]:
                phase_name = "instance_fusion"
                print(f"\n{'='*60}")
                print(f"[Phase 3.2] Instance Extraction and Fusion")
                print(f"{'='*60}")

                fusion = InstanceFusionStub(self.config)
                inst_report = fusion.extract_and_fuse_instances(scene_name, observations, objects)

                instances = inst_report.get("instances", [])
                print(f"  Extracted and fused {len(instances)} instances")

                self._log_phase(phase_name, "success", inst_report)
            else:
                instances = []
                self._log_phase("instance_fusion", "skipped", {})

            # Phase 3.3: Receptacle 平面检测
            if self.config.RECEPTACLE_DETECTION["enable"]:
                phase_name = "receptacle_detection"
                print(f"\n{'='*60}")
                print(f"[Phase 3.3] Receptacle Planes Identification")
                print(f"{'='*60}")

                detector = ReceptaclePlaneDetector(self.config)
                rec_report = detector.detect_receptacles(scene_name, instances)

                receptacles = rec_report.get("receptacles", [])
                print(f"  Detected {len(receptacles)} receptacles")

                self._log_phase(phase_name, "success", rec_report)
            else:
                receptacles = []
                self._log_phase("receptacle_detection", "skipped", {})

            # Phase 3.4: 语义感知放置
            if self.config.SEMANTIC_PLACEMENT["enable"]:
                phase_name = "semantics_aware_placement"
                print(f"\n{'='*60}")
                print(f"[Phase 3.4] Semantics-aware Object Placement")
                print(f"{'='*60}")

                placer = SemanticsAwarePlacer(self.config)
                place_report = placer.place_objects_semantically(scene_name, objects, receptacles)

                placements = place_report.get("placements", [])
                failed_count = place_report.get("failed_count", 0)
                print(f"  Placed {len(placements)} objects ({failed_count} failed)")

                self._log_phase(phase_name, "success", place_report)
            else:
                placements = []
                self._log_phase("semantics_aware_placement", "skipped", {})

            # Phase（可选）: 物理稳定性检查
            if self.config.PHYSICS_STABILIZATION["enable"]:
                phase_name = "physics_stabilization"
                print(f"\n{'='*60}")
                print(f"[Phase Optional] Physics Stabilization Check")
                print(f"{'='*60}")

                stabilizer = PhysicsStabilizerHeuristic(self.config)
                stabil_report = stabilizer.check_stability(placements, receptacles)

                stable_count = stabil_report.get("stable_count", 0)
                print(f"  Stability check: {stable_count}/{len(placements)} objects stable")

                self._log_phase(phase_name, "success", stabil_report)

            # 最终：生成布局并保存
            print(f"\n{'='*60}")
            print(f"[Phase Final] Layout Assembly and Output")
            print(f"{'='*60}")

            final_layout = self._assemble_final_layout(scene_name, placements, place_report)
            layout_file = self._save_layout(scene_name, final_layout)
            print(f"  Saved layout to: {layout_file}")

            # 完成报告
            end_time = datetime.now()
            self.report["end_time"] = end_time.isoformat()
            self.report["duration_sec"] = (end_time - start_time).total_seconds()
            self.report["pipeline_status"] = "success"

            print(f"\n{'='*60}")
            print(f"[Pipeline Complete] Status: {self.report['pipeline_status']}")
            print(f"  Total duration: {self.report['duration_sec']:.2f}s")
            print(f"{'='*60}\n")

            return self.report

        except Exception as e:
            self.report["pipeline_status"] = "failed"
            self.report["errors"].append(str(e))
            print(f"\n[ERROR] Pipeline failed: {e}\n")
            raise

    def _load_objects_from_scene(self, scene_name: str) -> List[Dict[str, Any]]:
        """从场景导出数据加载对象"""
        scene_info_path = os.path.join(
            "results", "scene_info", scene_name, f"{scene_name}_scene_info.json"
        )

        if os.path.isfile(scene_info_path):
            try:
                with open(scene_info_path, "r") as f:
                    data = json.load(f)
                    return data.get("objects", [])
            except Exception as e:
                print(f"[Warning] Failed to load scene info: {e}")

        # fallback：生成虚拟对象
        return [
            {"id": i, "model_id": f"obj_{i}", "position": [0, 0, 0]}
            for i in range(5)
        ]

    def _log_phase(self, phase_name: str, status: str, output: Dict[str, Any]):
        """记录阶段执行结果"""
        self.report["phases"].append({
            "phase_name": phase_name,
            "status": status,
            "output_keys": list(output.keys()) if isinstance(output, dict) else [],
        })

    def _assemble_final_layout(
        self,
        scene_name: str,
        placements: List[Dict[str, Any]],
        place_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """组装最终布局 JSON"""
        scene_paths = resolve_scene_paths(scene_name, require_semantic=False)
        scene_glb = str(scene_paths.stage_glb) if scene_paths else "unknown"

        layout = {
            "scene": scene_glb,
            "scene_name": scene_name,
            "timestamp": time.time(),
            "objects": placements,
            "sd_ovon_stats": {
                "total_objects": place_report.get("total_objects", len(placements)),
                "placed_count": place_report.get("placed_count", len(placements)),
                "failed_count": place_report.get("failed_count", 0),
                "source": "sd_ovon_pipeline",
            },
        }
        return layout

    def _save_layout(self, scene_name: str, layout: Dict[str, Any]) -> str:
        """保存布局到文件"""
        output_dir = self.config.get_output_dir("layouts")
        timestamp = int(time.time())
        output_file = os.path.join(output_dir, f"{scene_name}_sd_ovon_{timestamp}.json")

        with open(output_file, "w") as f:
            json.dump(layout, f, indent=2)

        return output_file


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Run SD-OVON object placement pipeline")
    parser.add_argument("--scene", required=True, help="Scene name (e.g., 00824-Dd4bFSTQ8gi)")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip coverage sampling phase")
    parser.add_argument("--skip-physics", action="store_true", help="Skip physics check")

    args = parser.parse_args()

    # 初始化配置
    config = SDOVONConfig()

    if args.skip_coverage:
        config.COVERAGE_SAMPLING["enable"] = False

    if args.skip_physics:
        config.PHYSICS_STABILIZATION["enable"] = False

    if args.output_dir:
        config.OUTPUT_BASE_DIR = args.output_dir

    # 运行 pipeline
    pipeline = SDOVONPipeline(config)

    try:
        report = pipeline.run_full_pipeline(args.scene)
        print(f"\n[Success] Pipeline completed for scene {args.scene}")
        return 0
    except Exception as e:
        print(f"\n[Failed] Pipeline error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
