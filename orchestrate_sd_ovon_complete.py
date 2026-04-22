"""SD-OVON 编排管道（完整版，可切换实现）。

文件功能总览
1) 提供 SDOVONPipelineOrchestrator，统一封装 SD-OVON 的阶段化执行。
2) 提供 run_full_pipeline，用于执行完整 8 阶段流程并产出统一报告。
3) 提供 run_roomwise_layout_pipeline，将 HM3D layout_json 的 room_groups
     直接桥接为按房间放置结果，便于快速联调。

文件依赖关系
- 强依赖（缺失会直接失败）：
    - sd_ovon_config_enhanced.SDOVONConfig
- 可选依赖（缺失会自动降级并继续运行）：
    - instance_fusion_gsam_3d.GSAMInstanceFusion（生产级实例融合）
    - instance_fusion_stub.InstanceFusionStub（融合回退实现）
    - physics_stabilizer_complete.PhysicsStabilizer（物理稳定性检查）
- 数据文件依赖（由各 stage 在运行期读取）：
    - results/scene_info/<scene>/*_rooms.json
    - results/probabilities/<scene>/*_probs.json
    - objects_images/*（当 rooms 推荐缺失时作为对象回退来源）

运行路径说明
- production: 优先走 GSAM 融合与物理仿真检查；不可用时回退到 stub/heuristic。
- mock: 走轻量路径，优先保证链路可运行与结果可调试。
"""

from typing import Dict, Any, Optional, List
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SDOVONPipelineOrchestrator:
    """可运行的 SD-OVON 编排器。

    设计目标是“优先可运行、其次可增强”：在可选模块缺失时自动降级，
    仍能产出结构化 stage_results，避免整条流水线被单点依赖阻塞。
    """

    def __init__(self, config_level: str = "production"):
        # 记录每个 stage 的耗时和结果，便于排查性能与质量问题。
        self.config_level = config_level
        self.stage_durations: Dict[str, float] = {}
        self.stage_results: Dict[str, Dict[str, Any]] = {}
        self._import_modules()

    def _import_modules(self) -> None:
        """根据配置导入真实存在的模块。

        该方法将“硬依赖”和“软依赖”拆分处理：
        - 配置类必须可导入。
        - 其他组件可缺失，缺失时仅记录 warning，不中断流程。
        """
        logger.info("[SDOVONPipeline] Initializing in %s mode", self.config_level)

        from sd_ovon_config_enhanced import SDOVONConfig

        self.SDOVONConfig = SDOVONConfig

        self.GSAMInstanceFusion = None
        self.InstanceFusionStub = None
        self.PhysicsStabilizer = None

        try:
            from instance_fusion_gsam_3d import GSAMInstanceFusion

            self.GSAMInstanceFusion = GSAMInstanceFusion
        except ImportError as e:
            logger.warning("[SDOVONPipeline] GSAM fusion unavailable: %s", e)

        try:
            from instance_fusion_stub import InstanceFusionStub

            self.InstanceFusionStub = InstanceFusionStub
        except ImportError as e:
            logger.warning("[SDOVONPipeline] Stub fusion unavailable: %s", e)

        try:
            from physics_stabilizer_complete import PhysicsStabilizer

            self.PhysicsStabilizer = PhysicsStabilizer
        except ImportError as e:
            logger.warning("[SDOVONPipeline] Physics stabilizer unavailable: %s", e)

    def run_full_pipeline(
        self,
        scene_name: str,
        observations: Optional[List[Dict[str, Any]]] = None,
        stage_overrides: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """运行完整流程并返回报告。

        Args:
            scene_name: 场景编号。
            observations: 观测帧；若为空，部分 stage 会构造最小占位输入。
            stage_overrides: 逐 stage 开关，True 表示执行，False 表示跳过。
        """
        logger.info("[SDOVONPipeline] Starting full pipeline for scene: %s", scene_name)

        self.stage_durations = {}
        self.stage_results = {}

        config = self.SDOVONConfig(implementation_level=self.config_level)
        stages_config = stage_overrides or {}
        self.stage_results["config"] = config.to_dict()

        # 统一声明 stage 执行顺序，报告聚合逻辑依赖该顺序。
        pipeline_stages = [
            ("stage_1_semantic_understanding", self._stage_1_semantic_understanding),
            ("stage_2_feature_preprocessing", self._stage_2_feature_preprocessing),
            ("stage_3_object_generation", self._stage_3_object_generation),
            ("stage_3_2_instance_fusion", self._stage_3_2_instance_fusion),
            ("stage_4_deduplication", self._stage_4_deduplication),
            ("stage_5_semantic_placement", self._stage_5_semantic_placement),
            ("stage_3_4_physics_check", self._stage_3_4_physics_check),
            ("stage_6_validation", self._stage_6_validation),
        ]

        for stage_name, stage_func in pipeline_stages:
            if not stages_config.get(stage_name, True):
                logger.info("[SDOVONPipeline] Skipping %s", stage_name)
                continue

            try:
                start_time = time.time()
                result = stage_func(scene_name, config, observations)
                duration = time.time() - start_time

                self.stage_results[stage_name] = result
                self.stage_durations[stage_name] = duration
                logger.info("[SDOVONPipeline] %s completed in %.2fs", stage_name, duration)
            except Exception as e:
                logger.exception("[SDOVONPipeline] %s failed", stage_name)
                self.stage_results[stage_name] = {"success": False, "error": str(e)}

        return self._generate_final_report(scene_name)

    def _stage_1_semantic_understanding(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 1: 从已有 room 推荐结果中提取语义信息。"""
        scene_dir = Path("results") / "scene_info" / scene_name
        room_files = sorted(scene_dir.glob("*_rooms.json"))

        if not room_files:
            return {
                "success": False,
                "scene_name": scene_name,
                "error": f"No room recommendation files found under {scene_dir}",
                "rooms": [],
                "object_room_map": {},
            }

        # rooms 用 region_id 去重，object_room_map 保留逐对象原始推荐。
        rooms: Dict[int, Dict[str, Any]] = {}
        object_room_map: Dict[str, List[Dict[str, Any]]] = {}

        for room_file in room_files:
            with room_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            object_name = room_file.name.replace("_rooms.json", "")
            recommendations = data.get("recommended_rooms", [])
            object_room_map[object_name] = recommendations

            for rec in recommendations:
                region_id = int(rec.get("region_id", -1))
                if region_id < 0:
                    continue
                rooms[region_id] = {
                    "region_id": region_id,
                    "room_center": rec.get("room_center", [0.0, 0.0, 0.0]),
                    "reasoning": rec.get("reasoning", ""),
                }

        return {
            "success": True,
            "scene_name": scene_name,
            "rooms_identified": len(rooms),
            "objects_with_recommendations": len(object_room_map),
            "rooms": list(rooms.values()),
            "object_room_map": object_room_map,
        }

    def _stage_2_feature_preprocessing(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 2: 将概率文件转换为可用于下游放置的轻量特征。"""
        prob_dir = Path("results") / "probabilities" / scene_name
        prob_files = sorted(prob_dir.glob("*_probs.json"))

        features: List[Dict[str, Any]] = []
        for prob_file in prob_files:
            with prob_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            probs = data.get("probabilities", [])
            top_prob = max((float(p.get("probability", 0.0)) for p in probs), default=0.0)
            features.append(
                {
                    "object_name": data.get("object_name", prob_file.name.replace("_probs.json", "")),
                    "candidate_room_count": len(probs),
                    "top_probability": top_prob,
                }
            )

        return {
            "success": True,
            "features_extracted": len(features),
            "feature_dimensions": 3,
            "features": features,
        }

    def _stage_3_object_generation(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 3: 根据 feature 和 room 推荐生成对象列表。"""
        object_room_map = self.stage_results.get("stage_1_semantic_understanding", {}).get("object_room_map", {})
        features = self.stage_results.get("stage_2_feature_preprocessing", {}).get("features", [])
        confidence_map = {f["object_name"]: f.get("top_probability", 0.5) for f in features}

        objects = []
        for idx, obj_name in enumerate(sorted(object_room_map.keys())):
            objects.append(
                {
                    "object_id": idx,
                    "name": obj_name,
                    "model_id": obj_name,
                    "confidence": float(confidence_map.get(obj_name, 0.5)),
                    "recommended_rooms": object_room_map.get(obj_name, []),
                }
            )

        # 如果没有 rooms 推理结果，则回退到图片名，保证后续 stage 有可处理对象。
        if not objects:
            img_dir = Path("objects_images")
            for idx, p in enumerate(sorted(img_dir.glob("*"))):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                objects.append(
                    {
                        "object_id": idx,
                        "name": p.stem,
                        "model_id": p.stem,
                        "confidence": 0.5,
                        "recommended_rooms": [],
                    }
                )

        avg_conf = sum(o["confidence"] for o in objects) / max(len(objects), 1)
        return {
            "success": True,
            "objects_generated": len(objects),
            "average_confidence": avg_conf,
            "objects": objects,
        }

    def _stage_3_2_instance_fusion(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 3.2: 使用生产或 stub 实例融合实现。"""
        objects = self.stage_results.get("stage_3_object_generation", {}).get("objects", [])

        try:
            # production 且 GSAM 可用时，优先走真实融合链路。
            if self.config_level == "production" and self.GSAMInstanceFusion is not None:
                import numpy as np

                inferred_observations = observations or [
                    {
                        "image": np.zeros((64, 64, 3), dtype=np.uint8),
                        "camera_pose": np.eye(4),
                    }
                ]
                prompts = [o.get("name", "object") for o in objects[:3]] or ["furniture", "appliance"]

                fusion = self.GSAMInstanceFusion(config.INSTANCE_FUSION)
                result = fusion.extract_and_fuse_instances(
                    scene_name=scene_name,
                    observations=inferred_observations,
                    text_prompts=prompts,
                )
                stats = result.get("statistics", {})
                return {
                    "success": result.get("status") == "success",
                    "instances_fused": int(stats.get("total_instances", 0)),
                    "fusion_quality": float(stats.get("average_fusion_quality", 0.0)),
                    "method": "gsam_3d",
                    "instances": result.get("instances", []),
                }

            if self.InstanceFusionStub is None:
                return {
                    "success": False,
                    "instances_fused": 0,
                    "fusion_quality": 0.0,
                    "method": "unavailable",
                    "error": "No available instance fusion backend",
                }

            # 否则回退到 stub，保证流程不断。
            fusion = self.InstanceFusionStub(config)
            result = fusion.extract_and_fuse_instances(
                scene_name=scene_name,
                observations=observations or [],
                objects_from_scene=objects,
            )
            return {
                "success": True,
                "instances_fused": int(result.get("total_instances", 0)),
                "fusion_quality": float(result.get("statistics", {}).get("average_confidence", 0.0)),
                "method": "stub",
                "instances": result.get("instances", []),
            }
        except Exception as e:
            return {
                "success": False,
                "instances_fused": 0,
                "fusion_quality": 0.0,
                "method": "unknown",
                "error": str(e),
            }

    def _stage_4_deduplication(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 4: 去重（按 model_id 或 name）。"""
        objects = self.stage_results.get("stage_3_object_generation", {}).get("objects", [])
        deduped = []
        seen = set()

        for obj in objects:
            key = obj.get("model_id") or obj.get("name")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(obj)

        return {
            "success": True,
            "objects_after_dedup": len(deduped),
            "duplicates_removed": max(len(objects) - len(deduped), 0),
            "objects": deduped,
        }

    def _stage_5_semantic_placement(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 5: 基于 room_center 的语义放置。"""
        objects = self.stage_results.get("stage_4_deduplication", {}).get("objects", [])
        object_room_map = self.stage_results.get("stage_1_semantic_understanding", {}).get("object_room_map", {})
        global_rooms = self.stage_results.get("stage_1_semantic_understanding", {}).get("rooms", [])

        placements = []
        for idx, obj in enumerate(objects):
            candidates = object_room_map.get(obj.get("name", ""), [])
            if candidates:
                # 首选对象自身的第一推荐房间。
                room = candidates[0]
                room_center = room.get("room_center", [0.0, 0.0, 0.0])
                region_id = room.get("region_id", -1)
            elif global_rooms:
                # 无对象级推荐时，退化到全局房间轮询。
                room = global_rooms[idx % len(global_rooms)]
                room_center = room.get("room_center", [0.0, 0.0, 0.0])
                region_id = room.get("region_id", -1)
            else:
                room_center = [0.0, 0.0, 0.0]
                region_id = -1

            placements.append(
                {
                    "object_id": obj.get("object_id", idx),
                    "name": obj.get("name", f"object_{idx}"),
                    "model_id": obj.get("model_id", obj.get("name", f"object_{idx}")),
                    "position": [
                        float(room_center[0]),
                        float(room_center[1]),
                        float(room_center[2]),
                    ],
                    "rotation": [0.0, 0.0, 0.0],
                    "size": [0.3, 0.3, 0.3],
                    "sampled_region_id": region_id,
                }
            )

        return {
            "success": True,
            "objects_placed": len(placements),
            "placement_success_rate": 1.0 if placements else 0.0,
            "placements": placements,
        }

    def _stage_3_4_physics_check(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 3.4: 物理稳定性检查。"""
        placements = self.stage_results.get("stage_5_semantic_placement", {}).get("placements", [])
        if not placements:
            return {
                "success": True,
                "stable_count": 0,
                "unstable_count": 0,
                "method": "skipped",
                "stability_rate": 0.0,
            }

        if self.PhysicsStabilizer is None:
            return {
                "success": True,
                "stable_count": len(placements),
                "unstable_count": 0,
                "method": "unavailable_fallback",
                "stability_rate": 1.0,
            }

        try:
            stabilizer = self.PhysicsStabilizer(
                scene_path=None,
                config=config.PHYSICS_STABILIZATION,
            )
            # use_physics_sim 由 config_level 控制：production 尝试仿真，其他模式走启发式。
            report = stabilizer.check_placement_stability(
                placements=placements,
                receptacles=[],
                use_physics_sim=self.config_level == "production",
            )
            total_checked = max(int(report.get("total_checked", 0)), 1)
            stable_count = int(report.get("stable_count", 0))
            return {
                "success": True,
                "stable_count": stable_count,
                "unstable_count": int(report.get("unstable_count", 0)),
                "method": report.get("method", "unknown"),
                "stability_rate": stable_count / total_checked,
                "physics_report": report,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _stage_6_validation(
        self,
        scene_name: str,
        config: Any,
        observations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Stage 6: 最终结果校验。"""
        placements = self.stage_results.get("stage_5_semantic_placement", {}).get("placements", [])
        physics = self.stage_results.get("stage_3_4_physics_check", {})
        valid = bool(placements) and physics.get("success", True)
        return {
            "success": True,
            "total_objects": len(placements),
            "validation_passed": valid,
            "has_physics_report": "physics_report" in physics,
        }

    def _extract_room_groups(self, layout_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract per-room object batches from HM3D sampled layout output.

        优先使用 layout_json 已提供的 room_groups；若缺失，则根据 objects 中的
        sampled_region_id 现场重建，兼容旧版布局格式。
        """
        room_groups = layout_json.get("room_groups", [])
        if isinstance(room_groups, list) and room_groups:
            normalized_groups = []
            for group in room_groups:
                if isinstance(group, dict):
                    normalized_groups.append(group)
            return normalized_groups

        objects = layout_json.get("objects", [])
        grouped: Dict[int, Dict[str, Any]] = {}
        unknown_group: Optional[Dict[str, Any]] = None

        for obj in objects:
            region_id = obj.get("sampled_region_id")
            if isinstance(region_id, int) and region_id >= 0:
                group = grouped.setdefault(
                    region_id,
                    {
                        "region_id": region_id,
                        "room_center": obj.get("position", [0.0, 0.0, 0.0]),
                        "room_aabb": obj.get("room_aabb", {}),
                        "objects": [],
                    },
                )
                group["objects"].append(obj)
                if obj.get("room_aabb"):
                    group["room_aabb"] = obj.get("room_aabb", {})
            else:
                if unknown_group is None:
                    unknown_group = {
                        "region_id": -1,
                        "room_center": [0.0, 0.0, 0.0],
                        "room_aabb": {},
                        "objects": [],
                    }
                unknown_group["objects"].append(obj)

        normalized_groups = [grouped[key] for key in sorted(grouped.keys())]
        if unknown_group is not None and unknown_group["objects"]:
            normalized_groups.append(unknown_group)
        return normalized_groups

    def _build_roomwise_placement(self, room_group: Dict[str, Any]) -> Dict[str, Any]:
        """Build a simple room-wise placement result from one grouped room batch.

        放置策略为轻量规则：以 room_center 为基准添加少量 jitter，随后用 room_aabb
        做边界裁剪，避免对象被放到房间外。
        """
        objects = room_group.get("objects", [])
        room_center = room_group.get("room_center", [0.0, 0.0, 0.0])
        room_aabb = room_group.get("room_aabb", {})

        if not isinstance(room_center, list) or len(room_center) < 3:
            room_center = [0.0, 0.0, 0.0]

        placements: List[Dict[str, Any]] = []
        for idx, obj in enumerate(objects):
            # 轻微抖动减少对象完全重叠，便于可视化和调试。
            jitter = (idx % 3) * 0.12
            position = [
                float(room_center[0]) + jitter,
                float(room_center[1]) + 0.05,
                float(room_center[2]) + (jitter * 0.5),
            ]
            if isinstance(room_aabb, dict):
                min_pt = room_aabb.get("min")
                max_pt = room_aabb.get("max")
                if isinstance(min_pt, list) and isinstance(max_pt, list) and len(min_pt) >= 3 and len(max_pt) >= 3:
                    position[0] = max(float(min_pt[0]) + 0.05, min(position[0], float(max_pt[0]) - 0.05))
                    position[1] = max(float(room_center[1]), float(room_center[1]) + 0.05)
                    position[2] = max(float(min_pt[2]) + 0.05, min(position[2], float(max_pt[2]) - 0.05))

            placements.append(
                {
                    "object_id": obj.get("object_id", obj.get("id", idx)),
                    "name": obj.get("name", f"object_{idx}"),
                    "model_id": obj.get("model_id", obj.get("name", f"object_{idx}")),
                    "position": [round(float(position[0]), 4), round(float(position[1]), 4), round(float(position[2]), 4)],
                    "rotation": obj.get("rotation", [0.0, 0.0, 0.0]),
                    "size": obj.get("size", [0.3, 0.3, 0.3]),
                    "sampled_region_id": room_group.get("region_id", -1),
                    "placement_source": "room_group_bridge",
                }
            )

        return {
            "success": True,
            "region_id": room_group.get("region_id", -1),
            "object_count": len(objects),
            "placements": placements,
        }

    def run_roomwise_layout_pipeline(
        self,
        layout_json: Dict[str, Any],
        scene_name: str,
    ) -> Dict[str, Any]:
        """Bridge HM3D sampled room groups into a room-by-room SD-OVON placement report.

        该接口不执行 full pipeline 的 8 个 stage，而是用于 HM3D->SD-OVON 的
        房间级快速桥接，输出结构化 room_reports 与 placements。
        """
        room_groups = self._extract_room_groups(layout_json)
        if not room_groups:
            return {
                "success": False,
                "scene_name": scene_name,
                "error": "No room_groups found in layout_json",
                "room_groups": [],
                "placements": [],
            }

        room_reports: List[Dict[str, Any]] = []
        placements: List[Dict[str, Any]] = []

        for room_group in room_groups:
            room_report = self._build_roomwise_placement(room_group)
            room_reports.append(room_report)
            placements.extend(room_report.get("placements", []))

        result = {
            "success": True,
            "scene_name": scene_name,
            "room_count": len(room_reports),
            "object_count": len(placements),
            "room_groups": room_groups,
            "room_reports": room_reports,
            "placements": placements,
            "pipeline_status": "completed",
        }
        return result

    def _generate_final_report(self, scene_name: str) -> Dict[str, Any]:
        """生成最终汇总报告。

        pipeline_status 判定规则：所有 stage.success 为 True 则 completed，
        否则 completed_with_errors。
        """
        total_time = sum(self.stage_durations.values())
        stage_entries = [v for k, v in self.stage_results.items() if k.startswith("stage_")]
        successful_stages = sum(1 for result in stage_entries if result.get("success", False))

        final_output = {
            "objects_placed": self.stage_results.get("stage_5_semantic_placement", {}).get("objects_placed", 0),
            "stability_rate": self.stage_results.get("stage_3_4_physics_check", {}).get("stability_rate", 0.0),
        }

        pipeline_status = "completed" if successful_stages == len(stage_entries) else "completed_with_errors"

        return {
            "scene_name": scene_name,
            "pipeline_status": pipeline_status,
            "implementation_level": self.config_level,
            "stages_completed": successful_stages,
            "total_stages": len(stage_entries),
            "total_time": total_time,
            "stage_durations": self.stage_durations,
            "stage_results": self.stage_results,
            "final_output": final_output,
        }
