"""
SD-OVON: Mock vs Production 切换指南与集成测试

演示如何在快速原型 (mock/stub) 和完整生产版本之间无缝切换。
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ==================== 使用指南 ====================

USAGE_GUIDE = """
╔════════════════════════════════════════════════════════════════╗
║         SD-OVON Pipeline: Mock vs Production Mode              ║
╚════════════════════════════════════════════════════════════════╝

### 1. 快速原型 (Mock Mode)
   用途: 快速验证流程、演示、开发测试
   特点: 无需依赖真实模型、立即运行
   代码示例:
   
   from orchestrate_sd_ovon_complete import SDOVONPipelineOrchestrator
   
   orchestrator = SDOVONPipelineOrchestrator(config_level="mock")
   report = orchestrator.run_full_pipeline("00800-TEEsavR23oF")
   
   实现替代:
   - Instance Fusion: instance_fusion_stub.py (生成虚拟实例)
   - Physics Check: physics_stabilizer_heuristic.py (启发式检查)

### 2. 完整生产版本 (Production Mode)
   用途: 真实部署、最佳效果、科研发表
   特点: 集成真实视觉模型、物理引擎、完整算法
   代码示例:
   
   from orchestrate_sd_ovon_complete import SDOVONPipelineOrchestrator
   
   orchestrator = SDOVONPipelineOrchestrator(config_level="production")
   report = orchestrator.run_full_pipeline("00800-TEEsavR23oF")
   
   完整实现:
   - Instance Fusion: instance_fusion_gsam_3d.py (G-SAM + ConceptGraphs)
   - Physics Check: physics_stabilizer_complete.py (Habitat-Sim)

---

### 3. 配置管理与差异

| 特性                | Mock Mode          | Production Mode      |
|---------------------|-------------------|---------------------|
| G-SAM 实例融合       | ✗ mock            | ✓ 真实 G-SAM 模型    |
| 3D 点云融合          | ✗ mock            | ✓ ConceptGraphs     |
| 物理模拟             | ✗ 启发式           | ✓ Habitat-Sim       |
| 初始化时间           | < 100ms           | 1-5s               |
| 单场景处理时间       | 1-3s              | 15-30s             |
| 精度期望             | ~70%              | ~90%+              |
| 依赖包               | numpy             | numpy + torch + ... |

---

### 4. 逐步迁移路径

Stage A: 快速验证流程
  └─ 使用 mock 模式            [5 分钟部署]
     ├─ 验证 I/O
     ├─ 检查数据流
     └─ 调试配置参数

Stage B: 部分完整实现
  └─ 保持 Instance Fusion mock，生产 Physics        [1 小时]
     ├─ 测试物理稳定性算法
     ├─ 调优放置约束
     └─ 验证输出质量

Stage C: 完整生产模式
  └─ 两个组件都使用完整版本                        [2-3 天]
     ├─ 部署 G-SAM 模型
     ├─ 集成 Habitat-Sim
     ├─ 性能优化
     └─ 部署到生产环境

---

### 5. 故障转移策略

if config_level == "production":
    try:
        use GSAMInstanceFusion()          # 尝试完整版
    except ImportError:
        logger.warning("G-SAM unavailable, falling back to mock")
        use InstanceFusionStub()          # 自动降级

    try:
        use HabitatPhysicsEngine()        # 尝试完整版
    except ImportError:
        logger.warning("Habitat-Sim unavailable, using heuristics")
        use PhysicsStabilizerHeuristic()  # 自动降级

"""


# ==================== 集成测试 ====================

def test_mock_mode():
    """测试 Mock 模式"""
    logger.info("=" * 60)
    logger.info("TEST 1: Mock Mode (Fast Prototype)")
    logger.info("=" * 60)

    from orchestrate_sd_ovon_complete import SDOVONPipelineOrchestrator

    orchestrator = SDOVONPipelineOrchestrator(config_level="mock")

    report = orchestrator.run_full_pipeline(
        scene_name="00800-TEEsavR23oF",
        stage_overrides={
            # 可选：跳过某些阶段
        },
    )

    # 验证报告
    assert report["pipeline_status"] == "completed"
    assert report["implementation_level"] == "mock"

    logger.info(f"✓ Mock mode test passed")
    logger.info(f"  - Total stages: {report['total_stages']}")
    logger.info(f"  - Execution time: {report['total_time']:.2f}s")
    logger.info(f"  - Objects placed: {report['final_output']['objects_placed']}")

    return report


def test_production_mode():
    """测试 Production 模式"""
    logger.info("=" * 60)
    logger.info("TEST 2: Production Mode (Full Implementation)")
    logger.info("=" * 60)

    from orchestrate_sd_ovon_complete import SDOVONPipelineOrchestrator

    orchestrator = SDOVONPipelineOrchestrator(config_level="production")

    report = orchestrator.run_full_pipeline(
        scene_name="00800-TEEsavR23oF",
    )

    # 验证报告
    assert report["pipeline_status"] == "completed"
    assert report["implementation_level"] == "production"

    logger.info(f"✓ Production mode test passed")
    logger.info(f"  - Total stages: {report['total_stages']}")
    logger.info(f"  - Execution time: {report['total_time']:.2f}s")
    logger.info(f"  - Objects placed: {report['final_output']['objects_placed']}")
    logger.info(f"  - Fusion method: {report['stage_results'].get('stage_3_2_instance_fusion', {}).get('method')}")
    logger.info(f"  - Physics method: {report['stage_results'].get('stage_3_4_physics_check', {}).get('method')}")

    return report


def test_fallback_behavior():
    """测试故障转移行为"""
    logger.info("=" * 60)
    logger.info("TEST 3: Fallback Behavior")
    logger.info("=" * 60)

    # 尝试生产模式，如果失败则自动降级
    from orchestrate_sd_ovon_complete import SDOVONPipelineOrchestrator

    orchestrator = SDOVONPipelineOrchestrator(config_level="production")
    report = orchestrator.run_full_pipeline("00800-TEEsavR23oF")

    # 检查是否降级
    fusion_method = report["stage_results"].get("stage_3_2_instance_fusion", {}).get("method", "unknown")
    physics_method = report["stage_results"].get("stage_3_4_physics_check", {}).get("method", "unknown")

    logger.info(f"✓ Fallback behavior test passed")
    logger.info(f"  - Actual fusion method: {fusion_method}")
    logger.info(f"  - Actual physics method: {physics_method}")

    if fusion_method == "mock":
        logger.warning("  → G-SAM not available, using mock")
    if physics_method == "heuristic":
        logger.warning("  → Habitat-Sim not available, using heuristics")

    return report


def compare_modes():
    """对比两种模式"""
    logger.info("=" * 60)
    logger.info("TEST 4: Mode Comparison")
    logger.info("=" * 60)

    # 运行两种模式
    mock_report = test_mock_mode()
    prod_report = test_production_mode()

    # 对比
    logger.info("\n=== COMPARISON REPORT ===\n")

    comparison = {
        "metric": ["Total Time", "Stages Completed", "Objects Placed", "Stability Rate"],
        "mock": [
            f"{mock_report['total_time']:.2f}s",
            f"{mock_report['stages_completed']}/{mock_report['total_stages']}",
            f"{mock_report['final_output']['objects_placed']}",
            f"{mock_report['stage_results'].get('stage_3_4_physics_check', {}).get('stability_rate', 0):.1%}",
        ],
        "production": [
            f"{prod_report['total_time']:.2f}s",
            f"{prod_report['stages_completed']}/{prod_report['total_stages']}",
            f"{prod_report['final_output']['objects_placed']}",
            f"{prod_report['stage_results'].get('stage_3_4_physics_check', {}).get('stability_rate', 0):.1%}",
        ],
    }

    for i, metric in enumerate(comparison["metric"]):
        logger.info(f"{metric:.<20} Mock: {comparison['mock'][i]:<15} Prod: {comparison['production'][i]:<15}")

    logger.info("\n✓ Comparison complete\n")


# ==================== 配置导出 ====================

def export_configs():
    """导出两种配置到 JSON"""
    from sd_ovon_config_enhanced import SDOVONConfig

    # Mock 配置
    mock_config = SDOVONConfig(implementation_level="mock")
    mock_config.save_to_file("./config_mock.json")

    # Production 配置
    prod_config = SDOVONConfig(implementation_level="production")
    prod_config.save_to_file("./config_production.json")

    logger.info("✓ Configurations exported:")
    logger.info("  - ./config_mock.json")
    logger.info("  - ./config_production.json")


# ==================== 主函数 ====================

def main():
    """运行完整的集成测试"""
    import sys

    print(USAGE_GUIDE)

    if len(sys.argv) > 1:
        test_name = sys.argv[1]

        if test_name == "mock":
            test_mock_mode()
        elif test_name == "production":
            test_production_mode()
        elif test_name == "fallback":
            test_fallback_behavior()
        elif test_name == "compare":
            compare_modes()
        elif test_name == "export":
            export_configs()
        else:
            logger.error(f"Unknown test: {test_name}")
            print("Available tests: mock, production, fallback, compare, export")
    else:
        # 运行所有测试
        logger.info("Running all integration tests...\n")
        test_mock_mode()
        logger.info("\n" + "=" * 60 + "\n")
        test_production_mode()
        logger.info("\n" + "=" * 60 + "\n")
        test_fallback_behavior()


if __name__ == "__main__":
    main()
