# SD-OVON 流程完整文档

Semantic Data Object-Video Optimization Network

## 📋 概览

SD-OVON 是一个集成了语义分割、物体识别和空间优化的室内场景生成流程。

**核心特性：**
- 🎨 基于扩散模型的语义理解
- 🚀 支持大规模场景合成
- ✅ 自动质量验证和去重
- 📊 可视化输出和分析工具

---

## 🏗️ 系统架构

```
Input: Scene + Semantic Annotations
    ↓
[Step 1: Semantic Understanding] → semantic_router.py
    - 加载场景和语义标签
    - 提取房间类型和用途
    ↓
[Step 2: Preprocess Features] → preprocess_features.py
    - 提取房间特征（大小、形状等）
    - 生成特征编码
    ↓
[Step 3: Generate Objects] → generate_sd_ovon_objects.py
    - 使用扩散模型生成候选物体
    - 评分和排序
    ↓
[Step 4: Deduplicate] → deduplicate_objects.py
    - 去除重复物体
    - 保留最佳候选
    ↓
[Step 5: Semantic Placement] → semantic_placement.py
    - 优化物体位置
    - 检查碰撞和约束
    ↓
[Step 6: Validate] → validate_sd_ovon_outputs.py
    - 验证输出质量
    - 生成报告
    ↓
Output: Positioned Objects Layout + Validation Report
```

---

## 📁 文件结构说明

### 核心配置
- **sd_ovon_config.py**：集中的配置管理
  - 扩散模型参数
  - 语义映射表
  - 放置约束条件

### 数据流处理
- **semantic_router.py**：第 1 阶段（语义理解）
- **preprocess_features.py**：第 2 阶段（特征提取）
- **generate_sd_ovon_objects.py**：第 3 阶段（物体生成）
- **deduplicate_objects.py**：第 4 阶段（去重处理）
- **semantic_placement.py**：第 5 阶段（空间优化）

### 工具和集成
- **orchestrate_sd_ovon.py**：编排入口（Phase E）
- **validate_sd_ovon_outputs.py**：验证工具
- **visualize_sd_ovon_results.py**：可视化工具
- **sample_and_place_objects.py**：与 Habitat 集成

---

## 🚀 快速开始

### 1. 基础配置

编辑 `sd_ovon_config.py`，根据需要调整参数：

```python
# 模型参数
DIFFUSION_MODEL = {
    "model_name": "stable-diffusion-v1-5",
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
}

# 语义映射
SEMANTIC_CATEGORIES = {
    "0": "wall",
    "1": "chair",
    "2": "table",
    # ...
}

# 放置约束
SEMANTIC_PLACEMENT = {
    "min_distance_between_objects": 0.3,
    "height_threshold": 0.1,
    "collision_check": True,
}
```

### 2. 单阶段运行

```python
# 只运行语义理解
from semantic_router import SemanticRouter
router = SemanticRouter()
scene_data = router.route_scene("scene_id")

# 只运行特征提取
from preprocess_features import FeaturePreprocessor
preprocessor = FeaturePreprocessor()
features = preprocessor.preprocess("scene_id")
```

### 3. 完整流程运行

```python
from orchestrate_sd_ovon import orchestrate_full_pipeline

result = orchestrate_full_pipeline(
    scene_name="00800-TEEsavR23oF",
    config_override=None,
    enable_visualization=True,
    validate_outputs=True,
)

print(result)
# Output:
# {
#   "success": True,
#   "stages_completed": 6,
#   "objects_placed": 42,
#   "validation_report": {...},
#   "output_files": {...}
# }
```

### 4. 验证和可视化

```python
from validate_sd_ovon_outputs import validate_sd_ovon_outputs
from visualize_sd_ovon_results import visualize_results

# 验证
report = validate_sd_ovon_outputs("00800-TEEsavR23oF")

# 可视化
visualize_results(
    "00800-TEEsavR23oF",
    show_semantics=True,
    show_objects=True,
    save_png=True,
)
```

---

## ⚙️ 配置详解

### DIFFUSION_MODEL（扩散模型）

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `model_name` | 模型名称 | `stable-diffusion-v1-5` |
| `guidance_scale` | 引导尺度（越大越符合提示） | `7.5` |
| `num_inference_steps` | 推理步数（越多质量越好，速度越慢） | `50` |
| `seed` | 随机种子 | `42` |

### SEMANTIC_PLACEMENT（语义放置）

| 参数 | 说明 | 默认值 |
|------|------|-------|
| `min_distance_between_objects` | 物体间最小距离（米） | `0.3` |
| `height_threshold` | 高度偏差阈值（米） | `0.1` |
| `collision_check` | 是否检查碰撞 | `True` |
| `max_iterations` | 最大放置迭代次数 | `100` |

---

## 🔍 输出文件说明

所有输出保存在配置指定的目录中：

### Layouts (输出目录/layouts/)
- 格式：JSON
- 内容：
  ```json
  {
    "scene_id": "00800-TEEsavR23oF",
    "objects": [
      {
        "id": "obj_0",
        "category": "chair",
        "position": [1.5, 0.0, 2.0],
        "rotation": [0, 0, 0],
        "confidence": 0.92,
        "source_stage": "semantic_placement"
      }
    ],
    "sd_ovon_stats": {
      "total_objects": 50,
      "placed_count": 48,
      "failed_count": 2,
      "processing_time_seconds": 15.3
    }
  }
  ```

### Validation Reports (输出目录/reports/)
- 格式：JSON
- 内容：验证结果、问题列表、统计信息

### Visualizations (输出目录/visualizations/)
- 格式：PNG
- 内容：语义分割地图 + 物体位置叠加

---

## 📊 与 Habitat 集成

使用 `sample_and_place_objects.py` 将生成的物体集成到 Habitat 场景：

```python
from sample_and_place_objects import SampleAndPlaceObjects

sampler = SampleAndPlaceObjects(
    scene_name="00800-TEEsavR23oF",
    backend="sd_ovon",  # 新增：使用 SD-OVON 后端
    use_ai_sampling=True,
)

# 方式 1：自动采样从 SD-OVON 输出
success = sampler.sample_and_place_objects(
    num_samples=42,
    source="sd_ovon_layout",  # 从 SD-OVON 生成的布局
)

# 方式 2：获取启用了 SD-OVON 的采样器进行自定义处理
sampler.initialize_sd_ovon_backend()
layout = sampler.load_sd_ovon_layout()
```

---

## 🐛 常见问题排查

### Q: 物体生成失败率很高怎么办？
**A:** 检查语义标签的质量。如果标签不完整或错误，生成会失败。可以：
1. 增加 `guidance_scale` 来增强模型引导
2. 增加 `num_inference_steps` 来提高生成质量
3. 检查 `SEMANTIC_CATEGORIES` 是否正确映射

### Q: 物体放置位置不对怎么办？
**A:** 调整 `SEMANTIC_PLACEMENT` 参数：
1. 增加 `min_distance_between_objects` 避免拥挤
2. 减少 `height_threshold` 对高度的容差
3. 启用 `collision_check` 防止碰撞

### Q: 处理速度太慢怎么办？
**A:** 性能优化建议：
1. 减少 `num_inference_steps`（质量 vs 速度权衡）
2. 使用 GPU 加速（CUDA）
3. 批量处理多个场景

---

## 📈 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| 单场景处理时间 | ~15-30s | 包括所有 6 阶段 |
| 典型物体生成数 | 40-60 个 | 根据房间大小 |
| 放置成功率 | 90-95% | 合理配置下 |
| 内存占用 | ~3-5 GB | GPU 内存 |

---

## 🔗 相关文档

- [Habitat-Sim Documentation](https://github.com/facebookresearch/habitat-sim)
- [Stable Diffusion](https://github.com/replicate/cog-stable-diffusion)
- [Scene Graphs](https://svl.stanford.edu/projects/wsvd/)

---

**维护者**：SD-OVON Team  
**最后更新**：2024  
**版本**：1.0
