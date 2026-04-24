# 本项目技术报告（已执行，全链路版）

## 1. 报告说明
- 本文档为已执行技术报告，不是计划文档。
- 报告基于当前代码实现整理，覆盖以下核心文件：
  - `export_scene_info.py`
  - `query_rooms_for_objects.py`
  - `sample_and_place_objects.py`
  - `test_layout.py`
  - `extract_room_instances.py`
  - `query_room_receptacle_objects.py`
  - `assign_objects_to_receptacle_instances.py`
  - `place_objects_on_instances.py`
  - `visualize_instance_pointcloud_viser.py`

## 2. 全链路概览（树状）
```text
统一前段（主干）
└── 场景信息导出 -> Qwen 房间推荐 -> 概率采样与自动初放
    ├── 手动微调模式
    │   └── test_layout.py 交互式编辑并保存最终布局
    └── 自动放置模式
        ├── extract_room_instances.py 提取房间 instances 与单 instance 点云
        ├── query_room_receptacle_objects.py 提取可放置 instance 上表面
        ├── assign_objects_to_receptacle_instances.py 物体到 instance 分配
        ├── place_objects_on_instances.py Habitat-Sim 碰撞检查放置
        └── visualize_instance_pointcloud_viser.py 可视化核验（调试/对齐）
```

说明：
1. 你要求的目标链路就是以上树状结构。
2. 当前代码已具备主要节点能力，但“主干到自动放置分支”的一键式编排仍属于待完全打通状态。
3. 目前可通过中间 JSON（如 `object_layout`、`surfaces_json`、`assignment_plan`）进行链路衔接。

## 3. 模块实施内容

### 3.1 `export_scene_info.py`：提取场景信息
已实现能力：
1. 解析 `semantic.txt`，建立 `semantic_id -> category / region_id / color_hex` 映射。
2. 通过 Habitat-Sim 读取场景语义对象与几何包围盒信息。
3. 汇总 `scene_info / categories / rooms` 等场景基础数据。
4. 支持单场景导出和全量导出（`--scene` / `--all`）。
5. 输出可作为后续房间推荐与放置流程输入。

### 3.2 `query_rooms_for_objects.py`：基于场景信息询问 Qwen3-VL
已实现能力：
1. 读取或现场导出 `scene_info`。
2. 遍历物体图片并发起图文查询。
3. 强约束模型只从候选房间中输出推荐结果（JSON）。
4. 对模型输出做 JSON 优先解析和回退解析，保证可用结果。
5. 输出 `results/scene_info/<scene>/<object>_rooms.json`。

### 3.3 `sample_and_place_objects.py`：概率采样与自动初放
已实现能力：
1. 读取/生成每个物体的房间概率文件。
2. 按概率采样房间，生成 `sampled_region_id` 与初始布局对象。
3. 支持 `placement=auto` 自动初放（房间内采样+碰撞约束）。
4. 支持 `placement=manual`，将布局交给编辑器人工微调。
5. 支持迭代式 `sample -> edit -> save` 流程。

### 3.4 `test_layout.py`：手动微调模式
已实现能力：
1. 提供交互式布局编辑（增删选、平移、旋转、切换布局）。
2. 支持房间感知选择（room-aware selection）。
3. 支持保存最终布局 JSON（`M`）。
4. 适合作为主干后的“人工精修分支”。

### 3.5 `extract_room_instances.py`：从房间提取 instances 与点云
已实现能力：
1. 房间级查询：
   - 输入 `scene + room_id`，返回该房间全部 `instances`。
2. 单 instance 查询：
   - 输入 `instance_id`，返回该实例点云摘要与生成链路信息。
3. 点云提取多级回退：
   - semantic mesh 颜色/空间裁剪
   - habitat-sim 直接字段
   - stage mesh 语义颜色
   - AABB/OBB 采样回退
4. 支持将点云导出为 `.ply/.xyz`，是自动放置分支的基础能力模块。

### 3.6 `query_room_receptacle_objects.py`：全房间可放置 instance 上表面提取
已实现能力：
1. 默认遍历全房间（不依赖单房间硬编码）。
2. 结合 LLM/启发式筛选可作为承载面的 instance。
3. 调用 `get_instance_point_cloud(...)` 获取实例点云。
4. 提取 top-surface 点云，必要时回退 AABB 顶面采样。
5. 输出房间级 `receptacle_instances` 与场景汇总统计。

### 3.7 `assign_objects_to_receptacle_instances.py`：物体到 instance 分配
已实现能力：
1. 输入对象可来自 `--object-layout` 或采样函数。
2. 严格房间约束：物体只能在其 `sampled_region_id` 房间内分配。
3. 支持图文 LLM 分配与启发式回退。
4. 输出 `assignment_plan`，并可继续调用放置模块。

### 3.8 `place_objects_on_instances.py`：自动放置与碰撞检查
已实现能力：
1. 根据分配结果在目标 `top_surface.points` 上采样落点。
2. 使用 `spawn_height` 上抬策略（默认 `0.3m`）。
3. 执行最小距离约束 `max(min_distance, radius_i + radius_j)`。
4. 启用 Habitat-Sim 物理步进与接触碰撞检测。
5. 输出 `layout + auto_placement_stats + failed_objects`。

### 3.9 `visualize_instance_pointcloud_viser.py`：viser 可视化核验
已实现能力：
1. 可视化 `extract_room_instances.py` 导出的 instance 点云与包围盒。
2. 可叠加场景 mesh，检查点云与场景对齐情况。
3. 支持三种匹配模式：
   - `manual`：手动角度
   - `auto`：90 度离散遍历估计朝向
   - `interactive`：终端命令循环调参
4. 支持从 `.ply/.xyz` 或 JSON 内嵌点云读取。
5. 用于自动分支调试、对齐校验与可视化验收。

## 4. 数据流与产物

### 4.1 主干产物
1. 场景信息：`scene_info_export/<scene>_scene_info.json` 或 `results/scene_info/...`
2. 房间推荐：`results/scene_info/<scene>/<object>_rooms.json`
3. 概率文件：`results/probabilities/<scene>/<object>_probs.json`
4. 初放布局：`results/layouts/<scene>/temp_*.json`

### 4.2 分支产物
1. 手动分支：
   - 编辑后布局：`results/layouts/<scene>/final_*.json`
2. 自动分支：
   - 房间实例与点云：`results/room_instances/<scene>/...`
   - 承载面结果：`results/receptacle_queries/<scene>/*_receptacle_surfaces_*.json`
   - 分配计划：`results/object_instance_assignments/<scene>/*_object_instance_plan.json`
   - 自动布局：`results/layouts/<scene>/*assigned_instance_layout*.json`

## 5. 工作流状态（现状与目标）

### 5.1 目标工作流（你定义的规划）
1. 主干统一：场景信息导出 -> Qwen 房间推荐 -> 概率采样与自动初放。
2. 主干后分叉：
   - 手动微调模式（`test_layout.py`）。
   - 自动放置模式（instance 提取 -> 承载面提取 -> 分配 -> 放置）。

### 5.2 当前实现状态
1. 两个分支核心能力都已实现。
2. 主干到自动分支可通过中间 JSON 衔接，语义约束可保持一致。
3. 一键化编排与完全无缝串联仍有整合空间。

## 6. 与本次需求对照
1. 技术报告新增 `extract_room_instances.py`：已完成。
2. 技术报告新增 `visualize_instance_pointcloud_viser.py`：已完成。
3. 全链路概览改为树状结构：已完成。
4. 概览前段统一为“场景信息导出 -> Qwen 房间推荐 -> 概率采样与自动初放”：已完成。
5. 后段改为手动微调模式与自动放置模式两分支，并标注打通状态：已完成。

## 7. 结论
- 报告现已与你定义的“树状主干+双分支”方案对齐。
- 自动分支的调试与验收链（`extract_room_instances.py` + `visualize_instance_pointcloud_viser.py`）已在报告中补齐。
