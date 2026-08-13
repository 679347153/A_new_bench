# 本项目技术报告（已执行，全链路版）

## 项目概述
本项目的主要目标是构建一个面向家庭场景物体布局变化的 benchmark。核心问题不是只在一个静态 3D 场景中放置一组物体，而是针对同一个 Habitat/HM3D 场景，批量生成多种合理但彼此不同的物体布局，用来模拟真实家庭环境中物体会随着时间推移、人类活动和使用习惯不断发生变化的现象。

在真实家庭中，同一个杯子、花瓶、闹钟、棋盘或小家具不会永远停留在固定位置：它们可能被人拿起、移动、临时放到桌面、柜架、地面或其他承载面上，也可能因为房间功能不同而呈现不同的空间分布。本项目希望把这种“动态但仍符合语义和几何约束的变化”转化为可复现的数据生成流程，从而为后续任务提供系统化评测基础。

因此，本 benchmark 的生成目标包括：
1. 对同一批物体，在同一场景中生成多个不同 layout，体现时间片之间的布局差异。
2. 保证物体位置符合房间语义，例如物体优先出现在适合其用途的房间或区域。
3. 保证物体最终落在合理承载面上，例如桌面、架子、柜面、床面或地面等可支撑实例。
4. 通过 Habitat-Sim 进行几何放置和碰撞检查，尽量避免悬空、穿模、重叠等无效布局。
5. 通过批量生成、manifest 记录和可视化检查，形成可复现、可审计、可扩展的数据集构建流程。

从整体流程看，项目先提取场景语义与房间信息，再结合物体图像和 VLM/启发式方法估计物体适合出现的房间概率；随后对每个 layout 使用不同随机种子重新采样房间、分配具体承载实例，并在承载面上完成最终放置。这样生成的一组 layout 可以被理解为同一家庭场景在不同时刻的状态快照，用于评估 embodied AI、目标导航、开放词汇物体定位、场景变化理解等任务。

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
  - `object_profiles.py`
  - `place_objects_on_instances.py`
  - `batch_generate_layouts.py`
  - `visualize_placed_layout.py`
  - `visualize_instance_pointcloud_viser.py`
  - `log_filter.py`
  - `lifespan_generate_layouts.py`
  - `lifespan_schema.py`
  - `lifespan_household_generator.py`
  - `lifespan_profiles.py`
  - `lifespan_event_generator.py`
  - `lifespan_state_engine.py`

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
        ├── object_profiles.py 统一物体尺寸/放置类别/承载面 affordance 规则
        ├── place_objects_on_instances.py Habitat-Sim 碰撞检查放置
        ├── batch_generate_layouts.py 复用缓存批量生成多个最终布局
        ├── visualize_placed_layout.py 加载最终布局检查放置效果/高度偏移
        └── visualize_instance_pointcloud_viser.py 可视化核验（调试/对齐）
    └── Lifespan 长期语义演化模式（当前为 semantic-only MVP）
        ├── resident_persona_profiles.json 候选人物画像池
        ├── lifespan_household_generator.py 场景人物分配/协作日程/月事件规划
        ├── lifespan_profiles.py 物体生命周期 profile
        ├── lifespan_event_generator.py routine/event 到 object event 展开
        ├── lifespan_state_engine.py 长期状态连续传播
        └── lifespan_generate_layouts.py 生成 household/event/state/snapshot/manifest
```

说明：
1. 你要求的目标链路就是以上树状结构。
2. 当前代码已具备主要节点能力；自动分支已支持在 `assign_objects_to_receptacle_instances.py` 中缺省 `--surfaces-json` 时自动触发承载面提取。
3. 链路通过中间 JSON（如 `object_layout`、`surfaces_json`、`assignment_plan`）进行稳定衔接。
4. Lifespan 分支已实现第一版语义轨迹生成闭环：可生成家庭人物、协作日程、洛杉矶随机月份整月事件、物体生命周期状态、snapshot request 和 semantic-only layout；下一步需要接入 Habitat 3D grounding，使其输出真实 `position/rotation`。

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
6. 自动初放使用 `object_profiles.py` 提供的半径/footprint 估计，减少不同阶段对物体尺寸理解不一致的问题。

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
4. 支持将点云导出为 `.ply`，JSON 仅保留点云文件路径字段，是自动放置分支的基础能力模块。

### 3.6 `query_room_receptacle_objects.py`：全房间可放置 instance 上表面提取
已实现能力：
1. 默认遍历全房间（不依赖单房间硬编码），支持 `--room-id` 局部处理。
2. 结合 LLM/启发式筛选可作为承载面的 instance，支持床和地面作为可放置承载体。
3. 调用 `get_instance_point_cloud(...)` 获取实例点云（上游含多级回退）。
4. 从实例点云提取 top-surface 点云；点云不足时回退 AABB 顶面采样。
5. 对上表面进行面积与尺寸有效性过滤，确保结果“可用优先、数量可降”。
6. 将上表面点云落盘为 `.ply`，JSON 中仅保留路径与摘要字段。
7. 输出房间级 `receptacle_instances` 与场景汇总统计，供后续分配/放置模块使用。
8. 当房间缺少显式 floor instance 时，自动构造 `room_floor` 合成承载面；优先使用 Habitat-Sim navmesh 点收缩到可导航地面范围，navmesh 不可用时回退房间 AABB 地面。

上表面点云提取逻辑（当前实现）：
1. 房间遍历与实例收集：按 `scene_info.rooms` 解析 `room_id` 列表，逐房间调用 `extract_room_instances(...)` 获取 `instances`。
2. 候选预过滤：对每个 instance 计算 AABB 几何特征 `(size_x, size_y, size_z, top_area_est, volume_est)`；剔除明显无效类别（如 `wall/ceiling/window/door/tap/faucet/shower/...`）与估计顶面积过小样本（`top_area_est < --candidate-min-top-area-est`，默认 `0.005 m^2`）。已知承载类别不会仅因粗略 AABB 顶面积偏小被提前剔除。
3. 排序阶段：优先 LLM，失败则启发式回退；启发式使用类别先验分数 + `top_area_est` 加分，且支持输出空集合（0 到 `--max-results`），避免“硬凑”无效承载体。
4. 实例点云获取：对入选候选调用 `get_instance_point_cloud(...)`，取 `point_cloud.points` 作为原始点集。
5. 顶面提取（Top-band）：
   - 若原始点云为空/非法，直接执行 `aabb_top_fallback`：在 AABB 顶平面 `y=max_y` 上均匀采样。
   - 若点云有效，先取最高 `Y` 带宽内点集：`band = max(0.02, y_range * 0.08)`。
   - 若点数不足，再扩大到 `wider_band = max(0.03, y_range * 0.18)`。
   - 若仍不足，则回退 AABB 顶平面采样。
   - 最终将点数裁剪到 `--surface-points-per-instance`（默认 `256`）。
6. 顶面几何摘要：
   - 统计 `point_count`、`centroid`、`bounds(min/max)`、`plane_height`。
   - 用 SVD 拟合估计法向 `normal`；若退化或朝向不稳定则回退 `[0,1,0]`。
7. 有效性过滤（关键质量门）：
   - 点数门限：`point_count >= --surface-min-points`（默认 `48`）。
   - 可用面积门限：由 `bounds` 估计 `usable_area_est = span_x * span_z`，要求 `usable_area_est >= --surface-min-area`（默认 `0.005 m^2`）。
   - 最小跨度门限：要求 `min(span_x, span_z) >= --surface-min-span`（默认 `0.02 m`）。
   - 任何一项不满足均丢弃，并在终端输出 `[Filter]` 原因日志。
8. 结果持久化：
   - 每个有效顶面写入 `surface_pointclouds/room_<room_id>_instance_<instance_id>_top_surface.ply`。
   - JSON 删除内嵌 `top_surface.points`，仅保留 `point_cloud_file`、`point_cloud_format=ply`、几何摘要与调试字段。
9. 产物接口：输出 `*_receptacle_surfaces_*.json`，下游 `assign_objects_to_receptacle_instances.py` 与 `place_objects_on_instances.py` 直接消费该结构。

关键参数（默认值）：
1. `--surface-points-per-instance=256`：每个上表面保存点数上限。
2. `--surface-min-points=48`：上表面最少点数要求。
3. `--surface-min-area=0.005`：上表面估计可用面积下限（平方米）。
4. `--surface-min-span=0.02`：上表面最小边跨度下限（米）。
5. `--candidate-min-top-area-est=0.005`：候选预筛顶面积下限（平方米）。

### 3.7 `object_profiles.py`：物体尺寸与承载面 affordance 规则
已实现能力：
1. 为每个物体提供统一的近似几何 profile：`radius / footprint_x / footprint_z / height / y_offset / placement_class`。
2. 支持 `object_profiles.json` 覆盖默认估计；若没有覆盖，则按物体名关键词稳定回退。
   - 查找位置包括项目根目录、`objects` 父目录和 `objects` 目录。
   - 支持按 `model_id`、去掉 `_4k` 的别名、补 `_4k` 的别名匹配。
3. 提供 `surface_requirement(...)`，把物体 footprint 转换为承载面最小跨度、面积和边缘余量要求。
4. 提供 `surface_affordance_score(...)`，按 `placement_class` 判断承载类别是否合理：
   - `floor_only`：轮椅、桌椅等优先且基本只允许地面。
   - `small_tabletop`：闹钟、相机、花瓶等优先桌面/柜面/架子。
   - `large_tabletop`：棋盘、茶具等优先较大的桌面/柜面。
   - `soft_surface`：枕头等优先床、沙发、椅子或地毯。
5. 下游分配和放置共用同一套 profile，减少“分配看起来合理但几何放不下”的断裂。
6. 若缺少手工 profile，会保留 `profile_source` 与 `missing_template_config` 等诊断字段，便于后续补充精确尺寸。

### 3.8 `assign_objects_to_receptacle_instances.py`：物体到 instance 分配
已实现能力：
1. 输入对象可来自 `--object-layout` 或采样函数。
2. 严格房间约束：物体只能在其 `sampled_region_id` 房间内分配。
3. 支持图文 LLM 分配与启发式回退。
4. 输出 `assignment_plan`，并可继续调用放置模块。
5. 分配前会用 `object_profiles.py` 对候选承载面做尺寸与 affordance 过滤，并把 `object_fit` 诊断写入 LLM prompt/debug。
6. LLM prompt 明确要求优先选择 `object_fit.fits=true` 且 affordance 合理的候选；LLM 输出无效时启发式回退也使用相同规则打分。
7. 当所有候选都被尺寸/affordance 过滤掉时，不会直接丢弃该物体，而是回退保留带 `object_fit` 诊断的候选，便于 LLM/启发式做最后选择并在 debug 中解释原因。

### 3.9 `place_objects_on_instances.py`：自动放置与碰撞检查
已实现能力：
1. 根据分配结果在目标 `top_surface.point_cloud_file`（PLY）上采样落点（兼容旧版 `top_surface.points`）。
2. 使用 `object profile y_offset` 对齐模型原点与承载面；仅当模板可碰撞时再使用 `spawn_height` 执行物理下落稳定。
3. 执行最小距离约束 `max(min_distance, radius_i + radius_j)`。
4. 对 `is_collidable=false` 的物体模板使用 KINEMATIC 直接放置，避免 DYNAMIC 重力步进把物体带到承载面下方。
5. 对可碰撞模板启用 Habitat-Sim 物理步进与接触碰撞检测。
6. 支持从分配计划中的 `backup_instance_ids` 依次尝试备用承载面；目标面失败时可自动尝试同房间其他合理 surface。
7. 采样落点时使用 profile 推导的 `edge_margin`，尽量避开承载面边缘，降低悬空与掉落概率。
8. 输出 `layout + auto_placement_stats + failed_objects`，每个成功放置物体记录 `placement_y_offset / template_collidable / physics_settle / placement_target_source / placement_radius / object_profile`，便于后续排查高度、碰撞和尺寸估计问题。
9. `auto_placement_stats.profile_diagnostics` 会列出缺少精确 profile 的物体，提示补充 `object_profiles.json`。

### 3.10 `batch_generate_layouts.py`：批量最终布局生成
已实现能力：
1. 面向同一场景与同一批物体图片，批量生成多个最终放置 layout。
2. 复用已有链路产物：
   - `scene_info`
   - 每个物体的房间推荐 JSON
   - 每个物体的房间概率文件
   - 全场景 receptacle surfaces JSON
3. 每个 layout 仅重新执行：
   - 根据 `base_seed + layout_index` 设置随机种子。
   - 调用 `sample_object_positions(..., mode="load")` 复用概率重新采样房间。
   - 复用 `assign_objects_to_receptacle_instances.py` 中的 LLM/启发式 helper 分配目标 instance。
   - 调用 `place_objects_on_instances(...)` 生成最终放置结果。
4. 默认 instance assignment 使用 LLM，并在整个 batch 内复用一个 SSH tunnel/client；可用 `--disable-assignment-llm` 切换启发式模式。
5. 支持 `--disable-surface-llm`、`--regenerate-room-queries`、`--regenerate-probabilities`、`--regenerate-surfaces` 控制缓存复用与重算。
6. 输出批次目录 `results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/`，包含多个 `layout_<idx>_seed_<seed>.json` 与 `manifest.json`。
7. `manifest.json` 汇总记录缓存路径、每个 layout 的 seed、输出路径、采样数量、分配数量、放置成功/失败统计与失败原因摘要。
8. 默认开启一次失败驱动 placement retry：若首轮存在 failed objects，会降低 `min_distance`、增加 `max_trials_per_object` 并重新放置整份 plan；仅当 retry 的 `placed_count` 更高时采用 retry 结果，retry 摘要写入 layout 与 manifest。可用 `--disable-placement-retry` 关闭。
9. retry 参数可调：
   - `--retry-min-distance-scale=0.8`
   - `--retry-trial-multiplier=2.0`
   - `--retry-seed-offset=10000`

执行示例：
```bash
python core/batch_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --num-layouts 10 \
  --ssh-password 666666
```

启发式快速验证：
```bash
python core/batch_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --num-layouts 2 \
  --disable-assignment-llm \
  --disable-surface-llm
```

### 3.11 `visualize_placed_layout.py`：最终布局可视化与高度调试
已实现能力：
1. 读取 `assign_objects_to_receptacle_instances.py` / `place_objects_on_instances.py` 生成的 layout JSON。
2. 使用 Habitat-Sim 加载 HM3D 场景与 `objects` 目录中的物体模板，复现已放置状态。
3. 支持交互式浏览：相机移动、切换 layout、切换物体、聚焦当前物体、保存截图。
4. 支持 `--headless` 保存总览与物体聚焦截图，用于无 GUI 环境快速验收。
5. 支持同场景多 layout 切换：
   - 默认扫描当前 layout 所在目录，按 `[` / `]` 切换上一个/下一个 layout。
   - 可用 `--layout-scan-dir results/layouts/<scene> --recursive-layout-scan` 跨多个 batch 目录比较。
6. 支持 `--debug-offset` 高度调试模式：
   - 默认 `--initial-y-offset=2.5`，加载后所有物体整体上移 2.5m，用于快速判断 layout 是否整体偏低；严格复现原始 layout 时可传 `--initial-y-offset 0`。
   - `B` 在 `selected/all` 作用域之间切换，可只调当前物体，也可让所有物体同时上下移动。
   - `U/O` 对调试作用域执行 Y 方向上/下微调。
   - HUD 显示当前物体实时偏移、调试作用域和累计 offset。
   - `M` 保存调整后的 layout；默认输出到原文件同目录的 `*_offset_debug.json`，也可用 `--output-layout` 指定。
7. 用途：当物体看起来位于承载面下方/上方时，直接在真实场景渲染中估计需要补偿的高度偏移，再把调试结果回写到 layout 供后续复查。

执行示例：
```bash
python core/visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json \
  --scene 00808-y9hTuugGdiq \
  --debug-offset --offset-step 0.02
```

严格复现原始 layout：
```bash
python core/visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --initial-y-offset 0
```

跨 batch 比较：
```bash
python core/visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --layout-scan-dir results/layouts/00808-y9hTuugGdiq \
  --recursive-layout-scan
```

### 3.12 `visualize_instance_pointcloud_viser.py`：viser 可视化核验
已实现能力：
1. 可视化 `extract_room_instances.py` 导出的 instance 点云与包围盒。
2. 可叠加场景 mesh，检查点云与场景对齐情况。
3. 支持三种匹配模式：
   - `manual`：手动角度
   - `auto`：90 度离散遍历估计朝向
   - `interactive`：终端命令循环调参
4. 支持从 `.ply/.xyz` 或 JSON 内嵌点云读取。
5. 用于自动分支调试、对齐校验与可视化验收。

### 3.13 `log_filter.py`：终端日志噪声过滤
已实现能力：
1. 过滤 Habitat/HM3D 高频噪声告警（如 `Metadata ... No Glob path result found ... unable to load templates ...`）。
2. 支持“管道模式”：从 stdin 读取日志并输出清洗结果。
3. 支持“包裹命令模式”：直接运行目标命令并实时过滤输出。
4. 支持用户自定义抑制规则（`--drop-regex`，可重复）。
5. 在 stderr 输出过滤统计（输入行数、输出行数、抑制行数、按规则计数）。

执行指引：
1. 管道过滤（已有日志文件）：
   - `python core/log_filter.py < raw.log > clean.log`
2. 实时过滤（包裹脚本运行）：
   - `python core/log_filter.py --run "python core/query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --disable-llm"`
3. 额外添加自定义噪声规则：
   - `python core/log_filter.py --run "python your_script.py" --drop-regex "some noisy regex"`
4. 关闭内置规则，仅使用自定义规则：
   - `python core/log_filter.py --no-default-rules --drop-regex "regex1" < raw.log > clean.log`
5. 不输出统计摘要：
   - `python core/log_filter.py --run "python your_script.py" --no-summary`

### 3.14 Lifespan 长期家庭语义演化模块

当前已实现第一版 semantic-only MVP，用于解决普通 `batch_generate_layouts.py` 只通过独立随机 seed 采样，难以表达长期家庭活动规律的问题。该分支不会替代现有 3D 自动放置链路，而是在其上游生成更真实的长期语义轨迹。

新增模块：
1. `lifespan_schema.py`
   - 提供 JSON 读写、schema 校验、prompt 压缩和通用工具。
   - 校验 `resident_persona_profiles.json`、`household_profile.json`、`resident_daily_routines.json`、`daily_important_events.json` 等核心结构。
2. `lifespan_household_generator.py`
   - 读取 `scene_info` 中的房间、卧室数量、家具/物体类别摘要。
   - 读取 `data/lifespan/resident_persona_profiles.json` 的 50 个候选人物画像。
   - 默认尝试通过 Qwen3-VL/LLM 生成场景特定 household；远端不可用时使用规则回退。
   - 输出人物数量、人物关系图、私人空间、共享空间、家庭习惯。
   - 为每个居民生成 weekday/weekend daily routine，并包含家庭协作事件。
   - 固定地点为 `Los Angeles, USA`，随机选择月份，并生成该月每天的重要事件。
3. `lifespan_profiles.py`
   - 从 object catalog 推断物体生命周期 profile。
   - 为物体标注 `mobility_class / home_location_type / persistence / activities / inventory`。
   - 支持 `consumable / semi_static / temporary / replaceable / routine_movable` 等长期动态类别。
4. `lifespan_event_generator.py`
   - 将 daily routine 和 monthly important events 展开为 object-level event log。
   - 当前支持 `MOVE / CONSUME / REPLENISH / CLEANUP` 等事件效果。
5. `lifespan_state_engine.py`
   - 按时间顺序执行事件，传播 object state。
   - 维护 `exists / quantity / location_state / semantic_target / condition / last_event_id`。
   - 生成 regular snapshots 的 `snapshot_requests`。
6. `lifespan_generate_layouts.py`
   - Lifespan 总入口。
   - 输出 `household_profile.json`、`resident_daily_routines.json`、`daily_important_events.json`、`event_log.json`、`state_history.json`、`snapshot_requests.json`、`manifest.json`、`validation_report.json` 和 `layouts/snapshot_*.json`。
   - 当前 layout 为 `lifespan_semantic_layout.v1`，顶层包含 `semantic_only=true`；物体 `position/rotation` 暂为 `null`，用于后续 3D grounding。

已生成/维护的数据文件：
1. `data/lifespan/resident_persona_profiles.json`
   - 50 个家庭场景候选人物画像。
   - 每条包含 `name / age / gender / occupation / personality / thoughts / routine_preferences / preferences`。
2. `data/lifespan/default_lifespan_config.json`
   - Lifespan 默认参数，包括 `duration_days / snapshots_per_day / location / month=random / resident_count_mode=infer_from_bedrooms`。
3. `data/lifespan/activity_templates.json`
   - 常见活动到物体效果的规则模板。

执行示例：
```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --duration-days 7 \
  --snapshots-per-day 07:00,12:00,18:00,22:00 \
  --object-limit 40 \
  --ssh-password 666666
```

不连接远端 Qwen 的本地 smoke test：
```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --duration-days 3 \
  --snapshots-per-day 07:00,18:00 \
  --object-limit 10 \
  --disable-lifespan-llm \
  --sequence-id smoke_lifespan_test
```

当前边界：
1. Lifespan 分支当前是 semantic-only MVP，不会直接生成可被 Habitat-Sim 加载的最终 3D pose。
2. 如果找不到真实 `scene_info`，脚本会使用 minimal fallback scene summary，并输出 warning；真实数据集生成建议先执行 `export_scene_info.py` 或通过 `--scene-info` 显式传入。
3. 下一步应将 `snapshot_requests.json` 接到 `assign_objects_to_receptacle_instances.py` 和 `place_objects_on_instances.py`，实现 changed objects 的 3D grounding，并复用 unchanged objects 的上一 snapshot pose。

## 4. 数据流与产物

### 4.1 主干产物
1. 场景信息：`scene_info_export/<scene>_scene_info.json` 或 `results/scene_info/...`
2. 房间推荐：`results/scene_info/<scene>/<object>_rooms.json`
3. 概率文件：`results/probabilities/<scene>/<object>_probs.json`
4. 初放布局：`results/layouts/<scene>/temp_*.json`
5. 可选物体 profile 覆盖：`object_profiles.json`

### 4.2 分支产物
1. 手动分支：
   - 编辑后布局：`results/layouts/<scene>/final_*.json`
2. 自动分支：
   - 房间实例与点云：`results/room_instances/<scene>/...`
   - 承载面结果：`results/receptacle_queries/<scene>/*_receptacle_surfaces_*.json`
   - 分配计划：`results/object_instance_assignments/<scene>/*_object_instance_plan.json`
   - 自动布局：`results/layouts/<scene>/*assigned_instance_layout*.json`
   - 批量布局：`results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/layout_*_seed_*.json`
   - 批量索引：`results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/manifest.json`
   - 可视化高度调试布局：`results/layouts/<scene>/*_offset_debug.json`
   - 自动放置统计：layout 顶层 `auto_placement_stats`，包含 `failed_by_reason / failed_objects / profile_diagnostics`
   - 批量 retry 记录：layout 顶层 `batch_generation.placement_retry` 与 manifest 中每个 layout 的 `placement_retry`
3. Lifespan 语义演化分支：
   - 候选人物画像：`data/lifespan/resident_persona_profiles.json`
   - 默认配置：`data/lifespan/default_lifespan_config.json`
   - 活动模板：`data/lifespan/activity_templates.json`
   - 长期序列目录：`results/lifespan/<scene>/lifespan_<YYYYmmdd_HHMMSS>/`
   - household：`household_profile.json`
   - 人物关系：`household_relationship_graph.json`
   - daily routine：`resident_daily_routines.json`
   - 月度事件：`monthly_calendar.json`、`daily_important_events.json`
   - object event：`event_log.json`
   - 状态历史：`state_history.json`
   - snapshot 请求：`snapshot_requests.json`
   - semantic-only layout：`layouts/snapshot_*_day_*.json`
   - 长期序列索引：`manifest.json`
   - 验证报告：`validation_report.json`

### 4.3 终端输出治理产物
1. 日志清洗脚本：`log_filter.py`
2. 可选清洗输出：用户可自行重定向为 `clean.log`（例如 `python core/log_filter.py < raw.log > clean.log`）

## 5. 工作流状态（现状与目标）

### 5.1 目标工作流（你定义的规划）
1. 主干统一：场景信息导出 -> Qwen 房间推荐 -> 概率采样与自动初放。
2. 主干后分叉：
   - 手动微调模式（`test_layout.py`）。
   - 自动放置模式（instance 提取 -> 承载面提取 -> 分配 -> 放置）。

### 5.2 当前实现状态
1. 两个分支核心能力都已实现。
2. 主干到自动分支可通过中间 JSON 衔接，语义约束可保持一致。
3. 自动分支已支持“分配脚本内自动补齐承载面查询”。
4. 批量生成已由 `batch_generate_layouts.py` 编排：同一场景同一批物体可以复用 scene_info、概率、承载面结果，并通过不同 seed 生成多个最终 layout。
5. 放置准确性增强已接入主链路：`object_profiles.py` 统一尺寸估计，assignment 使用 affordance/几何过滤，placement 支持备用承载面和失败驱动 retry。
6. 验收工具已支持同场景多 layout 切换、跨 batch 扫描和 selected/all 高度偏移调试。
7. Lifespan semantic-only MVP 已实现：可生成 scene-specific household、协作 daily routine、洛杉矶随机月份每日事件、object event log、state history、snapshot requests 与 semantic layout manifest。
8. Lifespan 3D grounding 尚未接入：当前 Lifespan layout 中 `position/rotation=null`，`semantic_only=true`，不能直接作为 Habitat 物理放置结果。

## 6. 与本次需求对照
1. 技术报告新增 `extract_room_instances.py`：已完成。
2. 技术报告新增 `visualize_instance_pointcloud_viser.py`：已完成。
3. 技术报告新增 `visualize_placed_layout.py`：已完成。
4. 技术报告新增 `batch_generate_layouts.py`：已完成。
5. 全链路概览改为树状结构：已完成。
6. 概览前段统一为“场景信息导出 -> Qwen 房间推荐 -> 概率采样与自动初放”：已完成。
7. 后段改为手动微调模式与自动放置模式两分支，并标注打通状态：已完成。
8. 自动放置后的高度偏移调试流程：已完成。
9. 同场景同物体批量生成最终 layout 流程：已完成。
10. 自动放置准确性增强：已完成，包括统一 object profile、承载面 affordance、navmesh 合成地面、备用承载面尝试与 placement retry。
11. 报告已根据当前代码更新默认参数、产物字段和可视化调试能力。
12. Lifespan 执行计划与第一版语义生成器：已完成，包括 50 个候选人物画像、默认配置、活动模板、长期事件与状态传播。
13. Lifespan 最终 3D layout grounding：未完成，是下一阶段核心任务。

## 7. 结论
- 报告现已与你定义的“树状主干+双分支”方案对齐。
- 自动分支已形成可复用、可批量、可调试的闭环：承载面提取、实例分配、物理放置、失败诊断、retry、可视化验收均已在报告中描述。
- Lifespan 分支已形成长期语义轨迹闭环，但仍是 semantic-only；要成为最终 benchmark layout，还需要将 `snapshot_requests.json` ground 到具体 receptacle instance 和 3D pose。
- 当前最值得继续迭代的数据资产是 `object_profiles.json` 与 `data/lifespan/resident_persona_profiles.json`：前者提升物体几何放置稳定性，后者提升家庭人物与长期行为多样性。
