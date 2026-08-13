# Lifespan 动态家庭布局生成执行计划

## 1. 背景与目标

当前项目已经能够针对同一 HM3D 场景和同一批物体，通过 Qwen3-VL 生成房间概率、重新采样、分配 receptacle instance，并用 Habitat-Sim 完成自动放置。该流程可以批量得到多个 layout，但其核心变化主要来自随机 seed 下的独立概率采样：

```text
static room probability -> random sample -> receptacle assignment -> physical placement
```

这种方式可以生成“不同布局”，但还不能充分模拟真实家庭中物体随时间变化的规律。例如，杯子不是每天随机出现在任意适合房间，而是会受到居民作息、个人习惯、活动事件、库存消耗、清洁整理、购物补货、物体损坏和新物品引入等因素驱动。

因此，本阶段目标是在现有项目上增加一层 **lifespan household dynamics** 编排，使批量 layout 不再是彼此独立的随机快照，而是一条可追溯的家庭演化轨迹：

```text
S_t0 -> event_1 -> S_t1 -> event_2 -> ... -> S_tK
```

最终希望机器人在同一场景中多次探索后，能够学习：

```text
P(object existence, location, quantity | time, history, household context)
```

而不是只学习一组静态或独立随机的位置分布。

## 2. 与现有项目的关系

Lifespan 模块不替代现有自动放置链路，而是作为其上层时间模拟器。现有模块继续承担场景解析、语义概率、承载面提取、instance 分配和几何放置。

现有能力复用关系如下：

| 现有模块 | 在 Lifespan 流程中的职责 |
| --- | --- |
| `core/export_scene_info.py` | 解析 HM3D 场景、房间、语义实例，作为长期家庭环境的静态结构 |
| `core/query_rooms_for_objects.py` | 生成物体适合房间的语义先验，不再直接代表每个时间点的最终概率 |
| `core/sample_and_place_objects.py` | 复用概率生成逻辑，作为初始 home-base 和候选位置分布来源 |
| `core/query_room_receptacle_objects.py` | 提取每个房间可承载物体的具体 instance surface |
| `core/object_profiles.py` | 复用物体尺寸、放置类别、affordance 规则，并扩展 lifespan 属性 |
| `core/assign_objects_to_receptacle_instances.py` | 将语义目标 room/receptacle grounded 到具体 instance |
| `core/place_objects_on_instances.py` | 完成最终 3D pose 放置、碰撞检查和失败统计 |
| `core/batch_generate_layouts.py` | 现有批量生成入口，可作为普通 independent baseline，也可被 lifespan 入口调用 |
| `benchmark/build_episodes.py` | 后续读取 lifespan layout manifest，生成长期导航任务 |
| `core/visualize_placed_layout.py` | 可视化连续 layout，检查同一家庭状态演化是否合理 |

新增 lifespan 层的核心变化是：先生成“家庭、居民、事件、物体生命周期和状态转移”，再调用现有放置链路生成每个时间点的 layout。

## 3. 总体执行架构

新增流程建议为：

```text
prepare static scene assets
  -> LLM reads scene_info and all candidate resident personas
  -> assign a scene-specific household by bedroom count and room layout
  -> build household relationship graph
  -> build object lifespan profiles
  -> LLM generates relationship-aware daily routines with collaboration
  -> LLM randomly selects a month for Los Angeles, USA
  -> LLM generates important daily events for the whole month
  -> expand activities into object events
  -> propagate object states chronologically
  -> export semantic snapshots
  -> ground changed objects into concrete receptacle instances
  -> run physical placement
  -> write layouts, lifespan manifest, event log, validation report
```

对应到项目中的推荐入口：

```text
core/lifespan_generate_layouts.py
```

该入口负责完整编排，默认复用现有缓存：

```text
scene_info
room recommendation
probabilities
receptacle surfaces
object catalog
object profiles
resident persona profiles
```

## 4. 推荐新增文件

### 4.1 `core/lifespan_schema.py`

定义所有长期动态相关的数据结构和 JSON schema 校验逻辑。

建议包含：

```text
HouseholdProfile
ResidentProfile
ObjectLifespanProfile
ActivityTemplate
HouseholdEvent
ObjectState
SnapshotState
LifespanManifest
ValidationReport
```

核心字段：

```json
{
  "object_id": "mug_A_001",
  "model_id": "mug_4k",
  "category": "mug",
  "exists": true,
  "room_id": 4,
  "receptacle_instance_id": 584,
  "position": [0.0, 0.0, 0.0],
  "rotation": [1.0, 0.0, 0.0, 0.0],
  "condition": "normal",
  "owner": "resident_0",
  "location_state": "home",
  "last_changed_at": "day_03_08:20"
}
```

### 4.2 `core/lifespan_profiles.py`

在现有 `object_profiles.py` 基础上补充长期动态属性。

建议每个物体增加：

```text
mobility_class:
  fixed
  semi_static
  routine_movable
  frequently_movable
  consumable
  temporary
  replaceable

home_location_type:
  private_room
  shared_room
  storage
  kitchen_storage
  tabletop
  floor

persistence:
  p_stay
  p_move_on_activity
  p_return_home
  p_clean_up
  p_disappear

activities:
  breakfast
  dinner
  work
  leisure
  cleaning
  shopping
  guest_event
```

输出建议保存到：

```text
data/lifespan/object_lifespan_profiles.json
```

如果物体来自 `ycb-v1.2` 或 `hssd-hab-v0.2.3` 且没有图片，则优先使用 `data/object_catalog/object_catalog.json` 中的 `semantic_text`。如果仍缺语义文本，则后续通过文件名、目录名、object config、scale、mesh metadata 生成弱描述，再让 Qwen 或规则补全 profile。

### 4.3 `data/lifespan/default_lifespan_config.json`

保存全局可控参数，避免写死在代码里。

示例：

```json
{
  "duration_days": 30,
  "snapshots_per_day": ["07:00", "12:00", "18:00", "22:00"],
  "location": "Los Angeles, USA",
  "month": "random",
  "resident_count_mode": "infer_from_bedrooms",
  "max_residents_per_bedroom": 2,
  "allow_guest_profiles": true,
  "resident_persona_pool": "data/lifespan/resident_persona_profiles.json",
  "routine_noise": 0.15,
  "placement_noise": 0.12,
  "special_event_rate": 0.25,
  "introduction_rate": 0.05,
  "failure_rate": 0.02,
  "cleanup_rate": 0.35,
  "consumption_rate": 1.0,
  "min_adjacent_stay_ratio": 0.7,
  "max_adjacent_stay_ratio": 0.95
}
```

其中 `month=random` 表示由大模型在 1-12 月中随机选择一个月份，并基于美国洛杉矶的季节、节假日和常见家庭活动生成该月每日重要事件。`resident_count_mode=infer_from_bedrooms` 表示居民数量不再手动固定，而是由场景卧室数量、房间陈列和候选居民画像共同决定。

### 4.3.1 `data/lifespan/resident_persona_profiles.json`

保存候选人物画像池。该文件已经包含 50 个覆盖家庭场景常见身份的人物 profile，包括婴幼儿、儿童、青少年、大学生、远程办公者、通勤父母、老人、室友、照护者、家政人员、临时访客、搬入/搬出居民等。

Lifespan 生成器应把该文件完整提供给大模型，让模型在理解具体场景后选择合理人物组合，而不是预先固定居民数量或机械随机抽样。

### 4.4 `data/lifespan/activity_templates.json`

定义活动到物体状态变化的模板。

示例：

```json
{
  "breakfast": {
    "typical_rooms": ["kitchen", "dining room"],
    "objects": [
      {"category": "mug", "effect": "MOVE", "target_state": "active"},
      {"category": "plate", "effect": "MOVE", "target_state": "active"},
      {"category": "apple", "effect": "CONSUME", "quantity": [0, 1]}
    ],
    "after_effects": [
      {"category": "plate", "effect": "MOVE", "target_state": "sink_or_table"},
      {"category": "mug", "effect": "MOVE", "target_state": "habit_location"}
    ]
  }
}
```

### 4.5 `core/lifespan_household_generator.py`

生成 persistent household profile，包括：

1. 读取 `scene_info`、房间类型、卧室数量、卫生间数量、厨房/客厅/书房等功能空间和主要家具陈列。
2. 读取 `data/lifespan/resident_persona_profiles.json` 中的全部候选人物画像。
3. 由大模型根据卧室数量估计长期生活人数：
   - 单卧室一般对应 1-2 名长期居民。
   - 双卧室一般对应 2-4 名长期居民。
   - 三卧室及以上可以对应核心家庭、多代家庭或室友家庭。
   - 老人、儿童、照护者、室友、临时访客需要与房间功能和床位数量保持一致。
4. 由大模型选择该场景下最可能存在的人物组合，并建立人物关系：
   - 伴侣关系。
   - 父母与子女关系。
   - 祖辈与孙辈关系。
   - 室友关系。
   - 照护者与被照护者关系。
   - 周期性访客或短期访客关系。
5. 为每名长期居民分配主卧室或私人空间，为共享角色分配共享活动区域。
6. 生成 household-level 习惯，例如购物日、清洁日、访客倾向、家庭整洁程度和共享物品使用习惯。
7. 输出人物选择原因，便于后续检查为什么该场景被分配为某种家庭结构。

大模型输出应使用结构化 JSON，并至少包含：

```json
{
  "scene_household_reasoning": "The scene has three bedrooms, one kitchen, one living room, and a study, so a two-parent family with one school-age child and one live-in grandparent is plausible.",
  "bedroom_count": 3,
  "resident_count": 4,
  "selected_residents": [
    {
      "resident_id": "resident_0",
      "source_profile_id": "persona_008_remote_worker_parent",
      "name_alias": "Parent A",
      "relationship_role": "parent",
      "assigned_private_room": "primary bedroom",
      "primary_shared_rooms": ["kitchen", "living room", "home office"]
    }
  ],
  "relationship_graph": [
    {"from": "resident_0", "to": "resident_1", "relation": "partner"},
    {"from": "resident_0", "to": "resident_2", "relation": "parent_child"}
  ]
}
```

建议该模块支持两种模式：

```text
rule_only
llm_assisted
```

第一版推荐默认使用 `llm_assisted` 进行人物选择和关系生成；如果 Qwen 不可用，则回退到规则模式：根据卧室数量采样候选人物模板并生成简单家庭关系。LLM 只负责生成结构化语义建议，不直接修改最终环境状态。

### 4.6 `core/lifespan_event_generator.py`

生成一个月内的事件时间线。

事件类型包括：

```text
MOVE
CONSUME
REPLENISH
INTRODUCE
REMOVE
DAMAGE
REPAIR
REPLACE
CLEANUP
GUEST_VISIT
SPECIAL_EVENT
```

输出：

```text
results/lifespan/<scene>/<sequence_id>/event_log.json
```

事件格式：

```json
{
  "event_id": "day_03_0730_breakfast_resident_0",
  "event_type": "routine_activity",
  "activity": "breakfast",
  "time": "day_03 07:30",
  "residents": ["resident_0"],
  "preconditions": [
    {"type": "exists", "category": "mug"}
  ],
  "effects": [
    {"type": "MOVE", "object_selector": {"category": "mug", "owner": "resident_0"}, "target": "breakfast_surface"},
    {"type": "CONSUME", "category": "apple", "quantity": 1}
  ],
  "parent_event_id": null
}
```

### 4.7 `core/lifespan_state_engine.py`

按时间顺序执行事件，对 object state 做连续传播。

必须保证：

1. 同一物体在任意时间只能有一个位置。
2. `exists=false` 的物体不能被 MOVE。
3. `CONSUME` 后的实例不能再次出现。
4. `REPLENISH` 应创建新实例或恢复库存数量。
5. `REPLACE` 必须由 `REMOVE(old)` 与 `INTRODUCE(new)` 组成。
6. 当前状态会反馈影响后续事件，例如库存过低提高购物概率。

输出：

```text
results/lifespan/<scene>/<sequence_id>/state_history.json
results/lifespan/<scene>/<sequence_id>/snapshot_requests.json
```

### 4.8 `core/lifespan_grounding.py`

将语义级状态 grounded 到具体房间、receptacle instance 和 3D pose。

核心策略：

1. 未改变物体默认复用上一 snapshot 的具体 pose。
2. 改变位置的物体才重新执行 instance assignment 和 placement。
3. 新出现物体执行完整分配和放置。
4. 被移除或消耗物体不写入当前 layout 的 `objects`。
5. 需要 collision check 时，把上一 snapshot 中未改变物体作为 fixed obstacles。

这里可能需要扩展 `core/place_objects_on_instances.py`：

```text
--fixed-layout <previous_layout.json>
--preserve-unchanged-objects
```

或者增加函数参数：

```python
place_objects_on_instances(..., fixed_objects=previous_objects)
```

这样可以避免新移动物体与未移动物体重叠。

### 4.9 `core/lifespan_generate_layouts.py`

新增总入口，完成完整生命周期 layout 生成。

推荐命令：

```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --config data/lifespan/default_lifespan_config.json \
  --duration-days 30 \
  --ssh-password 666666
```

建议也支持计划模式：

```bash
python core/lifespan_generate_layouts.py \
  --plan-json data/lifespan/lifespan_scene_plan.json \
  --ssh-password 666666
```

输出目录：

```text
results/lifespan/<scene>/lifespan_<YYYYmmdd_HHMMSS>/
  config_resolved.json
  household_profile.json
  object_lifespan_profiles.json
  monthly_calendar.json
  resident_routines.json
  event_log.json
  state_history.json
  snapshot_requests.json
  manifest.json
  validation_report.json
  layouts/
    snapshot_000_day_01_0700.json
    snapshot_001_day_01_1200.json
    snapshot_002_day_01_1800.json
    ...
```

## 5. Layout JSON 扩展方案

保持现有 `place_objects_on_instances.py` 输出结构兼容，只在顶层新增可选字段：

```json
{
  "lifespan_generation": {
    "sequence_id": "lifespan_20260807_153000",
    "scene": "00808-y9hTuugGdiq",
    "snapshot_index": 12,
    "day_index": 3,
    "time_label": "day_03_1800",
    "timestamp": "2026-10-03T18:00:00",
    "previous_layout": "snapshot_011_day_03_1200.json",
    "activity_context": ["return_home", "dinner"],
    "changed_object_count": 5,
    "present_object_count": 23,
    "absent_object_count": 3
  }
}
```

每个 object 增加可选字段：

```json
{
  "lifespan_state": {
    "exists": true,
    "owner": "resident_0",
    "location_state": "active",
    "change_type": "MOVE",
    "event_id": "day_03_1830_dinner",
    "previous_object_id": "mug_A_001",
    "previous_position": [1.2, 0.8, -0.3],
    "home_anchor": {
      "room_id": 4,
      "receptacle_instance_id": 584
    },
    "condition": "normal",
    "last_changed_at": "day_03_18:30"
  }
}
```

## 6. Manifest 扩展方案

`manifest.json` 应从“批量 layout 索引”升级为“长期家庭轨迹索引”。

建议字段：

```json
{
  "type": "lifespan_sequence_manifest",
  "scene": "00808-y9hTuugGdiq",
  "sequence_id": "lifespan_20260807_153000",
  "duration_days": 30,
  "snapshot_count": 120,
  "location": "Los Angeles, USA",
  "selected_month": 10,
  "resident_count_mode": "infer_from_bedrooms",
  "bedroom_count": 3,
  "resident_count": 4,
  "cache_paths": {
    "scene_info": "...",
    "probabilities_dir": "...",
    "surfaces_json": "..."
  },
  "statistics": {
    "avg_adjacent_stay_ratio": 0.83,
    "avg_changed_objects_per_snapshot": 4.2,
    "introduced_object_count": 5,
    "removed_object_count": 3,
    "consumed_object_count": 18,
    "placement_success_rate": 0.96
  },
  "snapshots": [
    {
      "snapshot_index": 0,
      "layout_path": "layouts/snapshot_000_day_01_0700.json",
      "day_index": 1,
      "time_label": "day_01_0700",
      "event_ids": [],
      "changed_object_count": 0,
      "placed_count": 24,
      "failed_count": 0
    }
  ]
}
```

## 7. 执行阶段

### 阶段 0：输入与缓存准备

目标：复用现有项目资产，不重复执行昂贵步骤。

执行内容：

1. 读取 `scene_info`，不存在则调用 `core/export_scene_info.py`。
2. 检查 object catalog 和 object config。
3. 检查房间推荐和概率文件，缺失时复用 `core/query_rooms_for_objects.py` 与 `core/sample_and_place_objects.py` 补齐。
4. 检查 receptacle surfaces，缺失时调用 `core/query_room_receptacle_objects.py`。
5. 检查 `object_profiles.json` 和 dataset-specific semantic text。

验收：

```text
scene_info 可读
概率文件完整
surfaces_json 可读
object profile 可解析
```

### 阶段 1：LLM 场景人物分配

目标：让每个场景先拥有合理的长期家庭身份。人物数量、人物关系和人物画像必须由该场景本身决定，而不是人为固定。

执行内容：

1. 读取 `scene_info`，统计卧室数量、卫生间数量、厨房、客厅、餐厅、书房、儿童房、客房、储藏室等功能空间。
2. 读取场景内主要家具陈列，例如床、婴儿床、书桌、餐桌、沙发、电视柜、轮椅、儿童玩具、书架等。
3. 读取 `data/lifespan/resident_persona_profiles.json` 中全部候选人物画像。
4. 将场景摘要和候选人物画像一起提供给大模型。
5. 由大模型根据卧室数量推断长期生活人数：
   - 生活人数应与卧室数量、床位、私人空间和共享空间容量匹配。
   - 卧室数量不是唯一约束；例如有儿童房、婴儿床、书房、辅助设备时，应影响人物选择。
   - 可以加入周期性访客或临时访客，但必须区分长期居民和非长期居民。
6. 由大模型选择该场景中可能存在的人物组合，并生成家庭关系图。
7. 为每名长期居民分配私人房间、常用共享房间、个人物体类别和主要活动区域。
8. 输出大模型选择依据，便于人工审查。

输出：

```text
household_profile.json
household_relationship_graph.json
```

验收：

```text
每个 resident 有稳定 ID
resident_count 与 bedroom_count 基本匹配
长期居民和临时访客被明确区分
每个长期居民有私人空间或合理共享空间
人物之间有明确 relationship
每个 private object 有 owner 或 shared 标记
```

### 阶段 2：生成 Object Lifespan Profile

目标：给每个物体建立长期动态属性。

执行内容：

1. 基于物体 category、model_id、semantic_text、object_profiles.py 推断 mobility class。
2. 建立 home-base 分布。
3. 建立 activity-compatible 分布。
4. 对 consumable 维护库存参数。
5. 对 temporary/replaceable 物体维护出现、移除、损坏概率。

输出：

```text
object_lifespan_profiles.json
```

验收：

```text
每个物体具有 mobility_class
每个可移动物体具有 home_anchor 候选
每个 consumable 具有 quantity/inventory 参数
```

### 阶段 3：LLM 生成人物协作 Daily Routine

目标：让每名人物的日常行为符合其性格、身份、房间陈列和家庭关系，并且人物之间存在协作，而不是互相独立活动。

执行内容：

1. 输入 `household_profile.json`、`household_relationship_graph.json`、场景房间与家具摘要。
2. 对每个长期居民生成 weekday/weekend daily routine。
3. routine 必须结合人物画像：
   - 儿童需要上学、作业、游戏和照护。
   - 远程办公者需要工作区、视频会议、咖啡/午餐行为。
   - 通勤者需要早晚出入、钥匙/包/水杯等 daily carry object。
   - 老人需要药物、阅读、休息和可达表面。
   - 照护者需要与被照护者共同活动。
4. routine 必须结合房间陈列：
   - 有书房时优先安排办公/学习。
   - 有儿童房或玩具时安排儿童活动。
   - 有大餐桌时安排家庭晚餐或聚会准备。
   - 有轮椅、助行器、药箱等线索时安排照护或老人活动。
5. routine 必须体现人物协作：
   - 父母与儿童共同早餐、晚餐、作业辅导、睡前整理。
   - 伴侣之间共享做饭、清洁、购物、休闲。
   - 祖辈可参与儿童照护、做饭或阅读陪伴。
   - 室友共享厨房、客厅和清洁责任，但私人物品归属清晰。
   - 照护者与老人存在 medication、meal assist、bathroom assist 等协作事件。
6. 输出高层 routine 后，再展开为 object-interaction activities，例如 `prepare_breakfast -> move mug/plate/food -> consume food -> dishes to sink`。

输出：

```text
resident_daily_routines.json
collaborative_activity_templates.json
event_log_routine_seed.json
```

验收：

```text
每个长期居民有 weekday/weekend routine
routine 与人物身份和房间陈列一致
至少包含若干 multi-resident collaborative activities
每个 high-level activity 能映射到 object effect
```

### 阶段 4：LLM 随机月份与整月特殊事件规划

目标：让场景固定出现在美国洛杉矶，并由大模型随机选择一个月份，然后为该月每一天生成重要事件。该事件表用于驱动长期物体生命周期和非平稳变化。

执行内容：

1. 固定地理位置：

```text
Los Angeles, USA
```

2. 由大模型随机选择一个月份 `month in [1, 12]`，并输出选择结果。
3. 大模型根据该月份在洛杉矶的季节、节假日、周末、家庭画像和场景空间，生成整个月每天的重要事件。
4. 每天至少包含一个 day-level important event，可以是 `normal_routine_day`，也可以是购物、清洁、访客、聚会、维修、包裹、新物体购买、食品补货、节日准备等。
5. 特殊事件必须考虑人物关系和场景容量：
   - 有儿童的家庭可出现 school project、sleepover、birthday、back-to-school preparation。
   - 有老人或照护者的家庭可出现 medication refill、doctor visit preparation、caregiver visit。
   - 有 host/guest 倾向的家庭可出现 dinner guest、weekend guest、holiday gathering。
   - 有 home office 的家庭可出现 work-from-home distribution shift。
6. 每个事件要给出 expected object effects，例如移动、消耗、补货、引入、移除、损坏、维修、替换或清洁归位。
7. 对复杂事件生成 `pre/main/post` 阶段，例如聚会前采购、聚会当天餐具和食物移动、聚会后清洁。

输出：

```text
monthly_calendar.json
daily_important_events.json
event_log_raw.json
```

验收：

```text
month 由大模型明确选择
location 固定为 Los Angeles, USA
整个月每天都有 important event
事件与人物画像、人物关系、房间陈列一致
复杂事件具有 pre/main/post 或 causal dependency
事件能够转化为 object state effects
```

### 阶段 5：连续状态传播

目标：让 Day N 的环境状态严格继承 Day 1 到 Day N-1 的历史。

执行内容：

1. 初始化 Day 1 初始物体状态。
2. 按时间顺序执行 event。
3. 对 MOVE/CONSUME/ADD/REMOVE/DAMAGE/REPAIR/REPLACE 更新状态。
4. 对无效事件执行 resample、delay、alternative 或 cancel。
5. 在 regular timestamps 和 event-aligned timestamps 导出 snapshot request。

输出：

```text
event_log.json
state_history.json
snapshot_requests.json
```

验收：

```text
不存在同一物体多位置
不存在已移除物体继续移动
不存在库存为 0 仍持续消费
snapshot state 可由 event history 重放得到
```

### 阶段 6：语义到 3D Grounding

目标：把语义状态变成现有 Habitat layout JSON。

执行内容：

1. 对第一个 snapshot 执行完整 assignment 和 placement。
2. 对后续 snapshot：
   - 未改变物体复用上一 layout 的 pose。
   - MOVE/ADD/REPLACE 的物体重新分配和放置。
   - REMOVE/CONSUME 的物体从当前 layout 移除。
3. 需要时把未改变物体作为 fixed obstacles 传给放置函数。
4. 保存 layout 顶层 `lifespan_generation` 元数据。
5. 保存 object-level `lifespan_state` 元数据。

输出：

```text
layouts/snapshot_*.json
manifest.json
```

验收：

```text
每个 snapshot 可被 visualize_placed_layout.py 加载
相邻 snapshot 保持较高 stay ratio
被事件影响的物体有明确 event_id
```

### 阶段 7：一致性验证

目标：自动发现不合理长期轨迹。

验证项：

1. Temporal consistency：相邻 snapshot 的变化率不过高或过低。
2. Causal consistency：事件前置条件满足。
3. Spatial consistency：物体位于有效房间和承载面。
4. Physical consistency：放置成功率、碰撞、悬空、穿模诊断。
5. Household consistency：居民作息与活动位置合理。
6. Dataset consistency：benchmark episode 不应采样当前 snapshot 中 absent 的物体。

输出：

```text
validation_report.json
```

关键指标：

```text
avg_adjacent_stay_ratio
avg_changed_objects_per_snapshot
placement_success_rate
event_failure_count
invalid_inventory_transition_count
invalid_absent_object_reference_count
spatial_invalid_count
```

### 阶段 8：Benchmark 集成

目标：让长期 layout 轨迹可以直接生成任务集。

需要修改：

```text
benchmark/build_episodes.py
benchmark/schemas.py
benchmark/metrics.py
```

建议新增字段：

```text
episode.scene_state.time_index
episode.scene_state.timestamp
episode.scene_state.lifespan_sequence_id
episode.scene_state.day_index
episode.scene_state.event_context
episode.seen_layout_count_before
episode.target_object_lifespan_state
```

episode 采样规则：

1. 只能从当前 snapshot 中 `exists=true` 的物体采样目标。
2. 对 image-goal，如果没有图片则允许使用 `semantic_text_goal` 或跳过 image-goal。
3. 对长期记忆任务，可以控制目标来自：
   - 稳定 home-base 物体。
   - 近期移动物体。
   - 新出现物体。
   - 曾经存在但当前 absent 的负样本任务。

## 8. 与 Qwen3-VL 的职责划分

Lifespan 流程中需要让 Qwen3-VL/LLM 在高层语义规划阶段承担更强职责，但仍然不让它直接生成每个 snapshot 的最终 3D pose。推荐职责如下：

| Qwen 适合做 | 不建议 Qwen 直接做 |
| --- | --- |
| 阅读 `scene_info`、房间数量、卧室数量、家具陈列和所有候选居民画像 | 直接输出 Day 1 到 Day 30 的所有 pose |
| 根据卧室数量和场景功能空间选择合理居民数量与人物组合 | 忽略卧室/床位/房间陈列机械随机选人 |
| 建立人物关系图，例如父母子女、伴侣、祖孙、室友、照护关系 | 生成互相独立且没有协作的人物 routine |
| 为每个人生成符合性格、身份、房间陈列和家庭关系的 daily routine | 让所有人物每天重复完全相同的行为模板 |
| 生成人物之间的协作活动，例如共同早餐、作业辅导、照护、做饭、清洁 | 直接修改最终 Habitat state |
| 随机选择洛杉矶场景下的月份，并生成该月每天的重要事件 | 记忆所有历史事件并保证最终状态一致性 |
| 判断物体适合哪些活动 | 直接修改最终 Habitat state |
| 将 semantic text 转换成 category/profile | 记忆所有历史事件并保证一致性 |
| 生成 activity template 候选 | 判断碰撞与几何稳定 |
| 对模糊物体给出 home-base 建议 | 直接决定消耗库存的长期一致性 |

核心原则：

```text
LLM plans household semantics and daily/monthly behavior.
State engine owns temporal truth.
Habitat placement owns physical truth.
```

因此推荐调用顺序是：

```text
LLM call 1:
  scene_info + resident_persona_profiles
  -> household_profile + relationship_graph

LLM call 2:
  household_profile + scene furniture summary
  -> resident_daily_routines + collaborative activities

LLM call 3:
  household_profile + daily_routines + location=Los Angeles
  -> random month + daily important events for the whole month

Optional LLM calls:
  object semantic text/profile correction
  ambiguous object-receptacle relation judgment
```

后续 `lifespan_state_engine.py` 必须把这些 LLM 输出转换成可验证的事件和状态转移。也就是说，LLM 负责“家庭故事和行为计划”，状态引擎负责“历史连续性和因果一致性”，Habitat 放置模块负责“空间和物理有效性”。

## 9. 与现有 batch baseline 的关系

保留 `core/batch_generate_layouts.py` 作为 baseline：

```text
independent_random_layouts
```

新增 lifespan 入口生成：

```text
lifespan_temporal_layouts
```

后续 benchmark 可以同时提供两种 split：

```text
baseline_independent
lifespan_monthly
```

这样可以直接比较：

1. 独立随机布局是否让历史经验失效。
2. Lifespan 布局是否能让机器人通过多次探索提升预测和导航效率。
3. 长期事件发生后，机器人是否能更新已有记忆。

## 10. 推荐实现顺序

### 第一步：实现 schema 与 LLM Household Planning

新增：

```text
core/lifespan_schema.py
core/lifespan_household_generator.py
core/lifespan_profiles.py
core/lifespan_event_generator.py
core/lifespan_state_engine.py
data/lifespan/default_lifespan_config.json
data/lifespan/activity_templates.json
data/lifespan/resident_persona_profiles.json
```

第一步先不调用 Habitat，也不生成 3D layout，而是完成三类高层规划：

1. 大模型读取 `scene_info` 和全部候选居民画像，输出该场景对应的家庭成员、人数和关系图。
2. 大模型根据人物性格、身份、房间陈列和人物关系，为每个人生成协作型 daily routine。
3. 大模型固定地点为美国洛杉矶，随机选择月份，并生成该月每天的重要事件。

输出：

```text
household_profile.json
household_relationship_graph.json
resident_daily_routines.json
monthly_calendar.json
daily_important_events.json
event_log.json
state_history.json
snapshot_requests.json
```

验证目标：

```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --config data/lifespan/default_lifespan_config.json \
  --dry-run-household-plan
```

验收目标：

```text
resident_count 由卧室数量和房间陈列推断
household_profile 中的人物来自 resident_persona_profiles.json
每个人有 weekday/weekend daily routine
daily routine 中存在 multi-resident collaboration
month 由 LLM 随机选择且 location 固定为 Los Angeles, USA
整个月每天都有 important event
```

### 第二步：离线状态传播

在不调用 Habitat 的情况下，把第一步生成的 daily routine 和 daily important events 转换为 object-level event log，并执行状态传播。

```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --config data/lifespan/default_lifespan_config.json \
  --dry-run-state-only \
  --ssh-password 666666
```

验证目标：

```text
event_log 按时间排序
routine 和 special event 均能转成 object effects
Day N 状态继承 Day 1 到 Day N-1 历史
CONSUME/REPLENISH/INTRODUCE/REMOVE 等变化具有因果一致性
snapshot_requests 覆盖 regular timestamps 和重要事件前后
```

### 第三步：接入现有 placement，生成少量 snapshot

先生成 1 个场景、3 天、每天 2 个 snapshot。

```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --duration-days 3 \
  --snapshots-per-day 2 \
  --disable-assignment-llm \
  --ssh-password 666666
```

验证目标：

```text
生成 6 个 layout
相邻 layout 不再完全独立
未变化物体复用上一 pose
变化物体有 event_id
layout 顶层包含 lifespan_generation
object 级别包含 lifespan_state
```

### 第四步：增加 fixed object collision

扩展 `place_objects_on_instances.py`，使重新放置 changed objects 时考虑上一 snapshot 的 unchanged objects。

验收：

```text
新增物体不与未移动物体重叠
MOVE 物体不覆盖 stable objects
placement_success_rate 不低于现有 batch baseline
```

### 第五步：接入 Qwen 物体语义 profile

除前三次高层 household/routine/monthly-event 调用外，只在以下物体语义阶段调用 Qwen：

1. 生成或修正 object lifespan profile。
2. 生成 activity template 候选。
3. 对 YCB/HSSD 缺图物体生成 semantic text。
4. 对模糊 object-category/receptacle relation 做少量判别。

验收：

```text
同一 scene 的 Qwen 调用集中在高层规划与少量语义补全
不会每个 snapshot 都重新问完整布局
Qwen 失败时规则回退可继续运行
```

### 第六步：生成完整 30 天轨迹

```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --duration-days 30 \
  --config data/lifespan/default_lifespan_config.json \
  --ssh-password 666666
```

验收：

```text
生成完整 manifest
生成 event_log/state_history/validation_report
layout 可视化连续切换
benchmark/build_episodes.py 可读取
```

### 第七步：benchmark 任务生成与评测

```bash
python benchmark/build_episodes.py \
  --layout-manifest results/lifespan/00808-y9hTuugGdiq/lifespan_<id>/manifest.json \
  --output-dir benchmark/episodes/lifespan_v1 \
  --episodes-per-layout 3
```

验收：

```text
episode 不采样 absent object
seen_layout_count_before 正确对应历史 snapshot 数量
metrics 可按 time_index/day_index 聚合
```

## 11. 需要重点修改的现有代码点

### 11.1 `core/place_objects_on_instances.py`

建议增加：

```text
fixed_objects
preserve_existing_pose
changed_object_ids
```

原因：lifespan snapshot 中大部分物体不应重新采样，否则会破坏时间连续性。

### 11.2 `core/assign_objects_to_receptacle_instances.py`

建议支持 object state 中的目标约束：

```text
preferred_room_id
preferred_receptacle_category
preferred_instance_id
previous_instance_id
home_anchor
activity_context
```

原因：活动事件已经给出了语义目标，assignment 不应完全重新选择。

### 11.3 `benchmark/build_episodes.py`

建议读取 lifespan manifest 中的：

```text
snapshot_index
day_index
timestamp
event_context
lifespan_sequence_id
```

并保证目标物体当前存在。

### 11.4 `core/visualize_placed_layout.py`

建议增加 lifespan 对比显示：

```text
changed objects 高亮
new/removed objects 列表
event_id HUD
snapshot timestamp HUD
```

## 12. 缺失信息与解决方案

当前可以直接开始实现，但以下信息会影响生成质量：

| 缺失信息 | 影响 | 解决方案 |
| --- | --- | --- |
| 每个 scene 的家庭规模设定 | 居民数量和私人空间不稳定 | 默认由 LLM 读取 scene_info、卧室数量、家具陈列和候选居民画像后推断，允许 config 覆盖 |
| YCB/HSSD 物体图片缺失 | image-goal 和 VLM 图像理解受限 | 使用 `semantic_text`；缺失时由文件名、目录名、object config 自动生成弱描述 |
| 精确物体尺寸和 y_offset 不完整 | 放置失败或高度偏差 | 继续补充 `object_profiles.json`，并记录 profile diagnostics |
| 柜子/冰箱/抽屉 inside relation 未完全建模 | 容器类物体只能近似放在表面 | 第一版用 surface approximation，后续增加 container volume |
| 真实居民作息数据 | 行为模式真实性受限 | 第一版由 Qwen 根据人物画像、人物关系和房间陈列生成 daily routine，规则模板作为回退，后续可接入真实 time-use survey |
| 物体库存初始数量 | consumable saw-tooth 质量受限 | 第一版按 category 采样初始库存，允许 config 覆盖 |

## 13. 验收标准

第一版 Lifespan 生成器完成后，应满足：

1. 单场景可生成 30 天、多 snapshot 的连续 layout。
2. 相邻 snapshot 中大部分物体保持不动，默认 stay ratio 在 `0.70-0.95`。
3. 物体移动由 event_log 中的活动或特殊事件解释。
4. 可消耗物体存在数量变化，并能触发 replenishment。
5. 新物体 introduction、旧物体 removal/replacement 可以被记录。
6. 所有 layout 保持现有 JSON 兼容格式，可被 `visualize_placed_layout.py` 加载。
7. `benchmark/build_episodes.py` 可以基于 lifespan manifest 生成任务集。
8. `validation_report.json` 能清楚列出失败原因，而不是只给最终成功率。

## 14. 最小可行版本范围

为了尽快形成可运行闭环，MVP 建议只实现：

1. 由 LLM 根据卧室数量和房间陈列选择 1 个合理家庭，居民数量不手动固定。
2. 7 天时长。
3. 每天 4 个 regular snapshots。
4. 只支持 MOVE、CONSUME、REPLENISH、INTRODUCE、REMOVE。
5. 只支持 breakfast、work、dinner、leisure、cleaning、shopping 六类活动。
6. changed objects 重新放置，unchanged objects 复用上一 pose。
7. Qwen 默认用于 household selection、collaborative daily routine 和 monthly important events；物体 profile 可先用启发式规则回退。

MVP 命令：

```bash
python core/lifespan_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --duration-days 7 \
  --snapshots-per-day 4 \
  --ssh-password 666666
```

MVP 产物：

```text
results/lifespan/00808-y9hTuugGdiq/lifespan_<id>/
  household_profile.json
  object_lifespan_profiles.json
  event_log.json
  state_history.json
  manifest.json
  validation_report.json
  layouts/
```

## 15. 后续论文实验建议

完成 lifespan 数据生成后，论文实验可以围绕以下问题展开：

1. Independent random layouts 与 lifespan layouts 的导航学习差异。
2. 随着 `seen_layout_count_before` 增加，机器人是否更快找到目标。
3. 对 stable object、routine movable object、temporary object、consumable object 分别统计性能。
4. 在 distribution shift 事件后，机器人是否能更新旧记忆。
5. 不同 `placement_noise`、`special_event_rate`、`cleanup_rate` 下的任务难度变化。

推荐新增指标：

```text
Object Location Prediction Error
Temporal Adaptation Rate
Memory Benefit over First Visit
Post-Shift Recovery Speed
Absent Object False Search Rate
```

## 16. 总结

Lifespan 版本的核心不是让 VLM 生成更多随机概率，而是把现有项目升级为：

```text
persistent household profile
+ hierarchical resident routine
+ event-driven state transition
+ object lifecycle simulation
+ physical scene grounding
+ benchmark episode generation
```

现有 `batch_generate_layouts.py` 仍然适合作为 independent layout baseline；新的 `lifespan_generate_layouts.py` 则应生成同一家庭在一个月内连续演化的状态轨迹。这样得到的数据集更适合研究机器人长期物体记忆、时空规律学习、动态目标导航和环境变化后的记忆更新。

---

## 当前实现状态同步（2026-08-13）

本文档中的第一阶段 semantic-only MVP 已经落地到代码中：

```text
core/lifespan_generate_layouts.py
core/lifespan_schema.py
core/lifespan_household_generator.py
core/lifespan_profiles.py
core/lifespan_event_generator.py
core/lifespan_state_engine.py
data/lifespan/default_lifespan_config.json
data/lifespan/activity_templates.json
data/lifespan/resident_persona_profiles.json
```

已实现能力：

1. 读取 `scene_info` 和 50 个候选人物画像。
2. 根据卧室数量、房间陈列和人物画像生成 scene-specific household；Qwen 不可用时使用规则回退。
3. 生成人物关系图、weekday/weekend daily routine 和协作活动。
4. 固定地点为 `Los Angeles, USA`，随机选择月份，并生成每天的重要事件。
5. 推断 object lifespan profiles。
6. 将 routine/month event 展开为 object-level event log。
7. 连续传播 object state，输出 `state_history.json` 和 `snapshot_requests.json`。
8. 写出 semantic-only `layouts/snapshot_*.json` 与 `manifest.json`。

已验证 smoke test：

```bash
python core/log_filter.py --run "python core/lifespan_generate_layouts.py --scene 00808-y9hTuugGdiq --duration-days 3 --snapshots-per-day 07:00,18:00 --object-limit 10 --disable-lifespan-llm --sequence-id smoke_lifespan_test"
```

当前未完成：

1. `core/lifespan_grounding.py` 尚未实现。
2. Lifespan snapshot 还没有接入 `assign_objects_to_receptacle_instances.py` 和 `place_objects_on_instances.py`。
3. 当前 semantic layout 中 `position/rotation=null`，不能直接用于 Habitat-Sim 导航任务。
4. `benchmark/build_episodes.py` 还未直接消费 Lifespan semantic manifest；需要先生成真实 3D layout。
