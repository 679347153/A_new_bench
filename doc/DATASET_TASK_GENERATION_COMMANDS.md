# 场景数据集与任务集生成命令指南

本文按执行顺序汇总本项目从数据整理、object catalog 构建、Qwen 查询、概率采样、承载面提取、批量 layout 生成、Lifespan 长期语义轨迹生成，到任务集生成与可视化检查的常用命令。

所有命令都通过 `python core/log_filter.py --run "..."` 包裹，用于过滤 Habitat/OpenGL/HM3D 的高频噪声日志。

默认 Qwen 连接方式为 SSH 密码登录：

```text
host: 7.216.187.6
ssh port: 30180
user: root
password: 666666
remote vLLM API: 127.0.0.1:8000
```

## 1. 整理项目数据目录

先预览迁移计划：

```bash
python core/log_filter.py --run "python core/prepare_project_structure.py --dry-run"
```

确认无误后执行整理：

```bash
python core/log_filter.py --run "python core/prepare_project_structure.py"
```

检查 HM3D 和 object config 搜索路径：

```bash
python core/log_filter.py --run "python -c 'from core.project_paths import resolve_hm3d_root, default_object_config_dirs_str; print(resolve_hm3d_root()); print(default_object_config_dirs_str())'"
```

## 2. 构建统一 Object Catalog

构建 legacy/YCB/HSSD 混合 object catalog。legacy 优先使用图片；YCB/HSSD 没有图片时使用 `semantic_text`：

```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets legacy,ycb,hssd --write-missing"
```

仅检查 YCB/HSSD 数量，不写文件：

```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets ycb,hssd --dry-run"
```

导出 HSSD 缺失语义描述的列表：

```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets hssd --write-missing --missing-output data/object_catalog/missing_semantic_text.csv"
```

## 3. 为 HSSD 补充 Semantic Text

小批量调用 Qwen 生成语义描述：

```bash
python core/log_filter.py --run "python core/generate_object_semantic_text.py --limit 20 --ssh-password 666666"
```

确认输出合理后扩大数量：

```bash
python core/log_filter.py --run "python core/generate_object_semantic_text.py --limit 200 --ssh-password 666666"
```

生成后重建 catalog：

```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets legacy,ycb,hssd --write-missing"
```

如果仍缺少语义文本，推荐先生成 object preview，再人工或半自动编辑：

```bash
python core/log_filter.py --run "python core/build_object_catalog.py --datasets hssd --write-missing --missing-output data/object_catalog/missing_semantic_text.csv"
```

## 4. 导出 Scene Info

单场景导出：

```bash
python core/log_filter.py --run "python core/export_scene_info.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --output-dir results/scene_info/00808-y9hTuugGdiq"
```

批量导出全部可用场景：

```bash
python core/log_filter.py --run "python core/export_scene_info.py --all --data-dir data/scenes/hm3d --output-dir results/scene_info"
```

## 5. 生成物体到房间的推荐

legacy 图片物体：

```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets legacy --images-dir data/object_images/legacy --output-dir results/scene_info --ssh-password 666666"
```

YCB text-only 物体：

```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets ycb --object-catalog data/object_catalog/object_catalog.json --output-dir results/scene_info --ssh-password 666666"
```

HSSD text-only 小批量测试：

```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets hssd --object-catalog data/object_catalog/object_catalog.json --limit-objects 20 --output-dir results/scene_info --ssh-password 666666"
```

混合物体集：

```bash
python core/log_filter.py --run "python core/query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --output-dir results/scene_info --ssh-password 666666"
```

## 6. 生成或复用房间概率分布

根据房间推荐生成概率并采样：

```bash
python core/log_filter.py --run "python core/sample_and_place_objects.py --scene 00808-y9hTuugGdiq --mode generate --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --rooms-info-dir results/scene_info --probabilities-dir results/probabilities --layouts-dir results/layouts"
```

后续复用概率文件，只重新 sample：

```bash
python core/log_filter.py --run "python core/sample_and_place_objects.py --scene 00808-y9hTuugGdiq --mode load --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --rooms-info-dir results/scene_info --probabilities-dir results/probabilities --layouts-dir results/layouts"
```

## 7. 提取可放置承载面

默认使用 Qwen 辅助承载面排序：

```bash
python core/log_filter.py --run "python core/query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --scene-info-path results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --output results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --ssh-password 666666"
```

无远端 LLM 的启发式模式：

```bash
python core/log_filter.py --run "python core/query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --scene-info-path results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --output results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --disable-llm"
```

## 8. 单次 Assignment + Final Layout

自动生成 surfaces、采样、分配并放置：

```bash
python core/log_filter.py --run "python core/assign_objects_to_receptacle_instances.py --scene 00808-y9hTuugGdiq --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

使用已有 surfaces，并关闭 LLM 使用启发式分配：

```bash
python core/log_filter.py --run "python core/assign_objects_to_receptacle_instances.py --scene 00808-y9hTuugGdiq --surfaces-json results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --disable-llm"
```

## 9. 批量生成多个独立随机 Layout

同一场景生成 10 个最终 layout，复用已有概率和 surfaces：

```bash
python core/log_filter.py --run "python core/batch_generate_layouts.py --scene 00808-y9hTuugGdiq --num-layouts 10 --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

无远端 LLM 的快速 smoke test：

```bash
python core/log_filter.py --run "python core/batch_generate_layouts.py --scene 00808-y9hTuugGdiq --num-layouts 2 --object-datasets ycb --object-catalog data/object_catalog/object_catalog.json --disable-assignment-llm --disable-surface-llm"
```

计划模式，读取多场景列表：

```bash
python core/log_filter.py --run "python core/batch_generate_layouts.py --plan-json scenes_plan.json --num-layouts 5 --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

计划文件示例：

```json
{
  "num_layouts": 5,
  "base_seed": 42,
  "object_datasets": "legacy,ycb,hssd",
  "limit_objects": 50,
  "scenes": [
    "00808-y9hTuugGdiq",
    {"scene": "00800-TEEsavR23oF", "num_layouts": 3, "base_seed": 100}
  ]
}
```

## 10. 生成 Lifespan 长期家庭语义轨迹

Lifespan 分支用于生成同一家庭在多天或整月内的语义演化轨迹。当前实现是 semantic-only MVP：会输出人物、日程、月事件、object state、snapshot request 和 semantic layout，但暂不输出可直接在 Habitat 中加载的真实 `position/rotation`。

本地规则回退 smoke test，不连接 Qwen：

```bash
python core/log_filter.py --run "python core/lifespan_generate_layouts.py --scene 00808-y9hTuugGdiq --duration-days 3 --snapshots-per-day 07:00,18:00 --object-limit 10 --disable-lifespan-llm --sequence-id smoke_lifespan_test"
```

使用 Qwen 进行 household selection、daily routine 和 monthly event planning：

```bash
python core/log_filter.py --run "python core/lifespan_generate_layouts.py --scene 00808-y9hTuugGdiq --duration-days 7 --snapshots-per-day 07:00,12:00,18:00,22:00 --object-limit 40 --ssh-password 666666"
```

显式传入已经导出的 scene_info：

```bash
python core/log_filter.py --run "python core/lifespan_generate_layouts.py --scene 00808-y9hTuugGdiq --scene-info results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --duration-days 7 --snapshots-per-day 07:00,12:00,18:00,22:00 --object-limit 40 --ssh-password 666666"
```

使用完整默认配置生成 30 天语义轨迹：

```bash
python core/log_filter.py --run "python core/lifespan_generate_layouts.py --scene 00808-y9hTuugGdiq --config data/lifespan/default_lifespan_config.json --object-limit 50 --ssh-password 666666"
```

Lifespan 主要输出：

```text
results/lifespan/<scene>/<sequence_id>/
  config_resolved.json
  scene_summary.json
  household_profile.json
  household_relationship_graph.json
  object_lifespan_profiles.json
  resident_daily_routines.json
  collaborative_activity_templates.json
  monthly_calendar.json
  daily_important_events.json
  event_log.json
  state_history.json
  snapshot_requests.json
  manifest.json
  validation_report.json
  layouts/snapshot_*.json
```

## 11. 可视化检查和手动修正

打开单个最终 3D layout：

```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json --scene 00808-y9hTuugGdiq"
```

打开 batch 中一个 layout，并用 `[` / `]` 切换同目录其他 layout：

```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq"
```

手动调试高度，默认加载时会给所有物体应用 `--initial-y-offset 2.5`：

```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq --debug-offset --offset-step 0.02"
```

严格复现原始 layout，不加默认 Y 偏移：

```bash
python core/log_filter.py --run "python core/visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq --initial-y-offset 0"
```

使用 `test_layout.py` 手动编辑：

```bash
python core/log_filter.py --run "python core/test_layout.py 00808-y9hTuugGdiq --layout scene_objects.json --ui-lang zh"
```

## 12. 生成任务集

如果已经有最终 3D layout，可以调用当前任务编排脚本生成 benchmark 任务：

```bash
python core/log_filter.py --run "python core/orchestrate_sd_ovon_complete.py --scene 00808-y9hTuugGdiq --layout results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json"
```

如果需要先生成观测数据：

```bash
python core/log_filter.py --run "python core/observation_generator.py --scene 00808-y9hTuugGdiq --layout results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json"
```

如果使用 `benchmark/build_episodes.py` 从 batch manifest 构建长期任务集：

```bash
python core/log_filter.py --run "python benchmark/build_episodes.py --layout-manifest results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/manifest.json --output-dir benchmark/episodes/layout_batch_v1 --episodes-per-layout 3"
```

注意：当前 Lifespan manifest 是 semantic-only，不能直接作为最终导航任务的 3D layout 输入。需要先将 `snapshot_requests.json` ground 到真实 3D layout，再交给 `benchmark/build_episodes.py`。

## 13. 输出结构速览

普通自动放置链路主要产物：

```text
results/scene_info/<scene>/<scene>_scene_info.json
results/scene_info/<scene>/<object>_rooms.json
results/probabilities/<scene>/<object>_probs.json
results/receptacle_queries/<scene>/<scene>_receptacle_surfaces_all_rooms.json
results/object_instance_assignments/<scene>/*_object_instance_plan.json
results/layouts/<scene>/*assigned_instance_layout*.json
results/layouts/<scene>/batch_<time>/layout_<idx>_seed_<seed>.json
results/layouts/<scene>/batch_<time>/manifest.json
```

Lifespan 语义演化链路主要产物：

```text
data/lifespan/resident_persona_profiles.json
data/lifespan/default_lifespan_config.json
data/lifespan/activity_templates.json
results/lifespan/<scene>/<sequence_id>/household_profile.json
results/lifespan/<scene>/<sequence_id>/resident_daily_routines.json
results/lifespan/<scene>/<sequence_id>/daily_important_events.json
results/lifespan/<scene>/<sequence_id>/event_log.json
results/lifespan/<scene>/<sequence_id>/state_history.json
results/lifespan/<scene>/<sequence_id>/snapshot_requests.json
results/lifespan/<scene>/<sequence_id>/layouts/snapshot_*.json
results/lifespan/<scene>/<sequence_id>/manifest.json
results/lifespan/<scene>/<sequence_id>/validation_report.json
```
