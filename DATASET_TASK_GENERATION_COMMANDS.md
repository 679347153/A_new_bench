# 场景数据集与任务集生成命令指南

本文按执行顺序汇总本项目从数据整理、对象 catalog 构建、Qwen 房间推荐、概率采样、承载面提取、批量 layout 生成到任务集生成的常用命令。所有命令都通过 `log_filter.py` 包裹，便于压缩 Habitat/OpenGL 的高频日志。

默认 Qwen 连接方式为 SSH 密码登录：

- host: `7.216.187.6`
- ssh port: `30180`
- user: `root`
- password: `666666`
- remote vLLM API: `127.0.0.1:8000`

## 1. 整理项目数据目录

先预览会移动哪些目录：

```bash
python log_filter.py --run "python prepare_project_structure.py --dry-run"
```

确认无误后执行迁移。迁移后的主要目录为：

- `data/scenes/hm3d`
- `data/object_datasets/ycb-v1.2`
- `data/object_datasets/hssd-hab-v0.2.3`
- `data/object_images/legacy`
- `data/object_catalog`
- `data/archives`

```bash
python log_filter.py --run "python prepare_project_structure.py"
```

检查数据目录：

```bash
python log_filter.py --run "python -c \"from project_paths import resolve_hm3d_root, default_object_config_dirs_str; print(resolve_hm3d_root()); print(default_object_config_dirs_str())\""
```

## 2. 构建对象 Catalog

构建统一对象表。legacy 对象使用图片；YCB/HSSD 没有图片时使用 `semantic_text`。

```bash
python log_filter.py --run "python build_object_catalog.py --datasets legacy,ycb,hssd --write-missing"
```

如果只想先检查对象数量，不写文件：

```bash
python log_filter.py --run "python build_object_catalog.py --datasets ycb,hssd --dry-run"
```

如果 HSSD 对象缺少语义描述，可先导出缺失列表：

```bash
python log_filter.py --run "python build_object_catalog.py --datasets hssd --write-missing --missing-output data/object_catalog/missing_semantic_text.csv"
```

## 3. 为 HSSD 补充 Semantic Text

HSSD 没有对象图片且部分配置没有描述性语义文本时，可用配置元数据批量请求 Qwen 生成简短描述。先小批量测试：

```bash
python log_filter.py --run "python generate_object_semantic_text.py --limit 20 --ssh-password 666666"
```

确认输出合理后继续生成更多条目：

```bash
python log_filter.py --run "python generate_object_semantic_text.py --limit 200 --ssh-password 666666"
```

生成后重建 catalog，让新增描述进入 `object_catalog.json`：

```bash
python log_filter.py --run "python build_object_catalog.py --datasets legacy,ycb,hssd --write-missing"
```

如果仍缺少描述，推荐方案是：先用 Habitat-Sim 离屏渲染对象预览图到 `data/object_previews/hssd`，再把“预览图 + config 元数据”一起发给 Qwen 生成更可靠的 `semantic_text`；若远端不可用，则人工编辑 `data/object_catalog/object_text_overrides.json` 覆盖关键对象。

## 4. 导出 Scene Info

单场景导出：

```bash
python log_filter.py --run "python export_scene_info.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --output-dir results/scene_info/00808-y9hTuugGdiq"
```

批量导出所有可用场景：

```bash
python log_filter.py --run "python export_scene_info.py --all --data-dir data/scenes/hm3d --output-dir results/scene_info"
```

## 5. 生成对象到房间的推荐

legacy 图片对象：

```bash
python log_filter.py --run "python query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets legacy --images-dir data/object_images/legacy --output-dir results/scene_info --ssh-password 666666"
```

YCB text-only 对象：

```bash
python log_filter.py --run "python query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets ycb --object-catalog data/object_catalog/object_catalog.json --output-dir results/scene_info --ssh-password 666666"
```

HSSD text-only 小批量测试：

```bash
python log_filter.py --run "python query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets hssd --object-catalog data/object_catalog/object_catalog.json --limit-objects 20 --output-dir results/scene_info --ssh-password 666666"
```

混合对象集：

```bash
python log_filter.py --run "python query_rooms_for_objects.py --scene 00808-y9hTuugGdiq --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --output-dir results/scene_info --ssh-password 666666"
```

## 6. 生成或复用概率分布

仅采样生成 layout 草稿，缺失概率时自动根据房间推荐生成：

```bash
python log_filter.py --run "python sample_and_place_objects.py --scene 00808-y9hTuugGdiq --mode generate --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --rooms-info-dir results/scene_info --probabilities-dir results/probabilities --layouts-dir results/layouts"
```

后续复用概率分布，只重新 sample：

```bash
python log_filter.py --run "python sample_and_place_objects.py --scene 00808-y9hTuugGdiq --mode load --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --rooms-info-dir results/scene_info --probabilities-dir results/probabilities --layouts-dir results/layouts"
```

## 7. 提取可放置承载面

默认使用 Qwen 辅助排序承载面：

```bash
python log_filter.py --run "python query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --scene-info-path results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --output results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --ssh-password 666666"
```

无远端 LLM 的启发式模式：

```bash
python log_filter.py --run "python query_room_receptacle_objects.py --scene 00808-y9hTuugGdiq --data-dir data/scenes/hm3d --scene-info-path results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json --output results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --disable-llm"
```

## 8. 单次 Assignment + Final Layout

自动生成 surfaces、采样、分配并放置：

```bash
python log_filter.py --run "python assign_objects_to_receptacle_instances.py --scene 00808-y9hTuugGdiq --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

使用已有 surfaces，启发式分配：

```bash
python log_filter.py --run "python assign_objects_to_receptacle_instances.py --scene 00808-y9hTuugGdiq --surfaces-json results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --disable-llm"
```

## 9. 批量生成多个 Layout

同一场景生成 10 个最终 layout，复用已有概率和 surfaces：

```bash
python log_filter.py --run "python batch_generate_layouts.py --scene 00808-y9hTuugGdiq --num-layouts 10 --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
```

无远端 LLM 的快速 smoke test：

```bash
python log_filter.py --run "python batch_generate_layouts.py --scene 00808-y9hTuugGdiq --num-layouts 2 --object-datasets ycb --object-catalog data/object_catalog/object_catalog.json --disable-assignment-llm --disable-surface-llm"
```

计划模式，读取多场景列表：

```bash
python log_filter.py --run "python batch_generate_layouts.py --plan-json scenes_plan.json --num-layouts 5 --object-datasets legacy,ycb,hssd --object-catalog data/object_catalog/object_catalog.json --images-dir data/object_images/legacy --limit-objects 50 --ssh-password 666666"
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

## 10. 可视化检查和手动修正

打开单个 layout：

```bash
python log_filter.py --run "python visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json --scene 00808-y9hTuugGdiq"
```

打开 batch 中一个 layout，并用 `[` / `]` 切换同目录其他 layout：

```bash
python log_filter.py --run "python visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq"
```

手动调试高度，默认可视化加载时会给所有物体应用 `--initial-y-offset 2.5`：

```bash
python log_filter.py --run "python visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq --debug-offset --offset-step 0.02"
```

严格复现原始 layout，不加默认 Y 偏移：

```bash
python log_filter.py --run "python visualize_placed_layout.py results/layouts/00808-y9hTuugGdiq/batch_20260101_120000/layout_000_seed_42.json --scene 00808-y9hTuugGdiq --initial-y-offset 0"
```

使用 `test_layout.py` 手动编辑：

```bash
python log_filter.py --run "python test_layout.py 00808-y9hTuugGdiq --layout scene_objects.json --ui-lang zh"
```

## 11. 生成任务集

如果已有最终 layout，可以调用当前任务编排脚本生成 benchmark 任务集：

```bash
python log_filter.py --run "python orchestrate_sd_ovon_complete.py --scene 00808-y9hTuugGdiq --layout results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json"
```

如果需要先生成观测数据：

```bash
python log_filter.py --run "python observation_generator.py --scene 00808-y9hTuugGdiq --layout results/layouts/00808-y9hTuugGdiq/00808-y9hTuugGdiq_assigned_instance_layout.json"
```

## 12. 输出结构速览

最终主要产物：

- `results/scene_info/<scene>/<scene>_scene_info.json`：场景语义、房间和实例明细。
- `results/scene_info/<scene>/<object>_rooms.json`：每个对象的候选房间推荐。
- `results/probabilities/<scene>/<object>_probs.json`：对象在候选房间上的采样概率。
- `results/receptacle_queries/<scene>/<scene>_receptacle_surfaces_all_rooms.json`：每个房间可放置承载面的候选实例和表面点云引用。
- `results/layouts/<scene>/batch_<time>/layout_<idx>_seed_<seed>.json`：最终布局。
- `results/layouts/<scene>/batch_<time>/manifest.json`：批量生成摘要、失败原因和复用路径。
- `benchmark/...`：根据布局和观测生成的任务集、episode 或评测产物。
