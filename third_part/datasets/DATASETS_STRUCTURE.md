# third_part/datasets 文件结构与功能说明

本文档基于当前 `third_part/datasets` 目录重新整理。该目录来自 Habitat-Lab 的 `habitat.datasets` 模块，主要负责多种 Habitat 任务数据集的注册、加载、反序列化，以及 Rearrangement 任务 episode 的程序化生成。

## 目录总览

```text
third_part/datasets/
|-- __init__.py
|-- registration.py
|-- utils.py
|-- eqa/
|   |-- __init__.py
|   `-- mp3d_eqa_dataset.py
|-- image_nav/
|   |-- __init__.py
|   `-- instance_image_nav_dataset.py
|-- object_nav/
|   |-- __init__.py
|   `-- object_nav_dataset.py
|-- pointnav/
|   |-- __init__.py
|   |-- pointnav_dataset.py
|   `-- pointnav_generator.py
|-- rearrange/
|   |-- __init__.py
|   |-- combine_datasets.py
|   |-- generate_episode_inits.py
|   |-- rearrange_dataset.py
|   |-- rearrange_generator.py
|   |-- run_episode_generator.py
|   |-- configs/
|   `-- samplers/
`-- vln/
    |-- __init__.py
    `-- r2r_vln_dataset.py
```

整体设计可以分成三层：

- 数据集注册层：`registration.py` 和各子包的 `__init__.py`。
- 数据集加载层：PointNav、ObjectNav、InstanceImageNav、EQA、VLN、Rearrange 的 Dataset 类。
- Rearrange episode 生成层：`rearrange_generator.py`、`run_episode_generator.py`、`samplers/` 和 `configs/`。

## 注册入口

### `__init__.py`

只做一件事：从 `habitat.datasets.registration` 暴露 `make_dataset`。

### `registration.py`

统一注册和实例化入口。

主要内容：

- 导入各子任务 `_try_register_*` 函数。
- 定义 `make_dataset(id_dataset, **kwargs)`：
  - 打日志。
  - 从 Habitat `registry` 中取对应 dataset 类。
  - 找不到则 assert。
  - 找到后用 `**kwargs` 实例化。
- 文件底部主动执行所有注册函数。

当前注册的数据集名称：

| Registry name | 对应类 |
| --- | --- |
| `PointNav-v1` | `PointNavDatasetV1` |
| `ObjectNav-v1` | `ObjectNavDatasetV1` |
| `InstanceImageNav-v1` | `InstanceImageNavDatasetV1` |
| `MP3DEQA-v1` | `Matterport3dDatasetV1` |
| `R2RVLN-v1` | `VLNDatasetV1` |
| `RearrangeDataset-v0` | `RearrangeDatasetV0` |

### 各子目录 `__init__.py`

这些文件都采用 try-register 模式：

- 正常情况下导入真实 Dataset 类，类上的 `@registry.register_dataset(...)` 完成注册。
- 如果导入失败，注册一个占位 Dataset 类；实例化时重新抛出原始 `ImportError`。
- 这样做是为了在缺少 `habitat-sim` 等重依赖时，仍能导入 Habitat 的部分 Python 包。

需要注意：`rearrange/__init__.py` 的导入失败占位类注册名是 `OrpNavDataset-v0`，而真实类注册名是 `RearrangeDataset-v0`。如果真实导入失败，`RearrangeDataset-v0` 不会被同名占位类接管，这很像上游遗留笔误。

## 通用工具

### `utils.py`

通用文本、词表、路径和物理配置工具。

主要内容：

- `tokenize(sentence, regex, keep, remove)`：小写化句子，并按正则拆 token。
- `load_str_list(fname)`：逐行读取字符串列表。
- `VocabDict`：词表封装，提供 `word2idx`、`idx2word`、`tokenize_and_index`、`stoi`、`itos` 和特殊 token。
- `VocabFromText`：从句子集合统计词频并构造词表。
- `get_action_shortest_path(...)`：用 `ShortestPathFollower` 在 simulator 中生成动作级最短路径，返回 `ShortestPathPoint` 序列。
- `check_and_gen_physics_config()`：如果 `data/default.physics_config.json` 不存在，则写入默认 Bullet physics 配置。

该文件被 EQA/VLN 词表、PointNav episode 生成、Rearrange physics config 初始化共同使用。

## PointNav

目录：`pointnav/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 注册 `PointNav-v1`。 |
| `pointnav_dataset.py` | PointNav 数据集加载器，也是 ObjectNav、InstanceImageNav、Rearrange 的基础加载框架。 |
| `pointnav_generator.py` | 随机生成 PointNav episode 的工具函数。 |

### `pointnav_dataset.py`

`PointNavDatasetV1(Dataset)` 负责加载 Point Navigation 数据集。

关键字段：

- `episodes: List[NavigationEpisode]`
- `content_scenes_path = "{data_path}/content/{scene}.json.gz"`
- `CONTENT_SCENES_PATH_FIELD = "content_scenes_path"`
- `DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"`

主要方法：

- `check_config_paths_exist(config)`：检查主数据文件和 `scenes_dir`。
- `get_scenes_to_load(config)`：返回数据集中可加载的 scene 列表；支持主文件和 content scene 拆分文件。
- `_get_scenes_from_folder(content_scenes_path, dataset_dir)`：扫描 content 目录，按文件名推断 scene id。
- `_load_from_file(fname, scenes_dir)`：当前版本只支持 gzip JSON，读取后调用 `from_json()`。
- `__init__(config)`：
  - 初始化 `self.episodes`。
  - 加载 `config.data_path.format(split=config.split)`。
  - 如果存在 content scene 拆分目录，按 `config.content_scenes` 加载对应 scene 文件。
  - 如果没有拆分目录，则用 `build_content_scenes_filter(config)` 过滤 episode。
- `from_json(json_str, scenes_dir)`：
  - JSON dict 转为 `NavigationEpisode`。
  - goals 转为 `NavigationGoal`。
  - shortest paths 转为 `ShortestPathPoint`。
  - 如果 scene id 以 `data/scene_datasets/` 开头，则去掉此前缀并拼到 `scenes_dir` 下。

与之前某些版本相比，当前文件没有 pickle/binary 加载分支，也没有 `to_binary()` / `from_binary()`。

### `pointnav_generator.py`

用于程序化生成 PointNav episode。

主要函数：

- `_ratio_sample_rate(ratio, ratio_threshold)`：对 geodesic/euclidean 比例接近 1 的简单 episode 做更激进拒绝采样。
- `is_compatible_episode(s, t, sim, near_dist, far_dist, geodesic_to_euclid_ratio)`：
  - 检查起点终点高度差。
  - 检查 geodesic 距离是否有效。
  - 检查距离范围。
  - 检查 geodesic/euclidean 比例。
  - 检查 nav island 半径。
- `_create_episode(...)`：构造 `NavigationEpisode`。
- `generate_pointnav_episode(...)`：
  - 随机采样目标点和起点。
  - 可选生成动作级 shortest path。
  - yield 满足条件的 `NavigationEpisode`。

## ObjectNav

目录：`object_nav/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 注册 `ObjectNav-v1`。 |
| `object_nav_dataset.py` | ObjectNav 数据集加载器，继承 PointNav 的文件加载流程。 |

### `object_nav_dataset.py`

`ObjectNavDatasetV1(PointNavDatasetV1)` 负责 Object Navigation 数据反序列化。

关键字段：

- `category_to_task_category_id`
- `category_to_scene_annotation_category_id`
- `goals_by_category`
- `episodes: List[ObjectGoalNavEpisode]`

主要方法：

- `dedup_goals(dataset)`：
  - 兼容旧格式。
  - 将 episode 中重复的 object goals 抽到顶层 `goals_by_category`。
  - episode 中只保留 `object_category` 和空 goals。
- `to_json()`：
  - 序列化时临时清空 episode goals，避免重复保存。
  - 序列化后再把 goals 填回内存对象。
- `__deserialize_goal(serialized_goal)`：
  - dict 转 `ObjectGoal`。
  - `view_points` 转 `ObjectViewLocation`。
  - view point 内部 `agent_state` 转 `AgentState`。
- `from_json(json_str, scenes_dir)`：
  - 读取 category 映射，兼容旧字段 `category_to_mp3d_category_id`。
  - 校验 task category 和 scene annotation category 的 key 一致。
  - 加载 `goals_by_category`。
  - episode 转为 `ObjectGoalNavEpisode`。
  - 按 `goals_key` 把共享 goals 填回 episode。
  - shortest path point 支持 dict，也兼容 `None`、`int`、`str` 形式的 action。

## Instance Image Navigation

目录：`image_nav/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 注册 `InstanceImageNav-v1`。 |
| `instance_image_nav_dataset.py` | InstanceImageNav 数据集加载器，目标由实例图像和视点组成。 |

### `instance_image_nav_dataset.py`

`InstanceImageNavDatasetV1(PointNavDatasetV1)` 负责 Instance Image Navigation 数据反序列化。

关键字段：

- `goals: Dict[str, InstanceImageGoal]`
- `episodes: List[InstanceImageGoalNavEpisode]`

主要方法：

- `to_json()`：
  - 序列化时清空 episode 内 goals。
  - 序列化后按 `goal_key` 恢复 goals。
- `_deserialize_goal(serialized_goal)`：
  - dict 转 `InstanceImageGoal`。
  - view point 转 `ObjectViewLocation` 和 `AgentState`。
  - image goal 参数转 `InstanceImageParameters`。
- `from_json(json_str, scenes_dir)`：
  - 要求顶层存在 `goals`。
  - 顶层 goals 先构造成共享 goal 字典。
  - episode 转 `InstanceImageGoalNavEpisode`。
  - 根据 `goal_key` 填回目标。

## EQA

目录：`eqa/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 注册 `MP3DEQA-v1`。 |
| `mp3d_eqa_dataset.py` | Matterport3D Embodied Question Answering 数据集加载器。 |

### `mp3d_eqa_dataset.py`

主要内容：

- `get_default_mp3d_v1_config(split="val")`：
  - 创建默认 `DatasetConfig`。
  - 默认路径是 `data/datasets/eqa/mp3d/v1/{split}.json.gz`。
- `Matterport3dDatasetV1(Dataset)`：
  - 从 gzip JSON 读取 EQA 数据。
  - `answer_vocab` 和 `question_vocab` 转成 `VocabDict`。
  - episode 转 `EQAEpisode`。
  - question 转 `QuestionData`。
  - goal 转 `ObjectGoal`。
  - goal 的 view point 转 `AgentState`。
  - shortest paths 转 `ShortestPathPoint`。
  - 最后用 `build_content_scenes_filter(config)` 过滤 scene。

实现细节：`from_json()` 中使用 `self.__dict__.update(deserialized)` 先把 JSON 顶层字段直接灌入实例，再逐项修正类型。这是 Habitat 上游风格，依赖 JSON 结构稳定。

## VLN

目录：`vln/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 注册 `R2RVLN-v1`。 |
| `r2r_vln_dataset.py` | R2R Vision-and-Language Navigation 数据集加载器。 |

### `r2r_vln_dataset.py`

`VLNDatasetV1(Dataset)` 负责 R2R VLN 数据加载。

主要行为：

- 检查 `data_path` 和 `scenes_dir`。
- 从 gzip JSON 加载数据。
- 构造 `instruction_vocab: VocabDict`。
- episode 转 `VLNEpisode`。
- instruction 转 `InstructionData`。
- goals 转 `NavigationGoal`。
- 修正 scene 路径。
- 用 `build_content_scenes_filter(config)` 按 scene 过滤。

## Rearrange 数据集

目录：`rearrange/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 注册 `RearrangeDataset-v0`。 |
| `rearrange_dataset.py` | 定义 Rearrange episode 数据结构和数据集加载器。 |
| `rearrange_generator.py` | 核心 episode 生成器。 |
| `run_episode_generator.py` | 命令行入口和默认配置 dataclass。 |
| `combine_datasets.py` | 合并多个 rearrange `.json.gz` 数据集。 |
| `generate_episode_inits.py` | 遍历 Habitat Env episode 并触发 reset，用于生成初始化缓存。 |
| `configs/` | episode 生成 YAML 配置。 |
| `samplers/` | scene、object、target、receptacle、AO state 采样器。 |

当前版本没有 `rearrange/navmesh_utils.py`。与更复杂版本相比，机器人导航可达性、无遮挡 raycast、scene-balanced 采样等逻辑不在本目录中。

### `rearrange_dataset.py`

`RearrangeEpisode(Episode)` 是重排任务 episode 的数据结构，额外记录：

- `ao_states`：articulated object 的关节状态，格式类似 `{instance_handle -> {link_index -> state}}`。
- `rigid_objs`：episode 中需要额外加载的刚体对象及其 transform。
- `targets`：目标物体实例对应的目标 transform。
- `markers`：抓取点、推拉点等兴趣点。
- `target_receptacles`：目标物体初始所在 receptacle 的父对象和 link。
- `goal_receptacles`：目标位置所在 receptacle 的父对象和 link。
- `name_to_receptacle`：物体实例 handle 到 receptacle unique name 的映射。

`RearrangeDatasetV0(PointNavDatasetV1)`：

- 注册名为 `RearrangeDataset-v0`。
- `to_json()` 使用 `DatasetFloatJSONEncoder` 序列化。
- `__init__(config)`：
  - 保存 config。
  - 如果配置路径不存在，调用 `habitat_sim.utils.datasets_download` 下载 `rearrange_task_assets`。
  - 调用 `check_and_gen_physics_config()`。
  - 复用 `PointNavDatasetV1` 的加载流程。
- `from_json()`：
  - JSON episode 转为 `RearrangeEpisode`。
  - episode id 被重写为顺序编号字符串。

当前版本没有 binary 压缩读写逻辑。

### `rearrange_generator.py`

`RearrangeEpisodeGenerator` 是程序化生成 Rearrange episode 的核心。

初始化阶段：

- 保存 generator config。
- 根据 `debug_visualization` 控制是否创建 renderer 和 debug 输出。
- 初始化 HabitatSim，先加载 `"NONE"` scene 以读取 SceneDataset。
- 调用：
  - `_get_resource_sets()`
  - `_get_scene_sampler()`
  - `_get_obj_samplers()`
  - `_get_ao_state_samplers()`

资源解析：

- `_get_resource_sets()`：
  - 从 `cfg.scene_sets` 中筛选 scene handles。
  - 从 `cfg.object_sets` 中筛选 object template handles。
  - 从 `cfg.receptacle_sets` 中构造 `ReceptacleSet`。
- `_get_scene_sampler()`：
  - 支持 `single` 和 `subset` 两种 scene sampler。
  - `subset` 会合并多个 scene set，再构造 `MultiSceneSampler`。
- `_get_obj_samplers()`：
  - 当前只支持 `type: "uniform"`。
  - 构造 `ObjectSampler`。
- `_get_object_target_samplers()`：
  - 当前只支持 `type: "uniform"`。
  - 构造 `ObjectTargetSampler`。
- `_get_ao_state_samplers()`：
  - 支持 `uniform`、`categorical`、`composite` 三类 AO joint state sampler。

单个 episode 生成流程：

1. 创建 `ReceptacleTracker` 管理 receptacle 容量限制。
2. 重置 sampler 状态。
3. 采样 scene，并重新配置 simulator。
4. 从 scene 目录加载对应 navmesh 文件。
5. 为目标物体采样初始 receptacle。
6. 为目标状态采样 goal receptacle。
7. 对目标/goal receptacle 额外增加一次容量计数，避免容量限制误伤。
8. 采样 articulated object 状态，例如打开冰箱门或抽屉。
9. 用 object samplers 放置物体，记录物体所在 receptacle。
10. 运行 `settle_sim()`，检查物体物理稳定性。
11. 用 target samplers 为目标物体采样 goal transform。
12. 校验目标起点和 goal 的距离至少为 `min_dist_from_start_to_goal`。
13. 收集最终 rigid object transforms、target transforms、receptacle 信息，返回 `RearrangeEpisode`。

重要方法：

- `generate_episodes(num_episodes, verbose)`：循环生成多个 episode。
- `generate_single_episode()`：完整单 episode 生成逻辑。
- `initialize_sim(scene_name, dataset_path)`：创建或重配 HabitatSim，加载额外 object configs。
- `visualize_scene_receptacles()`：调试绘制 receptacle。
- `settle_sim(target_object_names, duration, make_video)`：
  - 跑物理仿真若干秒。
  - 统计物体位移。
  - 如果非目标物体不稳定，且配置允许，会尝试删除不稳定物体来保留 episode。
  - 如果目标物体不稳定，则 episode 失败。

当前实现使用 `vdb: DebugVisualizer` 作为调试可视化对象。

### `run_episode_generator.py`

这是生成 Rearrange 数据集的命令行入口，同时定义默认配置。

配置 dataclass：

- `SceneSamplerParamsConfig`
- `SceneSamplerConfig`
- `RearrangeEpisodeGeneratorConfig`

`RearrangeEpisodeGeneratorConfig` 覆盖的主要配置：

- `min_dist_from_start_to_goal`
- `gpu_device_id`
- `dataset_path`
- `additional_object_paths`
- `correct_unstable_results`
- `scene_sets`
- `object_sets`
- `receptacle_sets`
- `scene_sampler`
- `max_objects_per_receptacle`
- `object_samplers`
- `object_target_samplers`
- `ao_state_samplers`
- `markers`

命令行参数：

- `--config`：YAML 配置文件。
- `--out`：输出路径。
- `--list`：列出 SceneDataset 资源。
- `--run`：运行生成器并写出数据集。
- `--debug`：输出 debug 图片/视频。
- `--verbose`：显示进度条。
- `--db-output`：debug 输出目录。
- `--limit-scene-set`：限制 scene set。
- `--num-episodes`：episode 数量。
- `--seed`：随机种子。

需要注意：当前 `if args.list:` 分支里先取了 `mm = ep_gen.sim.metadata_mediator`，然后调用 `print_metadata_mediator(mm)`；但 `print_metadata_mediator()` 的参数名和实现预期是 `ep_gen`，内部访问 `ep_gen.sim.metadata_mediator`。这会导致 `--list` 分支很可能出错。

### `combine_datasets.py`

小型脚本，用来合并多个 Rearrange 数据集：

- 输入多个 `.json.gz`。
- 读取每个文件的 `episodes` 并拼接。
- 输出 `{ "episodes": all_episodes, "config": dat["config"] }`。
- `config` 来自最后一个输入文件。

### `generate_episode_inits.py`

小型脚本：

- 读取 Habitat config。
- 创建 `habitat.Env`。
- 遍历 `env.number_of_episodes`。
- 每个 episode 调用 `env.reset()`。
- 每 100 个 episode 打印一次当前配置和数据集路径。

通常用于触发 episode 初始化或缓存生成。

## Rearrange Samplers

目录：`rearrange/samplers/`

| 文件 | 功能 |
| --- | --- |
| `__init__.py` | 统一导出采样器类。 |
| `scene_sampler.py` | scene 采样器。 |
| `object_sampler.py` | 物体初始放置采样器。 |
| `object_target_sampler.py` | 目标位姿采样器。 |
| `art_sampler.py` | articulated object joint 状态采样器。 |
| `receptacle.py` | receptacle 抽象、解析、过滤和容量跟踪。 |

### `samplers/__init__.py`

从各文件导出：

- `ArticulatedObjectStateSampler`
- `ArtObjCatStateSampler`
- `CompositeArticulatedObjectStateSampler`
- `ObjectSampler`
- `ObjectTargetSampler`
- `SceneSampler`
- `SingleSceneSampler`
- `MultiSceneSampler`

当前版本没有 `BalancedSceneSampler`。

### `scene_sampler.py`

类：

- `SceneSampler`：抽象基类，定义 `num_scenes()`、`reset()`、`sample()`。
- `SingleSceneSampler`：固定返回一个 scene。
- `MultiSceneSampler`：从给定 scene 列表中均匀随机选一个。

### `object_sampler.py`

`ObjectSampler` 负责“选物体、选 receptacle、尝试放置”。

关键方法：

- `reset()`：清空当前 scene 的 receptacle 缓存并重新决定采样数量。
- `sample_receptacle(sim, recep_tracker, cull_tilted_receptacles, tilt_tolerance)`：
  - 通过 `ReceptacleSet` 的包含/排除 substring 筛选 receptacle。
  - 可按 `sample_probs` 对 receptacle set 加权。
  - 可剔除倾斜 receptacle。
  - 支持 `OnTopOfReceptacle` 这种特殊采样器。
- `sample_object()`：从 object set 中随机选 object template handle。
- `sample_placement(sim, object_handle, receptacle, snap_down, vdb)`：
  - 从 receptacle 中采样位置。
  - 实例化刚体物体。
  - 按 `orientation_sampling` 设置朝向。
  - 可使用 `sutils.snap_down()` 放到支撑面上。
  - 用 `_is_accessible()` 做简单 navmesh 距离检查。
- `_is_accessible(sim, obj)`：
  - 如果 `nav_to_min_distance == -1`，直接通过。
  - 否则把物体位置 snap 到 navmesh，检查 XZ 平面距离是否小于阈值。
- `single_sample()`：一次完整 object/receptacle/placement 尝试。
- `set_num_samples()`：从 `[min, max]` 中决定本次 episode 的采样数量。
- `sample()`：重复采样，直到达到目标数量或失败；失败时清理已创建对象。

可配置行为：

- `num_samples`
- `orientation_sampling`: `none` / `up` / `all`
- `sample_region_ratio`
- `nav_to_min_distance`
- `sample_probs`
- `constrain_to_largest_nav_island`

### `object_target_sampler.py`

`ObjectTargetSampler(ObjectSampler)` 用于为已存在物体采样目标位置。

核心差异：

- 它不是从 template set 直接采样任务物体，而是从已经放入场景的 `object_instance_set` 中选择目标物体。
- 对目标物体的 template 在 goal receptacle 上创建一个临时对象，用临时对象 transform 作为目标位姿。
- 成功时返回 `{原物体 handle -> (临时目标对象, 原始 receptacle)}`。
- 如果未能为所有目标采样 goal，会删除已创建的临时目标对象并返回 `None`。

### `art_sampler.py`

用于采样 articulated object 的关节状态。

类：

- `ArticulatedObjectStateSampler`：
  - 对匹配 `ao_handle` 和 `link_name` 的 articulated object link，在连续范围内均匀采样 joint state。
- `ArtObjCatStateSampler`：
  - 继承上者，但从给定候选状态中离散采样。
- `CompositeArticulatedObjectStateSampler`：
  - 同时采样多个 AO/link 状态。
  - 只对与目标/goal receptacle 匹配的 link 采样。
  - 每次采样后用 contact test 验证组合状态。
  - 最多尝试 `max_iterations = 50` 次，失败返回 `None`。

### `receptacle.py`

Rearrange 采样中的“可放置区域”定义和解析逻辑。

核心类：

- `Receptacle`：
  - 抽象基类。
  - 表示 stage、rigid object 或 articulated object link 上的可采样区域。
  - 定义 `sample_uniform_local()`、`sample_uniform_global()`、`get_global_transform()`、`debug_draw()` 等接口。
- `OnTopOfReceptacle`：
  - 特殊 receptacle，用于“放在某个已采样目标上方”的组合关系。
- `AABBReceptacle`：
  - 用局部 AABB 表示体积/区域。
  - 支持 global stage receptacle 的旋转。
  - 可添加 wire box 可视化对象。
- `TriangleMeshReceptacle`：
  - 用三角网格表面作为采样区域。
  - 按三角形面积加权采样。
- `ReceptacleSet`：
  - YAML 配置中的 receptacle 集合，包含 object/receptacle 的 include/exclude substring。
- `ReceptacleTracker`：
  - 管理 `max_objects_per_receptacle`。
  - 当某个 receptacle 容量用完后，将它加入各 receptacle set 的排除列表。
  - 会根据 scene metadata 中的 `scene_filter_file` 加载过滤名单。

核心函数：

- `get_all_scenedataset_receptacles(sim)`：
  - 扫描 SceneDataset 的 stage、rigid object、articulated object template 中定义的 receptacle metadata。
- `filter_interleave_mesh(mesh)`：
  - 将 mesh 规整为三角形、仅保留 position 属性并 interleave。
- `import_tri_mesh(mesh_file)`：
  - 通过 Magnum importer 加载 mesh receptacle。
- `parse_receptacles_from_user_config(...)`：
  - 从 Habitat-Sim user config 中解析 `receptacle_aabb_*` 和 `receptacle_mesh_*`。
- `find_receptacles(sim, ignore_handles=None)`：
  - 扫描当前 simulator 中已实例化的 stage、rigid object、articulated object，返回所有 receptacle。

## Rearrange 配置文件

目录：`rearrange/configs/`

这些 YAML 文件是 `run_episode_generator.py --config` 的输入。常见字段：

- `dataset_path`：SceneDataset 配置路径。
- `additional_object_paths`：额外物体配置目录。
- `scene_sets`：用 substring 定义可采样 scene 集合。
- `object_sets`：用 substring 定义可采样物体集合。
- `receptacle_sets`：定义可放置区域集合。
- `scene_sampler`：选择 `single` 或 `subset`。
- `object_samplers`：定义初始物体放置。
- `object_target_samplers`：定义目标位姿采样。
- `ao_state_samplers`：定义 articulated object 状态采样。
- `markers`：定义任务相关兴趣点。
- `max_objects_per_receptacle`：限制 receptacle 容量。

当前配置文件：

| 文件 | 用途概括 |
| --- | --- |
| `configs/empty.yaml` | 基础空配置，定义 scene/object/receptacle/markers，但不采样物体和目标。 |
| `configs/test_config.yaml` | 简单测试配置：在 kitchen counter 上采样 1 个 kitchen object，并采样 1 个目标位置。 |
| `configs/all_receptacles.yaml` | 覆盖多类 receptacle 的较完整示例配置。 |
| `configs/bench_config.yaml` | 使用 `data/hab2_bench_assets/hab2_bench.scene_dataset_config.json` 的 benchmark 配置。 |
| `configs/in_fridge.yaml` | 采样物体到冰箱相关 receptacle。 |
| `configs/in_drawer.yaml` | 采样物体到抽屉相关 receptacle。 |
| `configs/hab/rearrange.yaml` | 通用 Habitat rearrange 任务配置。 |
| `configs/hab/rearrange_easy.yaml` | 简化 Rearrange 配置，通常用于较容易的移动目标任务。 |
| `configs/hab/prepare_groceries.yaml` | “准备食品/杂货”风格配置，包含更多食品和容器集合。 |
| `configs/hab/set_table.yaml` | “摆桌子”风格配置，关注餐桌和餐具目标。 |
| `configs/hab/tidy_house.yaml` | “整理房间”风格配置。 |

## 典型数据流

### 加载已有数据集

```text
make_dataset("ObjectNav-v1", config=...)
  -> registry.get_dataset("ObjectNav-v1")
  -> ObjectNavDatasetV1(config)
  -> PointNavDatasetV1.__init__()
  -> _load_from_file(...json.gz...)
  -> ObjectNavDatasetV1.from_json()
  -> 得到 typed episodes/goals/paths
```

PointNav、ObjectNav、InstanceImageNav、Rearrange 都复用类似的 `data_path + split + content_scenes` 加载机制。EQA 和 VLN 直接继承 `Dataset`，但也会在加载后做 scene 过滤。

### 生成 Rearrange 数据集

```text
run_episode_generator.py --config xxx.yaml --run --num-episodes N --out out.json.gz
  -> 读取默认 config
  -> merge YAML override
  -> 创建 RearrangeEpisodeGenerator
  -> 采样 scene/object/receptacle/AO state/target
  -> settle_sim 检查物理稳定性
  -> 生成 RearrangeEpisode
  -> RearrangeDatasetV0.to_json()
  -> 写出 gzip JSON
```

## 依赖关系简图

```text
registration.py
  -> 各子包 __init__.py
     -> 各 Dataset 类注册到 registry

PointNavDatasetV1
  -> ObjectNavDatasetV1
  -> InstanceImageNavDatasetV1
  -> RearrangeDatasetV0

utils.py
  -> EQA/VLN vocabulary
  -> PointNav shortest path generation
  -> Rearrange default physics config

RearrangeEpisodeGenerator
  -> RearrangeEpisode
  -> ObjectSampler
  -> ObjectTargetSampler
  -> SceneSampler
  -> ArticulatedObjectStateSampler
  -> Receptacle / ReceptacleTracker
```

## 当前版本需要注意的点

- 当前 `PointNavDatasetV1` 只支持 gzip JSON，不支持 `.pickle`。
- 当前 `RearrangeDatasetV0` 没有 binary 读写逻辑；资产缺失时会尝试下载 `rearrange_task_assets`。
- 当前 `rearrange/samplers/scene_sampler.py` 只有 `SingleSceneSampler` 和 `MultiSceneSampler`，没有 balanced scene sampler。
- 当前没有 `rearrange/navmesh_utils.py`，复杂机器人导航可达性检查不在此版本中。
- `rearrange/__init__.py` 的失败占位注册名疑似不一致：`OrpNavDataset-v0` vs `RearrangeDataset-v0`。
- `run_episode_generator.py` 的 `--list` 分支疑似传参错误：`print_metadata_mediator()` 预期 generator，但调用处传了 metadata mediator。
- `SceneSamplerConfig` 和 `RearrangeEpisodeGeneratorConfig` 中直接以 `SceneSamplerParamsConfig()`、`SceneSamplerConfig()` 作为 dataclass 默认值，而不是 `default_factory`；这在某些 Python/dataclasses 版本中可能引发可变默认值问题。
- 这些文件保留了 Habitat 包内 import 路径，例如 `habitat.datasets...`。在当前项目中能否直接运行，取决于 `habitat` 包路径是否正确映射到 `third_part` 或已安装 Habitat-Lab。
