# 场景数据集与任务集生成命令指南

本文档按执行顺序汇总本项目中“动态家庭场景 layout 数据集”和“导航任务 episode 集”的生成命令。

所有可执行命令均统一使用：

```bash
python log_filter.py --run "<真实命令>"
```

这样可以过滤 Habitat/HM3D 高频日志噪声，并在末尾输出过滤统计。

默认示例场景为：

```text
00808-y9hTuugGdiq
```

默认远程 Qwen3-VL 连接方式为 SSH 密码登录：

```text
host: 7.216.187.6
port: 30180
user: root
password: 666666
remote vLLM: 127.0.0.1:8000
```

> 说明：相关脚本已内置上述默认值。命令中显式写出 `--ssh-password 666666` 是为了让运行配置更清楚。

---

## 0. 环境与输入检查

在项目根目录执行：

```bash
python log_filter.py --run "python verify_workflow.py"
```

检查关键输入目录：

```bash
python log_filter.py --run "ls hm3d"
python log_filter.py --run "ls objects"
python log_filter.py --run "ls objects_images"
```

若使用默认密码方式连接 Qwen，Linux 环境需要安装 `sshpass`：

```bash
python log_filter.py --run "sshpass -V"
```

手动测试 SSH 登录：

```bash
python log_filter.py --run "SSHPASS=666666 sshpass -e ssh -p 30180 root@7.216.187.6"
```

测试远程 Qwen/vLLM 隧道与图文调用：

```bash
python log_filter.py --run "python qwen3_vl_connect.py \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --image objects_images/Camera_01.webp \
  --prompt 'Describe this object briefly.'"
```

---

## 1. 导出场景语义信息

单场景导出：

```bash
python log_filter.py --run "python export_scene_info.py \
  --scene 00808-y9hTuugGdiq \
  --output-dir ./results/scene_info/00808-y9hTuugGdiq"
```

全量导出：

```bash
python log_filter.py --run "python export_scene_info.py \
  --all \
  --output-dir ./results/scene_info"
```

预期产物：

```text
results/scene_info/<scene>/<scene>_scene_info.json
```

---

## 2. 生成物体房间推荐

对单个场景、同一批 `objects_images` 物体图片生成房间推荐：

```bash
python log_filter.py --run "python query_rooms_for_objects.py \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000 \
  --images-dir ./objects_images \
  --scene 00808-y9hTuugGdiq \
  --output-dir ./results/scene_info/"
```

预期产物：

```text
results/scene_info/<scene>/<object>_rooms.json
```

---

## 3. 生成或复用房间概率分布

首次生成概率，并输出一个初始 layout：

```bash
python log_filter.py --run "python sample_and_place_objects.py \
  --scene 00808-y9hTuugGdiq \
  --mode generate \
  --images-dir ./objects_images \
  --rooms-info-dir ./results/scene_info \
  --probabilities-dir ./results/probabilities \
  --layouts-dir ./results/layouts \
  --placement auto"
```

后续复用已有概率重新采样：

```bash
python log_filter.py --run "python sample_and_place_objects.py \
  --scene 00808-y9hTuugGdiq \
  --mode load \
  --images-dir ./objects_images \
  --rooms-info-dir ./results/scene_info \
  --probabilities-dir ./results/probabilities \
  --layouts-dir ./results/layouts \
  --placement auto"
```

预期产物：

```text
results/probabilities/<scene>/<object>_probs.json
results/layouts/<scene>/temp_*.json
```

---

## 4. 生成可放置承载面

使用 LLM 辅助筛选承载面：

```bash
python log_filter.py --run "python query_room_receptacle_objects.py \
  --scene 00808-y9hTuugGdiq \
  --data-dir ./hm3d \
  --scene-info-path ./results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json \
  --output ./results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000 \
  --surface-points-per-instance 256 \
  --surface-min-points 48 \
  --instance-pointcloud-points 2048"
```

不使用 LLM 的启发式快速版本：

```bash
python log_filter.py --run "python query_room_receptacle_objects.py \
  --scene 00808-y9hTuugGdiq \
  --data-dir ./hm3d \
  --scene-info-path ./results/scene_info/00808-y9hTuugGdiq/00808-y9hTuugGdiq_scene_info.json \
  --output ./results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json \
  --disable-llm"
```

预期产物：

```text
results/receptacle_queries/<scene>/<scene>_receptacle_surfaces_all_rooms.json
results/receptacle_queries/<scene>/surface_pointclouds/*.ply
```

---

## 5. 单次物体到承载实例分配与最终放置

如果已经有 sampled layout，可指定 `--object-layout`：

```bash
python log_filter.py --run "python assign_objects_to_receptacle_instances.py \
  --scene 00808-y9hTuugGdiq \
  --object-layout ./results/layouts/00808-y9hTuugGdiq/temp_auto_example.json \
  --surfaces-json ./results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000"
```

若不提供 `--object-layout`，脚本会按概率现场采样：

```bash
python log_filter.py --run "python assign_objects_to_receptacle_instances.py \
  --scene 00808-y9hTuugGdiq \
  --surfaces-json ./results/receptacle_queries/00808-y9hTuugGdiq/00808-y9hTuugGdiq_receptacle_surfaces_all_rooms.json \
  --images-dir ./objects_images \
  --rooms-info-dir ./results/scene_info \
  --probabilities-dir ./results/probabilities \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000"
```

预期产物：

```text
results/object_instance_assignments/<scene>/*_object_instance_plan.json
results/layouts/<scene>/*assigned_instance_layout.json
```

---

## 6. 批量生成同场景多时间 layout 数据集

推荐直接使用批量脚本。它会尽量复用已有 `scene_info / probabilities / surfaces`，只对每个 layout 重新 sample 和放置。

默认 LLM 版本：

```bash
python log_filter.py --run "python batch_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --num-layouts 10 \
  --base-seed 42 \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000"
```

无远端 LLM 的快速 smoke test：

```bash
python log_filter.py --run "python batch_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --num-layouts 2 \
  --base-seed 42 \
  --disable-assignment-llm \
  --disable-surface-llm"
```

强制重算概率和承载面：

```bash
python log_filter.py --run "python batch_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --num-layouts 10 \
  --regenerate-probabilities \
  --regenerate-surfaces \
  --ssh-password 666666"
```

预期产物：

```text
results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/
  manifest.json
  layout_000_seed_42.json
  layout_001_seed_43.json
  ...
```

---

## 7. 多场景计划模式批量生成

先准备计划文件，例如 `scenes_plan.json`：

```json
{
  "num_layouts": 5,
  "base_seed": 100,
  "scenes": [
    "00808-y9hTuugGdiq",
    {
      "scene": "00800-TEEsavR23oF",
      "num_layouts": 3,
      "base_seed": 200
    }
  ]
}
```

运行计划：

```bash
python log_filter.py --run "python batch_generate_layouts.py \
  --plan-json scenes_plan.json \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000"
```

预期产物：

```text
results/layouts/<scene>/batch_<YYYYmmdd_HHMMSS>/manifest.json
results/layouts/plan_<YYYYmmdd_HHMMSS>/plan_manifest.json
```

---

## 8. 可视化检查 layout 数据集

严格复现原始 layout：

```bash
python log_filter.py --run "python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --initial-y-offset 0"
```

调试物体高度偏移：

```bash
python log_filter.py --run "python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --debug-offset \
  --initial-y-offset 0 \
  --offset-step 0.02"
```

跨 batch 目录比较同一场景的多个 layout：

```bash
python log_filter.py --run "python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --layout-scan-dir results/layouts/00808-y9hTuugGdiq \
  --recursive-layout-scan \
  --initial-y-offset 0"
```

无窗口截图验收：

```bash
python log_filter.py --run "python visualize_placed_layout.py \
  results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/layout_000_seed_42.json \
  --scene 00808-y9hTuugGdiq \
  --headless \
  --headless-max-focus 20 \
  --initial-y-offset 0"
```

---

## 9. 由 layout 数据集生成导航 episode 任务集

从单个 batch manifest 生成 episode：

```bash
python log_filter.py --run "python -m benchmark.build_episodes \
  --layout-manifest results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/manifest.json \
  --version dynamic_household_v1 \
  --split val \
  --images-dir objects_images \
  --episodes-per-layout 3 \
  --min-subtasks 5 \
  --max-subtasks 10 \
  --success-radius 1.2 \
  --max-steps 500 \
  --seed 42"
```

从多个 scene 的 manifest 一次生成：

```bash
python log_filter.py --run "python -m benchmark.build_episodes \
  --layout-manifest \
    results/layouts/00808-y9hTuugGdiq/batch_<id_a>/manifest.json \
    results/layouts/00800-TEEsavR23oF/batch_<id_b>/manifest.json \
  --version dynamic_household_v1 \
  --split train \
  --images-dir objects_images \
  --episodes-per-layout 3"
```

预期产物：

```text
benchmark/episodes/<version>/<split>/<scene>/<layout_id>/*.json
benchmark/splits/benchmark_split_<version>.json
```

---

## 10. 运行任务集并生成轨迹

Oracle smoke test：

```bash
python log_filter.py --run "python -m benchmark.runner \
  --episodes benchmark/episodes/dynamic_household_v1/val \
  --output benchmark/eval/dynamic_household_v1/oracle.jsonl \
  --mode oracle \
  --sample-start-pose"
```

No-op 失败基线：

```bash
python log_filter.py --run "python -m benchmark.runner \
  --episodes benchmark/episodes/dynamic_household_v1/val \
  --output benchmark/eval/dynamic_household_v1/noop.jsonl \
  --mode noop"
```

接入自定义 agent：

```bash
python log_filter.py --run "python -m benchmark.runner \
  --episodes benchmark/episodes/dynamic_household_v1/val \
  --output benchmark/eval/my_agent/run.jsonl \
  --agent-module my_agent_module:create_agent \
  --agent-id my_agent \
  --sample-start-pose \
  --load-layout-objects"
```

预期产物：

```text
benchmark/eval/<version>/*.jsonl
```

---

## 11. 离线评测

评测 oracle：

```bash
python log_filter.py --run "python -m benchmark.evaluate \
  --episodes benchmark/episodes/dynamic_household_v1/val \
  --trajectories benchmark/eval/dynamic_household_v1/oracle.jsonl \
  --output-dir benchmark/eval/dynamic_household_v1/oracle"
```

评测 noop：

```bash
python log_filter.py --run "python -m benchmark.evaluate \
  --episodes benchmark/episodes/dynamic_household_v1/val \
  --trajectories benchmark/eval/dynamic_household_v1/noop.jsonl \
  --output-dir benchmark/eval/dynamic_household_v1/noop"
```

预期产物：

```text
benchmark/eval/<version>/<run_name>/summary.json
benchmark/eval/<version>/<run_name>/episode_results.jsonl
benchmark/eval/<version>/<run_name>/by_task_type.json
benchmark/eval/<version>/<run_name>/exploration_curve.json
```

---

## 12. 最小推荐流水线

如果只想快速从 layout 生成跑到 episode 评测，可按下面顺序执行：

```bash
python log_filter.py --run "python verify_workflow.py"

python log_filter.py --run "python batch_generate_layouts.py \
  --scene 00808-y9hTuugGdiq \
  --num-layouts 2 \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 --vllm-port 8000"

python log_filter.py --run "python -m benchmark.build_episodes \
  --layout-manifest results/layouts/00808-y9hTuugGdiq/batch_<YYYYmmdd_HHMMSS>/manifest.json \
  --version smoke_v1 \
  --split val \
  --images-dir objects_images \
  --episodes-per-layout 1"

python log_filter.py --run "python -m benchmark.runner \
  --episodes benchmark/episodes/smoke_v1/val \
  --output benchmark/eval/smoke_v1/oracle.jsonl \
  --mode oracle"

python log_filter.py --run "python -m benchmark.evaluate \
  --episodes benchmark/episodes/smoke_v1/val \
  --trajectories benchmark/eval/smoke_v1/oracle.jsonl \
  --output-dir benchmark/eval/smoke_v1/oracle"
```

将 `<YYYYmmdd_HHMMSS>` 替换为实际生成的 batch 目录名。

