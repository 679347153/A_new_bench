# AI物体房间推理 + 概率采样布局系统

完整工作流：Qwen3-VL推理 → 房间概率集合 → 概率采样 → 人工编辑微调

## 📋 二文件架构

### 文件1：`query_rooms_for_objects.py`
**目的**：批量查询物体在各场景中最有可能出现的房间位置

**输入**：
- 场景列表（单个或全量）
- 物体图片目录 (`objects_images/`) — 文件名即物体ID
- Qwen3-VL SSH密码认证参数

**处理流程**：
1. 加载/实时导出每个场景的语义信息 (scene_info JSON)
2. 建立SSH隧道连接远程Qwen3-VL服务
3. 对每张图片调用Qwen询问："该物体最有可能出现在房间的哪些地方？"
4. 解析回复，提取前5个房间推荐（region_id + 3D中心 + 置信分数）
5. 生成结构化JSON，保存到 `results/scene_info/{scene_name}/{object_name}_rooms.json`

**输出JSON结构**：
```json
{
  "scene_info": {
    "scene_name": "00800-TEEsavR23oF",
    "image": "mug_01.jpg",
    "object_name": "mug_01",
    "image_path": "/full/path",
    "query_timestamp": "2026-04-01 12:34:56",
    "model": "Qwen/Qwen3-VL-235B-A22B-Thinking"
  },
  "query": "看图片中的物体，它最有可能出现在房间的哪些地方？...",
  "raw_output": "<think>...</think>房间1: region_id=0, ...",
  "cleaned_output": "房间1: region_id=0, 中心=(-2.5, 1.2, 3.8), 置信度=0.95, 理由：客厅中央\n...",
  "recommended_rooms": [
    {
      "rank": 1,
      "region_id": 0,
      "confidence_score": 0.95,
      "room_center": [-2.5, 1.2, 3.8],
      "room_aabb": {"min": [...], "max": [...]},
      "reasoning": "客厅中央的茶几"
    },
    ...
  ],
  "metadata": {
    "total_objects_in_scene": 393,
    "total_rooms_in_scene": 11,
    "top_5_found": 5
  }
}
```

---

### 文件2：`sample_and_place_objects.py`
**目的**：根据概率分布采样物体位置，启动交互式编辑器进行微调

**两个工作模式**：
- `--mode generate`：首次运行，从query结果随机分配概率给前5房间
- `--mode load`：后续运行，直接读取已生成概率文件

**处理流程**：
1. 加载/生成概率文件（对每个物体，前5房间各分配概率，∑=1）
2. 采样：每个物体根据概率抽签选一个房间，作为初始位置
3. 生成中间布局JSON（包含所有采样物体的位置、旋转、置信度）
4. 自动启动 `test_layout.py` 编辑器进行人工微调
5. 用户保存后选择：
   - 继续循环（重新采样、编辑）
   - 保存最终布局并退出

**概率文件格式** (`results/probabilities/{scene_name}/{object_name}_probs.json`):
```json
{
  "object_name": "mug_01",
  "scene_name": "00800-TEEsavR23oF",
  "generated_timestamp": "2026-04-01 12:34:56",
  "probabilities": [
    {
      "rank": 1,
      "region_id": 0,
      "room_center": [-2.5, 1.2, 3.8],
      "probability": 0.35
    },
    {
      "rank": 2,
      "region_id": 2,
      "room_center": [-1.0, 0.9, 5.2],
      "probability": 0.28
    },
    ...
  ]
}
```

**采样布局JSON格式** (临时或最终):
```json
{
  "scene": "hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
  "timestamp": 1743676496.123,
  "objects": [
    {
      "id": 0,
      "model_id": "mug_01",
      "name": "Mug 01",
      "position": [-2.5, 1.2, 3.8],
      "rotation": [0.0, 0.0, 0.0],
      "confidence": 0.35,
      "source": "probability_sampling"
    },
    ...
  ]
}
```

---

## 🚀 快速开始

### 前置条件

1. **安装依赖**：
   ```bash
   pip install openai numpy
   # 若要实时导出scene_info，还需要habitat-sim（可选）
   pip install habitat-sim
   ```

2. **remote Qwen3-VL 服务**（假设已在运行）：
   - SSH可访问的服务器（支持密码认证）
   - 服务器内运行 vLLM 暴露 OpenAI 兼容API (默认 :8000)

3. **物体图片准备**：
   ```
   objects_images/
   ├── mug_01.jpg
   ├── chair_02.png
   ├── table_03.jpg
   └── ...
   ```
   文件名（不含扩展名）作为物体ID，与场景中的模板名对应。

### 步骤1：生成场景信息（可选）

如果没有 `results/scene_info/{scene_name}/*_scene_info.json` 文件：

```bash
# 使用export_scene_info.py生成
python export_scene_info.py --all --output-dir ./results/scene_info
# 或单个场景
python export_scene_info.py --scene 00800-TEEsavR23oF --output-dir ./results/scene_info
```

如果源代码没有这个脚本，文件1会在运行时自动尝试导出。

### 步骤2：查询Qwen获取房间推荐

```bash
python query_rooms_for_objects.py \
  --ssh-host 7.216.187.6 \
  --ssh-port 31822 \
  --ssh-user root \
  --ssh-password 666666 \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --images-dir ./objects_images \
  --scene 00800-TEEsavR23oF \
  --output-dir ./results/scene_info
```

**输出**：
```
results/scene_info/00800-TEEsavR23oF/
├── mug_01_rooms.json
├── chair_02_rooms.json
└── ...
```

### 步骤3a：首次采样（生成概率）

```bash
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode generate \
  --images-dir ./objects_images \
  --rooms-info-dir ./results/scene_info \
  --probabilities-dir ./results/probabilities \
  --layouts-dir ./results/layouts \
  --ui-lang zh
```

**过程**：
1. 加载 `results/scene_info/00800-TEEsavR23oF/*_rooms.json`
2. 为每个物体的前5房间随机分配概率 → 保存到 `results/probabilities/00800-TEEsavR23oF/*_probs.json`
3. 根据概率采样每个物体的位置
4. 启动 `test_layout.py` 编辑器，显示采样结果
5. 用户人工调整物体位置、旋转（按 [M] 保存）
6. 编辑器关闭后，提示保存最终布局

### 步骤3b：后续采样（复用已有概率）

```bash
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode load \
  --images-dir ./objects_images \
  --probabilities-dir ./results/probabilities \
  --layouts-dir ./results/layouts
```

**优势**：复用之前生成的概率分布，可快速迭代多次微调。

---

## 📂 目录结构变化

运行完整工作流后，目录树如下：

```
semantic/
├── results/
│   ├── scene_info/
│   │   ├── 00800-TEEsavR23oF/
│   │   │   ├── mug_01_rooms.json
│   │   │   ├── chair_02_rooms.json
│   │   │   └── ...
│   │   └── ...
│   ├── probabilities/
│   │   ├── 00800-TEEsavR23oF/
│   │   │   ├── mug_01_probs.json
│   │   │   ├── chair_02_probs.json
│   │   │   └── ...
│   │   └── ...
│   └── layouts/
│       ├── 00800-TEEsavR23oF/
│       │   ├── final_1743676123.json
│       │   ├── final_1743676234.json
│       │   └── ...
│       └── ...
├── objects_images/
│   ├── mug_01.jpg
│   ├── chair_02.png
│   └── ...
├── query_rooms_for_objects.py
├── sample_and_place_objects.py
├── test_layout.py     (already exists)
├── export_scene_info.py  (already exists)
└── ...
```

---

## ⚙️ 关键参数详解

### query_rooms_for_objects.py

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `--scale` | `00800-...` | 单个场景或 `all` 全量 |
| `--images-dir` | `./objects_images` | 物体图片目录 |
| `--output-dir` | `./results/scene_info` | 结果保存目录 |
| `--ssh-host` | 必需 | SSH服务器地址 |
| `--ssh-port` | 22 | SSH端口 |
| `--ssh-user` | 必需 | SSH用户名 |
| `--ssh-password` | - | SSH密码（非交互模式） |
| `--ssh-key` | - | SSH私钥路径（可选） |
| `--vllm-host` | 127.0.0.1 | vLLM API主机 |
| `--vllm-port` | 8000 | vLLM API端口 |
| `--model` | `Qwen/Qwen3-VL-235B-A22B-Thinking` | 模型名 |
| `--max-tokens` | 2048 | 回复最大长度 |
| `--timeout` | 3600 | 请求超时(秒) |

### sample_and_place_objects.py

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `--scene` | 必需 | 场景名 |
| `--mode` | `load` | `load` 或 `generate` |
| `--images-dir` | `./objects_images` | 物体图片目录 |
| `--rooms-info-dir` | `./results/scene_info` | query结果目录(--mode=generate时需要) |
| `--probabilities-dir` | `./results/probabilities` | 概率文件保存目录 |
| `--layouts-dir` | `./results/layouts` | 最终布局保存目录 |
| `--ui-lang` | `zh` | 编辑器UI语言 (`en` 或 `zh`) |

---

## 🔄 推荐工作流

### 场景A：初次体验（全流程）

```bash
# 1. 查询Qwen
python query_rooms_for_objects.py \
  --ssh-host <your_host> --ssh-port <port> --ssh-user <user> --ssh-password <pwd> \
  --images-dir ./objects_images \
  --scene 00800-TEEsavR23oF \
  --output-dir ./results/scene_info

# 2. 生成概率并首次微调
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode generate \
  --images-dir ./objects_images \
  --rooms-info-dir ./results/scene_info

# 3. 保存最终布局后，可在test_layout.py中加载进行更精细调整
python test_layout.py 00800-TEEsavR23oF --layout ./results/layouts/00800-TEEsavR23oF/final_*.json
```

### 场景B：快速迭代（后续调整）

```bash
# 重新采样、编辑、保存，无需再询问Qwen
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode load \
  --probabilities-dir ./results/probabilities

# 若要尝试不同的概率分布，重新生成
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode generate \
  --rooms-info-dir ./results/scene_info
```

---

## 🐛 常见问题

### Q1：query_rooms_for_objects.py 显示 SSH 隧道未就绪
**A**：
- 确认 `sshpass` 已安装：`sudo apt-get install -y sshpass`（Linux/macOS）或使用 `--ssh-key` 方式
- 检查SSH服务器地址、端口、用户名、密码正确
- 确保vLLM服务已启动且端口开放

### Q2：Qwen回复格式不规范，导致房间解析失败
**A**：
- 脚本使用正则表达式提取 `region_id=N`、`置信度=X.X` 等字段
- 若Qwen回复格式完全异常，可手动编辑对应的 `_rooms.json`，修改 `recommended_rooms` 字段
- 或调整 `QWEN_QUERY_TEMPLATE` 的提示词，要求更严格的输出格式

### Q3：编辑器中没有看到采样的物体
**A**：
- 检查物体模板名是否与 `objects/` 目录下的 `*.object_config.json` 匹配
- 确认 `objects_images/` 中的文件名（如 `mug_01.jpg`）要与物体模板ID一致
- 查看编辑器的错误日志，是否是模板加载失败

### Q4：如何修改概率分布？
**A**：
- 编辑 `results/probabilities/{scene}/{object}_probs.json` 中的 `probabilities` 字段
- 确保所有概率之和 = 1.0
- 再次运行 `sample_and_place_objects.py --mode load` 即可使用新概率

### Q5：如何在多个场景间复用一套物体？
**A**：
- 将 `results/scene_info/{scene1}/` 中的 `*_rooms.json` 复制到其他场景目录
- 对应更新 region_id 和房间中心坐标以匹配目标场景
- 或手工运行 query 对每个场景单独查询

---

## 📝 输出验证清单

| 步骤 | 预期输出 | 位置 |
|------|---------|------|
| 查询Qwen | `{object}_rooms.json` × N个物体 | `results/scene_info/{scene}/` |
| 生成概率 | `{object}_probs.json` × N个物体 | `results/probabilities/{scene}/` |
| 采样编辑 | `final_*.json` × 迭代次数 | `results/layouts/{scene}/` |

每个JSON文件都应包含以下关键字段：
- `recommended_rooms` / `probabilities` / `objects`（对应阶段）
- `timestamp`（生成时间戳）
- `scene_name` / `object_name`（元数据）

---

## 🔗 集成依赖

- **export_scene_info.py**：可选，用于实时导出scene_info
- **qwen3_vl_connect.py**：参考实现，query脚本已内置相同的隧道/API调用逻辑
- **test_layout.py**：交互式编辑器，sample脚本自动调用

---

## 📧 故障排查清单

1. **SSH隧道连接失败**
   - 运行 `ping <ssh_host>` 确认网络连通
   - 检查 `--ssh-password` 或 `--ssh-key` 认证参数
   - 尝试手工SSH连接确认凭证

2. **Qwen API调用超时**
   - 增加 `--timeout` 参数（默认3600秒）
   - 检查远程vLLM CPU/GPU负载
   - 查看vLLM日志是否有推理错误

3. **编辑器无法启动**
   - 确保 `test_layout.py` 在项目根目录
   - 检查Habitat环境配置（`hm3d_annotated_basis.scene_dataset_config.json` 路径）
   - 查看终端输出的详细错误信息

4. **最终布局未保存**
   - sample脚本会查询 `hm3d/minival/{scene}/configs/` 下最新修改的JSON
   - 若编辑器未按 [M] 保存，则无法自动检测
   - 手工复制编辑器配置文件到 `results/layouts/` 目录

---

祝使用愉快！有问题可参考各脚本的 docstring 或调整命令行参数。
