# 🚀 快速开始指南

## 三步完成完整工作流

### 前置准备
```bash
# 1. 安装依赖
pip install openai numpy

# 2. 准备物体图片（可选，用于测试）
# 文件夹: ./objects_images/
# 示例: mug_01.jpg, chair_02.png, ...
# 注: 文件名(不含扩展名) = 物体ID
```

### 工作流核心命令

#### 第1步：Qwen3-VL查询（获取房间推荐）
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

**输出**: 
```
results/scene_info/00800-TEEsavR23oF/
├── mug_01_rooms.json       ← 物体1的房间推荐
├── chair_02_rooms.json     ← 物体2的房间推荐
└── ...
```

**关键参数说明**:
- `--ssh-host`: 远程服务器地址
- `--ssh-port`: SSH端口（通常是31822）
- `--ssh-password`: SSH密码（需预装sshpass）
- `--vllm-port`: vLLM服务端口（通常是8000）

---

#### 第2步：生成概率并采样编辑（首次）
```bash
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode generate \
  --images-dir ./objects_images \
  --rooms-info-dir ./results/scene_info \
  --probabilities-dir ./results/probabilities \
  --layouts-dir ./results/layouts
```

**交互式流程**:
1. ✓ 根据房间推荐随机生成概率分布
2. ✓ 采样每个物体的位置（按概率选房间）
3. ✓ **自动启动编辑器** ← 可视化调整
4. ✓ 编辑器中按 [M] 保存
5. ✓ 关闭编辑器，选择是否继续循环

**输出**:
```
results/probabilities/00800-TEEsavR23oF/
├── mug_01_probs.json        ← 物体1的概率分布
├── chair_02_probs.json
└── ...

results/layouts/00800-TEEsavR23oF/
├── final_1743676123.json    ← 最终布局（可多次迭代）
└── ...
```

---

#### 第3步：后续重复采样（快速迭代）
```bash
python sample_and_place_objects.py \
  --scene 00800-TEEsavR23oF \
  --mode load \
  --layouts-dir ./results/layouts
```

**优势**: 直接复用之前生成的概率，快速重复采样和编辑，无需再询问Qwen

---

## 📁 输入输出一览

### 输入
| 来源 | 位置 | 格式 |
|------|------|------|
| 物体图片 | `./objects_images/` | JPG/PNG 任意图片 |
| 场景数据 | `./hm3d/minival/{scene}/` | 预置（无需修改） |
| 房间推荐 | `results/scene_info/{scene}/` | JSON（query脚本生成） |
| 概率分布 | `results/probabilities/{scene}/` | JSON（sample脚本生成或用户编辑） |

### 输出
| 步骤 | 文件 | 位置 | 用途 |
|------|------|------|------|
| 查询 | `{object}_rooms.json` | `results/scene_info/{scene}/` | Qwen回复 + 前5房间推荐 |
| 概率 | `{object}_probs.json` | `results/probabilities/{scene}/` | 每个房间的采样概率 |
| 布局 | `final_*.json` | `results/layouts/{scene}/` | 最终物体位置记录 |

---

## 🎮 编辑器快速参考

启动自动编辑器后的常用快捷键（来自 `test_layout.py`）：

### 相机移动
| 按键 | 功能 |
|------|------|
| **W/S** | 前进/后退 |
| **A/D** | 左移/右移 |
| **E/C** | 上升/下降 |
| **J/L** | 左右转向 |
| **I/K** | 俯仰 |

### 物体编辑
| 按键 | 功能 |
|------|------|
| **,/.** 或 **9/0** | 切换选中物体 |
| **T/G** | 前后移动 |
| **F/Y** | 左右移动 |
| **U/O** | 上下移动 |
| **1/2** 或 **!/@** | 左转/右转 |
| **B** | 删除选中物体 |
| **M** | **保存布局** |
| **[/]** | 切换布局文件 |
| **H** | 显示/隐藏帮助 |
| **ESC** 或 **Q** | 退出编辑器 |

---

## ⚙️ 常用命令备忘

### 只查询某个物体
```bash
# 如果希望对特定图片独立查询
cd objects_images/
# ... 将要查询的图片放入此目录
cd ..
python query_rooms_for_objects.py --ssh-host ... --scene 00800-... --images-dir ./objects_images
```

### 修改概率后重新采样
```bash
# 编辑某个概率文件
nano results/probabilities/00800-TEEsavR23oF/mug_01_probs.json
# 确保 "probability" 字段总和 = 1.0

# 再次采样（会自动使用修改后的概率）
python sample_and_place_objects.py --scene 00800-TEEsavR23oF --mode load
```

### 在编辑器中加载之前的布局
```bash
# 用编辑器打开已保存的布局进行再编辑
python test_layout.py 00800-TEEsavR23oF --layout ./results/layouts/00800-TEEsavR23oF/final_*.json
```

### 查看某个输出的详细内容
```bash
# 查看房间推荐
python -m json.tool results/scene_info/00800-TEEsavR23oF/mug_01_rooms.json | less

# 查看采样概率
python -m json.tool results/probabilities/00800-TEEsavR23oF/mug_01_probs.json | less
```

---

## 🔧 故障排查速查表

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| **SSH隧道连接失败** | `SSH tunnel did not become ready` | 检查密码、确认sshpass已装、ping服务器 |
| **Qwen无响应** | `Request failed: ...` | 增加 `--timeout` 参数；检查vLLM日志 |
| **编辑器未启动** | `Failed to launch editor` | 检查 `test_layout.py` 路径；确认habitat环境 |
| **找不到物体模板** | 编辑器提示 `Cannot find template` | 文件名应与 `objects/*_config.json` 匹配 |
| **房间识别错误** | 推荐房间坐标为全0 | Qwen回复格式不符；检查 `_rooms.json` 并手工修改 |

---

## 📖 详细文档

- **完整工作流说明**: [README_WORKFLOW.md](README_WORKFLOW.md)
- **脚本帮助信息**:
  ```bash
  python query_rooms_for_objects.py --help
  python sample_and_place_objects.py --help
  ```
- **验证系统完整性**:
  ```bash
  python verify_workflow.py
  ```

---

## 💡 使用建议

1. **首次运行**: 先处理 1-2 个物体，确保流程顺畅
2. **多次迭代**: 使用 `--mode load` 复用概率，快速调整
3. **概率调优**: 若某房间采样次数过多，降低概率权重
4. **批量处理**: 多个物体可并行查询Qwen，再统一采样编辑

---

**祝使用愉快！🎉**
