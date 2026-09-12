# Lifespan-Bench 论文与实验完善计划

生成日期：2026-09-07

本文档基于当前代码、`doc/` 文档、`benchmark/` 实现和 `paper/ICLR_zhangwei/iclr2026_conference.tex` 的检查结果，整理论文初稿还缺少什么，以及后续如何把它补成一篇更完整、可投稿、可复现实验支撑的论文。

## 1. 当前检查结论

### 1.1 已具备内容

当前论文已经具备基础论文骨架：

1. 标题、摘要、引言、贡献点。
2. Related Work，并已接入 ObjectNav、MultiON、ImageNav、GOAT-Bench、AI2-THOR、RoboTHOR、ProcTHOR、Habitat 2.0、iGibson、BEHAVIOR、动态家庭物体记忆、DynamicTHOR 等相关引用。
3. Benchmark Overview。
4. Grounded batch layout 生成路径说明。
5. Lifespan semantic-only 生成路径说明。
6. Task Suite。
7. Evaluation Protocol。
8. Dataset Structure。
9. Experiments 占位。
10. Discussion、Conclusion、Appendix 命令摘要。

当前代码已经具备以下支撑：

1. `core/batch_generate_layouts.py`：同一场景批量生成 grounded 3D layout。
2. `benchmark/build_episodes.py`：从 batch manifest 生成 episode。
3. `benchmark/runner.py`：支持 oracle/noop smoke agent 和外部 agent adapter。
4. `benchmark/evaluate.py`：输出 summary、episode_results、by_task_type、exploration_curve。
5. `core/lifespan_generate_layouts.py`：生成 semantic-only household lifespan sequence。
6. `data/lifespan/resident_persona_profiles.json`：50 个候选居民画像，经检查必需字段完整。
7. `data/object_catalog/object_catalog.json`：当前存在 8282 个 object entry，其中 YCB 77、HSSD 8205，且均已有 image 或 semantic text。

### 1.2 明确缺失项

论文层面缺失：

1. 6 个图仍是占位：
   - `fig:concept`
   - `fig:pipeline`
   - `fig:receptacle`
   - `fig:layout-examples`
   - `fig:episodes`
   - `fig:exploration-curve`
2. 实验表仍有大量 `TBD`：
   - dataset statistics
   - navigation results
   - object placement success / failure breakdown
   - runtime / cache reuse statistics
   - VLM success / heuristic fallback ratio
3. 目前只有 smoke 级别 benchmark 产物，缺少正式 train/val/test 统计。
4. 当前 PDF 里有图占位，但没有真实渲染图片、流程图、曲线图。
5. 方法部分还偏工程描述，需要进一步形式化：
   - object-room probability
   - receptacle candidate filtering
   - object-to-receptacle assignment score
   - lifespan state transition
   - episode sampling distribution
6. Related Work 已补齐基础引用，但还需要最后人工核验 BibTeX 元数据、页码和 venue。

工程/数据层面缺失：

1. Lifespan 分支当前是 `semantic_only=true`，`position/rotation=null`，尚未接入 3D grounding。
2. `benchmark/build_episodes.py` 当前主要从 grounded batch manifest 生成任务，不能直接消费 semantic-only lifespan manifest。
3. YCB/HSSD 可通过 semantic text 走开放词汇和语言目标，但 image-goal 仍依赖参考图片；当前若无图片，`image_goal` 任务会报错。
4. 当前 `object_catalog.json` 只统计到 YCB/HSSD，未看到 legacy 条目；需要确认 legacy 是否已迁移、是否故意排除，或 catalog 构建是否遗漏。
5. 当前没有真实 memory-aware agent baseline，只有 oracle/noop smoke agent。
6. 当前没有正式的大规模场景 split、layout plan 和结果 manifest。
7. `paper/` 被 `.gitignore` 忽略，论文 tex/bib/pdf 默认不会进入 Git 状态；需要决定是否继续忽略论文，或改为跟踪 `.tex/.bib`、忽略中间文件和 PDF。

## 2. 推荐完善目标

### 2.1 短期目标：形成可展示论文版本

目标是让论文具备完整故事线、真实图片、初步统计和 smoke 实验结果。

验收标准：

1. 所有 `Replace with ...` 图片占位替换为真实图或正式绘制图。
2. `TBD` 至少替换为 smoke/mini-benchmark 统计。
3. Experiments 至少包含：
   - generation statistics
   - placement success / failure reason
   - oracle/noop smoke validation
   - exploration curve 文件说明或初步曲线
4. PDF 可稳定编译，无 undefined citation。

### 2.2 中期目标：形成完整 benchmark paper

目标是支持论文中的核心 claim：机器人多次探索同一家庭场景后，可以学习物体分布规律。

验收标准：

1. 生成正式 train/val/test split。
2. 每个场景生成多个 grounded 3D layout。
3. 基于这些 layout 生成任务集。
4. 至少实现一个非 oracle 的 baseline：
   - reactive baseline：不使用历史 layout。
   - memory baseline：使用已见 layout 的 object-room/object-receptacle 统计。
5. 输出并绘制 `seen_layout_count_before -> SR/SPL` 曲线。
6. 论文中报告完整表格和曲线。

### 2.3 长期目标：Lifespan 语义轨迹接入 3D grounding

目标是让数据不只是独立随机 layout，而是由家庭居民、routine、事件和物体生命周期驱动的连续轨迹。

验收标准：

1. `snapshot_requests.json` 可以被转换为 grounded assignment plan。
2. unchanged objects 复用上一 snapshot pose。
3. changed objects 重新分配 room/receptacle/pose。
4. consumed/removed objects 从当前 layout 中移除。
5. `results/lifespan/<scene>/<sequence_id>/layouts/snapshot_*.json` 不再是 `position=null`，而是可被 Habitat-Sim 加载的 final layout。
6. `benchmark/build_episodes.py` 可读取 lifespan manifest，并避免采样 absent object。

## 3. 分阶段执行计划

### Phase 0：版本控制与论文文件管理

目的：避免论文修改“本地看到了但 Git 看不到”。

建议修改：

1. 修改 `.gitignore`，不要整体忽略 `paper/`。
2. 推荐保留：
   ```gitignore
   paper/**/*.aux
   paper/**/*.bbl
   paper/**/*.blg
   paper/**/*.fdb_latexmk
   paper/**/*.fls
   paper/**/*.log
   paper/**/*.out
   paper/**/*.synctex.gz
   paper/**/*.xdv
   ```
3. 根据需要决定是否跟踪 `paper/**/*.pdf`。
4. 至少跟踪：
   - `paper/ICLR_zhangwei/iclr2026_conference.tex`
   - `paper/ICLR_zhangwei/iclr2026_conference.bib`
   - `paper/ICLR_zhangwei/math_commands.tex`
   - `paper/ICLR_zhangwei/iclr2026_conference.sty`

输出：

1. Git 能显示论文源码变更。
2. 编译中间文件不污染 Git。

### Phase 1：生成论文图片

需要补齐 6 张图。

1. `fig:concept`：核心概念图
   - 内容：同一 HM3D home，多个 layout，机器人多次访问后更新记忆。
   - 形式：4 panel。
   - 数据来源：论文 schematic，可手工绘制。

2. `fig:pipeline`：双分支生成流程图
   - Branch A：grounded batch layout。
   - Branch B：lifespan semantic trajectory。
   - 数据来源：`doc/EXECUTABLE_TECH_SPEC.md` 和当前代码模块。

3. `fig:receptacle`：承载面提取图
   - 内容：room instances、filtered candidates、top surface point cloud、selected receptacle。
   - 数据来源：`results/receptacle_queries/<scene>/...` 和 `visualize_instance_pointcloud_viser.py`。

4. `fig:layout-examples`：同场景多 layout 示例
   - 内容：同一 scene 下 3 个 layout 渲染截图。
   - 数据来源：`core/visualize_placed_layout.py --headless` 或 GUI 截图。

5. `fig:episodes`：episode 构造图
   - 内容：batch manifest -> objects -> balanced subtasks -> episode JSON。
   - 数据来源：`benchmark/build_episodes.py`。

6. `fig:exploration-curve`：重复探索曲线
   - 内容：SR/SPL vs `seen_layout_count_before`。
   - 数据来源：`benchmark/eval/<version>/exploration_curve.json`。

输出：

1. `paper/ICLR_zhangwei/figures/fig_concept.pdf`
2. `paper/ICLR_zhangwei/figures/fig_pipeline.pdf`
3. `paper/ICLR_zhangwei/figures/fig_receptacle.pdf`
4. `paper/ICLR_zhangwei/figures/fig_layout_examples.pdf`
5. `paper/ICLR_zhangwei/figures/fig_episodes.pdf`
6. `paper/ICLR_zhangwei/figures/fig_exploration_curve.pdf`

### Phase 2：生成 mini-benchmark 统计

目的：先用小规模真实结果替换 `TBD`。

建议规模：

1. scenes：10 个。
2. layouts per scene：3 个。
3. episodes per layout：3 个。
4. subtasks per episode：5-10。
5. object datasets：先用 `ycb,hssd` 或确认 legacy 后使用 `legacy,ycb,hssd`。

需要统计：

1. 场景数。
2. layout 数。
3. episode 数。
4. object catalog 数量。
5. 每个 layout 平均物体数。
6. placement success rate。
7. failed reason 分布。
8. 平均生成时间。
9. cache reuse 情况。
10. LLM 成功率与 heuristic fallback 比例。

输出：

1. `paper/ICLR_zhangwei/tables/dataset_stats.json`
2. `paper/ICLR_zhangwei/tables/generation_stats.json`
3. `paper/ICLR_zhangwei/tables/placement_failure_breakdown.json`

### Phase 3：补齐 benchmark baseline

目的：让实验不只有 oracle/noop。

建议实现两个轻量 baseline：

1. Reactive baseline
   - 不使用历史 layout。
   - 根据当前 task prompt 或目标类别，在场景中均匀或启发式搜索。
   - 可先用模拟轨迹/近似路径长度实现 smoke 版。

2. Memory baseline
   - 从已见 layout 中统计 object -> room/receptacle 分布。
   - 对当前目标优先导航到历史高概率区域。
   - 输出 `memory_metrics`，用于 `dynamic_memory_accuracy` 和 `fixed_memory_accuracy`。

输出：

1. `benchmark/baselines/reactive_agent.py`
2. `benchmark/baselines/memory_agent.py`
3. `benchmark/eval/<version>/reactive.jsonl`
4. `benchmark/eval/<version>/memory.jsonl`
5. `benchmark/eval/<version>/summary_*.json`
6. `benchmark/eval/<version>/exploration_curve.json`

### Phase 4：让 Lifespan 接入 3D grounding

目的：让论文核心名称 Lifespan-Bench 与最终可执行数据完全一致。

需要实现：

1. 新增或完善 `core/lifespan_grounding.py`。
2. 读取：
   - `snapshot_requests.json`
   - previous grounded layout
   - surfaces json
   - object catalog
   - object profiles
3. 对每个 snapshot：
   - unchanged objects：复用上一 snapshot pose。
   - moved objects：调用 assignment + placement。
   - consumed/removed objects：从 objects 列表移除。
   - introduced/replenished objects：创建新 instance 并放置。
4. 输出 grounded lifespan layout：
   - `schema_version=lifespan_grounded_layout.v1`
   - `semantic_only=false`
   - `position/rotation` 均有效。
5. 更新 `benchmark/build_episodes.py`：
   - 支持 lifespan manifest。
   - 按 snapshot_index/state_index 设置 `seen_layout_count_before`。
   - 避免采样 `exists=false` 或 absent object。

输出：

1. `results/lifespan/<scene>/<sequence_id>/grounded_layouts/snapshot_*.json`
2. `results/lifespan/<scene>/<sequence_id>/grounded_manifest.json`
3. `benchmark/episodes/lifespan_v1/...`

### Phase 5：完善论文方法章节

目的：从“工程说明”提升到“论文方法”。

建议新增或强化以下公式：

1. Object-room prior：
   ```text
   P(r | o, S)
   ```
2. Receptacle candidate score：
   ```text
   score(c | o, r, S) = semantic_affordance + geometry_fit + surface_quality
   ```
3. Layout sampling：
   ```text
   L_S^{(k)} ~ P(L | S, O, seed_k)
   ```
4. Lifespan state transition：
   ```text
   X_{t+1} = T(X_t, E_t, H_S)
   ```
5. Episode sampling：
   ```text
   e ~ P(e | L_S^{(k)}, task_mix)
   ```
6. Exploration gain：
   ```text
   SRGain_k, SPLGain_k
   ```

输出：

1. 更正式的 Method section。
2. Algorithm 1：Grounded Batch Layout Generation。
3. Algorithm 2：Lifespan Semantic State Propagation。
4. Algorithm 3：Episode Generation and Evaluation。

### Phase 6：补齐正式实验表格

论文至少需要以下表格：

1. Dataset statistics
   - scenes
   - layouts
   - episodes
   - object assets
   - avg objects per layout

2. Generation quality
   - placed objects
   - failed objects
   - placement success rate
   - failure reason breakdown
   - runtime

3. Navigation results
   - noop
   - oracle
   - reactive baseline
   - memory baseline
   - SR
   - SPL
   - avg steps
   - avg path length

4. Repeated-exploration gain
   - seen layout count
   - SR
   - SPL
   - delta
   - gain percentage

5. Ablation
   - no VLM
   - no object profile
   - no receptacle geometry filtering
   - no placement retry
   - no memory

### Phase 7：最终论文检查

最终提交前检查：

1. 全文无 `TBD`。
2. 全文无 `Replace with ...`。
3. 所有 figure 都有真实文件和清晰 caption。
4. 所有 table 数值来自可复现 JSON。
5. 所有 citation 都能解析。
6. `xelatex -> bibtex -> xelatex -> xelatex` 通过。
7. Appendix 命令与 `doc/DATASET_TASK_GENERATION_COMMANDS.md` 一致。
8. `paper/` 的 Git 忽略策略已处理。

## 4. 当前最高优先级建议

推荐按下面顺序推进：

1. 先处理 `.gitignore`，让论文源码能被版本管理。
2. 确认 legacy object 是否需要加入 `object_catalog.json`。
3. 跑一个 10 scene x 3 layout 的 mini-benchmark，得到真实统计。
4. 用当前数据生成 3 张最关键图：
   - pipeline
   - layout examples
   - exploration curve
5. 实现 memory baseline 的最小版本。
6. 将 Lifespan semantic snapshots 接入 3D grounding。
7. 用 grounded lifespan 数据替换独立 batch layout 作为主实验。

## 5. 需要你确认的决策

1. 论文是否需要纳入 Git 跟踪？
   - 推荐：跟踪 `.tex/.bib/.sty`，忽略中间文件，PDF 可选。

2. 正式论文主线先采用哪种数据？
   - 保守方案：先用 grounded batch layout 作为主实验，Lifespan 作为未来/扩展。
   - 进取方案：先实现 Lifespan 3D grounding，再把 Lifespan trajectory 作为主实验。

3. image-goal 是否必须覆盖 YCB/HSSD？
   - 如果必须，需要批量渲染 object preview。
   - 如果不必须，YCB/HSSD 可以只进入 open-vocabulary 和 language-goal。

4. 论文实验规模先定多少？
   - 推荐 mini 版：10 scenes x 3 layouts x 3 episodes/layout。
   - 推荐正式版：50+ scenes x 10+ layouts x 5+ episodes/layout。

