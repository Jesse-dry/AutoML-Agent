# 主线整合交接：以你的 main 为基准的合并说明

> 给 miao-419。这份说明讲清楚：我把你推在 `origin/main` 的改动合并进
> `integration/v2.2` 后做了什么、当前状态、以及后续分工建议。

## 一、发生了什么

你在 `origin/main` 推了两个 commit：

- `958c0b2` — ECL 协议回归测试 E7-E10 + `ECL_TEAMMATE_TASK.md`
- `ad25c7c` — 三档动作空间 + 领域知识增强 + Solar/Wind 接入 + 36 次负对照实验总结

我（Jesse-dry）以「**你的方案为主**」把它们合并进 `integration/v2.2`（merge-base 是 `22d3fc2`）：

- 冲突文件**全部取你的版本**：`agent/feature_spec.py`、`agent/evolution_schema.py`、
  `agent/evolution_runner.py`、`experiments/run_self_evolving_agent.py`、
  `tests/test_evolution_suite.py`、`evaluation/spec_evaluator.py`。
- 我独有、你 main 上没有的子系统，**改接到你的架构上**（不是丢弃、不是并存两套）。

## 二、关键合并决策

1. **自进化 Agent 统一走你的架构**：`--dataset` + `--feature-tier` +
   `domain_knowledge.py` + 独立 spec evaluator（`solar_spec_evaluator.py` /
   `wind_spec_evaluator.py`）。
2. **`energy_registry` 只留给外循环**：`run_outer_loop.py` / `strategy_migration.py`
   继续用 `get_energy()` 解析赛道资源；自进化 Agent 不再走 `energy_registry`。
3. **我独有、改接到你架构的东西**：
   - **price 赛道** → 新增 `evaluation/price_spec_evaluator.py` + `--dataset price`
     分支 + `_build_price_scenario`（你的 `domain_knowledge.py` 里已含 price 段，直接复用）。
   - **LSTM 后端** → `models/replay_backends.py` 的 `LSTMBackend`（你 main 没动这个文件，
     merge 自动保留，无需回灌）。
   - **模型选择 `candidate.model`** → 回灌到 `evolution_schema.py`（`MODEL_CHOICES` +
     解析 + prompt）+ `evolution_runner.py`（`make_backend` + spec/model 状态同步）。
   - **`current` 特征类型 + cross 别名** → 回灌到 `feature_spec.py`（`current` 类型走你的
     `_check_source`/`exogenous_cols` 白名单；`_OP_ALIASES` 容错 LLM 常用 cross 缩写）。
   - **外循环** → `run_outer_loop.py` 把 `EvolutionRunner(energy=...)` 改成显式参数
     （`feature_tier`/`target_col`/`exogenous_cols`/`zone`/`scenario_builder`/
     `domain_knowledge`），`energy_registry.spec_evaluator` 直连新签名 evaluator。

## 三、当前状态

- 分支 `integration/v2.2`，commit 链（旧→新）：

  ```
  c43c7a3 feat(v4): 风险感知不确定性 Agent 切片
  1c9bc25 feat(ecl): Exp4 记忆对照实验入口
  d9bd1f1 feat(agent): 外循环 solar/price 接入 + ROADMAP 更新
  180aafc chore: gitignore 队友协作/请教文档
  64191e0 merge: 整合队友 main（以队友为主）
  864a285 feat(agent): 独有子系统改接到队友架构
  ```

- **测试全绿**：`evolution_suite`(E16-E19)、`lstm_backend`、`price_suite`、`wind_suite`、
  `solar_suite`、`evaluation_suite`、`uncertainty_agent`、`ecl_memory`。
- dry-run 冒烟通过：self-evolving `load`/`solar`/`price`、outer loop `load`/`price`
  （含跨 Task warm-start 迁移）。
- **尚未 push**：GitHub 网络不稳，port 443 时通时断。

## 四、后续怎么做

### 1. 同步（先解决双向分叉）

网络恢复后，建议按下面顺序合并，避免我们再在 `main` 和 `integration/v2.2` 双向分叉：

1. 我把 `integration/v2.2` 推上远程（或你直接拉 `integration/v2.2`）。
2. 把 `integration/v2.2` 合并回 `main`（`git merge integration/v2.2`）。
3. 你继续开发时，基于**合并后的 main** 新开 feature 分支，不再直接在 `main` 上堆 commit。

### 2. 我这边留的 TODO（需要补）

- `tests/test_multi_energy_outer_loop.py`：被删了（依赖旧 `--energy` 方案）。需按新架构
  重写外循环 solar/price 的测试（M1-M5 的替代）。
- `README.md`：两边各自加了内容（你的三档/负对照，我的 solar/ECL 接入），merge 自动合并了，
  但需要人工核对一遍有没有重复或冲突段落。
- **ECL E7-E10 测试**：本地没有 `ECL/electricity.txt`，E1-E6、E7-E10 里需要数据/回放的
  部分跑不了；有数据的机器上跑 `python tests/test_ecl_suite.py` 确认 E1-E10 全过。

### 3. 建议的协作约定

- 改动核心共享文件（`feature_spec.py` / `evolution_runner.py` / `evolution_schema.py` /
  `spec_evaluator.py`）前，先同步一下方向，避免同一文件两套语义再撞一次。
- feature 分支合入 main 前先 `git pull --rebase`；跨分支大改（像这次的「动作空间」vs
  「注册表」）先在小群里对齐方案再动手。

---

如有疑问，回传格式照旧：分支 / commit / 本地环境 / 运行命令 / 测试结果 / 产物目录 /
未完成事项。
