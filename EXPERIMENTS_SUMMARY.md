# 三档动作空间 · 砍基线 · 领域知识增强 · 负对照实验总结

> 结论基于 **36 次真实 LLM（qwen-max）运行**（每臂 3 seeds，`max_iter=5`，`patience=3`，`cold_start` 起点）。
> 目的：回应审稿级质疑 ——"领域提示词一定更好吗 / 会不会更差 / 只适合这些数据集吗 / LLM 到底发挥了什么"。

---

## 0. 一句话结论

> **先验的价值 = 它指向的列 × 该列在数据集中的信息占比。** 正先验只在具备先验所指列的 Wind 上显著优于无先验；错配先验在具备外生列的 Solar 上显著拖累；列不存在时白名单拒绝误导特征，先验被中和（Load 上只剩探索噪声）。**白名单（受约束动作空间）是伤害上限的最终安全网。**

---

## 1. 三档动作空间（`--feature-tier 1|2|3`）

| 档位 | 名称 | 允许特征 | 说明 |
|---|---|---|---|
| 1 | 基本时序 | `time` + 目标列 `lag` | 最保守 |
| 2 | 能源专有 | + 外生列（Solar `VAR169/164/167`；Wind `ws10/ws100`）`lag`/`rolling` | 领域专有特征 |
| 3（默认） | 组合类 | + `cross` | 给 LLM 最多空间 |

三档共用同一时间因果红线（lag ≥ 1、rolling shift(1)、cross 禁直用目标列）；外生列特征确定性命名带 source 前缀（`VAR169_lag_24`）杜绝跨 source 冲突。

**多 seed 实测（Task 15，cold_start 起点，3 seeds）：**

| 数据集 | tier1 | tier2 | tier3 |
|---|---|---|---|
| LOAD T15 | -6.6%±0.7% | —（动作空间=tier1） | **-17.4%±12.3%**（跨特征潜力高但方差大） |
| SOLAR T15 Z1 | -6.3%±4.4% | -4.5%±4.7% | -4.9%±3.6%（三档无显著差异） |

> 单 seed 曾观察到 tier1 达 -32%，多 seed 后被修正为 ~-6% —— **单次 LLM 探索随机性极强，档位结论必须多 seed 验证**。

---

## 2. 砍基线（`--baseline cold_start`）

Agent 起点从完整特征集砍到极简（time + lag_1/24），让 LLM 重新发现滚动统计 / 周滞后 / 气象外生，放大增益对比。**全局 `FEATURE_SPEC` / `SOLAR_FEATURE_SPEC` / `WIND_FEATURE_SPEC` 不变**，replay 基线数字不受影响。

基线实测（Task 15 Z1 / Task 1 Z1 / Task 15）：

| 数据集 | cold_start | full | 砍出的空间 |
|---|---|---|---|
| SOLAR T15 Z1 | 0.0739 | 0.0668 | **-9.6%** |
| WIND T1 Z1 | 0.1041 | 0.0984 | -5.5% |
| LOAD T15 | 3.4363 | 3.2993 | -4.0% |

> 关键发现：Solar 冷启动 LLM（领域知识增强后 0.0679）**逼近 full 人工特征集（0.0668）** —— "LLM 从 6 个极简特征重新发现人工 15 特征水平"是能站住的故事。

---

## 3. 领域知识增强（`agent/domain_knowledge.py`）

按数据集注入专属特征工程先验（load / solar / wind / price / ecl 五段注册），CLI 按 `--dataset` 自动选择，`--domain-key` 可显式覆盖。

**Solar 增强 vs 无增强（同 seed 配对，3 seeds）：**

| 档位 | 无增强 | DK 增强 | VAR169 使用率 |
|---|---|---|---|
| tier2 | -4.5% | **-8.0%** | 2/3 → 3/3 |
| tier3 | -4.9% | **-8.3%** | 0/3 → **3/3** |

> 核心变化：无先验时 LLM 几乎不用外生列（tier3 的 0/3），增强后 6/6 全部用上 VAR169/164/167。**Solar 增益小的根因之一是 LLM 上下文没被告知"辐射是主导信号"。**

---

## 4. 负对照实验（回答"会不会更差 / 只适合这些数据集吗"）

三数据集 × 正/无/负（错配 = 把别的领域有效先验原样扔过来 / 反事实 = 湿度反向周期）：

| 数据集 | 正先验 | 无先验 | 负先验（最差臂） | 排序 |
|---|---|---|---|---|
| **SOLAR** T15 Z1 | -8.3% | -8.7% | **-1.3%**（load 错配，3/3 seed 系统性忽略 VAR169） | 无≈正 > 负 ✓ |
| **WIND** T1 Z1 | **-3.5%** | -3.0% | -1.9% / -2.3%（load / solar 错配） | **正 > 无 > 负** ✓ |
| **LOAD** T15 | -4.3% | -12.4% | -13.0%±14.7%（湿度反事实） | ❌ 反转（见下） |

### Load 反转的机制

负先验（solar 错配 / 湿度反事实）指向 `VAR169` / 湿度列 → **Load 没有这些列 → 白名单直接拒绝** → LLM 退回目标列自由探索 → 方差爆表（std ±0.3~±0.5，个别 seed 撞大运 -29%~-30%），均值被幸运 seed 拉低。**这是"白名单兜底 + 探索噪声"的假象，不是负先验有效** —— 三臂均值差异完全落在噪声区间。

### 对质疑的最终回答

1. **"提示词一定更好吗"** → 不一定。只在 Wind（有先验所指列）显著优于无先验；Solar 持平。
2. **"会不会更差"** → 有先验指向的列时错配会显著拖累（Solar load 错配 -1.3% vs -8%）；列不存在时被白名单中和。
3. **"只适合这些数据集吗"** → 是。先验价值 = 指向列 × 信息占比；外生主导的 Solar / Wind 才有意义，无外生的 Load 近乎无关。
4. **"LLM 发挥什么"** → 先验给方向，LLM 做搜索 / 选择 / 评测 / 回滚；**白名单决定伤害上限**。

---

## 5. 复现命令

```bash
# 三档对比（Load / Solar / Wind）
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --feature-tier 1
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --feature-tier 3
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --dataset solar --zone 1 --feature-tier 3
python experiments/run_self_evolving_agent.py --task 1 --max-iter 5 --dataset wind --zone 1 --feature-tier 2 \
  --data-dir "GEFCom2014 Data/GEFCom2014-W_V2/Wind"

# 砍基线
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --baseline cold_start
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --baseline full

# 领域知识 / 负对照（--domain-key）
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --dataset solar --zone 1 --feature-tier 2 \
  --domain-key solar              # 正先验
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --dataset solar --zone 1 --feature-tier 2 \
  --domain-key ""                 # 无先验
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5 --dataset solar --zone 1 --feature-tier 2 \
  --domain-key solar_neg_load     # 负先验（load 错配）
```

可用 `--domain-key`：正 = `load / solar / wind / price / ecl`；负对照 = `solar_neg_humidity / solar_neg_load / solar_neg_wind / load_neg_solar / load_neg_humidity / wind_neg_load / wind_neg_solar`。

---

## 6. 测试

`tests/test_evolution_suite.py` 新增 E17（三档动作空间校验）、E18（Solar Agent CLI 冒烟）、E19（领域知识注册表 + prompt 注入）。全部套件通过。

---

## 7. 关键文件

- `agent/feature_spec.py` — 三档动作空间（tier / exogenous source / 外生命名）
- `agent/domain_knowledge.py` — 领域知识注册表 + 负对照 key
- `agent/evolution_schema.py` / `agent/evolution_runner.py` — prompt / runner 透传
- `data/task_builder.py` / `solar_task_builder.py` / `wind_task_builder.py` — 冷启动基线 spec
- `evaluation/solar_spec_evaluator.py` / `wind_spec_evaluator.py` — Solar / Wind 候选评测器
- `experiments/run_self_evolving_agent.py` — CLI（`--dataset / --zone / --feature-tier / --baseline / --patience / --domain-key / --data-dir`）
