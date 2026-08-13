# AutoML-Agent

LLM 驱动的 AutoML Agent —— 面向**多能源时序预测**（负荷 / 风电 / 电价）的**自动化特征工程**、**自进化特征优化**、**无泄漏闭环评测**与**决策价值评估**系统。

**核心创新**：大语言模型（LLM）作为**预测建模决策器**，在**受约束动作空间**（ADD / REMOVE / REPLACE / KEEP / ROLLBACK / STOP 六类动作，特征仅允许 lag / rolling / time / cross 四类确定性操作，参数受时间因果性与安全上限约束）内**自主设计与删改特征**。每轮 Agent 在**误差画像**（分段 RMSE + bias）驱动下提出 **3 个不同假设的候选方案**，逐一评测后**择优 / 自动回滚**，并把每次实验写入**经验记忆**（跨 Task 共享，场景相似度检索复用）。所有评测跑在**无泄漏滚动回放**（预测月 online_h1）之上，保证结果可信。LLM 不写代码——基于数据统计、自相关分析（ACF）、特征重要性、误差画像和历史经验，**自主决定**生成 / 删除什么特征；确定性执行引擎负责**安全执行**。两者构成完整的 **感知→决策→执行→评估→反馈→记忆** 闭环。最终，预测不只在 RMSE / CRPS 等精度指标上被评估，还会被接入**储能套利决策评估器**，把预测误差货币化为**套利利润 / Regret**（业务/决策驱动，详见 ROADMAP V2.1）。

**外循环（P1-B）**：把自进化 Agent 串联到 Task 1–15 滚动评测上，形成**滚动自适应进化双闭环智能体**——每完成一个 Task 保存**最佳策略 + 经验**，进入下一个 Task 前用**确定性漂移检测**（`evaluation/drift_detector.py`：均值 / 方差 / 分位 / ACF 周期 / 残余误差五类信号）度量 Task 间变化，再由 LLM 决定**继承 / 修改 / 重置**策略（`agent/strategy_migration.py`），以**warm-start**（`EvolutionRunner(init_spec=…)`）+ **自适应迭代预算**开始新一轮内循环。内循环负责"单 Task 自进化"，外循环负责"Task 间自适应迁移"。

数据集：
- [GEFCom2014-L_V2](https://www.sciencedirect.com/journal/international-journal-of-forecasting/vol/30/issue/2)（负荷赛道，15 任务）—— 主论文核心
- GEFCom2014-W_v2（风电赛道，15 Task × 10 Zone）—— ✅ 已接入（无泄漏滚动回放，LightGBM 基线 Mean RMSE 0.0998）
- GEFCom2014-P_V2（电价赛道，15 任务，单分区）—— **决策效能评估主线（价值出口）**：储能套利评估器把预测误差货币化为利润 / Regret（P-Value）

> **演进方向**：从「负荷预测」扩展为「多能源自主智能体」，最终以**决策价值**为价值出口 ——
> 预测不止于「准」，更在于「有用」。详见 [ROADMAP.md](ROADMAP.md)（V2.1 三层架构）。

---

## 整体架构

```
               GEFCom2014-L_V2 数据集
                       │
                       ▼
        data/availability.py + task_builder.py
      (逐 Task 可用历史拼接 → 血缘式特征构建 → 预测月窗口)
                       │
                       ▼
                    GEFComTask
       (history / train / val早停 / forecast_ts / y_true)
                       │
                       ▼
        evaluation/rolling_backtest.py（无泄漏滚动回放）
        online_h1 / recursive_month_ahead 双协议
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                  ▼
  LightGBM           LSTM            PatchTST
  (回放后端)      (可接入)         (可接入)
     │                 │                  │
     └────────┬────────┘
              │
              ▼
 ┌────────────────────────────────────────────┐
 │   自进化特征 Agent (evolution_runner.py)     │
 │   —— LLM 决策器，绝不写代码 ——               │
 │                                            │
 │  预测 → 误差画像(error_profiler)            │
 │       → 检索经验记忆(memory)                │
 │       → LLM 提出 3 个候选假设               │
 │       → 逐候选 apply_actions → 静态泄漏检查  │
 │       → 逐候选评测（预测月 online_h1 RMSE） │
 │       → Selector 择优 / 自动回滚            │
 │       → 实验写入记忆 → 下一轮               │
 │                                            │
 │  动作空间: ADD / REMOVE / REPLACE /         │
 │           KEEP / ROLLBACK / STOP            │
 └────────────────────────────────────────────┘
              │
              ▼
     实验产出 (experiments/output/evolution_task{id}/)
     summary.json / iteration_history.csv / best_features.txt
     error_profile.txt / run_manifest.json
     + memory/experiment_memory.jsonl（跨 Task 经验库）
              │
              ▼
 ┌──────────────────────────────────────────────┐
 │  外循环（P1-B）：滚动自适应进化双闭环           │
 │   Task k 完成 → 保存 Best Strategy + 经验      │
 │   → Task k+1 漂移检测（drift_detector.py）     │
 │     均值/方差/分位/ACF/残余误差 五类信号        │
 │   → LLM 策略迁移（strategy_migration.py）      │
 │     继承(low) / 修改(medium) / 重置(high)      │
 │   → warm-start EvolutionRunner(init_spec)     │
 │     + 自适应 max_iter                          │
 │   → record_strategy → Task k+2 …              │
 └──────────────────────────────────────────────┘
             │
             ▼
     外循环审计 (experiments/output/outer_loop/)
     task_{id:02d}/{drift_report, decision, strategy}.json
     outer_loop_summary.csv（drift / policy / transfer_gap）
     + memory/strategies.jsonl（每 Task 最佳策略库）
```

> **价值出口层（V2.1，业务/决策驱动）**：预测不只以精度评估，还会被接入**储能套利决策评估器**
> —— 把 Market/Price Agent 的电价预测喂给带约束的虚拟储能 LP，按预测决策、按真实结算，
> 产出**套利利润 / Regret**（预测误差的货币化）。三层架构 + P-Value 里程碑详见 [ROADMAP.md](ROADMAP.md)。

---

## 项目结构

```
AutoML-Agent/
├── data/
│   ├── preprocessing.py                 # 数据预处理流水线（时间戳消歧 / 填充 / 切分 / 基线特征）[legacy]
│   ├── gefcom_loader.py                 # GEFCom 统一加载器（train/benchmark/solution + 真值解析）
│   ├── availability.py                  # 每个 Task 的「可用历史 + 预测区间」定义
│   ├── task_builder.py                  # 血缘式特征规格 + 严格过去向特征构造（lag/rolling/time/cross）+ GEFComTask
│   ├── wind_loader.py                   # ★Wind 加载器（15 Task × 10 Zone：train/expvars/benchmark/solution + 真值解析）
│   ├── wind_task_builder.py             # ★Wind 特征规格（气象外生 U/V→风速/风向）+ WindTask 构建
│   └── price_loader.py                  # (TODO) Price 电价加载器（P-Value：决策效能主线）
│
├── evaluation/                          # 无泄漏滚动回放评测体系
│   ├── forecast_protocol.py             # 评测协议（online_h1 / recursive_month_ahead）
│   ├── leakage_checker.py               # 严格值级泄漏检查（Pass A 血缘 / Pass B 重算 / Pass C 别名）
│   ├── rolling_backtest.py              # 逐小时滚动回测（按协议回填，_features_at 与 build_features 逐位一致）
│   ├── evaluator.py                     # 指标计算 + 多 Task 汇总（复用 utils/metrics）
│   ├── error_profiler.py                # 误差画像（时段/负荷状态/变化状态分段 + bias + top-worst）
│   ├── spec_evaluator.py                # 候选特征集评测器（decision metric = 预测月 online_h1）
│   ├── drift_detector.py                # ★跨 Task 漂移检测（尾部窗口均值/方差/分位/ACF/残余误差 → score/level）
│   ├── task_replay.py                   # Task 1–15 回放主循环 + 审计输出（predictions/run_manifest）
│   ├── wind_replay.py                   # ★Wind 回放主循环（15 Task × 10 Zone 逐分区独立模型 + 气象外生特征）
│   └── arbitrage_evaluator.py           # (TODO) 储能套利评估器（LP + Regret，P-Value 价值出口）
│
├── models/
│   ├── baseline/
│   │   └── lgb_gefcom2014.py            # LightGBM 基线（legacy 协议）
│   ├── replay_backends.py               # 回放模型后端（LightGBM + 特征重要性 / Naive / Persistence）
│   ├── LSTM/
│   │   └── LSTM_baseline.py             # LSTM 基线（滑动窗口 + 归一化 + 早停）
│   ├── PatchTST/
│   │   ├── patch_tst_baseline.py        # PatchTST 训练脚本 (ICLR 2023)
│   │   ├── PatchTST_backbone.py         # Patching + Channel-Independent Transformer
│   │   ├── PatchTST_layers.py           # 自定义 Transformer 层
│   │   └── RevIN.py                     # 可逆实例归一化
│   └── Transformer/                     # (TODO) Transformer 模型
│
├── utils/
│   ├── metrics.py                       # 通用评估指标（RMSE/MAE/MAPE/SMAPE/R²）
│   └── data_loader.py                   # 滑动窗口 DataLoader（StandardScaler + shuffle）
│
├── agent/                               # 自进化 Agent 层
│   ├── feature_spec.py                  # ★血缘式 spec 工具 + 动作解释器（name/normalize/validate/apply_actions）
│   ├── evolution_runner.py              # ★自进化闭环状态机（多候选 → Selector 择优/自动回滚 → 记忆）
│   ├── evolution_schema.py              # ★LLM 输出 v2 schema 解析 + Prompt 构建
│   ├── strategy_migration.py            # ★跨 Task 策略迁移（LLM 决策 继承/修改/重置 + 确定性兜底）
│   ├── scripted_llm.py                  # ★确定性 LLM（测试 / --dry-run）
│   ├── feature_engine.py                # 确定性特征执行引擎（lag/rolling/time/cross）[legacy]
│   ├── feature_agent.py                 # LLM Agent 协议 + 闭环迭代调度器（v1，仅 ADD）[legacy]
│   ├── tuning_agent.py                  # (TODO) 超参调优 Agent
│   ├── report_agent.py                  # (TODO) 报告生成 Agent
│   └── market_agent.py                  # (TODO) 电价 Market/Price Agent（尖峰双层 + 归因，P-Value）
│
├── memory/
│   └── memory_manager.py                # ★经验记忆（experiment_memory.jsonl 轮级经验 + strategies.jsonl 策略级记忆，场景相似度检索跨 Task 共享）
│
├── experiments/
│   ├── run_self_evolving_agent.py       # ★自进化 Agent 实验（CLI：--task / --max-iter / --dry-run / --n-candidates）
│   ├── run_outer_loop.py                # ★外循环（P1-B）：逐 Task 漂移检测 → 策略迁移 → warm-start 自进化（CLI）
│   ├── run_task_replay.py               # Task 1–15 无泄漏滚动回放评测（CLI）
│   ├── run_wind_replay.py               # ★Wind Task 1–15 × Zone 1–10 无泄漏滚动回放评测（CLI）
│   ├── run_price_replay.py              # (TODO) Price Task 1–15 无泄漏滚动回放评测（CLI，P-Value）
│   ├── run_feature_agent.py             # v1 特征工程 Agent 实验（legacy）
│   ├── feature_agent_task15/            # Task 15 v1 实验输出
│   └── output/                          # 评测输出（predictions / manifest，gitignored）
│
├── tests/
│   ├── test_evaluation_suite.py         # 评测体系测试（T1–T6）
│   └── test_evolution_suite.py          # ★自进化 Agent 测试（E1–E16，E6 复现回滚 + E14 漂移 + E15 迁移 + E16 外循环）
│
└── GEFCom2014-L_V2/                     # 数据集 (gitignored)
```

---

## 环境配置

### 依赖

- Python 3.12+
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `torch` (PyTorch)
- `matplotlib`
- `requests`

```bash
pip install pandas numpy scikit-learn lightgbm torch matplotlib requests
```

### LLM API 配置

项目使用阿里云 DashScope 大模型 API（Qwen 系列）。在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-ws-your-api-key-here
```

> 使用 OpenAI 兼容接口。如需切换其他 LLM 提供商，修改 `agent/feature_agent.py` 中 `QwenClient` 的 `base_url` 和 `model` 参数即可。

---

## 快速开始

### 1. 数据预处理

```python
from data.preprocessing import preprocess_pipeline

result = preprocess_pipeline(
    data_dir="GEFCom2014-L_V2/Load",
    task_id=15,
    fill_load="interpolate",
    split_method="sequential",
    dropna_features=True,
)

train_df, val_df, test_df = result["train"], result["val"], result["test"]
feature_cols, target_col = result["feature_cols"], result["target_col"]
```

### 2. 训练基线模型

**LightGBM：**

```bash
python models/baseline/lgb_gefcom2014.py --task 15
```

产出：
- `lgb_baseline_task15.txt` — 训练好的模型
- `lgb_baseline_task15_metrics.json` — train/val/test 三阶段指标
- `lgb_baseline_task15_feature_importance.csv` — 特征重要性（供 LLM Agent 使用）
- `lgb_baseline_task15_predictions.csv` — 测试集预测结果

**LSTM：**

```bash
python models/LSTM/LSTM_baseline.py --task 15 --max-epochs 200 --patience 20
```

**PatchTST：**

```bash
python models/PatchTST/patch_tst_baseline.py --task 15 --max-epochs 200 --patience 20
```

> 所有模型输出统一格式的 `metrics.json`（由 `utils/metrics.py` 的 `compute_all_metrics()` 保证），可直接横向对比。

### 3. Task 1–15 无泄漏滚动回放评测

```bash
# 全量 1:15 回放（LightGBM / 基线 / persistence）
python experiments/run_task_replay.py --tasks 1:15 --model lightgbm --protocol online_h1
python experiments/run_task_replay.py --tasks 1:15 --model seasonal_naive_all
python experiments/run_task_replay.py --tasks 1:15 --model persistence
# 指定任务 / 快速泄漏检查
python experiments/run_task_replay.py --tasks 1:3 --model lightgbm --leak-check fast
```

产出（`experiments/output/task_replay/`）：
- 逐 Task RMSE 表 + **Mean / Std / Worst RMSE**
- `predictions/task_{01..15}.csv` — 逐小时 `y_true / y_pred / error`
- `run_manifest.json` — 协议 / 特征血缘哈希 / seed / git_commit 审计

> **协议说明**：`online_h1`（operational one-hour-ahead）预测 t 时只用 ≤t-1 的真实负荷，
> 适用于短期滚动预测 Agent，**不复现 GEFCom 官方 month-ahead 信息条件**；
> `recursive_month_ahead` 为 month-ahead 近似，预测月内只用预测值回填。
> 二者共享同一无泄漏特征工程（lag/rolling 严格过去窗口，训练/预测特征空间一致）。

### 3.1 Wind 风电赛道滚动回放（P0）

风电赛道（`GEFCom2014-W_v2/`）：15 Task × 10 Zone，预测月逐月推进（Task1=2012-10 … Task15=2013-12）。
目标为归一化出力 `TARGETVAR` [0,1]，特征含**气象外生**（U10/V10/U100/V100 → 风速/风向/切变），
预测月内气象列取 `TaskExpVars` 预报（决策时点可得，非泄漏）。逐分区独立模型，Task 得分 = 10 分区指标均值。

```bash
# 全量 1:15 × 10 Zone 回放（LightGBM / persistence / seasonal naive）
python experiments/run_wind_replay.py --tasks 1:15 --model lightgbm
python experiments/run_wind_replay.py --tasks 1:15 --model persistence
# 指定分区 / 任务 / 快速泄漏检查
python experiments/run_wind_replay.py --tasks 1:3 --zones 1:3 --model lightgbm --leak-check fast
```

产出（`experiments/output/wind_replay/`）：
- 逐 Task（Zone 均值）RMSE 表 + **Mean / Std / Worst**；`detail_summary_*.csv` 逐 Task×Zone 明细
- `predictions/task_{01..15}_zone_{01..10}.csv` — 逐小时 `y_true / y_pred / error`
- `run_manifest.json` — 数据集 / zones / 特征血缘哈希 / seed / git_commit 审计

> **Wind 基线**（LightGBM，online_h1）：Mean RMSE **0.0998**（归一化 [0,1] 量纲），Mean R² 0.873，
> 优于 persistence（对比见下文），验证风电天气驱动场景下气象外生特征有效。

### 4. 运行自进化 Agent（P1-A）

```bash
# 真实 LLM 调用（多候选 + 误差画像 + 经验记忆）
python experiments/run_self_evolving_agent.py --task 15 --max-iter 5

# 测试模式（ScriptedLLM 确定性演示，不调用 LLM；persistence 后端最快）
python experiments/run_self_evolving_agent.py --task 1 --max-iter 3 --dry-run --model persistence
```

产出（`experiments/output/evolution_task{id}/`）：
- `summary.json` — baseline → best RMSE 对比 + 每轮 outcome（improved / rolled_back / stopped）
- `iteration_history.csv` — 每轮 best_rmse / delta / outcome
- `best_features.txt` — 最优特征集；`error_profile_best_*.txt` — 最优特征集的误差画像
- `run_manifest.json` — task / 协议 / 候选数 / seed / git_commit 审计
- `memory/experiment_memory.jsonl` — 每轮实验经验（跨 Task 共享，下次运行自动检索复用）

> **自进化机制**：每轮 LLM 提出 ≤`--n-candidates`（默认 3）个**假设不同**的候选，逐个在
> 预测月 online_h1 滚动评测；若最优候选仍劣于当前 best，Selector **自动回滚**到 best 特征集继续。
> 即"第 N 轮退化 → 回滚 → 第 N+1 轮从 best 改善"是显式机制而非偶然
> （由 `tests/test_evolution_suite.py` E6 确定性复现）。

### 5. 运行外循环（P1-B，跨 Task 漂移检测 + 策略迁移）

```bash
# 全量 1:15 外循环（真实 LLM 迁移决策 + 进化闭环）
python experiments/run_outer_loop.py --tasks 1:15 --model lightgbm

# 测试模式（迁移走确定性映射，进化用 ScriptedLLM；persistence 最快）
python experiments/run_outer_loop.py --tasks 1:3 --model persistence --dry-run

# 审计参考基线（每 Task 多算一次 FEATURE_SPEC RMSE 做三向对比）
python experiments/run_outer_loop.py --tasks 1:15 --model lightgbm --with-reference-baseline
```

产出（`experiments/output/outer_loop/`）：
- `task_{id:02d}/{drift_report, decision, strategy, summary}.json` — 逐 Task 漂移报告 / 迁移决策 / 最优策略 / 评测摘要
- `outer_loop_summary.csv` — 总表（drift level / policy / max_iter / baseline·best RMSE / transfer_gap）
- `run_manifest.json` — 审计
- `memory/strategies.jsonl` — 每 Task 最佳策略库（迁移检索源）

> **双闭环机制**：内循环 = 策略生成→实验→误差诊断→反思→策略更新（P1-A）；
> 外循环 = Task 变化→漂移识别→经验迁移→策略适应→Memory 更新（P1-B）。
> 冷启动 Task 1 全量搜索并入库；后续 Task 先由 `drift_detector.py` 确定性度量
> Task 间变化（low / medium / high），再决定 **继承**（沿用上 Task 特征集 + 2 轮）、
> **修改**（继承 + 默认轮数重调优）或 **重置**（回到 FEATURE_SPEC + 更多轮数）。
> `transfer_gap` = 继承策略在本 Task 的 RMSE 相对变化，作为残余漂移信号滚动携带。

### 6. 运行 v1 特征工程实验（legacy）

```bash
# 真实 LLM 调用（v1：仅 ADD 特征）
python experiments/run_feature_agent.py --task 15 --max-iter 5

# 测试模式（不调用 LLM，使用内置示例输出）
python experiments/run_feature_agent.py --task 15 --max-iter 3 --dry-run
```

产出（保存至 `experiments/feature_agent_task15/`）：
- `result_*.json` — 基线 vs 最优迭代指标对比
- `iteration_history_*.csv` — 每轮迭代的特征数、RMSE、MAE、MAPE
- `best_features_*.txt` — 最优迭代使用的完整特征列表
- `metrics_curve_*.png` — RMSE/MAE/MAPE 三面板迭代曲线图

### 7. 编程方式调用（v1 legacy）

```python
from agent.feature_agent import run

# 一行启动 v1 闭环
result = run(
    data_dir="GEFCom2014-L_V2/Load",
    task_id=15,
    max_iterations=5,
    dry_run=False,
)
print(result["baseline_metrics"])  # 基线指标
print(result["best_metrics"])      # 最优迭代指标
print(result["history"])           # 迭代历史列表
```

---

## 自进化 Agent（P1-A）

当前主架构：在无泄漏滚动评测之上，让 LLM 同时拥有**添加 / 删除 / 替换特征**的能力，并通过
**误差驱动 + 多候选 + 显式回滚 + 经验记忆**实现真正的自进化。

### 每轮闭环

```
预测（预测月 online_h1 滚动回放）
    ↓
Error Profiling（error_profiler.py：时段/负荷状态/变化状态分段 RMSE + bias）
    ↓
检索历史经验（memory_manager.retrieve：场景相似度 top-k）
    ↓
LLM 提出 ≤3 个假设不同的候选（evolution_schema 校验）
    ↓
每候选：apply_actions → 静态泄漏检查 → evaluate_spec（预测月 RMSE + 画像）
    ↓
Selector：最优候选优于 best → 接受；否则 current:=best（自动回滚）
    ↓
实验写入 Experience Memory → 下一轮
```

### 动作空间（`agent/feature_spec.py`）

| 动作 | 含义 |
|------|------|
| `add_feature` | 追加一个血缘式 `feature_spec`（lag / rolling / time / cross） |
| `remove_feature` | 按名字删除特征 |
| `replace_feature` | 原位替换某特征（位置不变，name 由新 spec 推导） |
| `keep` | 保持当前特征集（收敛探测） |
| `rollback` | 显式要求从 best-so-far 状态重新探索 |
| `stop` | 提议本轮停止 |

- 特征名由 `name_from_spec` **确定性推导**（LLM 不需要发明列名，杜绝列名冲突）。
- `validate_spec_list` 复用泄漏检查器的血缘静态检查（lag ≥ 1、rolling 必须 shift(1)、
  cross 操作列必须是已定义的过去向特征）；非法候选整条作废，错误回灌给 LLM 重试。

### 多候选与回滚（`agent/evolution_runner.py`）

- 每轮 LLM 一次返回 `≤n_candidates` 个候选，要求**假设分化**（不同特征类型 / 不同误差段）。
- 每个候选独立评测（`evaluation/spec_evaluator.py`），**decision metric = 预测月 online_h1 滚动 RMSE**。
- **自动回滚是 Selector 的强制行为**：无候选改进即 `current := best`，不依赖 LLM 发 rollback。
  因此"第 N 轮退化 → 第 N+1 轮从 best 改善"是确定性机制，而非偶然。

### 经验记忆（`memory/memory_manager.py`）

- 每次候选实验写一行 JSONL（task_id / scenario{season, acf_24, acf_168, load_cv} / problem /
  actions / before_rmse / after_rmse / delta / outcome）。
- 下一轮（或下次运行、其它 Task）按**场景相似度**（季节 + ACF + CV 加权）检索 top-k，
  把"相似场景下哪些特征有效 / 哪些导致退化"直接喂给 LLM 上下文。

---

## 外循环：漂移检测 + 策略迁移（P1-B）

P1-B 把 P1-A 的**单 Task 自进化**串联到 Task 1–15 滚动评测上，形成**滚动自适应进化双闭环智能体**。
核心问题：**Task k → Task k+1 发生变化以后怎么办？** —— 先确定性度量变化，再由 LLM 决定策略去留。

```
                  Task k
                    │
                    ▼
             Drift Detection（drift_detector.py，确定性）
               均值/方差/分位/ACF24/ACF168/残余误差
                    │
        ┌───────────┴───────────┐
        │                       │
   Experience/Strategy Memory   当前数据统计（尾部窗口）
        │                       │
        └───────────┬───────────┘
                    ▼
              LLM 策略迁移（strategy_migration.py）
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       继承(low)   修改(medium)  重置(high)
       init=prev  init=prev     init=FEATURE_SPEC
       max_iter=2  max_iter=5   max_iter=8
          └─────────┼─────────┘
                    ▼
          EvolutionRunner(init_spec=…)   ← warm-start 内循环
                    ▼
             记录 Best Strategy + transfer_gap
                    ▼
                  Task k+1
```

### 漂移检测（`evaluation/drift_detector.py`，纯确定性，LLM 不参与计算）

- `compute_task_stats`：取 Task 历史**尾部 4 周窗口**（Task 历史是严格前缀关系，全长对比会被
  共享前缀稀释）计算 mean / std / q10·q50·q90 / ACF(24)·ACF(168) / cv / season。
- `detect_drift`：对比相邻 Task 统计量 → 五类信号全部归一化到 [0,1]：
  - LOAD：`mean_shift`（σ 计）、`std_shift`、`quantile_shift`（q10/50/90 平均相对变化）
  - Temporal：`acf24_change` / `acf168_change`（周期强度变化）
  - Residual：上一策略误差画像的 rmse / bias / peak_error（carryover context）+
    `transfer_gap` 趋势（继承策略在最近 Task 的退化幅度）
  - 聚合 `drift_score`（数据 + 时序 + 残余加权，RMS 族内聚合），level = low / medium / high。

### 策略迁移（`agent/strategy_migration.py`）

- `MigrationPlanner.plan()`：输入漂移报告 + 检索到的历史策略 + 上一 Task 策略 → LLM 输出
  `{policy: inherit|modify|reset, rationale, max_iter?}`；解析失败走**确定性兜底**
  （low→inherit / medium→modify / high→reset，max_iter 2 / 5 / 8）。
- `resolve_init_spec`：候选 init_spec 复用 `validate_spec_list` 血缘静态校验，违规回退 FEATURE_SPEC。
- **warm-start**：`EvolutionRunner(init_spec=决策.init_spec, max_iter=决策.max_iter)`
  （`evolution_runner.py` 新增 `init_spec` 参数），Round 0 即评测继承策略，
  `transfer-gap = (baseline_rmse − 上一 Task best_rmse) / 上一 Task best_rmse` 天然可测。

### 策略级记忆（`memory/memory_manager.py`）

- `StrategyRecord` 落盘 `memory/strategies.jsonl`（task_id / spec / rmse / scenario /
  stats / profile / policy / transfer_gap），`retrieve_strategies` 复用场景相似度检索，
  是迁移决策"检索相似 Task 经验"的数据源。

> **已知特性**：漂移检测度量的是**训练尾窗**的负荷形态。若波动集中在**预测月本身**（如 Task 11
> 2011-08 高波动月），预跑数据漂移可能为 low，但**残余信号（transfer_gap）会在该 Task 完成后**
> 抬升下一轮漂移分——这正是"双闭环"补盲区的机制。

---

## 特征工程系统详解（v1 Legacy）

> 以下是 v1 Agent 的架构说明，已被**自进化 Agent（P1-A）**取代（v1 仅支持 ADD 特征、跑在旧
> preprocess_pipeline 切分上）。保留此节用于理解 LLM 决策 + 确定性执行的演进脉络。

```
┌─────────────────────────────────────────────┐
│  feature_engine.py (执行引擎)                │
│  generate_lag_features / generate_rolling   │
│  generate_time_features / generate_cross     │
│  → 所有特征生成是确定性函数，LLM 不写代码     │
└────────────────────┬────────────────────────┘
                     │ 被调用
┌────────────────────▼────────────────────────┐
│  feature_agent.py (Agent 协议)               │
│  Context Builder → Prompt 渲染 → LLM 调用    │
│  → JSON Schema 校验 → execute_features_from_llm│
│  → 数据泄露检查 → 重训练 → 记录迭代历史        │
│  → LLM 只负责「决策生成什么特征」             │
└─────────────────────────────────────────────┘
```

### 特征执行引擎 (`agent/feature_engine.py`)

4 种特征类型的确定性函数，输入 DataFrame → 输出追加新列后的 DataFrame：

```python
from agent.feature_engine import (
    generate_lag_features,
    generate_rolling_features,
    generate_time_features,
    generate_cross_features,
    generate_all_features,
)

# 1. 滞后特征 — 捕获历史值对未来的影响
df = generate_lag_features(df, "LOAD", [1, 2, 24, 168])

# 2. 滚动窗口统计 — 捕获局部趋势与波动
df = generate_rolling_features(df, "LOAD", [6, 24], stats=["mean", "std", "max", "min"])

# 3. 时间特征 — 从 datetime 提取 + sin/cos 周期性编码
df = generate_time_features(df, "datetime", cyclical=True)

# 4. 交叉特征 — 两列算术运算（如温度 × 负荷）
df = generate_cross_features(df, "temp", "LOAD", "multiply")

# 5. 一键批量生成 + 生成报告
df, report = generate_all_features(
    df, target_col="LOAD", time_col="datetime",
    cross_pairs=[("temp", "LOAD", "multiply")]
)
```

### LLM Agent 协议 (`agent/feature_agent.py`)

**输入上下文**（自动从数据构建，5 层信息）：

| 层级 | 内容 | 用途 |
|------|------|------|
| A | 数据集基本信息（行数、列数、时间范围） | 了解数据规模 |
| B | 目标变量统计（均值、标准差、变异系数、分位数） | 了解预测目标特性 |
| C | 时序分析（ACF 摘要 + 趋势 + 季节性强度） | 识别时间依赖模式 |
| D | 当前特征 + 模型指标 | 了解现有特征和模型表现 |
| E | 迭代历史（已添加特征 + 指标 delta） | 判断上轮特征是否有效 |

```python
from agent.feature_agent import build_context_from_data

ctx = build_context_from_data(
    train_df,
    target_col="LOAD",
    time_col="datetime",
    feature_importance_df=feat_imp_df,    # 来自 LightGBM
    val_metrics={"RMSE": 8.53, "MAE": 6.70},
)
```

**LLM 严格输出格式**（JSON Schema 校验）：

```json
{
  "iteration": 1,
  "analysis": "日周期性显著(ACF_24=0.80)，温度与负荷相关性强，建议增加温度滞后和滚动峰谷统计",
  "new_features": [
    {"name": "lag_72_load", "type": "lag", "target_col": "LOAD", "params": {"lag": 72}},
    {"name": "rolling_max_24_load", "type": "rolling", "target_col": "LOAD", "params": {"window": 24, "stat": "max"}},
    {"name": "cross_temp_load", "type": "cross", "params": {"col1": "temp", "col2": "LOAD", "operation": "multiply"}}
  ]
}
```

**输出校验 + 执行**：

```python
from agent.feature_agent import validate_llm_output, execute_features_from_llm

# 多层校验：JSON 解析 → 结构 → 参数字段 → 类型/范围 → 列名存在性
validated = validate_llm_output(llm_json_str, available_columns=df.columns.tolist())

# 自动翻译为 feature_engine 调用
df_new, added_cols, skipped = execute_features_from_llm(df, validated)
```

### 安全机制

| 机制 | 说明 |
|------|------|
| **时间因果约束** | System Prompt 明确禁止使用未来信息；lag 必须 > 0 |
| **数据泄露检查** | `check_data_leakage()` 自动扫描新特征的时间因果性 |
| **安全 lag 上限** | 自动限制 lag/rolling window ≤ 验证集最小时间间隔，防止 NaN 泛滥 |
| **LLM 输出校验** | JSON Schema 多层校验 + 最多 3 次重试（错误信息反馈给 LLM 自我修正） |
| **确定性执行** | LLM 不生成代码，只输出结构化决策 → 执行引擎翻译为安全的函数调用 |

---

## 基线结果 (Task 15) — Legacy 协议

> ⚠️ 下列结果来自**旧评测协议**（训练月内部 70/15/15 顺序切分 + rolling 特征含当前行的自泄露），
> 属于项目历史，**不可与新的无泄漏 Task 1–15 滚动回放结果直接对比**。新评测体系见上方
> 「Task 1–15 无泄漏滚动回放评测」。

| 模型 | Train RMSE | Train MAPE | Val RMSE | Val MAPE | Test RMSE | Test MAPE | 参数量 |
|------|-----------|------------|----------|----------|-----------|-----------|--------|
| LightGBM | 7.74 | 4.86% | 8.53 | 4.97% | 9.23 | 5.78% | — |
| LSTM | 6.07 | 3.97% | 6.20 | 4.04% | 11.70 | 6.94% | 54,849 |
| **PatchTST** | **2.63** | **1.57%** | **3.22** | **1.95%** | **2.84** | **1.63%** | 139,301 |

> **PatchTST 全面最优**：通过 patching 将时序切分为 subseries-level tokens，配合通道独立 Transformer + RevIN 归一化，在全部三个集合上大幅领先。Test RMSE（2.84）仅为 LightGBM（9.23）的 31%。
>
> LSTM 过拟合明显（Train 6.07 → Test 11.70），LightGBM 泛化更稳（Train 7.74 → Test 9.23），PatchTST 泛化最佳（Train 2.63 → Test 2.84）。

---

## 设计原则

1. **关注点分离**：LLM 负责"决策"（生成 / 删除 / 替换什么特征），执行引擎负责"执行"（确定性函数），LLM 从不写代码
2. **统一指标接口**：所有模型共用 `utils/metrics.py`，确保指标定义一致、可直接横向对比
3. **误差驱动自进化**：每轮基于误差画像（分段 RMSE / bias）与 delta 指标做诊断，多候选择优 / 自动回滚；经验记忆跨 Task 复用
4. **安全优先**：时间因果性是最高优先级的约束，System Prompt + 血缘静态检查 + 值级泄漏检查（Pass A/B/C）三重保障
5. **鲁棒重试**：LLM 输出不符合 Schema 时自动重试（最多 3 次），错误信息即时反馈
6. **可审计评测**：所有结果跑在无泄漏滚动回放（预测月 online_h1）之上，产出 run_manifest + 逐小时预测，可复现可审计

---

## TODO

- [ ] P1-A 真实 LLM 全量验证（Task 1–15，发布前跑 `--leak-check full` 全量审计）
- [x] ~~漂移检测 Agent（P1-B：`evaluation/drift_detector.py` + `agent/strategy_migration.py`）~~
- [x] ~~多 Task 冷启动 / 策略迁移（P1-B：`run_outer_loop.py`，warm-start 继承历史最佳策略）~~
- [ ] P1-B 真实 LLM 全量验证（Task 1–15 外循环，迁移决策走真实 QwenClient）
- [ ] recursive_month_ahead 作为主评测口径（当前仅 online_h1 为主）
- [ ] 超参调优 Agent (`agent/tuning_agent.py`)
- [ ] 报告生成 Agent (`agent/report_agent.py`)
- [ ] Transformer 基线模型 (`models/Transformer/`)
- [x] ~~多能源扩展 · Wind 数据接入 + 基线回放（`data/wind_loader.py` / `wind_task_builder.py` / `evaluation/wind_replay.py` / `run_wind_replay.py`）~~
- [ ] 多能源扩展 · Wind 自进化 Agent 泛化 / 多专家协同（V3.0）
- [ ] 多能源扩展 · Price 决策效能主线（P-Value：`price_loader` + 储能套利评估器 + 尖峰双层，见 ROADMAP）
