# AutoML-Agent

LLM 驱动的 AutoML Agent —— 面向短期电力负荷预测的**自动化特征工程**、**自进化特征优化**与**无泄漏闭环评测**系统。

**核心创新**：大语言模型（LLM）作为**预测建模决策器**，在**受约束动作空间**（ADD / REMOVE / REPLACE / KEEP / ROLLBACK / STOP 六类动作，特征仅允许 lag / rolling / time / cross 四类确定性操作，参数受时间因果性与安全上限约束）内**自主设计与删改特征**。每轮 Agent 在**误差画像**（分段 RMSE + bias）驱动下提出 **3 个不同假设的候选方案**，逐一评测后**择优 / 自动回滚**，并把每次实验写入**经验记忆**（跨 Task 共享，场景相似度检索复用）。所有评测跑在**无泄漏滚动回放**（预测月 online_h1）之上，保证结果可信。LLM 不写代码——基于数据统计、自相关分析（ACF）、特征重要性、误差画像和历史经验，**自主决定**生成 / 删除什么特征；确定性执行引擎负责**安全执行**。两者构成完整的 **感知→决策→执行→评估→反馈→记忆** 闭环。

数据集：[GEFCom2014-L_V2](https://www.sciencedirect.com/journal/international-journal-of-forecasting/vol/30/issue/2)（Global Energy Forecasting Competition 2014，负荷预测赛道，含 15 个任务）

> **计划扩展**：后续将陆续增加更多电力负荷预测数据集，覆盖不同地区、时间粒度和数据特征，以验证模型和特征工程策略的通用性与鲁棒性。

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
```

---

## 项目结构

```
AutoML-Agent/
├── data/
│   ├── preprocessing.py                 # 数据预处理流水线（时间戳消歧 / 填充 / 切分 / 基线特征）[legacy]
│   ├── gefcom_loader.py                 # GEFCom 统一加载器（train/benchmark/solution + 真值解析）
│   ├── availability.py                  # 每个 Task 的「可用历史 + 预测区间」定义
│   └── task_builder.py                  # 血缘式特征规格 + 严格过去向特征构造（lag/rolling/time/cross）+ GEFComTask
│
├── evaluation/                          # 无泄漏滚动回放评测体系
│   ├── forecast_protocol.py             # 评测协议（online_h1 / recursive_month_ahead）
│   ├── leakage_checker.py               # 严格值级泄漏检查（Pass A 血缘 / Pass B 重算 / Pass C 别名）
│   ├── rolling_backtest.py              # 逐小时滚动回测（按协议回填，_features_at 与 build_features 逐位一致）
│   ├── evaluator.py                     # 指标计算 + 多 Task 汇总（复用 utils/metrics）
│   ├── error_profiler.py                # 误差画像（时段/负荷状态/变化状态分段 + bias + top-worst）
│   ├── spec_evaluator.py                # 候选特征集评测器（decision metric = 预测月 online_h1）
│   └── task_replay.py                   # Task 1–15 回放主循环 + 审计输出（predictions/run_manifest）
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
│   ├── scripted_llm.py                  # ★确定性 LLM（测试 / --dry-run）
│   ├── feature_engine.py                # 确定性特征执行引擎（lag/rolling/time/cross）[legacy]
│   ├── feature_agent.py                 # LLM Agent 协议 + 闭环迭代调度器（v1，仅 ADD）[legacy]
│   ├── tuning_agent.py                  # (TODO) 超参调优 Agent
│   └── report_agent.py                  # (TODO) 报告生成 Agent
│
├── memory/
│   └── memory_manager.py                # ★经验记忆（JSONL + 场景相似度检索，跨 Task 共享）
│
├── experiments/
│   ├── run_self_evolving_agent.py       # ★自进化 Agent 实验（CLI：--task / --max-iter / --dry-run / --n-candidates）
│   ├── run_task_replay.py               # Task 1–15 无泄漏滚动回放评测（CLI）
│   ├── run_feature_agent.py             # v1 特征工程 Agent 实验（legacy）
│   ├── feature_agent_task15/            # Task 15 v1 实验输出
│   └── output/                          # 评测输出（predictions / manifest，gitignored）
│
├── tests/
│   ├── test_evaluation_suite.py         # 评测体系测试（T1–T6）
│   └── test_evolution_suite.py          # ★自进化 Agent 测试（E1–E13，E6 复现回滚完成标准）
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

### 5. 运行 v1 特征工程实验（legacy）

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

### 6. 编程方式调用（v1 legacy）

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
- [ ] recursive_month_ahead 作为主评测口径（当前仅 online_h1 为主）
- [ ] 漂移检测 Agent (`agent/drift_agent.py`，ROADMAP V2.0)
- [ ] 超参调优 Agent (`agent/tuning_agent.py`)
- [ ] 报告生成 Agent (`agent/report_agent.py`)
- [ ] Transformer 基线模型 (`models/Transformer/`)
- [ ] 多 Task 冷启动：用历史经验记忆自动初始化特征集
- [ ] 多能源扩展（Wind：`GEFCom2014-W_V2`）
