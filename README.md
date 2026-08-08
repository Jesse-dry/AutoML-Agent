# AutoML-Agent

LLM 驱动的 AutoML Agent —— 面向短期电力负荷预测的**自动化特征工程**与**闭环迭代优化**系统。

**核心创新**：大语言模型（LLM）作为**预测建模决策器**，在**受约束动作空间**（仅允许 lag / rolling / time / cross 四类确定性特征操作，参数受时间因果性与安全上限约束）内**自主设计特征**，并通过每轮**实验反馈**（train/val/test 指标 delta）持续优化预测性能。LLM 不写代码——基于数据统计、自相关分析（ACF）、特征重要性和历史迭代反馈，**自主决定**生成什么特征；确定性执行引擎负责**安全执行**。两者构成一个完整的 **感知→决策→执行→评估→反馈** 闭环。

数据集：[GEFCom2014-L_V2](https://www.sciencedirect.com/journal/international-journal-of-forecasting/vol/30/issue/2)（Global Energy Forecasting Competition 2014，负荷预测赛道，含 15 个任务）

> **计划扩展**：后续将陆续增加更多电力负荷预测数据集，覆盖不同地区、时间粒度和数据特征，以验证模型和特征工程策略的通用性与鲁棒性。

---

## 整体架构

```
               GEFCom2014-L_V2 数据集
                       │
                       ▼
           data/preprocessing.py
     (时间戳解析 → 缺失填充 → 切分 → 基线特征)
                       │
                       ▼
              Train / Val / Test
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                  ▼
  LightGBM           LSTM            PatchTST
  (树模型基线)     (循环网络基线)    (Transformer基线)
     │                 │                  │
     ▼                 ▼                  ▼
  特征重要性        train/val/test 三阶段指标（原始量纲）
     │                 │
     └────────┬────────┘
              │
              ▼
 ┌────────────────────────────────────────────┐
 │       LLM 特征工程闭环 (核心模块)            │
 │                                            │
 │  ┌──────────────────────────────────┐      │
 │  │ Context Builder                  │      │
 │  │ 数据集统计 + ACF + 特征重要性     │      │
 │  │ + 趋势/季节性 + 历史 delta 指标   │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ System Prompt + User Prompt      │      │
 │  │ 角色/约束/领域知识/迭代历史       │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ Qwen LLM (DashScope API)         │      │
 │  │ AI 自主分析数据 → 输出特征方案    │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ JSON Schema 校验 + 重试 (≤3次)    │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ feature_engine 确定性执行         │      │
 │  │ LLM 不写代码，只决定"做什么"      │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ 数据泄露检查 check_data_leakage() │      │
 │  │ 时间因果性自动校验                │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ LightGBM 重训练 → 新指标          │      │
 │  └────────────┬─────────────────────┘      │
 │               ▼                            │
 │  ┌──────────────────────────────────┐      │
 │  │ 记录迭代历史 + 判断是否继续       │      │
 │  │ RMSE 改善 → 下一轮 / 退化 → 终止  │      │
 │  └──────────────────────────────────┘      │
 │                                            │
 └────────────────────────────────────────────┘
              │
              ▼
     实验产出 (experiments/)
     result.json / history.csv / best_features.txt / metrics_curve.png
```

---

## 项目结构

```
AutoML-Agent/
├── data/
│   └── preprocessing.py                 # 数据预处理流水线
│
├── models/
│   ├── baseline/
│   │   └── lgb_gefcom2014.py            # LightGBM 基线（产出特征重要性供 Agent 使用）
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
├── agent/
│   ├── feature_engine.py                # 确定性特征执行引擎（lag/rolling/time/cross）
│   ├── feature_agent.py                 # LLM Agent 协议 + 闭环迭代调度器
│   ├── tuning_agent.py                  # (TODO) 超参调优 Agent
│   └── report_agent.py                  # (TODO) 报告生成 Agent
│
├── experiments/
│   ├── run_feature_agent.py             # 端到端特征工程实验脚本（CLI）
│   └── feature_agent_task15/            # Task 15 实验输出
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

### 4. 运行 LLM 特征工程实验

```bash
# 真实 LLM 调用
python experiments/run_feature_agent.py --task 15 --max-iter 5

# 测试模式（不调用 LLM，使用内置示例输出）
python experiments/run_feature_agent.py --task 15 --max-iter 3 --dry-run
```

产出（保存至 `experiments/feature_agent_task15/`）：
- `result_*.json` — 基线 vs 最优迭代指标对比
- `iteration_history_*.csv` — 每轮迭代的特征数、RMSE、MAE、MAPE
- `best_features_*.txt` — 最优迭代使用的完整特征列表
- `metrics_curve_*.png` — RMSE/MAE/MAPE 三面板迭代曲线图

### 5. 编程方式调用

```python
from agent.feature_agent import run

# 一行启动完整闭环
result = run(
    data_dir="GEFCom2014-L_V2/Load",
    task_id=15,
    max_iterations=5,
    dry_run=False,
)
print(result["baseline_metrics"])  # 基线指标
print(result["best_metrics"])      # 最优迭代指标
print(result["iterations"])        # 迭代历史列表
```

---

## 特征工程系统详解

这是整个项目最核心的模块，体现 LLM + 自动化的价值。架构分为两层：**执行引擎**（确定性函数）和 **Agent 协议**（LLM 决策）。

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

1. **关注点分离**：LLM 负责"决策"（生成什么特征），执行引擎负责"执行"（确定性函数），LLM 从不写代码
2. **统一指标接口**：所有模型共用 `utils/metrics.py`，确保指标定义一致、可直接横向对比
3. **闭环自优化**：LLM 每轮接收 delta 指标，能判断上轮特征是否有效并自我纠偏
4. **安全优先**：时间因果性是最高优先级的约束，System Prompt + 代码检查双重保障
5. **鲁棒重试**：LLM 输出不符合 Schema 时自动重试（最多 3 次），错误信息即时反馈

---

## TODO

- [ ] 超参调优 Agent (`agent/tuning_agent.py`)
- [ ] 报告生成 Agent (`agent/report_agent.py`)
- [ ] Transformer 基线模型 (`models/Transformer/`)
- [ ] 多任务并行特征工程
- [ ] 模型选择 Agent（自动选择最优模型类型）
