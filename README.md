# AutoML-Agent

LLM-driven AutoML Agent for Short-term Electricity Load Forecasting with Automated Feature Engineering and Hyperparameter Optimization.

数据集：[GEFCom2014-L_V2](https://www.sciencedirect.com/journal/international-journal-of-forecasting/vol/30/issue/2)（Global Energy Forecasting Competition 2014，负荷预测赛道）

---

## 项目结构

```
AutoML-Agent/
├── data/
│   ├── preprocessing.py              # 数据预处理流水线（时间解析+缺失填充+特征工程+时序切分）
│   └── README_preprocessing.md       # 预处理详解（个人笔记，不入库）
│
├── models/
│   ├── baseline/
│   │   └── lgb_gefcom2014.py         # LightGBM 基线（特征重要性→LLM Agent）
│   ├── LSTM/
│   │   └── LSTM_baseline.py          # LSTM 基线（滑动窗口+归一化+早停）
│   ├── Transformer/                  # (TODO) Transformer 模型
│   └── PatchTST/                     # (TODO) PatchTST 模型
│
├── utils/
│   ├── metrics.py                    # 通用评估指标：RMSE / MAE / MAPE / SMAPE / R²
│   └── data_loader.py                # 滑动窗口 DataLoader（StandardScaler + shuffle 控制）
│
├── agent/
│   ├── feature_engine.py             # 特征执行引擎（确定性函数库：lag/rolling/time/cross）
│   ├── feature_agent.py              # LLM 特征工程 Agent（I/O 协议 + 上下文构建 + 输出校验）
│   ├── tuning_agent.py               # (TODO) 超参调优 Agent
│   └── report_agent.py               # (TODO) 报告生成 Agent
│
└── experiments/                      # 实验结果存放
```

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

### 2. LightGBM 基线

```bash
python models/baseline/lgb_gefcom2014.py --task 15
```

产出：
- `lgb_baseline_task15.txt` — 训练好的模型
- `lgb_baseline_task15_metrics.json` — 结构化指标字典
- `lgb_baseline_task15_feature_importance.csv` — 特征重要性（供 LLM Agent 使用）
- `lgb_baseline_task15_predictions.csv` — 测试集预测结果

### 3. LSTM 基线

```bash
python models/LSTM/LSTM_baseline.py --task 15 --max-epochs 200 --patience 20
```

核心流程：`StandardScaler 归一化 → 滑动窗口 → LSTM 训练 → 早停 → inverse_transform → 指标`

产出：
- `lstm_baseline_task15_best.pt` — 最佳 checkpoint
- `lstm_baseline_task15_metrics.json` — 结构化指标（与 LGB 同格式）
- `lstm_baseline_task15_predictions.csv` — 测试集预测结果

### 4. PatchTST 基线

```bash
python models/PatchTST/patch_tst_baseline.py --task 15 --max-epochs 200 --patience 20
```

核心流程：`StandardScaler → 滑动窗口 → Patching → Channel-Independent Transformer → RevIN 逆归一化 → 指标`

关键参数：
- `--seq-len` 历史窗口（默认 48 小时）
- `--patch-len` / `--stride` 控制 patch 切分（默认自动选择）
- `--d-model` / `--n-heads` / `--n-layers` 控制 Transformer 规模
- `--no-revin` 禁用 RevIN

产出：
- `patchtst_baseline_task15_best.pt` — 最佳 checkpoint
- `patchtst_baseline_task15_metrics.json` — 结构化指标（与 LGB/LSTM 同格式）
- `patchtst_baseline_task15_predictions.csv` — 测试集预测结果

---

## 特征工程系统 (LLM Agent 核心)

这是整个项目最核心的模块，体现 LLM + 自动化的价值。
架构分为两层：**执行引擎**（确定性函数）和 **Agent 协议**（LLM 决策 + 输入输出规范）。

### 架构

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

# 2. 滚动窗口统计 — 捕获局部趋势
df = generate_rolling_features(df, "LOAD", [6, 24], stats=["mean", "std", "max", "min"])

# 3. 时间特征 — 从 datetime 提取 + sin/cos 周期性编码
df = generate_time_features(df, "datetime", cyclical=True)

# 4. 交叉特征 — 两列算术运算
df = generate_cross_features(df, "temp", "LOAD", "multiply")

# 5. 一键批量生成 + 生成报告
df, report = generate_all_features(
    df, target_col="LOAD", time_col="datetime",
    cross_pairs=[("temp", "LOAD", "multiply")]
)
# report → {"n_original_cols": 4, "n_new_cols": 36, "new_columns": [...], ...}
```

### LLM Agent 协议 (`agent/feature_agent.py`)

**输入上下文**（自动从数据构建）：

```python
from agent.feature_agent import build_context_from_data, build_llm_prompt

ctx = build_context_from_data(
    train_df,
    target_col="LOAD",
    time_col="datetime",
    feature_importance_df=feat_imp_df,   # 来自 LightGBM
    val_metrics={"RMSE": 8.53, "MAE": 6.70},
)

# 渲染 LLM Prompt（含数据集统计 + ACF + 特征重要性 + 迭代历史）
prompt = build_llm_prompt(ctx)
# → 发送给任意支持 JSON 输出的 LLM
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

# 多层校验：JSON 结构 → 参数字段 → 类型/范围 → 列名存在性
validated = validate_llm_output(llm_json_str, available_columns=df.columns.tolist())

# 自动翻译为 feature_engine 调用
df_new, added_cols, skipped = execute_features_from_llm(df, validated)
```

**迭代上下文**：自动追踪指标 delta，帮助 LLM 判断上一轮特征是否有效。

```python
from agent.feature_agent import build_iteration_context, FeatureIterationHistory

# 构建带历史的迭代上下文
ctx_iter = build_iteration_context(
    ctx, iteration=2,
    previous_val_metrics={"RMSE": 8.53},
    previous_features_added=["lag_72_load", "rolling_max_24_load"],
)

# 追踪器自动记录每轮变化 + 识别 best iteration
history = FeatureIterationHistory()
history.record(...)
print(history.summary())          # DataFrame 概览
print(history.best_iteration())   # RMSE 改善最大的轮次
```

---

## 基线结果 (Task 15)

| 模型 | Train RMSE | Train MAPE | Val RMSE | Val MAPE | Test RMSE | Test MAPE | 参数量 |
|------|-----------|------------|----------|----------|-----------|-----------|--------|
| LightGBM | 7.74 | 4.86% | 8.53 | 4.97% | 9.23 | 5.78% | — |
| LSTM | 6.07 | 3.97% | 6.20 | 4.04% | 11.70 | 6.94% | 54,849 |
| **PatchTST** | **2.63** | **1.57%** | **3.22** | **1.95%** | **2.84** | **1.63%** | 139,301 |

> **PatchTST 全面最优**：通过 patching 将时序切分为 subseries-level tokens，配合通道独立 Transformer + RevIN 归一化，在全部三个集合上大幅领先。Test RMSE（2.84）仅为 LightGBM（9.23）的 31%，LSTM（11.70）的 24%。
>
> LSTM 过拟合明显（Train 6.07 → Test 11.70），LightGBM 泛化更稳（Train 7.74 → Test 9.23），PatchTST 泛化最佳（Train 2.63 → Test 2.84）。

---

## 环境

- Python 3.12+
- PyTorch
- LightGBM
- scikit-learn
- pandas, numpy
