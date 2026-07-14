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
│   ├── feature_agent.py              # (TODO) LLM 特征工程 Agent
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

---

## 基线结果 (Task 15)

| 模型 | Val RMSE | Val MAPE | Test RMSE | Test MAPE | 参数量 |
|------|----------|----------|-----------|-----------|--------|
| LightGBM | 8.53 | 4.97% | **9.23** | **5.78%** | — |
| LSTM | **6.20** | **4.04%** | 11.70 | 6.94% | 54,849 |

> LSTM 验证集更好但测试集更差 → 小样本过拟合（训练集仅 386 行）。树模型在小数据上泛化更稳。

---

## 环境

- Python 3.12+
- PyTorch
- LightGBM
- scikit-learn
- pandas, numpy
