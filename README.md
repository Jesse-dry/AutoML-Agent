# AutoML-Agent

An LLM-driven AutoML Agent for Short-term Electricity Load Forecasting with Automated Feature Engineering and Hyperparameter Optimization.

## 项目结构

```
├── data                  # 数据集目录
├── models                # 模型存放目录
│   ├── LSTM              # LSTM 模型
│   ├── Transformer       # Transformer 模型
│   └── PatchTST          # PatchTST 模型
├── agent                 # 智能代理模块
│   ├── feature_agent.py  # 特征工程代理
│   ├── tuning_agent.py   # 超参调优代理
│   └── report_agent.py   # 报告生成代理
├── experiments           # 实验结果存放目录
└── README.md
```

## 环境说明

- Python 3.10+
- PyTorch
- 其他依赖见各模块内部说明
