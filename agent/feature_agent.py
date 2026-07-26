"""
LLM 特征工程 Agent — 输入输出协议
==================================

这是整个 AutoML 项目的核心模块。Agent 负责「决策生成什么特征」，
但**不负责写代码**——所有特征生成由 feature_engine.py 中的确定性函数执行。

架构：
  ┌──────────────────────────────────────────────────┐
  │  Context Builder                                 │
  │  (数据集统计 + ACF + 特征重要性 + 历史指标)       │
  └──────────────────┬───────────────────────────────┘
                     │  结构化 Prompt
                     ▼
  ┌──────────────────────────────────────────────────┐
  │  LLM (任何支持 JSON 输出的模型)                   │
  │  → 分析数据 → 决策新特征 → 输出严格 JSON          │
  └──────────────────┬───────────────────────────────┘
                     │  严格 JSON
                     ▼
  ┌──────────────────────────────────────────────────┐
  │  Output Validator → Execute via feature_engine   │
  └──────────────────────────────────────────────────┘

模块结构：
  - FeatureAgentContext        — 输入上下文 dataclass
  - compute_acf_summary         — ACF 自相关计算
  - detect_trend_seasonality    — 趋势/周期性检测
  - build_context_from_data     — 从数据自动构建上下文
  - build_iteration_context     — 迭代上下文（含历史指标 delta）
  - build_llm_prompt            — 渲染上下文 → LLM prompt 字符串
  - FEATURE_AGENT_OUTPUT_SCHEMA — 输出 JSON Schema
  - validate_llm_output         — 校验 LLM 输出
  - execute_features_from_llm   — 将 LLM 决策翻译为 feature_engine 调用
  - FeatureIterationHistory     — 迭代历史追踪

用法：
    from agent.feature_agent import (
        build_context_from_data, build_llm_prompt,
        validate_llm_output, execute_features_from_llm,
    )

    # Step 1: 构建上下文
    ctx = build_context_from_data(train_df, target_col="LOAD", time_col="datetime",
                                   feature_importance_df=feat_imp_df, ...)
    # Step 2: 生成 prompt → 发给 LLM
    prompt = build_llm_prompt(ctx)
    # Step 3: LLM 返回 JSON → 校验
    validated = validate_llm_output(llm_json_str)
    # Step 4: 执行
    new_df = execute_features_from_llm(df, validated)
"""

from __future__ import annotations

import json
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---- 引用特征执行引擎 ----
from agent.feature_engine import (
    generate_lag_features,
    generate_rolling_features,
    generate_time_features,
    generate_cross_features,
    describe_new_columns,
)


# ============================================================
# 1. 输入上下文数据结构
# ============================================================

@dataclass
class FeatureAgentContext:
    """
    LLM Agent 的完整输入上下文。

    所有字段分四大类：
      A. 数据集基本信息
      B. 目标列统计
      C. 时序分析结果
      D. 当前特征 & 模型状态
      E. 迭代历史（可选）
    """

    # ---- A. 数据集基本信息 ----
    dataset_name: str = ""
    n_samples: int = 0
    n_features: int = 0
    sampling_frequency: str = ""          # 如 "1小时"、"15分钟"
    target_col: str = "LOAD"
    time_col: str = "datetime"
    history_window: Optional[int] = None   # 用于预测的历史窗口长度
    prediction_horizon: int = 1            # 预测步长

    # ---- B. 目标列统计 ----
    target_mean: float = 0.0
    target_std: float = 0.0
    target_min: float = 0.0
    target_max: float = 0.0
    target_cv: float = 0.0                 # 变异系数 = std/mean
    target_q05: float = 0.0
    target_q95: float = 0.0

    # ---- C. 时序分析结果 ----
    acf_summary: Dict[int, float] = field(default_factory=dict)
    # ↑ {1: 0.95, 24: 0.82, 168: 0.61, ...}
    has_daily_seasonality: bool = False     # 是否有日周期 (acf[24] > 0.3)
    has_weekly_seasonality: bool = False    # 是否有周周期 (acf[168] > 0.3)
    trend_strength: str = "未知"            # "强上升"/"强下降"/"弱趋势"/"无明显趋势"
    seasonality_strength: str = "未知"      # "强"/"中等"/"弱"/"无"

    # ---- D. 当前特征 & 模型状态 ----
    current_features: List[str] = field(default_factory=list)
    feature_importance: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame()
    )
    # ↑ 列: feature, importance_gain, importance_gain_norm
    top10_features: List[Dict] = field(default_factory=list)
    bottom10_features: List[Dict] = field(default_factory=list)

    current_val_metrics: Dict[str, float] = field(default_factory=dict)
    current_test_metrics: Dict[str, float] = field(default_factory=dict)

    # ---- E. 迭代历史 ----
    iteration: int = 0
    previous_val_metrics: Optional[Dict[str, float]] = None
    metrics_delta: Dict[str, float] = field(default_factory=dict)
    # ↑ {"RMSE": -0.5, "MAE": -0.3} 负值表示改善
    previous_features_added: List[str] = field(default_factory=list)
    max_iterations: int = 5
    stop_reason: str = ""                  # 若已停止，记录原因

    def to_dict(self) -> dict:
        """转为字典（供 prompt 渲染或序列化）。"""
        d = asdict(self)
        # DataFrame 不可直接序列化，转 records
        if isinstance(d.get("feature_importance"), pd.DataFrame):
            d["feature_importance"] = d["feature_importance"].to_dict(orient="records")
        return d


# ============================================================
# 2. 时序分析工具
# ============================================================

def compute_acf_summary(
    df: pd.DataFrame,
    target_col: str,
    lags: List[int] = None,
    max_lag: int = 336,
) -> Dict[int, float]:
    """
    计算目标列在指定滞后步数上的自相关系数 (ACF)。

    用 pandas 自带的 .autocorr() 实现，零依赖。
    ACF 值域 [-1, 1]，越接近 ±1 表示相关性越强。

    Parameters
    ----------
    df : pd.DataFrame
        时序数据（已按时间排序）
    target_col : str
        目标列名
    lags : list of int, optional
        关注的滞后步数。默认 [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
    max_lag : int
        若 lags 未指定，自动生成 1..max_lag 的等差序列

    Returns
    -------
    dict: {lag: acf_value}
    """
    if target_col not in df.columns:
        raise ValueError(f"列 '{target_col}' 不存在")

    series = df[target_col].dropna()
    if len(series) < 10:
        raise ValueError(f"数据量 ({len(series)}) 不足，至少需要 10 个点")

    if lags is None:
        # 智能选择：常用滞后 + 采样频率相关滞后
        lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
        # 过滤掉超过数据量 1/4 的滞后
        max_allowed = len(series) // 4
        lags = [k for k in lags if k <= max_allowed]

    result = {}
    for k in lags:
        if k < 1 or k >= len(series):
            continue
        try:
            acf = series.autocorr(lag=k)
            result[k] = round(float(acf), 4) if not np.isnan(acf) else 0.0
        except Exception:
            result[k] = 0.0

    return result


def detect_trend_seasonality(
    df: pd.DataFrame,
    target_col: str,
    acf_summary: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    检测趋势方向和季节性强弱。

    趋势检测：线性回归斜率 + 符号判断
    季节性检测：基于 ACF 在关键滞后 (24, 168) 的值

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    acf_summary : dict, optional
        预计算的 ACF，若未提供则自动计算

    Returns
    -------
    dict: {
        "has_daily_seasonality": bool,
        "has_weekly_seasonality": bool,
        "trend_strength": str,
        "seasonality_strength": str,
        "trend_slope": float,        # 每小时变化量
        "acf_24": float,
        "acf_168": float,
    }
    """
    series = df[target_col].dropna()
    n = len(series)

    # ---- 趋势检测 ----
    x = np.arange(n).reshape(-1, 1)
    y = series.values
    slope = 0.0

    if n >= 10:
        # 简单线性回归
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x.flatten() - x_mean) * (y - y_mean))
                      / np.sum((x.flatten() - x_mean) ** 2))

        # 归一化斜率（相对于均值的变化率）
        if y_mean != 0:
            normalized_slope = slope * n / y_mean  # 整个时间跨度内的相对变化
        else:
            normalized_slope = 0.0
    else:
        normalized_slope = 0.0

    # 趋势强度判断
    if abs(normalized_slope) > 0.3:
        trend_strength = "强上升" if slope > 0 else "强下降"
    elif abs(normalized_slope) > 0.1:
        trend_strength = "弱上升" if slope > 0 else "弱下降"
    else:
        trend_strength = "无明显趋势"

    # ---- 季节性检测 ----
    if acf_summary is None:
        acf_summary = compute_acf_summary(df, target_col)

    acf_24 = acf_summary.get(24, 0)
    acf_168 = acf_summary.get(168, 0)

    has_daily = acf_24 > 0.3
    has_weekly = acf_168 > 0.3

    # 综合季节性强度
    max_acf = max(acf_24, acf_168)
    if max_acf > 0.7:
        seasonality_strength = "强"
    elif max_acf > 0.3:
        seasonality_strength = "中等"
    elif max_acf > 0.1:
        seasonality_strength = "弱"
    else:
        seasonality_strength = "无"

    return {
        "has_daily_seasonality": has_daily,
        "has_weekly_seasonality": has_weekly,
        "trend_strength": trend_strength,
        "seasonality_strength": seasonality_strength,
        "trend_slope": round(slope, 6),
        "acf_24": round(acf_24, 4),
        "acf_168": round(acf_168, 4),
    }


# ============================================================
# 3. 上下文构建器
# ============================================================

def build_context_from_data(
    train_df: pd.DataFrame,
    target_col: str = "LOAD",
    time_col: str = "datetime",
    feature_importance_df: Optional[pd.DataFrame] = None,
    current_features: Optional[List[str]] = None,
    val_metrics: Optional[Dict[str, float]] = None,
    test_metrics: Optional[Dict[str, float]] = None,
    history_window: Optional[int] = None,
    prediction_horizon: int = 1,
    dataset_name: str = "",
    sampling_frequency: str = "",
) -> FeatureAgentContext:
    """
    从原始数据自动构建 FeatureAgentContext。

    自动完成：
      - 目标列统计（均值/方差/分位数）
      - ACF 自相关分析
      - 趋势/季节性检测
      - 特征重要性 Top-10 / Bottom-10 提取

    Parameters
    ----------
    train_df : pd.DataFrame
        训练集（已包含所有现有特征）
    target_col : str
        目标列名
    time_col : str
        时间列名（或 index 名）
    feature_importance_df : pd.DataFrame, optional
        特征重要性表（来自 LightGBM 等）。需包含列: feature, importance_gain
    current_features : list of str, optional
        当前使用的特征列名列表。若未提供，自动排除 target_col 和 time_col
    val_metrics : dict, optional
        验证集指标，如 {"RMSE": 8.53, "MAE": 6.70}
    test_metrics : dict, optional
        测试集指标
    history_window : int, optional
        用于预测的历史窗口长度（滑动窗口场景）
    prediction_horizon : int
        预测步长
    dataset_name : str
        数据集名称标签
    sampling_frequency : str
        采样频率描述。若未提供，尝试自动推断

    Returns
    -------
    FeatureAgentContext
    """
    df = train_df.copy()
    n_samples = len(df)

    # ---- 自动推断采样频率 ----
    if not sampling_frequency:
        try:
            if time_col in df.columns:
                time_series = pd.to_datetime(df[time_col])
            elif df.index.name == time_col or isinstance(df.index, pd.DatetimeIndex):
                time_series = df.index.to_series()
            else:
                time_series = None

            if time_series is not None:
                diffs = time_series.diff().dropna()
                if len(diffs) > 0:
                    median_diff = diffs.median()
                    seconds = median_diff.total_seconds()
                    if seconds <= 60:
                        sampling_frequency = f"{int(seconds)}秒"
                    elif seconds <= 3600:
                        sampling_frequency = f"{int(seconds / 60)}分钟"
                    elif seconds <= 86400:
                        sampling_frequency = f"{int(seconds / 3600)}小时"
                    else:
                        sampling_frequency = f"{int(seconds / 86400)}天"
        except Exception:
            pass

    if not sampling_frequency:
        sampling_frequency = "未知"

    # ---- 目标列统计 ----
    target = df[target_col].dropna()
    target_mean = float(target.mean())
    target_std = float(target.std())
    target_min = float(target.min())
    target_max = float(target.max())
    target_cv = round(target_std / target_mean, 4) if target_mean != 0 else 0.0
    target_q05 = float(target.quantile(0.05))
    target_q95 = float(target.quantile(0.95))

    # ---- ACF 分析 ----
    acf_summary = compute_acf_summary(df, target_col)

    # ---- 趋势/季节性 ----
    ts_info = detect_trend_seasonality(df, target_col, acf_summary)

    # ---- 特征列表 ----
    if current_features is None:
        # 自动推断：排除目标列、时间列、以及可能的元数据列
        exclude = {target_col, time_col, "task_id", "ZONEID", "TIMESTAMP"}
        current_features = [c for c in df.columns if c not in exclude]

    n_features = len(current_features)

    # ---- 特征重要性 ----
    top10 = []
    bottom10 = []
    fi_df = pd.DataFrame()

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        fi_df = feature_importance_df.copy()
        # 确保有归一化列
        if "importance_gain_norm" not in fi_df.columns:
            total = fi_df["importance_gain"].sum()
            if total > 0:
                fi_df["importance_gain_norm"] = fi_df["importance_gain"] / total * 100
            else:
                fi_df["importance_gain_norm"] = 0.0

        # 按 gain 降序
        fi_df = fi_df.sort_values("importance_gain", ascending=False)

        # Top-10
        for _, row in fi_df.head(10).iterrows():
            top10.append({
                "feature": row["feature"],
                "gain": round(float(row["importance_gain"]), 1),
                "norm_pct": round(float(row["importance_gain_norm"]), 2),
            })

        # Bottom-10（importance > 0 的最后 10 个，排除 gain=0 的无效特征）
        nonzero = fi_df[fi_df["importance_gain"] > 0]
        if len(nonzero) >= 10:
            bottom_candidates = nonzero.tail(10)
        else:
            bottom_candidates = fi_df.tail(min(10, len(fi_df)))
        for _, row in bottom_candidates.iterrows():
            bottom10.append({
                "feature": row["feature"],
                "gain": round(float(row["importance_gain"]), 1),
                "norm_pct": round(float(row.get("importance_gain_norm", 0)), 2),
            })

    # ---- 当前指标 ----
    current_val_metrics = val_metrics or {}
    current_test_metrics = test_metrics or {}

    return FeatureAgentContext(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        sampling_frequency=sampling_frequency,
        target_col=target_col,
        time_col=time_col,
        history_window=history_window,
        prediction_horizon=prediction_horizon,
        target_mean=round(target_mean, 4),
        target_std=round(target_std, 4),
        target_min=round(target_min, 4),
        target_max=round(target_max, 4),
        target_cv=target_cv,
        target_q05=round(target_q05, 4),
        target_q95=round(target_q95, 4),
        acf_summary=acf_summary,
        has_daily_seasonality=ts_info["has_daily_seasonality"],
        has_weekly_seasonality=ts_info["has_weekly_seasonality"],
        trend_strength=ts_info["trend_strength"],
        seasonality_strength=ts_info["seasonality_strength"],
        current_features=current_features,
        feature_importance=fi_df,
        top10_features=top10,
        bottom10_features=bottom10,
        current_val_metrics=current_val_metrics,
        current_test_metrics=current_test_metrics,
    )


def build_iteration_context(
    ctx: FeatureAgentContext,
    iteration: int,
    previous_val_metrics: Optional[Dict[str, float]],
    previous_features_added: Optional[List[str]] = None,
    max_iterations: int = 5,
) -> FeatureAgentContext:
    """
    基于初始上下文构建迭代上下文。

    在当前上下文基础上叠加上一轮的指标 delta 和已添加特征，
    帮助 LLM 判断「上一轮的特征是否有用」。

    Parameters
    ----------
    ctx : FeatureAgentContext
        初始上下文
    iteration : int
        当前迭代轮次（从 1 开始）
    previous_val_metrics : dict or None
        上一轮验证集指标。若为 None 则无 delta
    previous_features_added : list of str, optional
        上一轮新增的特征列名
    max_iterations : int
        最大迭代轮次

    Returns
    -------
    FeatureAgentContext
        带有迭代信息的上下文副本
    """
    new_ctx = deepcopy(ctx)
    new_ctx.iteration = iteration
    new_ctx.max_iterations = max_iterations
    new_ctx.previous_val_metrics = previous_val_metrics
    new_ctx.previous_features_added = previous_features_added or []

    # ---- 计算指标 delta ----
    if previous_val_metrics and ctx.current_val_metrics:
        delta = {}
        for key in ctx.current_val_metrics:
            if key in previous_val_metrics:
                diff = round(
                    ctx.current_val_metrics[key] - previous_val_metrics[key], 4
                )
                delta[key] = diff
        new_ctx.metrics_delta = delta

        # 自动判断是否该停止
        rmse_delta = delta.get("RMSE", delta.get("val_RMSE", 0))
        if rmse_delta > 0 and abs(rmse_delta) < 1e-4:
            pass  # 几乎无变化，但不强制停止
        if iteration > 0 and abs(rmse_delta) < 0.001 and iteration >= 3:
            new_ctx.stop_reason = "指标连续多轮无明显改善"

    return new_ctx


# ============================================================
# 4. Prompt 模板 & 渲染
# ============================================================

_PROMPT_TEMPLATE = """你是一位电力负荷预测领域的特征工程专家。

## 任务
分析当前数据集和模型状态，提出 {max_new_features_min}~{max_new_features_max} 个新特征。
你的目标是**提升模型在验证集上的 RMSE 和 MAE**。

---

## 数据集信息
- 数据集: {dataset_name}
- 训练样本数: {n_samples}
- 采样频率: {sampling_frequency}
- 预测目标: {target_col}
- 预测步长: {prediction_horizon}
{history_window_line}

## 目标列统计
| 指标 | 值 |
|------|-----|
| 均值 | {target_mean} |
| 标准差 | {target_std} |
| 变异系数 (CV) | {target_cv} |
| 最小值 | {target_min} |
| 最大值 | {target_max} |
| P05 | {target_q05} |
| P95 | {target_q95} |

## 时序分析
{acf_table}

- 日周期性 (ACF lag=24): **{has_daily}**
- 周周期性 (ACF lag=168): **{has_weekly}**
- 趋势判断: **{trend_strength}**
- 季节性强度: **{seasonality_strength}**

## 现有特征
总特征数: {n_features}

{feature_list_block}

{feature_importance_block}

## 当前模型指标
{current_metrics_block}

{iteration_history_block}

## 可用的特征类型

你可以提议以下 4 种类型的特征，每种类型有严格的参数格式：

### 1. lag —— 滞后特征
对某列做时间偏移，捕获「历史值对未来的影响」。
JSON 格式:
```json
{{
  "name": "lag_24_LOAD",
  "type": "lag",
  "target_col": "{target_col}",
  "params": {{ "lag": 24 }}
}}
```
params: lag (int, >=1, 滞后步数)

### 2. rolling —— 滚动窗口统计
对某列做滑动窗口聚合，捕获局部趋势。
JSON 格式:
```json
{{
  "name": "rolling_mean_6_LOAD",
  "type": "rolling",
  "target_col": "{target_col}",
  "params": {{ "window": 6, "stat": "mean" }}
}}
```
params: window (int, >=2, 窗口大小), stat (str, 取值: mean, std, max, min, median, sum, skew, kurt)

### 3. time —— 时间特征
从时间列提取日期/时间分量。
JSON 格式:
```json
{{
  "name": "time_hour_sin",
  "type": "time",
  "time_col": "{time_col}",
  "params": {{ "features": ["hour", "dayofweek"], "cyclical": true }}
}}
```
params: features (可选, list[str], 取值: year, month, day, dayofweek, dayofyear, hour, minute, quarter, weekofyear, is_weekend, is_month_start, is_month_end), cyclical (可选, bool, 是否生成 sin/cos 周期性编码)

### 4. cross —— 交叉特征
对两列做算术运算。
JSON 格式:
```json
{{
  "name": "temp_mul_LOAD",
  "type": "cross",
  "params": {{ "col1": "temp", "col2": "LOAD", "operation": "multiply" }}
}}
```
params: col1 (str, 第一个操作列), col2 (str, 第二个操作列), operation (str, 取值: add, subtract, multiply, divide)

---

## 思考要求

1. **结合 ACF 分析**: 如果某个滞后的 ACF 很高（>0.5），考虑对该滞后生成 lag 和 rolling 特征
2. **结合特征重要性**: Top-10 中的重要特征值得组合出 cross；Bottom-10 中的弱特征可以尝试不同窗口的 rolling 来增强
3. **结合指标**: 如果当前 MAE 仍然较大，考虑加入更多历史平滑特征 (rolling_mean)
4. **避免冗余**: 不要生成与已有特征高度重复的新特征（如已有 lag_24，没必要再提 lag_24 或 lag_23）
5. **命名规范**: name 字段用简洁英文加描述性后缀，如 lag_72_load、rolling_std_6_temp、cross_temp_load
6. **确保列存在**: col1/col2/target_col/time_col 必须是数据集中已有的列名

---

## 严格输出格式

你必须**只输出**一个 JSON 对象，不要有任何其他文字。JSON 格式如下：

```json
{{
  "iteration": {iteration_num},
  "analysis": "用中文简要分析当前数据特征和你的决策依据（100-200字）",
  "new_features": [
    {{
      "name": "feature_name_here",
      "type": "lag",
      "target_col": "{target_col}",
      "params": {{ "lag": 24 }}
    }}
  ]
}}
```

**重要**: 你的回复必须是一个可被 json.loads() 直接解析的合法 JSON。不要加 markdown 代码块标记。不要有任何前缀或后缀文字。
"""


def build_llm_prompt(ctx: FeatureAgentContext) -> str:
    """
    将 FeatureAgentContext 渲染为 LLM prompt 字符串。

    Parameters
    ----------
    ctx : FeatureAgentContext

    Returns
    -------
    str
        完整的 prompt
    """
    # ---- 辅助格式化 ----

    # ACF 表格
    if ctx.acf_summary:
        acf_rows = []
        for k, v in sorted(ctx.acf_summary.items()):
            bar = _acf_bar(v)
            acf_rows.append(f"| lag={k:>4d} | {v:>7.4f} | {bar} |")
        acf_table = (
            "| 滞后 | ACF | 强度 |\n"
            "|------|-----|------|\n"
            + "\n".join(acf_rows)
        )
    else:
        acf_table = "（无 ACF 数据）"

    # 特征列表（简洁版：只列前 30 个）
    if len(ctx.current_features) <= 30:
        feature_list_block = "```\n" + ", ".join(ctx.current_features) + "\n```"
    else:
        feature_list_block = (
            f"```\n{', '.join(ctx.current_features[:30])}\n"
            f"... (还有 {len(ctx.current_features) - 30} 个)\n```"
        )

    # 特征重要性
    if ctx.top10_features:
        top10_lines = []
        for f in ctx.top10_features:
            top10_lines.append(
                f"  {f['feature']:<30s} gain={f['gain']:>12.1f} ({f['norm_pct']:>6.2f}%)"
            )
        bottom10_lines = []
        for f in ctx.bottom10_features:
            bottom10_lines.append(
                f"  {f['feature']:<30s} gain={f['gain']:>12.1f} ({f['norm_pct']:>6.2f}%)"
            )
        feature_importance_block = (
            "### Top-10 特征重要性\n```\n"
            + "\n".join(top10_lines)
            + "\n```\n\n"
            "### Bottom-10 特征重要性\n```\n"
            + "\n".join(bottom10_lines)
            + "\n```"
        )
    else:
        feature_importance_block = "（无特征重要性数据）"

    # 当前指标
    metrics_block_parts = []
    if ctx.current_val_metrics:
        metrics_block_parts.append("#### 验证集\n```")
        for k, v in ctx.current_val_metrics.items():
            metrics_block_parts.append(f"  {k}: {v}")
        metrics_block_parts.append("```")
    if ctx.current_test_metrics:
        metrics_block_parts.append("#### 测试集\n```")
        for k, v in ctx.current_test_metrics.items():
            metrics_block_parts.append(f"  {k}: {v}")
        metrics_block_parts.append("```")
    current_metrics_block = (
        "\n".join(metrics_block_parts) if metrics_block_parts else "（无指标数据）"
    )

    # 迭代历史
    iteration_history_block = ""
    if ctx.iteration > 0:
        iteration_history_block = f"""## 迭代历史
当前是第 **{ctx.iteration}** / {ctx.max_iterations} 轮特征迭代。

"""

        if ctx.previous_features_added:
            iteration_history_block += (
                "上一轮新增特征:\n```\n"
                + "\n".join(f"  - {f}" for f in ctx.previous_features_added)
                + "\n```\n\n"
            )

        if ctx.metrics_delta:
            iteration_history_block += "指标变化 (当前 - 上一轮):\n```\n"
            for k, v in ctx.metrics_delta.items():
                direction = "↓改善" if v < 0 else ("↑恶化" if v > 0 else "→不变")
                iteration_history_block += f"  Δ{k}: {v:+.4f} {direction}\n"
            iteration_history_block += "```\n\n"

            # 智能建议
            rmse_key = next(
                (k for k in ctx.metrics_delta if "RMSE" in k.upper()), None
            )
            if rmse_key and ctx.metrics_delta[rmse_key] > 0.5:
                iteration_history_block += (
                    "⚠️ **上一轮特征添加后 RMSE 反而上升，"
                    "本轮请尝试不同类型的特征或更保守的窗口大小。**\n\n"
                )
            elif rmse_key and ctx.metrics_delta[rmse_key] < -0.5:
                iteration_history_block += (
                    "✅ **上一轮特征添加后 RMSE 明显改善，"
                    "本轮可以继续在相似方向深化（如扩大窗口范围）。**\n\n"
                )

        if ctx.stop_reason:
            iteration_history_block += (
                f"⚠️ 停止建议: {ctx.stop_reason}\n"
                "如果本轮没有新的特征思路，可以输出空 new_features 列表。\n\n"
            )

    prompt = _PROMPT_TEMPLATE.format(
        dataset_name=ctx.dataset_name or "电力负荷数据集",
        n_samples=ctx.n_samples,
        sampling_frequency=ctx.sampling_frequency,
        target_col=ctx.target_col,
        prediction_horizon=ctx.prediction_horizon,
        history_window_line=(
            f"- 历史窗口: {ctx.history_window} 步"
            if ctx.history_window is not None
            else ""
        ),
        target_mean=ctx.target_mean,
        target_std=ctx.target_std,
        target_cv=ctx.target_cv,
        target_min=ctx.target_min,
        target_max=ctx.target_max,
        target_q05=ctx.target_q05,
        target_q95=ctx.target_q95,
        acf_table=acf_table,
        has_daily="✅ 显著" if ctx.has_daily_seasonality else "❌ 不显著",
        has_weekly="✅ 显著" if ctx.has_weekly_seasonality else "❌ 不显著",
        trend_strength=ctx.trend_strength,
        seasonality_strength=ctx.seasonality_strength,
        n_features=ctx.n_features,
        feature_list_block=feature_list_block,
        feature_importance_block=feature_importance_block,
        current_metrics_block=current_metrics_block,
        iteration_history_block=iteration_history_block,
        iteration_num=ctx.iteration,
        max_new_features_min=2,
        max_new_features_max=5,
        time_col=ctx.time_col,
    )

    return prompt


def _acf_bar(value: float, max_width: int = 20) -> str:
    """生成 ACF 的可视化条形。"""
    if value == 0:
        return "·"
    abs_val = abs(value)
    width = int(abs_val * max_width)
    if width == 0:
        width = 1
    bar = "█" * width
    sign = "+" if value > 0 else "-"
    return f"{sign}{bar}"


# ============================================================
# 5. 输出 JSON Schema 定义
# ============================================================

# 每个特征类型的 params schema
_LAG_PARAMS_SCHEMA = {
    "type": "object",
    "required": ["lag"],
    "properties": {
        "lag": {
            "type": "integer",
            "minimum": 1,
            "description": "滞后步数",
        },
    },
    "additionalProperties": False,
}

_ROLLING_PARAMS_SCHEMA = {
    "type": "object",
    "required": ["window", "stat"],
    "properties": {
        "window": {
            "type": "integer",
            "minimum": 2,
            "description": "滚动窗口大小",
        },
        "stat": {
            "type": "string",
            "enum": ["mean", "std", "var", "max", "min", "median", "sum", "skew", "kurt"],
            "description": "统计量",
        },
    },
    "additionalProperties": False,
}

_TIME_PARAMS_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "year", "month", "day", "dayofweek", "dayofyear",
                    "hour", "minute", "quarter", "weekofyear",
                    "is_weekend", "is_month_start", "is_month_end",
                ],
            },
            "description": "要提取的时间特征列表",
        },
        "cyclical": {
            "type": "boolean",
            "description": "是否生成 sin/cos 周期性编码",
        },
    },
    "additionalProperties": False,
}

_CROSS_PARAMS_SCHEMA = {
    "type": "object",
    "required": ["col1", "col2", "operation"],
    "properties": {
        "col1": {"type": "string", "minLength": 1},
        "col2": {"type": "string", "minLength": 1},
        "operation": {
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
        },
    },
    "additionalProperties": False,
}

# 单个特征项 schema
_FEATURE_ITEM_SCHEMA = {
    "type": "object",
    "required": ["name", "type", "params"],
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 120,
            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
            "description": "特征名（英文、下划线、数字）",
        },
        "type": {
            "type": "string",
            "enum": ["lag", "rolling", "time", "cross"],
            "description": "特征类型",
        },
        "target_col": {
            "type": "string",
            "description": "仅 lag/rolling 类型需要",
        },
        "time_col": {
            "type": "string",
            "description": "仅 time 类型需要",
        },
        "params": {
            "type": "object",
            "description": "类型专属参数",
        },
    },
    "additionalProperties": False,
}

# 完整输出 schema
FEATURE_AGENT_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FeatureAgentOutput",
    "description": "LLM 特征工程 Agent 的严格输出格式",
    "type": "object",
    "required": ["iteration", "analysis", "new_features"],
    "properties": {
        "iteration": {
            "type": "integer",
            "minimum": 0,
            "description": "迭代轮次",
        },
        "analysis": {
            "type": "string",
            "minLength": 10,
            "maxLength": 1000,
            "description": "分析文本（中文，100-200字）",
        },
        "new_features": {
            "type": "array",
            "maxItems": 10,
            "items": _FEATURE_ITEM_SCHEMA,
            "description": "本轮新增特征列表（可为空数组表示停止）",
        },
    },
    "additionalProperties": False,
}


# ============================================================
# 6. 输出校验 & 解析
# ============================================================

# type → 额外必填字段
_TYPE_EXTRA_REQUIRED = {
    "lag": ["target_col"],
    "rolling": ["target_col"],
    "time": [],          # time_col 可选，默认为 context 中的 time_col
    "cross": [],         # col1/col2 在 params 里
}

# type → params 具体 schema
_TYPE_PARAMS_SCHEMA = {
    "lag": _LAG_PARAMS_SCHEMA,
    "rolling": _ROLLING_PARAMS_SCHEMA,
    "time": _TIME_PARAMS_SCHEMA,
    "cross": _CROSS_PARAMS_SCHEMA,
}


def validate_llm_output(
    raw: Union[str, dict],
    available_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    校验 LLM 输出的 JSON，返回标准化字典。

    校验规则：
      1. 合法的 JSON 结构
      2. 顶层必填字段: iteration, analysis, new_features
      3. 每个 feature 项: name, type, params 必填
      4. type 对应的额外必填字段 (target_col, time_col)
      5. params 符合各类型的子 schema
      6. （可选）引用的列名在 available_columns 中存在

    Parameters
    ----------
    raw : str or dict
        LLM 的原始输出（字符串会被解析为 JSON）
    available_columns : list of str, optional
        当前数据集所有可用列名，用于校验列引用

    Returns
    -------
    dict
        通过校验的标准化输出

    Raises
    ------
    ValueError
        校验失败时的详细错误信息
    """
    # Step 1: 解析 JSON
    if isinstance(raw, str):
        raw = _extract_json(raw)

    if not isinstance(raw, dict):
        raise ValueError(f"输出必须是 JSON 对象，实际类型: {type(raw).__name__}")

    # Step 2: 顶层字段检查
    missing_top = [
        k for k in ["iteration", "analysis", "new_features"] if k not in raw
    ]
    if missing_top:
        raise ValueError(f"缺少顶层必填字段: {missing_top}")

    if not isinstance(raw["new_features"], list):
        raise ValueError(
            f"new_features 必须是数组，实际类型: {type(raw['new_features']).__name__}"
        )

    if len(raw["new_features"]) > 10:
        raise ValueError(
            f"new_features 最多 10 个，实际: {len(raw['new_features'])}"
        )

    # Step 3: 逐项校验
    validated_features = []
    errors = []

    for i, feat in enumerate(raw["new_features"]):
        try:
            vf = _validate_single_feature(feat, i, available_columns)
            validated_features.append(vf)
        except ValueError as e:
            errors.append(str(e))

    if errors:
        raise ValueError(
            f"特征校验失败 ({len(errors)} 项):\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return {
        "iteration": int(raw["iteration"]),
        "analysis": str(raw["analysis"]),
        "new_features": validated_features,
    }


def _validate_single_feature(
    feat: dict,
    index: int,
    available_columns: Optional[List[str]] = None,
) -> dict:
    """校验单个特征项。"""
    prefix = f"new_features[{index}]"

    # 基础字段
    if not isinstance(feat, dict):
        raise ValueError(f"{prefix}: 必须是 JSON 对象")

    for field in ["name", "type", "params"]:
        if field not in feat:
            raise ValueError(f"{prefix}: 缺少必填字段 '{field}'")

    feat_type = feat["type"]
    if feat_type not in _TYPE_PARAMS_SCHEMA:
        raise ValueError(
            f"{prefix}: 不支持的特征类型 '{feat_type}'，"
            f"支持: {list(_TYPE_PARAMS_SCHEMA.keys())}"
        )

    # name 校验
    name = feat["name"]
    if not isinstance(name, str) or len(name) < 1 or len(name) > 120:
        raise ValueError(
            f"{prefix}: name 必须是 1-120 字符的字符串，实际: '{name}'"
        )
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ValueError(
            f"{prefix}: name '{name}' 格式非法，"
            f"必须以字母或下划线开头，只含字母、数字、下划线"
        )

    # 类型专属必填字段
    extra_required = _TYPE_EXTRA_REQUIRED.get(feat_type, [])
    for field in extra_required:
        if not feat.get(field):
            raise ValueError(
                f"{prefix}: 类型 '{feat_type}' 需要 '{field}' 字段"
            )

    # params 校验
    params = feat["params"]
    params_schema = _TYPE_PARAMS_SCHEMA[feat_type]
    _validate_params(params, params_schema, f"{prefix}.params")

    # 列引用校验
    if available_columns:
        col_set = set(available_columns)
        # target_col
        if feat_type in ("lag", "rolling") and feat.get("target_col"):
            tc = feat["target_col"]
            if tc not in col_set:
                raise ValueError(
                    f"{prefix}: target_col '{tc}' 不在可用列中。"
                    f"可用: {sorted(col_set)[:20]}..."
                )
        # time_col
        if feat_type == "time" and feat.get("time_col"):
            tc = feat["time_col"]
            if tc not in col_set:
                raise ValueError(
                    f"{prefix}: time_col '{tc}' 不在可用列中"
                )
        # cross: col1, col2 在 params 里
        if feat_type == "cross":
            for c in ["col1", "col2"]:
                if c in params and params[c] not in col_set:
                    raise ValueError(
                        f"{prefix}: {c} '{params[c]}' 不在可用列中"
                    )

    # 构建标准化输出
    result = {
        "name": name,
        "type": feat_type,
        "params": params,
    }
    if feat_type in ("lag", "rolling") and feat.get("target_col"):
        result["target_col"] = feat["target_col"]
    if feat_type == "time" and feat.get("time_col"):
        result["time_col"] = feat["time_col"]

    return result


def _validate_params(params: dict, schema: dict, path: str) -> None:
    """校验 params 是否符合子 schema。"""
    if not isinstance(params, dict):
        raise ValueError(f"{path}: params 必须是 JSON 对象")

    # 必填字段
    for req in schema.get("required", []):
        if req not in params:
            raise ValueError(f"{path}: 缺少必填字段 '{req}'")

    # 非法字段
    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}).keys())
        extra = set(params.keys()) - allowed
        if extra:
            raise ValueError(
                f"{path}: 包含不支持的字段: {extra}。"
                f"允许: {sorted(allowed)}"
            )

    # 字段值校验
    for key, val in params.items():
        if key not in schema.get("properties", {}):
            continue
        prop = schema["properties"][key]

        # 类型校验
        expected_type = prop.get("type")
        if expected_type:
            type_ok = False
            if expected_type == "string":
                type_ok = isinstance(val, str)
            elif expected_type == "integer":
                type_ok = isinstance(val, int) and not isinstance(val, bool)
            elif expected_type == "number":
                type_ok = isinstance(val, (int, float)) and not isinstance(val, bool)
            elif expected_type == "boolean":
                type_ok = isinstance(val, bool)
            elif expected_type == "array":
                type_ok = isinstance(val, list)

            if not type_ok:
                raise ValueError(
                    f"{path}.{key}: 期望类型 {expected_type}，"
                    f"实际: {type(val).__name__} (值: {val})"
                )

        # 数值范围
        if "minimum" in prop and isinstance(val, (int, float)):
            if val < prop["minimum"]:
                raise ValueError(
                    f"{path}.{key}: 值 {val} < 最小值 {prop['minimum']}"
                )
        if "maximum" in prop and isinstance(val, (int, float)):
            if val > prop["maximum"]:
                raise ValueError(
                    f"{path}.{key}: 值 {val} > 最大值 {prop['maximum']}"
                )

        # 枚举
        if "enum" in prop and val not in prop["enum"]:
            raise ValueError(
                f"{path}.{key}: '{val}' 不在允许值中: {prop['enum']}"
            )

        # 字符串长度
        if expected_type == "string":
            if "minLength" in prop and len(val) < prop["minLength"]:
                raise ValueError(
                    f"{path}.{key}: 字符串长度 {len(val)} < 最小 {prop['minLength']}"
                )
            if "maxLength" in prop and len(val) > prop["maxLength"]:
                raise ValueError(
                    f"{path}.{key}: 字符串长度 {len(val)} > 最大 {prop['maxLength']}"
                )
            if "pattern" in prop and not re.match(prop["pattern"], val):
                raise ValueError(
                    f"{path}.{key}: '{val}' 不符合正则: {prop['pattern']}"
                )


def _extract_json(raw: str) -> Union[dict, list, str]:
    """
    从 LLM 原始文本中提取 JSON 对象。

    容忍以下情况：
      - markdown ```json ... ``` 包裹
      - 前后有额外文字
      - 单引号替代双引号
    """
    original = raw
    raw = raw.strip()

    # 情况 1: markdown 代码块
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if code_block_match:
        raw = code_block_match.group(1).strip()

    # 情况 2: 找到第一个 { 和最后一个 }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        raw = raw[first_brace : last_brace + 1]

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试替换单引号为双引号
    try:
        # 简单替换（不处理嵌套引号）
        fixed = raw.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 最终失败 - 返回原始字符串，让上层报清晰的错误
    raise ValueError(
        f"无法从 LLM 输出中解析 JSON。原始输出前 300 字符:\n{original[:300]}"
    )


# ============================================================
# 7. LLM 决策 → feature_engine 执行桥接
# ============================================================

def execute_features_from_llm(
    df: pd.DataFrame,
    llm_output: dict,
    default_target_col: str = "LOAD",
    default_time_col: str = "datetime",
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    将 LLM 输出的特征决策翻译为 feature_engine 调用并执行。

    这是 LLM 决策到执行引擎的桥接层——
    LLM 说「我想要 lag_72_load (type=lag, params={lag:72})」，
    本函数就调用 generate_lag_features(df, "LOAD", [72])。

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame
    llm_output : dict
        经过 validate_llm_output 校验的输出
    default_target_col : str
        默认目标列（lag/rolling 类型若未指定 target_col 时使用）
    default_time_col : str
        默认时间列（time 类型若未指定 time_col 时使用）

    Returns
    -------
    (new_df, added_cols, skipped_features)
        - new_df: 加入新特征后的 DataFrame
        - added_cols: 成功添加的列名列表
        - skipped_features: 因报错而被跳过的特征描述列表
    """
    df_out = df.copy()
    added_cols = []
    skipped = []

    for feat in llm_output.get("new_features", []):
        feat_type = feat["type"]
        params = feat.get("params", {})
        col_before = set(df_out.columns)

        try:
            if feat_type == "lag":
                target_col = feat.get("target_col", default_target_col)
                lag = params["lag"]
                df_out = generate_lag_features(
                    df_out, target_col, [lag],
                )

            elif feat_type == "rolling":
                target_col = feat.get("target_col", default_target_col)
                window = params["window"]
                stat = params["stat"]
                df_out = generate_rolling_features(
                    df_out, target_col, [window], stats=[stat],
                )

            elif feat_type == "time":
                time_col = feat.get("time_col", default_time_col)
                features = params.get("features", None)
                cyclical = params.get("cyclical", True)
                df_out = generate_time_features(
                    df_out, time_col, features=features, cyclical=cyclical,
                )

            elif feat_type == "cross":
                col1 = params["col1"]
                col2 = params["col2"]
                operation = params["operation"]
                df_out = generate_cross_features(
                    df_out, col1, col2, operation,
                )

            # 记录新增列
            new_cols = [c for c in df_out.columns if c not in col_before]
            added_cols.extend(new_cols)

        except Exception as e:
            skipped.append(f"{feat.get('name', '?')} ({feat_type}): {e}")
            # 回滚 df（此特征生成失败，继续处理下一个）
            df_out = df.copy() if len(added_cols) == 0 else df_out
            # 只回滚本次尝试的列
            for c in set(df_out.columns) - col_before:
                if c in df_out.columns:
                    df_out = df_out.drop(columns=[c])

    return df_out, added_cols, skipped


# ============================================================
# 8. 迭代历史追踪器
# ============================================================

class FeatureIterationHistory:
    """
    追踪多轮特征工程的迭代历史。

    用法：
        history = FeatureIterationHistory()
        history.record(ctx_before, llm_output, added_cols, new_metrics)
        summary = history.summary()
    """

    def __init__(self):
        self.records: List[Dict] = []

    def record(
        self,
        iteration: int,
        llm_output: dict,
        added_columns: List[str],
        skipped_features: List[str],
        val_metrics_before: Optional[Dict[str, float]],
        val_metrics_after: Optional[Dict[str, float]],
        analysis: str = "",
    ):
        """记录一轮迭代。"""
        delta = {}
        if val_metrics_before and val_metrics_after:
            for k in val_metrics_after:
                if k in val_metrics_before:
                    delta[k] = round(
                        val_metrics_after[k] - val_metrics_before[k], 4
                    )

        self.records.append({
            "iteration": iteration,
            "analysis": analysis,
            "n_features_proposed": len(llm_output.get("new_features", [])),
            "n_features_added": len(added_columns),
            "n_features_skipped": len(skipped_features),
            "added_columns": added_columns,
            "skipped_features": skipped_features,
            "metrics_delta": delta,
            "val_metrics_before": val_metrics_before,
            "val_metrics_after": val_metrics_after,
        })

    def summary(self) -> pd.DataFrame:
        """返回迭代历史的 DataFrame 概览。"""
        rows = []
        for r in self.records:
            delta_str = ", ".join(
                f"{k}:{v:+.4f}" for k, v in r["metrics_delta"].items()
            )
            rows.append({
                "iteration": r["iteration"],
                "proposed": r["n_features_proposed"],
                "added": r["n_features_added"],
                "skipped": r["n_features_skipped"],
                "metrics_delta": delta_str,
                "added_cols": ", ".join(r["added_columns"][:5]),
            })
        return pd.DataFrame(rows)

    def best_iteration(self) -> Optional[int]:
        """返回 RMSE 改善最大的迭代轮次。"""
        best_iter = None
        best_delta = float("inf")
        for r in self.records:
            delta = r["metrics_delta"]
            rmse_key = next(
                (k for k in delta if "RMSE" in k.upper()), None
            )
            if rmse_key and delta[rmse_key] < best_delta:
                best_delta = delta[rmse_key]
                best_iter = r["iteration"]
        return best_iter


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    # 构造模拟数据
    dates = pd.date_range("2020-01-01", periods=500, freq="h")
    n = len(dates)
    df = pd.DataFrame({
        "datetime": dates,
        "LOAD": (
            500
            + 50 * np.sin(2 * np.pi * np.arange(n) / 24)
            + 30 * np.sin(2 * np.pi * np.arange(n) / 168)
            + np.random.normal(0, 15, n)
        ),
        "temp": (
            20 + 10 * np.sin(2 * np.pi * np.arange(n) / 24)
            + np.random.normal(0, 3, n)
        ),
        "humidity": (
            60 + 15 * np.sin(2 * np.pi * np.arange(n) / 48)
            + np.random.normal(0, 5, n)
        ),
    })

    # 模拟特征重要性
    feat_imp_df = pd.DataFrame({
        "feature": ["lag_1", "temp", "hour", "lag_24", "lag_168",
                      "temp_lag_24", "rolling_mean_24", "weekday",
                      "humidity", "month", "is_weekend"],
        "importance_gain": [
            1_500_000, 300_000, 100_000, 80_000, 50_000,
            40_000, 30_000, 20_000, 15_000, 5_000, 500,
        ],
    })

    # 模拟当前指标
    val_metrics = {"RMSE": 8.53, "MAE": 6.70, "MAPE": 4.97}

    print("=" * 60)
    print("1. ACF 分析测试")
    print("=" * 60)
    acf = compute_acf_summary(df, "LOAD")
    for k, v in sorted(acf.items()):
        print(f"  lag={k:>4d}: {v:>7.4f}  {_acf_bar(v)}")

    print()
    print("=" * 60)
    print("2. 趋势/季节性检测测试")
    print("=" * 60)
    ts_info = detect_trend_seasonality(df, "LOAD", acf)
    for k, v in ts_info.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 60)
    print("3. 上下文构建测试")
    print("=" * 60)
    ctx = build_context_from_data(
        df,
        target_col="LOAD",
        time_col="datetime",
        feature_importance_df=feat_imp_df,
        current_features=["lag_1", "temp", "hour", "lag_24", "lag_168",
                          "temp_lag_24", "rolling_mean_24", "weekday",
                          "humidity", "month", "is_weekend"],
        val_metrics=val_metrics,
        dataset_name="GEFCom2014 Task 15",
        history_window=24,
    )
    print(f"  样本数: {ctx.n_samples}")
    print(f"  特征数: {ctx.n_features}")
    print(f"  采样频率: {ctx.sampling_frequency}")
    print(f"  目标 CV: {ctx.target_cv}")
    print(f"  日周期: {ctx.has_daily_seasonality}")
    print(f"  周周期: {ctx.has_weekly_seasonality}")
    print(f"  趋势: {ctx.trend_strength}")
    print(f"  ACF_24: {ctx.acf_summary.get(24, 'N/A')}")
    print(f"  ACF_168: {ctx.acf_summary.get(168, 'N/A')}")
    print(f"  Top-3 特征: {[f['feature'] for f in ctx.top10_features[:3]]}")

    print()
    print("=" * 60)
    print("4. Prompt 渲染测试")
    print("=" * 60)
    prompt = build_llm_prompt(ctx)
    print(f"  Prompt 长度: {len(prompt)} 字符")
    print(f"  前 500 字符:\n{prompt[:500]}")
    print("  ...")

    print()
    print("=" * 60)
    print("5. LLM 输出校验测试")
    print("=" * 60)

    # 模拟正确的 LLM 输出
    correct_output = {
        "iteration": 1,
        "analysis": "当前日周期性显著(ACF_24=0.xx)，温度与负荷相关性强。"
                     "建议增加温度滞后特征和峰谷统计特征，同时利用ACF_72的高相关性"
                     "来捕获3天周期的规律。",
        "new_features": [
            {
                "name": "lag_72_load",
                "type": "lag",
                "target_col": "LOAD",
                "params": {"lag": 72},
            },
            {
                "name": "rolling_max_24_load",
                "type": "rolling",
                "target_col": "LOAD",
                "params": {"window": 24, "stat": "max"},
            },
            {
                "name": "cross_temp_load",
                "type": "cross",
                "params": {"col1": "temp", "col2": "LOAD", "operation": "multiply"},
            },
            {
                "name": "time_hour_cyclical",
                "type": "time",
                "time_col": "datetime",
                "params": {"features": ["hour"], "cyclical": True},
            },
        ],
    }

    available_cols = list(df.columns) + ["lag_1", "lag_24", "lag_168",
                                          "temp_lag_24", "rolling_mean_24",
                                          "hour", "weekday", "month",
                                          "humidity", "is_weekend"]
    validated = validate_llm_output(correct_output, available_columns=available_cols)
    print(f"  校验通过! iteration={validated['iteration']}")
    print(f"  analysis 前 60 字: {validated['analysis'][:60]}...")
    print(f"  new_features 数量: {len(validated['new_features'])}")
    for f in validated["new_features"]:
        print(f"    - {f['name']} ({f['type']}): {f['params']}")

    print()
    print("=" * 60)
    print("6. LLM 输出校验 — 错误处理测试")
    print("=" * 60)

    # 6a. 缺失字段
    try:
        validate_llm_output({"iteration": 1}, available_columns=available_cols)
    except ValueError as e:
        print(f"  [OK] 缺失字段: {str(e)[:80]}")

    # 6b. 非法类型
    bad_output = {
        "iteration": 1,
        "analysis": "test " * 5,
        "new_features": [{
            "name": "bad_feat",
            "type": "unknown_type",
            "params": {},
        }],
    }
    try:
        validate_llm_output(bad_output)
    except ValueError as e:
        print(f"  [OK] 非法类型: {str(e)[:80]}")

    # 6c. 非法 name 格式
    bad_output2 = {
        "iteration": 1,
        "analysis": "test " * 5,
        "new_features": [{
            "name": "123bad_name",
            "type": "lag",
            "target_col": "LOAD",
            "params": {"lag": 1},
        }],
    }
    try:
        validate_llm_output(bad_output2)
    except ValueError as e:
        print(f"  [OK] 非法 name: {str(e)[:80]}")

    # 6d. lag 缺少 target_col
    bad_output3 = {
        "iteration": 1,
        "analysis": "test " * 5,
        "new_features": [{
            "name": "missing_target",
            "type": "lag",
            "params": {"lag": 1},
        }],
    }
    try:
        validate_llm_output(bad_output3)
    except ValueError as e:
        print(f"  [OK] 缺少 target_col: {str(e)[:80]}")

    # 6e. 列不存在
    bad_output4 = {
        "iteration": 1,
        "analysis": "test " * 5,
        "new_features": [{
            "name": "bad_col",
            "type": "lag",
            "target_col": "NONEXISTENT",
            "params": {"lag": 1},
        }],
    }
    try:
        validate_llm_output(bad_output4, available_columns=available_cols)
    except ValueError as e:
        print(f"  [OK] 列不存在: {str(e)[:80]}")

    # 6f. JSON 提取（markdown 代码块包裹）
    md_wrapped = '```json\n{"iteration": 1, "analysis": "test analysis here", "new_features": []}\n```'
    extracted = _extract_json(md_wrapped)
    print(f"  [OK] Markdown 提取: {type(extracted).__name__} with {len(extracted)} keys")

    print()
    print("=" * 60)
    print("7. 执行桥接测试")
    print("=" * 60)
    df_new, added, skipped = execute_features_from_llm(
        df,
        validated,
        default_target_col="LOAD",
        default_time_col="datetime",
    )
    print(f"  新增列: {added}")
    print(f"  跳过: {skipped}")
    print(f"  df 列数: {len(df.columns)} → {len(df_new.columns)}")

    # 验证新增列确实存在且有效
    for col in added:
        assert col in df_new.columns, f"列 {col} 应该被添加"
        non_null = df_new[col].notna().sum()
        print(f"    {col}: {non_null}/{len(df_new)} 非空")

    print()
    print("=" * 60)
    print("8. 迭代历史追踪测试")
    print("=" * 60)
    history = FeatureIterationHistory()
    history.record(
        iteration=0,
        llm_output={"new_features": []},
        added_columns=["lag_24", "lag_168"],
        skipped_features=[],
        val_metrics_before={"RMSE": 10.0, "MAE": 8.0},
        val_metrics_after={"RMSE": 8.5, "MAE": 6.7},
        analysis="baseline",
    )
    history.record(
        iteration=1,
        llm_output=correct_output,
        added_columns=added,
        skipped_features=[],
        val_metrics_before={"RMSE": 8.5, "MAE": 6.7},
        val_metrics_after={"RMSE": 8.2, "MAE": 6.5},
        analysis="",
    )
    print(history.summary().to_string())
    print(f"  Best iteration: {history.best_iteration()}")

    print()
    print("=" * 60)
    print("9. 迭代上下文测试")
    print("=" * 60)
    ctx_iter2 = build_iteration_context(
        ctx,
        iteration=2,
        previous_val_metrics={"RMSE": 8.5, "MAE": 6.7},
        previous_features_added=["lag_72_load", "rolling_max_24_load"],
    )
    print(f"  iteration: {ctx_iter2.iteration}")
    print(f"  metrics_delta: {ctx_iter2.metrics_delta}")
    print(f"  previous_features_added: {ctx_iter2.previous_features_added}")

    prompt2 = build_llm_prompt(ctx_iter2)
    print(f"  迭代 prompt 长度: {len(prompt2)} 字符")
    # 验证迭代历史块出现在 prompt 中
    assert "迭代历史" in prompt2, "迭代 prompt 应包含迭代历史"
    assert "Δ" in prompt2, "迭代 prompt 应包含 delta"
    print("  [OK] 迭代 prompt 包含历史信息")

    print()
    print("=" * 60)
    print("全部测试通过！")
    print("=" * 60)
