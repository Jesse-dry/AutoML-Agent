# GEFCom Task 构建器：无泄漏特征工程 + train/val 切分
# ---------------------------------------------------------------
# 职责：
#   1. 血缘式 FEATURE_SPEC —— 每个特征带 lineage 元数据
#      （source / operation / lookback_start / lookback_end / uses_current_target），
#      这是 leakage_checker 静态检查与未来 LLM Agent 输出
#      "Feature + Lineage + Availability" 的技术基础。
#   2. build_features —— 唯一严格过去向特征构造器，训练与预测窗口共用，
#      保证特征空间一致（无温度特征，LOAD-only protocol）。
#   3. build_task —— 在"整段可用历史"上一次性构建特征（修复旧实现
#      val/test 丢失 lag/rolling 预热上下文的问题），再切 train/val。
#
# 因果性约定：
#   - lag_k      = LOAD.shift(k)，窗口 [t-k, t-k]
#   - rolling_*  = LOAD.shift(1).rolling(window, min_periods=window).stat()
#                 窗口严格为 [t-window .. t-1]，不含当前行（无自泄露）
#   - time       = 目标小时的日历属性，已知，无数据依赖
# ---------------------------------------------------------------
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from data.availability import Availability, available_history
from data.gefcom_loader import GEFCOM_DATA_DIR, load_ground_truth

TARGET_COL = "LOAD"
MAX_LAG = 168  # lag_168 / rolling_168 所需最大回看

# 血缘式特征规格（顺序即特征列顺序）
FEATURE_SPEC: List[dict] = [
    {"name": "hour", "type": "time", "attr": "hour",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "weekday", "type": "time", "attr": "weekday",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "month", "type": "time", "attr": "month",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "is_weekend", "type": "time", "attr": "is_weekend",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "lag_1", "type": "lag", "source": "LOAD", "k": 1,
     "lookback_start": -1, "lookback_end": -1, "uses_current_target": False},
    {"name": "lag_24", "type": "lag", "source": "LOAD", "k": 24,
     "lookback_start": -24, "lookback_end": -24, "uses_current_target": False},
    {"name": "lag_168", "type": "lag", "source": "LOAD", "k": 168,
     "lookback_start": -168, "lookback_end": -168, "uses_current_target": False},
    {"name": "rolling_mean_24", "type": "rolling", "source": "LOAD",
     "window": 24, "stat": "mean", "min_periods": 24,
     "lookback_start": -24, "lookback_end": -1, "uses_current_target": False},
    {"name": "rolling_std_24", "type": "rolling", "source": "LOAD",
     "window": 24, "stat": "std", "min_periods": 24,
     "lookback_start": -24, "lookback_end": -1, "uses_current_target": False},
    {"name": "rolling_mean_168", "type": "rolling", "source": "LOAD",
     "window": 168, "stat": "mean", "min_periods": 168,
     "lookback_start": -168, "lookback_end": -1, "uses_current_target": False},
]

# 冷启动基线（砍基线）：仅 time + lag_1/24，不含 rolling / lag_168。
# 作为自进化 Agent 的极简起点，让 LLM 用三档动作空间重新发现滚动统计、
# 周滞后等特征，放大增益对比。全局 FEATURE_SPEC 不变（replay 基线数字保持）。
COLD_START_FEATURE_SPEC: List[dict] = [
    {"name": "hour", "type": "time", "attr": "hour",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "weekday", "type": "time", "attr": "weekday",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "month", "type": "time", "attr": "month",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "is_weekend", "type": "time", "attr": "is_weekend",
     "lookback_start": 0, "lookback_end": 0, "uses_current_target": False},
    {"name": "lag_1", "type": "lag", "source": "LOAD", "k": 1,
     "lookback_start": -1, "lookback_end": -1, "uses_current_target": False},
    {"name": "lag_24", "type": "lag", "source": "LOAD", "k": 24,
     "lookback_start": -24, "lookback_end": -24, "uses_current_target": False},
]

FEATURE_COLS: List[str] = [s["name"] for s in FEATURE_SPEC]

DEFAULT_VAL_HOURS = 168


def feature_spec_hash(spec: List[dict] = FEATURE_SPEC) -> str:
    """特征规格的稳定哈希（用于 run_manifest 审计）。"""
    canonical = json.dumps(spec, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------
# cross 特征操作（build_features 与 evaluation/rolling_backtest._features_at
# 共用同一实现，保证 T5 parity 逐位一致）
# ---------------------------------------------------------------

def _safe_div(a, b):
    """除零安全除法：inf/-inf → NaN（LightGBM 无法处理 inf）。"""
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            res = a / b
    except ZeroDivisionError:
        return np.nan
    if isinstance(res, pd.Series):
        return res.replace([np.inf, -np.inf], np.nan)
    if isinstance(res, (float, np.floating)) and (np.isinf(res) or np.isnan(res)):
        return np.nan
    return res


_CROSS_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": _safe_div,
}


def build_features(
    df: pd.DataFrame,
    spec: List[dict] = FEATURE_SPEC,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    严格过去向特征构造器（Stateless）：df 须以 datetime 为索引，含 target 列。

    返回值：仅含特征列的 DataFrame（索引与 df 一致）。
    lag/rolling 只依赖 ≤ t-1 的信息；time 特征无数据依赖。
    """
    out = pd.DataFrame(index=df.index)
    for s in spec:
        name, stype = s["name"], s["type"]
        if stype == "time":
            idx = df.index
            attr = s["attr"]
            if attr == "hour":
                out[name] = idx.hour
            elif attr == "weekday":
                out[name] = idx.weekday  # 0=Mon .. 6=Sun
            elif attr == "month":
                out[name] = idx.month
            elif attr == "is_weekend":
                out[name] = (idx.weekday >= 5).astype(int)
            else:
                raise ValueError(f"未知 time 属性: {attr}")
        elif stype == "lag":
            out[name] = df[s["source"]].shift(s["k"])
        elif stype == "rolling":
            src = df[s["source"]].shift(1)  # 严格过去窗口：先 shift 再 rolling
            roll = src.rolling(window=s["window"], min_periods=s["min_periods"])
            out[name] = getattr(roll, s["stat"])()
        elif stype == "cross":
            # 两特征列算术组合；操作列必须是本 spec 中先定义的过去向特征
            col1, col2 = s["col1"], s["col2"]
            if col1 not in out.columns or col2 not in out.columns:
                raise ValueError(
                    f"cross {name} 依赖列 {col1}/{col2} 未定义或顺序在自身之后"
                )
            out[name] = _CROSS_OPS[s["operation"]](out[col1], out[col2])
        else:
            raise ValueError(f"未知特征类型: {stype}")
    return out


# ---------------------------------------------------------------
# FeatureTransformer 接口（预留）：
#   Stateless 特征（lag/rolling/calendar）可在整段历史上一次性 transform；
#   Stateful 特征（scaler / encoder / PCA / learned transform）必须
#   fit(train) 后再 transform(data)，绝不能整段历史先 fit。
# 阶段一只实现 Stateless（build_features 即其 transform）。
# ---------------------------------------------------------------
class FeatureTransformer(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "FeatureTransformer": ...

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame: ...


class StatelessTransformer(FeatureTransformer):
    """包装 build_features 的 Stateless 实现（fit 为无操作）。"""

    def __init__(self, spec: List[dict] = FEATURE_SPEC):
        self.spec = spec

    def fit(self, train_df: pd.DataFrame) -> "StatelessTransformer":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return build_features(data, spec=self.spec)


@dataclass
class GEFComTask:
    """一个 Task 的完整训练/评测对象。"""

    task_id: int
    history_df: pd.DataFrame  # 全量可用历史 + 特征（一次性构建）
    train_df: pd.DataFrame  # 训练段（含特征 + 目标，无 NaN）
    val_df: pd.DataFrame  # 早停验证段（历史末 val_hours，严格在预测区间前）
    forecast_ts: pd.DatetimeIndex
    y_true: pd.Series  # 预测月真值（以 forecast_ts 为索引）
    feature_cols: List[str]
    target_col: str = TARGET_COL
    n_train: int = field(default=0)
    n_val: int = field(default=0)
    n_forecast: int = field(default=0)


def build_task(
    task_id: int,
    data_dir: Path = GEFCOM_DATA_DIR,
    val_hours: int = DEFAULT_VAL_HOURS,
    spec: List[dict] = FEATURE_SPEC,
    eval_hours: int = 0,
) -> GEFComTask:
    """
    构建 Task：可用历史 → 整段一次性特征 → 切 train/val → 取决策窗口真值。

    - 特征在整段历史上一次性构建：val 行的 lag/rolling 由 train 段的历史值
      提供，无预热 NaN，也无需"单独为 val 重新生成特征"。
    - train = 历史[前 MAX_LAG 行之后 : 末 val_hours 之前]（前 168 行
      lag_168/rolling_168 为 NaN，drop）。
    - val   = 历史末尾 val_hours（默认 168h = 7 天），用于 LightGBM 早停，
      严格在决策窗口之前，不接触决策窗口任何信息。
    - eval_hours = 0（默认）：决策窗口 = GEFCom 预测月（forecast_ts / y_true
      来自 benchmark + 官方/增量真值），行为与旧版完全一致。
    - eval_hours > 0：决策窗口 = 历史末 eval_hours 小时（干净的留出段），
      用于 evaluate_spec 的 val 窗口可插拔选项（P1 默认不用）。
    """
    if task_id < 1 or task_id > 15:
        raise ValueError(f"task_id 必须在 1..15，got {task_id}")
    if val_hours < 24:
        raise ValueError(f"val_hours 过小（{val_hours}），需 ≥ 24")
    if eval_hours < 0:
        raise ValueError(f"eval_hours 必须 >= 0，got {eval_hours}")
    if data_dir is None:
        data_dir = GEFCOM_DATA_DIR  # 调用方透传 None 时兜底

    av: Availability = available_history(task_id, data_dir)
    feature_cols = [s["name"] for s in spec]

    # 整段历史一次性构建特征（stateless）
    feat = build_features(av.history_df, spec=spec, target_col=TARGET_COL)
    hist_all = feat.copy()
    hist_all[TARGET_COL] = av.history_df[TARGET_COL].values

    if eval_hours > 0:
        if len(hist_all) <= MAX_LAG + val_hours + eval_hours:
            raise ValueError(
                f"Task {task_id} 历史过短（{len(hist_all)} 行），无法满足 "
                f"lag_168 预热 + {val_hours}h 验证 + {eval_hours}h 决策窗口"
            )
        hist = hist_all.iloc[:-eval_hours].copy()  # 排除决策窗口
        forecast_ts = hist_all.index[-eval_hours:]
        y_true = hist_all[TARGET_COL].iloc[-eval_hours:]
    else:
        if len(hist_all) <= MAX_LAG + val_hours:
            raise ValueError(
                f"Task {task_id} 历史过短（{len(hist_all)} 行），无法满足 "
                f"lag_168 预热 + {val_hours}h 验证"
            )
        hist = hist_all
        y_true = load_ground_truth(task_id, data_dir)[TARGET_COL]
        forecast_ts = av.forecast_ts

    train = hist.iloc[MAX_LAG:-val_hours].copy()
    val = hist.iloc[-val_hours:].copy()

    train = train.dropna(subset=feature_cols + [TARGET_COL])

    return GEFComTask(
        task_id=task_id,
        history_df=hist,
        train_df=train,
        val_df=val,
        forecast_ts=forecast_ts,
        y_true=y_true,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        n_train=len(train),
        n_val=len(val),
        n_forecast=len(forecast_ts),
    )
