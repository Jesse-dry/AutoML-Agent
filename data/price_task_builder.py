# Price 任务构建器：外生负荷特征 + 无泄漏 train/val 切分
# ---------------------------------------------------------------
# 职责（镜像 data/wind_task_builder.py 的 Wind 版语义，但面向电价）：
#   1. PRICE_FEATURE_SPEC —— 血缘式特征规格：time + Zonal Price lag/rolling +
#      外生负荷 lag/rolling（source 为 Forecasted Total/Zonal Load，决策时点可得）。
#      复用 data/task_builder.build_features 作为唯一严格过去向构造器。
#   2. build_price_task —— 整段可用历史一次性建特征（无预热 NaN），再切 train/val；
#      预测窗口特征由 build_price_forecast_features 生成，外生列来自 train 文件的
#      预测日段（决策时点可得，外生非泄漏）。
#
# 因果性约定（与 Load/Wind 一致）：
#   - 目标 lag/rolling 只依赖 ≤ t-1 的 Zonal Price（shift 语义，无自泄露）
#   - 外生 lag/rolling 只依赖 ≤ t-1 的 Forecasted Load；当前小时负荷不进特征（严格过去）
#   - time 特征 = 目标小时的日历属性，已知
#
# 与 Wind 的差异：
#   - 无气象派生列（Forecasted Load 本身就是标量，无需 U/V→风速变换）
#   - 单分区，无 zone 维度
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from data.price_loader import (
    PRICE_EXOGENOUS_COLS,
    PRICE_TARGET_COL,
    load_price_forecast_exogenous,
    load_price_ground_truth,
    price_available_history,
)
from data.task_builder import MAX_LAG, build_features

TARGET_COL = PRICE_TARGET_COL  # "Zonal Price"

DEFAULT_VAL_HOURS = 168


# ---------------------------------------------------------------
# 血缘式特征规格（顺序即特征列顺序）
# ---------------------------------------------------------------

def _lag(name: str, source: str, k: int) -> dict:
    return {
        "name": name, "type": "lag", "source": source, "k": k,
        "lookback_start": -k, "lookback_end": -k, "uses_current_target": False,
    }


def _rolling(name: str, source: str, window: int, stat: str) -> dict:
    return {
        "name": name, "type": "rolling", "source": source,
        "window": window, "stat": stat, "min_periods": window,
        "lookback_start": -window, "lookback_end": -1, "uses_current_target": False,
    }


def _time(name: str, attr: str) -> dict:
    return {
        "name": name, "type": "time", "attr": attr,
        "lookback_start": 0, "lookback_end": 0, "uses_current_target": False,
    }


PRICE_FEATURE_SPEC: List[dict] = [
    # --- 时间特征 ---
    _time("hour", "hour"),
    _time("weekday", "weekday"),
    _time("month", "month"),
    _time("is_weekend", "is_weekend"),
    # --- 目标（Zonal Price）滞后 ---
    _lag("lag_1", TARGET_COL, 1),
    _lag("lag_24", TARGET_COL, 24),
    _lag("lag_168", TARGET_COL, 168),
    # --- 目标滚动统计 ---
    _rolling("rolling_mean_24", TARGET_COL, 24, "mean"),
    _rolling("rolling_std_24", TARGET_COL, 24, "std"),
    _rolling("rolling_mean_168", TARGET_COL, 168, "mean"),
    # --- 外生负荷 lag（决策时点可得，严格过去） ---
    _lag("total_load_lag_1", PRICE_EXOGENOUS_COLS[0], 1),
    _lag("total_load_lag_24", PRICE_EXOGENOUS_COLS[0], 24),
    _lag("zonal_load_lag_1", PRICE_EXOGENOUS_COLS[1], 1),
    _lag("zonal_load_lag_24", PRICE_EXOGENOUS_COLS[1], 24),
]

PRICE_FEATURE_COLS: List[str] = [s["name"] for s in PRICE_FEATURE_SPEC]


# ---------------------------------------------------------------
# PriceTask + 构建
# ---------------------------------------------------------------

@dataclass
class PriceTask:
    """一个 Task 的完整训练/评测对象（duck-type 兼容 GEFComTask）。"""

    task_id: int
    history_df: pd.DataFrame  # 全量可用历史 + 特征 + Zonal Price + 外生列
    train_df: pd.DataFrame  # 训练段（含特征 + 目标，无 NaN）
    val_df: pd.DataFrame  # 早停验证段（历史末 val_hours，严格在预测区间前）
    forecast_ts: pd.DatetimeIndex
    y_true: pd.Series  # 预测日真值（以 forecast_ts 为索引）
    exogenous_forecast_df: pd.DataFrame  # 预测日外生负荷（决策时点可得，datetime 索引）
    feature_cols: List[str]
    target_col: str = TARGET_COL
    n_train: int = field(default=0)
    n_val: int = field(default=0)
    n_forecast: int = field(default=0)


def build_price_forecast_features(
    observed_target: pd.Series,
    exogenous_frame: pd.DataFrame,
    forecast_ts: pd.DatetimeIndex,
    spec: List[dict] = PRICE_FEATURE_SPEC,
) -> pd.DataFrame:
    """
    预测窗口特征：观测目标（历史 ∪ 回填）+ 外生负荷（历史 ∪ 预测日预报）
    → build_features → 切 forecast_ts。

    外生列对预测日取决策时点已知的预报值（train 文件预测日段），外生非泄漏；
    目标 lag/rolling 由 observed_target 提供（online_h1 含真值 / recursive 含回填预测值）。
    """
    frame = pd.DataFrame(index=observed_target.index)
    frame[TARGET_COL] = observed_target
    for c in PRICE_EXOGENOUS_COLS:
        if c in exogenous_frame.columns:
            frame[c] = exogenous_frame[c]
    feat = build_features(frame, spec=spec, target_col=TARGET_COL)
    return feat.loc[forecast_ts]


def build_price_task(
    task_id: int,
    data_dir: Path = None,
    val_hours: int = DEFAULT_VAL_HOURS,
    spec: List[dict] = PRICE_FEATURE_SPEC,
) -> PriceTask:
    """
    构建 Task：可用历史 → 整段一次性特征 → 切 train/val → 取预测日真值 + 预报外生。

    - 特征在整段历史上一次性构建：val 行的 lag/rolling 由 train 段历史值提供，
      无预热 NaN。
    - train = 历史[前 MAX_LAG 行之后 : 末 val_hours 之前]
    - val   = 历史末尾 val_hours（默认 168h = 7 天），严格在预测区间之前
    - 预测窗口 = GEFCom 预测日（24h；forecast_ts / y_true 来自 benchmark + 官方/增量真值）
    """
    if task_id < 1 or task_id > 15:
        raise ValueError(f"task_id 必须在 1..15，got {task_id}")
    if val_hours < 24:
        raise ValueError(f"val_hours 过小（{val_hours}），需 ≥ 24")
    if data_dir is None:
        from data.price_loader import PRICE_DATA_DIR
        data_dir = PRICE_DATA_DIR

    av = price_available_history(task_id, data_dir)
    feature_cols = [s["name"] for s in spec]

    # 整段历史一次性建特征（stateless）
    hist = av.history_df
    feat = build_features(hist, spec=spec, target_col=TARGET_COL)
    hist_all = feat.copy()
    hist_all[TARGET_COL] = hist[TARGET_COL].values
    for c in PRICE_EXOGENOUS_COLS:
        hist_all[c] = hist[c].values

    if len(hist_all) <= MAX_LAG + val_hours:
        raise ValueError(
            f"Task {task_id} 历史过短（{len(hist_all)} 行），"
            f"无法满足 lag_168 预热 + {val_hours}h 验证"
        )

    train = hist_all.iloc[MAX_LAG:-val_hours].copy()
    val = hist_all.iloc[-val_hours:].copy()
    train = train.dropna(subset=feature_cols + [TARGET_COL])

    y_true = load_price_ground_truth(task_id, data_dir)[TARGET_COL]

    # 预测日外生负荷（外生，决策时点可得）
    exo_fc = load_price_forecast_exogenous(task_id, data_dir)
    exo_fc = exo_fc.loc[av.forecast_ts]

    return PriceTask(
        task_id=task_id,
        history_df=hist_all,
        train_df=train,
        val_df=val,
        forecast_ts=av.forecast_ts,
        y_true=y_true,
        exogenous_forecast_df=exo_fc,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        n_train=len(train),
        n_val=len(val),
        n_forecast=len(av.forecast_ts),
    )
