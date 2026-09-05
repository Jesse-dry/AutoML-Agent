# Wind 任务构建器：气象外生特征 + 无泄漏 train/val 切分
# ---------------------------------------------------------------
# 职责（镜像 data/task_builder.py 的 Load 版语义，但面向风电）：
#   1. 气象派生列 —— 由 U10/V10/U100/V100 风分量计算风速/风向/切变，
#      训练历史与预测窗口（expvars 预报）共用 compute_weather_features，保证一致。
#   2. WIND_FEATURE_SPEC —— 血缘式特征规格：time + TARGETVAR lag/rolling +
#      气象外生 lag/rolling（source 为派生列，uses_current_target=False）。
#      复用 data/task_builder.build_features 作为唯一严格过去向构造器。
#   3. build_wind_task —— 整段可用历史一次性建特征（修复旧实现预热上下文问题），
#      再切 train/val；预测窗口特征由 build_wind_forecast_features 生成，
#      气象列来自 TaskExpVars（决策时点可得，外生非泄漏）。
#
# 因果性约定（与 Load 版一致）：
#   - 目标 lag/rolling 只依赖 ≤ t-1 的 TARGETVAR（shift 语义，无自泄露）
#   - 气象 lag/rolling 只依赖 ≤ t-1 的气象值；当前小时气象不进特征（严格过去）
#   - time 特征 = 目标小时的日历属性，已知
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data.task_builder import MAX_LAG, _safe_div, build_features
from data.wind_loader import (
    WIND_TARGET_COL,
    load_wind_expvars,
    load_wind_ground_truth,
    wind_available_history,
)

TARGET_COL = WIND_TARGET_COL  # "TARGETVAR"

# 气象派生列（compute_weather_features 产出；spec 仅引用 ws10/ws100）
WIND_WEATHER_DERIVED_COLS = ["ws10", "ws100", "wd10", "wd100", "ws_ratio"]

DEFAULT_VAL_HOURS = 168


# ---------------------------------------------------------------
# 气象派生特征
# ---------------------------------------------------------------

def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    由 U10/V10/U100/V100 风分量计算气象派生列：
      ws10  / ws100  风速 = √(u²+v²)
      wd10  / wd100  风向 = atan2(v, u)（弧度）
      ws_ratio       风速垂直切变 = ws100 / ws10（安全除零）
    返回原 DataFrame 的副本 + 派生列（U/V 原始列保留）。
    """
    out = df.copy()
    for h in ("10", "100"):
        u = out[f"U{h}"]
        v = out[f"V{h}"]
        out[f"ws{h}"] = np.sqrt(u**2 + v**2)
        out[f"wd{h}"] = np.arctan2(v, u)
    out["ws_ratio"] = _safe_div(out["ws100"], out["ws10"])
    return out


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


WIND_FEATURE_SPEC: List[dict] = [
    # --- 时间特征 ---
    _time("hour", "hour"),
    _time("weekday", "weekday"),
    _time("month", "month"),
    _time("is_weekend", "is_weekend"),
    # --- 目标（TARGETVAR）滞后 ---
    _lag("lag_1", TARGET_COL, 1),
    _lag("lag_24", TARGET_COL, 24),
    _lag("lag_168", TARGET_COL, 168),
    # --- 目标滚动统计 ---
    _rolling("rolling_mean_24", TARGET_COL, 24, "mean"),
    _rolling("rolling_std_24", TARGET_COL, 24, "std"),
    _rolling("rolling_mean_168", TARGET_COL, 168, "mean"),
    # --- 气象外生（100m 风速为主，最贴出力） ---
    _lag("ws100_lag_1", "ws100", 1),
    _lag("ws100_lag_24", "ws100", 24),
    _lag("ws100_lag_168", "ws100", 168),
    _lag("ws10_lag_1", "ws10", 1),
    _lag("ws10_lag_24", "ws10", 24),
    _rolling("ws100_rolling_mean_24", "ws100", 24, "mean"),
]

# 冷启动基线（砍基线）：仅 time + TARGETVAR lag_1/24，不含 rolling / lag_168 / 气象外生。
# 作为 Wind 自进化 Agent 的极简起点，与 Load/Solar 口径一致（负对照可比）。
WIND_COLD_START_FEATURE_SPEC: List[dict] = [
    _time("hour", "hour"),
    _time("weekday", "weekday"),
    _time("month", "month"),
    _time("is_weekend", "is_weekend"),
    _lag("lag_1", TARGET_COL, 1),
    _lag("lag_24", TARGET_COL, 24),
]

WIND_FEATURE_COLS: List[str] = [s["name"] for s in WIND_FEATURE_SPEC]


# ---------------------------------------------------------------
# WindTask + 构建
# ---------------------------------------------------------------

@dataclass
class WindTask:
    """一个 Task×Zone 的完整训练/评测对象（duck-type 兼容 GEFComTask）。"""

    task_id: int
    zone: int
    history_df: pd.DataFrame  # 全量可用历史 + 特征 + TARGETVAR + 气象派生列
    train_df: pd.DataFrame  # 训练段（含特征 + 目标，无 NaN）
    val_df: pd.DataFrame  # 早停验证段（历史末 val_hours，严格在预测区间前）
    forecast_ts: pd.DatetimeIndex
    y_true: pd.Series  # 预测月真值（以 forecast_ts 为索引）
    weather_forecast_df: pd.DataFrame  # 预测月气象预报（决策时点可得，datetime 索引）
    feature_cols: List[str]
    target_col: str = TARGET_COL
    n_train: int = field(default=0)
    n_val: int = field(default=0)
    n_forecast: int = field(default=0)


def build_wind_forecast_features(
    observed_target: pd.Series,
    weather_frame: pd.DataFrame,
    forecast_ts: pd.DatetimeIndex,
    spec: List[dict] = WIND_FEATURE_SPEC,
) -> pd.DataFrame:
    """
    预测窗口特征：观测目标（历史 ∪ 回填）+ 外生气象（历史 ∪ 预报）→ build_features
    → 切 forecast_ts。

    气象列对预测月取决策时点已知的预报值（TaskExpVars），外生非泄漏；目标 lag/rolling
    由 observed_target 提供（online_h1 含真值 / recursive 含回填预测值）。
    """
    frame = pd.DataFrame(index=observed_target.index)
    frame[TARGET_COL] = observed_target
    for c in WIND_WEATHER_DERIVED_COLS:
        if c in weather_frame.columns:
            frame[c] = weather_frame[c]
    feat = build_features(frame, spec=spec, target_col=TARGET_COL)
    return feat.loc[forecast_ts]


def build_wind_task(
    task_id: int,
    zone: int,
    data_dir: Path = None,
    val_hours: int = DEFAULT_VAL_HOURS,
    spec: List[dict] = WIND_FEATURE_SPEC,
) -> WindTask:
    """
    构建 Task×Zone：可用历史 → 气象派生 → 整段一次性特征 → 切 train/val →
    取决策窗口真值 + 预报气象。

    - 特征在整段历史上一次性构建：val 行的 lag/rolling 由 train 段历史值提供，
      无预热 NaN。
    - train = 历史[前 MAX_LAG 行之后 : 末 val_hours 之前]
    - val   = 历史末尾 val_hours（默认 168h = 7 天），严格在预测区间之前
    - 决策窗口 = GEFCom 预测月（forecast_ts / y_true 来自 benchmark + 官方/增量真值）
    """
    if task_id < 1 or task_id > 15:
        raise ValueError(f"task_id 必须在 1..15，got {task_id}")
    if val_hours < 24:
        raise ValueError(f"val_hours 过小（{val_hours}），需 ≥ 24")
    if data_dir is None:
        from data.wind_loader import WIND_DATA_DIR
        data_dir = WIND_DATA_DIR

    av = wind_available_history(task_id, zone, data_dir)
    feature_cols = [s["name"] for s in spec]

    # 气象派生 + 整段历史一次性建特征（stateless）
    hist = compute_weather_features(av.history_df)
    feat = build_features(hist, spec=spec, target_col=TARGET_COL)
    hist_all = feat.copy()
    hist_all[TARGET_COL] = hist[TARGET_COL].values
    for c in WIND_WEATHER_DERIVED_COLS:
        hist_all[c] = hist[c].values

    if len(hist_all) <= MAX_LAG + val_hours:
        raise ValueError(
            f"Wind Task {task_id} Zone {zone} 历史过短（{len(hist_all)} 行），"
            f"无法满足 lag_168 预热 + {val_hours}h 验证"
        )

    train = hist_all.iloc[MAX_LAG:-val_hours].copy()
    val = hist_all.iloc[-val_hours:].copy()
    train = train.dropna(subset=feature_cols + [TARGET_COL])

    y_true = load_wind_ground_truth(task_id, zone, data_dir)[TARGET_COL]

    # 预测月气象预报（外生，决策时点可得）
    weather_fc = compute_weather_features(load_wind_expvars(task_id, zone, data_dir))
    weather_fc = weather_fc[WIND_WEATHER_DERIVED_COLS].loc[av.forecast_ts]

    return WindTask(
        task_id=task_id,
        zone=zone,
        history_df=hist_all,
        train_df=train,
        val_df=val,
        forecast_ts=av.forecast_ts,
        y_true=y_true,
        weather_forecast_df=weather_fc,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        n_train=len(train),
        n_val=len(val),
        n_forecast=len(av.forecast_ts),
    )
