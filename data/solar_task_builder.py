# Solar 任务构建器：气象外生特征 + 无泄漏 train/val 切分
# ---------------------------------------------------------------
# 职责（镜像 data/wind_task_builder.py 的 Wind 版语义，但面向光伏）：
#   1. SOLAR_FEATURE_SPEC —— 血缘式特征规格：time + POWER lag/rolling +
#      气象外生 lag/rolling（source 为 VAR169/VAR164/VAR167，
#      uses_current_target=False）。复用 data/task_builder.build_features 作为
#      唯一严格过去向构造器。
#   2. build_solar_task —— 整段可用历史一次性建特征，再切 train/val；
#      预测窗口特征由 build_solar_forecast_features 生成，气象列来自
#      predictors（决策时点可得，外生非泄漏）。
#
# 因果性约定（与 Load/Wind 版一致）：
#   - 目标 lag/rolling 只依赖 ≤ t-1 的 POWER（shift 语义，无自泄露）
#   - 气象 lag/rolling 只依赖 ≤ t-1 的气象值；当前小时气象不进特征（严格过去）
#   - time 特征 = 目标小时的日历属性，已知
#
# 与 Wind 的差异：光伏的气象外生无需派生（VAR169=SSRD 太阳辐射 / VAR164=TCC 云量 /
# VAR167=2T 温度 本身就是有物理含义的量），直接取 raw 列；train 文件不含气象，
# 气象来自单独 predictors 文件，已在 solar_loader.solar_available_history 里
# 合并进历史（历史段）+ 单独取预测月段。
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

from data.task_builder import MAX_LAG, build_features
from data.solar_loader import (
    SOLAR_TARGET_COL,
    SOLAR_WEATHER_COLS,
    load_solar_ground_truth,
    load_solar_predictors,
    solar_available_history,
)

TARGET_COL = SOLAR_TARGET_COL  # "POWER"

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


SOLAR_FEATURE_SPEC: List[dict] = [
    # --- 时间特征 ---
    _time("hour", "hour"),
    _time("weekday", "weekday"),
    _time("month", "month"),
    _time("is_weekend", "is_weekend"),
    # --- 目标（POWER）滞后 ---
    _lag("lag_1", TARGET_COL, 1),
    _lag("lag_24", TARGET_COL, 24),
    _lag("lag_168", TARGET_COL, 168),
    # --- 目标滚动统计 ---
    _rolling("rolling_mean_24", TARGET_COL, 24, "mean"),
    _rolling("rolling_std_24", TARGET_COL, 24, "std"),
    _rolling("rolling_mean_168", TARGET_COL, 168, "mean"),
    # --- 气象外生（SSRD 太阳辐射为主，最贴出力） ---
    _lag("VAR169_lag_1", "VAR169", 1),
    _lag("VAR169_lag_24", "VAR169", 24),
    _rolling("VAR169_rolling_mean_24", "VAR169", 24, "mean"),
    _lag("VAR164_lag_1", "VAR164", 1),
    _lag("VAR167_lag_1", "VAR167", 1),
]

SOLAR_FEATURE_COLS: List[str] = [s["name"] for s in SOLAR_FEATURE_SPEC]


# ---------------------------------------------------------------
# SolarTask + 构建
# ---------------------------------------------------------------

@dataclass
class SolarTask:
    """一个 Task×Zone 的完整训练/评测对象（duck-type 兼容 GEFComTask / WindTask）。"""

    task_id: int
    zone: int
    history_df: pd.DataFrame  # 全量可用历史 + 特征 + POWER + 气象外生列
    train_df: pd.DataFrame  # 训练段（含特征 + 目标，无 NaN）
    val_df: pd.DataFrame  # 早停验证段（历史末 val_hours，严格在预测区间前）
    forecast_ts: pd.DatetimeIndex
    y_true: pd.Series  # 预测月真值（以 forecast_ts 为索引）
    weather_forecast_df: pd.DataFrame  # 预测月气象外生（决策时点可得，datetime 索引）
    feature_cols: List[str]
    target_col: str = TARGET_COL
    n_train: int = field(default=0)
    n_val: int = field(default=0)
    n_forecast: int = field(default=0)


def build_solar_forecast_features(
    observed_target: pd.Series,
    weather_frame: pd.DataFrame,
    forecast_ts: pd.DatetimeIndex,
    spec: List[dict] = SOLAR_FEATURE_SPEC,
) -> pd.DataFrame:
    """
    预测窗口特征：观测目标（历史 ∪ 回填）+ 外生气象（历史 ∪ 预报）→ build_features
    → 切 forecast_ts。

    气象列对预测月取决策时点已知的预报值（predictors），外生非泄漏；目标 lag/rolling
    由 observed_target 提供（online_h1 含真值 / recursive 含回填预测值）。
    """
    frame = pd.DataFrame(index=observed_target.index)
    frame[TARGET_COL] = observed_target
    for c in SOLAR_WEATHER_COLS:
        if c in weather_frame.columns:
            frame[c] = weather_frame[c]
    feat = build_features(frame, spec=spec, target_col=TARGET_COL)
    return feat.loc[forecast_ts]


def build_solar_task(
    task_id: int,
    zone: int,
    data_dir: Path = None,
    val_hours: int = DEFAULT_VAL_HOURS,
    spec: List[dict] = SOLAR_FEATURE_SPEC,
) -> SolarTask:
    """
    构建 Task×Zone：可用历史（含气象外生）→ 整段一次性特征 → 切 train/val →
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
        from data.solar_loader import SOLAR_DATA_DIR
        data_dir = SOLAR_DATA_DIR

    av = solar_available_history(task_id, zone, data_dir)
    feature_cols = [s["name"] for s in spec]

    # 整段历史一次性建特征（stateless）；hist 已含 POWER + 气象外生列
    hist = av.history_df
    feat = build_features(hist, spec=spec, target_col=TARGET_COL)
    hist_all = feat.copy()
    hist_all[TARGET_COL] = hist[TARGET_COL].values
    for c in SOLAR_WEATHER_COLS:
        hist_all[c] = hist[c].values

    if len(hist_all) <= MAX_LAG + val_hours:
        raise ValueError(
            f"Solar Task {task_id} Zone {zone} 历史过短（{len(hist_all)} 行），"
            f"无法满足 lag_168 预热 + {val_hours}h 验证"
        )

    train = hist_all.iloc[MAX_LAG:-val_hours].copy()
    val = hist_all.iloc[-val_hours:].copy()
    train = train.dropna(subset=feature_cols + [TARGET_COL])

    y_true = load_solar_ground_truth(task_id, zone, data_dir)[TARGET_COL]

    # 预测月气象外生（决策时点可得，外生非泄漏）
    pred = load_solar_predictors(task_id, zone, data_dir)
    weather_fc = pred[SOLAR_WEATHER_COLS].loc[av.forecast_ts]

    return SolarTask(
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
