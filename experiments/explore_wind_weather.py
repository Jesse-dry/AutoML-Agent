# 探索实验：当前小时风速预报（ws100@t）作为外生特征对 Wind RMSE 的影响
# ---------------------------------------------------------------
# 现状：WIND_FEATURE_SPEC 只用严格过去气象特征（ws100_lag_1/24/168 等），
# 因为不想改动共享的 build_features/leakage_checker。但风电预测领域最重要的
# 信号是"目标小时的风速预报"（TaskExpVars 在决策时点可得，外生非泄漏）。
#
# 本实验在现有 WindTask 之上做 A/B：
#   A = 现有特征集（baseline）
#   B = A + ws100@t（当前小时 100m 风速）
#   C = B + 预测 clip 到 [0,1]
# 训练侧 ws100@t 取历史实际天气；预测侧取 expvars 预报——与真实部署一致。
# ---------------------------------------------------------------
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.wind_task_builder import (
    WIND_WEATHER_DERIVED_COLS,
    build_wind_forecast_features,
    build_wind_task,
)
from models.replay_backends import LightGBMBackend

TASKS = [1, 7, 9, 15]
ZONES = [1, 3, 5]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate(tid, zone, add_curr=False, clamp=False):
    t = build_wind_task(tid, zone)
    ext_cols = t.feature_cols + (["ws100"] if add_curr else [])
    b = LightGBMBackend()
    b.fit(t.train_df, t.val_df, ext_cols, t.target_col)

    observed = pd.concat([t.history_df[t.target_col], t.y_true])
    weather_full = pd.concat(
        [t.history_df[WIND_WEATHER_DERIVED_COLS],
         t.weather_forecast_df[WIND_WEATHER_DERIVED_COLS]]
    )
    X = build_wind_forecast_features(observed, weather_full, t.forecast_ts)
    if add_curr:
        X["ws100"] = weather_full["ws100"].loc[t.forecast_ts].values

    yp = np.asarray(b.predict(X))
    if clamp:
        yp = np.clip(yp, 0.0, 1.0)
    return rmse(t.y_true.values, yp)


def main():
    rows = []
    for tid in TASKS:
        for zone in ZONES:
            base = evaluate(tid, zone, add_curr=False)
            cur = evaluate(tid, zone, add_curr=True)
            cur_c = evaluate(tid, zone, add_curr=True, clamp=True)
            rows.append({"task": tid, "zone": zone,
                         "baseline": base, "ws100@t": cur, "ws100@t+clip": cur_c})
    df = pd.DataFrame(rows)
    print("\n=== 逐 Task×Zone RMSE（归一化 [0,1] 量纲）===")
    print(df.round(4).to_string(index=False))
    print("\n=== 均值 ===")
    print(f"baseline       : {df['baseline'].mean():.4f}")
    print(f"+ws100@t       : {df['ws100@t'].mean():.4f}  (↓{(1-df['ws100@t'].mean()/df['baseline'].mean())*100:.1f}%)")
    print(f"+ws100@t+clip  : {df['ws100@t+clip'].mean():.4f}  (↓{(1-df['ws100@t+clip'].mean()/df['baseline'].mean())*100:.1f}%)")


if __name__ == "__main__":
    main()
