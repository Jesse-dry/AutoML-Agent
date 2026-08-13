# Wind 风电赛道接入自动测试（W1–W6）
# ---------------------------------------------------------------
# 运行：python tests/test_wind_suite.py
#
# W1 Loader           —— 15 Task × Zone 可用历史：连续性、预测起点=历史终点+1h、
#                        expvars 与 benchmark 时间戳对齐
# W2 真值一致性        —— Task k 真值 == Task{k+1} train TARGETVAR（预测月段）；
#                        Task 15 == solution15（非缺失小时）
# W3 Task 构建         —— build_wind_task：train/val/forecast 长度、特征列、气象派生列
# W4 预测窗口特征      —— 索引==forecast_ts、预热后无 NaN、气象外生特征被消费
# W5 回放冒烟(persistence) —— 跑通 + RMSE 落在 [0,1] 量纲合理区间 + 泄漏检查通过
# W6 回放冒烟(lightgbm)  —— 单 Zone ML 路径跑通，RMSE 优于 persistence 幅度内
# ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.wind_loader import (
    WIND_TARGET_COL,
    load_wind_benchmark_ts,
    load_wind_expvars,
    load_wind_ground_truth,
    load_wind_solution,
    load_wind_train,
    wind_available_history,
)
from data.wind_task_builder import (
    WIND_FEATURE_COLS,
    WIND_WEATHER_DERIVED_COLS,
    build_wind_forecast_features,
    build_wind_task,
    compute_weather_features,
)
from evaluation.forecast_protocol import ONLINE_H1
from evaluation.wind_replay import replay_wind
from models.replay_backends import LightGBMBackend, PersistenceBackend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def rmse(y_true, y_hat):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_hat)) ** 2)))


# ---------------- W1 ----------------
def test_w1_loader():
    for tid in (1, 7, 15):
        for zone in (1, 10):
            av = wind_available_history(tid, zone)
            check(f"W1 T{tid} Z{zone} 历史逐小时连续",
                  (av.history_df.index.to_series().diff().dropna() ==
                   pd.Timedelta(hours=1)).all())
            check(f"W1 T{tid} Z{zone} 预测起点=历史终点+1h",
                  av.forecast_ts[0] == av.history_df.index[-1] + pd.Timedelta(hours=1),
                  f"{av.forecast_ts[0]} vs {av.history_df.index[-1] + pd.Timedelta(hours=1)}")
            check(f"W1 T{tid} Z{zone} 预测不重叠",
                  not av.forecast_ts.isin(av.history_df.index).any())

    # expvars 与 benchmark 时间戳逐行对齐（决策时点可得的外生预报）
    for tid in (1, 15):
        ts = load_wind_benchmark_ts(tid)
        exp = load_wind_expvars(tid, 1)
        check(f"W1 T{tid} expvars 与 benchmark 对齐",
              set(exp.index) == set(ts),
              f"expvars={len(exp)} benchmark={len(ts)}")


# ---------------- W2 ----------------
def test_w2_ground_truth():
    # Task k 真值 == Task{k+1} train 的 TARGETVAR（预测月段）
    for tid, zone in ((1, 1), (7, 5), (14, 10)):
        ts = load_wind_benchmark_ts(tid)
        gt = load_wind_ground_truth(tid, zone)[WIND_TARGET_COL]
        nxt = load_wind_train(tid + 1, zone)[WIND_TARGET_COL].reindex(ts)
        check(f"W2 T{tid} Z{zone} 真值==增量 train",
              np.allclose(gt.values, nxt.values, equal_nan=True))

    # Task 15 == solution15_W（非缺失小时）
    ts15 = load_wind_benchmark_ts(15)
    sol = load_wind_solution()
    for zone in (1, 10):
        gt15 = load_wind_ground_truth(15, zone)[WIND_TARGET_COL]
        raw = sol[sol["ZONEID"] == zone][WIND_TARGET_COL].reindex(ts15)
        mask = raw.notna()
        check(f"W2 T15 Z{zone} 真值==solution15（{int(mask.sum())}/744 非缺失小时）",
              np.allclose(gt15[mask].values, raw[mask].values))


# ---------------- W3 ----------------
def test_w3_task_build():
    t = build_wind_task(1, 1)
    check("W3 n_train==6240", t.n_train == 6240, f"got {t.n_train}")
    check("W3 n_val==168", t.n_val == 168, f"got {t.n_val}")
    check("W3 n_forecast==744", t.n_forecast == 744, f"got {t.n_forecast}")
    check("W3 特征列齐全",
          set(t.feature_cols) == set(WIND_FEATURE_COLS))
    check("W3 train 无 NaN 目标", t.train_df[t.target_col].notna().all())
    check("W3 历史含气象派生列",
          set(WIND_WEATHER_DERIVED_COLS).issubset(t.history_df.columns))
    check("W3 预报气象帧覆盖预测月",
          len(t.weather_forecast_df) == t.n_forecast and
          (t.weather_forecast_df.index == t.forecast_ts).all())

    # 气象派生正确性：ws10 == √(U10²+V10²)（用原始 train 帧验证）
    raw = load_wind_train(1, 1).head()
    df = compute_weather_features(raw)
    expect = np.sqrt(df["U10"] ** 2 + df["V10"] ** 2)
    check("W3 ws10 计算正确", np.allclose(df["ws10"].values, expect.values))
    expect100 = np.sqrt(df["U100"] ** 2 + df["V100"] ** 2)
    check("W3 ws100 计算正确", np.allclose(df["ws100"].values, expect100.values))


# ---------------- W4 ----------------
def test_w4_forecast_features():
    t = build_wind_task(1, 1)
    observed = pd.concat([t.history_df[t.target_col], t.y_true])
    weather_full = pd.concat(
        [t.history_df[WIND_WEATHER_DERIVED_COLS],
         t.weather_forecast_df[WIND_WEATHER_DERIVED_COLS]]
    )
    fw = build_wind_forecast_features(observed, weather_full, t.forecast_ts)
    check("W4 索引==forecast_ts", (fw.index == t.forecast_ts).all())
    check("W4 预热后无 NaN", fw.isna().sum().sum() == 0)
    check("W4 首小时气象外生特征有效",
          fw.loc[t.forecast_ts[0], "ws100_lag_1"] ==
          weather_full.loc[t.forecast_ts[0] - pd.Timedelta(hours=1), "ws100"])
    check("W4 首小时目标 lag_1 有效",
          fw.loc[t.forecast_ts[0], "lag_1"] == observed.loc[t.forecast_ts[0] - pd.Timedelta(hours=1)])


# ---------------- W5 ----------------
def test_w5_replay_persistence():
    payload = replay_wind([1], PersistenceBackend(), ONLINE_H1, zones=[1],
                          leak_check="sample")
    r = payload["zone_results"][1][0]
    v = float(r.metrics["RMSE"])
    check("W5 persistence RMSE 合理量纲", 0.0 < v < 0.5, f"RMSE={v:.4f}")
    check("W5 泄漏检查通过（未抛错）", True)


# ---------------- W6 ----------------
def test_w6_replay_lightgbm():
    payload_lgb = replay_wind([1], LightGBMBackend(), ONLINE_H1, zones=[1],
                              leak_check="sample")
    payload_pers = replay_wind([1], PersistenceBackend(), ONLINE_H1, zones=[1],
                               skip_leak_check=True)
    r_lgb = payload_lgb["zone_results"][1][0]
    r_pers = payload_pers["zone_results"][1][0]
    v_lgb = float(r_lgb.metrics["RMSE"])
    v_pers = float(r_pers.metrics["RMSE"])
    check("W6 lightgbm RMSE 合理量纲", 0.0 < v_lgb < 0.3, f"RMSE={v_lgb:.4f}")
    check("W6 lightgbm 不显著劣于 persistence",
          v_lgb <= v_pers + 0.02,
          f"lgb={v_lgb:.4f} pers={v_pers:.4f}")


def main():
    print("=" * 60)
    test_w1_loader()
    test_w2_ground_truth()
    test_w3_task_build()
    test_w4_forecast_features()
    test_w5_replay_persistence()
    test_w6_replay_lightgbm()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
