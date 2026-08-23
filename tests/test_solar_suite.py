# Solar 光伏赛道接入自动测试（S1–S6）
# ---------------------------------------------------------------
# 运行：python tests/test_solar_suite.py
#
# S1 Loader           —— 15 Task × Zone 可用历史：连续性、预测起点=历史终点+1h、
#                        predictors 与 benchmark 时间戳对齐
# S2 真值一致性        —— Task k 真值 == Task{k+1} train POWER（预测月段）；
#                        Task 15 == solution（非缺失小时）
# S3 Task 构建         —— build_solar_task：train/val/forecast 长度、特征列、气象外生列
# S4 预测窗口特征      —— 索引==forecast_ts、预热后无 NaN、气象外生特征被消费
# S5 回放冒烟(persistence) —— 跑通 + RMSE 落在 [0,1] 量纲合理区间 + 泄漏检查通过
# S6 回放冒烟(lightgbm)  —— 单 Zone ML 路径跑通，RMSE 优于 persistence 幅度内
# ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.solar_loader import (
    SOLAR_TARGET_COL,
    load_solar_benchmark_ts,
    load_solar_ground_truth,
    load_solar_predictors,
    load_solar_solution,
    load_solar_train,
    solar_available_history,
)
from data.solar_task_builder import (
    SOLAR_FEATURE_COLS,
    SOLAR_WEATHER_COLS,
    build_solar_forecast_features,
    build_solar_task,
)
from evaluation.forecast_protocol import ONLINE_H1
from evaluation.solar_replay import replay_solar
from models.replay_backends import LightGBMBackend, PersistenceBackend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


# ---------------- S1 ----------------
def test_s1_loader():
    for tid in (1, 7, 15):
        for zone in (1, 3):
            av = solar_available_history(tid, zone)
            check(f"S1 T{tid} Z{zone} 历史逐小时连续",
                  (av.history_df.index.to_series().diff().dropna() ==
                   pd.Timedelta(hours=1)).all())
            check(f"S1 T{tid} Z{zone} 预测起点=历史终点+1h",
                  av.forecast_ts[0] == av.history_df.index[-1] + pd.Timedelta(hours=1),
                  f"{av.forecast_ts[0]} vs {av.history_df.index[-1] + pd.Timedelta(hours=1)}")
            check(f"S1 T{tid} Z{zone} 预测不重叠",
                  not av.forecast_ts.isin(av.history_df.index).any())
            check(f"S1 T{tid} Z{zone} 历史含气象外生且无缺失",
                  all(av.history_df[c].notna().all() for c in SOLAR_WEATHER_COLS))

    # predictors 与 benchmark 时间戳逐行对齐（决策时点可得的外生预报）
    for tid in (1, 15):
        ts = load_solar_benchmark_ts(tid)
        pred = load_solar_predictors(tid, 1)
        check(f"S1 T{tid} predictors 与 benchmark 对齐",
              set(pred.index) >= set(ts),
              f"predictors={len(pred)} benchmark={len(ts)}")


# ---------------- S2 ----------------
def test_s2_ground_truth():
    # Task k 真值 == Task{k+1} train 的 POWER（预测月段）
    for tid, zone in ((1, 1), (7, 2), (14, 3)):
        ts = load_solar_benchmark_ts(tid)
        gt = load_solar_ground_truth(tid, zone)[SOLAR_TARGET_COL]
        nxt = load_solar_train(tid + 1, zone)[SOLAR_TARGET_COL].reindex(ts)
        check(f"S2 T{tid} Z{zone} 真值==增量 train",
              np.allclose(gt.values, nxt.values, equal_nan=True))

    # Task 15 == solution（非缺失小时）
    ts15 = load_solar_benchmark_ts(15)
    sol = load_solar_solution()
    for zone in (1, 3):
        gt15 = load_solar_ground_truth(15, zone)[SOLAR_TARGET_COL]
        raw = sol[sol["ZONEID"] == zone][SOLAR_TARGET_COL].reindex(ts15)
        mask = raw.notna()
        check(f"S2 T15 Z{zone} 真值==solution（{int(mask.sum())}/{len(ts15)} 非缺失小时）",
              np.allclose(gt15[mask].values, raw[mask].values))


# ---------------- S3 ----------------
def test_s3_task_build():
    t = build_solar_task(1, 1)
    check("S3 n_train==8424", t.n_train == 8424, f"got {t.n_train}")
    check("S3 n_val==168", t.n_val == 168, f"got {t.n_val}")
    check("S3 n_forecast==720", t.n_forecast == 720, f"got {t.n_forecast}")
    check("S3 特征列齐全",
          set(t.feature_cols) == set(SOLAR_FEATURE_COLS))
    check("S3 train 无 NaN 目标", t.train_df[t.target_col].notna().all())
    check("S3 历史含气象外生列",
          set(SOLAR_WEATHER_COLS).issubset(t.history_df.columns))
    check("S3 预报气象帧覆盖预测月",
          len(t.weather_forecast_df) == t.n_forecast and
          (t.weather_forecast_df.index == t.forecast_ts).all())


# ---------------- S4 ----------------
def test_s4_forecast_features():
    t = build_solar_task(1, 1)
    observed = pd.concat([t.history_df[t.target_col], t.y_true])
    weather_full = pd.concat(
        [t.history_df[SOLAR_WEATHER_COLS],
         t.weather_forecast_df[SOLAR_WEATHER_COLS]]
    )
    fw = build_solar_forecast_features(observed, weather_full, t.forecast_ts)
    check("S4 索引==forecast_ts", (fw.index == t.forecast_ts).all())
    check("S4 预热后无 NaN", fw.isna().sum().sum() == 0)
    check("S4 首小时气象外生特征有效",
          fw.loc[t.forecast_ts[0], "VAR169_lag_1"] ==
          weather_full.loc[t.forecast_ts[0] - pd.Timedelta(hours=1), "VAR169"])
    check("S4 首小时目标 lag_1 有效",
          fw.loc[t.forecast_ts[0], "lag_1"] == observed.loc[t.forecast_ts[0] - pd.Timedelta(hours=1)])


# ---------------- S5 ----------------
def test_s5_replay_persistence():
    payload = replay_solar([1], PersistenceBackend(), ONLINE_H1, zones=[1],
                           leak_check="sample")
    r = payload["zone_results"][1][0]
    v = float(r.metrics["RMSE"])
    check("S5 persistence RMSE 合理量纲", 0.0 < v < 0.5, f"RMSE={v:.4f}")
    check("S5 泄漏检查通过（未抛错）", True)


# ---------------- S6 ----------------
def test_s6_replay_lightgbm():
    payload_lgb = replay_solar([1], LightGBMBackend(), ONLINE_H1, zones=[1],
                               leak_check="sample")
    payload_pers = replay_solar([1], PersistenceBackend(), ONLINE_H1, zones=[1],
                                skip_leak_check=True)
    r_lgb = payload_lgb["zone_results"][1][0]
    r_pers = payload_pers["zone_results"][1][0]
    v_lgb = float(r_lgb.metrics["RMSE"])
    v_pers = float(r_pers.metrics["RMSE"])
    check("S6 lightgbm RMSE 合理量纲", 0.0 < v_lgb < 0.3, f"RMSE={v_lgb:.4f}")
    check("S6 lightgbm 不显著劣于 persistence",
          v_lgb <= v_pers + 0.02,
          f"lgb={v_lgb:.4f} pers={v_pers:.4f}")


def main():
    print("=" * 60)
    test_s1_loader()
    test_s2_ground_truth()
    test_s3_task_build()
    test_s4_forecast_features()
    test_s5_replay_persistence()
    test_s6_replay_lightgbm()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
