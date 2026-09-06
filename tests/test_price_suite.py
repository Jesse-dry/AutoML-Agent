# Price 电价赛道接入自动测试（P1–P6）
# ---------------------------------------------------------------
# 运行：python tests/test_price_suite.py
#
# P1 Loader           —— 15 Task 可用历史：连续性（容忍 DST 2h gap）、预测起点=历史终点+1h、
#                        外生负荷预报与 benchmark 时间戳对齐
# P2 真值一致性        —— Task k 真值 == Task{k+1} train Zonal Price（预测日段）；
#                        Task 15 == solution15（非缺失小时）
# P3 Task 构建         —— build_price_task：train/val/forecast 长度、特征列、外生列
# P4 预测窗口特征      —— 索引==forecast_ts、预热后无 NaN、外生负荷特征被消费
# P5 回放冒烟(persistence) —— 跑通 + RMSE 落在电价合理量纲 + 泄漏检查通过
# P6 回放冒烟(lightgbm)  —— 单 Task ML 路径跑通，RMSE 优于 persistence 幅度内
# ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.price_loader import (
    PRICE_TARGET_COL,
    load_price_benchmark_ts,
    load_price_forecast_exogenous,
    load_price_ground_truth,
    load_price_solution,
    load_price_train,
    price_available_history,
)
from data.price_task_builder import (
    PRICE_FEATURE_COLS,
    PRICE_EXOGENOUS_COLS,
    build_price_forecast_features,
    build_price_task,
)
from evaluation.forecast_protocol import ONLINE_H1
from evaluation.price_replay import replay_price
from models.replay_backends import LightGBMBackend, PersistenceBackend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def rmse(y_true, y_hat):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_hat)) ** 2)))


# ---------------- P1 ----------------
def test_p1_loader():
    for tid in (1, 7, 15):
        av = price_available_history(tid)
        diffs = av.history_df.index.to_series().diff().dropna()
        check(f"P1 T{tid} 历史连续（容忍 DST 2h gap）",
              diffs.isin([pd.Timedelta(hours=1), pd.Timedelta(hours=2)]).all())
        check(f"P1 T{tid} 预测起点=历史终点+1h",
              av.forecast_ts[0] == av.history_df.index[-1] + pd.Timedelta(hours=1),
              f"{av.forecast_ts[0]} vs {av.history_df.index[-1] + pd.Timedelta(hours=1)}")
        check(f"P1 T{tid} 预测不重叠",
              not av.forecast_ts.isin(av.history_df.index).any())
        check(f"P1 T{tid} 预测窗口=24h", av.n_forecast == 24, f"{av.n_forecast}")

    # 外生负荷预报与 benchmark 时间戳逐行对齐（决策时点可得）
    for tid in (1, 15):
        ts = load_price_benchmark_ts(tid)
        exo = load_price_forecast_exogenous(tid)
        check(f"P1 T{tid} 外生预报与 benchmark 对齐",
              set(exo.index) == set(ts),
              f"exo={len(exo)} benchmark={len(ts)}")
        check(f"P1 T{tid} 外生预报无 NaN", exo[PRICE_EXOGENOUS_COLS].notna().all().all())


# ---------------- P2 ----------------
def test_p2_ground_truth():
    # Task k 真值 == Task{k+1} train 的 Zonal Price（预测日段）
    for tid in (1, 7, 14):
        ts = load_price_benchmark_ts(tid)
        gt = load_price_ground_truth(tid)[PRICE_TARGET_COL]
        nxt = load_price_train(tid + 1)[PRICE_TARGET_COL].reindex(ts)
        check(f"P2 T{tid} 真值==增量 train",
              np.allclose(gt.values, nxt.values, equal_nan=True))

    # Task 15 == solution15（非缺失小时）
    ts15 = load_price_benchmark_ts(15)
    sol = load_price_solution()
    gt15 = load_price_ground_truth(15)[PRICE_TARGET_COL]
    raw = sol[PRICE_TARGET_COL].reindex(ts15)
    mask = raw.notna()
    check(f"P2 T15 真值==solution15（{int(mask.sum())}/24 非缺失小时）",
          np.allclose(gt15[mask].values, raw[mask].values))


# ---------------- P3 ----------------
def test_p3_task_build():
    t = build_price_task(1)
    check("P3 n_train==21191", t.n_train == 21191, f"got {t.n_train}")
    check("P3 n_val==168", t.n_val == 168, f"got {t.n_val}")
    check("P3 n_forecast==24", t.n_forecast == 24, f"got {t.n_forecast}")
    check("P3 特征列齐全", set(t.feature_cols) == set(PRICE_FEATURE_COLS))
    check("P3 train 无 NaN 目标", t.train_df[t.target_col].notna().all())
    check("P3 历史含外生负荷列",
          set(PRICE_EXOGENOUS_COLS).issubset(t.history_df.columns))
    check("P3 预报外生帧覆盖预测日",
          len(t.exogenous_forecast_df) == t.n_forecast and
          (t.exogenous_forecast_df.index == t.forecast_ts).all())


# ---------------- P4 ----------------
def test_p4_forecast_features():
    t = build_price_task(1)
    observed = pd.concat([t.history_df[t.target_col], t.y_true])
    exo_full = pd.concat(
        [t.history_df[PRICE_EXOGENOUS_COLS],
         t.exogenous_forecast_df[PRICE_EXOGENOUS_COLS]]
    )
    fw = build_price_forecast_features(observed, exo_full, t.forecast_ts)
    check("P4 索引==forecast_ts", (fw.index == t.forecast_ts).all())
    check("P4 预热后无 NaN", fw.isna().sum().sum() == 0)
    check("P4 首小时外生负荷特征有效",
          fw.loc[t.forecast_ts[0], "total_load_lag_1"] ==
          exo_full.loc[t.forecast_ts[0] - pd.Timedelta(hours=1), PRICE_EXOGENOUS_COLS[0]])
    check("P4 首小时目标 lag_1 有效",
          fw.loc[t.forecast_ts[0], "lag_1"] ==
          observed.loc[t.forecast_ts[0] - pd.Timedelta(hours=1)])


# ---------------- P5 ----------------
def test_p5_replay_persistence():
    payload = replay_price([1], PersistenceBackend(), ONLINE_H1, leak_check="sample")
    r = payload["results"][0]
    v = float(r.metrics["RMSE"])
    check("P5 persistence RMSE 合理量纲", 0.0 < v < 30.0, f"RMSE={v:.4f}")
    check("P5 泄漏检查通过（未抛错）", True)


# ---------------- P6 ----------------
def test_p6_replay_lightgbm():
    payload_lgb = replay_price([1], LightGBMBackend(), ONLINE_H1, leak_check="sample")
    payload_pers = replay_price([1], PersistenceBackend(), ONLINE_H1,
                                skip_leak_check=True)
    r_lgb = payload_lgb["results"][0]
    r_pers = payload_pers["results"][0]
    v_lgb = float(r_lgb.metrics["RMSE"])
    v_pers = float(r_pers.metrics["RMSE"])
    check("P6 lightgbm RMSE 合理量纲", 0.0 < v_lgb < 30.0, f"RMSE={v_lgb:.4f}")
    check("P6 lightgbm 不显著劣于 persistence",
          v_lgb <= v_pers + 5.0,
          f"lgb={v_lgb:.4f} pers={v_pers:.4f}")


def main():
    print("=" * 60)
    test_p1_loader()
    test_p2_ground_truth()
    test_p3_task_build()
    test_p4_forecast_features()
    test_p5_replay_persistence()
    test_p6_replay_lightgbm()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
