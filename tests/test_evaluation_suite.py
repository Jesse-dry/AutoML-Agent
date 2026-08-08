# 评测体系自动测试（阶段一：T1–T5 + 手工基线对账）
# ---------------------------------------------------------------
# 运行：python tests/test_evaluation_suite.py
#
# T1 Forecast boundary   —— 历史终点 < 预测起点，严格 +1h
# T2 Target poisoning    —— 改 LOAD[t]，feature[t] 不变，feature[t+1] 允许变
# T3 Future poisoning    —— 改 LOAD[t+1:]，feature[:t] 全部不变（强因果）
# T4 Information-policy  —— online_h1 回填真值 / recursive 全程禁真值
# T5 Train/inference parity —— build_features 与 _features_at 逐位一致
# T6 手工基线对账         —— persistence / naive_24 / naive_168 与 CLI 逐位一致
# ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.task_builder import TARGET_COL, build_features, build_task, FEATURE_SPEC
from data.availability import available_history
from evaluation.forecast_protocol import ONLINE_H1, RECURSIVE_MONTH_AHEAD
from evaluation.rolling_backtest import _features_at, build_forecast_features, rolling_predict
from models.replay_backends import SeasonalNaiveBackend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def rmse(y_true, y_hat):
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


# ---------------- T1 ----------------
def test_t1_boundary():
    t = build_task(1)
    check("T1 history < forecast start",
          t.history_df.index[-1] < t.forecast_ts[0],
          f"{t.history_df.index[-1]} < {t.forecast_ts[0]}")
    check("T1 forecast_start == history_end + 1h",
          t.forecast_ts[0] == t.history_df.index[-1] + pd.Timedelta(hours=1))
    check("T1 no forecast overlap", not t.forecast_ts.isin(t.history_df.index).any())


# ---------------- T2 ----------------
def test_t2_target_poisoning():
    t = build_task(1)
    hist = t.history_df
    probe = hist.index[1000]
    f_orig = build_features(hist)
    bad = hist.copy()
    bad.loc[probe, TARGET_COL] = 99999.0
    f_bad = build_features(bad)
    # feature[t] 不依赖 target[t]
    row_orig = f_orig.loc[probe]
    row_bad = f_bad.loc[probe]
    check("T2 feature[t] 不变", np.allclose(row_orig.values, row_bad.values, equal_nan=True),
          f"changed cols: {[c for c in f_orig.columns if not np.isclose(row_orig[c], row_bad[c], equal_nan=True)]}")
    # feature[t+1] 允许依赖 target[t]（如 lag_1）
    nxt = f_orig.index[f_orig.index.get_loc(probe) + 1]
    check("T2 feature[t+1] 随 target[t] 变化（lag_1）",
          f_bad.loc[nxt, "lag_1"] == 99999.0)


# ---------------- T3 ----------------
def test_t3_future_poisoning():
    t = build_task(1)
    hist = t.history_df
    cutoff = hist.index[5000]
    f_orig = build_features(hist)
    bad = hist.copy()
    bad.loc[bad.index > cutoff, TARGET_COL] = 1e9
    f_bad = build_features(bad)
    mask = f_orig.index <= cutoff
    check("T3 feature[:t] 全部不变",
          np.allclose(f_orig.loc[mask].values, f_bad.loc[mask].values, equal_nan=True))


# ---------------- T4 ----------------
def test_t4_information_policy():
    # 协议语义
    check("T4 online_h1 回填真值", ONLINE_H1.backfill(0.5, 10.0) == 10.0)
    check("T4 recursive 回填预测值（禁真值）", RECURSIVE_MONTH_AHEAD.backfill(0.5, 10.0) == 0.5)
    check("T4 recursive 标志", ONLINE_H1.recursive is False and RECURSIVE_MONTH_AHEAD.recursive is True)

    # 端到端：recursive 下 naive_24 输出必须与 online_h1（用真值回填）不同 ——
    # 因为预测月内的 lag_24 来自预测值而非真值，误差累积。
    t = build_task(1)
    nb = SeasonalNaiveBackend(24).fit(t.train_df, t.val_df, t.feature_cols, t.target_col)
    y_online = rolling_predict(nb, t, ONLINE_H1)
    y_recursive = rolling_predict(nb, t, RECURSIVE_MONTH_AHEAD)
    check("T4 recursive != online（禁止真值）",
          not np.allclose(y_recursive.values, y_online.values),
          f"recursive RMSE={rmse(t.y_true.values, y_recursive.values):.3f} "
          f"online RMSE={rmse(t.y_true.values, y_online.values):.3f}")


# ---------------- T5 ----------------
def test_t5_parity():
    t = build_task(1)
    observed = pd.concat([t.history_df[TARGET_COL], t.y_true])
    X = build_forecast_features(observed, t.forecast_ts)
    ok = True
    for ts in [t.forecast_ts[0], t.forecast_ts[100], t.forecast_ts[743]]:
        row_vec = _features_at(observed, ts)
        row_batch = X.loc[ts]
        for f in FEATURE_SPEC:
            v1 = row_vec[f["name"]]
            v2 = row_batch[f["name"]]
            if not (pd.isna(v1) and pd.isna(v2)) and not np.isclose(v1, v2):
                ok = False
    check("T5 train/inference 特征逐位一致", ok)


# ---------------- T6 ----------------
def test_t6_manual_baselines():
    t = build_task(1)
    observed = pd.concat([t.history_df[TARGET_COL], t.y_true])
    y_true = t.y_true.values
    expected = {
        "persistence": 7.188017,
        "seasonal_naive_24": 12.462008,
        "seasonal_naive_168": 28.468031,
    }
    for k, (name, exp) in {
        1: ("persistence", expected["persistence"]),
        24: ("seasonal_naive_24", expected["seasonal_naive_24"]),
        168: ("seasonal_naive_168", expected["seasonal_naive_168"]),
    }.items():
        yh = observed.shift(k).loc[t.forecast_ts].values
        got = rmse(y_true, yh)
        check(f"T6 {name} 手工对账", abs(got - exp) < 1e-4, f"got={got:.6f} exp={exp:.6f}")


def main():
    print("=" * 60)
    test_t1_boundary()
    test_t2_target_poisoning()
    test_t3_future_poisoning()
    test_t4_information_policy()
    test_t5_parity()
    test_t6_manual_baselines()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
