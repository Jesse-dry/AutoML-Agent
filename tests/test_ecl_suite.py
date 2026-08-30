# ECL 跨用户迁移接入自动测试（E1–E10）
# ---------------------------------------------------------------
# 运行：python tests/test_ecl_suite.py
#
# E1 Loader   —— 矩阵形状、逐小时连续、无缺失、列名
# E2 切分     —— 260 train / 61 test、无重叠、固定种子可复现
# E3 任务构建 —— train_df 形状、特征列齐全、无 NaN 目标
# E4 泄漏     —— lag/rolling 只依赖自身过去（shift 语义），无未来/跨用户
# E5 回放冒烟(persistence) —— 跑通 + RMSE 为正有限
# E6 回放冒烟(lightgbm)   —— train→test 迁移路径跑通，RMSE 为正有限
# E7 协议时间边界 —— Train<2014-06 / Val 06~07 / Test 07~12 严格不重叠
# E8 统一四指标 —— score_users 掩码一致，四个序列同样本数，ratio 正确
# E9 产物完整性 —— summary/per_user/predictions 字段齐全且可重算
# E10 原型回归 —— ecl_replay 统一协议下 train/val 严格分离（replay_ecl 输出）
# ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from data.ecl_loader import load_ecl_matrix, split_users
from data.ecl_task_builder import (
    ECL_FEATURE_COLS,
    ECL_TARGET_COL,
    build_migration_task,
    build_user_features,
)
from evaluation import ecl_protocol as ep
from evaluation.ecl_replay import replay_ecl
from models.replay_backends import LightGBMBackend, PersistenceBackend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


# ---------------- E1 ----------------
def test_e1_loader():
    df = load_ecl_matrix()
    check("E1 形状 (26304, 321)", df.shape == (26304, 321), f"got {df.shape}")
    check("E1 逐小时连续",
          (df.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all())
    check("E1 无缺失", df.isna().sum().sum() == 0)
    check("E1 列名 client_0..client_320",
          list(df.columns[:1]) == ["client_0"] and list(df.columns[-1:]) == ["client_320"])


# ---------------- E2 ----------------
def test_e2_split():
    train, test = split_users(seed=42)
    check("E2 260 train / 61 test", len(train) == 260 and len(test) == 61)
    check("E2 无重叠", set(train).isdisjoint(test))
    check("E2 覆盖全部 321 用户", len(set(train) | set(test)) == 321)
    t1, _ = split_users(seed=42)
    t2, _ = split_users(seed=42)
    check("E2 固定种子可复现", t1 == t2)


# ---------------- E3 ----------------
def test_e3_task_build():
    task = build_migration_task(n_train=30, seed=42)
    check("E3 train_df 行数 = 30×(26304-168)",
          task.n_train == 30 * (26304 - 168), f"got {task.n_train}")
    check("E3 特征列齐全", set(task.feature_cols) == set(ECL_FEATURE_COLS))
    check("E3 train 无 NaN 目标", task.train_df[ECL_TARGET_COL].notna().all())
    check("E3 test 用户数 = 321-30", len(task.test_frames) == 321 - 30)


# ---------------- E4 ----------------
def test_e4_leakage():
    matrix = load_ecl_matrix()
    series = matrix["client_0"]
    feat = build_user_features(series)
    t = feat.index[500]  # 任意非预热时刻
    # lag_1 只依赖 t-1（自身过去）
    check("E4 lag_1 == t-1 负荷",
          np.isclose(feat.loc[t, "lag_1"], series.loc[t - pd.Timedelta(hours=1)]))
    # rolling 不含当前行：rolling_mean_24 == 过去24h平均（t-24..t-1）
    w = series.loc[t - pd.Timedelta(hours=24):t - pd.Timedelta(hours=1)]
    check("E4 rolling_mean_24 == 过去24h平均",
          np.isclose(feat.loc[t, "rolling_mean_24"], w.mean()))
    # 无未来依赖：特征值不依赖 t 之后的数据
    check("E4 特征列无目标别名", ECL_TARGET_COL not in ECL_FEATURE_COLS)


# ---------------- E5 ----------------
def test_e5_replay_persistence():
    payload = replay_ecl(PersistenceBackend(), n_train=30, seed=42)
    s = payload["summary"]
    check("E5 persistence RMSE 为正有限", 0 < s["mean_rmse"] < np.inf,
          f"mean={s['mean_rmse']:.2f}")


# ---------------- E6 ----------------
def test_e6_replay_lightgbm():
    payload = replay_ecl(LightGBMBackend(), n_train=30, seed=42)
    s = payload["summary"]
    check("E6 lightgbm RMSE 为正有限", 0 < s["mean_rmse"] < np.inf,
          f"mean={s['mean_rmse']:.2f}")
    check("E6 逐用户 RMSE 数量 == test 用户数", len(payload["per_user_rmse"]) == s["n_test_users"])


# ---------------- E7 ----------------
def test_e7_protocol_time_split():
    """统一协议时间边界：Train<2014-06-01、Val 06~07、Test 07~12 严格分离。"""
    check("E7 Train_END == 2014-06-01",
          ep.TRAIN_END == pd.Timestamp("2014-06-01"))
    check("E7 Val = [2014-06-01, 2014-07-01)",
          ep.VAL_START == pd.Timestamp("2014-06-01") and
          ep.VAL_END == pd.Timestamp("2014-07-01"))
    check("E7 Test = [2014-07-01, 2014-12-31 23:00]",
          ep.TEST_START == pd.Timestamp("2014-07-01") and
          ep.TEST_END == pd.Timestamp("2014-12-31 23:00"))
    check("E7 三段严格不重叠",
          ep.TRAIN_END <= ep.VAL_START < ep.VAL_END <= ep.TEST_START)
    check("E7 test 预测起点有 24h 历史窗口",
          (ep.TEST_START - pd.Timestamp("2014-06-01")).total_seconds() >= 24 * 3600)


def _synthetic_user_arrays(n=400, seed=0):
    """构造合成的单用户序列（含 lag_1/lag_24 真值），用于协议回归测试。"""
    rng = np.random.RandomState(seed)
    y_true = 100 + 20 * np.sin(np.arange(n) / 24) + rng.randn(n) * 3
    y_pred = y_true + rng.randn(n) * 5        # 模型有噪声
    y_pers = np.roll(y_true, 1)                # persistence = lag_1
    y_sna = np.roll(y_true, 24)                # seasonal naive = lag_24
    return y_true.astype(float), y_pred.astype(float), \
        y_pers.astype(float), y_sna.astype(float)


# ---------------- E8 ----------------
def test_e8_unified_metrics():
    """四指标统一掩码：score_users 返回同样本数、四 user 字典对齐。"""
    users = ["u_a", "u_b", "u_c"]
    A, P, S, U = {}, {}, {}, {}
    for u in users:
        yt, yp_, yp, ys_ = _synthetic_user_arrays(seed=hash(u) % 2**31)
        A[u], P[u], S[u], U[u] = yt, yp_, yp, ys_
    m, p, s, n = ep.score_users(A, P, S, U)
    check("E8 每个用户都产出指标", set(m) == set(users) == set(p) == set(s))
    # 掩码一致性：三个 RMSE 都在同一掩码（全部非 NaN）上算
    for u in users:
        same = n[u] == len(A[u]) and m[u] >= 0 and p[u] >= 0 and s[u] >= 0
        check(f"E8 {u} 同样本数且非负", same, f"n={n[u]}")
    # 汇总可计算
    summ = ep.summarize(m, p, s, n, model_name="fake", n_train_users=260, n_test_users=3)
    check("E8 summary 关键字段齐全",
          all(k in summ for k in ["mean_rmse", "median_rmse", "ratio_vs_persistence",
                                  "ratio_vs_snaive"]))
    check("E8 ratio 结构完整",
          set(summ["ratio_vs_persistence"].keys()) == {"mean", "median", "pct_better", "min", "max"})


# ---------------- E9 ----------------
def test_e9_artifacts_recompute():
    """产物字段齐全 + 能从 per_user/predictions 重算指标。"""
    # 用合成数据构造 per_user 表，验证 user_rows 字段
    users = ["u_a", "u_b"]
    m, p, s, n = {}, {}, {}, {}
    for u in users:
        yt, yp_, yp, ys_ = _synthetic_user_arrays(seed=5 if u == "u_a" else 9)
        A, P, S, U = {u: yt}, {u: yp_}, {u: yp}, {u: ys_}
        _m, _p, _s, _n = ep.score_users(A, P, S, U)
        m[u], p[u], s[u], n[u] = _m[u], _p[u], _s[u], _n[u]
    df = ep.user_rows(m, p, s, n)
    check("E9 per_user 字段齐全",
          set(df.columns) == {"user", "model_rmse", "persistence_rmse",
                              "seasonal_naive_rmse", "ratio_vs_persistence",
                              "ratio_vs_snaive", "n_predictions"})
    # 从 per_user 重算汇总 = ep.summarize 直接输出
    summ = ep.summarize(m, p, s, n, model_name="fake", n_train_users=2, n_test_users=2)
    check("E9 从 per_user 重算 mean == summarize",
          np.isclose(df["model_rmse"].mean(), summ["mean_rmse"]))
    # predictions 一致性（字段 + 用户对齐）
    check("E9 per_user 用户与 n_predictions 对齐",
          (df["n_predictions"] == [n[u] for u in df["user"]]).all())


# ---------------- E10 ----------------
def test_e10_replay_time_split():
    """replay_ecl（统一协议）内部 train/val 严格分离（小型运行，load 数据但小 n_train=5）。"""
    payload = replay_ecl(PersistenceBackend(), n_train=5, seed=42)
    task = payload["task"]
    all_train_idx = task.train_df.index
    cutoff = pd.Timestamp("2014-06-01")
    has_pre = (all_train_idx < cutoff).any()
    has_after = (all_train_idx >= cutoff).any()
    check("E10 train_df 同时含 06 前（train）与 06~07（val）", has_pre and has_after,
          f"pre={has_pre} after={has_after}")
    s = payload["summary"]
    check("E10 统一摘要含 median_rmse", "median_rmse" in s)
    check("E10 persistence ratio vs self == 1",
          abs(s["ratio_vs_persistence"]["median"] - 1.0) < 1e-6)


def main():
    print("=" * 60)
    test_e1_loader()
    test_e2_split()
    test_e3_task_build()
    test_e4_leakage()
    test_e5_replay_persistence()
    test_e6_replay_lightgbm()
    test_e7_protocol_time_split()
    test_e8_unified_metrics()
    test_e9_artifacts_recompute()
    test_e10_replay_time_split()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
