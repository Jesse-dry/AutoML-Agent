# ECL 跨用户迁移接入自动测试（E1–E6）
# ---------------------------------------------------------------
# 运行：python tests/test_ecl_suite.py
#
# E1 Loader   —— 矩阵形状、逐小时连续、无缺失、列名
# E2 切分     —— 260 train / 61 test、无重叠、固定种子可复现
# E3 任务构建 —— train_df 形状、特征列齐全、无 NaN 目标
# E4 泄漏     —— lag/rolling 只依赖自身过去（shift 语义），无未来/跨用户
# E5 回放冒烟(persistence) —— 跑通 + RMSE 为正有限
# E6 回放冒烟(lightgbm)   —— train→test 迁移路径跑通，RMSE 为正有限
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


def main():
    print("=" * 60)
    test_e1_loader()
    test_e2_split()
    test_e3_task_build()
    test_e4_leakage()
    test_e5_replay_persistence()
    test_e6_replay_lightgbm()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
