# ECL 跨用户迁移评测主循环
# ---------------------------------------------------------------
# 与 GEFCom 四赛道的「滚动回放」不同，ECL 是「跨用户迁移」：
#   train 用户 → 训练一个用户无关模型 → test 用户（训练时从未见过）逐用户预测 + RMSE。
#
# 评测协议：
#   - 逐 test 用户算 RMSE（各自在整个时间序列上预测，无跨用户、无未来泄漏）
#   - 汇总：mean / std / best / worst RMSE（跨用户）
# 复用：build_features（特征）/ LightGBMBackend / utils.metrics.rmse。
# ---------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from data.ecl_task_builder import (
    ECL_FEATURE_SPEC,
    ECL_TARGET_COL,
    build_migration_task,
)
from models.replay_backends import ModelBackend
from utils.metrics import rmse

DEFAULT_VAL_HOURS = 720  # 早停验证段：最后 30 天


def replay_ecl(
    backend: ModelBackend,
    n_train: int = 260,
    seed: int = 42,
    val_hours: int = DEFAULT_VAL_HOURS,
    data_dir: Path = None,
    spec: List[dict] = ECL_FEATURE_SPEC,
) -> dict:
    """
    ECL 跨用户迁移评测。返回 {per_user_rmse, summary, task}。

    - 训练：池化 n_train 个 train 用户的 (特征, target)，按时间切最后 val_hours 小时做早停。
    - 测试：test 用户逐用户预测 + RMSE（用户无关模型，无跨用户特征）。
    """
    task = build_migration_task(data_dir, n_train, seed, spec)

    # 按时间切验证集（早停用）：val = 最后 val_hours 小时（索引可重复，按时间切）
    cutoff = task.train_df.index.max() - pd.Timedelta(hours=val_hours)
    train = task.train_df[task.train_df.index < cutoff]
    val = task.train_df[task.train_df.index >= cutoff]

    backend.fit(train, val, task.feature_cols, task.target_col, seed)

    # 逐测试用户预测 + RMSE（模型 + 两个朴素基线）
    per_user_rmse: Dict[str, float] = {}
    per_user_naive_rmse: Dict[str, float] = {}  # persistence = lag_1
    per_user_snaive_rmse: Dict[str, float] = {}  # seasonal naive = lag_24
    for u in task.test_cols:
        f = task.test_frames[u]
        y_true = f[task.target_col].to_numpy(dtype=float)
        y_pred = backend.predict(f[task.feature_cols])
        per_user_rmse[u] = rmse(y_true, y_pred)
        per_user_naive_rmse[u] = rmse(y_true, f["lag_1"].to_numpy(dtype=float))
        per_user_snaive_rmse[u] = rmse(y_true, f["lag_24"].to_numpy(dtype=float))

    # 相对 RMSE（模型误差 / 朴素基线误差），量纲抵消，跨用户可比
    def _ratio_agg(model: Dict, naive: Dict) -> Dict[str, float]:
        ratios = np.array([
            model[u] / naive[u] for u in model if naive[u] > 1e-6
        ])
        if len(ratios) == 0:
            return {"mean": float("nan"), "median": float("nan"), "pct_better": float("nan")}
        return {
            "mean": float(ratios.mean()),          # <1 = 模型优于朴素
            "median": float(np.median(ratios)),
            "pct_better": float((ratios < 1).mean() * 100),  # 优于朴素的用户占比 %
        }

    rmses = np.array(list(per_user_rmse.values()))
    summary = {
        "model": backend.name,
        "n_train_users": len(task.train_cols),
        "n_test_users": len(task.test_cols),
        "n_train_rows": len(train),
        "n_val_rows": len(val),
        "mean_rmse": float(rmses.mean()),
        "std_rmse": float(rmses.std()),
        "best_rmse": float(rmses.min()),
        "worst_rmse": float(rmses.max()),
        "best_user": task.test_cols[int(rmses.argmin())],
        "worst_user": task.test_cols[int(rmses.argmax())],
        "seed": seed,
        # 相对指标：模型 / 朴素基线（<1 更优，跨用户可比）
        "ratio_vs_naive": _ratio_agg(per_user_rmse, per_user_naive_rmse),
        "ratio_vs_snaive": _ratio_agg(per_user_rmse, per_user_snaive_rmse),
    }

    return {
        "per_user_rmse": per_user_rmse,
        "per_user_naive_rmse": per_user_naive_rmse,
        "per_user_snaive_rmse": per_user_snaive_rmse,
        "summary": summary,
        "task": task,
    }
