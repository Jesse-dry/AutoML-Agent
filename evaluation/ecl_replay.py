# ECL 跨用户迁移评测主循环（统一评测协议）
# ---------------------------------------------------------------
# 与 GEFCom 四赛道的「滚动回放」不同，ECL 是「跨用户迁移」：
#   train 用户 → 训练用户无关模型 → test 用户（从未见过）逐用户预测。
#
# 统一评测协议（见 evaluation/ecl_protocol.py，REVIEW 复核）：
#   Train  260 train 用户；目标时间 t < 2014-06-01
#   Val    260 train 用户；2014-06-01 <= t < 2014-07-01（早停）
#   Test   61  test 用户； 2014-07-01 <= t <= 2014-12-31（迁移评测）
#   统一 online one-step ahead：预测 t 只用该用户 <= t-1 的真实历史。
#   指标 = model / persistence(lag_1) / seasonal naive(lag_24) 三者在
#          相同 test 用户、相同时间戳、相同有效掩码上计算。
# 复用：build_features / LightGBMBackend / ecl_protocol.score_users。
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
from evaluation import ecl_protocol as ep
from models.replay_backends import ModelBackend
from utils.metrics import rmse


def replay_ecl(
    backend: ModelBackend,
    n_train: int = 260,
    seed: int = 42,
    data_dir: Path = None,
    spec: List[dict] = ECL_FEATURE_SPEC,
) -> dict:
    """
    ECL 跨用户迁移评测（统一协议）。

    Returns {per_user_rmse, per_user_persist_rmse, per_user_snaive_rmse,
             user_n_pred, summary, task, predictions}
    """
    task = build_migration_task(data_dir, n_train, seed, spec)

    # ---- 按统一时间边界切 train / val（train 用户）----
    all_train = task.train_df
    train = all_train[all_train.index < ep.TRAIN_END]
    val = all_train[(all_train.index >= ep.VAL_START) & (all_train.index < ep.VAL_END)]

    # 断言严格时间切分（REVIEW：验证集不来自留出窗随机）
    assert len(train) > 0 and len(val) > 0, "train/val 为空，切分出错"
    assert train.index.max() < val.index.min(), "train 与 val 目标时间未严格分离"

    backend.fit(train, val, task.feature_cols, task.target_col, seed)

    # ---- 逐 test 用户：test 时间段内预测 + 基线 ----
    user_actual, user_model, user_persist, user_snaive = {}, {}, {}, {}
    predictions_rows = []
    for u in task.test_cols:
        f = task.test_frames[u]
        f_test = f[(f.index >= ep.TEST_START) & (f.index <= ep.TEST_END)]
        if len(f_test) == 0:
            continue
        y_true = f_test[task.target_col].to_numpy(dtype=float)
        y_pred = backend.predict(f_test[task.feature_cols])
        y_pers = f_test["lag_1"].to_numpy(dtype=float)   # persistence = lag_1
        y_sna = f_test["lag_24"].to_numpy(dtype=float)   # seasonal naive = lag_24

        user_actual[u] = y_true
        user_model[u] = y_pred
        user_persist[u] = y_pers
        user_snaive[u] = y_sna

        for i, ts in enumerate(f_test.index):
            predictions_rows.append({
                "user": u, "timestamp": ts,
                "actual": y_true[i], "prediction": y_pred[i],
                "persistence": y_pers[i], "seasonal_naive": y_sna[i],
            })

    model_rmse, persist_rmse, snaive_rmse, n_pred = ep.score_users(
        user_actual, user_model, user_persist, user_snaive,
    )
    summary = ep.summarize(
        model_rmse, persist_rmse, snaive_rmse, n_pred,
        model_name=backend.name,
        n_train_users=len(task.train_cols),
        n_test_users=len(task.test_cols),
        n_train_windows=len(train),
        n_val_windows=len(val),
    )

    return {
        "per_user_rmse": model_rmse,
        "per_user_persist_rmse": persist_rmse,
        "per_user_snaive_rmse": snaive_rmse,
        "user_n_pred": n_pred,
        "summary": summary,
        "task": task,
        "predictions": pd.DataFrame(predictions_rows),
        "train_users": task.train_cols,
        "test_users": task.test_cols,
    }