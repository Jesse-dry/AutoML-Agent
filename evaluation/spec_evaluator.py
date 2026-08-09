# 候选特征集评测器：自进化 Agent 的 decision metric 抽象
# ---------------------------------------------------------------
# evaluate_spec(task_id, spec, protocol) -> {rmse, profile, ...}
#
# 候选 accept/reject 判定（用户已确认）：预测月 online_h1 滚动 RMSE。
# 等价一次 replay()：build_task(spec) → fit(train, val 早停) → rolling_predict。
# eval_hours>0 时可切换为历史末 val 窗口（P1 默认不用，接口已留）。
# ---------------------------------------------------------------
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from data.task_builder import GEFComTask, build_task
from evaluation.error_profiler import ErrorProfile, compute_error_profile
from evaluation.evaluator import evaluate_task
from evaluation.forecast_protocol import ForecastProtocol, ONLINE_H1
from evaluation.rolling_backtest import rolling_predict
from models.replay_backends import LightGBMBackend, ModelBackend


def _default_backend_factory() -> ModelBackend:
    """默认后端：LightGBM（early-stopping 用 task.val_df）。"""
    return LightGBMBackend()


def evaluate_spec(
    task_id: int,
    spec: List[dict],
    protocol: ForecastProtocol = ONLINE_H1,
    val_hours: int = 168,
    eval_hours: int = 0,
    backend_factory: Optional[Callable[[], ModelBackend]] = None,
    seed: int = 42,
    data_dir=None,
) -> Dict:
    """
    对候选特征集 spec 做一次完整评测，返回决策所需的全部信号。

    Returns
    -------
    dict: {
        task_id, spec, backend_name,
        rmse, mae,                      # 决策窗口（默认预测月 online_h1）
        y_true, y_pred, forecast_ts,    # 决策窗口逐小时
        profile: ErrorProfile,          # 误差画像（喂 LLM）
        feature_importance: DataFrame|None,
        best_iteration: int|None,
    }
    """
    if backend_factory is None:
        backend_factory = _default_backend_factory

    task: GEFComTask = build_task(
        task_id, data_dir=data_dir, val_hours=val_hours, spec=spec, eval_hours=eval_hours
    )
    backend = backend_factory()
    backend.fit(task.train_df, task.val_df, task.feature_cols, task.target_col, seed)

    y_pred = rolling_predict(backend, task, protocol, spec=spec)
    res = evaluate_task(task, y_pred, backend.name, protocol.name)

    y_true_arr = np.asarray(task.y_true.to_numpy(dtype=float))
    y_pred_arr = np.asarray(y_pred.reindex(task.forecast_ts).to_numpy(dtype=float))
    profile = compute_error_profile(y_true_arr, y_pred_arr, task.forecast_ts)

    importance = None
    if hasattr(backend, "feature_importance"):
        importance = backend.feature_importance()

    best_iter = getattr(backend, "best_iteration", None)

    return {
        "task_id": task_id,
        "spec": spec,
        "backend_name": backend.name,
        "rmse": float(res.metrics["RMSE"]),
        "mae": float(res.metrics["MAE"]) if res.metrics.get("MAE") is not None else None,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
        "forecast_ts": task.forecast_ts,
        "profile": profile,
        "feature_importance": importance,
        "best_iteration": best_iter,
    }
