# Wind 候选特征集评测器：自进化 Agent 的 decision metric（Wind 版）
# ---------------------------------------------------------------
# evaluate_wind_spec(task_id, zone, spec, protocol) -> {rmse, profile, ...}
#
# 与 Load 版 evaluation/spec_evaluator.evaluate_spec / Solar 版
# evaluation/solar_spec_evaluator 同构。差异在数据集适配：
#   build_task (Load)        → build_wind_task (Wind Task×Zone)
#   rolling_predict (Load)   → wind_rolling_predict
#   气象派生特征（ws10/ws100 风速）由 spec 的 source 引用，评测链路直接透传。
#
# 泄漏检查：候选 spec 的静态血缘检查（含 tier 档位）由
# agent/evolution_runner 的 validate_spec_list 在评测前完成（复用 Pass A）。
# ---------------------------------------------------------------
from typing import Callable, Dict, List, Optional

import numpy as np

from data.wind_task_builder import (
    WIND_TARGET_COL,
    build_wind_task,
)
from evaluation.error_profiler import compute_error_profile
from evaluation.evaluator import evaluate_task
from evaluation.forecast_protocol import ForecastProtocol, ONLINE_H1
from evaluation.wind_replay import wind_rolling_predict
from models.replay_backends import LightGBMBackend, ModelBackend


def _default_backend_factory() -> ModelBackend:
    """默认后端：LightGBM（early-stopping 用 task.val_df）。"""
    return LightGBMBackend()


def evaluate_wind_spec(
    task_id: int,
    spec: List[dict],
    protocol: ForecastProtocol = ONLINE_H1,
    val_hours: int = 168,
    eval_hours: int = 0,
    backend_factory: Optional[Callable[[], ModelBackend]] = None,
    seed: int = 42,
    data_dir=None,
    zone: Optional[int] = None,
    target_col: str = WIND_TARGET_COL,
    exogenous_cols: Optional[List[str]] = None,
) -> Dict:
    """
    对候选特征集 spec 做一次完整 Wind 评测，返回决策所需的全部信号。

    Returns
    -------
    dict: {
        task_id, spec, backend_name,
        rmse, mae,                      # 决策窗口（预测月 online_h1）
        y_true, y_pred, forecast_ts,    # 决策窗口逐小时
        profile: ErrorProfile,          # 误差画像（喂 LLM）
        feature_importance: DataFrame|None,
        best_iteration: int|None,
    }
    """
    if zone is None:
        raise ValueError("evaluate_wind_spec 需要 zone（Wind Task×Zone 逐分区评测）")
    if backend_factory is None:
        backend_factory = _default_backend_factory

    task = build_wind_task(task_id, zone, data_dir, val_hours, spec=spec)
    backend = backend_factory()
    backend.fit(task.train_df, task.val_df, task.feature_cols, task.target_col, seed)

    y_pred = wind_rolling_predict(backend, task, protocol, spec=spec)
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
