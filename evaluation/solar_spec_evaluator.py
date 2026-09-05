# Solar 候选特征集评测器：自进化 Agent 的 decision metric（Solar 版）
# ---------------------------------------------------------------
# evaluate_solar_spec(task_id, zone, spec, protocol) -> {rmse, profile, ...}
#
# 与 Load 版 evaluation/spec_evaluator.evaluate_spec 同构：对候选特征集 spec
# 做一次完整评测，返回决策所需的全部信号。差异在数据集适配：
#   build_task (Load)      → build_solar_task (Solar Task×Zone)
#   rolling_predict (Load) → solar_rolling_predict
#   外生特征列（VAR169/164/167）由 spec 的 source 引用，评测链路直接透传。
#
# 泄漏检查：候选 spec 的静态血缘检查（含 tier 档位）由
# agent/evolution_runner 的 validate_spec_list 在评测前完成（复用 Pass A）。
# ---------------------------------------------------------------
from typing import Callable, Dict, List, Optional

import numpy as np

from data.solar_task_builder import (
    SOLAR_TARGET_COL,
    build_solar_task,
)
from evaluation.error_profiler import compute_error_profile
from evaluation.evaluator import evaluate_task
from evaluation.forecast_protocol import ForecastProtocol, ONLINE_H1
from evaluation.solar_replay import solar_rolling_predict
from models.replay_backends import LightGBMBackend, ModelBackend


def _default_backend_factory() -> ModelBackend:
    """默认后端：LightGBM（early-stopping 用 task.val_df）。"""
    return LightGBMBackend()


def evaluate_solar_spec(
    task_id: int,
    spec: List[dict],
    protocol: ForecastProtocol = ONLINE_H1,
    val_hours: int = 168,
    eval_hours: int = 0,
    backend_factory: Optional[Callable[[], ModelBackend]] = None,
    seed: int = 42,
    data_dir=None,
    zone: Optional[int] = None,
    target_col: str = SOLAR_TARGET_COL,
    exogenous_cols: Optional[List[str]] = None,
) -> Dict:
    """
    对候选特征集 spec 做一次完整 Solar 评测，返回决策所需的全部信号。

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
        raise ValueError("evaluate_solar_spec 需要 zone（Solar Task×Zone 逐分区评测）")
    if backend_factory is None:
        backend_factory = _default_backend_factory

    task = build_solar_task(task_id, zone, data_dir, val_hours, spec=spec)
    backend = backend_factory()
    backend.fit(task.train_df, task.val_df, task.feature_cols, task.target_col, seed)

    y_pred = solar_rolling_predict(backend, task, protocol, spec=spec)
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
