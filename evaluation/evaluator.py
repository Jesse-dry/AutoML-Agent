# 评测指标 + 多 Task 汇总
# ---------------------------------------------------------------
# 指标复用 utils/metrics.compute_all_metrics（RMSE/MAE/MAPE/SMAPE/R2/N），
# 保证与既有模型横向可比。
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from data.task_builder import GEFComTask
from utils.metrics import compute_all_metrics


@dataclass
class TaskResult:
    task_id: int
    model: str
    protocol: str
    y_true: np.ndarray
    y_pred: np.ndarray
    forecast_ts: pd.DatetimeIndex
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def error(self) -> np.ndarray:
        return np.asarray(self.y_true) - np.asarray(self.y_pred)


def evaluate_task(
    task: GEFComTask,
    y_pred: pd.Series,
    model_name: str,
    protocol: str = "online_h1",
) -> TaskResult:
    """对单个 Task 计算指标。y_pred 须以 forecast_ts 为索引。"""
    y_true = task.y_true.to_numpy(dtype=float)
    yp = y_pred.reindex(task.forecast_ts).to_numpy(dtype=float)
    metrics = compute_all_metrics(y_true, yp, prefix="")
    return TaskResult(
        task_id=task.task_id,
        model=model_name,
        protocol=protocol,
        y_true=y_true,
        y_pred=yp,
        forecast_ts=task.forecast_ts,
        metrics=metrics,
    )


_METRIC_COLS = ["RMSE", "MAE", "MAPE", "SMAPE", "R2", "N"]


def summarize(results: List[TaskResult]) -> Dict[str, object]:
    """
    汇总多 Task 结果：
      table  — 每 Task 一行指标
      summary — mean/std/worst/best RMSE（+ worst/best task）
    """
    if not results:
        raise ValueError("results 为空，无法汇总")

    rows = []
    for r in results:
        row = {"task_id": r.task_id}
        for c in _METRIC_COLS:
            row[c] = r.metrics.get(c, None)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("task_id")

    rmses = table["RMSE"].astype(float)
    summary = {
        "model": results[0].model,
        "protocol": results[0].protocol,
        "mean_rmse": float(rmses.mean()),
        "std_rmse": float(rmses.std()),
        "worst_rmse": float(rmses.max()),
        "best_rmse": float(rmses.min()),
        "worst_task": int(rmses.idxmax()),
        "best_task": int(rmses.idxmin()),
        "n_tasks": len(results),
    }
    if table["MAE"].notna().any():
        summary["mean_mae"] = float(table["MAE"].astype(float).mean())
    if table["MAPE"].notna().any():
        summary["mean_mape"] = float(table["MAPE"].astype(float).mean())
    if table["R2"].notna().any():
        summary["mean_r2"] = float(table["R2"].astype(float).mean())

    return {"table": table, "summary": summary}
