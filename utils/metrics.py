"""
通用回归评估指标
================
所有模型（LightGBM、LSTM、Transformer 等）共用一套指标函数，
保证输出格式一致，方便后续 Agent 对比不同模型。

用法：
    from utils.metrics import compute_all_metrics

    metrics = compute_all_metrics(y_true, y_pred, prefix="val_")
    # → {"val_RMSE": ..., "val_MAE": ..., "val_MAPE": ..., ...}
"""

import numpy as np
from typing import Dict, Optional


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差 (Root Mean Square Error)"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差 (Mean Absolute Error)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 1e-8,
) -> Optional[float]:
    """
    平均绝对百分比误差 (Mean Absolute Percentage Error)

    自动跳过真实值接近零的样本，避免除零导致无穷大。
    若所有真实值都为零，返回 None。

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    zero_threshold : float
        真实值绝对值小于此阈值视为零，跳过

    Returns
    -------
    float or None
    """
    mask = np.abs(y_true) > zero_threshold
    if mask.sum() == 0:
        return None
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    )


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float = 1e-8,
) -> Optional[float]:
    """
    对称平均绝对百分比误差 (Symmetric MAPE)

    SMAPE = 200% * mean(|y - ŷ| / (|y| + |ŷ|))
    值域 [0, 200]，比 MAPE 更稳定。
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > zero_threshold
    if mask.sum() == 0:
        return None
    return float(np.mean(200.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 (R² Score)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, object]:
    """
    一键计算所有回归指标，返回结构化字典。

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    prefix : str
        指标名前缀，如 "val_" 或 "test_"

    Returns
    -------
    dict: {
        "{prefix}RMSE": float,
        "{prefix}MAE": float,
        "{prefix}MAPE": float | None,
        "{prefix}SMAPE": float | None,
        "{prefix}R2": float,
        "{prefix}N": int,
    }
    """
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    return {
        f"{prefix}RMSE": round(rmse(y_true, y_pred), 4),
        f"{prefix}MAE": round(mae(y_true, y_pred), 4),
        f"{prefix}MAPE": round(mape_val, 4) if mape_val is not None else None,
        f"{prefix}SMAPE": round(smape_val, 4) if smape_val is not None else None,
        f"{prefix}R2": round(r2_score(y_true, y_pred), 4),
        f"{prefix}N": len(y_true),
    }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    yt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    yp = np.array([110.0, 190.0, 310.0, 390.0, 510.0])

    print("真实值:", yt)
    print("预测值:", yp)
    print()
    print("单项指标:")
    print(f"  RMSE  = {rmse(yt, yp):.4f}")
    print(f"  MAE   = {mae(yt, yp):.4f}")
    print(f"  MAPE  = {mape(yt, yp):.4f}%")
    print(f"  SMAPE = {smape(yt, yp):.4f}%")
    print(f"  R2    = {r2_score(yt, yp):.4f}")
    print()

    # 结构化输出
    import json

    metrics = compute_all_metrics(yt, yp, prefix="test_")
    print("结构化指标 (JSON):")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # 边界测试：含零的真实值
    yt_zero = np.array([0.0, 100.0, 0.0, 200.0])
    yp_zero = np.array([5.0, 110.0, 0.0, 190.0])
    print(f"\n零值边界测试: MAPE = {mape(yt_zero, yp_zero):.4f}%")
