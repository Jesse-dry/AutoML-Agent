# ECL 统一评测协议（REVIEW 复核）
# ---------------------------------------------------------------
# 目标：所有模型（LightGBM / PatchTST / persistence / seasonal naive）
#       在【完全相同的用户、时间戳、有效掩码】上评测，保证可比。
#
# 统一协议：
#   Train   260 train 用户；目标时间     t < 2014-06-01 00:00
#   Val     260 train 用户；2014-06-01 00:00 <= t < 2014-07-01 00:00（早停）
#   Test    61  test 用户；2014-07-01 00:00 <= t <= 2014-12-31 23:00（迁移评测）
#   统一做 online one-step ahead：预测 t 只用该用户 <= t-1 的真实历史。
#
# 指标：逐用户 RMSE（model / persistence=lag_1 / seasonal naive=lag_24），
#       统一有效掩码（snaive 非 NaN，即预测样本三者都有值），
#       再汇总 mean/median/std + ratio 与胜出比例。
# ---------------------------------------------------------------
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRAIN_END = pd.Timestamp("2014-06-01 00:00:00")
VAL_START = pd.Timestamp("2014-06-01 00:00:00")
VAL_END = pd.Timestamp("2014-07-01 00:00:00")
TEST_START = pd.Timestamp("2014-07-01 00:00:00")
TEST_END = pd.Timestamp("2014-12-31 23:00:00")

LAG1_COL = "lag_1"
LAG24_COL = "lag_24"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def hash_list(users: List[str]) -> str:
    canonical = json.dumps(sorted(users), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def score_users(
    user_actual: Dict[str, np.ndarray],
    user_model_pred: Dict[str, np.ndarray],
    user_persist: Dict[str, np.ndarray],
    user_snaive: Dict[str, np.ndarray],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
    """
    对同一批用户、同一掩码计算三个 RMSE。

    输入各为用户 → 已对齐的预测/真值数组（长度一致，对应相同时间戳）。
    统一掩码：snaive 非 NaN（真值/模型/持久化均需非 NaN）。
    """
    model_rmse, persist_rmse, snaive_rmse, n_pred = {}, {}, {}, {}
    for u in user_actual:
        y_true = np.asarray(user_actual[u], dtype=float)
        y_pred = np.asarray(user_model_pred[u], dtype=float)
        y_pers = np.asarray(user_persist[u], dtype=float)
        y_sna = np.asarray(user_snaive[u], dtype=float)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_pers) | np.isnan(y_sna))
        y_true, y_pred = y_true[mask], y_pred[mask]
        y_pers, y_sna = y_pers[mask], y_sna[mask]
        if len(y_true) == 0:
            continue
        model_rmse[u] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        persist_rmse[u] = float(np.sqrt(np.mean((y_true - y_pers) ** 2)))
        snaive_rmse[u] = float(np.sqrt(np.mean((y_true - y_sna) ** 2)))
        n_pred[u] = int(len(y_true))
    return model_rmse, persist_rmse, snaive_rmse, n_pred


def _ratio_stats(model: Dict[str, float], naive: Dict[str, float]) -> Dict[str, float]:
    ratios = np.array(
        [model[u] / naive[u] for u in model if naive.get(u, 0) > 1e-6]
    )
    if len(ratios) == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "pct_better": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(ratios.mean()),
        "median": float(np.median(ratios)),
        "pct_better": float((ratios < 1).mean() * 100),
        "min": float(ratios.min()),
        "max": float(ratios.max()),
    }


def summarize(
    model_rmse: Dict[str, float],
    persist_rmse: Dict[str, float],
    snaive_rmse: Dict[str, float],
    n_pred: Dict[str, int],
    model_name: str,
    n_train_users: int,
    n_test_users: int,
    n_train_windows: Optional[int] = None,
    n_val_windows: Optional[int] = None,
    best_epoch: Optional[int] = None,
    best_val_rmse: Optional[float] = None,
) -> Dict[str, object]:
    """统一汇总：mean/median/std RMSE + ratio vs persistence/snaive。"""
    users = list(model_rmse)
    m = np.array([model_rmse[u] for u in users], dtype=float)
    p = np.array([persist_rmse[u] for u in users], dtype=float)
    s = np.array([snaive_rmse[u] for u in users], dtype=float)

    return {
        "model": model_name,
        "n_users": len(users),
        "n_train_users": n_train_users,
        "n_test_users": n_test_users,
        "n_total_predictions": int(sum(n_pred.values())),
        "mean_rmse": float(m.mean()),
        "median_rmse": float(np.median(m)),
        "std_rmse": float(m.std()),
        "best_rmse": float(m.min()) if len(m) else None,
        "worst_rmse": float(m.max()) if len(m) else None,
        "best_user": users[int(np.argmin(m))] if len(m) else None,
        "worst_user": users[int(np.argmax(m))] if len(m) else None,
        "mean_persistence_rmse": float(p.mean()),
        "mean_snaive_rmse": float(s.mean()),
        "ratio_vs_persistence": _ratio_stats(model_rmse, persist_rmse),
        "ratio_vs_snaive": _ratio_stats(model_rmse, snaive_rmse),
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "protocol": {
            "train_until": str(TRAIN_END),
            "val": f"{VAL_START} ~ {VAL_END}",
            "test": f"{TEST_START} ~ {TEST_END}",
            "horizon": "online one-step ahead (tes 只用历史, test 用户未见)",
        },
    }


def user_rows(
    model_rmse: Dict[str, float],
    persist_rmse: Dict[str, float],
    snaive_rmse: Dict[str, float],
    n_pred: Dict[str, int],
) -> pd.DataFrame:
    """逐用户指标表（REVIEW 要求 per_user_metrics.csv）。"""
    rows = []
    for u in model_rmse:
        rows.append({
            "user": u,
            "model_rmse": model_rmse[u],
            "persistence_rmse": persist_rmse[u],
            "seasonal_naive_rmse": snaive_rmse[u],
            "ratio_vs_persistence": (
                model_rmse[u] / persist_rmse[u] if persist_rmse[u] > 1e-6 else None),
            "ratio_vs_snaive": (
                model_rmse[u] / snaive_rmse[u] if snaive_rmse[u] > 1e-6 else None),
            "n_predictions": n_pred[u],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("user").reset_index(drop=True)
    return df


def write_manifest(path: Path, model_name: str, args: dict,
                   train_users: List[str], test_users: List[str],
                   n_train_windows: int = None, n_val_windows: int = None,
                   best_epoch: int = None, best_val_rmse: float = None,
                   extra: Optional[dict] = None) -> None:
    """run_manifest.json：REVIEW 要求的最低内容。"""
    manifest = {
        "model": model_name,
        "git_commit": git_commit(),
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cpu",
        "python": __import__("sys").version.split()[0],
        "command_args": args,
        "seed": args.get("seed"),
        "data": {
            "source": "ECL/electricity.txt (preprocessed, laiguokun)",
            "n_users": 321,
            "user_split_seed": args.get("seed"),
            "n_train_users": len(train_users),
            "n_test_users": len(test_users),
            "train_users_hash": hash_list(train_users),
            "test_users_hash": hash_list(test_users),
        },
        "time_split": {
            "train_until": str(TRAIN_END),
            "val": f"{VAL_START} ~ {VAL_END}",
            "test": f"{TEST_START} ~ {TEST_END}",
        },
        "horizon": "online one-step ahead",
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")