# ECL 跨用户迁移任务构建器
# ---------------------------------------------------------------
# 职责（ECL 特有，非滚动回放）：
#   1. ECL_FEATURE_SPEC —— 血缘式特征规格：time + 自身 lag/rolling。
#      source 为 ECL_TARGET_COL（"load"，即每个用户自己的负荷序列），
#      复用 data/task_builder.build_features 作为唯一严格过去向构造器。
#   2. build_user_features —— 对单个用户序列构建特征（无跨用户特征）。
#   3. build_migration_task —— 池化 train 用户的 (特征, target) 成用户无关训练集；
#      test 用户各自保留 (特征, target) 供逐用户评测。
#
# 迁移实验设计（跨用户迁移，不是时间回放）：
#   - train：随机 260 个用户，池化成一个大训练集（模型学「time + 自身 lag → 负荷」的
#     跨用户通用规律，不识别具体用户）。
#   - test：随机 61 个用户（模型训练时从未见过），逐用户预测 + 逐用户 RMSE。
#   - 特征无 user_id：用户规模差异（大户/小户）由「自身 lag」隐式携带
#     （lag_1 大的用户 → 预测也大）。
#
# 因果性约定（与 Load/Wind/Solar 一致）：
#   - lag/rolling 只依赖该用户 ≤ t-1 的自身负荷（shift 语义，无自泄露）
#   - time 特征 = 目标小时的日历属性，已知
#   - 无跨用户特征，故无跨用户泄漏
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd

from data.ecl_loader import (
    DEFAULT_N_TRAIN,
    DEFAULT_SEED,
    load_ecl_matrix,
    split_users,
)
from data.task_builder import MAX_LAG, build_features

ECL_TARGET_COL = "load"


# ---------------------------------------------------------------
# 血缘式特征规格（顺序即特征列顺序）
# ---------------------------------------------------------------

def _lag(name: str, source: str, k: int) -> dict:
    return {
        "name": name, "type": "lag", "source": source, "k": k,
        "lookback_start": -k, "lookback_end": -k, "uses_current_target": False,
    }


def _rolling(name: str, source: str, window: int, stat: str) -> dict:
    return {
        "name": name, "type": "rolling", "source": source,
        "window": window, "stat": stat, "min_periods": window,
        "lookback_start": -window, "lookback_end": -1, "uses_current_target": False,
    }


def _time(name: str, attr: str) -> dict:
    return {
        "name": name, "type": "time", "attr": attr,
        "lookback_start": 0, "lookback_end": 0, "uses_current_target": False,
    }


ECL_FEATURE_SPEC: List[dict] = [
    _time("hour", "hour"),
    _time("weekday", "weekday"),
    _time("month", "month"),
    _time("is_weekend", "is_weekend"),
    _lag("lag_1", ECL_TARGET_COL, 1),
    _lag("lag_24", ECL_TARGET_COL, 24),
    _lag("lag_168", ECL_TARGET_COL, 168),
    _rolling("rolling_mean_24", ECL_TARGET_COL, 24, "mean"),
    _rolling("rolling_std_24", ECL_TARGET_COL, 24, "std"),
    _rolling("rolling_mean_168", ECL_TARGET_COL, 168, "mean"),
]

ECL_FEATURE_COLS: List[str] = [s["name"] for s in ECL_FEATURE_SPEC]


# ---------------------------------------------------------------
# 单用户特征构建
# ---------------------------------------------------------------

def build_user_features(
    series: pd.Series,
    spec: List[dict] = ECL_FEATURE_SPEC,
    warmup: int = MAX_LAG,
) -> pd.DataFrame:
    """单个用户的负荷序列 → 特征 + target（datetime 索引）。

    特征 = time + 该用户自身 lag/rolling（无跨用户列）。
    drop 前 warmup 行（lag_168/rolling_168 的预热 NaN）。
    """
    frame = pd.DataFrame({ECL_TARGET_COL: series})
    feat = build_features(frame, spec=spec, target_col=ECL_TARGET_COL)
    feat[ECL_TARGET_COL] = series.values
    feat = feat.iloc[warmup:]
    return feat


# ---------------------------------------------------------------
# 迁移任务
# ---------------------------------------------------------------

@dataclass
class EclMigrationTask:
    """一次跨用户迁移实验的完整对象。"""

    train_df: pd.DataFrame  # 池化的训练集（特征 + target，用户无关，无 NaN）
    test_frames: Dict[str, pd.DataFrame]  # {user: 特征+target（datetime 索引）}
    feature_cols: List[str]
    target_col: str
    train_cols: List[str]
    test_cols: List[str]
    n_train: int = 0  # 训练集总行数
    n_test: int = 0  # 测试集总行数


def build_migration_task(
    data_dir: Path = None,
    n_train: int = DEFAULT_N_TRAIN,
    seed: int = DEFAULT_SEED,
    spec: List[dict] = ECL_FEATURE_SPEC,
) -> EclMigrationTask:
    """构建跨用户迁移任务：池化 train 用户 → 训练集；test 用户各自保留 → 评测。"""
    if data_dir is None:
        from data.ecl_loader import ECL_DATA_DIR
        data_dir = ECL_DATA_DIR

    matrix = load_ecl_matrix(data_dir)
    train_cols, test_cols = split_users(
        n_users=matrix.shape[1], n_train=n_train, seed=seed
    )
    feature_cols = [s["name"] for s in spec]

    # 池化 train 用户（用户无关；保留 datetime 索引以支持按时间切 val，索引可重复）
    train_frames = [build_user_features(matrix[u], spec) for u in train_cols]
    train_df = pd.concat(train_frames)
    train_df = train_df.dropna(subset=feature_cols + [ECL_TARGET_COL])

    # test 用户各自保留（保留 datetime 索引以对齐真值）
    test_frames = {}
    for u in test_cols:
        f = build_user_features(matrix[u], spec)
        f = f.dropna(subset=feature_cols + [ECL_TARGET_COL])
        test_frames[u] = f

    return EclMigrationTask(
        train_df=train_df,
        test_frames=test_frames,
        feature_cols=feature_cols,
        target_col=ECL_TARGET_COL,
        train_cols=train_cols,
        test_cols=test_cols,
        n_train=len(train_df),
        n_test=sum(len(f) for f in test_frames.values()),
    )
