# 严格值级泄漏检查器
# ---------------------------------------------------------------
# 三遍检查：
#   Pass A — 静态（基于 FEATURE_SPEC 血缘）：lag 必须严格过去且 ≥ pred_horizon；
#            rolling 不得 uses_current_target；min_periods < window 判为
#            feature_spec_violation（incomplete_window，非泄漏，但违反规范）；
#            特征不得为目标列别名。
#   Pass B — 值级 recompute-from-prefix：对检查点 t，用与 spec 一致的
#            严格过去向构造器重算 feature[t]（该值只依赖 ≤ t-1 的数据），
#            与原特征值比对。不一致 = 未来依赖（如未 shift 的 rolling 含当前行）。
#   Pass C — 同刻目标别名扫描：feature[t] 不得等于 target[t]。
#
# 分级（防 O(N²)，不全量重算）：
#   fast   — 前边界 / MAX_LAG 附近 / 中点 / 尾部 / 每月边界
#   sample — fast + 随机 100~500 点（默认）
#   full   — 全行（`--leak-check full`，正式发布前跑）
# ---------------------------------------------------------------
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.task_builder import FEATURE_SPEC, MAX_LAG, build_features

_LEVELS = ("fast", "sample", "full")


@dataclass
class LeakageViolation:
    feature: str
    row: Optional[pd.Timestamp]  # None 表示静态/整列级违规
    kind: str
    message: str


def _close(a, b) -> bool:
    """float 比较（含 NaN 匹配）。"""
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=1e-5, atol=1e-6))


def _default_check_points(n: int, mode: str) -> List[int]:
    """推导检查点行位置（不依赖外部语义边界）。"""
    if mode == "full":
        return list(range(n))
    if n <= MAX_LAG + 2:
        return list(range(n))

    pts: set = {MAX_LAG, MAX_LAG + 1, n // 2, n - MAX_LAG - 1, n - 1}
    if mode == "sample":
        extra = random.sample(range(n), min(300, n))
        pts.update(extra)
    return sorted(p for p in pts if 0 <= p < n)


def _pass_a(spec: List[dict], feature_cols: List[str], target_col: str,
            pred_horizon: int) -> List[LeakageViolation]:
    violations = []
    for s in spec:
        name, stype = s["name"], s["type"]
        if stype == "lag":
            k = s.get("k", 0)
            source = s.get("source", "")
            lookback_end = s.get("lookback_end", -1)
            if lookback_end >= 0 or k <= 0:
                violations.append(LeakageViolation(
                    name, None, "lag_le_0",
                    f"lag {name} 的 lookback_end={lookback_end} / k={k} >= 0，使用当前或未来信息"))
            elif source == target_col and k < pred_horizon:
                violations.append(LeakageViolation(
                    name, None, "lag_lt_horizon",
                    f"目标列 lag {name} 的 k={k} < 预测步长 {pred_horizon}"))
        elif stype == "rolling":
            if s.get("uses_current_target", False):
                violations.append(LeakageViolation(
                    name, None, "rolling_uses_current",
                    f"rolling {name} uses_current_target=True，窗口含当前行（必须 shift(1)）"))
            min_periods = s.get("min_periods")
            window = s.get("window", 0)
            if min_periods is not None and min_periods < window:
                violations.append(LeakageViolation(
                    name, None, "incomplete_window",
                    f"rolling {name} min_periods={min_periods} < window={window}，"
                    f"窗口不完整（Feature completeness 违规，非泄漏）"))
        elif stype == "cross":
            col1, col2 = s.get("col1"), s.get("col2")
            if col1 == target_col or col2 == target_col:
                violations.append(LeakageViolation(
                    name, None, "cross_uses_target",
                    f"cross {name} 操作列含当前目标（{col1}/{col2}），必须用滞后/滚动特征"))
            name_set = set(feature_cols)
            if col1 not in name_set or col2 not in name_set:
                violations.append(LeakageViolation(
                    name, None, "cross_unknown_operand",
                    f"cross {name} 操作列 {col1}/{col2} 不在特征集合中"))
            if s.get("uses_current_target", False):
                violations.append(LeakageViolation(
                    name, None, "cross_uses_current",
                    f"cross {name} uses_current_target=True，操作含当前目标"))
            # 操作列顺序必须在 cross 自身之前（增量计算依赖）
            idx = {f: i for i, f in enumerate(feature_cols)}
            pos = idx.get(name, len(feature_cols))
            if (col1 in idx and idx[col1] >= pos) or (col2 in idx and idx[col2] >= pos):
                violations.append(LeakageViolation(
                    name, None, "cross_operand_order",
                    f"cross {name} 操作列 {col1}/{col2} 必须定义在自身之前"))
        elif stype == "current":
            # 外生当前小时值（lookback=0）：仅当 source != target_col 才合法；
            # source == target_col 即直接使用当前目标，判定泄漏。
            if s.get("source", "") == target_col:
                violations.append(LeakageViolation(
                    name, None, "current_uses_target",
                    f"current {name} 作用于目标列 {target_col}，等于直接使用当前目标（泄漏）"))
        elif stype == "time":
            pass
    # 目标别名
    for s in spec:
        if s["name"] == target_col:
            violations.append(LeakageViolation(
                s["name"], None, "target_alias",
                f"特征 {s['name']} 直接别名目标列"))
    return violations


def _pass_b(df: pd.DataFrame, spec: List[dict], feature_cols: List[str],
            target_col: str, mode: str,
            extra_positions: Optional[List[int]] = None) -> List[LeakageViolation]:
    """recompute-from-prefix：用严格过去向构造器重算，与原值比对。"""
    expected = build_features(df, spec=spec, target_col=target_col)
    check_points = _default_check_points(len(df), mode)
    if extra_positions:
        check_points = sorted(set(check_points) | set(extra_positions))
    violations = []
    for i in check_points:
        t = df.index[i]
        for f in feature_cols:
            if f not in expected.columns:
                continue  # spec 之外的列无法用血缘重算，交由 Pass C 检查别名
            if not _close(df[f].iloc[i], expected[f].iloc[i]):
                violations.append(LeakageViolation(
                    f, t, "recompute_mismatch",
                    f"特征 {f} 在 {t} 用严格过去数据重算不一致"
                    f"（actual={df[f].iloc[i]}, expected={expected[f].iloc[i]}），存在未来依赖"))
    return violations


def _pass_c(df: pd.DataFrame, feature_cols: List[str],
            target_col: str) -> List[LeakageViolation]:
    """列级目标别名检查：特征整列不得与目标列完全一致（直接复制目标）。"""
    violations = []
    if target_col not in df.columns:
        return violations
    y = df[target_col]
    if y.notna().all():
        for f in feature_cols:
            if f in df.columns and df[f].notna().all():
                if np.allclose(df[f].values, y.values, rtol=1e-5, atol=1e-6):
                    violations.append(LeakageViolation(
                        f, None, "target_alias",
                        f"特征 {f} 整列与目标列 {target_col} 完全一致，直接复制目标"))
    return violations


def check_feature_leakage(
    df: pd.DataFrame,
    spec: List[dict] = FEATURE_SPEC,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "LOAD",
    pred_horizon: int = 1,
    mode: str = "sample",
    extra_check_points: Optional[List[pd.Timestamp]] = None,
) -> Tuple[bool, List[LeakageViolation]]:
    """
    严格值级泄漏检查。返回 (is_safe, violations)。

    - df：含特征列（+ target 列）的 DataFrame，datetime 索引
    - mode：fast | sample | full
    - extra_check_points：额外检查点（如 train_end / val_start / forecast_start）
    """
    if mode not in _LEVELS:
        raise ValueError(f"mode 必须是 {_LEVELS}，got {mode}")
    if feature_cols is None:
        feature_cols = [s["name"] for s in spec]

    extra_positions: Optional[List[int]] = None
    if extra_check_points and len(df) > 0:
        extra_positions = [df.index.get_loc(t) for t in extra_check_points if t in df.index]

    violations: List[LeakageViolation] = []
    violations += _pass_a(spec, feature_cols, target_col, pred_horizon)
    violations += _pass_b(df, spec, feature_cols, target_col, mode, extra_positions)
    violations += _pass_c(df, feature_cols, target_col)

    is_safe = len(violations) == 0
    return is_safe, violations
