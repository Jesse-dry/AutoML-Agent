# 误差画像（Error Profiling）：告诉 LLM 模型到底在哪里错
# ---------------------------------------------------------------
# 输入：决策窗口的 y_true / y_pred / 时间戳。
# 输出：整体 RMSE/BIAS + 分段误差（时段 / 工作日 / 负荷状态 / 变化状态）
#       + top-20 worst 时刻 + 排序后的 worst_segments。
#
# BIAS 约定：bias = mean(y_pred - y_true)。正 = 系统性高估，负 = 系统性低估。
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class ErrorProfile:
    rmse: float
    bias: float                     # mean(y_pred - y_true)
    bias_ratio: float               # bias / rmse
    n: int
    segments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    worst_segments: List[dict] = field(default_factory=list)   # [{segment, rmse, n, bias}]
    top_worst: List[dict] = field(default_factory=list)        # ≤ n_top_worst 条

    def worst_segment(self):
        """单个最差分段（供 memory 的 problem.worst_segment 使用）。"""
        if not self.worst_segments:
            return None
        w = self.worst_segments[0]
        return {"key": w["segment"], "rmse": w["rmse"], "bias": w["bias"]}


def _seg_metrics(mask: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if int(mask.sum()) == 0:
        return None
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yt - yp
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "n": int(mask.sum()),
        "bias": float(np.mean(yp - yt)),
    }


def _quantile_split(x: np.ndarray):
    """q33/q66 分位数，退化时回退到 std 阈值。"""
    lo = float(np.quantile(x, 0.33))
    hi = float(np.quantile(x, 0.66))
    if lo == hi:
        s = float(np.std(x))
        lo, hi = -0.5 * s, 0.5 * s
    return lo, hi


def compute_error_profile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ts: pd.DatetimeIndex,
    n_top_worst: int = 20,
) -> ErrorProfile:
    """
    计算误差画像。y_true / y_pred 为一维数组，ts 为同长度 DatetimeIndex。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    ts_valid = ts[valid]
    n = len(y_true)
    if n == 0:
        raise ValueError("compute_error_profile: 无有效 (y_true, y_pred) 点")

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(y_pred - y_true))
    bias_ratio = bias / rmse if rmse > 0 else 0.0

    hours = ts_valid.hour.to_numpy()
    dows = ts_valid.weekday.to_numpy()
    is_weekend = (dows >= 5).astype(int)

    # 负荷状态（y_true 分位数）
    load_lo, load_hi = _quantile_split(y_true)
    load_state = np.where(y_true < load_lo, "load_low",
                   np.where(y_true > load_hi, "load_peak", "load_normal"))

    # 变化状态（一阶差分分位数；首点标 stable）
    dy = np.zeros(n)
    dy[1:] = np.diff(y_true)
    ch_lo, ch_hi = _quantile_split(dy[1:])
    change = np.where(dy < ch_lo, "change_ramp_down",
              np.where(dy > ch_hi, "change_ramp_up", "change_stable"))

    segments: Dict[str, Dict[str, float]] = {}
    # 时段
    for h in range(24):
        m = hours == h
        seg = _seg_metrics(m, y_true, y_pred)
        if seg:
            segments[f"hour_{h:02d}"] = seg
    for w in (0, 1):
        m = is_weekend == w
        seg = _seg_metrics(m, y_true, y_pred)
        if seg:
            segments["weekend" if w else "weekday"] = seg
    for d in range(7):
        m = dows == d
        seg = _seg_metrics(m, y_true, y_pred)
        if seg:
            segments[f"dow_{d}"] = seg
    # 负荷 / 变化状态
    for sname in ("load_low", "load_normal", "load_peak"):
        seg = _seg_metrics(load_state == sname, y_true, y_pred)
        if seg:
            segments[sname] = seg
    for sname in ("change_stable", "change_ramp_up", "change_ramp_down"):
        seg = _seg_metrics(change == sname, y_true, y_pred)
        if seg:
            segments[sname] = seg
    # 复合：工作日晚峰（用户示例中的 weekday_peak 形态）
    seg = _seg_metrics((dows < 5) & (load_state == "load_peak"), y_true, y_pred)
    if seg:
        segments["weekday_peak"] = seg

    # worst_segments：按 rmse 降序，过滤过小分段（< 24 点），取 top-5
    ranked = sorted(
        ({"segment": k, **v} for k, v in segments.items() if v["n"] >= 24),
        key=lambda r: r["rmse"], reverse=True,
    )[:5]

    # top_worst：按 |error| 降序
    abs_err = np.abs(err)
    idx = np.argsort(abs_err)[::-1][:n_top_worst]
    top_worst = [
        {
            "timestamp": str(ts_valid[i]),
            "y_true": float(y_true[i]),
            "y_pred": float(y_pred[i]),
            "error": float(err[i]),
            "abs_error": float(abs_err[i]),
            "seg_tags": [
                f"hour_{hours[i]:02d}",
                "weekend" if is_weekend[i] else "weekday",
                f"dow_{dows[i]}",
                load_state[i],
                change[i],
            ],
        }
        for i in idx
    ]

    return ErrorProfile(
        rmse=rmse, bias=bias, bias_ratio=bias_ratio, n=n,
        segments=segments, worst_segments=ranked, top_worst=top_worst,
    )


def _bias_word(b: float) -> str:
    if b > 0:
        return "高估"
    if b < 0:
        return "低估"
    return "无偏"


def format_profile_for_llm(p: ErrorProfile, n_top_worst: int = 10) -> str:
    """把误差画像格式化为 LLM prompt 文本块。"""
    lines = []
    lines.append(
        f"## 误差画像（决策窗口 RMSE={p.rmse:.4f}, n={p.n}, "
        f"bias={p.bias:+.4f} ({_bias_word(p.bias)}, bias_ratio={p.bias_ratio:+.2f})"
    )

    lines.append("### 最差分段 (top-5 by RMSE)")
    for r in p.worst_segments:
        lines.append(
            f"  - {r['segment']:<18s} rmse={r['rmse']:.4f} n={r['n']:>4d} "
            f"bias={r['bias']:+.4f} ({_bias_word(r['bias'])})"
        )

    # 时段明细：hour 里 top/bottom 各 3
    hours = {k: v for k, v in p.segments.items() if k.startswith("hour_")}
    if hours:
        top_h = sorted(hours.items(), key=lambda kv: kv[1]["rmse"], reverse=True)[:3]
        bottom_h = sorted(hours.items(), key=lambda kv: kv[1]["rmse"])[:3]
        lines.append("### 时段 error hot/cold spots")
        for k, v in top_h:
            lines.append(f"  - {k} rmse={v['rmse']:.4f} n={v['n']} bias={v['bias']:+.4f}")
        for k, v in bottom_h:
            lines.append(f"  - {k} rmse={v['rmse']:.4f} n={v['n']} (best)")

    # 其它关键分段
    for key in ("weekday", "weekend", "load_low", "load_normal", "load_peak",
                "change_stable", "change_ramp_up", "change_ramp_down", "weekday_peak"):
        v = p.segments.get(key)
        if v:
            lines.append(f"  - {key:<18s} rmse={v['rmse']:.4f} n={v['n']:>4d} bias={v['bias']:+.4f}")

    lines.append(f"### top-{n_top_worst} 最差时刻")
    for r in p.top_worst[:n_top_worst]:
        lines.append(
            f"  - {r['timestamp']} y={r['y_true']:.1f} pred={r['y_pred']:.1f} "
            f"err={r['error']:+.2f}"
        )
    return "\n".join(lines)
