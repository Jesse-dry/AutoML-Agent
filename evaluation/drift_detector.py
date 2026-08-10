# Drift Detector：跨 Task 数据漂移确定性检测（P1-B 外循环）
# ---------------------------------------------------------------
# 职责：
#   1. compute_task_stats —— 从 Task 历史**尾部对齐窗口**提取确定性统计量
#      （Task 历史是严格前缀关系，全长对比会被共享前缀稀释 → 只比尾窗）
#   2. compute_scenario    —— 全长历史场景向量（与 memory_manager.Scenario /
#      evolution_runner._build_scenario 同公式，供策略检索）
#   3. detect_drift        —— 对比相邻 Task 统计量 + 上一模型残余误差，
#      输出 drift_score ∈ [0,1] / level(low|medium|high) / signals
#
# LLM 不参与漂移计算：本模块只做测量，解释与决策交给
# agent/strategy_migration.py（LLM）与 experiments/run_outer_loop.py。
# ---------------------------------------------------------------
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from memory.memory_manager import Scenario, season_from_month

# 窗口与阈值常量（可调）
DEFAULT_WINDOW_HOURS = 672          # 4 周尾部窗口
DRIFT_LOW = 0.30                    # < 0.30 → low（沿用上一 Task 策略）
DRIFT_HIGH = 0.55                   # ≥ 0.55 → high（重新搜索；需多轴信号或残余恶化叠加）
_MEAN_SIG_CAP = 2.0                 # mean_shift 以 σ 计，2σ 即满（1σ 属常见季节性月变）
_STD_CHG_CAP = 1.0                  # std 相对变化 100% 即满
_QUANTILE_CAP = 1.0                 # 分位相对变化 100% 即满
_RESID_TREND_CAP = 0.5              # rmse 相对变化 50% 即满


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))


def _norm(value: float, cap: float) -> float:
    """夹到 [0, cap] 再归一化为 [0,1]（信号统一量纲）。"""
    if cap <= 0:
        return 0.0
    return _clip(value, 0.0, cap) / cap


def _acf(load: pd.Series, lag: int) -> float:
    """自相关（自包含，pd.Series.autocorr；数据不足/零方差 → 0.0）。"""
    if len(load) <= lag + 1:
        return 0.0
    v = load.autocorr(lag=lag)
    if v is None or not np.isfinite(v):
        return 0.0
    return float(_clip(v, -1.0, 1.0))


def _tail_season(tail_idx: pd.DatetimeIndex) -> str:
    """尾部窗口的众数月 → 气象季节。"""
    months = tail_idx.month.to_numpy()
    if len(months) == 0:
        return "unknown"
    vals, counts = np.unique(months, return_counts=True)
    return season_from_month(int(vals[np.argmax(counts)]))


@dataclass
class TaskStats:
    """某 Task 尾部窗口的确定性统计量（漂移对比向量）。"""
    task_id: int
    tail_start: str
    n: int
    mean: float
    std: float
    q10: float
    q50: float
    q90: float
    acf_24: float
    acf_168: float
    cv: float
    season: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_task_stats(
    history_df: pd.DataFrame,
    task_id: int = 0,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    target_col: str = "LOAD",
) -> TaskStats:
    """
    从 history_df（datetime 索引）尾部 window_hours 提取统计量。

    - 窗口默认 672h（4 周）：相邻 Task 各取尾 4 周，窗口几乎不重叠，
      干净度量"最近一个月负荷形态"的月际变化，不被共享前缀稀释。
    """
    if window_hours < 24:
        raise ValueError(f"window_hours 过小（{window_hours}），需 ≥ 24")
    tail = history_df.tail(window_hours)
    load = tail[target_col].dropna()
    if len(load) == 0:
        raise ValueError(f"Task {task_id} 尾部窗口无有效 LOAD")

    mean = float(load.mean())
    std = float(load.std(ddof=0))
    cv = (std / mean) if mean != 0 else 0.0

    return TaskStats(
        task_id=task_id,
        tail_start=str(tail.index[0]),
        n=int(len(load)),
        mean=mean,
        std=std,
        q10=float(load.quantile(0.10)),
        q50=float(load.quantile(0.50)),
        q90=float(load.quantile(0.90)),
        acf_24=_acf(load, 24),
        acf_168=_acf(load, 168),
        cv=cv,
        season=_tail_season(tail.index),
    )


def compute_scenario(
    history_df: pd.DataFrame,
    forecast_ts: pd.DatetimeIndex,
    task_id: int = 0,
    target_col: str = "LOAD",
) -> Scenario:
    """
    全长历史场景向量（与 evolution_runner._build_scenario 同公式）：
    season = forecast 月季节；acf_24/acf_168 全长历史自相关；cv = std/mean。
    供策略检索（StrategyRecord.scenario）与 ExperienceRecords 同口径可比。
    """
    load = history_df[target_col].dropna()
    cv = (float(load.std()) / float(load.mean())) if load.mean() != 0 else 0.0
    return Scenario(
        season=season_from_month(forecast_ts[0].month),
        acf_24=_acf(load, 24),
        acf_168=_acf(load, 168),
        load_cv=cv,
    )


@dataclass
class DriftReport:
    """相邻 Task 漂移检测报告。"""
    task_id: int                    # 当前 Task
    compared_task_id: int           # 对比的上一个 Task
    drift_score: float              # ∈ [0,1]
    level: str                      # low | medium | high
    signals: Dict[str, float]       # mean_shift / std_shift / quantile_shift / acf24_change / acf168_change
    residual: Dict[str, float]      # 上一 Task 模型残余误差（carryover context）
    scores: Dict[str, float]        # data / temporal / residual 分量
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _residual_context(resid_prev: Optional[Any]) -> Dict[str, float]:
    """ErrorProfile（对象或 asdict 后的 dict）→ {rmse, bias, peak_error} 上下文。"""
    if resid_prev is None:
        return {}
    if isinstance(resid_prev, dict):
        p = resid_prev
        rmse = float(p.get("rmse", 0.0))
        bias = float(p.get("bias", 0.0))
        top = p.get("top_worst", []) or []
        peak = float(max((r.get("abs_error", 0.0) for r in top), default=rmse))
        return {"rmse": rmse, "bias": bias, "peak_error": peak}
    p = resid_prev
    rmse = float(getattr(p, "rmse", 0.0))
    bias = float(getattr(p, "bias", 0.0))
    top = getattr(p, "top_worst", []) or []
    peak = float(max((r["abs_error"] for r in top), default=rmse))
    return {"rmse": rmse, "bias": bias, "peak_error": peak}


def detect_drift(
    stats_prev: TaskStats,
    stats_cur: TaskStats,
    resid_prev: Optional[Any] = None,
    resid_trend: Optional[float] = None,
) -> DriftReport:
    """
    对比相邻 Task 的尾部统计量，输出漂移报告。

    信号（全部夹到 [0,1]）：
      - mean_shift   = |Δμ| / σ_pooled（1σ 满）
      - std_shift    = |σ_cur/σ_prev − 1|（100% 满）
      - quantile_shift = q10/q50/q90 平均相对变化
      - acf24_change / acf168_change = |ΔACF|（ACF 本身 ∈ [-1,1]）
      - residual     = |rmse 相对趋势|（50% 满），仅 resid_trend 提供时计入
    聚合：无残余时 数据 0.55 + 时序 0.45；有残余时 数据 0.30 + 时序 0.25 +
    残余 0.45（残余 = 策略转移健康度的直接度量，权重最高）。
    """
    sig: Dict[str, float] = {}

    # LOAD 数据信号（全部 _norm → [0,1]）
    mu_p, mu_c = stats_prev.mean, stats_cur.mean
    s_p, s_c = stats_prev.std, stats_cur.std
    pooled = np.sqrt(0.5 * (s_p ** 2 + s_c ** 2))
    sig["mean_shift"] = _norm(abs(mu_c - mu_p) / max(pooled, 1e-9), _MEAN_SIG_CAP)
    sig["std_shift"] = _norm(abs(s_c / s_p - 1.0) if s_p > 0 else 0.0, _STD_CHG_CAP)
    q_shifts = []
    for qkey in ("q10", "q50", "q90"):
        qp, qc = getattr(stats_prev, qkey), getattr(stats_cur, qkey)
        q_shifts.append(_norm(abs(qc - qp) / max(abs(qp), 1e-9), _QUANTILE_CAP))
    sig["quantile_shift"] = float(np.mean(q_shifts)) if q_shifts else 0.0

    # Temporal 信号
    sig["acf24_change"] = _clip(abs(stats_cur.acf_24 - stats_prev.acf_24), 0, 1.0)
    sig["acf168_change"] = _clip(abs(stats_cur.acf_168 - stats_prev.acf_168), 0, 1.0)

    # 残余信号（可选）
    resid_scores: Dict[str, float] = {}
    resid_ctx = _residual_context(resid_prev)
    if resid_trend is not None:
        resid_scores["residual"] = _norm(abs(resid_trend), _RESID_TREND_CAP)

    # 族内用 RMS 聚合：单轴极端信号（如纯方差倍增）也能达到可感知强度，
    # 而非被加权平均稀释到 low。
    data_score = float(np.sqrt(
        (sig["mean_shift"] ** 2 + sig["std_shift"] ** 2 + sig["quantile_shift"] ** 2) / 3.0
    ))
    temporal_score = float(np.sqrt(
        (sig["acf24_change"] ** 2 + sig["acf168_change"] ** 2) / 2.0
    ))

    scores = {"data": data_score, "temporal": temporal_score}
    if resid_scores:
        scores["residual"] = resid_scores["residual"]

    # 动态权重：有残余时残余权重最高（转移健康度直接度量）
    if "residual" in scores:
        w_data, w_temp, w_res = 0.30, 0.25, 0.45
    else:
        w_data, w_temp, w_res = 0.55, 0.45, 0.0

    drift_score = _clip(
        w_data * scores["data"] + w_temp * scores["temporal"] + w_res * scores.get("residual", 0.0),
        0.0, 1.0,
    )
    level = ("low" if drift_score < DRIFT_LOW
             else "high" if drift_score >= DRIFT_HIGH
             else "medium")

    return DriftReport(
        task_id=stats_cur.task_id,
        compared_task_id=stats_prev.task_id,
        drift_score=drift_score,
        level=level,
        signals=sig,
        residual=resid_ctx,
        scores=scores,
        meta={
            "prev_tail_start": stats_prev.tail_start,
            "cur_tail_start": stats_cur.tail_start,
            "prev_season": stats_prev.season,
            "cur_season": stats_cur.season,
            "window_hours": DEFAULT_WINDOW_HOURS,
        },
    )


def format_drift_for_llm(report: DriftReport) -> str:
    """漂移报告 → LLM prompt 文本块。"""
    lines = [
        f"## 跨 Task 漂移报告（Task {report.compared_task_id} → Task {report.task_id}）",
        f"drift_score={report.drift_score:.2f}  level={report.level}",
        "### 信号",
        f"  - mean_shift   = {report.signals.get('mean_shift', 0):.3f}（负荷均值漂移）",
        f"  - std_shift    = {report.signals.get('std_shift', 0):.3f}（负荷波动漂移）",
        f"  - quantile_shift = {report.signals.get('quantile_shift', 0):.3f}（分位漂移）",
        f"  - acf24_change = {report.signals.get('acf24_change', 0):.3f}（日周期强度变化）",
        f"  - acf168_change= {report.signals.get('acf168_change', 0):.3f}（周周期强度变化）",
    ]
    if report.residual:
        r = report.residual
        lines.append("### 上一 Task 模型残余误差（carryover 上下文）")
        lines.append(
            f"  - rmse={r.get('rmse', 0):.4f} bias={r.get('bias', 0):+.4f} "
            f"peak_error={r.get('peak_error', 0):.4f}"
        )
    lines.append(f"### 场景变化：{report.meta.get('prev_season')} → {report.meta.get('cur_season')}")
    return "\n".join(lines)
