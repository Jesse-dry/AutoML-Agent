# Experience Memory：结构化 JSONL 实验记忆
# ---------------------------------------------------------------
# P1 版本刻意简单：每行一个 json.dumps 的实验记录，append 追加，
# 检索 = 场景相似度（季节 + ACF24/ACF168/load_cv）top-k。
#
#   record(rec)    追加一条经验（线程安全：Lock + flush + fsync）
#   retrieve(sc)   按场景相似度返回 top_k 历史经验（跨 Task 共享）
#
# 相似度：sim = 0.4*season_match + 0.6*(1 - 归一化欧氏距离(acf_24, acf_168, load_cv))
#   season_match：同季 1 / 相邻季 0.5 / 否则 0
# ---------------------------------------------------------------
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SEASONS = ("winter", "spring", "summer", "autumn")
_SEASON_ADJACENT = {  # 相邻季（环状）
    "winter": {"spring"},
    "spring": {"summer"},
    "summer": {"autumn"},
    "autumn": {"winter"},
}
_MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}


def season_from_month(month: int) -> str:
    """月份 → 标准气象季节。"""
    if month < 1 or month > 12:
        raise ValueError(f"month 必须在 1..12，got {month}")
    return _MONTH_TO_SEASON[month]


@dataclass
class Scenario:
    """场景描述（用于记忆检索的相似度向量）。"""
    season: str
    acf_24: float
    acf_168: float
    load_cv: float


@dataclass
class ExperienceRecord:
    """一次候选实验的完整记录。"""
    task_id: int
    round: int
    scenario: Scenario
    problem: dict                       # {"worst_segment": {...}, "bias": ...}
    actions: list                        # 本轮最优候选的动作集
    spec_before: List[dict] = field(default_factory=list)
    spec_after: List[dict] = field(default_factory=list)
    before_rmse: float = 0.0
    after_rmse: float = 0.0
    delta_rmse: float = 0.0             # after - before（负 = 改善）
    outcome: str = ""                   # improved | rolled_back | no_candidate | stopped
    accepted: bool = False
    timestamp: str = ""


@dataclass
class StrategyRecord:
    """每 Task 最优策略（P1-B 外循环：跨 Task 迁移的载体）。

    - scenario：全长历史场景向量（检索键，与 ExperienceRecord 同口径可比）
    - stats：尾部窗口 TaskStats（漂移对比向量，evaluation/drift_detector）
    - profile：决策窗口 ErrorProfile（asdict 后的 dict，残余上下文）
    - transfer_gap：继承策略在本 Task 的 RMSE 相对变化（残余漂移的直接度量）
    """
    task_id: int
    spec: List[dict] = field(default_factory=list)
    rmse: float = 0.0
    scenario: Scenario = None            # type: ignore[assignment]
    stats: Dict[str, Any] = field(default_factory=dict)
    profile: Dict[str, Any] = field(default_factory=dict)
    policy: str = ""                    # cold_start | inherit | modify | reset
    init_max_iter: int = 0
    transfer_gap: Optional[float] = None
    timestamp: str = ""


def scenario_sim(q: Scenario, r: Scenario, cv_max: float) -> float:
    """场景相似度（模块级，供轮级经验与策略级检索共用）。

    sim = 0.4 * season_match + 0.6 * (1 - 归一化欧氏距离(acf_24, acf_168, load_cv))
    season_match：同季 1 / 相邻季 0.5 / 否则 0
    """
    season_sim = 1.0 if r.season == q.season else (
        0.5 if q.season in _SEASON_ADJACENT.get(r.season, set()) else 0.0
    )
    v = np.array([
        float(np.clip(q.acf_24, 0, 1)) - float(np.clip(r.acf_24, 0, 1)),
        float(np.clip(q.acf_168, 0, 1)) - float(np.clip(r.acf_168, 0, 1)),
        float(q.load_cv / cv_max) - float(r.load_cv / cv_max),
    ])
    dist = float(np.sqrt(np.mean(v ** 2)))   # ∈ [0,1]
    return 0.4 * season_sim + 0.6 * (1.0 - dist)


class MemoryManager:
    """JSONL 记忆管理器。runner 单进程内安全；跨进程请用独立 --memory-file。

    两类记忆：
      - 轮级经验  memory/experiment_memory.jsonl（ExperienceRecord）
      - 策略级    memory/strategies.jsonl（StrategyRecord，P1-B 迁移载体）
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else (
            Path(__file__).resolve().parent.parent / "memory" / "experiment_memory.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def strategies_path(self) -> Path:
        return self.path.parent / "strategies.jsonl"

    # ---- 写入 ----
    def record(self, rec: ExperienceRecord) -> None:
        if not rec.timestamp:
            rec.timestamp = datetime.now().isoformat(timespec="seconds")
        line = json.dumps(asdict(rec), ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    import os
                    os.fsync(f.fileno())
                except OSError:
                    pass

    # ---- 读取 ----
    def load_all(self) -> List[ExperienceRecord]:
        if not self.path.exists():
            return []
        recs = []
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        d["scenario"] = Scenario(**d["scenario"])
                        recs.append(ExperienceRecord(**d))
                    except Exception:
                        continue  # 容忍损坏行
        return recs

    def __len__(self) -> int:
        return len(self.load_all())

    # ---- 检索 ----
    def retrieve(self, scenario: Scenario, top_k: int = 5,
                 min_sim: float = 0.0) -> List[ExperienceRecord]:
        recs = self.load_all()
        if not recs:
            return []

        # 数值维度归一化参考界
        cv_max = max(0.5, max((r.scenario.load_cv for r in recs), default=0.5))

        scored = [(r, scenario_sim(scenario, r.scenario, cv_max)) for r in recs]
        scored = [(r, s) for r, s in scored if s > min_sim]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]

    # ---------------------------------------------------------
    # 策略级记忆（P1-B 外循环迁移载体）
    # ---------------------------------------------------------
    def record_strategy(self, rec: StrategyRecord) -> None:
        if not rec.timestamp:
            rec.timestamp = datetime.now().isoformat(timespec="seconds")
        line = json.dumps(asdict(rec), ensure_ascii=False, default=str)
        with self._lock:
            with open(self.strategies_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                try:
                    import os
                    os.fsync(f.fileno())
                except OSError:
                    pass

    def load_strategies(self) -> List[StrategyRecord]:
        if not self.strategies_path.exists():
            return []
        recs = []
        with self._lock:
            with open(self.strategies_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        d["scenario"] = Scenario(**d["scenario"])
                        recs.append(StrategyRecord(**d))
                    except Exception:
                        continue  # 容忍损坏行
        return recs

    def retrieve_strategies(self, scenario: Scenario, top_k: int = 3,
                            min_sim: float = 0.0) -> List[StrategyRecord]:
        recs = self.load_strategies()
        if not recs:
            return []

        # 数值维度归一化参考界（与轮级经验一致）
        cv_max = max(0.5, max((r.scenario.load_cv for r in recs), default=0.5))

        scored = [(r, scenario_sim(scenario, r.scenario, cv_max)) for r in recs]
        scored = [(r, s) for r, s in scored if s > min_sim]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]


def format_strategies_for_llm(recs: List[StrategyRecord]) -> str:
    """把检索到的历史最佳策略格式化为 prompt 文本块（迁移决策用）。"""
    if not recs:
        return "（无历史最佳策略）"
    lines = []
    for r in recs:
        gap = f"{r.transfer_gap:+.3f}" if r.transfer_gap is not None else "-"
        feats = ",".join(s["name"] for s in r.spec)
        lines.append(
            f"  Task {r.task_id} [{r.policy}] rmse={r.rmse:.4f} gap={gap} "
            f"{r.scenario.season} acf24={r.scenario.acf_24:.2f} acf168={r.scenario.acf_168:.2f} "
            f"cv={r.scenario.load_cv:.2f} n_feat={len(r.spec)} feats={feats[:200]}"
        )
    return "\n".join(lines)


def format_memories_for_llm(recs: List[ExperienceRecord]) -> str:
    """把检索到的历史经验格式化为 prompt 文本块。"""
    if not recs:
        return "（无相关历史经验）"
    lines = []
    for r in recs:
        worst = (r.problem or {}).get("worst_segment", {})
        lines.append(
            f"  Task {r.task_id} R{r.round} [{r.outcome}] "
            f"{r.scenario.season} acf24={r.scenario.acf_24:.2f} acf168={r.scenario.acf_168:.2f} "
            f"cv={r.scenario.load_cv:.2f} worst={worst.get('key', '-')} "
            f"ΔRMSE={r.delta_rmse:+.4f} ({r.before_rmse:.3f}→{r.after_rmse:.3f}) "
            f"actions={json.dumps(r.actions, ensure_ascii=False)[:160]}"
        )
    return "\n".join(lines)
