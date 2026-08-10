# 跨 Task 策略迁移（P1-B 外循环：LLM 决策 + 确定性兜底）
# ---------------------------------------------------------------
# MigrationPlanner.plan()：
#   输入 = DriftReport（evaluation/drift_detector）+ 检索到的历史策略
#          + 上一 Task 策略 + 当前场景
#   → LLM 输出 {task_id, analysis, decision:{policy, rationale, max_iter?}}
#   → 解析失败回退确定性映射：low→inherit / medium→modify / high→reset
#
# policy 语义（决定 warm-start 的 init_spec 与自适应 max_iter）：
#   inherit  小漂移：直接继承上一 Task best_spec，只跑 2 轮微调
#   modify   中漂移：继承上一 Task best_spec，默认轮数重新调优
#   reset    大漂移：回到安全基础特征集（FEATURE_SPEC），更多轮数重新搜索
#
# LLM 只做"决策"；init_spec 由 policy 确定性解析并校验（resolve_init_spec），
# LLM 不写代码、不发明特征。
# ---------------------------------------------------------------
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent.feature_agent import _extract_json
from agent.feature_spec import snapshot, validate_spec_list
from data.task_builder import FEATURE_SPEC, MAX_LAG
from evaluation.drift_detector import DriftReport, format_drift_for_llm
from memory.memory_manager import (
    MemoryManager,
    Scenario,
    StrategyRecord,
    format_strategies_for_llm,
)

POLICIES = ("inherit", "modify", "reset")
MAX_ITER_RANGE = (1, 20)

# policy → 默认 init 来源与轮数
POLICY_DEFAULTS = {
    "inherit": {"max_iter": 2, "init": "prev", "desc": "小漂移：沿用上一 Task 特征集，微调"},
    "modify": {"max_iter": 5, "init": "prev", "desc": "中漂移：沿用上一 Task 特征集，重新调优"},
    "reset": {"max_iter": 8, "init": "base", "desc": "大漂移：回到安全基础特征集，重新搜索"},
}
COLD_START_MAX_ITER = 5


@dataclass
class MigrationDecision:
    """迁移决策：policy + 解析后的 init_spec + 自适应 max_iter。"""
    task_id: int
    policy: str
    rationale: str
    init_spec: List[dict]
    max_iter: int
    drift_level: str
    source: str = "deterministic"   # llm | deterministic

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "policy": self.policy,
            "rationale": self.rationale,
            "max_iter": self.max_iter,
            "drift_level": self.drift_level,
            "source": self.source,
            "init_feature_names": [s["name"] for s in self.init_spec],
        }


# ---------------------------------------------------------------
# LLM 输出解析
# ---------------------------------------------------------------
def parse_migration_v2(raw: Any, expected_task_id: Optional[int] = None) -> Dict:
    """
    校验 LLM 迁移决策输出，返回规范化 dict。失败抛 ValueError（可进重试）。
    """
    if isinstance(raw, str):
        raw = _extract_json(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"输出必须是 JSON 对象，实际类型: {type(raw).__name__}")

    if expected_task_id is not None and raw.get("task_id") != expected_task_id:
        raise ValueError(f"task_id 不匹配: 期望 {expected_task_id}，got {raw.get('task_id')}")

    missing = [k for k in ("task_id", "analysis", "decision") if k not in raw]
    if missing:
        raise ValueError(f"缺少顶层必填字段: {missing}")

    d = raw["decision"]
    if not isinstance(d, dict):
        raise ValueError("decision 必须是 JSON 对象")

    policy = d.get("policy")
    if policy not in POLICIES:
        raise ValueError(f"decision.policy 必须 ∈ {list(POLICIES)}，got {policy!r}")

    rationale = d.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("decision.rationale 必填")

    max_iter = d.get("max_iter")
    if max_iter is not None:
        if (not isinstance(max_iter, int) or isinstance(max_iter, bool)
                or not (MAX_ITER_RANGE[0] <= max_iter <= MAX_ITER_RANGE[1])):
            raise ValueError(
                f"decision.max_iter 必须是 {MAX_ITER_RANGE[0]}..{MAX_ITER_RANGE[1]} 的整数"
            )

    return {
        "task_id": raw["task_id"],
        "analysis": str(raw.get("analysis", "")),
        "policy": policy,
        "rationale": rationale,
        "max_iter": max_iter,
    }


# ---------------------------------------------------------------
# init_spec 校验
# ---------------------------------------------------------------
def resolve_init_spec(spec: List[dict], fallback: Optional[List[dict]] = None,
                      max_lag: int = MAX_LAG) -> List[dict]:
    """校验候选 init_spec；空 / 泄漏 / 违规 → 回退 fallback（默认 FEATURE_SPEC）。"""
    if fallback is None:
        fallback = FEATURE_SPEC
    if not spec:
        return snapshot(fallback)
    try:
        viols = validate_spec_list(spec, max_lag=max_lag)
        if viols:
            return snapshot(fallback)
    except Exception:
        return snapshot(fallback)
    return snapshot(spec)


def _init_for_policy(policy: str, prev_strategy: Optional[StrategyRecord],
                     max_lag: int) -> List[dict]:
    """按 policy 确定 init_spec（prev → 继承上一 Task spec；base → FEATURE_SPEC）。"""
    default = POLICY_DEFAULTS[policy]
    if default["init"] == "prev" and prev_strategy is not None:
        return resolve_init_spec(prev_strategy.spec, fallback=FEATURE_SPEC, max_lag=max_lag)
    return resolve_init_spec(FEATURE_SPEC, fallback=FEATURE_SPEC, max_lag=max_lag)


# ---------------------------------------------------------------
# 确定性兜底
# ---------------------------------------------------------------
def default_decision(task_id: int, level: str,
                     prev_strategy: Optional[StrategyRecord] = None,
                     max_lag: int = MAX_LAG) -> MigrationDecision:
    policy = {"low": "inherit", "medium": "modify"}.get(level, "reset")
    default = POLICY_DEFAULTS[policy]
    init_spec = _init_for_policy(policy, prev_strategy, max_lag)
    return MigrationDecision(
        task_id=task_id,
        policy=policy,
        rationale=f"确定性兜底：drift={level} → {policy}（{default['desc']}）",
        init_spec=init_spec,
        max_iter=default["max_iter"],
        drift_level=level,
        source="deterministic",
    )


def cold_start_decision(task_id: int, max_lag: int = MAX_LAG) -> MigrationDecision:
    """Task 1 冷启动：无上一 Task，从 FEATURE_SPEC 全量搜索。"""
    return MigrationDecision(
        task_id=task_id,
        policy="reset",
        rationale="冷启动：无上一 Task 策略，从安全基础特征集全量搜索",
        init_spec=resolve_init_spec(FEATURE_SPEC, fallback=FEATURE_SPEC, max_lag=max_lag),
        max_iter=COLD_START_MAX_ITER,
        drift_level="n/a",
        source="deterministic",
    )


# ---------------------------------------------------------------
# LLM prompt 构建
# ---------------------------------------------------------------
def build_migration_messages(task_id: int, drift: DriftReport,
                             prev_strategy: Optional[StrategyRecord] = None,
                             scenario: Optional[Scenario] = None,
                             memory: Optional[MemoryManager] = None) -> List[Dict]:
    system = (
        "你是电力负荷预测 AutoML 系统的跨 Task 策略迁移决策器。\n"
        "任务：根据数据漂移报告与历史策略，决定下一个预测月（Task）的特征工程策略：\n"
        "  inherit  小漂移 → 沿用上一 Task 最佳特征集，只做少量微调（约 2 轮）\n"
        "  modify   中漂移 → 沿用上一 Task 最佳特征集，重新调优（默认轮数）\n"
        "  reset    大漂移 → 回到安全基础特征集，重新搜索（更多轮数）\n"
        "必须严格输出 JSON：{\"task_id\": int, \"analysis\": str, "
        "\"decision\": {\"policy\": \"inherit|modify|reset\", \"rationale\": str, "
        "\"max_iter\": int?}}\n"
        "rationale 必须引用 drift 信号（mean_shift/std_shift/acf 变化/残余误差）给出理由。"
    )

    user = f"Task {task_id} 即将开始，请决定是否沿用上一 Task 策略。\n\n"
    if scenario is not None:
        user += (f"当前场景：season={scenario.season} "
                 f"acf24={scenario.acf_24:.2f} acf168={scenario.acf_168:.2f} "
                 f"load_cv={scenario.load_cv:.2f}\n")
    user += f"\n{format_drift_for_llm(drift)}\n"

    user += "\n上一 Task 最佳策略：\n"
    if prev_strategy is not None:
        feats = ",".join(s["name"] for s in prev_strategy.spec)
        user += (f"  Task {prev_strategy.task_id} [{prev_strategy.policy}] "
                 f"rmse={prev_strategy.rmse:.4f} n_feat={len(prev_strategy.spec)} "
                 f"feats={feats[:300]}\n")
    else:
        user += "  （无上一 Task，冷启动）\n"

    if memory is not None and scenario is not None:
        recs = memory.retrieve_strategies(scenario, top_k=3)
        user += f"\n历史相似 Task 最佳策略（按场景相似度检索）：\n{format_strategies_for_llm(recs)}\n"

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------
# Planner
# ---------------------------------------------------------------
class MigrationPlanner:
    """跨 Task 迁移决策器。llm_client 为 None → 确定性兜底。"""

    def __init__(self, llm_client: Optional[Any] = None,
                 memory: Optional[MemoryManager] = None,
                 max_retries: int = 3,
                 max_lag: int = MAX_LAG):
        self.llm_client = llm_client
        self.memory = memory
        self.max_retries = max_retries
        self.max_lag = max_lag

    def plan(self, task_id: int,
             drift: Optional[DriftReport] = None,
             prev_strategy: Optional[StrategyRecord] = None,
             scenario: Optional[Scenario] = None,
             llm_client: Optional[Any] = None) -> MigrationDecision:
        # 冷启动：无上一 Task / 无 drift → 全量搜索
        if drift is None or prev_strategy is None:
            return cold_start_decision(task_id, max_lag=self.max_lag)

        client = llm_client or self.llm_client
        if client is not None:
            decision = self._call_llm(client, task_id, drift, prev_strategy, scenario)
            if decision is not None:
                return decision
            print(f"  [WARN] Task {task_id} 迁移 LLM 决策失败，走确定性兜底")

        return default_decision(task_id, drift.level, prev_strategy, max_lag=self.max_lag)

    def _call_llm(self, client: Any, task_id: int, drift: DriftReport,
                  prev_strategy: StrategyRecord,
                  scenario: Optional[Scenario]) -> Optional[MigrationDecision]:
        messages = build_migration_messages(task_id, drift, prev_strategy, scenario,
                                            self.memory)
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = client.chat(messages, temperature=0.2)
                parsed = parse_migration_v2(raw, expected_task_id=task_id)
                break
            except ValueError as e:
                if attempt == self.max_retries:
                    return None
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"]
                    + f"\n\n【修正】你上一次输出有格式错误，请严格按 JSON 格式重出：\n{e}",
                }

        policy = parsed["policy"]
        init_spec = _init_for_policy(policy, prev_strategy, self.max_lag)
        max_iter = (parsed["max_iter"] if parsed["max_iter"] is not None
                    else POLICY_DEFAULTS[policy]["max_iter"])
        return MigrationDecision(
            task_id=task_id,
            policy=policy,
            rationale=parsed["rationale"],
            init_spec=init_spec,
            max_iter=max_iter,
            drift_level=drift.level,
            source="llm",
        )
