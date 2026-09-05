# 自进化 Agent：LLM 输出 v2 schema + Prompt 构建
# ---------------------------------------------------------------
# 顶层输出（一次调用返回全部候选）：
#   { "round": int, "analysis": str,
#     "candidates": [ { "candidate_id": int, "hypothesis": str, "actions": [...] } ] }
#
# 动作类型与 feature_spec 的解析/校验由 agent/feature_spec.py 承担，
# 本模块只做 JSON 结构 + 顶层字段校验，动作参数留给 apply_actions 校验。
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.feature_agent import _extract_json
from agent.feature_spec import ACTION_TYPES
from data.task_builder import MAX_LAG

MAX_ACTIONS_PER_CANDIDATE = 5


def parse_llm_v2(raw: Any, n_candidates_max: int = 3) -> Dict:
    """
    校验 LLM 输出 v2，返回标准化 dict。失败抛 ValueError（可进重试循环）。
    """
    if isinstance(raw, str):
        raw = _extract_json(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"输出必须是 JSON 对象，实际类型: {type(raw).__name__}")

    missing = [k for k in ("round", "analysis", "candidates") if k not in raw]
    if missing:
        raise ValueError(f"缺少顶层必填字段: {missing}")

    cands = raw["candidates"]
    if not isinstance(cands, list) or not (1 <= len(cands) <= n_candidates_max):
        raise ValueError(
            f"candidates 必须是 1..{n_candidates_max} 的数组，实际 {len(cands) if isinstance(cands, list) else type(cands).__name__}"
        )

    validated = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            raise ValueError(f"candidates[{i}] 必须是 JSON 对象")
        for f in ("candidate_id", "hypothesis", "actions"):
            if f not in c:
                raise ValueError(f"candidates[{i}] 缺少必填字段 '{f}'")
        acts = c["actions"]
        if not isinstance(acts, list) or len(acts) > MAX_ACTIONS_PER_CANDIDATE:
            raise ValueError(
                f"candidates[{i}].actions 必须是 0..{MAX_ACTIONS_PER_CANDIDATE} 的数组"
            )
        for j, a in enumerate(acts):
            if not isinstance(a, dict) or a.get("type") not in ACTION_TYPES:
                raise ValueError(
                    f"candidates[{i}].actions[{j}] 非法，type 必须是 "
                    f"{sorted(ACTION_TYPES)}"
                )
        validated.append({
            "candidate_id": int(c["candidate_id"]),
            "hypothesis": str(c["hypothesis"]),
            "actions": acts,
        })

    return {
        "round": int(raw["round"]),
        "analysis": str(raw["analysis"]),
        "candidates": validated,
    }


# ---------------------------------------------------------------
# 上下文（runner 每轮构建 → build_llm_v2_messages 渲染）
# ---------------------------------------------------------------

@dataclass
class EvolutionContext:
    task_id: int
    round: int
    max_iterations: int
    n_candidates: int
    dataset_name: str = ""
    season: str = ""
    acf_24: float = 0.0
    acf_168: float = 0.0
    load_cv: float = 0.0
    current_features: List[str] = field(default_factory=list)
    best_rmse: float = 0.0
    baseline_rmse: float = 0.0
    best_round: int = 0
    error_profile_text: str = ""
    feature_importance_text: str = ""
    memories_text: str = ""
    round_history_text: str = ""
    max_lag: int = MAX_LAG
    # ★ 三档动作空间
    feature_tier: int = 3
    target_col: str = "LOAD"
    exogenous_cols: tuple = ()
    # ★ 领域知识增强提示（按数据集注入）
    domain_knowledge: str = ""


def _spec_help(max_lag: int, feature_tier: int = 3,
               target_col: str = "LOAD", exogenous_cols: tuple = ()) -> str:
    """feature_spec 类型说明（喂 system prompt），按档位渲染可用操作与 source 范围。"""
    sources = [target_col]
    if feature_tier >= 2:
        sources += list(exogenous_cols)
    src_str = " / ".join(f'"{s}"' for s in sources)
    lines = [
        "## feature_spec 允许的类型（血缘格式，name 由系统推导，无需你给列名）",
        "",
        f"1. time   {{\"type\":\"time\",  \"attr\":\"hour|weekday|month|is_weekend\"}}",
        f"2. lag    {{\"type\":\"lag\",   \"source\":\"{target_col}\", \"k\":1..{max_lag}}}",
    ]
    if feature_tier >= 2:
        lines.append(
            f"   - lag/rolling 的 source 可取 {src_str}（能源领域外生列，如气象变量）"
        )
    lines.append(
        f"3. rolling{{\"type\":\"rolling\",\"source\":\"{target_col}\", "
        f"\"window\":2..{max_lag}, \"stat\":\"mean|std|var|max|min|median|sum|skew|kurt\"}}"
    )
    if feature_tier >= 3:
        lines += [
            "4. cross  {\"type\":\"cross\", \"col1\":\"已有特征名\", \"col2\":\"已有特征名\", "
            "\"operation\":\"add|subtract|multiply|divide\"}",
            "   - cross 的 col1/col2 必须是**当前特征集里已存在**的特征（建议 lag / rolling / time），"
            f"禁止直接用 {target_col}；若想用某特征做交叉，先在同一候选里 add 它。",
        ]
    else:
        lines.append(
            f"（当前档位 tier={feature_tier}：未开放 cross 组合特征）"
        )
    lines += [
        "",
        "动作类型：",
        "  add_feature     {\"type\":\"add_feature\", \"feature_spec\":{...}}",
        "  remove_feature  {\"type\":\"remove_feature\", \"feature\":\"现有特征名\"}",
        "  replace_feature {\"type\":\"replace_feature\", \"feature\":\"现有特征名\", \"feature_spec\":{...}}",
        "  keep            {\"type\":\"keep\"}",
        "  rollback        {\"type\":\"rollback\"}   # 从 best-so-far 状态重新开始探索",
        "  stop            {\"type\":\"stop\"}       # 提议本轮停止",
    ]
    return "\n".join(lines)


def _system_prompt(max_lag: int, n_candidates: int, feature_tier: int = 3,
                   target_col: str = "LOAD", exogenous_cols: tuple = (),
                   domain_knowledge: str = "") -> str:
    return f"""你是能源预测的**特征工程决策 Agent**。你只做决策，不写代码；
所有特征由确定性引擎执行。你的目标是提升**预测月 online_h1 滚动 RMSE**。

## 时间因果红线（最高优先级）
- 特征只能使用 ≤ t-1 的信息：lag/rolling 必须严格过去，禁止任何未来信息。
- cross 只能组合"过去向特征"，禁止直接用当前时刻的 {target_col}。

{_spec_help(max_lag, feature_tier, target_col, tuple(exogenous_cols))}
{domain_knowledge}

## 每轮动作
每轮返回 **{n_candidates} 个候选假设**（可 1~{n_candidates} 个），每个候选包含：
- hypothesis：基于误差画像 + 特征重要性的诊断与假设（必须与其它候选**不同假设**，
  例如：A 加强日周期、B 加强周周期、C 针对峰值/特定时段误差）；
- actions：一组有序动作（0~5 个），对该候选要评估的特征集做修改。

## 特征设计原则
1. 误差驱动：先看误差画像的 worst_segment / bias，针对性设计（如晚峰低估 →
   加 hour×weekday 交互或对应时段 lag）；不要只看整体 RMSE 泛泛加 lag。
2. 冗余抑制：已有 lag_24 不要再加 lag_23/25；已有 rolling_mean_24 不要重复。
3. 保守回看：所有 lag/rolling 的 k/window ≤ {max_lag}。
4. 考虑回滚：若上轮已回滚（rolled_back），应从 best 状态出发提出**不同方向**的假设，
   不要重复失败过的动作组合。

## 严格输出格式
只输出一个 JSON 对象，不要任何额外文字：
{{
  "round": 当前轮次,
  "analysis": "中文诊断（≤500字）",
  "candidates": [
    {{
      "candidate_id": 1,
      "hypothesis": "中文假设（≥10字）",
      "actions": [ {{"type":"add_feature","feature_spec":{{"type":"lag","source":"{target_col}","k":48}}}} ]
    }}
  ]
}}
"""


def build_llm_v2_messages(ctx: EvolutionContext) -> List[Dict]:
    """渲染 system + user 两条消息。"""
    sys = _system_prompt(ctx.max_lag, ctx.n_candidates, ctx.feature_tier,
                         ctx.target_col, tuple(ctx.exogenous_cols),
                         ctx.domain_knowledge)

    feat_list = ", ".join(ctx.current_features) if ctx.current_features else "（无）"
    tier_desc = {
        1: "基本时序（目标列 lag/time）",
        2: "能源专有（+ 外生列 lag/rolling）",
        3: "组合类（+ cross 组合特征）",
    }.get(ctx.feature_tier, f"tier={ctx.feature_tier}")
    user = f"""## 任务
Task {ctx.task_id}（{ctx.dataset_name}），第 {ctx.round}/{ctx.max_iterations} 轮。
目标：在预测月 online_h1 滚动评测下降低 RMSE。返回 {ctx.n_candidates} 个**假设不同**的候选。

## 场景
- 季节: {ctx.season or "?"}  ACF(lag24)={ctx.acf_24:.3f}  ACF(lag168)={ctx.acf_168:.3f}  load_cv={ctx.load_cv:.3f}
- 动作空间档位: tier={ctx.feature_tier}（{tier_desc}）
- 当前特征（{len(ctx.current_features)} 个）: {feat_list}
- baseline RMSE = {ctx.baseline_rmse:.4f}；当前 best RMSE = {ctx.best_rmse:.4f}（第 {ctx.best_round} 轮）
{ctx.round_history_text}

{ctx.error_profile_text}

{ctx.feature_importance_text}

{ctx.memories_text}

## 输出
严格按 system 提示的 JSON 格式输出 {ctx.n_candidates} 个候选。
"""
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def format_round_history(records: List[Dict]) -> str:
    """把迭代历史格式化为 prompt 文本。records: [{round, outcome, best_rmse, delta_rmse, candidate_note}]"""
    if not records:
        return ""
    lines = ["## 迭代历史"]
    for r in records:
        lines.append(
            f"  R{r['round']} [{r['outcome']}] best={r['best_rmse']:.4f} "
            f"Δ={r['delta_rmse']:+.4f} {r.get('candidate_note', '')}"
        )
    return "\n".join(lines)


def format_feature_importance(df) -> str:
    """把特征重要性 DataFrame 格式化为 prompt 文本。"""
    if df is None or len(df) == 0:
        return ""
    lines = ["## 特征重要性 (gain)"]
    for _, row in df.head(10).iterrows():
        lines.append(
            f"  {row['feature']:<24s} gain={float(row['importance_gain']):>12.1f} "
            f"({float(row['importance_gain_norm']):>6.2f}%)"
        )
    if len(df) > 10:
        lines.append(f"  ... (共 {len(df)} 个)")
    return "\n".join(lines)
