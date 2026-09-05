# 血缘式特征 spec 工具 + 自进化 Agent 动作解释器
# ---------------------------------------------------------------
# P1-A 的 Agent 状态 = 有序血缘 spec 列表（与 data/task_builder.FEATURE_SPEC
# 同构，由 build_features 执行）。本模块提供：
#
#   name_from_spec     确定性列名（同 spec 恒同列，LLM 不需发明列名）
#   normalize_spec     LLM 原始 feature_spec → 完整血缘 dict（推导 name/
#                      lookback/min_periods/uses_current_target，校验约束）
#   validate_spec_list 候选 spec 全集的静态检查（复用 leakage_checker Pass A
#                      + 重复名 / 回看越界 / cross 依赖）
#   apply_actions      动作解释器：ADD/REMOVE/REPLACE/KEEP → 新 spec 列表
#   snapshot           深拷贝快照（回滚语义的基础）
#
# 动作空间：add_feature / remove_feature / replace_feature / keep / rollback / stop。
# rollback / stop 不在 apply_actions 内解释（由 evolution_runner 顶层处理）。
#
# ★ 三档动作空间（feature_tier）：
#   Tier 1（基本时序）  time + 目标列 lag —— 最保守，仅目标列时序特征
#   Tier 2（能源专有）  + 外生列（exogenous_cols，如 Solar 气象 VAR169/164/167）
#                       的 lag / rolling —— 能源领域专有特征
#   Tier 3（组合类）    + cross —— 组合特征，给 LLM 最多空间
#   默认 Tier 3（兼容历史行为）；cross 始终要求操作列是已定义的过去向特征，
#   保持"受约束动作空间 + 时间因果"的核心叙事（不开放任意列直用）。
# ---------------------------------------------------------------
from copy import deepcopy
from typing import Dict, List

from data.task_builder import MAX_LAG
from evaluation.leakage_checker import LeakageViolation, _pass_a

# ---------------------------------------------------------------
# 允许的取值
# ---------------------------------------------------------------
TARGET_COL = "LOAD"

FEATURE_TIER_FULL = 3  # 默认档位 = 历史全能力（含 cross），保证向后兼容
FEATURE_TIERS = (1, 2, 3)

TIME_ATTRS = {"hour", "weekday", "month", "is_weekend"}
ROLLING_STATS = {"mean", "std", "var", "max", "min", "median", "sum", "skew", "kurt"}
CROSS_OPS = {"add", "subtract", "multiply", "divide"}
_OP_WORDS = {"add": "plus", "subtract": "minus", "multiply": "multiply", "divide": "div"}

# cross operation 容错别名（LLM 常用缩写 → 合法值），降低非法 operation 导致的 retry
_OP_ALIASES = {
    "add": "add", "plus": "add", "sum": "add",
    "subtract": "subtract", "minus": "subtract", "sub": "subtract", "diff": "subtract",
    "multiply": "multiply", "mul": "multiply", "times": "multiply", "product": "multiply",
    "divide": "divide", "div": "divide", "ratio": "divide",
}


def _normalize_cross_op(op) -> str:
    """把 LLM 常见的 operation 缩写/别名映射到合法值（非字符串原样返回，供后续报错）。"""
    if not isinstance(op, str):
        return op
    return _OP_ALIASES.get(op.strip().lower(), op)

ACTION_TYPES = {
    "add_feature", "remove_feature", "replace_feature",
    "keep", "rollback", "stop",
}


def _check_tier(tier: int) -> None:
    if tier not in FEATURE_TIERS:
        raise ValueError(f"feature_tier 必须是 {FEATURE_TIERS} 之一，got {tier!r}")


def _check_source(source, target_col, exogenous_cols, feature_tier, stype) -> None:
    """按档位校验 lag/rolling 的 source 可选范围。

    Tier 1: 仅目标列；Tier 2+: 目标列 ∪ 外生列。
    """
    allowed = [target_col]
    if feature_tier >= 2:
        allowed += list(exogenous_cols)
    if source not in allowed:
        raise ValueError(
            f"{stype} 的 source 必须是 {allowed}，got {source!r}"
            f"（feature_tier={feature_tier}，外生列={list(exogenous_cols) or '无'}）"
        )


# ---------------------------------------------------------------
# 确定性命名
# ---------------------------------------------------------------

def name_from_spec(spec_entry: dict, target_col: str = TARGET_COL) -> str:
    """从 spec 确定性生成列名。"""
    stype = spec_entry["type"]
    if stype == "time":
        return spec_entry["attr"]
    if stype == "lag":
        source = spec_entry.get("source", target_col)
        return f"lag_{spec_entry['k']}" if source == target_col else f"{source}_lag_{spec_entry['k']}"
    if stype == "rolling":
        source = spec_entry.get("source", target_col)
        base = f"rolling_{spec_entry['stat']}_{spec_entry['window']}"
        return base if source == target_col else f"{source}_{base}"
    if stype == "current":
        return f"{spec_entry['source']}_current"
    if stype == "cross":
        return f"{spec_entry['col1']}_{_OP_WORDS[spec_entry['operation']]}_{spec_entry['col2']}"
    raise ValueError(f"未知特征类型: {stype}")


# ---------------------------------------------------------------
# LLM feature_spec → 完整血缘 dict
# ---------------------------------------------------------------

def _lookback_of(specs: List[dict], name: str):
    """取某特征的血缘 lookback 区间（未找到返回保守默认）。"""
    for s in specs:
        if s.get("name") == name:
            return s.get("lookback_start", -1), s.get("lookback_end", 0)
    return -1, 0


def normalize_spec(
    raw: dict,
    existing_specs: List[dict],
    target_col: str = TARGET_COL,
    max_lag: int = MAX_LAG,
    feature_tier: int = FEATURE_TIER_FULL,
    exogenous_cols: List[str] = None,
) -> dict:
    """
    把 LLM 提供的扁平 feature_spec 归一化为完整血缘 dict。

    - name / lookback_* / min_periods / uses_current_target 由本函数推导并覆盖
      （LLM 提供的 name 一律忽略，确定性命名消除列名冲突）。
    - 约束不满足抛 ValueError（apply_actions 会使其候选作废，错误进 retry 反馈）。
    - feature_tier 控制动作空间（见模块头）：
        Tier 1: 仅目标列 lag/time（无 rolling / 无外生 / 无 cross）
        Tier 2: + 外生列 lag/rolling（能源专有）
        Tier 3: + cross（组合类）
    """
    _check_tier(feature_tier)
    exogenous_cols = list(exogenous_cols or [])

    stype = raw.get("type")
    if stype not in ("time", "lag", "rolling", "cross", "current"):
        raise ValueError(f"feature_spec 的 type 必须是 time/lag/rolling/cross/current，got {stype!r}")

    if stype == "time":
        attr = raw.get("attr")
        if attr not in TIME_ATTRS:
            raise ValueError(f"time 特征的 attr 必须是 {sorted(TIME_ATTRS)}，got {attr!r}")
        return {
            "name": attr, "type": "time", "attr": attr,
            "lookback_start": 0, "lookback_end": 0, "uses_current_target": False,
        }

    if stype == "current":
        source = raw.get("source")
        if not source:
            raise ValueError("current 特征必须提供 source 外生列")
        if source == target_col:
            raise ValueError(f"current 特征只能作用于外生列，禁止 source={target_col}（用当前目标=泄漏）")
        _check_source(source, target_col, exogenous_cols, feature_tier, stype)
        return {
            "name": f"{source}_current", "type": "current", "source": source,
            "lookback_start": 0, "lookback_end": 0, "uses_current_target": False,
        }

    if stype == "lag":
        source = raw.get("source", target_col)
        _check_source(source, target_col, exogenous_cols, feature_tier, stype)
        k = raw.get("k")
        if not isinstance(k, int) or isinstance(k, bool) or not (1 <= k <= max_lag):
            raise ValueError(f"lag k 必须是 1..{max_lag} 的整数，got {k!r}")
        name = f"lag_{k}" if source == target_col else f"{source}_lag_{k}"
        return {
            "name": name, "type": "lag", "source": source, "k": k,
            "lookback_start": -k, "lookback_end": -k, "uses_current_target": False,
        }

    if stype == "rolling":
        source = raw.get("source", target_col)
        _check_source(source, target_col, exogenous_cols, feature_tier, stype)
        window = raw.get("window")
        if not isinstance(window, int) or isinstance(window, bool) or not (2 <= window <= max_lag):
            raise ValueError(f"rolling window 必须是 2..{max_lag} 的整数，got {window!r}")
        stat = raw.get("stat")
        if stat not in ROLLING_STATS:
            raise ValueError(f"rolling stat 必须是 {sorted(ROLLING_STATS)}，got {stat!r}")
        base = f"rolling_{stat}_{window}"
        name = base if source == target_col else f"{source}_{base}"
        return {
            "name": name, "type": "rolling",
            "source": source, "window": window, "stat": stat,
            "min_periods": window,  # 强制窗口完整（incomplete_window 违规）
            "lookback_start": -window, "lookback_end": -1, "uses_current_target": False,
        }

    # cross
    if feature_tier < 3:
        raise ValueError(
            f"cross 特征仅在 feature_tier=3（组合类）开放，当前 tier={feature_tier}"
        )
    col1, col2 = raw.get("col1"), raw.get("col2")
    op = _normalize_cross_op(raw.get("operation"))
    if op not in CROSS_OPS:
        raise ValueError(f"cross operation 必须是 {sorted(CROSS_OPS)}，got {op!r}")
    existing_names = [s.get("name") for s in existing_specs]
    for c in (col1, col2):
        if c not in existing_names:
            raise ValueError(f"cross 操作列 {c} 不在现有特征中（必须先 add 再 cross）")
    # 操作列血缘（保守求交，保证 lookback 不夸大回看）
    lb1_s, lb1_e = _lookback_of(existing_specs, col1)
    lb2_s, lb2_e = _lookback_of(existing_specs, col2)
    uses_cur = any(
        s.get("name") in (col1, col2) and s.get("uses_current_target", False)
        for s in existing_specs
    )
    return {
        "name": f"{col1}_{_OP_WORDS[op]}_{col2}", "type": "cross",
        "col1": col1, "col2": col2, "operation": op,
        "lookback_start": min(lb1_s, lb2_s), "lookback_end": max(lb1_e, lb2_e),
        "uses_current_target": bool(uses_cur),
    }


# ---------------------------------------------------------------
# 候选 spec 全集的静态检查
# ---------------------------------------------------------------

def validate_spec_list(
    spec: List[dict],
    target_col: str = TARGET_COL,
    max_lag: int = MAX_LAG,
    pred_horizon: int = 1,
    feature_tier: int = FEATURE_TIER_FULL,
    exogenous_cols: List[str] = None,
) -> List[LeakageViolation]:
    """
    候选 spec 的静态检查：复用 leakage_checker._pass_a（血缘泄漏）+ 新增：
      - 重复特征名 → duplicate_feature
      - lag.k / rolling.window 超过 max_lag → lookback_exceeds_max
      - 档位外 source / cross → tier_disallowed（跨过 normalize 的硬校验防线）
    （cross 依赖缺失/顺序由 _pass_a 的 cross 分支覆盖。）
    """
    exogenous_cols = list(exogenous_cols or [])
    feature_cols = [s["name"] for s in spec]
    violations: List[LeakageViolation] = list(
        _pass_a(spec, feature_cols, target_col, pred_horizon)
    )

    seen: Dict[str, int] = {}
    for i, s in enumerate(spec):
        name = s["name"]
        if name in seen:
            violations.append(LeakageViolation(
                name, None, "duplicate_feature",
                f"特征名 {name} 重复（第 {seen[name]}、{i + 1} 位）"))
        seen[name] = i + 1

        stype = s["type"]
        if stype in ("lag", "rolling", "current"):
            src = s.get("source", target_col)
            allowed = [target_col] + ([*exogenous_cols] if feature_tier >= 2 else [])
            if src not in allowed:
                violations.append(LeakageViolation(
                    name, None, "tier_disallowed",
                    f"{stype} {name} 的 source={src!r} 超出 tier={feature_tier} "
                    f"允许范围 {allowed}"))
        if stype == "cross" and feature_tier < 3:
            violations.append(LeakageViolation(
                name, None, "tier_disallowed",
                f"cross {name} 在 tier={feature_tier} 未开放（需 tier=3）"))

        if stype == "lag" and s.get("k", 0) > max_lag:
            violations.append(LeakageViolation(
                name, None, "lookback_exceeds_max",
                f"lag {name} 的 k={s['k']} > max_lag={max_lag}，会破坏切分安全"))
        elif stype == "rolling" and s.get("window", 0) > max_lag:
            violations.append(LeakageViolation(
                name, None, "lookback_exceeds_max",
                f"rolling {name} 的 window={s['window']} > max_lag={max_lag}，会破坏切分安全"))

    return violations


# ---------------------------------------------------------------
# 动作解释器
# ---------------------------------------------------------------

def snapshot(spec: List[dict]) -> List[dict]:
    """深拷贝 spec 快照（best_spec 永不突变的基础）。"""
    return deepcopy(spec)


def apply_actions(
    base_spec: List[dict],
    actions: List[dict],
    target_col: str = TARGET_COL,
    max_lag: int = MAX_LAG,
    feature_tier: int = FEATURE_TIER_FULL,
    exogenous_cols: List[str] = None,
) -> List[dict]:
    """
    按序解释动作，返回新 spec 列表。任何非法动作抛 ValueError（候选作废）。

    说明：
      - add_feature    追加 normalize 后的 spec（cross 操作列必须已存在）
      - remove_feature 按 name 删除（不存在 → ValueError）
      - replace_feature 原位替换（位置不变，name 由新 spec 推导）
      - keep           无操作
      - rollback / stop 顶层处理，本函数忽略（不报错）
    - feature_tier / target_col / exogenous_cols 透传给 normalize_spec（三档动作空间）。
    """
    spec = snapshot(base_spec)
    for a in actions:
        atype = a.get("type")
        if atype not in ACTION_TYPES:
            raise ValueError(f"未知动作类型 {atype!r}，允许: {sorted(ACTION_TYPES)}")

        if atype == "add_feature":
            new_spec = normalize_spec(
                a["feature_spec"], spec,
                target_col=target_col, max_lag=max_lag,
                feature_tier=feature_tier, exogenous_cols=exogenous_cols,
            )
            if new_spec["name"] in [s["name"] for s in spec]:
                raise ValueError(
                    f"add_feature: 特征 {new_spec['name']} 已存在，不可重复添加")
            spec.append(new_spec)

        elif atype == "remove_feature":
            name = a["feature"]
            names = [s["name"] for s in spec]
            if name not in names:
                raise ValueError(f"remove_feature: 特征 {name!r} 不存在于当前特征集")
            spec = [s for s in spec if s["name"] != name]

        elif atype == "replace_feature":
            name = a["feature"]
            names = [s["name"] for s in spec]
            if name not in names:
                raise ValueError(f"replace_feature: 特征 {name!r} 不存在于当前特征集")
            new_spec = normalize_spec(
                a["feature_spec"], spec,
                target_col=target_col, max_lag=max_lag,
                feature_tier=feature_tier, exogenous_cols=exogenous_cols,
            )
            spec = [new_spec if s["name"] == name else s for s in spec]

        elif atype == "keep":
            pass

        # rollback / stop：由 evolution_runner 顶层识别，apply_actions 不解释
    return spec
