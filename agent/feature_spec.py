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
# ---------------------------------------------------------------
from copy import deepcopy
from typing import Dict, List

from data.task_builder import MAX_LAG
from evaluation.leakage_checker import LeakageViolation, _pass_a

# ---------------------------------------------------------------
# 允许的取值（P1 LOAD-only protocol）
# ---------------------------------------------------------------
TARGET_COL = "LOAD"

TIME_ATTRS = {"hour", "weekday", "month", "is_weekend"}
ROLLING_STATS = {"mean", "std", "var", "max", "min", "median", "sum", "skew", "kurt"}
CROSS_OPS = {"add", "subtract", "multiply", "divide"}
_OP_WORDS = {"add": "plus", "subtract": "minus", "multiply": "multiply", "divide": "div"}

ACTION_TYPES = {
    "add_feature", "remove_feature", "replace_feature",
    "keep", "rollback", "stop",
}


# ---------------------------------------------------------------
# 确定性命名
# ---------------------------------------------------------------

def name_from_spec(spec_entry: dict) -> str:
    """从 spec 确定性生成列名。"""
    stype = spec_entry["type"]
    if stype == "time":
        return spec_entry["attr"]
    if stype == "lag":
        return f"lag_{spec_entry['k']}"
    if stype == "rolling":
        return f"rolling_{spec_entry['stat']}_{spec_entry['window']}"
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
) -> dict:
    """
    把 LLM 提供的扁平 feature_spec 归一化为完整血缘 dict。

    - name / lookback_* / min_periods / uses_current_target 由本函数推导并覆盖
      （LLM 提供的 name 一律忽略，确定性命名消除列名冲突）。
    - 约束不满足抛 ValueError（apply_actions 会使其候选作废，错误进 retry 反馈）。
    """
    stype = raw.get("type")
    if stype not in ("time", "lag", "rolling", "cross"):
        raise ValueError(f"feature_spec 的 type 必须是 time/lag/rolling/cross，got {stype!r}")

    if stype == "time":
        attr = raw.get("attr")
        if attr not in TIME_ATTRS:
            raise ValueError(f"time 特征的 attr 必须是 {sorted(TIME_ATTRS)}，got {attr!r}")
        return {
            "name": attr, "type": "time", "attr": attr,
            "lookback_start": 0, "lookback_end": 0, "uses_current_target": False,
        }

    if stype == "lag":
        source = raw.get("source", target_col)
        if source != target_col:
            raise ValueError(f"P1 阶段 lag 只能作用于 {target_col}，got source={source!r}")
        k = raw.get("k")
        if not isinstance(k, int) or isinstance(k, bool) or not (1 <= k <= max_lag):
            raise ValueError(f"lag k 必须是 1..{max_lag} 的整数，got {k!r}")
        return {
            "name": f"lag_{k}", "type": "lag", "source": source, "k": k,
            "lookback_start": -k, "lookback_end": -k, "uses_current_target": False,
        }

    if stype == "rolling":
        source = raw.get("source", target_col)
        if source != target_col:
            raise ValueError(f"P1 阶段 rolling 只能作用于 {target_col}，got source={source!r}")
        window = raw.get("window")
        if not isinstance(window, int) or isinstance(window, bool) or not (2 <= window <= max_lag):
            raise ValueError(f"rolling window 必须是 2..{max_lag} 的整数，got {window!r}")
        stat = raw.get("stat")
        if stat not in ROLLING_STATS:
            raise ValueError(f"rolling stat 必须是 {sorted(ROLLING_STATS)}，got {stat!r}")
        return {
            "name": f"rolling_{stat}_{window}", "type": "rolling",
            "source": source, "window": window, "stat": stat,
            "min_periods": window,  # 强制窗口完整（incomplete_window 违规）
            "lookback_start": -window, "lookback_end": -1, "uses_current_target": False,
        }

    # cross
    col1, col2 = raw.get("col1"), raw.get("col2")
    op = raw.get("operation")
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
) -> List[LeakageViolation]:
    """
    候选 spec 的静态检查：复用 leakage_checker._pass_a（血缘泄漏）+ 新增：
      - 重复特征名 → duplicate_feature
      - lag.k / rolling.window 超过 max_lag → lookback_exceeds_max
    （cross 依赖缺失/顺序由 _pass_a 的 cross 分支覆盖。）
    """
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


def apply_actions(base_spec: List[dict], actions: List[dict]) -> List[dict]:
    """
    按序解释动作，返回新 spec 列表。任何非法动作抛 ValueError（候选作废）。

    说明：
      - add_feature    追加 normalize 后的 spec（cross 操作列必须已存在）
      - remove_feature 按 name 删除（不存在 → ValueError）
      - replace_feature 原位替换（位置不变，name 由新 spec 推导）
      - keep           无操作
      - rollback / stop 顶层处理，本函数忽略（不报错）
    """
    spec = snapshot(base_spec)
    for a in actions:
        atype = a.get("type")
        if atype not in ACTION_TYPES:
            raise ValueError(f"未知动作类型 {atype!r}，允许: {sorted(ACTION_TYPES)}")

        if atype == "add_feature":
            new_spec = normalize_spec(a["feature_spec"], spec)
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
            new_spec = normalize_spec(a["feature_spec"], spec)
            spec = [new_spec if s["name"] == name else s for s in spec]

        elif atype == "keep":
            pass

        # rollback / stop：由 evolution_runner 顶层识别，apply_actions 不解释
    return spec
