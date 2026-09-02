# P1-A 自进化 Agent 测试套件（E1–E13）
# ---------------------------------------------------------------
# 运行：python tests/test_evolution_suite.py
#
# E1  spec 命名 / 归一化确定性往返
# E2  apply_actions（add/remove/replace/keep、缺失/重名、replace 保位）
# E3  validate_spec_list 静态检查（cross 依赖 / 回看越界 / 重复名 / 窗口不完整）
# E4  T5 parity 扩展（自定义 spec：扩展 rolling stat + cross）
# E5  evaluate_spec(baseline) 与手算基线逐位一致
# E6  ★回滚复现（完成标准：改善→回归→回滚→从 best 再改善）
# E7  显式 rollback 动作不破坏候选
# E8  memory record/retrieve 往返 + 相似度检索
# E9  error_profiler 正确性
# E10 _check_task_leakage 透传 spec（合法通过 / 泄漏中止）
# E11 Pass A cross 分支
# E12 EvolutionRunner + ScriptedLLM 端到端（memory 落盘）
# E13 CLI 冒烟
# ---------------------------------------------------------------
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evolution_runner import EvolutionRunner
from agent.feature_spec import apply_actions, name_from_spec, normalize_spec, snapshot, validate_spec_list
from agent.scripted_llm import ScriptedLLM, sequence
from data.task_builder import FEATURE_SPEC, build_task, feature_spec_hash
from evaluation.error_profiler import compute_error_profile, format_profile_for_llm
from evaluation.forecast_protocol import ONLINE_H1
from evaluation.leakage_checker import check_feature_leakage
from evaluation.rolling_backtest import _features_at, build_forecast_features
from evaluation.spec_evaluator import evaluate_spec
from evaluation.task_replay import LeakageError, replay
from memory.memory_manager import (
    MemoryManager,
    Scenario,
    StrategyRecord,
    ExperienceRecord,
)
from models.replay_backends import PersistenceBackend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


def _names(spec):
    return [s["name"] for s in spec]


def _v2(round_no, candidates):
    return json.dumps(
        {"round": round_no, "analysis": "test", "candidates": candidates},
        ensure_ascii=False,
    )


def _add_action(feature_spec):
    return {"type": "add_feature", "feature_spec": feature_spec}


# ---------------------------------------------------------------
# 模拟 spec_evaluator（确定性，按 spec 内容返回预设 RMSE）
# ---------------------------------------------------------------

def _mock_eval(rules):
    def _evaluate(task_id, zone, spec, protocol, val_hours=168, backend_factory=None,
                  seed=42, data_dir=None):
        rmse = float(rules([s["name"] for s in spec]))
        y_true = 100.0 + 10.0 * np.sin(np.linspace(0, 6.28, 48))
        y_pred = y_true + rmse / 5.0
        ts = pd.date_range("2020-01-01", periods=48, freq="h")
        return {
            "task_id": task_id, "spec": spec, "backend_name": "mock",
            "rmse": rmse, "mae": rmse,
            "y_true": y_true, "y_pred": y_pred, "forecast_ts": ts,
            "profile": compute_error_profile(y_true, y_pred, ts),
            "feature_importance": None, "best_iteration": None,
        }
    return _evaluate


# ---------------- E1 ----------------
def test_e1_spec_naming():
    # 确定性命名
    check("E1 lag 命名", name_from_spec({"type": "lag", "k": 48}) == "lag_48")
    check("E1 rolling 命名", name_from_spec({"type": "rolling", "stat": "max", "window": 24}) == "rolling_max_24")
    check("E1 time 命名", name_from_spec({"type": "time", "attr": "is_weekend"}) == "is_weekend")
    check("E1 cross 命名", name_from_spec({"type": "cross", "col1": "lag_1", "col2": "lag_24", "operation": "subtract"}) == "lag_1_minus_lag_24")

    # normalize 推导 lineage
    ns = normalize_spec({"type": "lag", "source": "LOAD", "k": 48}, snapshot(FEATURE_SPEC))
    check("E1 normalize lag 推导", ns["name"] == "lag_48" and ns["lookback_start"] == -48
          and ns["lookback_end"] == -48 and ns["uses_current_target"] is False)
    ns2 = normalize_spec({"type": "rolling", "source": "LOAD", "window": 24, "stat": "mean"}, snapshot(FEATURE_SPEC))
    check("E1 normalize rolling min_periods", ns2["min_periods"] == 24 and ns2["lookback_end"] == -1)
    ns3 = normalize_spec({"type": "cross", "col1": "lag_24", "col2": "lag_168", "operation": "subtract"}, snapshot(FEATURE_SPEC))
    check("E1 normalize cross lookback", ns3["name"] == "lag_24_minus_lag_168"
          and ns3["lookback_start"] == -168 and ns3["lookback_end"] == -24)

    # 非法输入
    try:
        normalize_spec({"type": "lag", "source": "LOAD", "k": 0}, snapshot(FEATURE_SPEC))
        check("E1 lag k=0 拒绝", False)
    except ValueError:
        check("E1 lag k=0 拒绝", True)
    try:
        normalize_spec({"type": "lag", "source": "LOAD", "k": 999}, snapshot(FEATURE_SPEC))
        check("E1 lag k>max 拒绝", False)
    except ValueError:
        check("E1 lag k>max 拒绝", True)
    try:
        normalize_spec({"type": "rolling", "source": "LOAD", "window": 1, "stat": "mean"}, snapshot(FEATURE_SPEC))
        check("E1 rolling window<2 拒绝", False)
    except ValueError:
        check("E1 rolling window<2 拒绝", True)
    try:
        normalize_spec({"type": "cross", "col1": "nope", "col2": "lag_24", "operation": "add"}, snapshot(FEATURE_SPEC))
        check("E1 cross 操作列缺失拒绝", False)
    except ValueError:
        check("E1 cross 操作列缺失拒绝", True)


# ---------------- E2 ----------------
def test_e2_apply_actions():
    base = snapshot(FEATURE_SPEC)
    # add
    s1 = apply_actions(base, [_add_action({"type": "lag", "source": "LOAD", "k": 48})])
    check("E2 add", _names(s1) == _names(base) + ["lag_48"])
    # remove
    s2 = apply_actions(base, [{"type": "remove_feature", "feature": "rolling_std_24"}])
    check("E2 remove", "rolling_std_24" not in _names(s2) and len(s2) == len(base) - 1)
    # replace 保位：rolling_mean_24 原位换成 rolling_mean_48（index 7）
    s3 = apply_actions(base, [{"type": "replace_feature", "feature": "rolling_mean_24",
                               "feature_spec": {"type": "rolling", "source": "LOAD", "window": 48, "stat": "mean"}}])
    names3 = _names(s3)
    check("E2 replace 保位", names3.index("rolling_mean_48") == 7 and "rolling_mean_24" not in names3)
    # keep
    s4 = apply_actions(base, [{"type": "keep"}])
    check("E2 keep", _names(s4) == _names(base))
    # remove 不存在 → ValueError
    try:
        apply_actions(base, [{"type": "remove_feature", "feature": "no_such_feature"}])
        check("E2 remove 不存在拒绝", False)
    except ValueError:
        check("E2 remove 不存在拒绝", True)
    # add 重名 → 跳过而非抛错（避免连坐整单候选）；warnings 出参记录提示
    _warn = []
    s5 = apply_actions(base, [_add_action({"type": "lag", "source": "LOAD", "k": 1})],
                       warnings=_warn)
    check("E2 add 重名跳过", _names(s5) == _names(base) and len(_warn) == 1)
    # 混合动作：1 个重名 + 1 个新特征 → 新特征仍生效，重名被跳过
    _warn2 = []
    s6 = apply_actions(base, [
        _add_action({"type": "lag", "source": "LOAD", "k": 1}),   # 重名（跳过）
        _add_action({"type": "lag", "source": "LOAD", "k": 48}),  # 新（生效）
    ], warnings=_warn2)
    check("E2 重名连坐解除", _names(s6) == _names(base) + ["lag_48"] and len(_warn2) == 1)


# ---------------- E3 ----------------
def test_e3_validate_spec_list():
    base = snapshot(FEATURE_SPEC)
    # cross 含 LOAD → 拒
    bad_cross = base + [{"name": "x", "type": "cross", "col1": "LOAD", "col2": "lag_24",
                         "operation": "subtract", "lookback_start": -1, "lookback_end": 0, "uses_current_target": False}]
    v = validate_spec_list(bad_cross)
    check("E3 cross 含 LOAD 拒", any(x.kind == "cross_uses_target" for x in v))
    # cross 操作列缺失 → 拒
    miss = base + [{"name": "x", "type": "cross", "col1": "ghost", "col2": "lag_24",
                    "operation": "add", "lookback_start": -1, "lookback_end": 0, "uses_current_target": False}]
    v = validate_spec_list(miss)
    check("E3 cross 操作列缺失拒", any(x.kind == "cross_unknown_operand" for x in v))
    # cross 顺序错误（操作列在自身之后）→ 拒
    order = [{"name": "x", "type": "cross", "col1": "lag_24", "col2": "lag_168",
              "operation": "subtract", "lookback_start": -1, "lookback_end": 0, "uses_current_target": False},
             *base]
    v = validate_spec_list(order)
    check("E3 cross 顺序错误拒", any(x.kind == "cross_operand_order" for x in v))
    # lag k=0 → 拒
    lag0 = base + [{"name": "lag_0", "type": "lag", "source": "LOAD", "k": 0,
                    "lookback_start": 0, "lookback_end": 0, "uses_current_target": False}]
    v = validate_spec_list(lag0)
    check("E3 lag k=0 拒", any(x.kind == "lag_le_0" for x in v))
    # lag k>168 → 拒
    lagbig = base + [{"name": "lag_200", "type": "lag", "source": "LOAD", "k": 200,
                      "lookback_start": -200, "lookback_end": -200, "uses_current_target": False}]
    v = validate_spec_list(lagbig)
    check("E3 lag 越界拒", any(x.kind == "lookback_exceeds_max" for x in v))
    # rolling min_periods<window → 拒（incomplete_window）
    rbad = base + [{"name": "r", "type": "rolling", "source": "LOAD", "window": 24, "stat": "mean",
                    "min_periods": 12, "lookback_start": -24, "lookback_end": -1, "uses_current_target": False}]
    v = validate_spec_list(rbad)
    check("E3 incomplete_window 拒", any(x.kind == "incomplete_window" for x in v))
    # 重复名 → 拒
    dup = base + [{"name": "lag_1", "type": "lag", "source": "LOAD", "k": 1,
                   "lookback_start": -1, "lookback_end": -1, "uses_current_target": False}]
    v = validate_spec_list(dup)
    check("E3 重复名拒", any(x.kind == "duplicate_feature" for x in v))
    # 合法 spec 通过
    good = base + [{"name": "lag_48", "type": "lag", "source": "LOAD", "k": 48,
                    "lookback_start": -48, "lookback_end": -48, "uses_current_target": False}]
    check("E3 合法 spec 通过", len(validate_spec_list(good)) == 0)


# ---------------- E4 ----------------
def test_e4_parity_extension():
    spec = snapshot(FEATURE_SPEC) + [
        {"name": "lag_48", "type": "lag", "source": "LOAD", "k": 48,
         "lookback_start": -48, "lookback_end": -48, "uses_current_target": False},
        {"name": "rolling_max_24", "type": "rolling", "source": "LOAD", "window": 24, "stat": "max",
         "min_periods": 24, "lookback_start": -24, "lookback_end": -1, "uses_current_target": False},
        {"name": "rolling_skew_168", "type": "rolling", "source": "LOAD", "window": 168, "stat": "skew",
         "min_periods": 168, "lookback_start": -168, "lookback_end": -1, "uses_current_target": False},
        {"name": "lag_24_minus_lag_168", "type": "cross", "col1": "lag_24", "col2": "lag_168",
         "operation": "subtract", "lookback_start": -168, "lookback_end": -24, "uses_current_target": False},
    ]
    t = build_task(1, spec=spec)
    observed = pd.concat([t.history_df["LOAD"], t.y_true])
    X = build_forecast_features(observed, t.forecast_ts, spec=spec)
    ok = True
    for ts in [t.forecast_ts[0], t.forecast_ts[100], t.forecast_ts[-1]]:
        rowv = _features_at(observed, ts, spec)
        rowb = X.loc[ts]
        for s in spec:
            a, b = rowv[s["name"]], rowb[s["name"]]
            if not (pd.isna(a) and pd.isna(b)) and not np.isclose(a, b, equal_nan=True):
                ok = False
                print(f"    mismatch {s['name']} @ {ts}: {a} vs {b}")
    check("E4 自定义 spec parity（扩展 stat + cross）", ok)


# ---------------- E5 ----------------
def test_e5_evaluate_spec_consistency():
    res = evaluate_spec(1, FEATURE_SPEC, ONLINE_H1,
                        backend_factory=lambda: PersistenceBackend())
    check("E5 evaluate_spec(baseline) == 手算 7.188017",
          abs(res["rmse"] - 7.188017) < 1e-4, f"got={res['rmse']:.6f}")
    check("E5 返回 profile", res["profile"].n == len(res["y_true"]))


# ---------------- E6（完成标准） ----------------
def test_e6_rollback_reproduction():
    # 确定性脚本：R1 改善 → R2 回归 → R3 从 best 再改善
    script = sequence([
        _v2(1, [{"candidate_id": 1, "hypothesis": "加 lag_48",
                 "actions": [_add_action({"type": "lag", "source": "LOAD", "k": 48})]}]),
        _v2(2, [{"candidate_id": 1, "hypothesis": "加 rolling_max_24（会导致回归）",
                 "actions": [_add_action({"type": "rolling", "source": "LOAD", "window": 24, "stat": "max"})]}]),
        _v2(3, [{"candidate_id": 1, "hypothesis": "加 lag_72（从 best 出发改善）",
                 "actions": [_add_action({"type": "lag", "source": "LOAD", "k": 72})]}]),
    ])

    def rules(names):
        s = set(names)
        if "lag_48" in s and "rolling_max_24" in s:
            return 6.2   # 回归
        if "lag_72" in s:
            return 5.5   # 新 best
        if "lag_48" in s:
            return 5.7   # 首次改善
        return 5.9       # baseline

    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryManager(Path(tmp) / "mem.jsonl")
        runner = EvolutionRunner(
            task_id=1, spec_evaluator=_mock_eval(rules), llm_client=ScriptedLLM(script),
            memory=memory, max_iter=3,
        )
        result = runner.run(verbose=False)

        check("E6 baseline=5.9", abs(result["baseline_rmse"] - 5.9) < 1e-9)
        outcomes = [r["outcome"] for r in result["summary"]]
        check("E6 轮次状态机 [improved, rolled_back, improved]",
              outcomes[1:] == ["improved", "rolled_back", "improved"], f"{outcomes}")

        # 回滚后 current==best，且不含回归特征
        r2 = result["round_records"][1]
        check("E6 R2 后 current==best",
              runner.current_spec == runner.best_spec)
        check("E6 R2 回滚移除回归特征",
              "rolling_max_24" not in _names(runner.best_spec)
              and "lag_48" in _names(runner.best_spec))
        # R3 从 best 出发得到新 best
        check("E6 R3 新 best=5.5", abs(result["best_rmse"] - 5.5) < 1e-9
              and result["best_round"] == 3)
        # memory 3 条，R2 记 rolled_back
        recs = memory.load_all()
        check("E6 memory 3 条", len(recs) == 3)
        check("E6 memory R2=rolled_back", recs[1].outcome == "rolled_back"
              and recs[1].accepted is False)


# ---------------- E7 ----------------
def test_e7_explicit_rollback():
    script = sequence([
        _v2(1, [{"candidate_id": 1, "hypothesis": "h", "actions": [_add_action({"type": "lag", "source": "LOAD", "k": 48})]}]),
        _v2(2, [{"candidate_id": 1, "hypothesis": "h", "actions": [
            {"type": "rollback"},
            _add_action({"type": "lag", "source": "LOAD", "k": 72}),
        ]}]),
    ])

    def rules(names):
        if "lag_72" in names:
            return 5.5
        if "lag_48" in names:
            return 5.7
        return 5.9

    with tempfile.TemporaryDirectory() as tmp:
        memory = MemoryManager(Path(tmp) / "mem.jsonl")
        runner = EvolutionRunner(
            task_id=1, spec_evaluator=_mock_eval(rules), llm_client=ScriptedLLM(script),
            memory=memory, max_iter=2,
        )
        result = runner.run(verbose=False)
        # R2 显式 rollback 候选应从 best 出发，被正确评估（improved）
        check("E7 显式 rollback 候选正常评估",
              result["summary"][-1]["outcome"] == "improved")
        r2_cand = result["round_records"][-1].get("best_candidate")
        # 显式 rollback：base 应为 R2 时刻的 best（baseline + lag_48），不是 current 派生状态
        expected_base = _names(snapshot(FEATURE_SPEC)) + ["lag_48"]
        check("E7 候选 base==best(baseline+lag_48)", r2_cand is not None
              and _names(r2_cand["base_spec"]) == expected_base)


# ---------------- E8 ----------------
def test_e8_memory_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        mm = MemoryManager(Path(tmp) / "mem.jsonl")
        r1 = ExperienceRecord(task_id=1, round=1, scenario=Scenario("summer", 0.90, 0.80, 0.20),
                              problem={"worst_segment": {"key": "load_peak", "rmse": 8.0}},
                              actions=[{"type": "add_feature"}],
                              before_rmse=5.0, after_rmse=4.5, delta_rmse=-0.5,
                              outcome="improved", accepted=True)
        r2 = ExperienceRecord(task_id=6, round=2, scenario=Scenario("winter", 0.30, 0.20, 0.50),
                              problem={}, actions=[{"type": "remove_feature"}],
                              before_rmse=6.0, after_rmse=6.3, delta_rmse=0.3,
                              outcome="rolled_back", accepted=False)
        mm.record(r1)
        mm.record(r2)
        check("E8 len==2", len(mm) == 2)
        check("E8 load_all 往返", len(mm.load_all()) == 2
              and mm.load_all()[0].scenario.season == "summer")
        # 相似度检索：夏季近场景 → r1 优先
        near = mm.retrieve(Scenario("summer", 0.89, 0.81, 0.21), top_k=1)
        check("E8 检索近场景", len(near) == 1 and near[0].task_id == 1)
        # 冬季查询 → r2 优先
        winter = mm.retrieve(Scenario("winter", 0.31, 0.21, 0.49), top_k=1)
        check("E8 检索冬季", len(winter) == 1 and winter[0].task_id == 6)

        # 策略级记忆（P1-B）：往返 + 场景相似度检索
        sr1 = StrategyRecord(task_id=1, spec=snapshot(FEATURE_SPEC), rmse=4.5,
                             scenario=Scenario("summer", 0.90, 0.80, 0.20),
                             stats={"mean": 100.0}, policy="cold_start",
                             init_max_iter=5, transfer_gap=None)
        sr2 = StrategyRecord(task_id=6, spec=snapshot(FEATURE_SPEC), rmse=3.9,
                             scenario=Scenario("winter", 0.30, 0.20, 0.50),
                             stats={"mean": 80.0}, policy="inherit",
                             init_max_iter=2, transfer_gap=0.12)
        mm.record_strategy(sr1)
        mm.record_strategy(sr2)
        check("E8 策略落盘", mm.strategies_path.exists()
              and len(mm.load_strategies()) == 2)
        loaded = mm.load_strategies()[0]
        check("E8 策略往返字段", loaded.spec == sr1.spec and loaded.rmse == sr1.rmse
              and loaded.scenario.season == "summer" and loaded.policy == "cold_start"
              and loaded.transfer_gap is None)
        near_s = mm.retrieve_strategies(Scenario("summer", 0.89, 0.81, 0.21), top_k=1)
        check("E8 策略检索近场景", len(near_s) == 1 and near_s[0].task_id == 1)
        winter_s = mm.retrieve_strategies(Scenario("winter", 0.31, 0.21, 0.49), top_k=1)
        check("E8 策略检索冬季", len(winter_s) == 1 and winter_s[0].task_id == 6)


# ---------------- E9 ----------------
def test_e9_error_profiler():
    n = 744
    ts = pd.date_range("2020-01-01", periods=n, freq="h")
    y_true = 100.0 + 20.0 * np.sin(np.arange(n) / 24.0 * 2 * np.pi) \
        + np.random.RandomState(0).randn(n) * 3.0
    y_pred = y_true + np.random.RandomState(1).randn(n) * 2.0
    p = compute_error_profile(y_true, y_pred, ts)
    check("E9 n 一致", p.n == n)
    hour_sum = sum(v["n"] for k, v in p.segments.items() if k.startswith("hour_"))
    check("E9 hour 段样本和==n", hour_sum == n)
    check("E9 top_worst≤20", len(p.top_worst) <= 20)
    check("E9 top_worst 降序", all(p.top_worst[i]["abs_error"] >= p.top_worst[i + 1]["abs_error"]
                                   for i in range(len(p.top_worst) - 1)))
    check("E9 bias 符号正确", abs(p.bias - float(np.mean(y_pred - y_true))) < 1e-6)
    txt = format_profile_for_llm(p)
    check("E9 format 含分段", "最差分段" in txt and "hour_00" in txt or "bias" in txt)


# ---------------- E10 ----------------
def test_e10_leak_check_spec_threading():
    from evaluation.forecast_protocol import get_protocol
    proto = get_protocol("online_h1")
    # 合法自定义 spec 通过
    good = snapshot(FEATURE_SPEC) + [
        {"name": "lag_48", "type": "lag", "source": "LOAD", "k": 48,
         "lookback_start": -48, "lookback_end": -48, "uses_current_target": False}]
    payload = replay([1], PersistenceBackend(), proto, leak_check="fast", spec=good)
    check("E10 合法 spec 通过", payload is not None)
    # 泄漏 spec（lag_0）被 LeakageError 中止
    leaky = snapshot(FEATURE_SPEC) + [
        {"name": "lag_0", "type": "lag", "source": "LOAD", "k": 0,
         "lookback_start": 0, "lookback_end": 0, "uses_current_target": False}]
    try:
        replay([1], PersistenceBackend(), proto, leak_check="fast", spec=leaky)
        check("E10 泄漏 spec 被中止", False)
    except LeakageError:
        check("E10 泄漏 spec 被中止", True)


# ---------------- E11 ----------------
def test_e11_pass_a_cross():
    good_cross = snapshot(FEATURE_SPEC) + [
        {"name": "lag_24_minus_lag_168", "type": "cross", "col1": "lag_24", "col2": "lag_168",
         "operation": "subtract", "lookback_start": -168, "lookback_end": -24, "uses_current_target": False}]
    v = validate_spec_list(good_cross)
    check("E11 cross(lag_24,lag_168) 通过", len(v) == 0, f"{[x.kind for x in v]}")
    bad_cross = snapshot(FEATURE_SPEC) + [
        {"name": "cross_target", "type": "cross", "col1": "LOAD", "col2": "lag_24",
         "operation": "subtract", "lookback_start": -1, "lookback_end": 0, "uses_current_target": False}]
    v = validate_spec_list(bad_cross)
    check("E11 cross 含 LOAD 拒", any(x.kind == "cross_uses_target" for x in v))


# ---------------- E12 ----------------
def test_e12_runner_end_to_end():
    script = sequence([
        _v2(1, [{"candidate_id": 1, "hypothesis": "h", "actions": [_add_action({"type": "lag", "source": "LOAD", "k": 48})]},
                {"candidate_id": 2, "hypothesis": "h2", "actions": [_add_action({"type": "rolling", "source": "LOAD", "window": 48, "stat": "std"})]}]),
        _v2(2, [{"candidate_id": 1, "hypothesis": "h", "actions": [{"type": "keep"}]}]),
    ])

    def rules(names):
        if "lag_48" in names:
            return 5.7
        if "rolling_std_48" in names:
            return 5.8
        return 5.9

    with tempfile.TemporaryDirectory() as tmp:
        mem_path = Path(tmp) / "mem.jsonl"
        memory = MemoryManager(mem_path)
        runner = EvolutionRunner(
            task_id=1, spec_evaluator=_mock_eval(rules), llm_client=ScriptedLLM(script),
            memory=memory, max_iter=2, n_candidates=2,
        )
        result = runner.run(verbose=False)
        check("E12 多候选择优 lag_48", result["best_rmse"] < 5.8
              and "lag_48" in _names(result["best_spec"]))
        check("E12 memory 落盘", mem_path.exists() and len(memory) >= 2)
        check("E12 result 字段齐全", "baseline_rmse" in result and "best_spec" in result
              and "summary" in result)

        # warm-start（P1-B）：init_spec 作为 Round 0 基线，
        # baseline_rmse == init_spec 在该 Task 的 RMSE（跨 Task 迁移的 transfer-gap 基础）
        init_spec = snapshot(FEATURE_SPEC) + [
            {"name": "lag_48", "type": "lag", "source": "LOAD", "k": 48,
             "lookback_start": -48, "lookback_end": -48, "uses_current_target": False}]
        runner_ws = EvolutionRunner(
            task_id=2, spec_evaluator=_mock_eval(rules), llm_client=ScriptedLLM(script),
            max_iter=1, init_spec=init_spec, init_spec_label="Task 1 best",
        )
        res_ws = runner_ws.run(verbose=False)
        check("E12 warm-start 基线=继承策略",
              abs(res_ws["baseline_rmse"] - 5.7) < 1e-9
              and res_ws["summary"][0]["note"] == "baseline Task 1 best",
              f"baseline={res_ws['baseline_rmse']:.4f} note={res_ws['summary'][0]['note']}")
        check("E12 warm-start baseline_spec==init_spec",
              res_ws["baseline_spec"] == init_spec)


# ---------------- E13 ----------------
def test_e13_cli_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "out"
        memfile = Path(tmp) / "mem.jsonl"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "run_self_evolving_agent.py"),
            "--task", "1", "--max-iter", "1", "--dry-run",
            "--model", "persistence", "--n-candidates", "2",
            "--outdir", str(outdir), "--memory-file", str(memfile),
        ]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=180, env=env)
        check("E13 CLI 返回码 0", proc.returncode == 0,
              f"rc={proc.returncode}\n{proc.stderr[-500:]}")
        check("E13 CLI 产物", outdir.exists()
              and (outdir / "run_manifest.json").exists()
              and (outdir / "best_features.txt").exists())


# ---------------- E14 ----------------
def test_e14_drift_detector():
    from data.availability import available_history
    from evaluation.drift_detector import (
        compute_task_stats,
        detect_drift,
        format_drift_for_llm,
    )

    def _mk(level, amp24, amp168, noise, seed=0):
        n = 1000
        t = np.arange(n)
        y = (level + amp24 * np.sin(2 * np.pi * t / 24)
             + amp168 * np.sin(2 * np.pi * t / 168)
             + np.random.RandomState(seed).randn(n) * noise)
        return pd.DataFrame(
            {"LOAD": y}, index=pd.date_range("2020-01-01", periods=n, freq="h")
        )

    A = _mk(100, 20, 8, 3)
    B = _mk(180, 20, 8, 3)      # 均值 +80（>2σ 池）
    C = _mk(100, 40, 16, 6)     # 波动倍增（std ≈ 2×）
    D = _mk(100, 20, 8, 40)     # 噪声淹没 → ACF 削弱
    sA, sB, sC, sD = (compute_task_stats(x, task_id=i)
                      for i, x in enumerate([A, B, C, D], 1))

    r_self = detect_drift(sA, sA)
    check("E14 自对比 score=0", r_self.drift_score == 0.0 and r_self.level == "low")

    r_ab = detect_drift(sA, sB)
    check("E14 均值漂移信号>0.5", r_ab.signals["mean_shift"] > 0.5,
          f"={r_ab.signals['mean_shift']:.2f}")
    check("E14 均值漂移 medium+", r_ab.level in ("medium", "high"),
          f"={r_ab.level} score={r_ab.drift_score:.2f}")

    r_ac = detect_drift(sA, sC)
    check("E14 波动倍增 std_shift>0.3", r_ac.signals["std_shift"] > 0.3,
          f"={r_ac.signals['std_shift']:.2f}")
    check("E14 波动倍增 medium", r_ac.level == "medium",
          f"={r_ac.level} score={r_ac.drift_score:.2f}")

    r_ad = detect_drift(sA, sD)
    check("E14 ACF 削弱 → high", r_ad.signals["acf24_change"] > 0.3 and r_ad.level == "high",
          f"acf24={r_ad.signals['acf24_change']:.2f} lvl={r_ad.level}")

    # 残余恶化（transfer-gap）应抬升 drift_score 并计入 scores
    r_abr = detect_drift(sA, sB, resid_trend=0.8)
    check("E14 残余恶化抬升 score", r_abr.drift_score > r_ab.drift_score
          and "residual" in r_abr.scores,
          f"{r_ab.drift_score:.2f}->{r_abr.drift_score:.2f}")

    txt = format_drift_for_llm(r_ab)
    check("E14 format 含字段", "drift_score" in txt and "mean_shift" in txt)

    # 真实数据：Task 1 vs 15（相隔一年、跨季节）→ medium+；自对比 → 0
    s1 = compute_task_stats(available_history(1).history_df, task_id=1)
    s15 = compute_task_stats(available_history(15).history_df, task_id=15)
    r_real = detect_drift(s1, s15)
    check("E14 真实 T1->T15 medium+", r_real.level in ("medium", "high")
          and r_real.drift_score > 0.2,
          f"={r_real.drift_score:.2f} {r_real.level}")


# ---------------- E15 ----------------
def test_e15_migration_decision():
    from agent.strategy_migration import (
        MigrationPlanner,
        build_migration_messages,
        cold_start_decision,
        default_decision,
        parse_migration_v2,
        resolve_init_spec,
    )
    from evaluation.drift_detector import DriftReport

    # ---- parse_migration_v2 ----
    good = parse_migration_v2({
        "task_id": 5, "analysis": "a",
        "decision": {"policy": "modify", "rationale": "mean_shift 高", "max_iter": 6},
    }, expected_task_id=5)
    check("E15 解析合法", good["policy"] == "modify" and good["max_iter"] == 6)
    for bad, name in [
        ({"task_id": 5, "analysis": "a", "decision": {"policy": "nope", "rationale": "x"}},
         "非法 policy 拒"),
        ({"task_id": 5, "analysis": "a", "decision": {"policy": "reset"}}, "缺 rationale 拒"),
        ({"task_id": 5, "analysis": "a",
          "decision": {"policy": "reset", "rationale": "x", "max_iter": 99}}, "max_iter 越界拒"),
    ]:
        try:
            parse_migration_v2(bad)
            check(f"E15 {name}", False)
        except ValueError:
            check(f"E15 {name}", True)
    try:
        parse_migration_v2({"task_id": 4, "analysis": "a",
                            "decision": {"policy": "reset", "rationale": "x"}},
                           expected_task_id=5)
        check("E15 task_id 不匹配拒", False)
    except ValueError:
        check("E15 task_id 不匹配拒", True)

    # ---- 确定性映射 ----
    prev_spec = snapshot(FEATURE_SPEC) + [{
        "name": "lag_48", "type": "lag", "source": "LOAD", "k": 48,
        "lookback_start": -48, "lookback_end": -48, "uses_current_target": False}]
    prev = StrategyRecord(task_id=4, spec=prev_spec, rmse=4.0,
                          scenario=Scenario("summer", 0.90, 0.80, 0.20))
    d_low = default_decision(5, "low", prev)
    check("E15 low→inherit 继承 prev", d_low.policy == "inherit" and d_low.max_iter == 2
          and d_low.init_spec == prev_spec, f"init={len(d_low.init_spec)}")
    d_med = default_decision(5, "medium", prev)
    check("E15 medium→modify", d_med.policy == "modify" and d_med.max_iter == 5
          and d_med.init_spec == prev_spec)
    d_hi = default_decision(5, "high", prev)
    check("E15 high→reset", d_hi.policy == "reset" and d_hi.max_iter == 8
          and d_hi.init_spec == snapshot(FEATURE_SPEC))
    d_cold = cold_start_decision(1)
    check("E15 冷启动 reset", d_cold.policy == "reset"
          and d_cold.init_spec == snapshot(FEATURE_SPEC) and d_cold.max_iter == 5)

    # ---- resolve_init_spec：泄漏回退 / 合法保留 ----
    leaky = snapshot(FEATURE_SPEC) + [{
        "name": "lag_0", "type": "lag", "source": "LOAD", "k": 0,
        "lookback_start": 0, "lookback_end": 0, "uses_current_target": False}]
    check("E15 泄漏 init_spec 回退", resolve_init_spec(leaky) == snapshot(FEATURE_SPEC))
    check("E15 合法 init_spec 保留", resolve_init_spec(prev_spec) == prev_spec)

    # ---- MigrationPlanner：LLM 路径 + 确定性兜底 + 冷启动 ----
    drift = DriftReport(task_id=5, compared_task_id=4, drift_score=0.70, level="high",
                        signals={"mean_shift": 0.5}, residual={},
                        scores={"data": 0.5}, meta={})
    script_llm = ScriptedLLM(lambda _: json.dumps({
        "task_id": 5, "analysis": "漂移大",
        "decision": {"policy": "reset", "rationale": "std_shift 高，重新搜索", "max_iter": 8},
    }, ensure_ascii=False))
    dec_llm = MigrationPlanner(llm_client=script_llm).plan(
        5, drift=drift, prev_strategy=prev, scenario=prev.scenario)
    check("E15 LLM 决策 reset", dec_llm.policy == "reset" and dec_llm.source == "llm"
          and dec_llm.max_iter == 8 and dec_llm.init_spec == snapshot(FEATURE_SPEC))
    dec_det = MigrationPlanner().plan(5, drift=drift, prev_strategy=prev)
    check("E15 确定性兜底", dec_det.source == "deterministic" and dec_det.policy == "reset")
    dec_cold = MigrationPlanner().plan(1)
    check("E15 planner 冷启动", dec_cold.policy == "reset" and dec_cold.max_iter == 5)

    # ---- prompt 渲染 ----
    msgs = build_migration_messages(5, drift, prev)
    check("E15 messages 含 drift 文本", "drift_score" in msgs[-1]["content"]
          and "上一 Task 最佳策略" in msgs[-1]["content"])


# ---------------- E16 ----------------
def test_e16_outer_loop_cli():
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp) / "outer"
        memfile = Path(tmp) / "mem.jsonl"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "run_outer_loop.py"),
            "--tasks", "1:3", "--model", "persistence", "--dry-run",
            "--n-candidates", "2", "--outdir", str(outdir),
            "--memory-file", str(memfile), "--quiet",
        ]
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=300, env=env)
        check("E16 CLI 返回码 0", proc.returncode == 0,
              f"rc={proc.returncode}\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")

        # 策略落盘 3 条（strategies.jsonl 与记忆文件同目录）
        from memory.memory_manager import MemoryManager
        mm = MemoryManager(memfile)
        recs = mm.load_strategies()
        check("E16 策略落盘 3 条", len(recs) == 3 and mm.strategies_path.exists(),
              f"len={len(recs)}")

        # 逐 Task 审计产物
        check("E16 审计产物",
              (outdir / "task_01" / "decision.json").exists()
              and (outdir / "task_01" / "strategy.json").exists()
              and (outdir / "task_02" / "drift_report.json").exists()
              and (outdir / "task_03" / "summary.json").exists()
              and (outdir / "outer_loop_summary.csv").exists())

        # 冷启动 Task1 drift=null；Task2/3 有漂移报告
        d1 = json.loads((outdir / "task_01" / "drift_report.json").read_text(encoding="utf-8"))
        d2 = json.loads((outdir / "task_02" / "drift_report.json").read_text(encoding="utf-8"))
        d3 = json.loads((outdir / "task_03" / "drift_report.json").read_text(encoding="utf-8"))
        check("E16 冷启动 drift=null", d1 is None)
        check("E16 Task2/3 drift 有效",
              d2 is not None and d2["level"] in ("low", "medium", "high")
              and d3 is not None and d3["level"] in ("low", "medium", "high"))

        # 继承链：Task2 的 decision.init_spec == Task1 的 best spec
        s1 = json.loads((outdir / "task_01" / "strategy.json").read_text(encoding="utf-8"))
        dec2 = json.loads((outdir / "task_02" / "decision.json").read_text(encoding="utf-8"))
        t1_names = {s["name"] for s in s1["spec"]}
        check("E16 Task2 继承 Task1 特征集",
              set(dec2["init_feature_names"]) == t1_names
              and dec2["policy"] in ("inherit", "modify", "reset"),
              f"policy={dec2['policy']}")

        # 总表 3 行
        import csv
        with open(outdir / "outer_loop_summary.csv", encoding="utf-8-sig") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        check("E16 总表 3 行", n_rows == 3, f"n={n_rows}")


def main():
    print("=" * 60)
    test_e1_spec_naming()
    test_e2_apply_actions()
    test_e3_validate_spec_list()
    test_e4_parity_extension()
    test_e5_evaluate_spec_consistency()
    test_e6_rollback_reproduction()
    test_e7_explicit_rollback()
    test_e8_memory_roundtrip()
    test_e9_error_profiler()
    test_e10_leak_check_spec_threading()
    test_e11_pass_a_cross()
    test_e12_runner_end_to_end()
    test_e13_cli_smoke()
    test_e14_drift_detector()
    test_e15_migration_decision()
    test_e16_outer_loop_cli()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
