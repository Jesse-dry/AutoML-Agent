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
from memory.memory_manager import MemoryManager, Scenario, ExperienceRecord
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
    def _evaluate(task_id, spec, protocol, val_hours=168, backend_factory=None,
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
    # add 重名 → ValueError
    try:
        apply_actions(base, [_add_action({"type": "lag", "source": "LOAD", "k": 1})])
        check("E2 add 重名拒绝", False)
    except ValueError:
        check("E2 add 重名拒绝", True)


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
                              text=True, timeout=180, env=env)
        check("E13 CLI 返回码 0", proc.returncode == 0,
              f"rc={proc.returncode}\n{proc.stderr[-500:]}")
        check("E13 CLI 产物", outdir.exists()
              and (outdir / "run_manifest.json").exists()
              and (outdir / "best_features.txt").exists())


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
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
