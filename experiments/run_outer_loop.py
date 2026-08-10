# P1-B 外循环 CLI：跨 Task 漂移检测 + 策略迁移（滚动自适应进化双闭环）
# ---------------------------------------------------------------
# 用法：
#   python experiments/run_outer_loop.py --tasks 1:3 --dry-run --model persistence
#   python experiments/run_outer_loop.py --tasks 1:15 --model lightgbm           # 真实 LLM 迁移决策
#   python experiments/run_outer_loop.py --tasks 1:3 --dry-run --with-reference-baseline
#
# 流程（不复用 replay()——那是静态 backend 循环、无跨 Task 状态）：
#   逐 Task：
#     compute_task_stats（尾部窗口）→ detect_drift(对比 prev 策略) → MigrationPlanner
#     → EvolutionRunner(init_spec=决策.init_spec, max_iter=决策.max_iter)   # warm-start
#     → record_strategy（含 transfer_gap）→ 下一步
#
# 产出（--outdir 默认 experiments/output/outer_loop）：
#   task_{id:02d}/{drift_report, decision, strategy, summary}.json  逐 Task 审计
#   outer_loop_summary.csv  总表（Task / drift / policy / rmse / transfer_gap）
#   run_manifest.json
# ---------------------------------------------------------------
import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.evolution_runner import EvolutionRunner  # noqa: E402
from agent.feature_agent import QwenClient  # noqa: E402
from agent.feature_spec import snapshot  # noqa: E402
from agent.scripted_llm import ScriptedLLM  # noqa: E402
from agent.strategy_migration import MigrationPlanner  # noqa: E402
from data.availability import available_history  # noqa: E402
from data.task_builder import FEATURE_SPEC  # noqa: E402
from evaluation.drift_detector import (  # noqa: E402
    TaskStats,
    compute_scenario,
    compute_task_stats,
    detect_drift,
    format_drift_for_llm,
)
from evaluation.forecast_protocol import get_protocol  # noqa: E402
from evaluation.spec_evaluator import evaluate_spec  # noqa: E402
from memory.memory_manager import MemoryManager, StrategyRecord  # noqa: E402
from models.replay_backends import make_backend  # noqa: E402
from run_self_evolving_agent import _demo_script  # noqa: E402
from run_task_replay import parse_tasks  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _residual_trend(strategies: Dict[int, StrategyRecord]) -> Optional[float]:
    """残余漂移信号：最近一个已完成 Task 的 transfer_gap（上一策略迁移到它的退化幅度）。"""
    if not strategies:
        return None
    last = strategies[max(strategies)]
    return last.transfer_gap


def _stats_from_record(rec: StrategyRecord) -> TaskStats:
    """StrategyRecord.stats（dict）→ TaskStats。"""
    return TaskStats(**rec.stats)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def run_outer_loop(args) -> List[Dict]:
    tasks: List[int] = parse_tasks(args.tasks)
    protocol = get_protocol(args.protocol)
    backend_factory = lambda: make_backend(args.model)  # noqa: E731

    outdir = Path(args.outdir) if args.outdir else (
        PROJECT_ROOT / "experiments" / "output" / "outer_loop"
    )
    memory = MemoryManager(Path(args.memory_file)) if args.memory_file else MemoryManager()

    # LLM 客户端：dry-run 用 ScriptedLLM（进化闭环）；迁移决策走确定性兜底。
    if args.dry_run:
        llm_client: Optional[QwenClient] = None
        evolution_llm = ScriptedLLM(_demo_script(args.n_candidates))
        print(f"模式: DRY RUN（迁移=确定性映射，进化=ScriptedLLM）  Tasks: {tasks}  模型: {args.model}")
    else:
        try:
            llm_client = QwenClient()
            evolution_llm = llm_client
            print(f"模式: API（{llm_client.model}）  Tasks: {tasks}  模型: {args.model}")
        except ValueError as e:
            print(f"[FATAL] API Key 未配置: {e}（可用 --dry-run）", file=sys.stderr)
            return []

    planner = MigrationPlanner(llm_client=llm_client, memory=memory)

    strategies: Dict[int, StrategyRecord] = {}
    prev_strategy: Optional[StrategyRecord] = None
    rows: List[Dict] = []

    for tid in tasks:
        print(f"\n{'=' * 70}\nTask {tid} / {tasks[-1]}\n{'=' * 70}")
        av = available_history(tid)
        stats_cur = compute_task_stats(av.history_df, task_id=tid)
        scenario = compute_scenario(av.history_df, av.forecast_ts, task_id=tid)
        print(f"  场景: {scenario.season} acf24={scenario.acf_24:.2f} "
              f"acf168={scenario.acf_168:.2f} cv={scenario.load_cv:.2f} "
              f"尾窗 {stats_cur.tail_start}")

        # ---- 漂移检测（对比 prev 策略的尾部统计 + 残余趋势） ----
        drift = None
        if prev_strategy is not None:
            drift = detect_drift(
                _stats_from_record(prev_strategy), stats_cur,
                resid_prev=prev_strategy.profile,
                resid_trend=_residual_trend(strategies),
            )
            print(f"  Drift Task {prev_strategy.task_id}→{tid}: "
                  f"score={drift.drift_score:.3f} level={drift.level}")

        # ---- 迁移决策（LLM 或确定性兜底） ----
        decision = planner.plan(tid, drift=drift, prev_strategy=prev_strategy,
                                scenario=scenario)
        print(f"  迁移: policy={decision.policy} max_iter={decision.max_iter} "
              f"init_features={len(decision.init_spec)} source={decision.source}")
        if drift is not None and args.verbose:
            print(format_drift_for_llm(drift))

        # ---- 内循环：EvolutionRunner（warm-start init_spec + 自适应 max_iter） ----
        runner = EvolutionRunner(
            task_id=tid,
            backend_factory=backend_factory,
            protocol=protocol,
            llm_client=evolution_llm,
            memory=memory,
            n_candidates=args.n_candidates,
            max_iter=decision.max_iter,
            val_hours=args.val_hours,
            seed=args.seed,
            init_spec=decision.init_spec,
            init_spec_label=(f"Task {prev_strategy.task_id} best"
                             if prev_strategy is not None else "FEATURE_SPEC"),
        )
        result = runner.run(verbose=args.verbose)
        print(f"  Task {tid}: baseline RMSE={result['baseline_rmse']:.4f} → "
              f"best RMSE={result['best_rmse']:.4f} (round {result['best_round']})")

        # ---- 参考基线（可选）：FEATURE_SPEC 在该 Task 的 RMSE ----
        ref_rmse = None
        if args.with_reference_baseline:
            ref = evaluate_spec(tid, snapshot(FEATURE_SPEC), protocol,
                                val_hours=args.val_hours,
                                backend_factory=backend_factory, seed=args.seed)
            ref_rmse = float(ref["rmse"])
            print(f"  参考基线 FEATURE_SPEC RMSE = {ref_rmse:.4f}")

        # ---- 策略入库（含 transfer_gap = 继承策略在本 Task 的退化幅度） ----
        gap: Optional[float] = None
        if prev_strategy is not None and prev_strategy.rmse > 0:
            gap = (result["baseline_rmse"] - prev_strategy.rmse) / prev_strategy.rmse

        profile = runner._current_result["profile"] if runner._current_result else None
        strategy = StrategyRecord(
            task_id=tid,
            spec=result["best_spec"],
            rmse=result["best_rmse"],
            scenario=scenario,
            stats=stats_cur.to_dict(),
            profile=asdict(profile) if profile else {},
            policy=decision.policy,
            init_max_iter=decision.max_iter,
            transfer_gap=gap,
        )
        memory.record_strategy(strategy)
        strategies[tid] = strategy
        prev_strategy = strategy

        # ---- 逐 Task 审计落盘 ----
        row = {
            "task_id": tid,
            "forecast_month": av.forecast_month,
            "drift_level": drift.level if drift else "n/a",
            "drift_score": round(drift.drift_score, 4) if drift else None,
            "policy": decision.policy,
            "decision_source": decision.source,
            "max_iter": decision.max_iter,
            "baseline_rmse": round(result["baseline_rmse"], 4),
            "best_rmse": round(result["best_rmse"], 4),
            "best_round": result["best_round"],
            "ref_rmse": round(ref_rmse, 4) if ref_rmse is not None else None,
            "transfer_gap": round(gap, 4) if gap is not None else None,
            "n_features": len(result["best_spec"]),
        }
        rows.append(row)
        _write_task_audit(outdir, tid, drift, decision, strategy, result, row)

    _write_outputs(outdir, rows, args)
    return rows


def _write_task_audit(outdir: Path, tid: int, drift, decision, strategy,
                      result, row: Dict) -> None:
    d = outdir / f"task_{tid:02d}"
    _write_json(d / "drift_report.json", drift.to_dict() if drift else None)
    _write_json(d / "decision.json", decision.to_dict())
    _write_json(d / "strategy.json", asdict(strategy))
    _write_json(d / "summary.json", {
        **row,
        "best_features": [s["name"] for s in strategy.spec],
        "baseline_features": [s["name"] for s in result["baseline_spec"]],
    })


def _write_outputs(outdir: Path, rows: List[Dict], args) -> None:
    import pandas as pd
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"outer_loop_summary_{ts}.csv", index=False, encoding="utf-8-sig")
    df.to_csv(outdir / "outer_loop_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "tasks": [r["task_id"] for r in rows],
        "protocol": args.protocol,
        "model": args.model,
        "n_candidates": args.n_candidates,
        "seed": args.seed,
        "val_hours": args.val_hours,
        "dry_run": args.dry_run,
        "migration": "deterministic" if args.dry_run else "llm",
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tasks": len(rows),
    }
    _write_json(outdir / "run_manifest.json", manifest)
    print(f"\n  审计输出 -> {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B 跨 Task 漂移检测 + 策略迁移（外循环）")
    parser.add_argument("--tasks", default="1:15", help="任务范围，如 1:15 / 1,3,5")
    parser.add_argument("--model", default="lightgbm",
                        help="lightgbm | persistence | seasonal_naive_24 | seasonal_naive_168")
    parser.add_argument("--protocol", default="online_h1",
                        choices=["online_h1", "recursive_month_ahead"])
    parser.add_argument("--dry-run", action="store_true",
                        help="迁移决策走确定性兜底 + 进化闭环用 ScriptedLLM")
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=0,
                        help=">0 时覆盖迁移决策的每 Task max_iter（调试用）")
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--with-reference-baseline", action="store_true",
                        help="每 Task 多算一次 FEATURE_SPEC 参考 RMSE（审计对比）")
    parser.add_argument("--quiet", action="store_true", help="抑制逐 Task verbose")
    args = parser.parse_args()
    args.verbose = not args.quiet

    rows = run_outer_loop(args)
    if not rows:
        return 1

    mean_best = sum(r["best_rmse"] for r in rows) / len(rows)
    print(f"\n{'=' * 70}")
    print(f"外循环完成：{len(rows)} Task  |  mean best RMSE = {mean_best:.4f}")
    for r in rows:
        print(f"  Task {r['task_id']:>2} [{r['policy']:<7}] drift={r['drift_level']:<6} "
              f"best={r['best_rmse']:.4f} gap={r['transfer_gap'] if r['transfer_gap'] is not None else '-':<7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
