# P1-A 自进化特征工程 Agent CLI
# ---------------------------------------------------------------
# 用法：
#   python experiments/run_self_evolving_agent.py --task 1 --max-iter 3 --dry-run --model persistence
#   python experiments/run_self_evolving_agent.py --task 15 --max-iter 5          # 真实 LLM（需 DASHSCOPE_API_KEY）
#   python experiments/run_self_evolving_agent.py --task 4 --n-candidates 3 --max-iter 6
#
# 产出（--outdir 默认 experiments/output/evolution_task{id}）：
#   summary.json / iteration_history.csv / best_features.txt / run_manifest.json
#   error_profile_best.txt / memory 写入 memory/experiment_memory.jsonl
# ---------------------------------------------------------------
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.domain_knowledge import (  # noqa: E402
    build_domain_knowledge_section,
    get_domain_knowledge,
    get_domain_knowledge_by_key,
)
from agent.evolution_runner import EvolutionRunner  # noqa: E402
from agent.feature_agent import QwenClient, compute_acf_summary  # noqa: E402
from agent.feature_spec import snapshot  # noqa: E402
from agent.scripted_llm import ScriptedLLM  # noqa: E402
from data.solar_loader import SOLAR_DATA_DIR, solar_available_history  # noqa: E402
from data.solar_task_builder import (  # noqa: E402
    SOLAR_COLD_START_FEATURE_SPEC,
    SOLAR_FEATURE_SPEC,
    SOLAR_TARGET_COL,
    SOLAR_WEATHER_COLS,
)
from data.task_builder import COLD_START_FEATURE_SPEC, FEATURE_SPEC  # noqa: E402
from data.wind_loader import WIND_DATA_DIR, wind_available_history  # noqa: E402
from data.wind_task_builder import (  # noqa: E402
    WIND_COLD_START_FEATURE_SPEC,
    WIND_FEATURE_SPEC,
    WIND_TARGET_COL,
)
from evaluation.error_profiler import format_profile_for_llm  # noqa: E402
from evaluation.forecast_protocol import get_protocol  # noqa: E402
from evaluation.solar_spec_evaluator import evaluate_solar_spec  # noqa: E402
from evaluation.wind_spec_evaluator import evaluate_wind_spec  # noqa: E402
from memory.memory_manager import (  # noqa: E402
    MemoryManager,
    Scenario,
    season_from_month,
)
from models.replay_backends import make_backend  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _build_solar_scenario(task_id: int, data_dir, zone):
    """Solar 场景构建器（注入 EvolutionRunner.scenario_builder）。"""
    av = solar_available_history(task_id, zone, data_dir)
    pwr = av.history_df[SOLAR_TARGET_COL].dropna()
    cv = float(pwr.std() / pwr.mean()) if pwr.mean() != 0 else 0.0
    acf = compute_acf_summary(av.history_df, SOLAR_TARGET_COL, lags=[24, 168])
    season = season_from_month(av.forecast_ts[0].month)
    return Scenario(
        season=season,
        acf_24=float(acf.get(24, 0.0)),
        acf_168=float(acf.get(168, 0.0)),
        load_cv=cv,
    )


def _build_wind_scenario(task_id: int, data_dir, zone):
    """Wind 场景构建器（注入 EvolutionRunner.scenario_builder）。"""
    av = wind_available_history(task_id, zone, data_dir)
    w = av.history_df[WIND_TARGET_COL].dropna()
    cv = float(w.std() / w.mean()) if w.mean() != 0 else 0.0
    acf = compute_acf_summary(av.history_df, WIND_TARGET_COL, lags=[24, 168])
    season = season_from_month(av.forecast_ts[0].month)
    return Scenario(
        season=season,
        acf_24=float(acf.get(24, 0.0)),
        acf_168=float(acf.get(168, 0.0)),
        load_cv=cv,
    )


def _demo_script(n_candidates: int = 3, target_col: str = "LOAD",
                 feature_tier: int = 3, exogenous_cols: tuple = ()):
    """--dry-run 的确定性演示脚本：每轮 n_candidates 个不同方向的候选。

    target_col / feature_tier / exogenous_cols 数据集感知：
      - lag/rolling source 用 target_col（Solar 时自动 POWER）
      - tier≥2 时演示外生列 lag（Solar 气象 VAR169）
      - tier≥3 才用 cross 组合（否则 fallback 到 lag/rolling）
    """
    import json as _json

    exo = tuple(exogenous_cols or ())
    exo_src = exo[0] if exo else None

    def _lag(k):
        return {"type": "add_feature",
                "feature_spec": {"type": "lag", "source": target_col, "k": k}}

    def _rolling(window, stat="std"):
        return {"type": "add_feature",
                "feature_spec": {"type": "rolling", "source": target_col,
                                 "window": window, "stat": stat}}

    def _exo_lag(k=24):
        return {"type": "add_feature",
                "feature_spec": {"type": "lag", "source": exo_src, "k": k}}

    hyp_map = {
        1: "补中程滞后与波动刻画（日周期增强）",
        2: "补更长滞后 / 滚动统计（趋势增强）",
        3: "补长周期滞后 / 滚动（周周期增强）",
    }
    rounds = [
        [{"candidate_id": 1, "hypothesis": hyp_map[1], "actions": [_lag(48), _rolling(48, "std")]},
         {"candidate_id": 2, "hypothesis": hyp_map[1], "actions": [
             ({"type": "add_feature",
               "feature_spec": {"type": "cross", "col1": "lag_24",
                                "col2": "lag_1", "operation": "subtract"}}
              if feature_tier >= 3 else _rolling(24, "mean"))]}],
        [{"candidate_id": 1, "hypothesis": hyp_map[2], "actions": [_lag(72)]},
         {"candidate_id": 2, "hypothesis": hyp_map[2], "actions": [
             ({"type": "remove_feature", "feature": "rolling_std_48"}
              if feature_tier >= 3 else _rolling(48, "mean")),
             (_exo_lag(24) if feature_tier >= 2 and exo_src else _rolling(168, "mean"))]}],
        [{"candidate_id": 1, "hypothesis": hyp_map[3], "actions": [_lag(120), _rolling(168, "std")]},
         {"candidate_id": 2, "hypothesis": hyp_map[3], "actions": [
             ({"type": "keep"} if feature_tier >= 3 else _lag(96))]}],
    ]

    def _script(round_no: int) -> str:
        idx = min(round_no - 1, len(rounds) - 1)
        payload = {
            "round": round_no,
            "analysis": "[DRY RUN] 演示脚本：生成多个不同方向候选。",
            "candidates": rounds[idx][:n_candidates],
        }
        return _json.dumps(payload, ensure_ascii=False)

    return _script


def _write_outputs(args, result, profile_text, outdir) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 迭代表
    summary_rows = result["summary"]
    df = pd.DataFrame(summary_rows)
    df.to_csv(outdir / f"iteration_history_{ts}.csv", index=False, encoding="utf-8-sig")

    # 汇总 JSON
    summary = {
        "timestamp": ts,
        "task": result["task_id"],
        "protocol": args.protocol,
        "model": args.model,
        "n_candidates": args.n_candidates,
        "baseline_rmse": result["baseline_rmse"],
        "best_rmse": result["best_rmse"],
        "best_round": result["best_round"],
        "n_evaluations": result["n_evaluations"],
        "best_features": [s["name"] for s in result["best_spec"]],
        "rounds": summary_rows,
    }
    with open(outdir / f"summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # best_features
    with open(outdir / "best_features.txt", "w", encoding="utf-8") as f:
        f.write(f"# Best features (Task {result['task_id']}, round {result['best_round']})\n")
        f.write(f"# RMSE: {result['baseline_rmse']:.4f} → {result['best_rmse']:.4f}\n\n")
        for s in result["best_spec"]:
            f.write(f"{s['name']}\n")

    # 误差画像
    if profile_text:
        with open(outdir / f"error_profile_best_{ts}.txt", "w", encoding="utf-8") as f:
            f.write(profile_text + "\n")

    # run_manifest
    manifest = {
        "task": result["task_id"],
        "dataset": getattr(args, "dataset", "load"),
        "zone": getattr(args, "zone", None),
        "feature_tier": getattr(args, "feature_tier", 3),
        "baseline": getattr(args, "baseline", "cold_start"),
        "domain_key": getattr(args, "domain_key", None),
        "protocol": args.protocol,
        "model": args.model,
        "n_candidates": args.n_candidates,
        "max_iter": args.max_iter,
        "val_hours": args.val_hours,
        "seed": args.seed,
        "baseline_rmse": result["baseline_rmse"],
        "best_rmse": result["best_rmse"],
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  审计输出 -> {outdir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-A 自进化特征工程 Agent")
    parser.add_argument("--task", type=int, default=1, help="GEFCom Task 1..15")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="用 ScriptedLLM 演示脚本")
    parser.add_argument("--protocol", default="online_h1", choices=["online_h1", "recursive_month_ahead"])
    parser.add_argument("--n-candidates", type=int, default=3)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--model", default="lightgbm",
                        help="lightgbm | persistence | seasonal_naive_24 | seasonal_naive_168")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--memory-file", default=None, help="记忆文件路径（默认 memory/experiment_memory.jsonl）")
    parser.add_argument("--max-lag", type=int, default=168)
    # ★ 三档动作空间 + 数据集
    parser.add_argument("--dataset", default="load", choices=["load", "solar", "wind"],
                        help="load | solar | wind（solar/wind 需 --zone）")
    parser.add_argument("--zone", type=int, default=1, help="Solar zone 1..3 / Wind zone 1..10")
    parser.add_argument("--feature-tier", type=int, default=3,
                        help="动作空间档位：1=基本时序 2=能源专有 3=组合类")
    parser.add_argument("--baseline", default="cold_start", choices=["cold_start", "full"],
                        help="Agent 起点基线：cold_start=极简起点（砍基线）| full=完整特征集")
    parser.add_argument("--patience", type=int, default=2,
                        help="连续无改善轮数上限（默认 2；多 seed 对比建议 3 让每档跑满）")
    parser.add_argument("--domain-key", default=None,
                        help="领域先验 key（默认按 --dataset 自动选）。"
                             "负对照实验：solar_neg_humidity / solar_neg_load / solar_neg_wind /"
                             " load_neg_solar / load_neg_humidity / wind_neg_load / wind_neg_solar；"
                             "空字符串 '' 表示无先验。")
    parser.add_argument("--data-dir", default=None,
                        help="数据目录覆盖（wind 数据默认在 GEFCom2014 Data/GEFCom2014-W_V2/Wind）")
    args = parser.parse_args()

    if not (1 <= args.task <= 15):
        print(f"[ERROR] task 必须在 1..15", file=sys.stderr)
        return 1
    if not (1 <= args.n_candidates <= 3):
        print(f"[ERROR] n_candidates 必须在 1..3", file=sys.stderr)
        return 1
    if args.feature_tier not in (1, 2, 3):
        print(f"[ERROR] feature_tier 必须是 1|2|3", file=sys.stderr)
        return 1
    if args.dataset == "solar" and not (1 <= args.zone <= 3):
        print(f"[ERROR] Solar zone 必须在 1..3", file=sys.stderr)
        return 1
    if args.dataset == "wind" and not (1 <= args.zone <= 10):
        print(f"[ERROR] Wind zone 必须在 1..10", file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else (
        PROJECT_ROOT / "experiments" / "output" / f"evolution_{args.dataset}_task{args.task}"
    )
    memory = MemoryManager(Path(args.memory_file)) if args.memory_file else MemoryManager()

    # ---- 数据集适配：load / solar ----
    # 领域知识增强：每个数据集注入专属特征工程先验（domain_knowledge.py）
    if args.domain_key is not None:
        domain_text = get_domain_knowledge_by_key(args.domain_key)  # 支持负对照 key
    else:
        domain_text = get_domain_knowledge(args.dataset)
    domain_knowledge = build_domain_knowledge_section(text=domain_text) if domain_text else ""
    if args.dataset == "solar":
        target_col = SOLAR_TARGET_COL
        exogenous_cols = list(SOLAR_WEATHER_COLS)
        spec_evaluator = evaluate_solar_spec
        scenario_builder = _build_solar_scenario
        data_dir = SOLAR_DATA_DIR
        dataset_name = f"GEFCom2014 Solar Task {args.task} Zone {args.zone}"
        init_spec = snapshot(
            SOLAR_COLD_START_FEATURE_SPEC if args.baseline == "cold_start"
            else SOLAR_FEATURE_SPEC)
        init_spec_label = ("SOLAR_COLD_START" if args.baseline == "cold_start"
                           else "SOLAR_FEATURE_SPEC")
    elif args.dataset == "wind":
        target_col = WIND_TARGET_COL
        exogenous_cols = ["ws10", "ws100"]  # wind 气象派生列（spec 的 source）
        spec_evaluator = evaluate_wind_spec
        scenario_builder = _build_wind_scenario
        data_dir = Path(args.data_dir) if args.data_dir else WIND_DATA_DIR
        dataset_name = f"GEFCom2014 Wind Task {args.task} Zone {args.zone}"
        init_spec = snapshot(
            WIND_COLD_START_FEATURE_SPEC if args.baseline == "cold_start"
            else WIND_FEATURE_SPEC)
        init_spec_label = ("WIND_COLD_START" if args.baseline == "cold_start"
                           else "WIND_FEATURE_SPEC")
    else:
        target_col = "LOAD"
        exogenous_cols = []
        spec_evaluator = None  # 默认 evaluate_spec（Load）
        scenario_builder = None
        data_dir = None
        dataset_name = f"GEFCom2014 Task {args.task}"
        init_spec = snapshot(
            COLD_START_FEATURE_SPEC if args.baseline == "cold_start"
            else FEATURE_SPEC)
        init_spec_label = ("COLD_START" if args.baseline == "cold_start"
                           else "FEATURE_SPEC")

    # LLM 客户端
    if args.dry_run:
        llm_client = ScriptedLLM(_demo_script(
            args.n_candidates, target_col=target_col,
            feature_tier=args.feature_tier, exogenous_cols=tuple(exogenous_cols)))
        print(f"模式: DRY RUN（ScriptedLLM）  Task: {args.task}  "
              f"数据集: {args.dataset}  协议: {args.protocol}  tier={args.feature_tier}  "
              f"基线: {args.baseline}")
    else:
        try:
            llm_client = QwenClient()
            print(f"模式: API（{llm_client.model}）  Task: {args.task}  "
                  f"数据集: {args.dataset}  协议: {args.protocol}  tier={args.feature_tier}  "
                  f"基线: {args.baseline}")
        except ValueError as e:
            print(f"[FATAL] API Key 未配置: {e}（可用 --dry-run）", file=sys.stderr)
            return 1

    runner = EvolutionRunner(
        task_id=args.task,
        backend_factory=lambda: make_backend(args.model),
        protocol=get_protocol(args.protocol),
        llm_client=llm_client,
        memory=memory,
        n_candidates=args.n_candidates,
        max_iter=args.max_iter,
        val_hours=args.val_hours,
        seed=args.seed,
        max_lag=args.max_lag,
        dataset_name=dataset_name,
        feature_tier=args.feature_tier,
        target_col=target_col,
        exogenous_cols=exogenous_cols,
        zone=args.zone if args.dataset in ("solar", "wind") else None,
        scenario_builder=scenario_builder,
        spec_evaluator=spec_evaluator,
        data_dir=data_dir,
        init_spec=init_spec,
        init_spec_label=init_spec_label,
        patience=args.patience,
        domain_knowledge=domain_knowledge,
    )

    try:
        result = runner.run(verbose=True)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # 输出 best 的误差画像
    profile_text = ""
    if runner._current_result is not None:
        profile_text = format_profile_for_llm(runner._current_result["profile"])

    print(f"\n{'=' * 60}")
    print(f"Baseline RMSE {result['baseline_rmse']:.4f} → Best RMSE {result['best_rmse']:.4f} "
          f"({result['best_rmse'] - result['baseline_rmse']:+.4f}, round {result['best_round']})")
    print(f"评测 spec 数: {result['n_evaluations']}  记忆文件: {memory.path}")

    _write_outputs(args, result, profile_text, outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
