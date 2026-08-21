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

from agent.evolution_runner import EvolutionRunner  # noqa: E402
from agent.feature_agent import QwenClient  # noqa: E402
from agent.scripted_llm import ScriptedLLM  # noqa: E402
from evaluation.error_profiler import format_profile_for_llm  # noqa: E402
from evaluation.forecast_protocol import get_protocol  # noqa: E402
from memory.memory_manager import MemoryManager  # noqa: E402
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


def _wind_demo_script(n_candidates: int = 3):
    """--dry-run 的 Wind 确定性演示脚本（source 用 TARGETVAR / 气象外生列）。"""
    import json as _json

    rounds = [
        [
            {"candidate_id": 1, "hypothesis": "补中程目标滞后 lag_48，加强出力惯性",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "TARGETVAR", "k": 48}}]},
            {"candidate_id": 2, "hypothesis": "补 48h 风速滚动标准差，刻画风资源波动",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "ws100", "window": 48, "stat": "std"}}]},
            {"candidate_id": 3, "hypothesis": "补 72h 风速滞后，捕捉天气系统演变",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "ws100", "k": 72}}]},
        ],
        [
            {"candidate_id": 1, "hypothesis": "补目标 lag_72 覆盖更长惯性",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "TARGETVAR", "k": 72}}]},
            {"candidate_id": 2, "hypothesis": "移除低信息量的 ws10_lag_24 试探",
             "actions": [{"type": "remove_feature", "feature": "ws10_lag_24"}]},
            {"candidate_id": 3, "hypothesis": "加当前小时风速预报 ws100@t（决策时点可得，试探 train/serve 偏移收益）",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "current", "source": "ws100"}}]},
        ],
        [
            {"candidate_id": 1, "hypothesis": "补目标 lag_120 试探更长惯性",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "TARGETVAR", "k": 120}}]},
            {"candidate_id": 2, "hypothesis": "补 168h 目标滚动标准差",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "TARGETVAR", "window": 168, "stat": "std"}}]},
            {"candidate_id": 3, "hypothesis": "保持当前特征集收敛探测",
             "actions": [{"type": "keep"}]},
        ],
    ]

    def _script(round_no: int) -> str:
        idx = min(round_no - 1, len(rounds) - 1)
        payload = {
            "round": round_no,
            "analysis": "[DRY RUN] Wind 演示脚本：生成多个不同方向候选。",
            "candidates": rounds[idx][:n_candidates],
        }
        return _json.dumps(payload, ensure_ascii=False)

    return _script


def _demo_script(n_candidates: int = 3, energy: str = "load"):
    """--dry-run 的确定性演示脚本：每轮 n_candidates 个不同方向的候选。"""
    import json as _json

    if energy == "wind":
        return _wind_demo_script(n_candidates)

    rounds = [
        [
            {"candidate_id": 1, "hypothesis": "补中程日周期滞后 lag_48，加强日周期信号",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "LOAD", "k": 48}}]},
            {"candidate_id": 2, "hypothesis": "补 48h 滚动标准差，刻画日间波动",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "LOAD", "window": 48, "stat": "std"}}]},
            {"candidate_id": 3, "hypothesis": "构造日周期-周周期交互（lag24 - lag168），强化跨周期差分",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "cross", "col1": "lag_24", "col2": "lag_168", "operation": "subtract"}}]},
        ],
        [
            {"candidate_id": 1, "hypothesis": "补 lag_72 覆盖更长滞后",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "LOAD", "k": 72}}]},
            {"candidate_id": 2, "hypothesis": "移除低信息量的 rolling_std_24 试探",
             "actions": [{"type": "remove_feature", "feature": "rolling_std_24"}]},
            {"candidate_id": 3, "hypothesis": "补 48h 滚动均值",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "LOAD", "window": 48, "stat": "mean"}}]},
        ],
        [
            {"candidate_id": 1, "hypothesis": "补 lag_120 试探更长日周期分量",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "LOAD", "k": 120}}]},
            {"candidate_id": 2, "hypothesis": "补 168h 滚动标准差",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "LOAD", "window": 168, "stat": "std"}}]},
            {"candidate_id": 3, "hypothesis": "保持当前特征集收敛探测",
             "actions": [{"type": "keep"}]},
        ],
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
    parser.add_argument("--energy", default="load", choices=["load", "wind"],
                        help="能源赛道：load（负荷）| wind（风电）")
    parser.add_argument("--zone", type=int, default=1, help="Wind 分区 1..10（energy=wind 时生效）")
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
    args = parser.parse_args()

    if not (1 <= args.task <= 15):
        print(f"[ERROR] task 必须在 1..15", file=sys.stderr)
        return 1
    if args.energy == "wind" and not (1 <= args.zone <= 10):
        print(f"[ERROR] zone 必须在 1..10", file=sys.stderr)
        return 1
    if not (1 <= args.n_candidates <= 3):
        print(f"[ERROR] n_candidates 必须在 1..3", file=sys.stderr)
        return 1

    if args.energy == "wind":
        outdir_suffix = f"evolution_wind_task{args.task}_z{args.zone}"
    else:
        outdir_suffix = f"evolution_task{args.task}"
    outdir = Path(args.outdir) if args.outdir else (
        PROJECT_ROOT / "experiments" / "output" / outdir_suffix
    )
    memory = MemoryManager(Path(args.memory_file)) if args.memory_file else MemoryManager()

    # LLM 客户端
    if args.dry_run:
        llm_client = ScriptedLLM(_demo_script(args.n_candidates, args.energy))
        print(f"模式: DRY RUN（ScriptedLLM）  Task: {args.task}  energy={args.energy}  协议: {args.protocol}")
    else:
        try:
            llm_client = QwenClient()
            print(f"模式: API（{llm_client.model}）  Task: {args.task}  协议: {args.protocol}")
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
        energy=args.energy,
        zone=args.zone,
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
