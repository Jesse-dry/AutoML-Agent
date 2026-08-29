# ECL 跨用户迁移评测 CLI（统一评测协议）
# ---------------------------------------------------------------
# 用法示例：
#   python experiments/run_ecl_replay.py --model lightgbm
#   python experiments/run_ecl_replay.py --model persistence
#   python experiments/run_ecl_replay.py --model lightgbm --n-train 30 --seed 42
#
# 输出：逐 test 用户 RMSE 表 + 跨用户 mean/median/std + 相对指标；
#       --outdir 指定产物目录（manifest/summary/per_user/predictions）。
# 返回码：0 成功；1 运行失败。
# ---------------------------------------------------------------
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import ecl_protocol as ep  # noqa: E402
from evaluation.ecl_replay import replay_ecl  # noqa: E402
from models.replay_backends import make_backend  # noqa: E402


def _print_summary(outdir, payload, model_name, args) -> None:
    s = payload["summary"]
    print("=" * 70)
    print(f"[{model_name}] 统一协议评测 | {s['n_users']} test 用户 "
          f"| {s['n_train_users']} train 用户")
    print(f"  Mean RMSE   = {s['mean_rmse']:.4f}")
    print(f"  Median RMSE = {s['median_rmse']:.4f}")
    print(f"  Std RMSE    = {s['std_rmse']:.4f}")
    print(f"  Best/Worst  = {s['best_rmse']:.4f} ({s['best_user']}) / "
          f"{s['worst_rmse']:.4f} ({s['worst_user']})")
    r1, r2 = s["ratio_vs_persistence"], s["ratio_vs_snaive"]
    print(
        f"  ratio vs persistence(lag_1): mean={r1['mean']:.4f} "
        f"median={r1['median']:.4f} pct_better={r1['pct_better']:.1f}%"
    )
    print(
        f"  ratio vs snaive24(lag_24)  : mean={r2['mean']:.4f} "
        f"median={r2['median']:.4f} pct_better={r2['pct_better']:.1f}%"
    )

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # run_manifest.json
        manifest = {
            "model": model_name,
            "git_commit": ep.git_commit(),
            "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command_args": vars(args),
            "data": {
                "source": "ECL/electricity.txt (preprocessed)",
                "n_train_users": s["n_train_users"],
                "n_test_users": s["n_test_users"],
                "train_users_hash": ep.hash_list(payload["train_users"]),
                "test_users_hash": ep.hash_list(payload["test_users"]),
            },
            "summary": {k: v for k, v in s.items()},
        }
        (outdir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # metrics_summary.json
        (outdir / "metrics_summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        # per_user_metrics.csv
        ep.user_rows(payload["per_user_rmse"], payload["per_user_persist_rmse"],
                     payload["per_user_snaive_rmse"], payload["user_n_pred"]).to_csv(
            outdir / "per_user_metrics.csv", index=False, encoding="utf-8-sig")
        # predictions.csv
        if payload.get("predictions") is not None and len(payload["predictions"]) > 0:
            payload["predictions"].to_csv(
                outdir / "predictions.csv", index=False, encoding="utf-8")
        print(f"\n  产物 -> {outdir}")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ECL 跨用户迁移评测（统一协议，train 用户 → test 用户）")
    parser.add_argument("--model", default="lightgbm",
                        help="lightgbm | persistence")
    parser.add_argument("--n-train", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default=None,
                        help="产物目录（默认 experiments/output/ecl_replay）")
    args = parser.parse_args()

    try:
        backend = make_backend(args.model)
        outdir = Path(args.outdir) if args.outdir else (
            PROJECT_ROOT / "experiments" / "output" / "ecl_replay"
        )
        payload = replay_ecl(
            backend, n_train=args.n_train, seed=args.seed,
        )
        _print_summary(outdir, payload, backend.name, args)
        return 0

    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())