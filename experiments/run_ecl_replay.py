# ECL 跨用户迁移评测 CLI
# ---------------------------------------------------------------
# 用法示例：
#   python experiments/run_ecl_replay.py --model lightgbm
#   python experiments/run_ecl_replay.py --model persistence
#   python experiments/run_ecl_replay.py --model lightgbm --n-train 30 --seed 42   # 快速冒烟
#
# 输出：逐 test 用户 RMSE 表 + 跨用户 mean/std/best/worst RMSE。
# 返回码：0 成功；1 运行失败。
# ---------------------------------------------------------------
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.ecl_replay import replay_ecl  # noqa: E402
from models.replay_backends import make_backend  # noqa: E402


def _print_table(per_user_rmse, summary) -> None:
    import pandas as pd

    rows = [{"user": u, "rmse": round(v, 3)} for u, v in sorted(per_user_rmse.items())]
    df = pd.DataFrame(rows).set_index("user")
    print(df.to_string())
    s = summary
    print(
        f"\n  Mean RMSE={s['mean_rmse']:.4f}  Std RMSE={s['std_rmse']:.4f}  "
        f"Best={s['best_rmse']:.4f} ({s['best_user']})  "
        f"Worst={s['worst_rmse']:.4f} ({s['worst_user']})"
    )
    if "ratio_vs_naive" in s:
        r = s["ratio_vs_naive"]
        r2 = s["ratio_vs_snaive"]
        print(
            f"\n  相对指标（模型误差 / 朴素基线误差，<1 表示模型更优）:"
        )
        print(
            f"    vs persistence(lag_1): mean={r['mean']:.3f}  median={r['median']:.3f}  "
            f"优于朴素用户占比={r['pct_better']:.1f}%"
        )
        print(
            f"    vs snaive24(lag_24)  : mean={r2['mean']:.3f}  median={r2['median']:.3f}  "
            f"优于snaive用户占比={r2['pct_better']:.1f}%"
        )
    print(
        f"  n_train_users={s['n_train_users']}  n_test_users={s['n_test_users']}  "
        f"n_train_rows={s['n_train_rows']}  n_val_rows={s['n_val_rows']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ECL 跨用户迁移评测（train 用户 → test 用户）")
    parser.add_argument("--model", default="lightgbm",
                        help="lightgbm | persistence")
    parser.add_argument("--n-train", type=int, default=260,
                        help="训练用户数（默认 260，测试用户 = 321 - n_train）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机划分用户种子（可复现）")
    parser.add_argument("--val-hours", type=int, default=720,
                        help="早停验证段小时数（默认 720 = 30 天）")
    args = parser.parse_args()

    try:
        backend = make_backend(args.model)
        print(f"模型: {backend.name}  n_train={args.n_train}  seed={args.seed}  "
              f"val_hours={args.val_hours}")
        print("=" * 70)

        payload = replay_ecl(
            backend,
            n_train=args.n_train,
            seed=args.seed,
            val_hours=args.val_hours,
        )
        _print_table(payload["per_user_rmse"], payload["summary"])
        return 0

    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
