# GEFCom2014-S Task 1–15 × Zone 1–3 无泄漏滚动回放评测 CLI
# ---------------------------------------------------------------
# 用法示例：
#   python experiments/run_solar_replay.py --tasks 1:15 --model lightgbm
#   python experiments/run_solar_replay.py --tasks 1:3  --model persistence
#   python experiments/run_solar_replay.py --tasks 1:15 --model lightgbm --zones 1:3 --outdir experiments/output/solar_replay
#
# 输出：逐 Task（3 分区均值）指标表 + 逐 Task×Zone 明细；--outdir 指定审计输出。
# 返回码：0 成功；1 泄漏检查违规 / 运行失败。
# ---------------------------------------------------------------
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.solar_replay import LeakageError, replay_solar  # noqa: E402
from evaluation.forecast_protocol import get_protocol  # noqa: E402
from models.replay_backends import SeasonalNaiveBackend, make_backend  # noqa: E402


def parse_tasks(s: str):
    """解析 '1:15' / '1,3,5' / '1-15' / '1:15:2' / '3' 为 Task 列表。"""
    s = s.strip()
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            segs = part.split(":")
            if len(segs) == 2:
                start, end = int(segs[0]), int(segs[1])
                out.extend(range(start, end + 1))
            elif len(segs) == 3:
                start, end, step = (int(x) for x in segs)
                out.extend(range(start, end + 1, step))
            else:
                raise ValueError(f"无法解析任务范围: {part!r}")
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    tasks = sorted(set(out))
    if any(t < 1 or t > 15 for t in tasks):
        raise ValueError("task 必须在 1..15")
    return tasks


def parse_zones(s: str):
    """解析 '1:3' / '1,2,3' / '1-3' 为 Zone 列表（1..3）。"""
    s = s.strip()
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            segs = part.split(":")
            if len(segs) == 2:
                start, end = int(segs[0]), int(segs[1])
                out.extend(range(start, end + 1))
            else:
                raise ValueError(f"无法解析分区范围: {part!r}")
        elif "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    zones = sorted(set(out))
    if any(z < 1 or z > 3 for z in zones):
        raise ValueError("zone 必须在 1..3")
    return zones


def _print_table(summary_dict) -> None:
    table = summary_dict["task_table"].copy()
    table["RMSE"] = table["RMSE"].map(lambda v: f"{v:.4f}")
    table["MAE"] = table["MAE"].map(lambda v: f"{v:.4f}")
    table["MAPE"] = table["MAPE"].map(lambda v: f"{v:.4f}" if v is not None else "-")
    table["R2"] = table["R2"].map(lambda v: f"{v:.4f}" if v is not None else "-")
    print(table.to_string())
    s = summary_dict["summary"]
    print(
        f"\n  Mean RMSE={s['mean_rmse']:.4f}  Std RMSE={s['std_rmse']:.4f}  "
        f"Best RMSE={s['best_rmse']:.4f} (Task {s['best_task']})  "
        f"Worst RMSE={s['worst_rmse']:.4f} (Task {s['worst_task']})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GEFCom2014-S Task 1-15 x Zone 1-3 无泄漏滚动回放评测")
    parser.add_argument("--tasks", default="1:15",
                        help="任务范围，如 1:15 / 1,3,5 / 1-5 / 1:15:2")
    parser.add_argument("--zones", default="1:3",
                        help="分区范围，如 1:3 / 1,2,3 / 1-3")
    parser.add_argument("--model", default="lightgbm",
                        help="lightgbm | seasonal_naive_24 | seasonal_naive_168 | seasonal_naive_all | persistence")
    parser.add_argument("--protocol", default="online_h1",
                        help="online_h1 | recursive_month_ahead")
    parser.add_argument("--leak-check", default="sample", choices=["fast", "sample", "full"])
    parser.add_argument("--outdir", default=None,
                        help="审计输出目录（默认 experiments/output/solar_replay）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-hours", type=int, default=168)
    parser.add_argument("--skip-leak-check", action="store_true")
    args = parser.parse_args()

    try:
        tasks = parse_tasks(args.tasks)
        zones = parse_zones(args.zones)
        protocol = get_protocol(args.protocol)
        outdir = Path(args.outdir) if args.outdir else (
            PROJECT_ROOT / "experiments" / "output" / "solar_replay"
        )

        if args.model == "seasonal_naive_all":
            print(f"模型: seasonal_naive_all  协议: {protocol.name}  "
                  f"Tasks: {tasks}  Zones: {zones}")
            print("=" * 70)
            for k in (24, 168):
                backend = SeasonalNaiveBackend(k)
                print(f"\n--- {backend.name} ---")
                payload = replay_solar(tasks, backend, protocol,
                                       val_hours=args.val_hours,
                                       zones=zones,
                                       leak_check=args.leak_check,
                                       skip_leak_check=args.skip_leak_check,
                                       seed=args.seed, outdir=outdir)
                _print_table(payload)
            return 0

        backend = make_backend(args.model)
        print(f"模型: {backend.name}  协议: {protocol.name}  Tasks: {tasks}  "
              f"Zones: {zones}  seed={args.seed}  val_hours={args.val_hours}  "
              f"leak_check={args.leak_check}")
        print("=" * 70)
        payload = replay_solar(tasks, backend, protocol,
                               val_hours=args.val_hours,
                               zones=zones,
                               leak_check=args.leak_check,
                               skip_leak_check=args.skip_leak_check,
                               seed=args.seed, outdir=outdir)
        _print_table(payload)
        return 0

    except (LeakageError, ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
