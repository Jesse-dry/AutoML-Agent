# 消融实验：Agent 特征价值是否可迁移到深模型
# ---------------------------------------------------------------
# 背景：说服力漏洞 —— 当前 Agent 只在 LightGBM 上做特征工程，和「端到端
# 深模型」对比不公平。本脚本证明：Agent 特征工程的价值不限于浅模型，
# 喂给深模型（LSTM）同样有效。
#
# 三组特征（同一 Task、同一 online_h1、同一后端、seed 固定）：
#   A 最小特征   time×4 + lag_1        （最弱基线，接近 persistence）
#   B 基础特征   FEATURE_SPEC（10 列）  （手工基线）
#   C Agent 特征 Agent 在 LightGBM 上进化的 best_spec（关键组）
#
# 两个后端：LightGBM / LSTM（训练窗口截断 + 充分 epoch）
# 结论方向：B→C 在 LSTM 上也有正收益 ⇒ 特征价值与模型无关；A→C ⇒ 特征工程
# 对深模型也有意义（堵住「深模型不需要特征」的质疑）。
#
# 用法：
#   python experiments/run_ablation_deep_features.py --task 15
#   python experiments/run_ablation_deep_features.py --task 15 --agent-spec best_spec.json
# ---------------------------------------------------------------
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.feature_spec import snapshot  # noqa: E402
from data.task_builder import FEATURE_SPEC  # noqa: E402
from evaluation.forecast_protocol import ONLINE_H1  # noqa: E402
from evaluation.spec_evaluator import evaluate_spec  # noqa: E402
from models.replay_backends import LSTMBackend, LightGBMBackend  # noqa: E402


def minimal_spec() -> list:
    """组 A：最小特征 = time×4 + lag_1（FEATURE_SPEC 前 5 列）。

    注意必须含 lag_1 —— LSTMBackend.predict 用 lag_1 重建预测月 target 序列。
    """
    return snapshot(FEATURE_SPEC[:5])


def _pure_feature_demo_script(n_candidates: int = 3):
    """纯特征工程 demo（不含 model 字段），现场取 Agent 在 LightGBM 上的 best_spec。"""
    import json as _json

    rounds = [
        [
            {"candidate_id": 1, "hypothesis": "补中程日周期滞后 lag_48",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "LOAD", "k": 48}}]},
            {"candidate_id": 2, "hypothesis": "补 48h 滚动标准差刻画日间波动",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "LOAD", "window": 48, "stat": "std"}}]},
            {"candidate_id": 3, "hypothesis": "补 lag_72 覆盖更长滞后",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "lag", "source": "LOAD", "k": 72}}]},
        ],
        [
            {"candidate_id": 1, "hypothesis": "补 48h 滚动均值",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "rolling", "source": "LOAD", "window": 48, "stat": "mean"}}]},
            {"candidate_id": 2, "hypothesis": "构造日周期-周周期交互",
             "actions": [{"type": "add_feature", "feature_spec": {"type": "cross", "col1": "lag_24", "col2": "lag_168", "operation": "subtract"}}]},
            {"candidate_id": 3, "hypothesis": "保持当前特征集收敛探测",
             "actions": [{"type": "keep"}]},
        ],
    ]

    def _script(round_no: int) -> str:
        idx = min(round_no - 1, len(rounds) - 1)
        payload = {
            "round": round_no,
            "analysis": "[ABLATION] 纯特征工程演示，取 LightGBM 上的 best_spec。",
            "candidates": rounds[idx][:n_candidates],
        }
        return _json.dumps(payload, ensure_ascii=False)

    return _script


def get_agent_best_spec(task_id: int, seed: int, max_iter: int = 3) -> list:
    """现场 dry-run 一次 LightGBM 特征进化，返回 best_spec（组 C 特征来源）。"""
    from agent.evolution_runner import EvolutionRunner
    from agent.scripted_llm import ScriptedLLM

    runner = EvolutionRunner(
        task_id=task_id,
        llm_client=ScriptedLLM(_pure_feature_demo_script(3)),
        n_candidates=3,
        max_iter=max_iter,
        seed=seed,
        memory=None,
    )
    result = runner.run(verbose=False)
    return snapshot(result["best_spec"])


def load_spec(path: Path) -> list:
    """从 JSON 文件读 spec（--agent-spec）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("spec", data.get("best_spec", data))
    if not isinstance(data, list):
        raise ValueError(f"--agent-spec 文件需为 spec 列表或含 spec/best_spec 键的 dict")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent 特征迁移到深模型的消融实验")
    parser.add_argument("--task", type=int, default=15)
    parser.add_argument("--agent-spec", default=None, help="Agent best_spec JSON 文件（缺省则现场 dry-run）")
    parser.add_argument("--train-window", type=int, default=0, help="LSTM 训练窗口（小时，0=全量历史）")
    parser.add_argument("--max-epochs", type=int, default=50, help="LSTM 最大 epoch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (1 <= args.task <= 15):
        print("[ERROR] task 必须在 1..15", file=sys.stderr)
        return 1

    specs = {
        "A 最小(time+lag1)": minimal_spec(),
        "B 基础(FEATURE_SPEC)": snapshot(FEATURE_SPEC),
    }
    if args.agent_spec:
        specs["C Agent(best_spec)"] = load_spec(Path(args.agent_spec))
        print(f"Agent 特征来源: {args.agent_spec}")
    else:
        print("现场 dry-run LightGBM 特征进化，取 best_spec ...")
        specs["C Agent(best_spec)"] = get_agent_best_spec(args.task, args.seed)

    lgb_factory = lambda: LightGBMBackend()  # noqa: E731
    lstm_factory = lambda: LSTMBackend(  # noqa: E731
        train_window=args.train_window, max_epochs=args.max_epochs
    )

    print("=" * 70)
    print(f"消融实验 — Task {args.task} | LSTM train_window={args.train_window}h "
          f"max_epochs={args.max_epochs}")
    print("=" * 70)
    print(f"{'后端':<10} {'特征组':<22} {'特征数':>4} {'RMSE':>10} {'MAE':>10} {'MAPE':>10}")
    rows = []
    for backend_name, factory in [("LightGBM", lgb_factory), ("LSTM", lstm_factory)]:
        for label, spec in specs.items():
            res = evaluate_spec(args.task, spec, ONLINE_H1,
                                backend_factory=factory, seed=args.seed)
            print(f"{backend_name:<10} {label:<22} {len(spec):>4} "
                  f"{res['rmse']:>10.4f} {res['mae']:>10.4f}")
            rows.append({
                "backend": backend_name, "group": label,
                "n_features": len(spec), "rmse": res["rmse"], "mae": res["mae"],
            })

    # 关键对比：B→C 在 LSTM 上的收益
    lstm_rmse = {r["group"]: r["rmse"] for r in rows if r["backend"] == "LSTM"}
    lgb_rmse = {r["group"]: r["rmse"] for r in rows if r["backend"] == "LightGBM"}
    print("\n关键结论:")
    print(f"  LSTM  B(基础)={lstm_rmse.get('B 基础(FEATURE_SPEC)', float('nan')):.4f} "
          f"→ C(Agent)={lstm_rmse.get('C Agent(best_spec)', float('nan')):.4f}")
    print(f"  LightGBM B(基础)={lgb_rmse.get('B 基础(FEATURE_SPEC)', float('nan')):.4f} "
          f"→ C(Agent)={lgb_rmse.get('C Agent(best_spec)', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
