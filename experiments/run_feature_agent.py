"""
特征工程 Agent 端到端实验
========================
运行完整的 LLM 特征工程迭代闭环，产出：
  1. 迭代指标变化曲线（证明 LLM 生成的特征有效）
  2. 最优自动特征集（对比手动基线）
  3. 完整实验日志

用法：
  python experiments/run_feature_agent.py [--task 15] [--max-iter 5]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.feature_agent import (
    QwenClient,
    FeatureIterationRunner,
    build_context_from_data,
    build_messages,
    validate_llm_output,
    execute_features_from_llm,
    FeatureIterationHistory,
)
from agent.feature_engine import generate_all_features
from data.preprocessing import preprocess_pipeline


def main():
    parser = argparse.ArgumentParser(description="LLM 特征工程 Agent 实验")
    parser.add_argument("--task", type=int, default=15)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ---- 输出目录 ----
    output_dir = PROJECT_ROOT / "experiments" / f"feature_agent_task{args.task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_{timestamp}.log"

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 60)
    log(f"LLM 特征工程 Agent 实验")
    log(f"  时间: {timestamp}")
    log(f"  Task: {args.task}")
    log(f"  最大迭代: {args.max_iter}")
    log(f"  模式: {'DRY RUN' if args.dry_run else 'API (Qwen3.7 Max)'}")
    log("=" * 60)

    # ================================================================
    # Step 1: 加载数据
    # ================================================================
    log("\n[Step 1] 加载并预处理数据...")
    data_dir = str(PROJECT_ROOT / "GEFCom2014-L_V2" / "Load")

    result = preprocess_pipeline(
        data_dir=data_dir,
        task_id=args.task,
        fill_load="interpolate",
        fill_weather="interpolate",
        split_method="sequential",
        dropna_features=True,
    )

    train_df = result["train"]
    val_df = result["val"]
    test_df = result["test"]
    feature_cols = result["feature_cols"]
    target_col = result["target_col"]

    log(f"  Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    log(f"  初始特征 ({len(feature_cols)}): {feature_cols}")

    # ================================================================
    # Step 2: 初始化 LLM 客户端
    # ================================================================
    log("\n[Step 2] 初始化 LLM 客户端...")

    if args.dry_run:
        llm_client = QwenClient(dry_run=True)
        log("  模式: DRY RUN")
    else:
        try:
            llm_client = QwenClient()
            log(f"  模型: {llm_client.model}")
        except ValueError as e:
            log(f"  [FATAL] API Key 未配置: {e}")
            return 1

    # ================================================================
    # Step 3: 运行特征工程迭代
    # ================================================================
    log("\n[Step 3] 启动特征工程迭代闭环...")

    runner = FeatureIterationRunner(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col="datetime" if "datetime" in train_df.columns else train_df.index.name,
        llm_client=llm_client,
        max_iterations=args.max_iter,
        pred_horizon=1,
        dataset_name=f"GEFCom2014 Task {args.task}",
    )

    output = runner.run(verbose=True)

    # ================================================================
    # Step 4: 输出结果
    # ================================================================
    log("\n" + "=" * 60)
    log("[Step 4] 实验结果汇总")
    log("=" * 60)

    baseline = output["baseline_metrics"]
    best = output["best_metrics"]
    final = output["final_metrics"]

    log(f"\nBaseline  Val RMSE: {baseline.get('RMSE', '?'):.4f}")
    log(f"Best      Val RMSE: {best.get('RMSE', '?'):.4f}  (iteration {output['best_iteration']})")
    log(f"Final     Val RMSE: {final.get('RMSE', '?'):.4f}")

    if isinstance(baseline.get("RMSE"), float) and isinstance(best.get("RMSE"), float):
        improvement = baseline["RMSE"] - best["RMSE"]
        improvement_pct = improvement / baseline["RMSE"] * 100
        log(f"\nRMSE 改善: {baseline['RMSE']:.4f} -> {best['RMSE']:.4f} "
            f"({improvement:+.4f}, {improvement_pct:+.1f}%)")

    log(f"\n累计新增特征 ({output['total_features_added']}):")
    for f in output["all_added_features"]:
        log(f"  - {f}")

    log(f"\n最优特征集 ({len(output['best_features'])} 个):")
    for f in output["best_features"]:
        log(f"  - {f}")

    # ================================================================
    # Step 5: 保存产出文件
    # ================================================================
    log(f"\n[Step 5] 保存实验产出 → {output_dir}")

    # 5a. 完整结果 JSON
    result_json = {
        "timestamp": timestamp,
        "task": args.task,
        "max_iterations": args.max_iter,
        "baseline_metrics": {k: v for k, v in baseline.items() if isinstance(v, (int, float, str, type(None)))},
        "best_metrics": {k: v for k, v in best.items() if isinstance(v, (int, float, str, type(None)))},
        "final_metrics": {k: v for k, v in final.items() if isinstance(v, (int, float, str, type(None)))},
        "best_iteration": output["best_iteration"],
        "total_features_added": output["total_features_added"],
        "all_added_features": output["all_added_features"],
        "best_features": output["best_features"],
    }
    json_path = output_dir / f"result_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False, default=str)
    log(f"  [ok] 结果 JSON → {json_path}")

    # 5b. 迭代历史 CSV
    try:
        summary_df = output["summary"]
        csv_path = output_dir / f"iteration_history_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False, encoding="utf-8")
        log(f"  [ok] 迭代历史 → {csv_path}")
    except Exception:
        pass

    # 5c. 最优特征集文件
    features_path = output_dir / f"best_features_{timestamp}.txt"
    with open(features_path, "w", encoding="utf-8") as f:
        f.write(f"# 最优特征集 (Task {args.task}, iteration {output['best_iteration']})\n")
        f.write(f"# Val RMSE: {best.get('RMSE', '?')}\n")
        f.write(f"# 总特征数: {len(output['best_features'])}\n\n")
        for feat in output["best_features"]:
            f.write(feat + "\n")
    log(f"  [ok] 最优特征集 → {features_path}")

    # ================================================================
    # Step 6: 生成指标变化曲线
    # ================================================================
    log(f"\n[Step 6] 生成指标变化曲线...")
    try:
        _plot_metrics_curve(output, baseline, output_dir, timestamp)
        log(f"  [ok] 指标曲线 → {output_dir / f'metrics_curve_{timestamp}.png'}")
    except Exception as e:
        log(f"  [WARN] 图表生成失败: {e}")

    log(f"\n{'=' * 60}")
    log("实验完成!")
    log(f"所有产出: {output_dir}")
    log(f"{'=' * 60}")

    return 0


def _plot_metrics_curve(output, baseline, output_dir, timestamp):
    """绘制迭代过程中的指标变化曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = output["history"]
    if not history.records:
        return

    iterations = [0]  # baseline
    rmse_vals = [baseline.get("RMSE", float("nan"))]
    mae_vals = [baseline.get("MAE", float("nan"))]
    mape_vals = [baseline.get("MAPE", float("nan"))]
    labels = ["Baseline"]

    for r in history.records:
        m = r.get("val_metrics_after", {})
        iterations.append(r["iteration"])
        rmse_vals.append(m.get("RMSE", float("nan")))
        mae_vals.append(m.get("MAE", float("nan")))
        mape_vals.append(m.get("MAPE", float("nan")))
        labels.append(f"Iter {r['iteration']}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # RMSE
    ax = axes[0]
    ax.plot(iterations, rmse_vals, "o-", color="#1a73e8", linewidth=2, markersize=8)
    best_idx = rmse_vals.index(min([v for v in rmse_vals if not np.isnan(v)]))
    ax.plot(iterations[best_idx], rmse_vals[best_idx], "D", color="#e74c3c", markersize=12,
            label=f"Best: {rmse_vals[best_idx]:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE Change")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # MAE
    ax = axes[1]
    ax.plot(iterations, mae_vals, "s-", color="#34a853", linewidth=2, markersize=8)
    best_mae_idx = mae_vals.index(min([v for v in mae_vals if not np.isnan(v)]))
    ax.plot(iterations[best_mae_idx], mae_vals[best_mae_idx], "D", color="#e74c3c", markersize=12,
            label=f"Best: {mae_vals[best_mae_idx]:.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAE")
    ax.set_title("MAE Change")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # MAPE
    ax = axes[2]
    ax.plot(iterations, mape_vals, "^-", color="#fbbc04", linewidth=2, markersize=8)
    best_mape_idx = mape_vals.index(min([v for v in mape_vals if not np.isnan(v)]))
    ax.plot(iterations[best_mape_idx], mape_vals[best_mape_idx], "D", color="#e74c3c", markersize=12,
            label=f"Best: {mape_vals[best_mape_idx]:.2f}%")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE Change")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("LLM Feature Engineering Agent — Iteration Metrics", fontsize=14, fontweight="bold")
    plt.tight_layout()

    chart_path = output_dir / f"metrics_curve_{timestamp}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
