"""
LightGBM 基线模型 — GEFCom2014 电力负荷预测
============================================
传统机器学习基线，用于：
  1. 验证数据 pipeline 正确性
  2. 建立"简单模型能做到多好"的参考线
  3. 输出特征重要性，供后续 LLM 特征 Agent 使用

用法：
  python models/baseline/lgb_gefcom2014.py              # 默认 Task 15
  python models/baseline/lgb_gefcom2014.py --task 1     # 指定 Task
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# 将项目根目录加入 path，确保能 import data.preprocessing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import preprocess_pipeline
from utils.metrics import compute_all_metrics

# ============================================================
# 工具函数
# ============================================================


def setup_logging(log_dir: Path) -> tuple:
    """配置日志：控制台 + 文件双写。返回 (logger, log_filepath)。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"lgb_baseline_{timestamp}.log"

    logger = logging.getLogger("LGB_Baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_path


# ============================================================
# LightGBM 训练回调
# ============================================================


class MetricsHistory:
    """收集训练过程中每个 epoch 的指标，训练结束后可导出。"""

    def __init__(self):
        self.records = []

    def callback(self, env):
        """LightGBM 回调接口。"""
        iteration = env.iteration
        evals = env.evaluation_result_list  # [(name, metric, value, higher_is_better)]
        record = {"iteration": iteration}
        for name, metric, value, _ in evals:
            record[f"{name}_{metric}"] = value
        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# ============================================================
# 主流程
# ============================================================


def run_lgb_baseline(
    data_dir: str = None,
    task_id: int = 15,
    output_dir: str = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    运行 LightGBM 基线训练 + 评估，产出模型、指标、特征重要性。

    Parameters
    ----------
    data_dir : str
        GEFCom2014-L_V2/Load 目录路径。默认从项目根目录推导。
    task_id : int
        Task 编号 1~15
    output_dir : str
        输出目录。默认 models/baseline/output/
    num_boost_round : int
        最大迭代轮数
    early_stopping_rounds : int
        早停轮数
    learning_rate : float
        学习率
    seed : int
        随机种子

    Returns
    -------
    dict: 包含所有指标、路径的汇总字典
    """
    # ---- 路径准备 ----
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "GEFCom2014-L_V2" / "Load")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 日志 ----
    logger, log_path = setup_logging(output_dir)
    logger.info(f"=== LightGBM Baseline | Task {task_id} | seed={seed} ===")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")

    # ---- 1. 加载预处理好的数据 ----
    logger.info("Step 1/5: 加载并预处理数据...")
    try:
        result = preprocess_pipeline(
            data_dir=data_dir,
            task_id=task_id,
            fill_load="interpolate",
            fill_weather="interpolate",
            split_method="sequential",
            dropna_features=True,
        )
    except FileNotFoundError as e:
        logger.error(f"数据文件不存在: {e}")
        raise

    train_df = result["train"]
    val_df = result["val"]
    test_df = result["test"]
    feature_cols = result["feature_cols"]
    target_col = result["target_col"]

    logger.info(
        f"  数据加载完成: Train={train_df.shape}, "
        f"Val={val_df.shape}, Test={test_df.shape}"
    )
    logger.info(f"  特征数: {len(feature_cols)}, 目标列: {target_col}")
    logger.info(
        f"  Train 时间范围: {train_df.index.min()} ~ {train_df.index.max()}"
    )
    logger.info(
        f"  Test  时间范围: {test_df.index.min()} ~ {test_df.index.max()}"
    )

    # ---- 2. 构造 LightGBM Dataset ----
    logger.info("Step 2/5: 构造 LightGBM Dataset...")
    train_data = lgb.Dataset(
        train_df[feature_cols], label=train_df[target_col],
    )
    val_data = lgb.Dataset(
        val_df[feature_cols], label=val_df[target_col], reference=train_data,
    )

    # ---- 3. 训练参数 ----
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": seed,
        "num_threads": -1,
    }
    logger.info(f"  Hyperparams: lr={learning_rate}, leaves=31, seed={seed}")

    # ---- 4. 训练 + 日志回调 ----
    logger.info(
        f"Step 3/5: 开始训练 (max_rounds={num_boost_round}, "
        f"early_stop={early_stopping_rounds})..."
    )

    metrics_history = MetricsHistory()
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
        lgb.log_evaluation(period=100),
        metrics_history.callback,
    ]

    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    best_iteration = gbm.best_iteration
    logger.info(f"  训练完成, best_iteration={best_iteration}")

    # ---- 5. 验证集 & 测试集评估（显式预测 + 结构化指标） ----
    logger.info("Step 4/5: 评估模型...")

    # 验证集预测
    val_preds = gbm.predict(val_df[feature_cols])
    val_metrics = compute_all_metrics(
        val_df[target_col].values, val_preds, prefix="val_"
    )

    # 测试集预测
    test_preds = gbm.predict(test_df[feature_cols])
    test_metrics = compute_all_metrics(
        test_df[target_col].values, test_preds, prefix="test_"
    )

    # 汇总所有指标
    all_metrics = {
        **val_metrics,
        **test_metrics,
        "best_iteration": best_iteration,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_time_range": f"{train_df.index.min()} ~ {train_df.index.max()}",
        "test_time_range": f"{test_df.index.min()} ~ {test_df.index.max()}",
    }

    logger.info(f"  验证集 → RMSE={val_metrics['val_RMSE']:.4f}, "
                f"MAE={val_metrics['val_MAE']:.4f}, "
                f"MAPE={val_metrics['val_MAPE']:.2f}%")
    logger.info(f"  测试集 → RMSE={test_metrics['test_RMSE']:.4f}, "
                f"MAE={test_metrics['test_MAE']:.4f}, "
                f"MAPE={test_metrics['test_MAPE']:.2f}%")

    # ---- 6. 保存产出 ----
    logger.info("Step 5/5: 保存模型、指标、特征重要性...")

    # 6a. 模型（LightGBM C++ 后端不支持中文路径，用 model_to_string 绕开）
    model_path = output_dir / f"lgb_baseline_task{task_id}.txt"
    model_str = gbm.model_to_string()
    with open(model_path, "w", encoding="utf-8") as f:
        f.write(model_str)
    logger.info(f"  [ok] 模型 → {model_path}")

    # 6b. 结构化指标 (JSON)
    metrics_path = output_dir / f"lgb_baseline_task{task_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  [ok] 指标 → {metrics_path}")

    # 6c. 特征重要性 (CSV) — 供 LLM 特征 Agent 使用
    importance_gain = gbm.feature_importance(importance_type="gain")
    importance_split = gbm.feature_importance(importance_type="split")

    feat_imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": importance_gain,
        "importance_split": importance_split,
    })
    # 按 gain 降序排列
    feat_imp_df = feat_imp_df.sort_values("importance_gain", ascending=False)
    # 加归一化列
    total_gain = feat_imp_df["importance_gain"].sum()
    feat_imp_df["importance_gain_norm"] = (
        feat_imp_df["importance_gain"] / total_gain * 100 if total_gain > 0 else 0
    )

    feat_imp_path = output_dir / f"lgb_baseline_task{task_id}_feature_importance.csv"
    feat_imp_df.to_csv(feat_imp_path, index=False, encoding="utf-8")
    logger.info(f"  [ok] 特征重要性 → {feat_imp_path}")

    # 打印 Top-10 特征重要性
    logger.info("\n--- 特征重要性 Top-10 (gain) ---")
    for _, row in feat_imp_df.head(10).iterrows():
        logger.info(
            f"  {row['feature']:30s}  "
            f"gain={row['importance_gain']:10.1f}  "
            f"({row['importance_gain_norm']:5.1f}%)"
        )

    # 6d. 训练过程指标历史 (CSV)
    hist_df = metrics_history.to_dataframe()
    hist_path = output_dir / f"lgb_baseline_task{task_id}_training_history.csv"
    hist_df.to_csv(hist_path, index=False, encoding="utf-8")
    logger.info(f"  [ok] 训练历史 → {hist_path}")

    # 6e. 测试集预测结果 (CSV) — 方便后续对比分析
    pred_df = test_df[[target_col]].copy()
    pred_df["prediction"] = test_preds
    pred_df["error"] = pred_df[target_col] - pred_df["prediction"]
    pred_df["abs_error"] = np.abs(pred_df["error"])
    pred_path = output_dir / f"lgb_baseline_task{task_id}_predictions.csv"
    pred_df.to_csv(pred_path, encoding="utf-8")
    logger.info(f"  [ok] 预测结果 → {pred_path}")

    # ---- 汇总 ----
    summary = {
        "task_id": task_id,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_importance_path": str(feat_imp_path),
        "training_history_path": str(hist_path),
        "predictions_path": str(pred_path),
        "log_path": str(log_path),
        "metrics": all_metrics,
    }

    logger.info(f"\n{'='*60}")
    logger.info("训练完成! 产出文件:")
    for key, val in summary.items():
        if key.endswith("_path") and key != "log_path":
            logger.info(f"  {key}: {val}")
    logger.info(f"{'='*60}")

    return summary


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LightGBM 基线 — GEFCom2014 电力负荷预测"
    )
    parser.add_argument(
        "--task", type=int, default=15,
        help="Task 编号 (1~15, 默认 15)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="GEFCom2014-L_V2/Load 目录路径",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录 (默认 models/baseline/output/)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.05,
        help="学习率 (默认 0.05)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (默认 42)",
    )
    args = parser.parse_args()

    summary = run_lgb_baseline(
        data_dir=args.data_dir,
        task_id=args.task,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # 最终打印关键指标（方便 Agent / 脚本解析）
    print("\n" + json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
