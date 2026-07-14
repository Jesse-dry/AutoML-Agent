"""
LSTM 基线模型 — GEFCom2014 电力负荷预测
========================================
深度学习基线，用于：
  1. 和 LightGBM 对比，验证深度学习在时序预测上的表现
  2. 建立"简单 LSTM 能做到多好"的参考线
  3. 输出结构化指标，供后续 Agent 对比

关键设计：
  - 数据必须经过 StandardScaler 归一化再送入 LSTM
  - 预测后 inverse_transform 恢复原始量纲再算 MAE/RMSE/MAPE
  - 训练集 shuffle=True，验证/测试集 shuffle=False（保证连续预测曲线）

用法：
  python models/LSTM/LSTM_baseline.py              # 默认 Task 15
  python models/LSTM/LSTM_baseline.py --task 1     # 指定 Task
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 将项目根目录加入 path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import preprocess_pipeline
from utils.data_loader import create_dataloaders
from utils.metrics import compute_all_metrics


# ============================================================
# LSTM 模型
# ============================================================


class LSTMRegressor(nn.Module):
    """纯净版 LSTM 回归器。

    batch_first=True，输入 (Batch, Seq_len, Features) → 输出 (Batch, Output_size)
    只取最后一个时间步的隐藏状态做预测（适用于单步/短步预测）。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # PyTorch LSTM 要求 num_layers=1 时 dropout 必须为 0
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，帮助 LSTM 更快收敛。"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 只取最后一个时间步 → (batch_size, hidden_size)
        last_out = lstm_out[:, -1, :]
        return self.linear(last_out)


# ============================================================
# 工具函数
# ============================================================


def setup_logging(log_dir: Path) -> tuple:
    """日志双写：控制台 + 文件。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"lstm_baseline_{timestamp}.log"

    logger = logging.getLogger("LSTM_Baseline")
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


def get_device() -> torch.device:
    """自动选择可用设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple Silicon) 对 LSTM 支持不完整，回退 CPU
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 训练 + 评估
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """评估：返回 (平均 loss, 所有预测值, 所有真实值)。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()
        n_batches += 1

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, preds, targets


# ============================================================
# 主流程
# ============================================================


def run_lstm_baseline(
    data_dir: str = None,
    task_id: int = 15,
    output_dir: str = None,
    # 数据参数
    seq_len: int = 24,
    pred_len: int = 1,
    batch_size: int = 32,
    # 模型参数
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    # 训练参数
    learning_rate: float = 0.001,
    max_epochs: int = 200,
    patience: int = 20,
    clip_grad: float = 1.0,
    seed: int = 42,
) -> Dict:
    """
    运行 LSTM 基线训练 + 评估。

    Returns
    -------
    dict: 包含指标、路径的汇总字典（与 LightGBM 基线格式一致）
    """
    # ---- 路径 & 日志 ----
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "GEFCom2014-L_V2" / "Load")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logging(output_dir)
    device = get_device()

    logger.info(f"=== LSTM Baseline | Task {task_id} | device={device} | seed={seed} ===")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")

    # ---- 随机种子 ----
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Step 1: 加载预处理数据 ----
    logger.info("Step 1/6: 加载预处理数据...")
    result = preprocess_pipeline(
        data_dir=data_dir,
        task_id=task_id,
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

    logger.info(
        f"  数据: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}"
    )
    logger.info(f"  特征数: {len(feature_cols)}, 目标: {target_col}")

    # ---- Step 2: 构造 DataLoader（含归一化） ----
    logger.info(
        f"Step 2/6: 构造滑动窗口 DataLoader "
        f"(seq_len={seq_len}, pred_len={pred_len}, batch={batch_size})..."
    )
    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        scaler_type="standard",
        num_workers=0,
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    target_scaler = loaders["target_scaler"]
    input_size = len(feature_cols) + 1  # +1 因为 target 在 DataLoader 的第 0 列
    logger.info(f"  Train batches: {len(train_loader)}, input_size={input_size}")

    # ---- Step 3: 构建模型 ----
    logger.info(
        f"Step 3/6: 构建 LSTM "
        f"(hidden={hidden_size}, layers={num_layers}, dropout={dropout})..."
    )
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=pred_len,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  参数量: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ---- Step 4: 训练（含早停） ----
    logger.info(
        f"Step 4/6: 训练 (max_epochs={max_epochs}, patience={patience}, "
        f"lr={learning_rate})..."
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []  # 每 epoch 的 loss 记录

    checkpoint_path = output_dir / f"lstm_baseline_task{task_id}_best.pt"

    for epoch in range(1, max_epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_grad
        )
        # 验证
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        # 学习率调度
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": current_lr,
        })

        # 每 10 轮或最佳轮打印
        if epoch % 10 == 0 or val_loss < best_val_loss:
            marker = " *" if val_loss < best_val_loss else ""
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"lr={current_lr:.2e}{marker}"
            )

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  早停触发! best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
                break

    if epoch >= max_epochs:
        logger.info(f"  训练完成, best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")

    # ---- 加载最佳模型 ----
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ---- Step 5: 评估（inverse_transform 后算指标） ----
    logger.info("Step 5/6: 评估 (inverse_transform → 真实量纲 → 指标)...")

    # 验证集
    _, val_preds_scaled, val_targets_scaled = evaluate(
        model, val_loader, criterion, device
    )
    val_preds = target_scaler.inverse_transform(val_preds_scaled).flatten()
    val_targets = target_scaler.inverse_transform(val_targets_scaled).flatten()
    val_metrics = compute_all_metrics(val_targets, val_preds, prefix="val_")

    # 测试集
    _, test_preds_scaled, test_targets_scaled = evaluate(
        model, test_loader, criterion, device
    )
    test_preds = target_scaler.inverse_transform(test_preds_scaled).flatten()
    test_targets = target_scaler.inverse_transform(test_targets_scaled).flatten()
    test_metrics = compute_all_metrics(test_targets, test_preds, prefix="test_")

    # 汇总
    all_metrics = {
        **val_metrics,
        **test_metrics,
        "best_epoch": best_epoch,
        "best_val_loss_scaled": round(best_val_loss, 6),
        "total_params": total_params,
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

    # ---- Step 6: 保存产出 ----
    logger.info("Step 6/6: 保存模型、指标、预测结果...")

    # 6a. 模型（已保存在 checkpoint_path）
    logger.info(f"  [ok] 模型 → {checkpoint_path}")

    # 6b. 结构化指标 (JSON)
    metrics_path = output_dir / f"lstm_baseline_task{task_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  [ok] 指标 → {metrics_path}")

    # 6c. 训练历史 (CSV)
    hist_df = pd.DataFrame(history)
    hist_path = output_dir / f"lstm_baseline_task{task_id}_training_history.csv"
    hist_df.to_csv(hist_path, index=False, encoding="utf-8")
    logger.info(f"  [ok] 训练历史 → {hist_path}")

    # 6d. 测试集预测结果 (CSV)
    pred_df = pd.DataFrame({
        "actual": test_targets,
        "prediction": test_preds,
        "error": test_targets - test_preds,
        "abs_error": np.abs(test_targets - test_preds),
    })
    pred_path = output_dir / f"lstm_baseline_task{task_id}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    logger.info(f"  [ok] 预测结果 → {pred_path}")

    # ---- 汇总 ----
    summary = {
        "task_id": task_id,
        "model_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
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
        description="LSTM 基线 — GEFCom2014 电力负荷预测"
    )
    parser.add_argument("--task", type=int, default=15)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    # 数据
    parser.add_argument("--seq-len", type=int, default=24, help="历史窗口长度")
    parser.add_argument("--pred-len", type=int, default=1, help="预测步长")
    parser.add_argument("--batch-size", type=int, default=32)
    # 模型
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    # 训练
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    summary = run_lstm_baseline(
        data_dir=args.data_dir,
        task_id=args.task,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )

    print("\n" + json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
