"""
PatchTST 基线模型 — GEFCom2014 电力负荷预测
============================================
基于 ICLR 2023 "A Time Series is Worth 64 Words" 的 PatchTST 架构，
适配到本项目的 GEFCom2014 数据集和统一评估体系。

核心思路：
  - 将 seq_len 长的时序切分为 overlapping patches
  - 每个 patch 作为 Transformer 的 token
  - 通道独立（Channel-Independent）：每个变量共享同一个 Encoder
  - RevIN 处理分布偏移
  - 预测后 inverse_transform 恢复原始量纲再算指标

用法：
  python models/PatchTST/patch_tst_baseline.py                   # 默认 Task 15
  python models/PatchTST/patch_tst_baseline.py --task 1          # 指定 Task
  python models/PatchTST/patch_tst_baseline.py --pred-len 24     # 多步预测
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
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import preprocess_pipeline
from utils.data_loader import create_dataloaders
from utils.metrics import compute_all_metrics

from PatchTST_backbone import PatchTST_backbone


# ============================================================
# PatchTST 模型封装
# ============================================================


class PatchTSTModel(nn.Module):
    """
    将 PatchTST 主干适配到本项目的数据格式。

    本项目的 DataLoader 产出:
      x: [bs, seq_len, n_features]  — 第 0 列是 target
      y: [bs, pred_len]

    PatchTST_backbone 期望:
      输入: [bs, nvars, seq_len]
      输出: [bs, nvars, pred_len]

    Wrapper 负责:
      1. 转置 x → [bs, nvars, seq_len]
      2. 调用 backbone 得到 [bs, nvars, pred_len]
      3. 取出第 0 通道 (target) → [bs, pred_len]
    """

    def __init__(self, backbone: PatchTST_backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [bs, seq_len, n_features]
            第 0 列为目标变量

        Returns
        -------
        [bs, pred_len]  — 只返回目标变量的预测值
        """
        # 转置到 PatchTST 格式: [bs, seq_len, nvars] → [bs, nvars, seq_len]
        z = x.permute(0, 2, 1)

        # PatchTST 前向
        out = self.backbone(z)  # [bs, nvars, pred_len]

        # 只取目标变量（第 0 通道）
        out = out[:, 0, :]  # [bs, pred_len]

        return out


# ============================================================
# 工具函数
# ============================================================


def setup_logging(log_dir: Path) -> tuple:
    """日志双写：控制台 + 文件。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"patchtst_baseline_{timestamp}.log"

    logger = logging.getLogger("PatchTST_Baseline")
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
    return torch.device("cpu")


def get_default_patch_config(seq_len: int) -> dict:
    """
    根据输入长度自动推荐 patch 配置。

    原则：patch_num 控制在 8~32 之间，
         让 Transformer 有足够的 token 但又不过多。
    """
    if seq_len <= 48:
        # 短序列：小 patch，密步长
        return {"patch_len": 4, "stride": 2}
    elif seq_len <= 168:
        # 中等序列
        return {"patch_len": 8, "stride": 4}
    else:
        # 长序列：大 patch
        return {"patch_len": 16, "stride": 8}


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


def run_patchtst_baseline(
    data_dir: str = None,
    task_id: int = 15,
    output_dir: str = None,
    # 数据参数
    seq_len: int = 48,
    pred_len: int = 1,
    batch_size: int = 32,
    # Patch 参数
    patch_len: int = None,
    stride: int = None,
    # 模型参数
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    head_dropout: float = 0.0,
    activation: str = "gelu",
    revin: bool = True,
    individual: bool = False,
    # 训练参数
    learning_rate: float = 0.001,
    max_epochs: int = 200,
    patience: int = 20,
    clip_grad: float = 1.0,
    seed: int = 42,
) -> Dict:
    """
    运行 PatchTST 基线训练 + 评估。

    Returns
    -------
    dict: 包含指标、路径的汇总字典（与 LSTM、LightGBM 基线格式一致）
    """
    # ---- 路径 & 日志 ----
    if data_dir is None:
        data_dir = str(_PROJECT_ROOT / "GEFCom2014-L_V2" / "Load")
    if output_dir is None:
        output_dir = str(Path(__file__).resolve().parent / "output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = setup_logging(output_dir)
    device = get_device()

    logger.info(f"=== PatchTST Baseline | Task {task_id} | device={device} | seed={seed} ===")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")

    # ---- 随机种子 ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    # 自动配置 patch 参数
    if patch_len is None or stride is None:
        auto_patch = get_default_patch_config(seq_len)
        patch_len = patch_len or auto_patch["patch_len"]
        stride = stride or auto_patch["stride"]

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

    # 计算 patch 数量
    patch_num = int((seq_len - patch_len) / stride + 1)
    logger.info(
        f"  Train batches: {len(train_loader)}, input_size={input_size}"
    )
    logger.info(
        f"  Patch: len={patch_len}, stride={stride}, "
        f"num={patch_num} (padding_patch='end' → {patch_num + 1})"
    )

    # ---- Step 3: 构建 PatchTST 模型 ----
    logger.info(
        f"Step 3/6: 构建 PatchTST "
        f"(d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, "
        f"d_ff={d_ff}, revin={revin})..."
    )

    # 输入通道数 = 特征数 + 目标变量（DataLoader 把 target 放在第 0 列）
    c_in = len(feature_cols) + 1

    backbone = PatchTST_backbone(
        c_in=c_in,
        context_window=seq_len,
        target_window=pred_len,
        patch_len=patch_len,
        stride=stride,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        attn_dropout=attn_dropout,
        head_dropout=head_dropout,
        activation=activation,
        revin=revin,
        individual=individual,
        padding_patch="end",  # 保证覆盖最后的数据点
        pe="zeros",
        learn_pe=True,
    )

    model = PatchTSTModel(backbone).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")

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
    history = []

    checkpoint_path = output_dir / f"patchtst_baseline_task{task_id}_best.pt"

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_grad
        )
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": current_lr,
        })

        if epoch % 10 == 0 or val_loss < best_val_loss:
            marker = " *" if val_loss < best_val_loss else ""
            logger.info(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"lr={current_lr:.2e}{marker}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"  早停触发! best_epoch={best_epoch}, "
                    f"best_val_loss={best_val_loss:.6f}"
                )
                break

    if epoch >= max_epochs:
        logger.info(
            f"  训练完成, best_epoch={best_epoch}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

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
        "trainable_params": trainable_params,
        "n_features": len(feature_cols),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_time_range": f"{train_df.index.min()} ~ {train_df.index.max()}",
        "test_time_range": f"{test_df.index.min()} ~ {test_df.index.max()}",
    }

    logger.info(
        f"  验证集 → RMSE={val_metrics['val_RMSE']:.4f}, "
        f"MAE={val_metrics['val_MAE']:.4f}, "
        f"MAPE={val_metrics['val_MAPE']:.2f}%"
    )
    logger.info(
        f"  测试集 → RMSE={test_metrics['test_RMSE']:.4f}, "
        f"MAE={test_metrics['test_MAE']:.4f}, "
        f"MAPE={test_metrics['test_MAPE']:.2f}%"
    )

    # ---- Step 6: 保存产出 ----
    logger.info("Step 6/6: 保存模型、指标、预测结果...")

    # 6a. 模型（已保存在 checkpoint_path）
    logger.info(f"  [ok] 模型 → {checkpoint_path}")

    # 6b. 结构化指标 (JSON)
    metrics_path = output_dir / f"patchtst_baseline_task{task_id}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  [ok] 指标 → {metrics_path}")

    # 6c. 训练历史 (CSV)
    hist_df = pd.DataFrame(history)
    hist_path = output_dir / f"patchtst_baseline_task{task_id}_training_history.csv"
    hist_df.to_csv(hist_path, index=False, encoding="utf-8")
    logger.info(f"  [ok] 训练历史 → {hist_path}")

    # 6d. 测试集预测结果 (CSV)
    pred_df = pd.DataFrame({
        "actual": test_targets,
        "prediction": test_preds,
        "error": test_targets - test_preds,
        "abs_error": np.abs(test_targets - test_preds),
    })
    pred_path = output_dir / f"patchtst_baseline_task{task_id}_predictions.csv"
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
        description="PatchTST 基线 — GEFCom2014 电力负荷预测"
    )
    parser.add_argument("--task", type=int, default=15,
                        help="Task 编号 (1~15, 默认 15)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="GEFCom2014-L_V2/Load 目录路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 models/PatchTST/output/)")
    # 数据
    parser.add_argument("--seq-len", type=int, default=48,
                        help="历史窗口长度 (默认 48，适合小时级数据)")
    parser.add_argument("--pred-len", type=int, default=1,
                        help="预测步长 (默认 1)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="批次大小")
    # Patch 参数
    parser.add_argument("--patch-len", type=int, default=None,
                        help="Patch 长度 (默认自动选择)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Patch 步长 (默认自动选择)")
    # 模型
    parser.add_argument("--d-model", type=int, default=128,
                        help="模型隐藏维度")
    parser.add_argument("--n-heads", type=int, default=8,
                        help="注意力头数 (默认 8，需整除 d_model)")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="Transformer encoder 层数")
    parser.add_argument("--d-ff", type=int, default=256,
                        help="前馈网络隐藏层维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="残差 dropout")
    parser.add_argument("--attn-dropout", type=float, default=0.0,
                        help="注意力 dropout")
    parser.add_argument("--head-dropout", type=float, default=0.0,
                        help="预测头 dropout")
    parser.add_argument("--no-revin", action="store_true",
                        help="禁用 RevIN")
    parser.add_argument("--individual", action="store_true",
                        help="每通道独立预测头")
    # 训练
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心轮数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    summary = run_patchtst_baseline(
        data_dir=args.data_dir,
        task_id=args.task,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        head_dropout=args.head_dropout,
        revin=not args.no_revin,
        individual=args.individual,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )

    print("\n" + json.dumps(summary["metrics"], indent=2, ensure_ascii=False))
