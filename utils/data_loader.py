"""
滑动窗口 DataLoader
==================
为 LSTM / Transformer 等时序深度学习模型构造训练样本。

核心设计：
  1. 用列名定位特征和目标（不用整数索引，避免列顺序变化导致 bug）
  2. 工厂函数 create_dataloaders 一站式完成：选列 → 归一化 → 切窗口 → DataLoader
  3. 训练集 shuffle=True，验证/测试集 shuffle=False（保证可画出连续预测曲线）
  4. 返回 scaler，预测后 inverse_transform 恢复原始量纲再算指标

用法：
    from utils.data_loader import create_dataloaders

    loaders = create_dataloaders(
        train_df, val_df, test_df,
        feature_cols=['hour', 'temp', ...],
        target_col='LOAD',
        seq_len=24, pred_len=1, batch_size=32,
    )
    # loaders['train_loader'], loaders['target_scaler'], ...
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """
    滑动窗口时序数据集。

    对 (总样本数, 特征数) 的数组，按 seq_len 切出输入序列，
    按 pred_len 切出目标序列。target 固定在第 0 列。
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int = 1,
    ):
        """
        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_features)
            已经归一化的数组，第 0 列必须是目标变量
        seq_len : int
            历史窗口长度（用过去多少步预测）
        pred_len : int
            预测步长（默认 1 = 单步预测）
        """
        if len(data) < seq_len + pred_len:
            raise ValueError(
                f"数据长度 ({len(data)}) 不足，需要至少 "
                f"seq_len + pred_len = {seq_len + pred_len} 行"
            )

        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # X: 过去 seq_len 步的 [全部特征]
        x = self.data[index : index + self.seq_len, :]
        # y: 未来 pred_len 步的 [目标变量（第 0 列）]
        y_start = index + self.seq_len
        y = self.data[y_start : y_start + self.pred_len, 0]
        return x, y


# ============================================================
# 工厂函数
# ============================================================


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "LOAD",
    seq_len: int = 24,
    pred_len: int = 1,
    batch_size: int = 32,
    scaler_type: str = "standard",
    num_workers: int = 0,
) -> Dict:
    """
    一站式构造 train / val / test 的 DataLoader。

    内部自动完成：
      1. 按列名提取特征 + 目标
      2. StandardScaler 归一化（仅在训练集上 fit）
      3. 把 target 放到第 0 列，拼接成 (n, 1+len(feature_cols)) 数组
      4. 构造 TimeSeriesDataset → DataLoader

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        预处理后的时序 DataFrame（以 datetime 为索引）
    feature_cols : list of str
        用作模型输入的特征列名
    target_col : str
        目标列名（默认 "LOAD"）
    seq_len : int
        历史窗口长度
    pred_len : int
        预测步长
    batch_size : int
        批次大小
    scaler_type : str
        "standard" (Z-score) 或 "minmax"
    num_workers : int
        DataLoader 的并行 worker 数（Windows 下建议 0）

    Returns
    -------
    dict:
        {
            "train_loader": DataLoader,   # shuffle=True
            "val_loader":   DataLoader,   # shuffle=False
            "test_loader":  DataLoader,   # shuffle=False
            "feature_scaler": StandardScaler,  # 拟合在 train 特征上
            "target_scaler":  StandardScaler,  # 拟合在 train 目标上
            "feature_cols":   list,
            "target_col":     str,
            "seq_len":        int,
            "pred_len":       int,
        }
    """
    # ---- 校验 ----
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise KeyError(f"特征列在 DataFrame 中不存在: {missing}")
    if target_col not in train_df.columns:
        raise KeyError(f"目标列 '{target_col}' 在 DataFrame 中不存在")

    # ---- 选列 ----
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    y_train = train_df[target_col].values.astype(np.float32).reshape(-1, 1)
    y_val = val_df[target_col].values.astype(np.float32).reshape(-1, 1)
    y_test = test_df[target_col].values.astype(np.float32).reshape(-1, 1)

    # ---- 归一化 ----
    if scaler_type == "standard":
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
    else:
        # MinMax 可后续扩展
        from sklearn.preprocessing import MinMaxScaler
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

    # 只在训练集上 fit！
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    # ---- 拼接：target 放第 0 列 ----
    # 最终数组 shape = (n_samples, 1 + len(feature_cols))
    train_arr = np.concatenate([y_train_scaled, X_train_scaled], axis=1)
    val_arr = np.concatenate([y_val_scaled, X_val_scaled], axis=1)
    test_arr = np.concatenate([y_test_scaled, X_test_scaled], axis=1)

    # ---- 构造 Dataset ----
    train_dataset = TimeSeriesDataset(train_arr, seq_len=seq_len, pred_len=pred_len)
    val_dataset = TimeSeriesDataset(val_arr, seq_len=seq_len, pred_len=pred_len)
    test_dataset = TimeSeriesDataset(test_arr, seq_len=seq_len, pred_len=pred_len)

    # ---- 构造 DataLoader ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,            # 训练集打乱
        drop_last=True,           # 丢弃最后不完整的 batch
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,            # 验证集不打乱，保持时序
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,            # 测试集不打乱，保证可画连续曲线
        drop_last=False,
        num_workers=num_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "seq_len": seq_len,
        "pred_len": pred_len,
    }


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.preprocessing import preprocess_pipeline

    # 加载 Task 15 数据
    data_dir = str(Path(__file__).resolve().parent.parent / "GEFCom2014-L_V2" / "Load")
    result = preprocess_pipeline(data_dir, task_id=15, dropna_features=True)

    # 构造 DataLoader
    loaders = create_dataloaders(
        result["train"],
        result["val"],
        result["test"],
        feature_cols=result["feature_cols"],
        target_col=result["target_col"],
        seq_len=24,
        pred_len=1,
        batch_size=32,
    )

    print("=== DataLoader 测试 ===")
    for name in ["train_loader", "val_loader", "test_loader"]:
        loader = loaders[name]
        n_batches = len(loader)
        x, y = next(iter(loader))
        print(f"  {name}: {n_batches} batches, X shape={x.shape}, y shape={y.shape}")

    print(f"  feature_scaler mean[:3] = {loaders['feature_scaler'].mean_[:3]}")
    print(f"  target_scaler mean      = {loaders['target_scaler'].mean_[0]:.4f}")
    print(f"  target_scaler scale     = {loaders['target_scaler'].scale_[0]:.4f}")
    print("  shuffle: train=True, val=False, test=False  [OK]")
