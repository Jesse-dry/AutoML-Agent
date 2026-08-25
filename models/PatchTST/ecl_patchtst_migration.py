# PatchTST 跨用户迁移（ECL）
# ---------------------------------------------------------------
# 用 PatchTST backbone（channel-independence + RevIN）做 ECL 跨用户迁移：
#   - 训练：随机 n_train 个用户的窗口样本混合训练（用户无关，共享权重），
#     RevIN 逐序列归一化 → 天然治"用户规模差 140 倍"。
#   - 评测：test 用户（训练时从未见过）预测低频段，逐用户 RMSE + 相对指标
#     （vs persistent-naive lag_1 / seasonal-naive lag_24）。
#
# 参考原版 PatchTST_baseline 的封装方式，但改用单通道（c_in=1）。
# 用法：
#   python models/PatchTST/ecl_patchtst_migration.py --n-train 260 --epochs 30
#   python models/PatchTST/ecl_patchtst_migration.py --n-train 20 --epochs 2 --sample-ratio 0.05  # 冒烟
# ---------------------------------------------------------------
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 PatchTST_backbone 可 import

from PatchTST_backbone import PatchTST_backbone  # noqa: E402

from data.ecl_loader import load_ecl_matrix, split_users  # noqa: E402
from utils.metrics import rmse  # noqa: E402

# 数据时间约定
# 训练段：2012-01-01 ~ TRAIN_END（两年半 ≈ 21900h）
# 评测段：TEST_START ~ 2014-12-31（半年，test 用户）
TRAIN_END = pd.Timestamp("2014-07-01")
TEST_START = pd.Timestamp("2014-07-01")
TEST_END = pd.Timestamp("2014-12-31 23:00:00")

ECL_TARGET = "load"


class ECLPatchTST(nn.Module):
    """单通道 PatchTST 封装。输入 [bs, seq_len] → 输出 [bs, pred_len]。”

    RevIN 在 backbone 内部（norm/denorm）。c_in=1，用户无关（每个样本独立归一化）。
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
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
    ):
        super().__init__()
        self.backbone = PatchTST_backbone(
            c_in=1,
            context_window=seq_len,
            target_window=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            head_dropout=head_dropout,
            activation=activation,
            revin=revin,
            individual=individual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [bs, seq_len] → [bs, pred_len]"""
        z = x.unsqueeze(1)  # [bs, 1, seq_len]
        out = self.backbone(z)  # [bs, 1, pred_len]
        return out[:, 0, :]  # [bs, pred_len]


class UserWindowDataset(Dataset):
    """单通道滑窗数据集：每个样本 = 一个用户的一段 seq_len 窗口 → 下一时刻。

    训练：max_len=None 且不设 eval_ts → 全量滑窗（限制在训练段内）。
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        users: List[str],
        seq_len: int,
        pred_len: int = 1,
        max_time: Optional[pd.Timestamp] = None,
        sample_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        rng = np.random.RandomState(seed)

        X_list, y_list = [], []
        for u in users:
            s = matrix[u].values  # float array
            n = len(s)
            if max_time is not None:
                n = matrix.index.searchsorted(max_time)  # 只用 < max_time 的历史
            # 每个窗口：X = s[i : i+seq_len], y = s[i+seq_len]
            # 有效起始 i 满足 i+seq_len+pred_len-1 < n, 即 i <= n-1-seq_len-pred_len+1
            n_samples = n - seq_len - pred_len + 1
            if n_samples <= 0:
                continue
            # 随机抽样子集
            if sample_ratio < 1.0:
                n_pick = max(1, int(n_samples * sample_ratio))
                idx = np.sort(rng.choice(n_samples, size=n_pick, replace=False))
            else:
                idx = np.arange(n_samples)
            for i in idx:
                X_list.append(s[i : i + seq_len].tolist())
                y_list.append(s[i + seq_len : i + seq_len + pred_len].tolist())
        self.X = torch.tensor(np.array(X_list, dtype=np.float32))
        self.y = torch.tensor(np.array(y_list, dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_test_matrix(
    matrix: pd.DataFrame,
    users: List[str],
    seq_len: int,
    pred_len: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """每个 test 用户 → (X_windows, y_true, y_naive, y_snaive)。

    窗口 = 该用户 [t-seq_len, t-1] 的真值（online 单步，无未来泄漏）；
    naive = y[t-1]（lag_1），snaive = y[t-24]（lag_24）；y_true = y[t]。
    """
    result = {}
    ts = matrix.index
    t_idx = {t: i for i, t in enumerate(ts)}
    for u in users:
        s = matrix[u].values
        # 评测段时间戳（每小时）
        times = ts[(ts >= start) & (ts <= end)]
        X_w, y_t, y_n, y_s = [], [], [], []
        for t in times:
            i = t_idx[t]
            if i - seq_len < 0:
                continue  # 窗口起点出界
            X_w.append(s[i - seq_len : i].tolist())
            y_t.append(s[i])
            y_n.append(s[i - 1])
            y_s.append(s[i - 24] if i - 24 >= 0 else np.nan)
        result[u] = (
            np.array(X_w, dtype=np.float32),
            np.array(y_t, dtype=np.float32),
            np.array(y_n, dtype=np.float32),
            np.array(y_s, dtype=np.float32),
        )
    return result


def _ratio_stats(model_rmse: np.ndarray, naive_rmse: np.ndarray) -> Dict[str, float]:
    """相对指标：model_rmse / naive_rmse（<1 = 模型更优）。"""
    mask = naive_rmse > 1e-6
    ratios = model_rmse[mask] / naive_rmse[mask]
    return {
        "mean": float(ratios.mean()) if len(ratios) else float("nan"),
        "median": float(np.median(ratios)) if len(ratios) else float("nan"),
        "pct_better": float((ratios < 1).mean() * 100) if len(ratios) else float("nan"),
    }


@torch.no_grad()
def predict_windows(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """批量预测窗口 → 预测值数组 [n_windows, pred_len]"""
    model.eval()
    outs = []
    Xt = torch.tensor(X)
    for i in range(0, len(Xt), batch_size):
        batch = Xt[i : i + batch_size]
        outs.append(model(batch).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description="PatchTST ECL 跨用户迁移")
    parser.add_argument("--n-train", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-ratio", type=float, default=0.1,
                        help="训练窗口抽样比例（控制时间/内存，全量=1.0）")
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    print(f"设备: {device}")

    # ---- 数据 ----
    print("加载 ECL 矩阵 + 随机划分用户...")
    matrix = load_ecl_matrix()
    train_cols, test_cols = split_users(
        n_users=matrix.shape[1], n_train=args.n_train, seed=args.seed
    )
    print(f"train 用户 {len(train_cols)} / test 用户 {len(test_cols)}")

    # ---- 训练集（train 用户，训练段 2.5 年）----
    t0 = time.time()
    print(f"构建训练数据集（抽样 {args.sample_ratio}）...")
    train_ds = UserWindowDataset(
        matrix, train_cols, args.seq_len, args.pred_len,
        max_time=TRAIN_END, sample_ratio=args.sample_ratio, seed=args.seed,
    )
    print(f"  训练样本数: {len(train_ds)}  （构建用时 {time.time()-t0:.1f}s）")

    # 切 val（按时间尾部 ~5%，从训练段末尾）
    n_val = int(len(train_ds) * 0.05)
    rng = np.random.RandomState(args.seed + 1)
    val_idx = rng.choice(len(train_ds), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(train_ds)), val_idx)
    train_sub = torch.utils.data.Subset(train_ds, train_idx)
    val_sub = torch.utils.data.Subset(train_ds, val_idx)
    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- 模型 ----
    model = ECLPatchTST(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        revin=True,
        individual=False,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 训练（early stopping on val_rmse）----
    print(f"开始训练（epochs={args.epochs}, patience={args.patience}, lr={args.lr}）...")
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, nb = 0.0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        train_epoch_time = time.time() - t0

        # val 早停
        model.eval()
        val_preds, val_t = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_preds.append(model(xb.to(device)).cpu().numpy())
                val_t.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_t = np.concatenate(val_t)
        val_rmse = float(np.sqrt(np.mean((val_preds - val_t) ** 2)))
        print(
            f"  epoch {epoch:2d} | loss={total_loss/max(nb,1):.4f} | "
            f"val_rmse={val_rmse:.4f} | {train_epoch_time:.1f}s"
        )
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  [early stop] {args.patience} epochs 无改善")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ---- 评测：test 用户预测低频段 + 相对指标 ----
    print("\n评测 test 用户低频段（{} ~ {}）...".format(TEST_START.date(), TEST_END.date()))
    test_data = build_test_matrix(
        matrix, test_cols, args.seq_len, args.pred_len,
        TEST_START, TEST_END,
    )
    user_rmse, user_naive_rmse, user_snaive_rmse = {}, {}, {}
    for u, (Xw, yt, yn, ys_) in test_data.items():
        if len(Xw) == 0:
            continue
        y_hat = predict_windows(model, Xw, args.eval_batch)
        mask = ~np.isnan(ys_)
        user_rmse[u] = rmse(yt[mask], y_hat[:, 0][mask])
        user_naive_rmse[u] = rmse(yt[mask], yn[mask])
        user_snaive_rmse[u] = rmse(yt[mask], ys_[mask])

    model_rmse_arr = np.array([v for v in user_rmse.values()])
    naive_rmse_arr = np.array([user_naive_rmse[u] for u in user_rmse])
    snaive_rmse_arr = np.array([user_snaive_rmse[u] for u in user_rmse])

    r_vs_naive = _ratio_stats(model_rmse_arr, naive_rmse_arr)
    r_vs_snaive = _ratio_stats(model_rmse_arr, snaive_rmse_arr)

    print("\n" + "=" * 60)
    print(f"PatchTST 跨用户迁移 | {len(user_rmse)} test 用户 | seq_len={args.seq_len}")
    print(f"  Mean RMSE = {model_rmse_arr.mean():.2f}  Std = {model_rmse_arr.std():.2f}")
    print(
        f"  ratio vs naive(lag_1): mean={r_vs_naive['mean']:.3f} "
        f"median={r_vs_naive['median']:.3f} pct_better={r_vs_naive['pct_better']:.1f}%"
    )
    print(
        f"  ratio vs snaive24   : mean={r_vs_snaive['mean']:.3f} "
        f"median={r_vs_snaive['median']:.3f} pct_better={r_vs_snaive['pct_better']:.1f}%"
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())