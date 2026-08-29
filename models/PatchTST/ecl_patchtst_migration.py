# PatchTST 跨用户迁移（ECL）— 统一评测协议 + 严格时间切分验证
# ---------------------------------------------------------------
# 用 PatchTST backbone（channel-independence + RevIN）做 ECL 跨用户迁移：
#   - 训练：随机 n_train 用户，目标时间 t < TRAIN_END（2.5 年）的窗口混合训练，
#     RevIN 逐序列归一化 → 治"用户规模差 140 倍"。
#   - 验证：严格时间切分（REVIEW 修正）——train 用户目标时间
#     [VAL_START, VAL_END) = 2014-06-01 ~ 2014-07-01，用于早停，
#     不再从训练窗中随机抽样（rng.choice 会让相邻滑窗并入 train/val，验证偏乐观）。
#   - 评测：61 个 test 用户（训练时从未见过），2014-07-01 ~ 2014-12-31，
#     与 LightGBM / persistence / seasonal naive 走同一统一协议
#     （evaluation/ecl_protocol.py），指标完全可比。
#
# 修订依据：ECL_REVIEW_NOTES.md（统一协议 / 严格 val 切分 / 可复现产物）。
#
# 用法：
#   python models/PatchTST/ecl_patchtst_migration.py --n-train 260 --epochs 25
#   python models/PatchTST/ecl_patchtst_migration.py --n-train 20 --epochs 2 --sample-ratio 0.05  # 冒烟
# ---------------------------------------------------------------
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 PatchTST_backbone 可 import

from PatchTST_backbone import PatchTST_backbone  # noqa: E402

from data.ecl_loader import load_ecl_matrix, split_users  # noqa: E402
from evaluation import ecl_protocol as ep  # noqa: E402


class ECLPatchTST(nn.Module):
    """单通道 PatchTST 封装。输入 [bs, seq_len] → 输出 [bs, pred_len]。

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
    """单通道滑窗数据集，按【目标时间】过滤窗口（REVIEW 要求）。

    每个样本 = 目标时间 target_time 下，窗口输入 s[t-seq_len : t] → 目标 s[t].
    用 min/max_target 把窗口分到 train（t < TRAIN_END）或 val
    （VAL_START <= t < VAL_END）集合，训练/验证目标时间严格不相交。
    每次构建记录 self.target_times（DatetimeIndex），用于交割断言。
    """

    def __init__(
        self,
        matrix: pd.DataFrame,
        users: List[str],
        seq_len: int,
        pred_len: int = 1,
        min_target: Optional[pd.Timestamp] = None,
        max_target: Optional[pd.Timestamp] = None,
        sample_ratio: float = 1.0,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        rng = np.random.RandomState(seed)

        ts = matrix.index
        self.target_times: List[pd.Timestamp] = []
        X_list, y_list = [], []
        for u in users:
            s = matrix[u].values
            n = len(s)
            # 有效起始 i 满足 i + seq_len + pred_len - 1 < n
            n_samples = n - seq_len - pred_len + 1
            if n_samples <= 0:
                continue
            starts = np.arange(n_samples)
            target_idx = starts + seq_len + pred_len - 1  # 窗口内最后一个目标位置
            target_t = ts[target_idx].to_numpy()  # datetime array

            # 目标时间过滤
            keep = np.ones(n_samples, dtype=bool)
            if min_target is not None:
                keep &= target_t >= np.datetime64(min_target)
            if max_target is not None:
                keep &= target_t < np.datetime64(max_target)
            if not keep.any():
                continue

            sel_starts = starts[keep]
            sel_target_t = target_t[keep]

            # 抽样
            if sample_ratio < 1.0 and len(sel_starts) > 0:
                n_pick = max(1, int(len(sel_starts) * sample_ratio))
                pick = np.sort(rng.choice(len(sel_starts), size=n_pick, replace=False))
                sel_starts = sel_starts[pick]
                sel_target_t = sel_target_t[pick]

            # 向量化构窗
            windows = np.lib.stride_tricks.sliding_window_view(
                s, seq_len)  # shape (n - seq_len + 1, seq_len)
            for st, tgt in zip(sel_starts, sel_target_t):
                X_list.append(windows[st].tolist())
                y_list.append(s[st + seq_len : st + seq_len + pred_len].tolist())
                self.target_times.append(pd.Timestamp(tgt))

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
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """每个 test 用户 → (X_windows, y_true, y_persist, y_snaive)。

    online 单步：窗口 = [t-seq_len, t-1] 该用户真实历史；persist=y[t-1]；
    snaive=y[t-24]；y_true=y[t]。向量化构窗，逐用户返回。
    """
    ts = matrix.index
    target_positions = np.arange(seq_len, len(ts))  # 目标索引（窗口起点 i=t-seq_len）
    result = {}
    for u in users:
        s = matrix[u].values
        windows = np.lib.stride_tricks.sliding_window_view(s, seq_len)
        # 全局时间掩码 → 目标索引
        times = ts[target_positions]
        in_range = (times >= start) & (times <= end)
        idx = target_positions[in_range]
        X_w = np.array([windows[t - seq_len].tolist() for t in idx], dtype=np.float32)
        y_t = s[idx].astype(np.float32)
        y_p = np.where(idx - 1 >= 0, s[idx - 1], np.nan).astype(np.float32)
        y_s = np.where(idx - 24 >= 0, s[idx - 24], np.nan).astype(np.float32)
        result[u] = (X_w, y_t, y_p, y_s)
    return result


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
    parser = argparse.ArgumentParser(description="PatchTST ECL 跨用户迁移（统一协议）")
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
    parser.add_argument("--outdir", default=None,
                        help="产物目录（默认 experiments/output/ecl_patchtst/<run_id>）")
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

    t0 = time.time()
    # ---- train / val 严格按目标时间切分 ----
    print(f"构建 train 窗口（目标 < {ep.TRAIN_END.date()}，抽样 {args.sample_ratio}）...")
    train_ds = UserWindowDataset(
        matrix, train_cols, args.seq_len, args.pred_len,
        min_target=None, max_target=ep.TRAIN_END,
        sample_ratio=args.sample_ratio, seed=args.seed,
    )
    print(f"  train 窗口: {len(train_ds)}  （构建 {time.time()-t0:.1f}s）")

    print(f"构建 val 窗口（{ep.VAL_START.date()} <= t < {ep.VAL_END.date()}，不抽样）...")
    val_ds = UserWindowDataset(
        matrix, train_cols, args.seq_len, args.pred_len,
        min_target=ep.VAL_START, max_target=ep.VAL_END,
        sample_ratio=1.0, seed=args.seed,
    )
    print(f"  val 窗口: {len(val_ds)}")

    # ---- REVIEW 要求：严格时间交割断言 ----
    tr_times = np.array([pd.Timestamp(t) for t in train_ds.target_times])
    va_times = np.array([pd.Timestamp(t) for t in val_ds.target_times])
    assert len(tr_times) > 0 and len(va_times) > 0, "train/val 窗口为空"
    assert tr_times.max() < va_times.min(), (
        f"train 目标时间与 val 目标时间未严格分离: train_max={tr_times.max()} "
        f"val_min={va_times.min()}")
    assert np.intersect1d(tr_times, va_times).size == 0, "train/val 目标时间相交！"
    print(f"  [OK] 严格时间交割: train_max_time={tr_times.max()}  "
          f"val_min_time={va_times.min()}  窗口数 train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- 模型 ----
    model = ECLPatchTST(
        seq_len=args.seq_len, pred_len=args.pred_len,
        patch_len=args.patch_len, stride=args.stride,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, revin=True, individual=False,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- 训练（early stopping on val_rmse）----
    print(f"开始训练（epochs={args.epochs}, patience={args.patience}, lr={args.lr}）...")
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    training_history = []
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

        model.eval()
        val_preds, val_t = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_preds.append(model(xb.to(device)).cpu().numpy())
                val_t.append(yb.numpy())
        val_preds = np.concatenate(val_preds)
        val_t = np.concatenate(val_t)
        val_rmse = float(np.sqrt(np.mean((val_preds - val_t) ** 2)))
        training_history.append({
            "epoch": epoch,
            "train_loss": round(total_loss / max(nb, 1), 4),
            "val_rmse": round(val_rmse, 4),
            "epoch_seconds": round(train_epoch_time, 2),
        })
        print(
            f"  epoch {epoch:2d} | loss={total_loss/max(nb,1):.4f} | "
            f"val_rmse={val_rmse:.4f} | {train_epoch_time:.1f}s"
        )
        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            best_epoch = epoch
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
    print(f"  best_epoch={best_epoch} best_val_rmse={best_val:.4f}")

    # ---- 评测：test 用户低频段（统一协议）----
    print("\n评测 test 用户（{} ~ {}）...".format(ep.TEST_START.date(), ep.TEST_END.date()))
    test_data = build_test_matrix(matrix, test_cols, args.seq_len, ep.TEST_START, ep.TEST_END)
    user_actual, user_model, user_persist, user_snaive = {}, {}, {}, {}
    predictions_rows = []
    for u, (Xw, yt, yp, ys_) in test_data.items():
        if len(Xw) == 0:
            continue
        y_hat = predict_windows(model, Xw, args.eval_batch)[:, 0]
        user_actual[u] = yt
        user_model[u] = y_hat
        user_persist[u] = yp
        user_snaive[u] = ys_
        # 时间戳对齐：test 段内目标索引
        ts = matrix.index
        target_pos = np.arange(args.seq_len, len(ts))
        times = ts[target_pos]
        in_range = (times >= ep.TEST_START) & (times <= ep.TEST_END)
        t_sel = times[in_range]
        for i, tt in enumerate(t_sel):
            predictions_rows.append({
                "user": u, "timestamp": tt,
                "actual": yt[i], "prediction": y_hat[i],
                "persistence": yp[i], "seasonal_naive": ys_[i],
            })

    model_rmse, persist_rmse, snaive_rmse, n_pred = ep.score_users(
        user_actual, user_model, user_persist, user_snaive,
    )
    summary = ep.summarize(
        model_rmse, persist_rmse, snaive_rmse, n_pred,
        model_name=f"patchtst_s{args.seq_len}",
        n_train_users=len(train_cols),
        n_test_users=len(test_cols),
        n_train_windows=len(train_ds),
        n_val_windows=len(val_ds),
        best_epoch=best_epoch,
        best_val_rmse=best_val,
    )

    # ---- 产物输出 ----
    outdir = Path(args.outdir) if args.outdir else (
        PROJECT_ROOT / "experiments" / "output" / "ecl_patchtst" /
        time.strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": f"patchtst_{args.seq_len}",
        "git_commit": ep.git_commit(),
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "cpu" if device.type == "cpu" else "cuda",
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "command_args": vars(args),
        "data": {
            "source": "ECL/electricity.txt (preprocessed, laiguokun)",
            "n_users": 321,
            "n_train_users": len(train_cols),
            "n_test_users": len(test_cols),
            "train_users_hash": ep.hash_list(train_cols),
            "test_users_hash": ep.hash_list(test_cols),
        },
        "time_split": {
            "train_until": str(ep.TRAIN_END),
            "val": f"{ep.VAL_START} ~ {ep.VAL_END}",
            "test": f"{ep.TEST_START} ~ {ep.TEST_END}",
        },
        "model_params": {
            "n_params": n_params,
            "seq_len": args.seq_len, "pred_len": args.pred_len,
            "patch_len": args.patch_len, "stride": args.stride,
            "d_model": args.d_model, "n_layers": args.n_layers,
            "revin": True,
        },
        "n_train_windows": len(train_ds),
        "n_val_windows": len(val_ds),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val,
        "summary": {k: v for k, v in summary.items()},
    }
    (outdir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (outdir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ep.user_rows(model_rmse, persist_rmse, snaive_rmse, n_pred).to_csv(
        outdir / "per_user_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(predictions_rows).to_csv(
        outdir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_history).to_csv(
        outdir / "training_history.csv", index=False, encoding="utf-8-sig")
    torch.save(model.state_dict(), outdir / "best_model.pt")

    # ---- 打印 ----
    print("\n" + "=" * 60)
    print(f"PatchTST 跨用户迁移 | {summary['n_users']} test 用户 | seq_len={args.seq_len}")
    print(f"  Mean RMSE = {summary['mean_rmse']:.4f} | Median = {summary['median_rmse']:.4f} "
          f"| Std = {summary['std_rmse']:.4f}")
    r1, r2 = summary["ratio_vs_persistence"], summary["ratio_vs_snaive"]
    print(
        f"  ratio vs persistence(lag_1): mean={r1['mean']:.4f} "
        f"median={r1['median']:.4f} pct_better={r1['pct_better']:.1f}%"
    )
    print(
        f"  ratio vs snaive24(lag_24)  : mean={r2['mean']:.4f} "
        f"median={r2['median']:.4f} pct_better={r2['pct_better']:.1f}%"
    )
    print(f"  产物 -> {outdir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())