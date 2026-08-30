# 回放评测模型后端（统一接口）
# ---------------------------------------------------------------
# ModelBackend 抽象：fit(train, val) / predict(X)。滚动回放中每个
# Task 会新建一个后端实例（fit 只在当期可用历史上进行，无跨 Task 泄漏）。
#
# 实现：
#   - LightGBMBackend     参数原样复用 models/baseline/lgb_gefcom2014.py，
#                         另加 deterministic/force_col_wise 保证可复现；无 scaler（树模型）
#   - SeasonalNaiveBackend  seasonal naive k 小时：y_hat[t] = y[t-k]
#   - PersistenceBackend     y_hat[t] = y[t-1]（= seasonal naive 1）
# ---------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd


class ModelBackend(ABC):
    """回放模型统一接口。"""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seed: int = 42,
    ) -> "ModelBackend": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def feature_importance(self) -> Optional[pd.DataFrame]:
        """可选：特征重要性（gain）。默认 None（naive/persistence 不实现）。

        runner 检测到 None 时跳过特征重要性上下文段。
        """
        return None


class LightGBMBackend(ModelBackend):
    name = "lightgbm"

    def __init__(
        self,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100,
        learning_rate: float = 0.05,
    ):
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.learning_rate = learning_rate
        self._feature_cols: List[str] = []
        self.best_iteration: int = 0

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seed: int = 42,
    ) -> "LightGBMBackend":
        train = train_df.dropna(subset=feature_cols + [target_col])
        val = val_df.dropna(subset=feature_cols + [target_col])
        self._feature_cols = list(feature_cols)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": self.learning_rate,
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
            "deterministic": True,  # 可复现
            "force_col_wise": True,
        }
        train_data = lgb.Dataset(train[feature_cols], label=train[target_col])
        val_data = lgb.Dataset(
            val[feature_cols], label=val[target_col], reference=train_data
        )
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=200),
            ],
        )
        self.best_iteration = self._model.best_iteration

        # 特征重要性（gain），供自进化 runner 构建上下文
        imp = self._model.feature_importance(importance_type="gain")
        total = float(imp.sum())
        self._importance = pd.DataFrame({
            "feature": self._feature_cols,
            "importance_gain": imp,
            "importance_gain_norm": (imp / total * 100 if total > 0 else 0),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        return self

    def feature_importance(self) -> Optional[pd.DataFrame]:
        return getattr(self, "_importance", None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X[self._feature_cols])


class SeasonalNaiveBackend(ModelBackend):
    """Seasonal Naive：y_hat[t] = y[t-k]（直接用 lag_k 特征列）。"""

    def __init__(self, k: int):
        self.k = int(k)
        self.name = f"seasonal_naive_{self.k}"

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seed: int = 42,
    ) -> "SeasonalNaiveBackend":
        col = f"lag_{self.k}"
        if col not in feature_cols:
            raise ValueError(f"Seasonal Naive {self.k} 需要特征列 '{col}'")
        self._lag_col = col
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self._lag_col].to_numpy(dtype=float)


class PersistenceBackend(SeasonalNaiveBackend):
    """Persistence：y_hat[t] = y[t-1]。"""

    def __init__(self):
        super().__init__(k=1)
        self.name = "persistence"


class LSTMBackend(ModelBackend):
    """LSTM 深度后端：把表格特征作为 channel 输入序列模型。

    - fit 在 train+val 连续历史上训练（滑窗 + 早停），scaler 只在 train 段 fit
    - predict 用 ``lag_1`` 特征重建预测月 target 序列，逐点滑窗预测（online_h1）
    - 无特征重要性（返回 None，runner 跳过该上下文段）

    与 ``models/LSTM/LSTM_baseline.py`` 复用同一 ``LSTMRegressor`` 与
    ``utils/data_loader.TimeSeriesDataset``，差异仅在：
      这里按 ``ModelBackend`` 契约从 train/val DataFrame 就地训练，
      predict 时把「逐行表格特征」还原成连续窗口。
    """

    name = "lstm"

    def __init__(
        self,
        seq_len: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_epochs: int = 30,
        patience: int = 8,
        batch_size: int = 128,
        train_window: int = 5000,
    ):
        self.seq_len = int(seq_len)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.train_window = int(train_window)
        self._feature_cols: List[str] = []
        self._target_col = "LOAD"
        self.best_iteration: int = 0

    def _to_array(self, df: pd.DataFrame, feature_cols: List[str],
                  target_col: str) -> np.ndarray:
        """构造 (n, 1+len(feature_cols)) 数组：target 第 0 列 + 特征，已缩放。"""
        y = self._target_scaler.transform(df[[target_col]].values)
        if feature_cols and self._feature_scaler is not None:
            x = self._feature_scaler.transform(df[feature_cols].values)
            return np.concatenate([y, x], axis=1)
        return y

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        seed: int = 42,
    ) -> "LSTMBackend":
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import DataLoader

        from models.LSTM.LSTM_baseline import LSTMRegressor
        from utils.data_loader import TimeSeriesDataset

        self._feature_cols = list(feature_cols)
        self._target_col = target_col
        torch.manual_seed(seed)
        np.random.seed(seed)

        train = train_df.dropna(subset=feature_cols + [target_col])
        val = val_df.dropna(subset=feature_cols + [target_col])
        # 训练窗口截断：时序模型用近期历史足够（LightGBM 用全量、LSTM 降本提速）
        if self.train_window > 0 and len(train) > self.train_window:
            train = train.iloc[-self.train_window:]
        # 连续历史（截断后 train 仍与 val 相邻），供 predict 重建窗口
        self._context = pd.concat([train, val]).sort_index()

        # scaler 只在 train 段 fit（防 val/test 泄漏）
        self._target_scaler = StandardScaler().fit(train[[target_col]].values)
        self._feature_scaler = (
            StandardScaler().fit(train[feature_cols].values) if feature_cols else None
        )

        train_arr = self._to_array(train, feature_cols, target_col)
        val_arr = self._to_array(val, feature_cols, target_col)

        train_loader = DataLoader(
            TimeSeriesDataset(train_arr, self.seq_len, pred_len=1),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TimeSeriesDataset(val_arr, self.seq_len, pred_len=1),
            batch_size=self.batch_size, shuffle=False, drop_last=False,
        )

        input_size = len(feature_cols) + 1  # 动态：Agent 特征数会变
        self._model = LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=1,
            dropout=self.dropout,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        no_improve = 0
        for epoch in range(1, self.max_epochs + 1):
            self._model.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = criterion(self._model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            self._model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for x, y in val_loader:
                    val_loss += criterion(self._model(x), y).item() * len(y)
                    n += len(y)
            val_loss = val_loss / max(n, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        self.best_iteration = best_epoch
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if len(X) == 1:
            raise NotImplementedError(
                "LSTMBackend 仅支持 online_h1（完整预测月特征表），recursive 逐点预测暂未实现"
            )
        if "lag_1" not in X.columns:
            raise ValueError(
                "LSTMBackend.predict 需要特征列 'lag_1' 重建预测月 target 序列"
            )
        forecast_ts = X.index
        hour = pd.Timedelta(hours=1)

        context = self._context
        context_target = context[self._target_col]
        # 重建预测月 target（前 N-1 个）：lag_1@t == target(t-1)，故 target(t) == X.loc[t+1,'lag_1']
        pred_target = X["lag_1"].shift(-1).iloc[:-1]
        full_target = pd.concat([context_target, pred_target])

        if self._feature_cols:
            full_feat = pd.concat([context[self._feature_cols], X[self._feature_cols]])
        else:
            full_feat = None

        y_hats = []
        for t in forecast_ts:
            win_target = (
                full_target.loc[t - hour * self.seq_len: t - hour]
                .to_numpy(dtype=float).reshape(-1, 1)
            )
            win_target_scaled = self._target_scaler.transform(win_target)
            if full_feat is not None:
                win_feat = (
                    full_feat.loc[t - hour * self.seq_len: t - hour]
                    .to_numpy(dtype=float)
                )
                win_feat_scaled = self._feature_scaler.transform(win_feat)
                win = np.concatenate([win_target_scaled, win_feat_scaled], axis=1)
            else:
                win = win_target_scaled

            x_t = torch.FloatTensor(win).unsqueeze(0)
            with torch.no_grad():
                pred_scaled = self._model(x_t).item()
            y_hats.append(
                float(self._target_scaler.inverse_transform([[pred_scaled]])[0, 0])
            )
        return np.array(y_hats)

    def feature_importance(self) -> Optional[pd.DataFrame]:
        return None


def make_backend(model_name: str, **kwargs) -> ModelBackend:
    """模型工厂。`seasonal_naive_all` 由 CLI 展开为 24/168 两个后端。"""
    name = model_name.lower()
    if name == "lightgbm":
        return LightGBMBackend(**kwargs)
    if name == "lstm":
        return LSTMBackend(**kwargs)
    if name == "seasonal_naive_24":
        return SeasonalNaiveBackend(24)
    if name == "seasonal_naive_168":
        return SeasonalNaiveBackend(168)
    if name == "persistence":
        return PersistenceBackend()
    raise ValueError(
        f"未知模型 '{model_name}'，可选: lightgbm / lstm / seasonal_naive_24 / "
        f"seasonal_naive_168 / seasonal_naive_all / persistence"
    )
