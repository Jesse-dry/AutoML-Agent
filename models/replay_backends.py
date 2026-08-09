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


def make_backend(model_name: str, **kwargs) -> ModelBackend:
    """模型工厂。`seasonal_naive_all` 由 CLI 展开为 24/168 两个后端。"""
    name = model_name.lower()
    if name == "lightgbm":
        return LightGBMBackend(**kwargs)
    if name == "seasonal_naive_24":
        return SeasonalNaiveBackend(24)
    if name == "seasonal_naive_168":
        return SeasonalNaiveBackend(168)
    if name == "persistence":
        return PersistenceBackend()
    raise ValueError(
        f"未知模型 '{model_name}'，可选: lightgbm / seasonal_naive_24 / "
        f"seasonal_naive_168 / seasonal_naive_all / persistence"
    )
