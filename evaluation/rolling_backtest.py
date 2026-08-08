# 滚动回测：按 Forecast Protocol 对预测月逐小时滚动预测
# ---------------------------------------------------------------
# 信息契约（horizon = 1，逐小时）：
#   预测小时 t 的特征只依赖 ≤ t-1 的 LOAD。
#
#   online_h1   ：observed = 历史 ∪ 预测月真实观测值（向量化，
#                与逐小时循环等价——预测 t 时 t 之前真值已知）
#   recursive   ：observed 逐小时回填自己的预测值（必须逐点计算，
#                避免对增长序列反复 build_features 的 O(N²)）
#
# _features_at 是 build_features 的逐点等价视图（T5 parity 测试保证一致）。
# ---------------------------------------------------------------
from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from data.task_builder import TARGET_COL, FEATURE_SPEC, GEFComTask, build_features
from evaluation.forecast_protocol import ONLINE_H1, ForecastProtocol
from models.replay_backends import ModelBackend

_HOUR = timedelta(hours=1)


def build_forecast_features(
    observed: pd.Series,
    forecast_ts: pd.DatetimeIndex,
    spec: List[dict] = FEATURE_SPEC,
) -> pd.DataFrame:
    """
    在 observed（历史 ∪ 预测月回填值）上一次性构建特征，返回预测月部分。
    用于 online_h1（observed 含真实观测，可向量化）。
    """
    frame = pd.DataFrame({TARGET_COL: observed})
    feat = build_features(frame, spec=spec)
    return feat.loc[forecast_ts]


def _features_at(
    observed: pd.Series, t: pd.Timestamp, spec: List[dict] = FEATURE_SPEC
) -> Dict[str, float]:
    """
    单点特征计算（recursive 协议用）。与 build_features 在相同输入下
    逐位一致：lag = observed[t-k]，rolling = mean/std(observed[t-w..t-1])。
    """
    row: Dict[str, float] = {}
    for s in spec:
        stype = s["type"]
        name = s["name"]
        if stype == "time":
            attr = s["attr"]
            if attr == "hour":
                row[name] = t.hour
            elif attr == "weekday":
                row[name] = t.weekday()
            elif attr == "month":
                row[name] = t.month
            elif attr == "is_weekend":
                row[name] = 1 if t.weekday() >= 5 else 0
        elif stype == "lag":
            row[name] = observed.get(t - _HOUR * s["k"], np.nan)
        elif stype == "rolling":
            w = s["window"]
            vals = observed.loc[t - _HOUR * w : t - _HOUR]
            if s["stat"] == "mean":
                row[name] = vals.mean()
            elif s["stat"] == "std":
                row[name] = vals.std()
            else:
                raise ValueError(f"未知 rolling 统计量: {s['stat']}")
        else:
            raise ValueError(f"未知特征类型: {stype}")
    return row


def rolling_predict(
    backend: ModelBackend,
    task: GEFComTask,
    protocol: ForecastProtocol = ONLINE_H1,
    spec: List[dict] = FEATURE_SPEC,
) -> pd.Series:
    """
    对 task 的预测月做逐小时滚动预测，返回以 forecast_ts 为索引的预测序列。

    online_h1 可向量化；recursive 必须逐小时（observed 回填预测值）。
    """
    history_load = task.history_df[TARGET_COL]

    if not protocol.recursive:
        observed = pd.concat([history_load, task.y_true])
        X = build_forecast_features(observed, task.forecast_ts, spec)
        y_hat = backend.predict(X)
        return pd.Series(y_hat, index=task.forecast_ts)

    # recursive：预测月内只能回填自己的预测值
    observed = history_load.copy()
    y_hats = []
    for t in task.forecast_ts:
        x = _features_at(observed, t, spec)
        yh = backend.predict(pd.DataFrame([x]))[0]
        y_hats.append(yh)
        observed.loc[t] = protocol.backfill(yh, task.y_true.loc[t])
    return pd.Series(y_hats, index=task.forecast_ts)
