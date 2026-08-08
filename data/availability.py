# GEFCom2014 数据可用性定义
# ---------------------------------------------------------------
# 职责：明确"每个 Task 当时可用的历史数据"与"预测区间"。
#
#   available_history(task_id) -> Availability
#     history_df  : 截至该 task 预测月前一刻的全部可得历史（拼接 L1..Lk，
#                   drop 2001–2005 前导 LOAD-NaN 块，逐小时连续）
#     available_until : 历史最后一刻（= 预测月前 1 小时）
#     forecast_ts     : 预测月逐小时时间戳（来自 benchmark 文件）
# ---------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from data.gefcom_loader import (
    GEFCOM_DATA_DIR,
    TASK_IDS,
    load_benchmark,
    load_train,
)


@dataclass(frozen=True)
class Availability:
    """某 Task 的"可用历史 + 预测区间"定义。"""

    task_id: int
    history_df: pd.DataFrame  # datetime 索引，逐小时连续，LOAD 无 NaN
    available_until: pd.Timestamp  # history_df 最后一刻
    forecast_ts: pd.DatetimeIndex  # 预测月逐小时（来自 benchmark）

    @property
    def forecast_start(self) -> pd.Timestamp:
        return self.forecast_ts[0]

    @property
    def forecast_end(self) -> pd.Timestamp:
        return self.forecast_ts[-1]

    @property
    def n_history(self) -> int:
        return len(self.history_df)

    @property
    def n_forecast(self) -> int:
        return len(self.forecast_ts)

    @property
    def forecast_month(self) -> str:
        return self.forecast_start.strftime("%Y-%m")


def available_history(
    task_id: int, data_dir: Path = GEFCOM_DATA_DIR
) -> Availability:
    """
    构建 Task 的可用历史 + 预测区间。

    - 拼接 L1..Lk 的 train 文件（L1 为全量历史，L2..Lk 为增量，天然连续）
    - drop 前导 LOAD-NaN 块（Task 1 的 2001–2005，不能用于训练，不填充）
    - 断言：历史逐小时连续；预测起点 == 历史终点 + 1 小时
    """
    if task_id not in TASK_IDS:
        raise ValueError(f"task_id 必须在 1..15，got {task_id}")

    frames: List[pd.DataFrame] = [
        load_train(k, data_dir) for k in range(1, task_id + 1)
    ]
    hist = pd.concat(frames)

    # 防御性去重排序（已验证拼接处无重复时间戳）
    hist = hist[~hist.index.duplicated(keep="first")].sort_index()

    # drop 前导 LOAD-NaN 块，而非 bfill 填充（避免伪造常量历史）
    load_notna = hist["LOAD"].notna()
    if not load_notna.all():
        first_valid = hist.index[load_notna][0]
        hist = hist.loc[first_valid:]
    if hist["LOAD"].isna().any():
        raise ValueError(f"Task {task_id} 历史存在内部 LOAD NaN，无法安全训练")

    # 断言逐小时连续
    diffs = hist.index.to_series().diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError(f"Task {task_id} 历史时间不连续")

    forecast_ts = load_benchmark(task_id, data_dir).index
    if forecast_ts[0] != hist.index.max() + pd.Timedelta(hours=1):
        raise ValueError(
            f"Task {task_id} 预测起点 {forecast_ts[0]} != 历史终点 + 1h "
            f"({hist.index.max() + pd.Timedelta(hours=1)})"
        )

    return Availability(
        task_id=task_id,
        history_df=hist,
        available_until=hist.index.max(),
        forecast_ts=forecast_ts,
    )


def get_task_metadata(data_dir: Path = GEFCOM_DATA_DIR) -> pd.DataFrame:
    """
    产出 Task 1–15 的可用性元数据表（每 Task 一行）：
      task_id / forecast_month / available_until / forecast_start / forecast_end
      n_history / n_forecast
    """
    rows = []
    for k in TASK_IDS:
        av = available_history(k, data_dir)
        rows.append(
            {
                "task_id": av.task_id,
                "forecast_month": av.forecast_month,
                "available_until": av.available_until,
                "forecast_start": av.forecast_start,
                "forecast_end": av.forecast_end,
                "n_history": av.n_history,
                "n_forecast": av.n_forecast,
            }
        )
    return pd.DataFrame(rows)
