# GEFCom2014-P（电价）数据统一加载器
# ---------------------------------------------------------------
# 职责：把 Price 赛道 Task 1–15 的 train / benchmark / solution 文件统一加载为
#       datetime 索引的 DataFrame，并解析每个 Task 预测日的"真值"。语义对齐
#       Load 版（data/gefcom_loader.py）/ Wind 版（data/wind_loader.py）。
#
# Price 数据约定（GEFCom2014-P_V2）：
#   - 文件为普通 CSV（非 Wind 的 zip 套 zip），单分区 ZONEID=1（无 Zone 维度）。
#   - Task k/Task{k}_P.csv  历史 + 预测日（完整前缀）：
#       列 ZONEID, timestamp, Forecasted Total Load, Forecasted Zonal Load, Zonal Price。
#       历史 2011-01-01 → 预测日前一天 23:00 的 Zonal Price 为真实值；
#       预测日 00:00–23:00（24h）Zonal Price 为 NaN 占位（待预测）。
#       外生「Forecasted Total/Zonal Load」全文件无 NaN（决策时点可得的负荷预报）。
#   - Task k/Benchmark{k}_P.csv  预测日 99 分位模板（24 行 × 101 列：ZONEID, timestamp,
#       0.01..0.99）；本加载器只消费其 timestamp。⚠️ Task 7 特例：Benchmark7_P_new3.csv。
#   - Solution to Task 15/Solution to Task15_P.csv  Task 15 预测日官方真值 Zonal Price。
#
# 与 Load/Wind 的关键差异（接入时单独验证，勿套用）：
#   - 预测窗口 = 1 天（24h），非整月（Load/Wind 为 744h）。15 个 Task 预测 15 个
#     特定日期（非连续）：06-16, 06-17, 06-24, 07-04, 07-09, 07-13, 07-16, 07-18,
#     07-19, 07-20, 07-24, 07-25, 12-07, 12-08, 12-17。
#   - 目标 Zonal Price 为连续肥尾值（均值≈48.6，p99≈156，max≈364），非归一化；
#     Load ~150MW / Wind 归一化[0,1] / Price 电价 是三套不同物理量纲，不可横向对比。
#   - 单分区：无 Zone 维度，逐 Task 独立建模（不像 Wind 有 10 分区均值）。
#
# 时间戳格式：Price V2 为严格 "MMDDYYYY H:MM"（如 01012011 0:00 → 2011-01-01 00:00；
#   06162013 22:00 → 2013-06-16 22:00）。月/日固定 2 位有前导零、年 4 位、小时无前导零。
#   与 Load 的变长 "MMDDYY/MMDDYYYY" 消歧解析器（data/preprocessing）、Wind 的
#   "YYYYMMDD"（data/wind_loader）均不同，这里用专有解析器 parse_price_timestamp
#   （%m%d%Y，严格无歧义）。
#
# 真值来源（与 Load/Wind 一致）：
#   - k = 1..14：Task{k+1}_P.csv 的 Zonal Price（预测日段）== Task k 预测日真实电价
#     （Task{k+1} 文件覆盖到 Task k 预测日之后，时间戳与 benchmark{k} 逐行一致）
#   - k = 15：官方 Solution to Task15_P.csv
#
# DST 边界瑕疵（官方数据固有，仅影响训练历史，不影响预测窗口）：
#   2013-03-10（美东夏令时开始）这一天，官方 train 文件里 "01:00" 出现两次、
#   "02:00" 缺失（第二个 01:00 实为 02:00 的电价，数据录入所致）。2011/2012 全年
#   及 2013 秋季（11-03）均无此现象。price_available_history 对重复时间戳
#   keep="first" 去重、并对紧随其后的 2h gap 放行（其余必须严格 1h 连续）。
#   预测日（06-16…12-17）均非 DST 日，故预测窗口 / 真值不受影响。
# ---------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from data.availability import Availability

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRICE_DATA_DIR = PROJECT_ROOT / "GEFCom2014-P_v2"
PRICE_TASK_IDS = tuple(range(1, 16))

PRICE_TARGET_COL = "Zonal Price"
PRICE_EXOGENOUS_COLS = ["Forecasted Total Load", "Forecasted Zonal Load"]
_NUM_COLS = [PRICE_TARGET_COL] + PRICE_EXOGENOUS_COLS


# ---------------------------------------------------------------
# 时间戳解析（Price V2 格式，严格 MMDDYYYY H:MM）
# ---------------------------------------------------------------

def parse_price_timestamp(ts_str: str) -> Optional[datetime]:
    """解析 "MMDDYYYY H:MM" → datetime。

    月/日固定 2 位有前导零、年 4 位、小时无前导零（0:00–23:00）。
    格式严格，无 Load 的变长歧义。
    """
    try:
        date_str, time_str = ts_str.strip().split(" ")
        dt = datetime.strptime(date_str, "%m%d%Y")
        return dt.replace(hour=int(time_str.split(":")[0]))
    except Exception:
        return None


def _load_price_frame(path: Path) -> pd.DataFrame:
    """读 Price CSV（裸文件），统一时间戳解析 + 数值化，datetime 索引。

    返回 DataFrame 含数值列（Zonal Price + 外生）；ZONEID/timestamp 列删除
    （单分区 ZONEID=1 无意义）。
    """
    df = pd.read_csv(path)
    df["datetime"] = df["timestamp"].map(parse_price_timestamp)
    n_fail = int(df["datetime"].isna().sum())
    if n_fail > 0:
        raise ValueError(f"{path.name}: {n_fail} 行时间戳解析失败")
    df = df.set_index("datetime").sort_index()
    # DST 边界去重（2013-03-10 "01:00" 重复，见文件头注释）：keep first，保证索引唯一
    df = df[~df.index.duplicated(keep="first")]
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    drop_cols = [c for c in ("timestamp", "ZONEID") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


# ---------------------------------------------------------------
# 路径
# ---------------------------------------------------------------

def price_task_paths(
    task_id: int, data_dir: Path = PRICE_DATA_DIR
) -> Dict[str, Path]:
    """返回某 Task 涉及的全部数据文件路径。

    ⚠️ Task 7 的 benchmark 文件名特例为 Benchmark7_P_new3.csv（官方数据如此）。
    """
    if task_id not in PRICE_TASK_IDS:
        raise ValueError(
            f"task_id 必须在 {PRICE_TASK_IDS[0]}..{PRICE_TASK_IDS[-1]}，got {task_id}"
        )
    data_dir = Path(data_dir)
    d = data_dir / f"Task {task_id}"
    benchmark_name = (
        "Benchmark7_P_new3.csv" if task_id == 7 else f"Benchmark{task_id}_P.csv"
    )
    return {
        "train": d / f"Task{task_id}_P.csv",
        "benchmark": d / benchmark_name,
        # ⚠️ 目录名无空格 "Solution to Task15"（Load/Wind 为 "Solution to Task 15" 有空格）
        "solution": data_dir / "Solution to Task15" / "Solution to Task15_P.csv",
    }


# ---------------------------------------------------------------
# 加载
# ---------------------------------------------------------------

def load_price_train(
    task_id: int, data_dir: Path = PRICE_DATA_DIR
) -> pd.DataFrame:
    """加载 Task 的 train 文件（datetime 索引，含 Zonal Price + 外生列）。

    预测日 24h 的 Zonal Price 为 NaN 占位；外生列全文件非 NaN。
    """
    p = price_task_paths(task_id, data_dir)
    return _load_price_frame(p["train"])


def load_price_benchmark_ts(
    task_id: int, data_dir: Path = PRICE_DATA_DIR
) -> pd.DatetimeIndex:
    """加载 Task 预测日 24h 逐小时时间戳（来自 benchmark 文件）。"""
    p = price_task_paths(task_id, data_dir)
    df = pd.read_csv(p["benchmark"])
    ts = pd.DatetimeIndex(df["timestamp"].map(parse_price_timestamp).dropna())
    return ts.sort_values().unique()


def load_price_solution(data_dir: Path = PRICE_DATA_DIR) -> pd.DataFrame:
    """加载 Task 15 官方真值（datetime 索引，单列 Zonal Price）。"""
    p = price_task_paths(15, data_dir)
    return _load_price_frame(p["solution"])


def load_price_forecast_exogenous(
    task_id: int, data_dir: Path = PRICE_DATA_DIR
) -> pd.DataFrame:
    """预测日 24h 的 Forecasted Total/Zonal Load（决策时点可得，外生非泄漏）。"""
    df = load_price_train(task_id, data_dir)
    ts = load_price_benchmark_ts(task_id, data_dir)
    return df[PRICE_EXOGENOUS_COLS].reindex(ts)


def load_price_ground_truth(
    task_id: int, data_dir: Path = PRICE_DATA_DIR
) -> pd.DataFrame:
    """
    解析 Task 预测日的真实 Zonal Price（以预测日时间戳为索引，单列）。

    - task 15：官方 Solution to Task15_P.csv
    - task k (1..14)：Task{k+1}_P.csv 的 Zonal Price（预测日段）

    任何预测小时缺失真值则抛错（价格不可线性插值造假）。
    """
    forecast_ts = load_price_benchmark_ts(task_id, data_dir)

    if task_id == 15:
        truth = load_price_solution(data_dir)[PRICE_TARGET_COL]
    else:
        truth = load_price_train(task_id + 1, data_dir)[PRICE_TARGET_COL]

    y_true = truth.reindex(forecast_ts)
    if y_true.isna().any():
        n_missing = int(y_true.isna().sum())
        raise ValueError(f"Task {task_id} 预测日真值缺失 {n_missing} 小时")
    return y_true.to_frame(PRICE_TARGET_COL)


# ---------------------------------------------------------------
# 可用性（Price 单文件即完整前缀，但预测日 24h 为 NaN 占位需剔除）
# ---------------------------------------------------------------

def price_available_history(
    task_id: int, data_dir: Path = PRICE_DATA_DIR
) -> Availability:
    """
    构建 Task 的可用历史 + 预测区间。

    Price 的 Task{k} train 文件本身就是"2011-01-01 → 预测日前一天"的完整前缀，
    无需像 Load 那样拼接 L1..Lk；预测日 24h 的 NaN 占位行剔除后即为可用历史。
    """
    full = load_price_train(task_id, data_dir)

    # 剔除预测日 NaN 占位段（历史 = Zonal Price 非 NaN 段）
    price_notna = full[PRICE_TARGET_COL].notna()
    if not price_notna.all():
        hist = full.loc[price_notna].copy()
    else:
        hist = full.copy()

    # 注：重复时间戳已在 _load_price_frame 统一去重（DST 边界，见文件头注释）

    if hist[PRICE_TARGET_COL].isna().any():
        raise ValueError(f"Task {task_id} 历史存在内部 Zonal Price NaN，无法安全训练")

    # 断言逐小时连续（容忍 DST 春季 2h gap，其余必须严格 1h）
    diffs = hist.index.to_series().diff().dropna()
    if not diffs.isin([pd.Timedelta(hours=1), pd.Timedelta(hours=2)]).all():
        raise ValueError(f"Task {task_id} 历史时间不连续")

    forecast_ts = load_price_benchmark_ts(task_id, data_dir)
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


def get_price_task_metadata(data_dir: Path = PRICE_DATA_DIR) -> pd.DataFrame:
    """
    产出 Price Task 1–15 可用性元数据表（每 Task 一行）：
      task_id / forecast_day / available_until / forecast_start / forecast_end
      n_history / n_forecast
    """
    rows = []
    for tid in PRICE_TASK_IDS:
        av = price_available_history(tid, data_dir)
        rows.append(
            {
                "task_id": av.task_id,
                "forecast_day": av.forecast_start.strftime("%Y-%m-%d"),
                "available_until": av.available_until,
                "forecast_start": av.forecast_start,
                "forecast_end": av.forecast_end,
                "n_history": av.n_history,
                "n_forecast": av.n_forecast,
            }
        )
    return pd.DataFrame(rows)
