# GEFCom2014-S（光伏）数据统一加载器
# ---------------------------------------------------------------
# 职责：把 Solar 赛道 Task 1–15 × Zone 1–3 的 train / predictors / benchmark /
#       solution 文件统一加载为 datetime 索引的 DataFrame，并解析每个 Task
#       预测月的"真值"。语义与 Load 版（data/gefcom_loader.py）/ Wind 版
#       （data/wind_loader.py）对齐。
#
# Solar 数据约定（GEFCom2014-S_V2）：
#   - Task {k}/train{k}.csv     该 task 的完整可用历史（从 2012-04-01 起逐月增长的
#       完整前缀），列 ZONEID,TIMESTAMP,POWER。单文件即完整前缀（无需像 Load 那样
#       拼接 L1..Lk）。
#   - Task {k}/predictors{k}.csv  气象外生（ECMWF NWP 12 变量 VAR78~VAR228），
#       覆盖「完整历史 + 预测月」，列 ZONEID,TIMESTAMP,VAR*。决策时点可得，外生非泄漏。
#   - Task {k}/benchmark{k:02d}.csv  预测月 99 分位模板（3 分区 × 720~744h）；
#       注意 benchmark 文件名补零到 2 位（benchmark01..benchmark09），
#       train/predictors 不补零（train1/predictors1）。本加载器只消费其时间戳。
#   - Solution to Task 15/Solution to Task 15.csv  Task 15 预测月官方真值 POWER。
#
# 时间戳格式与 Wind 相同：严格 "YYYYMMDD H:MM"（如 20120401 1:00），
# 无 Load 的变长歧义，不能复用 data/preprocessing.load_single_task，
# 这里用专有解析器 parse_solar_timestamp。
#
# 真值来源（与 Load/Wind 一致）：
#   - k = 1..14：Task{k+1} train 的 POWER == Task k 预测月的真实出力
#     （Task{k+1} train 覆盖到 Task k 预测月末，时间戳与 benchmark{k} 逐行一致）
#   - k = 15：官方 Solution to Task 15.csv（按 zone 过滤）
#
# 光伏特有现象：夜间出力恒 0（约 43.8% 小时 POWER=0），目标非负 [0,1]。
# ---------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data.availability import Availability

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOLAR_DATA_DIR = PROJECT_ROOT / "GEFCom2014-S_V2" / "Solar"
SOLAR_TASK_IDS = tuple(range(1, 16))
SOLAR_ZONES = tuple(range(1, 4))  # 3 个分区（ZONEID = 1/2/3）

SOLAR_TARGET_COL = "POWER"
# 气象外生列：从 predictors 12 个 VAR 里选 3 个最贴光伏出力的
SOLAR_WEATHER_COLS = ["VAR169", "VAR164", "VAR167"]
#   VAR169 = SSRD 地表太阳短波辐射（光伏出力的直接驱动）
#   VAR164 = TCC  总云量 0-1（云遮太阳）
#   VAR167 = 2T   2 米温度 K（面板效率）
_NUM_COLS = ["POWER", "VAR169", "VAR164", "VAR167"]


# ---------------------------------------------------------------
# 时间戳解析（与 Wind 相同的严格 YYYYMMDD H:MM）
# ---------------------------------------------------------------

def parse_solar_timestamp(ts_str: str) -> Optional[datetime]:
    """解析 "YYYYMMDD H:MM" → datetime。格式严格，无 Load 的变长歧义。"""
    try:
        date_str, time_str = ts_str.strip().split(" ")
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.replace(hour=int(time_str.split(":")[0]))
    except Exception:
        return None


def _load_solar_frame(path: Path) -> pd.DataFrame:
    """读 Solar CSV（普通 CSV，无 Wind 那种 zip 套 zip），统一时间戳解析 +
    数值化 + datetime 索引。返回 DataFrame 含 ZONEID 列 + 数值列；时间列删除。
    """
    df = pd.read_csv(path)
    df["datetime"] = df["TIMESTAMP"].map(parse_solar_timestamp)
    n_fail = int(df["datetime"].isna().sum())
    if n_fail > 0:
        raise ValueError(f"{path.name}: {n_fail} 行时间戳解析失败")
    df = df.set_index("datetime").sort_index()
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "TIMESTAMP" in df.columns:
        df = df.drop(columns=["TIMESTAMP"])
    return df


def _fill_target_nan(series: pd.Series, who: str) -> pd.Series:
    """对目标列少量缺失做线性插值（limit 24，双向），插不动则抛错。

    与仓库 preprocessing.fill_missing_values 的插值惯例一致。
    """
    s = series.copy()
    n_missing = int(s.isna().sum())
    if n_missing == 0:
        return s
    s = s.interpolate(method="linear", limit=24, limit_direction="both")
    s = s.ffill().bfill()
    remaining = int(s.isna().sum())
    if remaining > 0:
        raise ValueError(f"{who} 缺失 {n_missing} 小时（插值后仍缺 {remaining}）")
    print(f"  [INFO] {who}: 插值填充 {n_missing} 个缺失小时")
    return s


# ---------------------------------------------------------------
# 路径
# ---------------------------------------------------------------

def solar_task_paths(
    task_id: int, data_dir: Path = SOLAR_DATA_DIR
) -> Dict[str, Path]:
    """返回某 Task 涉及的全部数据文件路径。"""
    if task_id not in SOLAR_TASK_IDS:
        raise ValueError(f"task_id 必须在 {SOLAR_TASK_IDS[0]}..{SOLAR_TASK_IDS[-1]}，got {task_id}")
    data_dir = Path(data_dir)
    d = data_dir / f"Task {task_id}"
    return {
        "train": d / f"train{task_id}.csv",
        "benchmark": d / f"benchmark{task_id:02d}.csv",  # 注意：benchmark 补零到 2 位
        "predictors": d / f"predictors{task_id}.csv",
        "solution": data_dir / "Solution to Task 15" / "Solution to Task 15.csv",
    }


# ---------------------------------------------------------------
# 加载
# ---------------------------------------------------------------

def load_solar_train(
    task_id: int, zone: int, data_dir: Path = SOLAR_DATA_DIR
) -> pd.DataFrame:
    """加载 Task×Zone 的 train 历史（datetime 索引，含 POWER）。"""
    p = solar_task_paths(task_id, data_dir)
    df = _load_solar_frame(p["train"])
    if "ZONEID" in df.columns:
        df = df[df["ZONEID"] == zone]
    return df


def load_solar_predictors(
    task_id: int, zone: int, data_dir: Path = SOLAR_DATA_DIR
) -> pd.DataFrame:
    """加载 Task×Zone 的气象外生（datetime 索引，VAR78~VAR228，覆盖历史 + 预测月）。"""
    p = solar_task_paths(task_id, data_dir)
    df = _load_solar_frame(p["predictors"])
    if "ZONEID" in df.columns:
        df = df[df["ZONEID"] == zone]
    return df


def load_solar_benchmark_ts(
    task_id: int, data_dir: Path = SOLAR_DATA_DIR
) -> pd.DatetimeIndex:
    """加载 Task 预测月逐小时时间戳（来自 benchmark 文件，3 分区时间戳一致）。"""
    p = solar_task_paths(task_id, data_dir)
    df = pd.read_csv(p["benchmark"])
    ts = pd.DatetimeIndex(df["TIMESTAMP"].map(parse_solar_timestamp).dropna())
    return ts.sort_values().unique()


def load_solar_solution(data_dir: Path = SOLAR_DATA_DIR) -> pd.DataFrame:
    """加载 Task 15 官方真值（3 分区全，datetime 索引 + ZONEID + POWER）。"""
    p = solar_task_paths(15, data_dir)
    return _load_solar_frame(p["solution"])


def load_solar_ground_truth(
    task_id: int, zone: int, data_dir: Path = SOLAR_DATA_DIR
) -> pd.DataFrame:
    """
    解析 Task 预测月的真实 POWER（以预测月时间戳为索引，单列）。

    - task 15：官方 Solution to Task 15.csv（按 zone 过滤）
    - task k (1..14)：Task{k+1} train 的 POWER（已验证与 benchmark 时间戳逐行一致）

    任何预测小时缺失真值则抛错。
    """
    forecast_ts = load_solar_benchmark_ts(task_id, data_dir)

    if task_id == 15:
        sol = load_solar_solution(data_dir)
        sol_zone = sol[sol["ZONEID"] == zone] if "ZONEID" in sol else sol
        truth = sol_zone[SOLAR_TARGET_COL].reindex(forecast_ts)
    else:
        truth = load_solar_train(task_id + 1, zone, data_dir)[SOLAR_TARGET_COL].reindex(forecast_ts)

    truth = _fill_target_nan(truth, f"Solar Task {task_id} Zone {zone} 真值")
    return truth.to_frame(SOLAR_TARGET_COL)


# ---------------------------------------------------------------
# 可用性（镜像 wind_available_history 语义，但气象外生在单独 predictors 文件）
# ---------------------------------------------------------------

def solar_available_history(
    task_id: int, zone: int, data_dir: Path = SOLAR_DATA_DIR
) -> Availability:
    """
    构建 Task×Zone 的可用历史 + 预测区间。

    Solar 的 Task{k} train 文件是"2012-04-01 → 预测月前一刻"的完整前缀，
    无需拼接；气象外生（VAR169/VAR164/VAR167）来自 predictors 文件的历史段，
    与 train 时间戳逐行对齐。
    """
    hist = load_solar_train(task_id, zone, data_dir)
    pred = load_solar_predictors(task_id, zone, data_dir)

    # 目标列缺失插值
    hist = hist.copy()
    hist[SOLAR_TARGET_COL] = _fill_target_nan(
        hist[SOLAR_TARGET_COL], f"Solar Task {task_id} Zone {zone} 历史"
    )

    # 合并气象外生（历史段，predictors 覆盖完整历史）
    for c in SOLAR_WEATHER_COLS:
        hist[c] = pred[c].reindex(hist.index)

    # 断言逐小时连续
    diffs = hist.index.to_series().diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError(f"Solar Task {task_id} Zone {zone} 历史时间不连续")

    forecast_ts = load_solar_benchmark_ts(task_id, data_dir)
    if forecast_ts[0] != hist.index.max() + pd.Timedelta(hours=1):
        raise ValueError(
            f"Solar Task {task_id} Zone {zone} 预测起点 {forecast_ts[0]} != 历史终点 + 1h "
            f"({hist.index.max() + pd.Timedelta(hours=1)})"
        )

    return Availability(
        task_id=task_id,
        history_df=hist,
        available_until=hist.index.max(),
        forecast_ts=forecast_ts,
    )


def get_solar_task_metadata(
    zones: List[int] = None, data_dir: Path = SOLAR_DATA_DIR
) -> pd.DataFrame:
    """
    产出 Solar Task×Zone 可用性元数据表（每 Task×Zone 一行）：
      task_id / zone / forecast_month / available_until / forecast_start / forecast_end
      n_history / n_forecast
    """
    if zones is None:
        zones = list(SOLAR_ZONES)
    rows = []
    for tid in SOLAR_TASK_IDS:
        for zone in zones:
            av = solar_available_history(tid, zone, data_dir)
            rows.append(
                {
                    "task_id": av.task_id,
                    "zone": zone,
                    "forecast_month": av.forecast_month,
                    "available_until": av.available_until,
                    "forecast_start": av.forecast_start,
                    "forecast_end": av.forecast_end,
                    "n_history": av.n_history,
                    "n_forecast": av.n_forecast,
                }
            )
    return pd.DataFrame(rows)
