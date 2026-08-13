# GEFCom2014-W（风电）数据统一加载器
# ---------------------------------------------------------------
# 职责：把 Wind 赛道 Task 1–15 × Zone 1–10 的 train / expvars / benchmark /
#       solution 文件统一加载为 datetime 索引的 DataFrame，并解析每个 Task
#       预测月的"真值"。语义与 Load 版（data/gefcom_loader.py）对齐。
#
# Wind 数据约定（GEFCom2014-W_V2）：
#   - Task k/Task{k}_W_Zone1_10.zip   每分区 train 文件（zip 内）：
#       2012-01-01 → 预测月前一刻的逐小时历史，列 ZONEID,TIMESTAMP,TARGETVAR,U10,V10,U100,V100。
#       单文件即 Task k 的完整可用前缀（无需像 Load 那样拼接 L1..Lk）。
#   - Task k/TaskExpVars{k}_W_Zone1_10.zip  每分区预测月气象预报（zip 内），
#       列 ZONEID,TIMESTAMP,U10,V10,U100,V100，与 benchmark 时间戳逐行对齐（决策时点可得，外生）。
#   - Task k/benchmark{k}_W.csv        预测月 99 分位模板（10 分区 × 744h）；本加载器只消费其时间戳。
#   - Solution to Task 15/solution15_W.csv  Task 15 预测月的官方真值 TARGETVAR（10 分区全）。
#
# 时间戳格式与 Load 不同：Wind V2 为严格 "YYYYMMDD H:MM"（如 20120101 1:00），
# 与 Load 的变长 "MMDDYY H:MM" 不兼容，不能复用 data/preprocessing.load_single_task，
# 这里用专有解析器 parse_wind_timestamp（格式严格，无歧义）。
#
# 真值来源（与 Load 一致）：
#   - k = 1..14：Task{k+1}_W_Zone{zone}.csv 的 TARGETVAR == Task k 预测月的真实出力
#     （Task{k+1} train 覆盖到 Task k 预测月末，时间戳与 benchmark{k} 逐行一致）
#   - k = 15：官方 solution15_W.csv（按 zone 过滤）
# ---------------------------------------------------------------
import io
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from data.availability import Availability

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WIND_DATA_DIR = PROJECT_ROOT / "GEFCom2014-W_v2"
WIND_TASK_IDS = tuple(range(1, 16))
WIND_ZONES = tuple(range(1, 11))

WIND_TARGET_COL = "TARGETVAR"
WIND_RAW_WEATHER_COLS = ["U10", "V10", "U100", "V100"]
_NUM_COLS = ["TARGETVAR", "U10", "V10", "U100", "V100"]


# ---------------------------------------------------------------
# 时间戳解析（Wind V2 格式，严格 YYYYMMDD H:MM）
# ---------------------------------------------------------------

def parse_wind_timestamp(ts_str: str) -> Optional[datetime]:
    """解析 "YYYYMMDD H:MM" → datetime。格式严格，无 Load 的变长歧义。"""
    try:
        date_str, time_str = ts_str.strip().split(" ")
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.replace(hour=int(time_str.split(":")[0]))
    except Exception:
        return None


def _read_zip_csv(zip_path: Path, member: str) -> pd.DataFrame:
    """从 zip 内读取单个 CSV 成员（不落盘）。"""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as f:
            return pd.read_csv(io.BytesIO(f.read()))


def _load_wind_frame(path: Path, zip_member: Optional[str] = None) -> pd.DataFrame:
    """读 Wind CSV（zip 内或裸文件），统一时间戳解析 + 数值化，datetime 索引。

    返回 DataFrame 含 ZONEID 列 + 数值列；时间列删除。
    """
    df = _read_zip_csv(path, zip_member) if zip_member else pd.read_csv(path)
    df["datetime"] = df["TIMESTAMP"].map(parse_wind_timestamp)
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

    Wind 数据存在极少量缺失小时（solution15 每分区 6~8/744；个别 train 文件
    少量），与仓库 preprocessing.fill_missing_values 的插值惯例一致。
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

def wind_task_paths(
    task_id: int, zone: int, data_dir: Path = WIND_DATA_DIR
) -> Dict[str, Path]:
    """返回某 Task×Zone 涉及的全部数据文件路径（zip 路径 + zip 内成员名）。"""
    if task_id not in WIND_TASK_IDS:
        raise ValueError(f"task_id 必须在 {WIND_TASK_IDS[0]}..{WIND_TASK_IDS[-1]}，got {task_id}")
    if zone not in WIND_ZONES:
        raise ValueError(f"zone 必须在 {WIND_ZONES[0]}..{WIND_ZONES[-1]}，got {zone}")
    data_dir = Path(data_dir)
    d = data_dir / f"Task {task_id}"
    return {
        "train_zip": d / f"Task{task_id}_W_Zone1_10.zip",
        "train_member": f"Task{task_id}_W_Zone1_10/Task{task_id}_W_Zone{zone}.csv",
        "expvars_zip": d / f"TaskExpVars{task_id}_W_Zone1_10.zip",
        "expvars_member": (
            f"TaskExpVars{task_id}_W_Zone1_10/TaskExpVars{task_id}_W_Zone{zone}.csv"
        ),
        "benchmark": d / f"benchmark{task_id}_W.csv",
        "solution": data_dir / "Solution to Task 15" / "solution15_W.csv",
    }


# ---------------------------------------------------------------
# 加载
# ---------------------------------------------------------------

def load_wind_train(
    task_id: int, zone: int, data_dir: Path = WIND_DATA_DIR
) -> pd.DataFrame:
    """加载 Task×Zone 的 train 历史（datetime 索引，含 TARGETVAR + U10/V10/U100/V100）。"""
    p = wind_task_paths(task_id, zone, data_dir)
    df = _load_wind_frame(p["train_zip"], p["train_member"])
    if "ZONEID" in df.columns:
        df = df[df["ZONEID"] == zone]
    return df


def load_wind_expvars(
    task_id: int, zone: int, data_dir: Path = WIND_DATA_DIR
) -> pd.DataFrame:
    """加载 Task×Zone 预测月气象预报（datetime 索引，U10/V10/U100/V100）。"""
    p = wind_task_paths(task_id, zone, data_dir)
    df = _load_wind_frame(p["expvars_zip"], p["expvars_member"])
    if "ZONEID" in df.columns:
        df = df[df["ZONEID"] == zone]
    return df


def load_wind_benchmark_ts(
    task_id: int, data_dir: Path = WIND_DATA_DIR
) -> pd.DatetimeIndex:
    """加载 Task 预测月逐小时时间戳（来自 benchmark 文件，10 分区时间戳一致）。"""
    p = wind_task_paths(task_id, 1, data_dir)
    df = pd.read_csv(p["benchmark"])
    ts = pd.DatetimeIndex(df["TIMESTAMP"].map(parse_wind_timestamp).dropna())
    return ts.sort_values().unique()


def load_wind_solution(data_dir: Path = WIND_DATA_DIR) -> pd.DataFrame:
    """加载 Task 15 官方真值（10 分区全，datetime 索引 + ZONEID + TARGETVAR）。"""
    p = wind_task_paths(15, 1, data_dir)
    return _load_wind_frame(p["solution"])


def load_wind_ground_truth(
    task_id: int, zone: int, data_dir: Path = WIND_DATA_DIR
) -> pd.DataFrame:
    """
    解析 Task 预测月的真实 TARGETVAR（以预测月时间戳为索引，单列）。

    - task 15：官方 solution15_W.csv（按 zone 过滤）
    - task k (1..14)：Task{k+1} train 的 TARGETVAR（已验证与 benchmark 时间戳逐行一致）

    任何预测小时缺失真值则抛错。
    """
    forecast_ts = load_wind_benchmark_ts(task_id, data_dir)

    if task_id == 15:
        sol = load_wind_solution(data_dir)
        sol_zone = sol[sol["ZONEID"] == zone] if "ZONEID" in sol else sol
        truth = sol_zone[WIND_TARGET_COL].reindex(forecast_ts)
    else:
        truth = load_wind_train(task_id + 1, zone, data_dir)[WIND_TARGET_COL].reindex(forecast_ts)

    truth = _fill_target_nan(truth, f"Wind Task {task_id} Zone {zone} 真值")
    return truth.to_frame(WIND_TARGET_COL)


# ---------------------------------------------------------------
# 可用性（镜像 data/availability.py 语义，但 Wind 单文件即完整前缀）
# ---------------------------------------------------------------

def wind_available_history(
    task_id: int, zone: int, data_dir: Path = WIND_DATA_DIR
) -> Availability:
    """
    构建 Task×Zone 的可用历史 + 预测区间。

    Wind 的 Task{k} train 文件本身就是"2012-01-01 → 预测月前一刻"的完整前缀，
    无需像 Load 那样拼接 L1..Lk。
    """
    hist = load_wind_train(task_id, zone, data_dir)

    # 个别 train 文件有少量 TARGETVAR 缺失小时，线性插值后仍缺则抛错
    hist = hist.copy()
    hist[WIND_TARGET_COL] = _fill_target_nan(
        hist[WIND_TARGET_COL], f"Wind Task {task_id} Zone {zone} 历史"
    )

    # 断言逐小时连续
    diffs = hist.index.to_series().diff().dropna()
    if not (diffs == pd.Timedelta(hours=1)).all():
        raise ValueError(f"Wind Task {task_id} Zone {zone} 历史时间不连续")

    forecast_ts = load_wind_benchmark_ts(task_id, data_dir)
    if forecast_ts[0] != hist.index.max() + pd.Timedelta(hours=1):
        raise ValueError(
            f"Wind Task {task_id} Zone {zone} 预测起点 {forecast_ts[0]} != 历史终点 + 1h "
            f"({hist.index.max() + pd.Timedelta(hours=1)})"
        )

    return Availability(
        task_id=task_id,
        history_df=hist,
        available_until=hist.index.max(),
        forecast_ts=forecast_ts,
    )


def get_wind_task_metadata(
    zones: List[int] = None, data_dir: Path = WIND_DATA_DIR
) -> pd.DataFrame:
    """
    产出 Wind Task×Zone 可用性元数据表（每 Task×Zone 一行）：
      task_id / zone / forecast_month / available_until / forecast_start / forecast_end
      n_history / n_forecast
    """
    if zones is None:
        zones = list(WIND_ZONES)
    rows = []
    for tid in WIND_TASK_IDS:
        for zone in zones:
            av = wind_available_history(tid, zone, data_dir)
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
