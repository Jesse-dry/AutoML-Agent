# GEFCom2014 Load 数据统一加载器
# ---------------------------------------------------------------
# 职责：把 Task 1–15 的 train / benchmark / solution 文件统一加载为
# datetime 索引的 DataFrame，并解析每个 Task 预测月的"真值"。
#
# 数据约定（GEFCom2014 Load 赛道）：
#   - Task {n}/L{n}-train.csv   该 task 的增量历史（L1 = 全量历史，L2–L15 = 仅前一个月）
#   - Task {n}/L{n}-benchmark.csv  预测月模板（99 分位），其时间戳即预测月 —— 常量值
#                              是 naive 基准（去年同月），不是真值
#   - Solution to Task 15/solution15_L.csv  Task 15 预测月的官方真值 LOAD
#
# 真值来源：
#   - k = 1..14：L{k+1}-train.csv 的 LOAD 列 == task k 预测月的真实负荷
#     （L{k+1}-train 的时间戳与 L{k}-benchmark 逐行一致，已验证）
#   - k = 15：官方 solution15_L.csv
# ---------------------------------------------------------------
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from data.preprocessing import load_single_task

GEFCOM_DATA_DIR = Path(__file__).resolve().parent.parent / "GEFCom2014-L_V2" / "Load"
TASK_IDS = tuple(range(1, 16))


def task_paths(data_dir: Path, task_id: int) -> Dict[str, Path]:
    """返回某 Task 涉及的全部数据文件路径。"""
    if task_id not in TASK_IDS:
        raise ValueError(f"task_id 必须在 {TASK_IDS[0]}..{TASK_IDS[-1]}，got {task_id}")
    data_dir = Path(data_dir)
    paths = {
        "train": data_dir / f"Task {task_id}" / f"L{task_id}-train.csv",
        "benchmark": data_dir / f"Task {task_id}" / f"L{task_id}-benchmark.csv",
    }
    if task_id == 15:
        paths["solution"] = data_dir / "Solution to Task 15" / "solution15_L.csv"
        paths["solution_temperature"] = (
            data_dir / "Solution to Task 15" / "solution15_L_temperature.csv"
        )
    return paths


def load_train(task_id: int, data_dir: Path = GEFCOM_DATA_DIR) -> pd.DataFrame:
    """加载 Task 的增量训练历史（datetime 索引，含 LOAD + w1..w25）。"""
    return load_single_task(str(task_paths(data_dir, task_id)["train"]))


def load_benchmark(task_id: int, data_dir: Path = GEFCOM_DATA_DIR) -> pd.DataFrame:
    """
    加载 Task 的预测月模板。返回 datetime 索引 DataFrame；
    通常只消费 `.index`（即预测月逐小时时间戳）。
    """
    return load_single_task(str(task_paths(data_dir, task_id)["benchmark"]))


def load_solution(task_id: int, data_dir: Path = GEFCOM_DATA_DIR) -> Optional[pd.DataFrame]:
    """
    加载 Task 的官方真值。仅 Task 15 有 solution 文件，其余返回 None。
    """
    if task_id != 15:
        return None
    path = task_paths(data_dir, task_id).get("solution")
    if path is None or not path.exists():
        return None
    return load_single_task(str(path))


def load_ground_truth(task_id: int, data_dir: Path = GEFCOM_DATA_DIR) -> pd.DataFrame:
    """
    解析 Task 预测月的真实 LOAD，以预测月时间戳为索引。

    - task 15：官方 solution15_L.csv
    - task k (1..14)：L{k+1}-train.csv 的 LOAD（已验证与 benchmark 时间戳逐行一致）

    返回单列 DataFrame（列名 LOAD）。任何预测小时缺失真值则抛错。
    """
    forecast_ts = load_benchmark(task_id, data_dir).index

    if task_id == 15:
        sol = load_solution(15, data_dir)
        if sol is None:
            raise FileNotFoundError(f"Task 15 solution 文件缺失: {task_paths(data_dir, 15)}")
        truth = sol["LOAD"]
    else:
        truth = load_train(task_id + 1, data_dir)["LOAD"]

    y_true = truth.reindex(forecast_ts)
    if y_true.isna().any():
        n_missing = int(y_true.isna().sum())
        raise ValueError(f"Task {task_id} 预测月真值缺失 {n_missing} 小时")
    return y_true.to_frame("LOAD")
