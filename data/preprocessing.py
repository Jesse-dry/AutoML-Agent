"""
GEFCom2014-L_V2 电力负荷数据预处理模块
==========================================
功能：
  1. 时间索引解析（处理变长 TIMESTAMP 格式）
  2. 缺失值填充（LOAD + 气象站温度）
  3. 时序切分（训练/验证/测试集）
  4. 人工基线特征构造（时间特征 + 滞后特征 + 滚动特征）
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 1. 时间戳解析
# ============================================================

def _get_all_candidates(ts_str: str) -> list:
    """
    返回一条 TIMESTAMP 的所有合法 datetime 解释（用于歧义检测）。

    尝试 4 位年份和 2 位年份两种假设，对每种假设尝试所有月/日拆分。
    返回所有合法 datetime，按「更可能正确」的启发式排序：
      1. 4 位年份优先于 2 位年份（长日期串中 4 位年更常见）
      2. 2 位月份优先于 1 位月份（MMDD 惯例）

    对于单条时间戳无法确定的歧义（如 "112001" 中 "2001" 是 4 位年
    还是 "20" + "01"=2位年），由调用方利用数据集时间范围做出最终选择。
    """
    try:
        parts = ts_str.strip().split(" ")
        if len(parts) < 2:
            return []

        date_str = parts[0]
        time_str = parts[1]
        hour = int(time_str.split(":")[0])

        all_candidates = []

        # --- 假设 1: 4 位年份 ---
        if len(date_str) >= 5:
            candidate_4y = int(date_str[-4:])
            candidate_md = date_str[:-4]
            if 2000 <= candidate_4y <= 2030 and 2 <= len(candidate_md) <= 4:
                for m_len in (2, 1):
                    if m_len > len(candidate_md):
                        continue
                    ms, ds = candidate_md[:m_len], candidate_md[m_len:]
                    if not ms or not ds or len(ds) > 2:
                        continue
                    m, d = int(ms), int(ds)
                    if 1 <= m <= 12 and 1 <= d <= 31:
                        try:
                            dt = datetime(candidate_4y, m, d, hour)
                            all_candidates.append(dt)
                        except ValueError:
                            pass

        # --- 假设 2: 2 位年份 ---
        candidate_2y = 2000 + int(date_str[-2:])
        candidate_md = date_str[:-2]
        if 2 <= len(candidate_md) <= 4:
            for m_len in (2, 1):
                if m_len > len(candidate_md):
                    continue
                ms, ds = candidate_md[:m_len], candidate_md[m_len:]
                if not ms or not ds or len(ds) > 2:
                    continue
                m, d = int(ms), int(ds)
                if 1 <= m <= 12 and 1 <= d <= 31:
                    try:
                        dt = datetime(candidate_2y, m, d, hour)
                        all_candidates.append(dt)
                    except ValueError:
                        pass

        # 去重（两种年份假设可能产生相同的日期）
        seen = set()
        unique = []
        for dt in all_candidates:
            if dt not in seen:
                seen.add(dt)
                unique.append(dt)

        return unique

    except Exception:
        return []


def _select_best_candidate(
    candidates: list, time_range: tuple = None,
    lower_bound: datetime = None, upper_bound: datetime = None,
) -> Optional[datetime]:
    """
    从多个候选 datetime 中选择最佳的一个。

    选择策略（按优先级）：
      1. 若给定 lower_bound / upper_bound（来自前后锚点行），
         筛选落在 [lower, upper] 内的候选，再按「到范围中心的距离」择优
      2. 若 time_range 给定，筛选落在范围内的候选，按中心距离择优；
         若都不在范围内，选离边界最近的
      3. 无约束 → 返回第一个（贪婪，4位年+2位月优先）
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # ---- 确定用于筛选和评分的范围 ----
    t_min, t_max = None, None

    if lower_bound is not None or upper_bound is not None:
        # 锚点约束：来自前后无歧义行的时序单调性
        t_min = lower_bound if lower_bound is not None else datetime(1990, 1, 1)
        t_max = upper_bound if upper_bound is not None else datetime(2040, 1, 1)
    elif time_range is not None:
        t_min, t_max = time_range

    if t_min is not None and t_max is not None:
        span = (t_max - t_min).total_seconds()
        if span <= 0:
            span = 3600

        # Step A: 筛选落在范围内的候选
        in_range = [dt for dt in candidates if t_min <= dt <= t_max]

        if in_range:
            # 有落在范围内的 → 选离范围中心最近的
            center = t_min + (t_max - t_min) / 2
            best_dt = in_range[0]
            best_dist = float("inf")
            for dt in in_range:
                dist = abs((dt - center).total_seconds()) / span
                if dist < best_dist:
                    best_dist = dist
                    best_dt = dt
            return best_dt

        # Step B: 都不在范围内 → 选离最近边界最近的
        best_dt = candidates[0]
        best_dist = float("inf")
        for dt in candidates:
            if dt < t_min:
                dist = (t_min - dt).total_seconds() / span
            else:
                dist = (dt - t_max).total_seconds() / span
            if dist < best_dist:
                best_dist = dist
                best_dt = dt
        return best_dt

    # 无约束 → 贪婪
    return candidates[0]


def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    解析 GEFCom2014 的变长 TIMESTAMP 格式（贪婪版本）。

    支持的格式：
      - MMDDYY H:MM    （如 112001 1:00  → 2001-11-20 01:00）
      - MMDDYYYY H:MM  （如 9302010 20:00 → 2010-09-30 20:00）
      - MMDDYY HH:MM   （如 1112011 1:00  → 2011-11-01 01:00）

    生成所有合法候选（4位年/2位年 × 月1~2位），返回贪婪最优：
    4位年优先 → 2位月优先。对歧义场景（如 "112001" 中 "2001"
    可能是 4位年或 "20"+2位年"01"），不做二次消歧。
    批量解析建议通过 load_single_task() 享受自动消歧。
    """
    candidates = _get_all_candidates(ts_str)
    return candidates[0] if candidates else None


def parse_timestamp_with_context(
    ts_str: str, time_range: tuple = None
) -> Optional[datetime]:
    """
    结合数据集已知时间范围，消除歧义的时间戳解析。

    当一条 TIMESTAMP 有多个合法解释时（如 "112001 1:00" 既是
    Jan 1, 2001（4位年假说）又是 Nov 20, 2001（2位年假说）），
    利用已知时间范围选择与上下文最一致的。

    Parameters
    ----------
    ts_str : str
        原始 TIMESTAMP 字符串
    time_range : tuple of (min_datetime, max_datetime), optional
        数据集的已知时间范围。

    Returns
    -------
    datetime or None
    """
    candidates = _get_all_candidates(ts_str)
    return _select_best_candidate(candidates, time_range)


def parse_solution_timestamp(date_str: str, hour: int) -> datetime:
    """
    解析 Solution 文件的日期格式: MM/DD/YYYY + hour
    例: 12/1/2011, 1 → 2011-12-01 01:00
    """
    month, day, year = map(int, date_str.split("/"))
    return datetime(year, month, day, hour)


# ============================================================
# 2. 数据加载
# ============================================================

def load_single_task(train_path: str, benchmark_path: str = None) -> pd.DataFrame:
    """
    加载单个 Task 的训练数据。

    两步时间戳解析（解决变长格式歧义）：
      Pass 1 — 找出无歧义的行（只有 1 个合法候选），用它们建立
               数据集的时间范围。歧义行暂用贪婪选择。
      Pass 2 — 用 Pass 1 的干净时间范围，对所有行重新选择最佳候选，
               消除两种歧义：
                 ① 年份歧义："2001" 是 4 位年份还是 "20"+2位年"01"
                 ② 月日歧义：md_part="111" → 11月1日 还是 1月11日

    Parameters
    ----------
    train_path : str
        训练 CSV 文件路径
    benchmark_path : str, optional
        Benchmark CSV 文件路径（用于获取预测时间范围）

    Returns
    -------
    pd.DataFrame
        包含解析后时间索引的 DataFrame
    """
    df = pd.read_csv(train_path)

    # 获取每条 TIMESTAMP 的所有合法候选 datetime
    all_candidates = df["TIMESTAMP"].apply(_get_all_candidates)
    n_candidates = all_candidates.apply(len)

    # ---- Pass 1: 用无歧义行建立时间范围 ----
    unambiguous_mask = n_candidates == 1
    n_unambiguous = unambiguous_mask.sum()
    n_ambiguous = (n_candidates > 1).sum()

    if n_unambiguous > 0:
        # 只用无歧义行建立干净的时间范围
        unambiguous_dt = all_candidates[unambiguous_mask].apply(
            lambda cands: cands[0]
        )
        t_min = unambiguous_dt.min()
        t_max = unambiguous_dt.max()
        from datetime import timedelta
        # 适当扩展以容纳歧义行可能的日期偏移
        time_range = (
            t_min - timedelta(days=365),
            t_max + timedelta(days=365),
        )
        if n_ambiguous > 0:
            print(
                f"  [INFO] {Path(train_path).name}: "
                f"{n_unambiguous} 行无歧义 → 时间范围 [{t_min.date()}, {t_max.date()}]; "
                f"{n_ambiguous} 行有歧义待消解"
            )
    else:
        # 极端情况：所有行都有歧义，用贪婪选择兜底
        time_range = None
        print(
            f"  [WARNING] {Path(train_path).name}: "
            f"所有行都有歧义，无法建立可靠时间范围"
        )

    # ---- Pass 2: 块级消歧 ----
    # 核心思路：歧义行通常以连续「块」出现（如同一天的24小时），
    # 块内所有行应做出一致的选择。对每个歧义块，尝试每种候选解释，
    # 选择使"块到前后锚点的时序间隔最小"的解释。
    if time_range is not None and n_ambiguous > 0:
        # 建立锚点列表：(行位置, datetime)
        anchor_positions = []  # (iloc_position, datetime)
        for i in range(len(df)):
            if unambiguous_mask.iloc[i]:
                anchor_positions.append((i, all_candidates.iloc[i][0]))

        # 识别歧义块（连续的歧义行）
        amb_blocks = []  # list of (start_iloc, end_iloc)
        i = 0
        while i < len(df):
            if n_candidates.iloc[i] > 1:
                start = i
                while i < len(df) and n_candidates.iloc[i] > 1:
                    i += 1
                amb_blocks.append((start, i - 1))
            else:
                i += 1

        # 对每个歧义块，用锚点约束做块级消歧
        for block_start, block_end in amb_blocks:
            # 找到前后最近的锚点
            before_anchor = None  # 在块之前的最近锚点
            after_anchor = None   # 在块之后的最近锚点

            for pos, dt in anchor_positions:
                if pos < block_start:
                    before_anchor = dt  # 不断更新为更近的
                elif pos > block_end:
                    after_anchor = dt
                    break  # 第一个块后的锚点就是最近的

            # 该块内任取一行，获取候选数量（块内所有行候选数相同）
            n_cands = n_candidates.iloc[block_start]

            # 对每个候选索引 k（0=贪婪，1=备选），评估整块的拟合度
            best_k = 0
            best_gap = float("inf")

            for k in range(n_cands):
                # 取该块第一行和最后一行的第 k 个候选
                block_first = all_candidates.iloc[block_start][k]
                block_last = all_candidates.iloc[block_end][k]

                # 计算到锚点的时序间隔
                total_gap = 0.0

                if before_anchor is not None:
                    if block_first < before_anchor:
                        total_gap += float("inf")  # 违反时序单调性
                    else:
                        total_gap += (
                            block_first - before_anchor
                        ).total_seconds()

                if after_anchor is not None:
                    if block_last > after_anchor:
                        total_gap += float("inf")  # 违反时序单调性
                    else:
                        total_gap += (
                            after_anchor - block_last
                        ).total_seconds()

                if total_gap < best_gap:
                    best_gap = total_gap
                    best_k = k

            # 将该块所有行的候选替换为只有最佳选择
            for idx in range(block_start, block_end + 1):
                all_candidates.iloc[idx] = [all_candidates.iloc[idx][best_k]]

        # 最终选择：所有行取第一个（也是唯一一个，歧义行已被消解为单元素）
        df["datetime"] = all_candidates.apply(
            lambda cands: cands[0] if cands else None
        )

        n_resolved = sum(
            1 for start, end in amb_blocks
        )
        if n_resolved > 0:
            print(
                f"  [INFO] {Path(train_path).name}: "
                f"块级消歧处理了 {n_resolved} 个歧义块"
            )
    else:
        df["datetime"] = all_candidates.apply(
            lambda cands: _select_best_candidate(
                cands, time_range
            ) if cands else None
        )

    # 统计最终解析结果
    n_failed = df["datetime"].isna().sum()
    if n_failed > 0:
        print(
            f"  [WARNING] {Path(train_path).name}: "
            f"{n_failed} 行时间戳解析失败"
        )

    # 剔除解析失败的行
    df = df.dropna(subset=["datetime"]).copy()

    # 设置时间索引
    df = df.set_index("datetime").sort_index()

    # 将 LOAD 转为数值（空字符串 → NaN）
    df["LOAD"] = pd.to_numeric(df["LOAD"], errors="coerce")

    # 将 w1~w25 转为数值
    w_cols = [f"w{i}" for i in range(1, 26)]
    for c in w_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 删除原始的 ZONEID 和 TIMESTAMP 列（保留 ZONEID 用于分组）
    if "TIMESTAMP" in df.columns:
        df = df.drop(columns=["TIMESTAMP"])

    return df


def load_all_tasks(data_dir: str) -> Dict[int, pd.DataFrame]:
    """
    加载所有 15 个 Task 的数据。

    Returns
    -------
    dict: {task_id: DataFrame}
    """
    data_dir = Path(data_dir)
    tasks = {}

    for task_id in range(1, 16):
        train_path = data_dir / f"Task {task_id}" / f"L{task_id}-train.csv"
        if train_path.exists():
            print(f"Loading Task {task_id}...")
            df = load_single_task(str(train_path))
            df["task_id"] = task_id
            tasks[task_id] = df
            print(f"  -> {len(df)} 行, "
                  f"时间范围: {df.index.min()} ~ {df.index.max()}, "
                  f"LOAD 缺失: {df['LOAD'].isna().sum()}")
        else:
            print(f"Task {task_id}: 文件不存在 ({train_path})")

    return tasks


def load_solution(solution_path: str) -> pd.DataFrame:
    """
    加载 Solution 文件 (solution15_L_temperature.csv)。

    格式: date, hour, LOAD, w1~w25
    """
    df = pd.read_csv(solution_path)
    df["datetime"] = df.apply(
        lambda row: parse_solution_timestamp(str(row["date"]), int(row["hour"])),
        axis=1,
    )
    df = df.set_index("datetime").sort_index()
    df["LOAD"] = pd.to_numeric(df["LOAD"], errors="coerce")

    w_cols = [f"w{i}" for i in range(1, 26)]
    for c in w_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ============================================================
# 3. 缺失值填充
# ============================================================

def fill_missing_values(
    df: pd.DataFrame,
    load_method: str = "interpolate",
    weather_method: str = "interpolate",
    interpolation_limit: int = 24,
) -> pd.DataFrame:
    """
    填充缺失值。

    Parameters
    ----------
    df : pd.DataFrame
        原始数据（以 datetime 为索引）
    load_method : str
        LOAD 填充方式:
        - "interpolate": 线性插值
        - "forward": 前向填充
        - "drop": 删除缺失行
    weather_method : str
        气象数据填充方式（同上）
    interpolation_limit : int
        插值时最大连续缺失数（超过则不填充）

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    w_cols = [f"w{i}" for i in range(1, 26)]

    # --- 填充 LOAD ---
    n_missing_load = df["LOAD"].isna().sum()
    if n_missing_load > 0:
        print(f"  LOAD 缺失: {n_missing_load} / {len(df)} ({100*n_missing_load/len(df):.1f}%)")

        if load_method == "drop":
            df = df.dropna(subset=["LOAD"])
        elif load_method == "interpolate":
            df["LOAD"] = df["LOAD"].interpolate(
                method="linear",
                limit=interpolation_limit,
                limit_direction="both",
            )
            # 对仍然缺失的（超过 limit 的连续段），用前向/后向填充兜底
            df["LOAD"] = df["LOAD"].ffill().bfill()
        elif load_method == "forward":
            df["LOAD"] = df["LOAD"].ffill().bfill()

        remaining = df["LOAD"].isna().sum()
        print(f"    填充后剩余缺失: {remaining}")

    # --- 填充气象数据 ---
    for c in w_cols:
        n_missing = df[c].isna().sum()
        if n_missing > 0:
            if weather_method == "interpolate":
                df[c] = df[c].interpolate(
                    method="linear",
                    limit=interpolation_limit,
                    limit_direction="both",
                )
                df[c] = df[c].ffill().bfill()
            elif weather_method == "forward":
                df[c] = df[c].ffill().bfill()

    # 最终检查
    total_missing = df["LOAD"].isna().sum() + sum(df[c].isna().sum() for c in w_cols)
    if total_missing > 0:
        print(f"  [WARNING] 最终仍有 {total_missing} 个缺失值，将删除这些行")
        df = df.dropna(subset=["LOAD"] + w_cols)

    return df


# ============================================================
# 4. 时序切分
# ============================================================

def split_timeseries(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    method: str = "sequential",
    gap: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    时序切分（保持时间顺序）。

    Parameters
    ----------
    df : pd.DataFrame
        已排序的时序数据
    train_ratio, val_ratio, test_ratio : float
        三个集合的比例（相加应为 1.0）
    method : str
        - "sequential": 简单按比例顺序切分（默认）
        - "yearly": 按年份切分
    gap : int
        train/val 之间的间隔（小时），防止数据泄露

    Returns
    -------
    (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio) + gap

    if method == "sequential":
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end + gap : val_end].copy()
        test_df = df.iloc[val_end:].copy()

    elif method == "yearly":
        # 按年份切分：用最后两年做 val/test
        years = sorted(df.index.year.unique())
        if len(years) >= 3:
            train_years = years[:-2]
            val_year = years[-2]
            test_year = years[-1]
            train_df = df[df.index.year.isin(train_years)].copy()
            val_df = df[df.index.year == val_year].copy()
            test_df = df[df.index.year == test_year].copy()
        else:
            # 年份不够，回退到 sequential
            print("  [WARNING] 年份不足3年，回退到 sequential 切分")
            return split_timeseries(df, train_ratio, val_ratio, test_ratio, "sequential", gap)

    print(f"  Train: {train_df.index.min()} ~ {train_df.index.max()} ({len(train_df)} 行)")
    print(f"  Val:   {val_df.index.min()} ~ {val_df.index.max()} ({len(val_df)} 行)")
    print(f"  Test:  {test_df.index.min()} ~ {test_df.index.max()} ({len(test_df)} 行)")

    return train_df, val_df, test_df


# ============================================================
# 5. 基线特征工程
# ============================================================

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造人工基线特征集。

    特征清单：
    ┌──────────────────────┬─────────────────────────────────┐
    │ 类别                 │ 特征                            │
    ├──────────────────────┼─────────────────────────────────┤
    │ 时间特征             │ hour, weekday, month, is_weekend│
    │ 负荷滞后 (Lag)       │ lag_1, lag_24, lag_168          │
    │ 温度基础             │ temp (w1), temp_lag_24           │
    │ 负荷滚动统计         │ rolling_mean_24_load            │
    └──────────────────────┴─────────────────────────────────┘

    Parameters
    ----------
    df : pd.DataFrame
        包含 LOAD 和 w1~w25 列的时序数据（以 datetime 为索引）

    Returns
    -------
    pd.DataFrame
        原数据 + 特征列
    """
    df = df.copy()

    # ---- 5.1 时间特征 ----
    df["hour"] = df.index.hour.astype(np.int32)
    df["weekday"] = df.index.weekday.astype(np.int32)  # 0=Mon, 6=Sun
    df["month"] = df.index.month.astype(np.int32)
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int32)

    # ---- 5.2 负荷滞后特征 ----
    df["lag_1"] = df["LOAD"].shift(1)        # 1小时前
    df["lag_24"] = df["LOAD"].shift(24)      # 24小时前（昨日同时）
    df["lag_168"] = df["LOAD"].shift(168)    # 168小时前（上周同时）

    # ---- 5.3 温度基础特征 ----
    # 使用 w1（第一个气象站）作为温度代表
    df["temp"] = df["w1"]
    df["temp_lag_24"] = df["w1"].shift(24)

    # 也可以构造多站均值/极值（增强版本）
    w_cols = [f"w{i}" for i in range(1, 26)]
    df["temp_mean"] = df[w_cols].mean(axis=1)
    df["temp_min"] = df[w_cols].min(axis=1)
    df["temp_max"] = df[w_cols].max(axis=1)
    df["temp_std"] = df[w_cols].std(axis=1)

    # ---- 5.4 负荷滚动统计 ----
    df["rolling_mean_24_load"] = df["LOAD"].rolling(window=24, min_periods=1).mean()
    df["rolling_std_24_load"] = df["LOAD"].rolling(window=24, min_periods=1).std()
    df["rolling_mean_168_load"] = df["LOAD"].rolling(window=168, min_periods=1).mean()

    # ---- 5.5 温度滚动统计 ----
    df["rolling_mean_24_temp"] = df["temp_mean"].rolling(window=24, min_periods=1).mean()

    # ---- 清理：删除因 shift/rolling 产生的 NaN 行（可选）----
    # 注意：lag_168 会导致前168行为 NaN，根据需求决定是否 drop
    # df = df.dropna()  # 如需严格无 NaN，取消注释

    return df


# ============================================================
# 6. 完整预处理流水线
# ============================================================

def preprocess_pipeline(
    data_dir: str,
    task_id: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    fill_load: str = "interpolate",
    fill_weather: str = "interpolate",
    split_method: str = "sequential",
    dropna_features: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    一站式预处理流水线。

    Parameters
    ----------
    data_dir : str
        GEFCom2014-L_V2/Load 目录路径
    task_id : int
        要处理的任务编号 (1-15)
    train_ratio, val_ratio, test_ratio : float
        训练/验证/测试比例
    fill_load : str
        LOAD 缺失值填充方式
    fill_weather : str
        气象数据缺失值填充方式
    split_method : str
        切分方式
    dropna_features : bool
        是否删除因 lag/rolling 导致的 NaN 行

    Returns
    -------
    dict: {
        "train": DataFrame (含所有特征),
        "val": DataFrame,
        "test": DataFrame,
        "raw": 原始完整数据,
        "feature_cols": 特征列名列表,
        "target_col": 目标列名,
    }
    """
    data_dir = Path(data_dir)

    # Step 1: 加载
    print("=" * 60)
    print(f"Step 1: 加载 Task {task_id} 数据")
    train_path = data_dir / f"Task {task_id}" / f"L{task_id}-train.csv"
    df = load_single_task(str(train_path))
    print(f"  原始数据: {len(df)} 行")

    # Step 2: 填充缺失值
    print(f"\nStep 2: 缺失值填充")
    df = fill_missing_values(df, load_method=fill_load, weather_method=fill_weather)
    print(f"  填充后: {len(df)} 行")

    # Step 3: 构造特征
    print(f"\nStep 3: 构造基线特征")
    df = build_baseline_features(df)

    # 定义特征列和目标列
    target_col = "LOAD"
    base_feature_cols = [
        # 时间特征
        "hour", "weekday", "month", "is_weekend",
        # 负荷滞后
        "lag_1", "lag_24", "lag_168",
        # 温度基础
        "temp", "temp_lag_24",
        "temp_mean", "temp_min", "temp_max", "temp_std",
        # 负荷滚动
        "rolling_mean_24_load", "rolling_std_24_load", "rolling_mean_168_load",
        # 温度滚动
        "rolling_mean_24_temp",
    ]
    weather_cols = [f"w{i}" for i in range(1, 26)]

    # 确保所有特征列存在
    feature_cols = [c for c in base_feature_cols if c in df.columns]

    print(f"  特征数: {len(feature_cols)}, 目标列: {target_col}")

    # Step 4: 处理 NaN 特征行
    if dropna_features:
        before = len(df)
        df = df.dropna(subset=feature_cols)
        after = len(df)
        if before > after:
            print(f"  删除含 NaN 特征的行: {before} → {after}")

    # Step 5: 时序切分
    print(f"\nStep 4: 时序切分 ({split_method})")
    train_df, val_df, test_df = split_timeseries(
        df, train_ratio, val_ratio, test_ratio, method=split_method
    )

    print(f"\n{'=' * 60}")
    print(f"预处理完成!")
    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")
    print(f"  特征列: {feature_cols}")

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "raw": df,
        "feature_cols": feature_cols,
        "weather_cols": weather_cols,
        "target_col": target_col,
    }


# ============================================================
# 7. 主入口（演示 / 测试）
# ============================================================

if __name__ == "__main__":
    # 数据目录
    DATA_DIR = Path(__file__).parent.parent / "GEFCom2014-L_V2" / "Load"

    print("GEFCom2014-L_V2 数据预处理\n")
    print(f"数据目录: {DATA_DIR}\n")

    # ---- 演示：解析时间戳 ----
    print("=" * 60)
    print("[测试] 时间戳解析")
    test_timestamps = [
        "112001 1:00",
        "122001 0:00",
        "9302010 20:00",
        "1012010 0:00",
        "1112011 1:00",
        "1212011 1:00",
    ]
    for ts in test_timestamps:
        result = parse_timestamp(ts)
        print(f"  '{ts}' → {result}")

    # ---- 演示：加载单个 Task ----
    print(f"\n{'=' * 60}")
    print("[测试] 加载 Task 1")

    # 先看各 Task 的时间范围
    for tid in [1, 5, 10, 15]:
        train_path = DATA_DIR / f"Task {tid}" / f"L{tid}-train.csv"
        if train_path.exists():
            df = load_single_task(str(train_path))
            load_na = df["LOAD"].isna().sum()
            load_ok = len(df) - load_na
            print(f"  Task {tid}: {len(df)} 行, "
                  f"{df.index.min()} ~ {df.index.max()}, "
                  f"LOAD有效={load_ok}, LOAD缺失={load_na}")

    # ---- 完整流水线 ----
    print(f"\n{'=' * 60}")
    print("[演示] Task 15 完整预处理流水线")

    result = preprocess_pipeline(
        data_dir=str(DATA_DIR),
        task_id=15,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        fill_load="interpolate",
        fill_weather="interpolate",
        split_method="sequential",
        dropna_features=True,
    )

    # 展示结果
    for name in ["train", "val", "test"]:
        df = result[name]
        print(f"\n--- {name} 集预览 (前3行) ---")
        cols_to_show = ["LOAD"] + result["feature_cols"]
        print(df[cols_to_show].head(3).to_string())

    print(f"\n[完成] 数据已准备好，可用于模型训练。")
