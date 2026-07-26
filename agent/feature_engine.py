"""
特征执行引擎（Feature Execution Engine）
========================================

这是 AutoML Agent 的特征工程基础层——所有特征生成能力用纯 Python 函数实现。
LLM Agent 只负责「决策生成什么特征、传什么参数」，不负责写代码。

核心理念：
  - 每个函数都是确定性、可复现的变换
  - 输入原始 DataFrame → 输出加入新特征后的 DataFrame
  - 函数之间独立、可自由组合
  - 统一的参数校验 + 清晰的报错信息，方便 LLM 纠错重试

模块结构：
  - generate_lag_features      — 滞后特征
  - generate_rolling_features  — 滚动窗口统计
  - generate_time_features     — 时间特征（从 datetime 列提取）
  - generate_cross_features    — 交叉特征（两列运算）
  - generate_all_features      — 一键批量生成（便捷函数）
  - describe_new_columns       — 描述新增列（供 LLM 了解生成了什么）

用法：
    from agent.feature_engine import (
        generate_lag_features,
        generate_rolling_features,
        generate_time_features,
        generate_cross_features,
        generate_all_features,
    )

    df = generate_lag_features(df, "LOAD", [1, 2, 24, 168])
    df = generate_rolling_features(df, "LOAD", [6, 24], stats=["mean", "std"])
    df = generate_time_features(df, "datetime")
    df = generate_cross_features(df, "temp", "LOAD", "multiply")
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================
# 1. 滞后特征（Lag Features）
# ============================================================

def generate_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_list: List[int],
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    为指定列生成滞后（lag）特征。

    对于电力负荷预测，lag=24 表示「昨天同一时刻」，
    lag=168 表示「上周同一时刻」。

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame，数据按时间顺序排列
    target_col : str
        要生成滞后的目标列名
    lag_list : list of int
        滞后步数列表，如 [1, 2, 24, 168]。
        所有值必须 ≥ 1。
    group_col : str, optional
        分组列名。若提供，则在每个分组内独立计算滞后
        （如按 "station_id" 分组，避免跨站点串数据）

    Returns
    -------
    pd.DataFrame
        在原 DataFrame 基础上追加 lag 列，
        列名格式: "{target_col}_lag_{k}"。
        滞后产生的 NaN 保留（调用方可自行 dropna / ffill）。

    Raises
    ------
    ValueError
        lag_list 为空、包含非正整数、或 target_col 不存在
    """
    # ---- 参数校验 ----
    if target_col not in df.columns:
        raise ValueError(
            f"目标列 '{target_col}' 在 DataFrame 中不存在。"
            f"当前列: {list(df.columns)}"
        )
    if not lag_list:
        raise ValueError("lag_list 不能为空")
    if not all(isinstance(k, int) and k >= 1 for k in lag_list):
        invalid = [k for k in lag_list if not (isinstance(k, int) and k >= 1)]
        raise ValueError(f"lag_list 中所有值必须 ≥1 的整数，非法值: {invalid}")

    df_out = df.copy()

    # ---- 生成滞后 ----
    for k in lag_list:
        col_name = f"{target_col}_lag_{k}"
        if col_name in df_out.columns:
            warnings.warn(f"列 '{col_name}' 已存在，将被覆盖")

        if group_col is not None:
            if group_col not in df_out.columns:
                raise ValueError(f"分组列 '{group_col}' 不存在")
            df_out[col_name] = df_out.groupby(group_col)[target_col].shift(k)
        else:
            df_out[col_name] = df_out[target_col].shift(k)

    return df_out


# ============================================================
# 2. 滚动窗口统计特征（Rolling Window Statistics）
# ============================================================

def generate_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    window_list: List[int],
    stats: List[str] = None,
    center: bool = False,
    min_periods: Optional[int] = None,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    为指定列生成滚动窗口统计特征。

    例如 window=6 表示用过去 6 个时间步做统计，
    window=24 表示过去一天（小时级数据）的统计。

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame，数据按时间顺序排列
    target_col : str
        目标列名
    window_list : list of int
        滚动窗口大小列表，如 [6, 12, 24]。
        所有值必须 ≥ 2（窗口为 1 无意义）
    stats : list of str, optional
        统计量列表，默认 ['mean', 'std', 'max', 'min']。
        支持: 'mean', 'std', 'var', 'max', 'min', 'median',
              'sum', 'skew', 'kurt'
    center : bool
        是否居中窗口（默认 False = 只看过去）
    min_periods : int, optional
        窗口内最少非空值数量。默认等于窗口大小（即不计算不完整窗口）
    group_col : str, optional
        分组列名。若提供，则在每个分组内独立计算滚动统计

    Returns
    -------
    pd.DataFrame
        追加滚动统计列，列名格式: "{target_col}_rolling_{window}_{stat}"。
        窗口初期产生的 NaN 保留。

    Raises
    ------
    ValueError
        参数不合法时抛出
    """
    # ---- 默认值 ----
    if stats is None:
        stats = ["mean", "std", "max", "min"]

    # ---- 参数校验 ----
    if target_col not in df.columns:
        raise ValueError(
            f"目标列 '{target_col}' 在 DataFrame 中不存在。"
            f"当前列: {list(df.columns)}"
        )
    if not window_list:
        raise ValueError("window_list 不能为空")
    if not all(isinstance(w, int) and w >= 2 for w in window_list):
        invalid = [w for w in window_list if not (isinstance(w, int) and w >= 2)]
        raise ValueError(f"window_list 中所有值必须 ≥2 的整数，非法值: {invalid}")

    valid_stats = {"mean", "std", "var", "max", "min", "median", "sum", "skew", "kurt"}
    invalid_stats = [s for s in stats if s not in valid_stats]
    if invalid_stats:
        raise ValueError(
            f"不支持的统计量: {invalid_stats}。支持: {sorted(valid_stats)}"
        )

    # ---- 统计量 → pandas rolling method 映射 ----
    stat_method_map = {
        "mean": "mean",
        "std": "std",
        "var": "var",
        "max": "max",
        "min": "min",
        "median": "median",
        "sum": "sum",
        "skew": "skew",
        "kurt": "kurt",
    }

    df_out = df.copy()

    # ---- 确定滚动对象 ----
    if group_col is not None:
        if group_col not in df_out.columns:
            raise ValueError(f"分组列 '{group_col}' 不存在")
        roller_factory = lambda w: df_out.groupby(group_col)[target_col].rolling(
            window=w, center=center, min_periods=min_periods or w
        )
    else:
        roller_factory = lambda w: df_out[target_col].rolling(
            window=w, center=center, min_periods=min_periods or w
        )

    # ---- 生成滚动统计 ----
    for w in window_list:
        roller = roller_factory(w)
        for stat in stats:
            col_name = f"{target_col}_rolling_{w}_{stat}"
            if col_name in df_out.columns:
                warnings.warn(f"列 '{col_name}' 已存在，将被覆盖")

            method = stat_method_map[stat]

            if group_col is not None:
                # groupby rolling 返回 MultiIndex，需要 reset
                rolled = getattr(roller, method)()
                # 将 MultiIndex 的最后一个 level (原始 index) 对齐回去
                df_out[col_name] = rolled.reset_index(level=0, drop=True)
            else:
                df_out[col_name] = getattr(roller, method)()

    return df_out


# ============================================================
# 3. 时间特征（Time Features）
# ============================================================

def generate_time_features(
    df: pd.DataFrame,
    time_col: str,
    features: List[str] = None,
    cyclical: bool = True,
) -> pd.DataFrame:
    """
    从 datetime 列中提取时间特征。

    生成的特征包括：
      基础: year, month, day, dayofweek, dayofyear,
            hour, minute, quarter, weekofyear
      布尔标记: is_weekend, is_month_start, is_month_end
      周期性编码: hour_sin, hour_cos, month_sin, month_cos,
                 dayofweek_sin, dayofweek_cos
      （周期性编码将循环时间映射到单位圆，让模型理解 23 点离 0 点很近）

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame
    time_col : str
        datetime 类型的列名。如果不是 datetime，会自动尝试 pd.to_datetime 转换
    features : list of str, optional
        要生成的特征列表。默认全部生成。
        可选值见上方「基础」和「布尔标记」列表，
        周期性编码由 `cyclical` 参数独立控制
    cyclical : bool
        是否生成周期性编码（sin/cos）。默认 True

    Returns
    -------
    pd.DataFrame
        追加时间特征列

    Raises
    ------
    ValueError
        time_col 不存在、无法转为 datetime、或 features 中有不支持的值
    """
    # ---- 参数校验 ----
    if time_col not in df.columns:
        raise ValueError(
            f"时间列 '{time_col}' 在 DataFrame 中不存在。"
            f"当前列: {list(df.columns)}"
        )

    df_out = df.copy()

    # ---- 确保 datetime 类型 ----
    if not pd.api.types.is_datetime64_any_dtype(df_out[time_col]):
        try:
            df_out[time_col] = pd.to_datetime(df_out[time_col])
        except Exception as e:
            raise ValueError(
                f"无法将列 '{time_col}' 转为 datetime 类型: {e}"
            ) from e

    dt = df_out[time_col]

    # ---- 默认特征集 ----
    all_basic_features = [
        "year", "month", "day", "dayofweek", "dayofyear",
        "hour", "minute", "quarter", "weekofyear",
    ]
    all_bool_features = [
        "is_weekend", "is_month_start", "is_month_end",
    ]

    if features is None:
        features = all_basic_features + all_bool_features

    # ---- 校验 features ----
    valid_all = set(all_basic_features + all_bool_features)
    invalid = [f for f in features if f not in valid_all]
    if invalid:
        raise ValueError(
            f"不支持的时间特征: {invalid}。支持: {sorted(valid_all)}"
        )

    # ---- 生成基础特征 ----
    feature_extractors = {
        "year": lambda d: d.dt.year,
        "month": lambda d: d.dt.month,
        "day": lambda d: d.dt.day,
        "dayofweek": lambda d: d.dt.dayofweek,       # Monday=0, Sunday=6
        "dayofyear": lambda d: d.dt.dayofyear,
        "hour": lambda d: d.dt.hour,
        "minute": lambda d: d.dt.minute,
        "quarter": lambda d: d.dt.quarter,
        "weekofyear": lambda d: d.dt.isocalendar().week.astype(int),
    }

    for feat in features:
        if feat in feature_extractors:
            col_name = f"{time_col}_{feat}"
            if col_name in df_out.columns:
                warnings.warn(f"列 '{col_name}' 已存在，将被覆盖")
            df_out[col_name] = feature_extractors[feat](dt)

    # ---- 布尔标记 ----
    bool_extractors = {
        "is_weekend": lambda d: d.dt.dayofweek >= 5,
        "is_month_start": lambda d: d.dt.is_month_start,
        "is_month_end": lambda d: d.dt.is_month_end,
    }

    for feat in features:
        if feat in bool_extractors:
            col_name = f"{time_col}_{feat}"
            if col_name in df_out.columns:
                warnings.warn(f"列 '{col_name}' 已存在，将被覆盖")
            df_out[col_name] = bool_extractors[feat](dt).astype(int)

    # ---- 周期性编码 ----
    if cyclical:
        cyclical_specs = {
            "hour": (24, dt.dt.hour),
            "month": (12, dt.dt.month),
            "dayofweek": (7, dt.dt.dayofweek),
        }

        for name, (period, values) in cyclical_specs.items():
            sin_col = f"{time_col}_{name}_sin"
            cos_col = f"{time_col}_{name}_cos"

            if sin_col in df_out.columns:
                warnings.warn(f"列 '{sin_col}' 已存在，将被覆盖")
            if cos_col in df_out.columns:
                warnings.warn(f"列 '{cos_col}' 已存在，将被覆盖")

            rad = 2 * np.pi * values / period
            df_out[sin_col] = np.sin(rad)
            df_out[cos_col] = np.cos(rad)

    return df_out


# ============================================================
# 4. 交叉特征（Cross Features）
# ============================================================

def generate_cross_features(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    operation: str,
) -> pd.DataFrame:
    """
    对两列进行算术运算，生成交叉特征。

    典型用法：
      - 温度 × 负荷 → 体感用电强度
      - 负荷 / 温度 → 单位温度用电量
      - 负荷(t) - 负荷(t-1) → 一阶差分（也可用 lag + cross 组合）

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame
    col1 : str
        第一个操作数列名
    col2 : str
        第二个操作数列名
    operation : str
        运算类型: 'add' (加), 'subtract' (减), 'multiply' (乘),
                  'divide' (除), 'ratio' (比值，同 divide),
                  'diff' (col1 - col2 的差分别名)

    Returns
    -------
    pd.DataFrame
        追加一列，列名格式: "{col1}_{op}_{col2}"

    Raises
    ------
    ValueError
        列不存在、operation 不支持、或除零场景
    """
    # ---- 参数校验 ----
    for col in [col1, col2]:
        if col not in df.columns:
            raise ValueError(
                f"列 '{col}' 在 DataFrame 中不存在。"
                f"当前列: {list(df.columns)}"
            )

    operation = operation.lower().strip()
    valid_ops = {
        "add", "subtract", "multiply", "divide", "ratio", "diff",
    }
    if operation not in valid_ops:
        raise ValueError(
            f"不支持的运算 '{operation}'。支持: {sorted(valid_ops)}"
        )

    # ---- 运算映射 ----
    op_symbol_map = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "ratio": "/",
        "diff": "-",
    }

    df_out = df.copy()
    op_symbol = op_symbol_map[operation]

    if operation in ("subtract", "diff"):
        col_name = f"{col1}_minus_{col2}"
    elif operation in ("divide", "ratio"):
        col_name = f"{col1}_div_{col2}"
    elif operation == "add":
        col_name = f"{col1}_plus_{col2}"
    else:
        col_name = f"{col1}_{operation}_{col2}"

    if col_name in df_out.columns:
        warnings.warn(f"列 '{col_name}' 已存在，将被覆盖")

    # ---- 执行运算 ----
    if operation in ("add",):
        df_out[col_name] = df_out[col1] + df_out[col2]
    elif operation in ("subtract", "diff"):
        df_out[col_name] = df_out[col1] - df_out[col2]
    elif operation in ("multiply",):
        df_out[col_name] = df_out[col1] * df_out[col2]
    elif operation in ("divide", "ratio"):
        # 避免除零警告
        with np.errstate(divide="ignore", invalid="ignore"):
            result = df_out[col1] / df_out[col2]
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_out[col_name] = result

    return df_out


# ============================================================
# 5. 便捷函数：一键批量生成
# ============================================================

def generate_all_features(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    lag_list: List[int] = None,
    rolling_windows: List[int] = None,
    rolling_stats: List[str] = None,
    cross_pairs: List[Tuple[str, str, str]] = None,
    group_col: Optional[str] = None,
    time_features: List[str] = None,
    cyclical: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    一键批量生成所有类型特征，并返回生成报告。

    这是一个便捷聚合函数，内部依次调用上述 4 个独立函数。
    适合在 LLM Agent 决策完成后，一次性执行所有特征生成。

    Parameters
    ----------
    df : pd.DataFrame
        原始 DataFrame
    target_col : str
        目标列名（用于 lag 和 rolling）
    time_col : str
        时间列名（用于时间特征提取）
    lag_list : list of int, optional
        滞后步数。默认 [1, 2, 24, 168]
    rolling_windows : list of int, optional
        滚动窗口。默认 [6, 12, 24]
    rolling_stats : list of str, optional
        滚动统计量。默认 ['mean', 'std', 'max', 'min']
    cross_pairs : list of (col1, col2, operation), optional
        交叉特征三元组。默认 None = 不生成
    group_col : str, optional
        分组列名（传给 lag 和 rolling）
    time_features : list of str, optional
        时间特征列表。默认 None = 全部
    cyclical : bool
        是否生成周期性编码。默认 True

    Returns
    -------
    (pd.DataFrame, dict)
        - 追加了所有特征后的 DataFrame
        - 生成报告 dict:
            {
                "n_original_cols": int,
                "n_new_cols": int,
                "n_total_cols": int,
                "new_columns": [str, ...],
                "lag_count": int,
                "rolling_count": int,
                "time_count": int,
                "cross_count": int,
            }
    """
    # ---- 默认值 ----
    if lag_list is None:
        lag_list = [1, 2, 24, 168]
    if rolling_windows is None:
        rolling_windows = [6, 12, 24]
    if rolling_stats is None:
        rolling_stats = ["mean", "std", "max", "min"]

    original_cols = set(df.columns)
    df_out = df.copy()

    lag_count = 0
    rolling_count = 0
    time_count = 0
    cross_count = 0

    # ---- 1. 滞后特征 ----
    df_out = generate_lag_features(df_out, target_col, lag_list, group_col=group_col)
    lag_count = len(lag_list)

    # ---- 2. 滚动统计 ----
    df_out = generate_rolling_features(
        df_out, target_col, rolling_windows,
        stats=rolling_stats, group_col=group_col,
    )
    rolling_count = len(rolling_windows) * len(rolling_stats)

    # ---- 3. 时间特征 ----
    df_out = generate_time_features(
        df_out, time_col, features=time_features, cyclical=cyclical,
    )
    # 数一下新增的时间列
    time_count = len(set(df_out.columns) - original_cols) - lag_count - rolling_count

    # ---- 4. 交叉特征 ----
    if cross_pairs:
        for col1, col2, op in cross_pairs:
            df_out = generate_cross_features(df_out, col1, col2, op)
        cross_count = len(cross_pairs)

    # ---- 生成报告 ----
    new_cols = [c for c in df_out.columns if c not in original_cols]
    report = {
        "n_original_cols": len(original_cols),
        "n_new_cols": len(new_cols),
        "n_total_cols": len(df_out.columns),
        "new_columns": new_cols,
        "lag_count": lag_count,
        "rolling_count": rolling_count,
        "time_count": time_count,
        "cross_count": cross_count,
    }

    return df_out, report


# ============================================================
# 6. 工具函数：描述新增列
# ============================================================

def describe_new_columns(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
) -> pd.DataFrame:
    """
    对比特征生成前后，返回新增列的描述信息。

    适用于 LLM Agent 查看「生成了什么特征」，帮助它做出下一步决策。

    Parameters
    ----------
    df_before : pd.DataFrame
        特征生成前的 DataFrame
    df_after : pd.DataFrame
        特征生成后的 DataFrame

    Returns
    -------
    pd.DataFrame
        包含列名、数据类型、非空数量、均值/标准差（数值列）的概览
    """
    new_cols = [c for c in df_after.columns if c not in df_before.columns]

    if not new_cols:
        return pd.DataFrame({"message": ["没有新增列"]})

    desc = df_after[new_cols].describe().T
    desc["dtype"] = [str(df_after[c].dtype) for c in new_cols]
    desc["non_null"] = [df_after[c].notna().sum() for c in new_cols]
    desc["null_ratio"] = [
        round(df_after[c].isna().mean(), 4) for c in new_cols
    ]

    cols_order = ["dtype", "non_null", "null_ratio", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    desc = desc[[c for c in cols_order if c in desc.columns]]

    return desc


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    # 构造模拟电力负荷数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="h")
    n = len(dates)

    df = pd.DataFrame({
        "datetime": dates,
        "LOAD": (
            500
            + 50 * np.sin(2 * np.pi * np.arange(n) / 24)       # 日内周期
            + 30 * np.sin(2 * np.pi * np.arange(n) / 168)      # 周内周期
            + np.random.normal(0, 15, n)                        # 噪声
        ),
        "temp": 20 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + np.random.normal(0, 3, n),
        "humidity": 60 + 15 * np.sin(2 * np.pi * np.arange(n) / 48) + np.random.normal(0, 5, n),
    })

    print("=" * 60)
    print("1. 滞后特征测试")
    print("=" * 60)
    df1 = generate_lag_features(df, "LOAD", [1, 2, 24])
    lag_cols = [c for c in df1.columns if c.startswith("LOAD_lag")]
    print(f"  新增列: {lag_cols}")
    print(f"  LOAD_lag_1 前 5 行:\n{df1['LOAD_lag_1'].head(6).to_string()}")
    print(f"  LOAD_lag_24[24:28]:\n{df1['LOAD_lag_24'].iloc[24:28].to_string()}")
    # 验证：滞后 24 步的值应该等于原始 24 步前的值
    assert np.allclose(
        df1["LOAD_lag_24"].iloc[24:30].values,
        df["LOAD"].iloc[0:6].values,
        rtol=1e-5,
    ), "滞后 24 步校验失败！"
    print("  [OK] lag_24 校验通过")

    print()
    print("=" * 60)
    print("2. 滚动统计特征测试")
    print("=" * 60)
    df2 = generate_rolling_features(df, "LOAD", [6, 12], stats=["mean", "std", "max", "min"])
    roll_cols = [c for c in df2.columns if "rolling" in c]
    print(f"  新增列: {roll_cols}")
    # 验证：滚动均值的前 window-1 行应为 NaN
    assert df2["LOAD_rolling_6_mean"].iloc[:5].isna().all(), "前 5 行应为 NaN"
    assert df2["LOAD_rolling_6_mean"].iloc[5:10].notna().all(), "第 6 行起应有值"
    print("  [OK] NaN 位置校验通过")

    print()
    print("=" * 60)
    print("3. 时间特征测试")
    print("=" * 60)
    df3 = generate_time_features(df, "datetime")
    time_cols = [c for c in df3.columns if c.startswith("datetime_")]
    print(f"  新增列 ({len(time_cols)}): {time_cols}")
    # 验证周期性编码值域
    for col in ["datetime_hour_sin", "datetime_hour_cos"]:
        vals = df3[col].dropna()
        assert (-1.0 <= vals.min() and vals.max() <= 1.0), f"{col} 应在 [-1, 1]"
    print("  [OK] 周期性编码值域在 [-1, 1]")

    # 验证 is_weekend
    weekend_dates = df3[df3["datetime_is_weekend"] == 1]["datetime"]
    if len(weekend_dates) > 0:
        print(f"  周末样本数: {len(weekend_dates)}, 示例: {weekend_dates.iloc[0]}")

    print()
    print("=" * 60)
    print("4. 交叉特征测试")
    print("=" * 60)
    df4 = generate_cross_features(df, "temp", "LOAD", "multiply")
    cross_col = "temp_multiply_LOAD"
    assert cross_col in df4.columns
    expected = df["temp"] * df["LOAD"]
    assert np.allclose(df4[cross_col].values, expected.values, rtol=1e-5)
    print(f"  [OK] '{cross_col}' = temp * LOAD 校验通过")

    # 测试 subtract
    df4b = generate_cross_features(df, "LOAD", "LOAD", "subtract")
    # LOAD - LOAD = 0
    assert (df4b["LOAD_minus_LOAD"].dropna() == 0).all()
    print("  [OK] LOAD - LOAD = 0 校验通过")

    # 测试除零保护
    df_zero = df.copy()
    df_zero.loc[0, "LOAD"] = 0.0
    df_div = generate_cross_features(df_zero, "temp", "LOAD", "divide")
    assert np.isnan(df_div.loc[0, "temp_div_LOAD"])
    assert not np.isnan(df_div.loc[1, "temp_div_LOAD"])
    print("  [OK] 除零 → NaN 校验通过")

    print()
    print("=" * 60)
    print("5. 批量生成 + 报告测试")
    print("=" * 60)
    df_all, report = generate_all_features(
        df,
        target_col="LOAD",
        time_col="datetime",
        cross_pairs=[("temp", "LOAD", "multiply"), ("LOAD", "temp", "divide")],
    )
    print(f"  报告:")
    for k, v in report.items():
        if k == "new_columns":
            print(f"    {k}: [{len(v)} 列] {v[:5]}...")
        else:
            print(f"    {k}: {v}")

    print()
    print("=" * 60)
    print("6. describe_new_columns 测试")
    print("=" * 60)
    desc = describe_new_columns(df, df_all)
    print(desc.to_string())

    print()
    print("=" * 60)
    print("7. 错误处理测试")
    print("=" * 60)
    # 列不存在
    try:
        generate_lag_features(df, "NONEXISTENT", [1, 2])
    except ValueError as e:
        print(f"  [OK] 列不存在 → ValueError: {str(e)[:60]}...")

    # 非法 lag
    try:
        generate_lag_features(df, "LOAD", [0, -1])
    except ValueError as e:
        print(f"  [OK] 非法 lag → ValueError: {str(e)[:60]}...")

    # 非法窗口
    try:
        generate_rolling_features(df, "LOAD", [1])
    except ValueError as e:
        print(f"  [OK] window < 2 → ValueError: {str(e)[:60]}...")

    # 非法统计量
    try:
        generate_rolling_features(df, "LOAD", [6], stats=["percentile"])
    except ValueError as e:
        print(f"  [OK] 非法 stat → ValueError: {str(e)[:60]}...")

    # 非法运算
    try:
        generate_cross_features(df, "LOAD", "temp", "power")
    except ValueError as e:
        print(f"  [OK] 非法运算 → ValueError: {str(e)[:60]}...")

    # 空 lag_list
    try:
        generate_lag_features(df, "LOAD", [])
    except ValueError as e:
        print(f"  [OK] 空 lag_list → ValueError: {str(e)}")

    print()
    print("=" * 60)
    print("全部测试通过！")
    print("=" * 60)
