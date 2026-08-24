# ECL 跨用户迁移数据加载器
# ---------------------------------------------------------------
# 职责：加载预处理版 ECL（laiguokun electricity.txt），生成时间戳索引，
#       随机划分 train/test 用户（固定种子，可复现）。
#
# ECL 数据约定（预处理版，laiguokun/multivariate-time-series-data）：
#   - 纯数字矩阵：26304 行 × 321 列，逗号分隔，无表头、无时间戳列。
#   - 行 = 小时：26304 = 3 年（2012 闰年 366 天 + 2013/2014 各 365 天）。
#     按仓库约定，行 i 对应 2012-01-01 00:00 + i 小时（本加载器据此生成索引）。
#   - 列 = 321 个居民用户（原 UCI 的 MT_001..MT_321 列名在预处理时被剥掉，
#     本加载器重新命名 client_0..client_320）。
#   - 值 = 小时负荷 kW，非负，无缺失。用户间差异极大（平均 23 ~ 3335 kW，约 140 倍）。
#
# 迁移实验（ECL 特有，非滚动回放）：
#   随机划分 260 train / 61 test（np.random.RandomState(seed) 固定，可复现），
#   test 用户是模型训练时从未见过的用户。
# ---------------------------------------------------------------
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ECL_DATA_DIR = PROJECT_ROOT / "ECL"
ECL_FILENAME = "electricity.txt"

ECL_N_USERS = 321
ECL_START = "2012-01-01 00:00:00"
ECL_FREQ = "h"
ECL_TARGET_SUFFIX = "client_"

DEFAULT_N_TRAIN = 260
DEFAULT_SEED = 42


def load_ecl_matrix(data_dir: Path = ECL_DATA_DIR) -> pd.DataFrame:
    """加载 electricity.txt → (26304 × 321) DataFrame。

    - datetime 索引（2012-01-01 00:00 起，每小时一行）
    - 列名 client_0..client_320（321 个用户）
    - 值 kW，无缺失
    """
    path = Path(data_dir) / ECL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"ECL 数据缺失: {path}")
    df = pd.read_csv(path, header=None)
    df.columns = [f"{ECL_TARGET_SUFFIX}{i}" for i in range(df.shape[1])]
    df.index = pd.date_range(ECL_START, periods=len(df), freq=ECL_FREQ)
    df.index.name = "datetime"
    return df


def split_users(
    n_users: int = ECL_N_USERS,
    n_train: int = DEFAULT_N_TRAIN,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[str], List[str]]:
    """随机划分用户（固定种子，可复现）。

    返回 (train_cols, test_cols) —— 各为 client_* 列名列表。
    321 个用户打乱后，前 n_train 个为训练，其余为测试。
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_users)
    train = [f"{ECL_TARGET_SUFFIX}{i}" for i in idx[:n_train]]
    test = [f"{ECL_TARGET_SUFFIX}{i}" for i in idx[n_train:]]
    return train, test
