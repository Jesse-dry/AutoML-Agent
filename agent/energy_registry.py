# energy 赛道注册表
# ---------------------------------------------------------------
# 把「能源赛道」的资源解析从散落的 if-else 收敛成一张配置表。
# 每个 energy 一个 EnergySpec：数据字段 + 统一签名的函数。
#
# 接入新赛道（Solar/Price）= 在 ENERGY_REGISTRY 里加一行 + 改 CLI --energy choices。
#
# 统一签名约定：
#   availability_fn(task_id, zone, data_dir) -> Availability
#   spec_evaluator(task_id, zone, spec, protocol, val_hours, backend_factory,
#                  seed, data_dir) -> dict
#   单分区赛道（Load/Price）zone 传 None，wrapper 忽略。
# ---------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from data.availability import available_history
from data.gefcom_loader import GEFCOM_DATA_DIR
from data.price_loader import (
    PRICE_DATA_DIR,
    PRICE_EXOGENOUS_COLS,
    PRICE_TARGET_COL,
    price_available_history,
)
from data.price_task_builder import PRICE_FEATURE_SPEC
from data.solar_loader import (
    SOLAR_DATA_DIR,
    SOLAR_TARGET_COL,
    SOLAR_WEATHER_COLS,
    SOLAR_ZONES,
    solar_available_history,
)
from data.solar_task_builder import SOLAR_FEATURE_SPEC
from data.task_builder import FEATURE_SPEC, TARGET_COL
from data.wind_loader import WIND_DATA_DIR, WIND_TARGET_COL, wind_available_history
from data.wind_task_builder import WIND_FEATURE_SPEC, WIND_WEATHER_DERIVED_COLS
from evaluation.spec_evaluator import (
    evaluate_price_spec,
    evaluate_solar_spec,
    evaluate_spec,
    evaluate_wind_spec,
)


@dataclass(frozen=True)
class EnergySpec:
    """一条能源赛道的完整资源描述（数据字段 + 统一签名函数）。"""

    key: str
    target_col: str
    data_dir: str
    zones: tuple                     # 空 tuple = 单分区
    dataset_label: str               # 任务描述前缀（"GEFCom2014" / "GEFCom2014-W"）
    label: str                       # 简短赛道名（"电力负荷预测" / "风电出力预测"）
    base_spec: List[dict]            # 血缘式特征 spec（runner 会 snapshot）
    allowed_sources: frozenset       # {target_col} ∪ 外生列（门控外生/current 特征）
    domain: str                      # 领域知识（已含 target_col，喂 system prompt）
    availability_fn: Callable        # (task_id, zone, data_dir) -> Availability
    spec_evaluator: Callable         # 统一签名（见文件头）


def _load_availability(task_id: int, zone: Optional[int] = None,
                       data_dir=None) -> "Availability":
    """Load 单分区 wrapper：忽略 zone，对齐统一签名。"""
    if data_dir is None:
        return available_history(task_id)
    return available_history(task_id, data_dir)


def _load_spec_evaluator(task_id: int, zone: Optional[int] = None, spec: List[dict] = None,
                         protocol=None, val_hours: int = 168, backend_factory=None,
                         seed: int = 42, data_dir=None) -> Dict:
    """Load 单分区 wrapper：忽略 zone，对齐统一签名。"""
    return evaluate_spec(task_id, spec, protocol, val_hours=val_hours,
                         backend_factory=backend_factory, seed=seed, data_dir=data_dir)


def _price_availability(task_id: int, zone: Optional[int] = None, data_dir=None) -> "Availability":
    """Price 单分区 wrapper：忽略 zone，对齐统一签名。"""
    if data_dir is None:
        return price_available_history(task_id)
    return price_available_history(task_id, data_dir)


def _price_spec_evaluator(task_id: int, zone: Optional[int] = None, spec: List[dict] = None,
                          protocol=None, val_hours: int = 168, backend_factory=None,
                          seed: int = 42, data_dir=None) -> Dict:
    """Price 单分区 wrapper：忽略 zone，对齐统一签名。"""
    return evaluate_price_spec(task_id, spec, protocol, val_hours=val_hours,
                               backend_factory=backend_factory, seed=seed, data_dir=data_dir)


ENERGY_REGISTRY = {
    "load": EnergySpec(
        key="load",
        target_col=TARGET_COL,
        data_dir=str(GEFCOM_DATA_DIR),
        zones=(),
        dataset_label="GEFCom2014",
        label="电力负荷预测",
        base_spec=FEATURE_SPEC,
        allowed_sources=frozenset({TARGET_COL}),
        domain=(
            "你是**电力负荷预测**的特征工程决策 Agent。目标列 LOAD 为电力负荷。\n"
            "负荷特性：日/周周期强、规律明显；工作日/周末差异大；晚峰低谷是主要误差来源。"
        ),
        availability_fn=_load_availability,
        spec_evaluator=_load_spec_evaluator,
    ),
    "wind": EnergySpec(
        key="wind",
        target_col=WIND_TARGET_COL,
        data_dir=str(WIND_DATA_DIR),
        zones=tuple(range(1, 11)),
        dataset_label="GEFCom2014-W",
        label="风电出力预测",
        base_spec=WIND_FEATURE_SPEC,
        allowed_sources=frozenset({WIND_TARGET_COL, *WIND_WEATHER_DERIVED_COLS}),
        domain=(
            "你是**风电出力预测**的特征工程决策 Agent。目标列 TARGETVAR 为归一化风电出力 [0,1]。\n"
            "风电特性：随机性强、天气驱动、非平稳；出力随风速非线性上升，存在爬坡事件\n"
            "（短时大幅变化），日/周周期弱于负荷。外生气象预报（ws10/ws100 等）在决策时点\n"
            "可得，是最重要的特征来源；注意训练侧用历史实际天气、预测侧用气象预报，\n"
            "存在 train/serve 分布偏移，Agent 应感知这一风险。"
        ),
        availability_fn=wind_available_history,
        spec_evaluator=evaluate_wind_spec,
    ),
    "solar": EnergySpec(
        key="solar",
        target_col=SOLAR_TARGET_COL,
        data_dir=str(SOLAR_DATA_DIR),
        zones=SOLAR_ZONES,
        dataset_label="GEFCom2014-S",
        label="光伏出力预测",
        base_spec=SOLAR_FEATURE_SPEC,
        allowed_sources=frozenset({SOLAR_TARGET_COL, *SOLAR_WEATHER_COLS}),
        domain=(
            "你是**光伏出力预测**的特征工程决策 Agent。目标列 POWER 为归一化光伏出力 [0,1]。\n"
            "光伏特性：夜间恒 0（约 44% 小时）、白天由太阳辐射驱动，日周期极强但夜间无信号；\n"
            "天气是白天出力波动的关键。外生气象预报（VAR169=SSRD 太阳辐射、VAR164=TCC 云量、\n"
            "VAR167=2T 温度）在决策时点可得，是白天最重要的特征来源；夜间段特征应退化为恒 0，\n"
            "Agent 应避免为夜间堆叠无效特征。"
        ),
        availability_fn=solar_available_history,
        spec_evaluator=evaluate_solar_spec,
    ),
    "price": EnergySpec(
        key="price",
        target_col=PRICE_TARGET_COL,
        data_dir=str(PRICE_DATA_DIR),
        zones=(),
        dataset_label="GEFCom2014-P",
        label="电价预测",
        base_spec=PRICE_FEATURE_SPEC,
        allowed_sources=frozenset({PRICE_TARGET_COL, *PRICE_EXOGENOUS_COLS}),
        domain=(
            "你是**电价预测**的特征工程决策 Agent。目标列 Zonal Price 为连续肥尾电价\n"
            "（均值≈48.6、p99≈156、max≈364）。电价由市场均衡决定：尖峰/肥尾、均值回归、\n"
            "需求驱动；预测窗口 = 1 天（24h）。外生「Forecasted Total/Zonal Load」负荷预报\n"
            "在决策时点可得，是电价的重要驱动。尖峰日误差远大于常规日，Agent 应关注尾部风险。"
        ),
        availability_fn=_price_availability,
        spec_evaluator=_price_spec_evaluator,
    ),
}


def get_energy(key: str) -> EnergySpec:
    """按 key 查注册表，未知 key 抛错。"""
    if key not in ENERGY_REGISTRY:
        raise ValueError(f"未知 energy '{key}'，可选: {sorted(ENERGY_REGISTRY)}")
    return ENERGY_REGISTRY[key]
