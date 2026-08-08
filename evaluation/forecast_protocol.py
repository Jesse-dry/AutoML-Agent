# Forecast Protocol：滚动预测的信息策略
# ---------------------------------------------------------------
# 明确区分两种评测协议（它们共享同一特征工程，但 observed 回填不同）：
#
#   ONLINE_H1  (online_h1, operational one-hour-ahead)
#     预测 t 时只用 ≤ t-1 的真实 LOAD；获得 t 的真实值后预测 t+1。
#     —— 短期滚动负荷预测 Agent 的协议，可向量化，不复现 GEFCom
#        官方 month-ahead 信息条件。
#
#   RECURSIVE_MONTH_AHEAD (recursive_month_ahead)
#     forecast origin 之后，预测月内只用上一小时的**预测值**回填
#     （recursive forecast），全程禁止任何 y_true。
#     —— GEFCom 官方 month-ahead 的近似复现，必须逐小时循环。
# ---------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Optional


class ForecastProtocol(ABC):
    """滚动预测信息策略抽象。"""

    name: str = "base"

    @property
    @abstractmethod
    def recursive(self) -> bool:
        """True 表示预测月内必须用预测值回填（逐小时），False 可用真值（可向量化）。"""

    @abstractmethod
    def backfill(self, y_hat_t: Optional[float], y_true_t: Optional[float]) -> float:
        """预测小时 t 之后，observed 序列在 t 处回填的值。"""


class OnlineH1Protocol(ForecastProtocol):
    name = "online_h1"

    @property
    def recursive(self) -> bool:
        return False

    def backfill(self, y_hat_t: Optional[float], y_true_t: Optional[float]) -> float:
        # 实时预报：预测 t 后立刻观测到真实值
        return y_true_t


class RecursiveMonthAheadProtocol(ForecastProtocol):
    name = "recursive_month_ahead"

    @property
    def recursive(self) -> bool:
        return True

    def backfill(self, y_hat_t: Optional[float], y_true_t: Optional[float]) -> float:
        # month-ahead：预测月内只能用自己的预测值
        return y_hat_t


ONLINE_H1 = OnlineH1Protocol()
RECURSIVE_MONTH_AHEAD = RecursiveMonthAheadProtocol()

_PROTOCOLS = {
    ONLINE_H1.name: ONLINE_H1,
    RECURSIVE_MONTH_AHEAD.name: RECURSIVE_MONTH_AHEAD,
}


def get_protocol(name: str) -> ForecastProtocol:
    if name not in _PROTOCOLS:
        raise ValueError(f"未知协议 '{name}'，可选: {list(_PROTOCOLS)}")
    return _PROTOCOLS[name]
