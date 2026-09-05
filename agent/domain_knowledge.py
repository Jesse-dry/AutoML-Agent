# 领域知识注册表：每个数据集一段"特征工程先验提示"
# ---------------------------------------------------------------
# 目的：把数据集特定的领域知识显式喂给 LLM，指导其特征工程方向。
# 设计：按 dataset 名注册一段领域先验（中文，可执行的特征设计原则），
#       由 CLI 按 --dataset 注入 system prompt。
#
# 写每段先验的原则：
#   - 讲"这个赛道的主导信号是什么 / 目标由什么驱动"（帮助 LLM 选对 source）；
#   - 给"可执行的特征方向"（不是泛泛的领域科普，而是能转化为 add_feature 的具体建议）；
#   - 提示"容易踩的坑"（如 solar 夜间恒 0、load 别退化成 persistence）。
# ---------------------------------------------------------------
from typing import Dict

DOMAIN_KNOWLEDGE: Dict[str, str] = {
    # ---- GEFCom2014 负荷（主赛道）----
    "load": """
## 领域先验：电力负荷预测
- 负荷由**日周期（24h）+ 周周期（168h）**双驱动，工作日/周末形态明显不同；
  温度（天气）驱动的冷/热负荷是季节性变化主因。
- 特征方向：
  · 保留 lag_1（短时惯性）但不依赖它——只靠 lag_1 会退化成 persistence；
  · 优先补 lag_24 / lag_168 及它们与 hour/weekday 的交互（cross）；
  · rolling_mean（近 24h/168h 水平）能稳定刻画近期趋势，比裸 lag 更抗噪。
- 易踩的坑：已有 lag_24 时再加 lag_23/25 冗余；rolling 窗口别超过周周期。
""",
    # ---- GEFCom2014 光伏（本轮重点：外生列主导）----
    "solar": """
## 领域先验：光伏出力预测
- 光伏出力（POWER）**由太阳辐射直接驱动**：VAR169（SSRD 地表太阳辐射）是
  第一主导信号；VAR164（总云量）决定遮挡、VAR167（2m 温度）影响面板效率。
- **外生列是主导，目标列滞后是辅助**——这是 Solar 与 Load 最大的区别：
  · 优先构造 VAR169 的 lag / rolling（辐射的过去值预测未来出力）；
  · 做 POWER 与 VAR169 的 cross（如 lag_1 × VAR169_lag_1），捕捉"同辐射下历史出力"；
  · 云量 VAR164 / 温度 VAR167 的 lag 次之。
- 夜间出力恒 0（约 44% 小时），白天的误差才是有意义的部分；误差画像的
  worst_segment 若集中在夜间，请聚焦白天时段（hour 6~19）的特征。
- 易踩的坑：只对 POWER 做滚动统计而不用 VAR169，等于扔掉主导信息。
""",
    # ---- GEFCom2014 风电（未接入 agent，预留）----
    "wind": """
## 领域先验：风电出力预测
- 风电出力（TARGETVAR）由**风速驱动**且强非线性（切入/额定/切出功率曲线），
  波动大、尖峰多，persistence 基线相对弱。
- 特征方向：
  · 外生风速 U10/V10/U100/V100 的 lag / rolling 比目标列滞后信息量大；
  · 可用 wind speed 派生（合成风速、切变）做 cross（若已注册为外生列）；
  · 关注误差画像的高波动段（风速骤变时）。
- 易踩的坑：风电自相关弱于负荷，过深的目标列滞后收益递减。
""",
    # ---- GEFCom2014 电价（未接入 agent，决策价值主线，预留）----
    "price": """
## 领域先验：电价预测
- 电价（Zonal Price）**强均值回归 + 尖峰重尾**：lag_1 信息量高，但尖峰（峰段）
  误差才是决策价值所在（储能套利：峰段预测准 → 利润高）。
- 特征方向：
  · 保留 lag_1（均值回归）同时补峰段专用特征（峰时段的 lag / rolling 分桶）；
  · 负荷-电价耦合：若负荷列可用，负荷 lag 与电价的 cross 有价值；
  · 关注误差画像 worst_segment 是否集中在峰段——那是 P-Value 主线最该修的误差。
- 易踩的坑：只看整体 RMSE 忽略峰段；尖峰在 MAPE 上会被淹没，看 RMSE/分位误差。
""",
    # ---- ECL 居民用电（未接入 agent，跨用户迁移，预留）----
    "ecl": """
## 领域先验：居民用电（跨用户迁移）
- 321 个用户负荷差异极大（用户规模差可达 140 倍），工作日/周末、早晚峰形态因
  用户而异；跨用户迁移场景下**用户无关特征**优先（时间型 + 归一化趋势）。
- 特征方向：
  · hour / weekday / is_weekend 及它们与 lag 的 cross 跨用户最稳；
  · rolling 相对量（相对近期均值的变化）比绝对滞后更可迁移；
  · 目标 lag 用短周期（lag_1/24），深滞后跨用户不稳定。
- 易踩的坑：绝对 lag 值跨用户不可比，迁移到未见用户时优先相对/交互特征。
""",
}


# ---------------------------------------------------------------
# 负对照先验（--domain-key 指定；不注册进 DOMAIN_KNOWLEDGE，避免污染生产默认）
# ---------------------------------------------------------------
# ① 反事实：湿度驱动 + 反向周期（物理上错误，检验 LLM 是否被带偏）
_NEG_HUMIDITY_TEXT = """
## 领域先验：光伏出力预测
- 光伏出力（POWER）**主要受前一小时湿度影响，且呈反向周期**：湿度越高出力越低，
  湿度滞后 1/2/4 小时是预测的关键特征。
- 特征方向：
  · 优先构造湿度相关列的 lag（前一小时湿度最能解释出力）；
  · 用湿度与 POWER 的 cross 强化"反向周期"；
  · 云量/温度/辐射对出力影响可以忽略，不必使用。
- 夜间出力恒 0 约 44% 小时，白天时段的特征集中在湿度滞后上。
- 易踩的坑：太阳辐射（VAR169）不是主要驱动，不要浪费预算在它上面。
"""

NEGATIVE_DOMAIN_KNOWLEDGE: Dict[str, str] = {
    # 反事实：湿度驱动 + 反向周期
    "solar_neg_humidity": _NEG_HUMIDITY_TEXT,
    # 错配：load 先验扔给 solar（居民负荷 vs 光伏）
    "solar_neg_load": DOMAIN_KNOWLEDGE["load"],
    # 干扰：wind 先验扔给 solar（风速功率曲线 vs 光伏）
    "solar_neg_wind": DOMAIN_KNOWLEDGE["wind"],
    # load 的负对照：solar 先验错配 / 湿度反事实（负荷并非辐射驱动）
    "load_neg_solar": DOMAIN_KNOWLEDGE["solar"],
    "load_neg_humidity": _NEG_HUMIDITY_TEXT,
    # wind 的负对照：load 先验错配 / solar 先验错配（风电并非负荷/辐射驱动）
    "wind_neg_load": DOMAIN_KNOWLEDGE["load"],
    "wind_neg_solar": DOMAIN_KNOWLEDGE["solar"],
}


def get_domain_knowledge(dataset: str) -> str:
    """按 dataset 名取领域提示词；未注册的数据集返回空（不注入额外先验）。"""
    return DOMAIN_KNOWLEDGE.get(dataset, "")


def get_domain_knowledge_by_key(key: str) -> str:
    """按 key 取先验：先查正注册表，再查负对照注册表；空 key 返回空（无先验）。

    用于 --domain-key 显式指定先验（正 / 负 / 无），支撑三臂对照实验。
    """
    if not key:
        return ""
    if key in DOMAIN_KNOWLEDGE:
        return DOMAIN_KNOWLEDGE[key]
    if key in NEGATIVE_DOMAIN_KNOWLEDGE:
        return NEGATIVE_DOMAIN_KNOWLEDGE[key]
    raise KeyError(
        f"未知 domain-key {key!r}；可用正先验={sorted(DOMAIN_KNOWLEDGE)}，"
        f"负对照={sorted(NEGATIVE_DOMAIN_KNOWLEDGE)}"
    )


def build_domain_knowledge_section(dataset: str = "", text: str = "") -> str:
    """渲染成 system prompt 片段（带标题）；无先验时返回空串。

    - dataset 传入时按注册表取先验（正注册表）；
    - text 传入时直接使用（负对照实验的任意先验文本）。
    """
    if text is None:
        text = ""
    if not text and dataset:
        text = get_domain_knowledge(dataset)
    if not text:
        return ""
    return f"\n{text.strip()}\n"
