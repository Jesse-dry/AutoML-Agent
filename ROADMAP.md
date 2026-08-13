# AutoML-Agent 演进路线图 V2.1（多能源扩展 + 业务/决策驱动版）

> 本文件为项目演进路线的**唯一依据**（多能源扩展版）。
> 核心变化：从「单数据集（GEFCom2014 Load）特征工程闭环」升级为「**面向多能源场景的滚动自进化风险感知自主机器学习智能体**」，
> 并以 **决策价值（业务/决策驱动）** 为最终价值出口 —— 预测不止于「准」，更在于「有用」。

---

## 一、项目定位升级

| | 旧定位（V1.0） | 新定位（V2.0 起） |
|---|---|---|
| 一句话 | 在 GEFCom2014 上做 LLM 特征工程 | **面向多能源场景的滚动自进化风险感知自主机器学习智能体**，价值出口为**决策效能评估（储能套利）** |
| 数据 | 单一负荷数据集 | 负荷 + 风电 + 光伏 + 多用户 + 长序列 + 家庭能源 |
| 能力 | 单任务特征优化 | 跨能源任务 · 跨时间演化 · 跨场景迁移 |

**数据支撑链条**：

```
GEFCom2014 Load（电力负荷）── 主论文
  ├── GEFCom2014 Wind（风电）── 新能源泛化
  ├── GEFCom2014 Solar（光伏）── 场景专家
  ├── GEFCom2014 Price（电价）── 决策效能评估主线（价值出口）
  ├── ECL（370 用户负荷）──── 跨用户迁移
  ├── ETT（变压器长序列）──── 模型选择
  └── Pecan Street（家庭能源）── 应用展示
```

项目不再是"在 GEFCom2014 上调 LightGBM 特征"，而是"**构建一个能够跨能源任务、跨时间演化、跨场景迁移、并能把预测转成决策价值的自主智能体**"——更贴近当前 AI Agent + AutoML + Energy AI 的研究方向。

**项目架构（业务/决策驱动，三层）**：

```
┌──────────────────────────────────────────────────────────────┐
│  价值出口层（新增 · 核心）—— 业务/决策驱动                       │
│  Market/Price Agent（电价预测 + 尖峰捕获 + 归因报告）           │
│    → Storage Arbitrage Evaluator（储能套利决策层）             │
│      LP 决策 → 按预测决策 / 按真实结算 → 利润 / Regret         │
└───────────────────────────┬──────────────────────────────────┘
                            │ 预测输出（点 / 分位数）
┌───────────────────────────┴──────────────────────────────────┐
│  能力引擎层                                                    │
│  自进化 Agent（V2.0 双闭环）· 多能源专家协同（V3.0）· 风险感知（V4.0）│
└───────────────────────────┬──────────────────────────────────┘
                            │ 无泄漏滚动回放
┌───────────────────────────┴──────────────────────────────────┐
│  底层环境层                                                    │
│  GEFCom2014 Load（负荷）· Wind（风电）· Solar（光伏）           │
│  Task 1–15 滚动回放 · 10 分区独立模型                          │
└──────────────────────────────────────────────────────────────┘
```

**评估双主线（不是取代，是分层）**：
- **精度主线（科学基线）**：RMSE / MAE / CRPS / Coverage —— 保证预测统计上可靠，是决策层的**诚实性锚点**（防止 Agent 为套利利润过度博弈尖峰）
- **决策主线（业务展示）**：储能套利利润 / Regret（完美预知 LP 利润 − Agent 利润）= **预测误差的货币化** —— 回答"预测准了然后呢"

> 关系：精度是基底层，决策价值是展示层。同一个自主智能体，既要预测统计可靠（CRPS），
> 又要能把预测转成真金白银（套利利润）——这正是 Energy AI 与运筹交叉的前沿叙事
> （Decision-focused Forecasting）。

---

## 二、多数据集版图

### 总览

| # | 数据集 | 任务 | 采样/规模 | 用途 | 验证创新 | 优先级 |
|---|--------|------|-----------|------|----------|--------|
| 1 | GEFCom2014 Load | 负荷 | 15 个 Task，1h | 主论文核心 | 滚动自进化 + 风险感知 | 必备 |
| 2 | GEFCom2014 Wind | 风电 | 15 Task × 10 分区（Zone），1h | 新能源泛化 | 滚动自进化 + 多专家 + 风险 | ★★★★★ |
| 3 | GEFCom2014 Solar | 光伏 | 15 Task，1h，含天气预测变量 | 场景专家 | 多专家协同 | ★★★★★ |
| 4 | ECL | 用户级负荷 | 370 用户，15min | 跨用户迁移 | 时序经验记忆 | ★★★★★ |
| 5 | ETT | 变压器长序列 | 4 个变体，1h/15min | 模型选择 | 模型选择 Agent | ★★★★ |
| 6 | Pecan Street | 家庭能源 | 数百家庭，1min | 应用展示 | 家庭场景专家 | ★★★★ |
| 7 | GEFCom2014 Price | 电价 | 15 Task，1h，单分区（ZONEID=1） | **决策效能评估主线（价值出口）** | 决策导向评估（储能套利 + 尖峰捕获） | ★★★★★ |

> ※ **数据校正**：GEFCom2014 官方共 **4 个赛道** —— **Load / Price / Solar / Wind**（另有 `GEFCom2014-E.xlsx` 误差分析表），均为 15 个 Task。四套原始压缩包已全部在本仓库 `GEFCom2014 Data/`（`GEFCom2014-{L,P,S,W}_V2.zip`），光伏不必另找 GEFCom2012 或 NREL/DKASC。

### 各数据集与 Agent 设计

#### 1. GEFCom2014 Load（核心，保持）

- 负荷特性：周期强、规律明显，Agent 容易学到 `lag_24` 这类基础模式
- 天然 Task1→Task15 序列，验证**滚动自进化**是否随任务积累变强
- 原任务即为概率预测（分位数赛道），天然契合**风险感知**

#### 2. GEFCom2014 Wind（风电）★★★★★

**为什么适合**：负荷周期强、规律明显；而风电**随机性强、天气驱动强、非平稳**。风电更能检验 Agent 是**真的理解了场景**，还是只是找到了 `lag_24` 这种捷径。

**数据结构**：15 个 Task × 10 个分区（Zone），1h 采样。每 Task = `benchmark{k}_W.csv`（50 分位模板）+ `Task{k}_W_Zone1_10.zip`（10 分区历史）+ `TaskExpVars{k}_W_Zone1_10.zip`（气象预测变量）；官方真值文件仅 `Solution to Task 15/solution15_W.csv`（真值语义待 loader 泛化时逐项验证）。

为风电设计的 Agent：

```
Weather Expert
  输入：风速 · 风向 · 温度 · 湿度
Uncertainty Expert
  输出：P10 / P50 / P90
Scenario Agent
  识别：低风速稳定 / 高波动天气 / 风暴天气
```

**创新包装**：面向高随机性新能源出力预测的**风险感知型自主学习智能体**。

#### 3. GEFCom2014 Solar（光伏）★★★★★

**特点**：夜间为 0、白天周期、天气影响、季节变化——离散状态明显，是分场景专家的理想验证场。

**数据结构**：15 个 Task，1h 采样。每 Task = `train{k}.csv` + `benchmark{k}.csv` + `predictors{k}.csv`（天气变量）；官方真值文件仅 `Solution to Task 15/Solution to Task 15.csv`。

```
Night Expert（夜间恒 0）
Sunny Expert（晴日强周期）
Cloudy Expert（云层波动）
Extreme Weather Expert（极端天气）
```

**证明点**：多专家协同**不仅对负荷有效**，在状态离散的光伏上同样成立。

#### 4. ECL（Electricity Consumption Load）★★★★★

- 来源：UCI，`ElectricityLoadDiagrams20112014`
- 特点：370 个用户，15 分钟采样，2011–2014

**实验设计（跨用户迁移）**：

```
训练：User 1–300
测试：User 301–370
比较：普通模型 vs 带 Memory 的 Agent
```

**创新包装**：基于时序经验记忆的**跨用户迁移预测**——验证 Agent 能否把用户 A 的经验迁移到用户 B。

#### 5. ETT（Electricity Transformer Temperature）★★★★

- 经典长序列预测数据集（变压器油温 + 电力负荷 + 时间特征）
- 4 个变体：ETTh1 / ETTh2 / ETTm1 / ETTm2
- 用途：长序列预测，验证**模型选择 Agent**：

```
短期：LightGBM
长期：PatchTST
复杂：Transformer
```

#### 6. Pecan Street（家庭能源系统）★★★★

- 真实家庭数据：家庭用电 + 光伏 + EV + 储能（需数据协议）
- 强调"**能源系统**"而非"电力预测"

```
Household Energy Agent
  场景：普通家庭 / 有 PV 家庭 / 有 EV 家庭 / 储能家庭
```

#### 7. GEFCom2014 Price（电价）★★★★★（决策效能评估主线）

**为什么适合**：电价由市场均衡决定——尖峰 / 肥尾（均值 48，p99 156，max 363）、均值回归、需求驱动。
它是整个项目从"预测"走向"决策"的**价值出口**：预测误差可以直接翻译成储能套利的真实利润。

**数据结构**：15 Task、1h、**单分区（ZONEID=1）**。每 Task = `Task{k}_P.csv`（目标 `Zonal Price` +
外生「Forecasted Total/Zonal Load」，决策时点可得）+ `Benchmark{k}_P.csv`（99 分位模板）+ 官方真值
（k<15 在 `Task{k+1}_P.csv`，Task15 在 `Solution to Task15`）。timestamp 为 Load 格式（`load_single_task`
可直接解析）；预测窗口语义与 Load/Wind 略异（Task1 起于 2013-06-16，非整月），接入时需单独验证。

> ⚠️ **数据边界**：Load / Wind / Price 是**三套不同物理系统**（Load 赛道 ~150MW、Price 赛道 ~18GW、
> Wind 为 ERCOT 风电），**跨赛道的 NetLoad / 空间拓扑 不成立**。市场均衡分析必须在**电价赛道内**进行
> （"负荷→价格"凸性敏感度），储能套利评估**只用电价赛道**，自洽。

为电价设计的 Agent / 评估器：

```
Market/Price Agent
  尖峰双层：分类预警（是否尖峰）+ 数值回归（尖峰幅度）
  经济交叉特征：负荷→价格凸性敏感度（赛道内）
  LLM 归因报告：风电骤降 / 负荷激增 等事件溯源
Storage Arbitrage Evaluator（价值出口）
  LP：max Σ(P̂_t·E_dis − P̂_t·E_ch/η)  s.t. SOC/功率边界 + 期末 SOC 约束
  结算：按预测决策、按真实价格结算；Regret = 完美预知利润 − Agent 利润
```

**创新包装**：**决策导向评估**（Decision-focused Forecasting）——同一预测智能体的预测误差被
货币化为套利利润与 Regret，比单纯 RMSE 有更强的业务纵深。

---

## 三、版本演进（多能源化）

一条主线贯穿：**LLM 作为预测建模决策器，在受约束动作空间内自主决策，通过实验反馈持续优化**。动作空间从"特征"逐步扩展为"特征 + 模型 + 参数 + 集成"。

### V1.0：LLM 特征工程闭环 Agent（✅ 已完成）

- 单任务（Task 15）特征闭环：RMSE 8.51 → 5.75（↓32%）
- 证明：LLM 不仅生成特征，且具备闭环自我纠偏能力（第 4 轮退化、第 5 轮恢复）

### V2.0：滚动自适应进化双闭环智能体（P0 · 第一优先级）

```
                环境变化（季节 / 负荷模式 / 天气 / 能源类型切换）
                       │
                       ▼
        ┌─── 外循环：场景自适应闭环 ───────────┐
        │  数据漂移检测 · 季节变化 · 负荷模式变化 │
        └──────────────┬──────────────────────┘
                       │ 场景判定 + 经验召回
                       ▼
        ┌─── 内循环：模型优化闭环（V1.0）────────┐
        │  特征生成 → 模型训练 → 参数调整 → 误差分析 │
        └──────────────┬──────────────────────┘
                       │
                       ▼
            经验更新（成功/失败入库） → 策略进化
```

**多能源化**：任务序列从 `Task1→Task15`（Load）扩展为 **Load + Wind 双能源任务流**，Agent 在跨能源任务序列中积累场景经验。

| 新增模块 | 文件 | 职责 |
|----------|------|------|
| Task Replay Engine | `data/gefcom_loader.py` `task_builder.py` `task_replay.py` | Load + Wind 多任务滚动回放 |
| Drift Detection ✅（Load 版已落地，P1-B） | `evaluation/drift_detector.py` | 尾部窗口均值/方差/分位/ACF 周期/残余误差漂移检测 → score/level（确定性） |
| Strategy Migration ✅（Load 版已落地，P1-B） | `agent/strategy_migration.py` `experiments/run_outer_loop.py` | 漂移报告 → LLM 决策 继承/修改/重置 → warm-start 自进化 → 策略入库 |
| Experience Memory | `memory/feature_memory.json` `scenario_memory.json` `failure_memory.json` `memory/experiment_memory.jsonl` `strategies.jsonl` | 场景→动作→结果→经验入库 |

### V3.0：分场景多能源专家协同（P1 · 第二优先级）

```
                    Coordinator Agent（LLM 动态权重）
                            │
        ┌───────────┬───────┼───────┬───────────┬────────────┐
        │           │       │       │           │            │
   Load Expert  Wind Expert Solar   Weather    Anomaly     Base
  (负荷周期)   (随机风电)  Expert    Expert    Expert   (V1.0闭环)
                          (昼夜/天) (风速温度) (极端事件)
        │           │       │       │           │            │
        └───────────┴───────┼───────┴───────────┴────────────┘
                            ▼
                 Fusion Prediction Agent
                            │
                            ▼
                      最终预测结果
```

| 新增 Agent | 文件 | 场景示例（权重动态分配） |
|-----------|------|--------------------------|
| Load Expert | `agent/trend_agent.py`（复用） | 正常日：70% 周期 + 20% 趋势 + 10% 气象 |
| Wind Expert | `agent/wind_agent.py` | 风暴日：60% 气象 + 30% 异常 + 10% 周期 |
| Solar Expert | `agent/solar_agent.py` | 晴天：80% 周期 + 20% 气象 / 阴天：对调 |
| Weather Expert | `agent/weather_agent.py` | 高温日：40% 气象 + 30% 异常 + 30% 周期 |
| Anomaly Expert | `agent/anomaly_agent.py` | — |
| Coordinator | `agent/coordinator_agent.py` | LLM 基于 V2.0 场景判定生成权重 |

### V4.0：风险感知型概率预测（P2 · 第三优先级）

- **量化模型**：Quantile LightGBM 输出 `q0.1 / q0.5 / q0.9`
- **区间校准**：Conformal Prediction，提供 90% 预测区间
- **评估指标升级**：Pinball Loss / CRPS / Coverage / Interval Width
- **决策目标升级**：`Score = 预测误差 + 不确定性 + 稳定性 + 计算成本`（可能选择 RMSE 略高但风险更低的模型）

| 模块 | 文件 |
|------|------|
| 概率预测协调器 | `agent/uncertainty_agent.py` |

### 价值出口层：储能套利决策评估（业务/决策驱动，跨版本）

预测只是手段，**决策价值才是价值出口**。本层把 Market/Price Agent 的预测接进一个
带约束的虚拟储能电池 LP，用套利利润与 Regret 反向评估预测质量：

- **决策模型**：`max Σ(P̂_t · E_dis_t − P̂_t · E_ch_t / η)`，约束 = 电池容量 / 充放电功率 /
  往返效率 η / SOC 边界 / **期末 SOC = 期初**（防清仓套现伪造利润）
- **结算规则**：按预测价格 P̂ 做充放电决策，按真实价格 P 结算 —— 决策导向评估的定义
- **Regret** = 完美预知 LP 利润 − Agent 决策利润 = **预测误差的货币化**
- **与 V4.0 衔接**：分位数预测 → 随机套利（预测不确定性大时保守、小时激进），
  风险感知落到价值层，而非止步于 Coverage 指标

### V5.0：完整自主能源预测 AutoML（终极形态）

```
                    Energy Agent（总控）
                          │
              Task Understanding Agent
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
  数据诊断 Agent      场景识别 Agent      多专家预测 Agent (V3)
        │                 │                 │
  模型选择 Agent      超参优化 Agent      风险评估 Agent (V4)
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                   Autonomous Forecast System
```

---

## 四、创新点 ↔ 数据集验证矩阵

| 创新 | 最佳验证数据集 | 原因 | 验证指标 |
|------|--------------|------|----------|
| 创新 1：滚动自适应进化 | GEFCom Load + Wind | 天然 Task 序列，跨能源任务流 | 序列上的 RMSE 趋势、退化次数 |
| 创新 2：多专家协同 | Load + Wind + Solar | 三种能源特性差异大，专家分工可解释 | 加权融合 vs 单模型 |
| 创新 3：风险感知概率预测 | GEFCom2014（原生概率赛道） | 原始任务即分位数预测 | Pinball / CRPS / Coverage / Interval Width |
| 创新 4：经验迁移 | ECL + Pecan Street | 有大量用户/家庭，可构造迁移场景 | 有 Memory vs 无 Memory 的测试误差 |
| 创新 5：决策导向评估（储能套利） | GEFCom2014 Price | 电价肥尾尖峰，预测误差可直接货币化 | 套利利润 / Regret / 尖峰捕获率 |

---

## 五、实验矩阵（论文 / 申报）

| 实验 | 主题 | 数据 | 比较 | 指标 | 证明 |
|------|------|------|------|------|------|
| **Exp 1** | 单能源预测能力 | GEFCom Load | LightGBM / PatchTST / LLM-Agent | RMSE / MAE / MAPE | Agent 有效 |
| **Exp 2** | 跨能源泛化 | Load + Wind + Solar | 单模型 vs 多专家架构 | RMSE / 分能源对比 | 多专家架构有效 |
| **Exp 3** | 滚动自进化 | GEFCom Task1–15 | 逐 Task 指标序列 | RMSE 收敛趋势 | Agent 随任务积累变强 |
| **Exp 4** | 跨用户迁移 | ECL | 普通模型 vs 有 Memory Agent | RMSE（User 301–370） | Memory 机制有效 |
| **Exp 5** | 风险预测 | GEFCom | 点预测 vs 概率预测 | Pinball / CRPS / Coverage / Interval Width | 风险感知有效 |
| **Exp 6** | 决策价值 | GEFCom2014 Price | RMSE 最优 Agent vs 利润最优 Agent | 套利利润 / Regret / 尖峰捕获率 | 决策导向评估有效（预测能赚钱） |

> 建议论文主线：**Exp 1 → Exp 3 → Exp 2 → Exp 5 → Exp 4 → Exp 6**
> （单能源证明 → 自进化 → 泛化 → 风险 → 迁移 → 决策价值，逻辑递进）。

---

## 六、分阶段实施计划

| 优先级 | 阶段 | 内容 | 依赖 |
|--------|------|------|------|
| **P0** | V2.0 滚动自进化 | Task Replay（Load+Wind）+ Drift Detection + Experience Memory | V1.0 已就绪 |
| **P1** | V3.0 多能源专家 | Energy Expert 层级 + Coordinator | 需要 P0 的场景判定 |
| **P2** | V4.0 风险感知 | Quantile + Conformal + 概率指标 | 可在 P0 后并行 |
| **P-Value** | 决策价值闭环（价值出口） | Price 接入 + 储能套利评估器 + 尖峰双层 | 可独立于 P1/P2 推进（电价赛道数据自洽） |
| **P3** | 工程完善 | 模型选择 Agent、调优 Agent、报告 Agent | 依赖 P1/P2 |

**数据集落地顺序**：

1. 先扩 **GEFCom2014 Wind**（同一竞赛体系，数据格式与 Load 最接近，改动最小）
2. 再补 **ECL**（UCI 公开下载，无协议门槛，做跨用户迁移）
3. **ETT**（长序列，github 公开）
4. **GEFCom2014 Solar**（光伏，已在本地 `GEFCom2014-S_V2.zip`）与 **Pecan Street**（需数据协议，最后）

> ※ **GEFCom2014 Price（电价）**：定位为**决策效能评估主线（价值出口）**，不再是从属于 V4.0 的载体。
> 接入里程碑：`price_loader` + 基线 → 储能套利评估器（LP + Regret 结算）→ 尖峰双层 → 赛道内经济交叉。
> 数据已在本地 `GEFCom2014-P_V2.zip`：15 Task、单分区（ZONEID=1），目标 `Zonal Price` + 外生「预测总/分区负荷」
> （决策时点可得，同 Wind expvars 语义）；timestamp 为 Load 格式（`load_single_task` 可直接解析）；
> 预测窗口语义与 Load/Wind 略异（Task1 起于 2013-06-16，非整月），接入时需单独验证。
> 接入成本低（wind_loader 模式可复用）。⚠️ 三赛道为不同物理系统，跨赛道 NetLoad/空间拓扑不成立。

---

## 七、TODO 清单

### P0 —— 论文主线（V2.0）

- [x] **多任务滚动预测框架（Load 版）** ⭐⭐⭐⭐⭐ `data/gefcom_loader.py` + `task_builder.py` + `task_replay.py`
  - Task15 only → Load 全 15 Task 无泄漏滚动回放已落地；Wind 已接入（见下）
- [x] **Wind 风电赛道接入（P0）** ⭐⭐⭐⭐⭐ `data/wind_loader.py` + `data/wind_task_builder.py` + `evaluation/wind_replay.py` + `experiments/run_wind_replay.py`
  - 15 Task × 10 Zone 无泄漏滚动回放（逐分区独立模型，Task 得分 = 10 分区均值）
  - 气象外生特征：U10/V10/U100/V100 → 风速/风向/切变；预测月气象取 TaskExpVars 预报（决策时点可得）
  - Wind LightGBM 基线 Mean RMSE = 0.0998（归一化 [0,1] 量纲），Mean R² = 0.873（tests W1–W6 全绿）
  - 待扩：Wind 自进化 Agent 泛化（feature_agent/evolution_runner LOAD 硬编码 + feature_spec 的 source==target 限制 + memory 无 energy 字段）
  - **候选 Agent 动作**（`experiments/explore_wind_weather.py` 探索实验）：加"当前小时风速预报 ws100@t"（TaskExpVars 决策时点可得，外生非泄漏）。平均 RMSE ↓3.3%，但**收益场景相关**（Task15-z5 ↓7% / Task1-z5 ↓7.8% / Task7-z1、Task9-z1 略差 ↑0.3~0.6%）→ 不宜固化进固定特征，列为后续 Wind 自进化 Agent 的可选动作，由 Agent 按场景判定是否启用。启用需支持新的 spec 特征类型（exogenous/current，source 非目标列、lookback 0），涉及 build_features / _features_at / leakage_checker Pass A / feature_spec.normalize_spec 的扩展（均 additive，Load 零回归）
- [x] **Experience Memory（Load 版）** ⭐⭐⭐⭐⭐ `memory/experiment_memory.jsonl`（轮级）+ `memory/strategies.jsonl`（策略级）
  - 滚动自进化 + 跨 Task 迁移的基础
- [x] **Drift Detection + Strategy Migration（Load 版，P1-B）** ⭐⭐⭐⭐⭐ `evaluation/drift_detector.py` + `agent/strategy_migration.py` + `experiments/run_outer_loop.py`
  - 均值/方差/分位/周期/残余误差漂移 → 继承/修改/重置 → warm-start 双闭环
  - 待扩：极端天气漂移、Wind 能源切换场景

### P1 —— 核心创新（V3.0）

- [ ] **多能源专家协同** ⭐⭐⭐⭐⭐ `agent/coordinator_agent.py` + `wind_agent.py` + `solar_agent.py` + `weather_agent.py` + `anomaly_agent.py`
- [ ] **模型选择 Agent** ⭐⭐⭐⭐ 短期 LightGBM / 长期 PatchTST / 复杂 Transformer / Ensemble

### P2 —— 研究高度（V4.0）

- [ ] **风险感知概率预测** ⭐⭐⭐⭐⭐ `agent/uncertainty_agent.py`
  - Quantile LightGBM + Conformal + Pinball/CRPS/Coverage/Interval Width + 综合评分
  - 载体：GEFCom2014 Wind（原生分位数赛道）· Price（尖峰，与决策价值层联动，见 P-Value）
- [ ] **超参调优 Agent** ⭐⭐⭐ 不做独立 Optuna 包装，LLM 对 `model + parameter + feature + ensemble` 统一决策

### P3 —— 工程完善

- [ ] **报告生成 Agent** `agent/report_agent.py`（实验报告 / 模型比较 / 错误分析 / 可解释性）
- [ ] Transformer 基线（`models/Transformer/`，配合 ETT 长序列实验）

### P-Value —— 决策价值闭环（业务/决策驱动 · 价值出口）

- [ ] **Price 电价赛道接入** ⭐⭐⭐⭐⭐ `data/price_loader.py` + `data/price_task_builder.py` + `evaluation/price_replay.py` + `experiments/run_price_replay.py`（wind_loader 模式复用；预测窗口语义单独验证）
- [ ] **储能套利评估器** ⭐⭐⭐⭐⭐ `evaluation/arbitrage_evaluator.py` —— LP（容量/功率/η/SOC 边界 + **期末 SOC=期初**）→ 按预测决策、按真实结算 → **利润 / Regret**
- [ ] **尖峰双层模型** ⭐⭐⭐⭐ `agent/market_agent.py` —— 分类预警（是否尖峰）+ 回归估计（尖峰幅度）+ LLM 归因报告（风电骤降 / 负荷激增）
- [ ] **赛道内经济交叉特征** ⭐⭐⭐⭐ 负荷→价格凸性敏感度（二阶导异动）喂给 Agent 场景判断

---

## 八、与旧版 roadmap 的关系

旧版 `roadmap.md`（单能源版）已归档删除。本文件为其多能源升级版，新增内容：

1. 项目定位升级（单能源 → 多能源自主智能体）
2. 多数据集版图（Load / Wind / Solar / ECL / ETT / Pecan Street / Price）
3. 创新点 ↔ 数据集验证矩阵 + 6 实验矩阵
4. V2.0 任务序列扩展为跨能源；V3.0 专家能源化
5. **数据校正**：GEFCom2014 官方含 Load/Price/Solar/Wind 四赛道，Solar 数据已本地化（此前误写「无 Solar 赛道」）
6. **业务/决策驱动升维**：新增价值出口层（Price 决策效能评估主线 + 储能套利评估器），创新 5 + Exp 6，从「精度叙事」升级为「精度 + 决策价值双层叙事」
