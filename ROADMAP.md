# AutoML-Agent 演进路线图 V2.2（多能源扩展 · 自主智能体版）

> 本文件为项目演进路线的**唯一依据**（多能源扩展版）。
> 核心变化：从「单数据集（GEFCom2014 Load）特征工程闭环」升级为「**面向多能源场景的滚动自进化风险感知自主机器学习智能体**」。
> 一条主线贯穿：**LLM 作为预测建模决策器**，在受约束动作空间内自主决策**特征 / 模型 / 参数 / 输出协议（点预测 → 概率预测）**，
> 通过无泄漏实验反馈持续优化。预测不止于「准」，更在于「知道自己什么时候不确定」（风险感知）。

---

## 一、项目定位升级

| | 旧定位（V1.0） | 新定位（V2.0 起） |
|---|---|---|
| 一句话 | 在 GEFCom2014 上做 LLM 特征工程 | **面向多能源场景的滚动自进化风险感知自主机器学习智能体** |
| 数据 | 单一负荷数据集 | 负荷 + 风电 + 光伏 + 电价 + 多用户 + 长序列 + 家庭能源 |
| 能力 | 单任务特征优化 | 跨能源任务 · 跨时间演化 · 跨场景迁移 · 跨输出协议（点 → 概率） |

**数据支撑链条**：

```
GEFCom2014 Load（电力负荷）── 主论文核心
  ├── GEFCom2014 Wind（风电）── 随机性泛化
  ├── GEFCom2014 Solar（光伏）── 离散状态专家
  ├── GEFCom2014 Price（电价）── 肥尾尖峰 · 概率预测极端场景
  ├── ECL（370 用户负荷）──── 跨用户迁移
  ├── ETT（变压器长序列）──── 模型选择
  └── Pecan Street（家庭能源）── 应用展示
```

项目不再是"在 GEFCom2014 上调 LightGBM 特征"，而是"**构建一个能够跨能源任务、跨时间演化、跨场景迁移、并把预测升级为风险感知概率输出的自主机器学习智能体**"——更贴近当前 AI Agent + AutoML + Energy AI 的研究方向。

**项目架构（自主智能体，两层 + 可选应用）**：

```
┌──────────────────────────────────────────────────────────────┐
│  能力引擎层（核心）—— LLM 决策器                                │
│  自进化 Agent（V2.0 双闭环）· 多能源专家协同（V3.0）· 风险感知（V4.0）│
│  动作空间：特征 → 模型 → 参数 → 输出协议（点 / 分位数）          │
└───────────────────────────┬──────────────────────────────────┘
                            │ 无泄漏滚动回放
┌───────────────────────────┴──────────────────────────────────┐
│  底层环境层                                                    │
│  GEFCom2014 Load（负荷）· Wind（风电）· Solar（光伏）· Price（电价）│
│  Task 1–15 滚动回放 · 逐分区/逐任务独立模型                     │
└──────────────────────────────────────────────────────────────┘
            │
            ▼  （可选 · 非主线 · 应用展示附录）
   储能套利演示：把电价预测误差货币化为利润/Regret（一句话带过，证明"有用"）
```

**评估主线（精度 + 概率，均落在智能体能力上）**：
- **点预测精度**：RMSE / MAE / MAPE —— 保证点预测统计可靠，是能力的**基线锚点**
- **概率预测校准**：Pinball Loss / CRPS / Coverage / Interval Width —— 保证智能体"知道不确定"，是 V4.0 的**核心指标**

> 关系：点预测是基础，概率预测是升维。同一个自主智能体，既要预测得准（RMSE），
> 又要在肥尾/尖峰场景下输出可信的分位数（CRPS/Coverage）——这才是"自主预测建模"的完整能力，
> 而非止步于单一误差指标。GEFCom2014 本就是**分位数预测竞赛**，概率预测是回归竞赛本意，不是外挂。

---

## 二、多数据集版图

### 总览

| # | 数据集 | 任务 | 采样/规模 | 用途 | 验证创新 | 优先级 |
|---|--------|------|-----------|------|----------|--------|
| 1 | GEFCom2014 Load | 负荷 | 15 个 Task，1h | 主论文核心 | 滚动自进化 + 风险感知 | 必备 |
| 2 | GEFCom2014 Wind | 风电 | 15 Task × 10 分区（Zone），1h | 随机性泛化 | 滚动自进化 + 多专家 + 风险 | ★★★★★ |
| 3 | GEFCom2014 Solar | 光伏 | 15 Task，1h，含天气预测变量 | 离散状态专家 | 多专家协同 | ★★★★★ |
| 4 | GEFCom2014 Price | 电价 | 15 Task，1h，单分区（ZONEID=1） | **第四赛道：肥尾尖峰 · 概率极端场景** | 风险感知（尾部概率校准） | ★★★★★ |
| 5 | ECL | 用户级负荷 | 370 用户，15min | 跨用户迁移 | 时序经验记忆 | ★★★★★ |
| 6 | ETT | 变压器长序列 | 4 个变体，1h/15min | 模型选择 | 模型选择 Agent | ★★★★ |
| 7 | Pecan Street | 家庭能源 | 数百家庭，1min | 应用展示 | 家庭场景专家 | ★★★★ |

> ※ **数据校正**：GEFCom2014 官方共 **4 个赛道** —— **Load / Price / Solar / Wind**（另有 `GEFCom2014-E.xlsx` 误差分析表），均为 15 个 Task。四套原始压缩包已全部在本仓库 `GEFCom2014 Data/`（`GEFCom2014-{L,P,S,W}_V2.zip`），光伏不必另找 GEFCom2012 或 NREL/DKASC。

### 各数据集与 Agent 设计

#### 1. GEFCom2014 Load（核心，保持）

- 负荷特性：周期强、规律明显，Agent 容易学到 `lag_24` 这类基础模式
- 天然 Task1→Task15 序列，验证**滚动自进化**是否随任务积累变强
- 原任务即为概率预测（分位数赛道），天然契合**风险感知**

#### 2. GEFCom2014 Wind（风电）★★★★★

**为什么适合**：负荷周期强、规律明显；而风电**随机性强、天气驱动强、非平稳**。风电更能检验 Agent 是**真的理解了场景**，还是只是找到了 `lag_24` 这种捷径。

**数据结构**：15 个 Task × 10 个分区（Zone），1h 采样。每 Task = `benchmark{k}_W.csv`（50 分位模板）+ `Task{k}_W_Zone1_10.zip`（10 分区历史）+ `TaskExpVars{k}_W_Zone1_10.zip`（气象预测变量）；官方真值文件仅 `Solution to Task 15/solution15_W.csv`。

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

#### 4. GEFCom2014 Price（电价）★★★★★（第四赛道 · 概率预测极端场景）

**为什么适合**：电价由市场均衡决定——尖峰 / 肥尾（均值 48，p99 156，max 363）、均值回归、需求驱动。
它是四赛道里**肥尾尖峰最极端**的：点预测在长尾上必然吃亏（尖峰日 RMSE 可飙到 29），
只有概率预测才能捕捉尾部风险——**检验智能体「风险感知」能力的终极试金石**。

**数据结构**：15 Task、1h、**单分区（ZONEID=1）**，预测窗口 = **1 天（24h）**（非整月）。
每 Task = `Task{k}_P.csv`（目标 `Zonal Price` + 外生「Forecasted Total/Zonal Load」，决策时点可得）+
`Benchmark{k}_P.csv`（99 分位模板）+ 官方真值（k<15 在 `Task{k+1}_P.csv`，Task15 在 `Solution to Task15`）。
时间戳 = `MMDDYYYY H:MM`（专有解析器）。✅ 已接入（数据层 + 24h 滚动回放基线，见 P0）。

> ⚠️ **数据边界**：Load / Wind / Price 是**三套不同物理系统**（Load ~150MW、Price ~18GW、Wind 为 ERCOT 风电），
> **跨赛道 NetLoad / 空间拓扑不成立**，专家协同是"按物理系统各自建模 + Coordinator 加权"，不做跨赛道物理耦合。

为电价设计的 Agent：

```
Price Expert（电价专家）
  尖峰双层：分类预警（是否尖峰）+ 分位数回归（尾部幅度）
  外生负荷特征：Forecasted Total/Zonal Load（决策时点可得）
  场景：常规日 / 尖峰日 / 极端尖峰日
Uncertainty Expert（复用 V4.0）
  输出：P10 / P50 / P90 —— 尖峰日区间更宽、更保守
```

**创新包装**：**风险感知概率预测的极端场景验证**——同一个智能体，在规律性强的负荷上给点预测，
在肥尾尖峰的电价上自动切换到分位数输出、并放大尾部区间。

#### 5. ECL（Electricity Consumption Load）★★★★★

- 来源：UCI，`ElectricityLoadDiagrams20112014`
- 特点：370 个用户，15 分钟采样，2011–2014

**实验设计（跨用户迁移）**：

```
训练：User 1–300
测试：User 301–370
比较：普通模型 vs 带 Memory 的 Agent
```

**创新包装**：基于时序经验记忆的**跨用户迁移预测**——验证 Agent 能否把用户 A 的经验迁移到用户 B。

#### 6. ETT（Electricity Transformer Temperature）★★★★

- 经典长序列预测数据集（变压器油温 + 电力负荷 + 时间特征）
- 4 个变体：ETTh1 / ETTh2 / ETTm1 / ETTm2
- 用途：长序列预测，验证**模型选择 Agent**：

```
短期：LightGBM
长期：PatchTST
复杂：Transformer
```

#### 7. Pecan Street（家庭能源系统）★★★★

- 真实家庭数据：家庭用电 + 光伏 + EV + 储能（需数据协议）
- 强调"**能源系统**"而非"电力预测"

```
Household Energy Agent
  场景：普通家庭 / 有 PV 家庭 / 有 EV 家庭 / 储能家庭
```

---

## 三、版本演进（多能源化）

一条主线贯穿：**LLM 作为预测建模决策器，在受约束动作空间内自主决策，通过实验反馈持续优化**。动作空间从"特征"逐步扩展为"特征 + 模型 + 参数 + 输出协议"。

### V1.0：LLM 特征工程闭环 Agent（✅ 已完成）

- 单任务（Task 15）特征闭环：RMSE 8.51 → 5.75（↓32%）
- 证明：LLM 不仅生成特征，且具备闭环自我纠偏能力（第 4 轮退化、第 5 轮恢复）

### V2.0：滚动自适应进化双闭环智能体（P0 · 已完成主线）

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

**多能源化**：任务序列从 `Task1→Task15`（Load）扩展为 **Load + Wind + Price 多能源任务流**（Solar 接入中），Agent 在跨能源任务序列中积累场景经验。

| 新增模块 | 文件 | 职责 |
|----------|------|------|
| Task Replay Engine | `data/gefcom_loader.py` `task_builder.py` `task_replay.py` | Load + Wind 多任务滚动回放 |
| Drift Detection ✅（Load 版已落地，P1-B） | `evaluation/drift_detector.py` | 尾部窗口均值/方差/分位/ACF 周期/残余误差漂移检测 → score/level（确定性） |
| Strategy Migration ✅（Load 版已落地，P1-B） | `agent/strategy_migration.py` `experiments/run_outer_loop.py` | 漂移报告 → LLM 决策 继承/修改/重置 → warm-start 自进化 → 策略入库 |
| Experience Memory | `memory/experiment_memory.jsonl` `strategies.jsonl` | 场景→动作→结果→经验入库 |

### V3.0：分场景多能源专家协同（P1 · 核心创新）

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
| Price Expert | `agent/price_agent.py` | 尖峰日：分位数回归 + 尾部放大 |
| Weather Expert | `agent/weather_agent.py` | 高温日：40% 气象 + 30% 异常 + 30% 周期 |
| Anomaly Expert | `agent/anomaly_agent.py` | — |
| Coordinator | `agent/coordinator_agent.py` | LLM 基于 V2.0 场景判定生成权重 |

### V4.0：风险感知型概率预测（P1 · 核心创新，与 V3.0 并行）

- **量化模型**：Quantile LightGBM 输出 `q0.1 / q0.5 / q0.9`
- **区间校准**：Conformal Prediction，提供 90% 预测区间
- **评估指标升级**：Pinball Loss / CRPS / Coverage / Interval Width
- **决策目标升级**：`Score = 预测误差 + 不确定性 + 稳定性 + 计算成本`（可能选择 RMSE 略高但风险更低的模型）

| 模块 | 文件 |
|------|------|
| 概率预测协调器 | `agent/uncertainty_agent.py` |

**载体**：GEFCom2014 Wind（原生分位数赛道）· **Price（肥尾尖峰，尾部风险试金石）** · Load（规律基线）。

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
| 创新 2：多专家协同 | Load + Wind + Solar + Price | 四种能源特性差异大，专家分工可解释 | 加权融合 vs 单模型 |
| 创新 3：风险感知概率预测 | GEFCom2014（原生概率赛道） | 原始任务即分位数预测；电价肥尾尖峰是尾部风险试金石 | Pinball / CRPS / Coverage / Interval Width / 尖峰捕获率 |
| 创新 4：经验迁移 | ECL + Pecan Street | 有大量用户/家庭，可构造迁移场景 | 有 Memory vs 无 Memory 的测试误差 |

---

## 五、实验矩阵（论文 / 申报）

| 实验 | 主题 | 数据 | 比较 | 指标 | 证明 |
|------|------|------|------|------|------|
| **Exp 1** | 单能源预测能力 | GEFCom Load | LightGBM / PatchTST / LLM-Agent | RMSE / MAE / MAPE | Agent 有效 |
| **Exp 2** | 跨能源泛化 | Load + Wind + Solar + Price | 单模型 vs 多专家架构 | RMSE / 分能源对比 | 多专家架构有效 |
| **Exp 3** | 滚动自进化 | GEFCom Task1–15 | 逐 Task 指标序列 | RMSE 收敛趋势 | Agent 随任务积累变强 |
| **Exp 4** | 跨用户迁移 | ECL | 普通模型 vs 有 Memory Agent | RMSE（User 301–370） | Memory 机制有效 |
| **Exp 5** | 风险感知概率预测 | GEFCom Load+Wind+Price | 点预测 vs 概率预测（分位数校准） | Pinball / CRPS / Coverage / Interval Width | 风险感知有效 |
| **Exp 5b** | 尾部风险（尖峰） | GEFCom2014 Price | 点预测 vs 分位数（尖峰日） | 尖峰捕获率 / 尾部 Pinball | 概率预测在肥尾场景更可靠 |

> 建议论文主线：**Exp 1 → Exp 3 → Exp 2 → Exp 5 → Exp 4**
> （单能源证明 → 自进化 → 泛化 → 风险概率 → 迁移，逻辑递进，全程落在智能体能力上）。
> 储能套利（决策价值）作为附录一句话展示，不进主线实验。

---

## 六、分阶段实施计划

| 优先级 | 阶段 | 内容 | 依赖 |
|--------|------|------|------|
| **P0** | V2.0 滚动自进化 | Task Replay（Load+Wind+Price）+ Drift Detection + Experience Memory | V1.0 已就绪 |
| **P1** | V3.0 多能源专家 + V4.0 风险感知（并行） | Energy Expert 层级 + Coordinator；Quantile + Conformal + 概率指标 | 需要 P0 的场景判定 |
| **P3** | 工程完善 | 模型选择 Agent、调优 Agent、报告 Agent | 依赖 P1 |
| **附录** | 储能套利演示（可选） | 极简 LP 脚本，一句话展示"误差→收益" | 可独立，非主线 |

**数据集落地顺序**：

1. **GEFCom2014 Wind**（同一竞赛体系，数据格式与 Load 最接近，改动最小）✅
2. **GEFCom2014 Price**（第四赛道）✅ 已接入：数据层 + 24h 滚动回放基线 Mean RMSE 6.96
3. **GEFCom2014 Solar**（光伏，队友并行接入中，本地 `GEFCom2014-S_V2.zip`）
4. **ECL**（UCI 公开下载，无协议门槛，做跨用户迁移）
5. **ETT**（长序列，github 公开）
6. **Pecan Street**（需数据协议，最后）

> ※ **GEFCom2014 Price（电价）**：定位为**第四赛道**——跨能源泛化的一员 + **V4.0 风险感知的极端场景载体**
> （肥尾尖峰是概率预测的试金石）。储能套利评估降级为附录演示，不进主线。
> 数据语义已落地（预测窗口=1天、`MMDDYYYY` 时间戳、DST 瑕疵、两命名坑），见 `data/price_loader.py` 文件头。
> ⚠️ 三赛道为不同物理系统，跨赛道 NetLoad/空间拓扑不成立，专家协同按物理系统各自建模。

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
- [x] **Price 电价赛道接入（第四赛道）** ⭐⭐⭐⭐⭐ `data/price_loader.py` + `data/price_task_builder.py` + `evaluation/price_replay.py` + `experiments/run_price_replay.py` + `tests/test_price_suite.py`（P1–P6 全绿）
  - 预测窗口 = **1 天（24h）**（非整月），15 Task 预测 15 个特定日期（06-16…12-17）；单分区 ZONEID=1；目标 `Zonal Price`（肥尾）+ 外生 `Forecasted Total/Zonal Load`（决策时点可得）
  - 时间戳 = `MMDDYYYY H:MM`（专有解析器 `%m%d%Y`）；两个命名坑：Task7 benchmark=`Benchmark7_P_new3.csv`、solution 目录=`Solution to Task15`（无空格）
  - DST 瑕疵：2013-03-10 "01:00" 重复 + "02:00" 缺失（官方数据固有），loader 统一去重 + 容忍 2h gap
  - Price LightGBM 基线 Mean RMSE = **6.96**（online_h1），vs persistence 10.27（↓32%）；尖峰日 Task8/9/15 RMSE 18.8/29.1/12.0
- [x] **Experience Memory（Load 版）** ⭐⭐⭐⭐⭐ `memory/experiment_memory.jsonl`（轮级）+ `memory/strategies.jsonl`（策略级）
  - 滚动自进化 + 跨 Task 迁移的基础
- [x] **Drift Detection + Strategy Migration（Load 版，P1-B）** ⭐⭐⭐⭐⭐ `evaluation/drift_detector.py` + `agent/strategy_migration.py` + `experiments/run_outer_loop.py`
  - 均值/方差/分位/周期/残余误差漂移 → 继承/修改/重置 → warm-start 双闭环
  - 待扩：极端天气漂移、Wind/Price 能源切换场景

### P1 —— 核心创新（V3.0 + V4.0 并行）

- [ ] **多能源专家协同（V3.0）** ⭐⭐⭐⭐⭐ `agent/coordinator_agent.py` + `wind_agent.py` + `solar_agent.py` + `price_agent.py` + `weather_agent.py` + `anomaly_agent.py`
- [ ] **风险感知概率预测（V4.0）** ⭐⭐⭐⭐⭐ `agent/uncertainty_agent.py`
  - Quantile LightGBM + Conformal + Pinball/CRPS/Coverage/Interval Width + 综合评分
  - 载体：GEFCom2014 Wind（原生分位数赛道）· Price（肥尾尖峰，尾部风险试金石）· Load（规律基线）
- [ ] **模型选择 Agent** ⭐⭐⭐⭐ 短期 LightGBM / 长期 PatchTST / 复杂 Transformer / Ensemble
- [ ] **超参调优 Agent** ⭐⭐⭐ 不做独立 Optuna 包装，LLM 对 `model + parameter + feature + ensemble` 统一决策

### P3 —— 工程完善

- [ ] **报告生成 Agent** `agent/report_agent.py`（实验报告 / 模型比较 / 错误分析 / 可解释性）
- [ ] Transformer 基线（`models/Transformer/`，配合 ETT 长序列实验）

### 附录 —— 储能套利演示（可选 · 非主线）

> 不作为创新点 / 不进实验矩阵。仅作为"预测误差可货币化"的一句话应用展示，用极简脚本
> 证明智能体的电价预测能转化为套利收益。若时间紧可整体砍掉，不影响主线。

- [ ] **储能套利演示脚本** ⭐⭐ `evaluation/arbitrage_evaluator.py` —— LP（容量/功率/η/SOC 边界 + **期末 SOC=期初**）→ 按预测决策、按真实结算 → 利润 / Regret
- [ ] **尖峰概率子任务**（归入 V4.0）⭐⭐⭐⭐ 尖峰分类预警 + 分位数尾部估计，喂给 uncertainty_agent 做风险感知

---

## 八、与旧版 roadmap 的关系

旧版 `roadmap.md`（单能源版）已归档删除。本文件为其多能源升级版，新增内容：

1. 项目定位升级（单能源 → 多能源自主智能体）
2. 多数据集版图（Load / Wind / Solar / Price / ECL / ETT / Pecan Street）
3. 创新点 ↔ 数据集验证矩阵 + 6 实验矩阵（Exp 5b 尾部风险）
4. V2.0 任务序列扩展为跨能源；V3.0 专家能源化（含 Price Expert）
5. **数据校正**：GEFCom2014 官方含 Load/Price/Solar/Wind 四赛道，Solar 数据已本地化（此前误写「无 Solar 赛道」）

**V2.2 叙事回调（本版）**：撤销 V2.1 的「业务/决策驱动升维」——储能套利（决策价值出口）曾一度升为主线，
现**降级为附录演示**，主线回归「自主机器学习智能体」（V1→V2→V3→V4→V5 纯智能体叙事）。
Price 赛道定位从「决策价值出口」改为「第四预测赛道 + V4.0 风险感知的极端场景载体」。
核心立场不变：**LLM 作为预测建模决策器，自主决策特征 / 模型 / 输出协议，闭环自进化**——项目价值在「预测过程」，不在「预测变现」。
