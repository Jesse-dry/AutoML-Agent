# AutoML-Agent 演进路线图 V2.0（多能源扩展版）

> 本文件为项目演进路线的**唯一依据**（多能源扩展版）。
> 核心变化：从「单数据集（GEFCom2014 Load）特征工程闭环」升级为「**面向多能源场景的滚动自进化风险感知自主机器学习智能体**」。

---

## 一、项目定位升级

| | 旧定位（V1.0） | 新定位（V2.0 起） |
|---|---|---|
| 一句话 | 在 GEFCom2014 上做 LLM 特征工程 | **面向多能源场景的滚动自进化风险感知自主机器学习智能体** |
| 数据 | 单一负荷数据集 | 负荷 + 风电 + 光伏 + 多用户 + 长序列 + 家庭能源 |
| 能力 | 单任务特征优化 | 跨能源任务 · 跨时间演化 · 跨场景迁移 |

**数据支撑链条**：

```
GEFCom2014 Load（电力负荷）── 主论文
  ├── GEFCom2014 Wind（风电）── 新能源泛化
  ├── GEFCom2014 Solar → GEFCom2012 光伏（场景专家）※校正
  ├── ECL（370 用户负荷）──── 跨用户迁移
  ├── ETT（变压器长序列）──── 模型选择
  └── Pecan Street（家庭能源）── 应用展示
```

项目不再是"在 GEFCom2014 上调 LightGBM 特征"，而是"**构建一个能够跨能源任务、跨时间演化、跨场景迁移的自主预测智能体**"——更贴近当前 AI Agent + AutoML + Energy AI 的研究方向。

---

## 二、多数据集版图

### 总览

| # | 数据集 | 任务 | 采样/规模 | 用途 | 验证创新 | 优先级 |
|---|--------|------|-----------|------|----------|--------|
| 1 | GEFCom2014 Load | 负荷 | 15 个 Task，1h | 主论文核心 | 滚动自进化 + 风险感知 | 必备 |
| 2 | GEFCom2014 Wind | 风电 | 15 个 Task，1h | 新能源泛化 | 滚动自进化 + 多专家 + 风险 | ★★★★★ |
| 3 | GEFCom2012 Solar ※ | 光伏 | 含昼夜/天气 | 场景专家 | 多专家协同 | ★★★★★ |
| 4 | ECL | 用户级负荷 | 370 用户，15min | 跨用户迁移 | 时序经验记忆 | ★★★★★ |
| 5 | ETT | 变压器长序列 | 4 个变体，1h/15min | 模型选择 | 模型选择 Agent | ★★★★ |
| 6 | Pecan Street | 家庭能源 | 数百家庭，1min | 应用展示 | 家庭场景专家 | ★★★★ |

> ※ **数据校正**：GEFCom2014 官方只有 **Load / Wind / Price** 三个赛道，**没有 Solar 赛道**。光伏实验建议改用 **GEFCom2012 光伏赛道**（同竞赛体系、数据格式接近），或 NREL / DKASC 光伏数据集。多专家设计不受影响。

### 各数据集与 Agent 设计

#### 1. GEFCom2014 Load（核心，保持）

- 负荷特性：周期强、规律明显，Agent 容易学到 `lag_24` 这类基础模式
- 天然 Task1→Task15 序列，验证**滚动自进化**是否随任务积累变强
- 原任务即为概率预测（分位数赛道），天然契合**风险感知**

#### 2. GEFCom2014 Wind（风电）★★★★★

**为什么适合**：负荷周期强、规律明显；而风电**随机性强、天气驱动强、非平稳**。风电更能检验 Agent 是**真的理解了场景**，还是只是找到了 `lag_24` 这种捷径。

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

#### 3. GEFCom2012 Solar（光伏）★★★★★（数据源已校正）

**特点**：夜间为 0、白天周期、天气影响、季节变化——离散状态明显，是分场景专家的理想验证场。

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

---

## 五、实验矩阵（论文 / 申报）

| 实验 | 主题 | 数据 | 比较 | 指标 | 证明 |
|------|------|------|------|------|------|
| **Exp 1** | 单能源预测能力 | GEFCom Load | LightGBM / PatchTST / LLM-Agent | RMSE / MAE / MAPE | Agent 有效 |
| **Exp 2** | 跨能源泛化 | Load + Wind + Solar | 单模型 vs 多专家架构 | RMSE / 分能源对比 | 多专家架构有效 |
| **Exp 3** | 滚动自进化 | GEFCom Task1–15 | 逐 Task 指标序列 | RMSE 收敛趋势 | Agent 随任务积累变强 |
| **Exp 4** | 跨用户迁移 | ECL | 普通模型 vs 有 Memory Agent | RMSE（User 301–370） | Memory 机制有效 |
| **Exp 5** | 风险预测 | GEFCom | 点预测 vs 概率预测 | Pinball / CRPS / Coverage / Interval Width | 风险感知有效 |

> 建议论文主线：**Exp 1 → Exp 3 → Exp 2 → Exp 5 → Exp 4**（从单能源证明 → 自进化 → 泛化 → 风险 → 迁移，逻辑递进）。

---

## 六、分阶段实施计划

| 优先级 | 阶段 | 内容 | 依赖 |
|--------|------|------|------|
| **P0** | V2.0 滚动自进化 | Task Replay（Load+Wind）+ Drift Detection + Experience Memory | V1.0 已就绪 |
| **P1** | V3.0 多能源专家 | Energy Expert 层级 + Coordinator | 需要 P0 的场景判定 |
| **P2** | V4.0 风险感知 | Quantile + Conformal + 概率指标 | 可在 P0 后并行 |
| **P3** | 工程完善 | 模型选择 Agent、调优 Agent、报告 Agent | 依赖 P1/P2 |

**数据集落地顺序**：

1. 先扩 **GEFCom2014 Wind**（同一竞赛体系，数据格式与 Load 最接近，改动最小）
2. 再补 **ECL**（UCI 公开下载，无协议门槛，做跨用户迁移）
3. **ETT**（长序列，github 公开）
4. **GEFCom2012 Solar**（光伏，需单独获取）与 **Pecan Street**（需数据协议，最后）

---

## 七、TODO 清单

### P0 —— 论文主线（V2.0）

- [x] **多任务滚动预测框架（Load 版）** ⭐⭐⭐⭐⭐ `data/gefcom_loader.py` + `task_builder.py` + `task_replay.py`
  - Task15 only → Load 全 15 Task 无泄漏滚动回放已落地；待扩 Wind
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
- [ ] **超参调优 Agent** ⭐⭐⭐ 不做独立 Optuna 包装，LLM 对 `model + parameter + feature + ensemble` 统一决策

### P3 —— 工程完善

- [ ] **报告生成 Agent** `agent/report_agent.py`（实验报告 / 模型比较 / 错误分析 / 可解释性）
- [ ] Transformer 基线（`models/Transformer/`，配合 ETT 长序列实验）

---

## 八、与旧版 roadmap 的关系

旧版 `roadmap.md`（单能源版）已归档删除。本文件为其多能源升级版，新增内容：

1. 项目定位升级（单能源 → 多能源自主智能体）
2. 多数据集版图（Load / Wind / Solar / ECL / ETT / Pecan Street）
3. 创新点 ↔ 数据集验证矩阵 + 5 实验矩阵
4. V2.0 任务序列扩展为跨能源；V3.0 专家能源化
5. **数据校正**：GEFCom2014 无 Solar 赛道，改用 GEFCom2012 或 NREL/DKASC
