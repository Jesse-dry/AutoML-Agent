# ECL 跨用户迁移赛道接入任务书

> 给你的一句话任务：把 UCI 的 **ECL（ElectricityLoadDiagrams20112014）** 数据集接进项目，产出「数据层 + 跨用户迁移基线 + 测试」，让 ECL 成为 Load/Wind/Solar/Price 之后的**第 5 个数据集**。
>
> ⚠️ **这次不是「照葫芦画瓢」**：Wind/Solar/Price 都是「照 Wind 抄」（滚动回放那套），但 ECL 是**跨用户迁移**——没有 Task/benchmark/predictors/solution，没有「预测月」。你必须先和负责人对齐接入范式，再写代码。红线照旧（§7）。

---

## 0. 先搞懂「项目在干嘛」+ ECL 的定位

读三份文档（仓库根目录）：

1. `README.md` —— 项目整体在做什么
2. `ROADMAP.md` —— 重点看「二、多数据集版图」里 **ECL 那一段**（第 5 行）和「五、实验矩阵」的 **Exp 4 跨用户迁移**
3. `CLAUDE.md`（如果本地有）—— 常用命令

**ECL 在项目里的定位**（一句话）：验证 **Exp 4 经验迁移** —— Agent 能不能把「用户 A 的经验」迁移到「用户 B」。

- 训练：User 1–300 的负荷
- 测试：User 301–370 的负荷（模型**从没见过的用户**）
- 比较：**普通模型 vs 带 Memory 的 Agent**（创新点是「时序经验记忆」）

这和 GEFCom 四赛道是**两回事**：
| | GEFCom（L/W/S/P） | ECL |
|---|---|---|
| 评测范式 | 时间序列滚动回放（Task 1→15 预测未来月） | **跨用户迁移**（train 用户 → test 新用户） |
| 目标列 | 单 target（LOAD/TARGETVAR/POWER/Zonal Price） | **每用户一列**（370 个目标序列） |
| 真值来源 | benchmark + solution 官方文件 | **无官方预测窗口**，需自定义 |
| 采样 | 1 小时 | **15 分钟**（要聚合到 1h） |

> 所以「数据字典」这一步（§2）不是确认 benchmark 文件名，而是**确认「迁移实验」到底怎么定义**——这个必须先和负责人对齐，否则后面全错。

---

## 1. 数据源（已查证，不用你找）

- **下载**：https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip （约 249MB 压缩）
- **解压后主文件**：`LD2011_2014.txt`（约 678MB）
- **格式**：分号 `;` 分隔；第一列 `yyyy-mm-dd hh:mm:ss`（15min 粒度），其余 **370 列 = 370 个用户**，值为 kW 的浮点
- **已知坑**（实测确认）：
  - 15min 粒度：每天 96 个点；**聚合到 1h = 每 4 点求和**（kW×0.25h 累加 = kWh，直接 sum 即可）
  - DST：3 月有 23h 天（1:00–2:00 零值）、10 月有 25h 天（1:00–2:00 聚合了两小时）——聚合 1h 后要容忍这个
  - 官方说「无缺失」，但 DST 天的时间轴不规范，聚合时要防「时间戳重复/缺失」

---

## 2. 第一步交付：ECL 数据字典 + 迁移实验设计（先做这个，别急着写代码）

**写一份说明，回答这几个问题**（这是负责人 review 的关口）：

1. **时间聚合**：15min → 1h 怎么聚（sum？）？DST 的 23h/25h 天聚合后时间戳长什么样？聚合后每个用户有多少小时？
2. **用户切分**：370 个用户怎么切成 train(1–300)/test(301–370)？用户的列名长什么样（`MT_001`？）？
3. **迁移实验的预测目标**：给定 test 用户的历史（前 N 小时），预测它的未来负荷？还是「用 train 用户训练一个用户无关模型，直接预测 test 用户整段」？
4. **评测协议**：跨用户迁移下，怎么算 RMSE？（逐用户算再平均？还是所有 test 用户拼起来算一个总 RMSE？）——**这个必须和负责人确认**
5. **数据量**：370 用户 × 3 年 × 1h ≈ 970 万行。首版要不要采样用户（比如 train 抽 50 个、test 抽 10 个）来降本？

> 这一步可以用 AI 写探查脚本（`pd.read_csv(sep=';')` + 时间戳解析 + 聚合），但**结论和设计必须你自己写**，且 §4 的实验设计要负责人点头再动代码。

---

## 3. 第二步：接入（新范式，参考但别照抄）

参考现有文件（理解「怎么复用共享核心」，不是「照着改」）：

| 参考文件 | 学什么 |
|---|---|
| `data/price_loader.py` | 单一目标列的加载 + 专有时间戳解析器（ECL 要写自己的 15min→1h 解析） |
| `data/task_builder.py` 的 `build_features` | 严格过去向特征构造器（**复用，不改**） |
| `evaluation/leakage_checker.py` | 时间因果红线（**复用，不改**） |

**新增文件**（照赛道命名惯例）：
1. `data/ecl_loader.py` —— 读 LD2011_2014.txt，15min→1h 聚合，按用户切分，产出「用户 × 时间」的负荷矩阵
2. `data/ecl_task_builder.py` —— 迁移任务构建：train 用户集合 → 训练特征；test 用户 → 预测特征（复用 `build_features`）
3. `evaluation/ecl_replay.py` —— 跨用户迁移评测（**不是滚动回放**，是新协议，先和负责人定）
4. `experiments/run_ecl_replay.py` —— CLI 入口
5. `tests/test_ecl_suite.py` —— E1–E6（照 S1–S6 的覆盖思路，但语义换成迁移）

> 核心提醒：`ecl_*.py` 里 `import` 共享的 `build_features` / `check_feature_leakage`，**不要改它们**（§7）。

---

## 4. 第三步：测试 + 基线数字

1. `tests/test_ecl_suite.py` 覆盖（照 S1–S6 思路改语义）：
   - E1 Loader：时间聚合正确（15min→1h 无重复/缺失）、用户列名正确
   - E2 切分正确：train 用户 1–300、test 用户 301–370，无泄漏（test 用户不参与训练）
   - E3 迁移任务构建：特征列齐全、train/test 无 NaN 目标
   - E4 泄漏检查：迁移协议下无未来信息（**这是重点**，迁移场景的时间因果要单独想清楚）
   - E5 persistence 冒烟
   - E6 lightgbm 冒烟（train 用户训练 → test 用户预测，RMSE 合理量纲）
2. 跑出跨用户迁移基线（LightGBM），和 GEFCom 基线并列记一行。

**验收**：E1–E6 全绿 + 一个合理的迁移基线 RMSE。ECL 的负荷量纲是「每用户 kW·h」，和 GEFCom 的「归一化/兆瓦」不同，**基线数字是否合理要负责人判断**。

---

## 5. 交付清单（做完这些才算完成）

- [ ] `data/ecl_loader.py` + 文件头数据语义说明（§2 的数据字典）
- [ ] `data/ecl_task_builder.py`
- [ ] `evaluation/ecl_replay.py`（跨用户迁移协议，已和负责人对齐）
- [ ] `experiments/run_ecl_replay.py`
- [ ] `tests/test_ecl_suite.py`（E1–E6 全绿）
- [ ] ECL LightGBM 迁移基线 RMSE
- [ ] 没改任何共享核心文件（§7）
- [ ] `.gitignore` 加 ECL 数据目录（不入库）

---

## 6. AI 协作 playbook（同 Solar 版，但更强调「先对齐再写」）

1. **先让 AI 读范式**：喂给它 GEFCom 四赛道的 loader/replay，让它总结「接入一个数据集的套路」，你对照理解「数据从哪来、怎么防泄漏」。
2. **数据字典 + 实验设计自己写**（§2），AI 帮你写探查脚本，但「迁移协议怎么定义」是你和负责人的事。
3. **一小步一验证**：AI 写 `ecl_loader.py` 一个函数 → 你跑测试 → 通过再写下一个。**绝不让它一口气吐四五个文件**。
4. **拿测试当裁判**：E1–E6 全绿才算数。
5. **迁移场景的时间因果**（§4 的 E4）是最容易错的地方——迁移协议里「train 用户的历史」和「test 用户的历史」怎么不泄漏，单独想清楚再写。

---

## 7. 红线（现阶段一个字别改，除非负责人 review）

- `evaluation/leakage_checker.py` —— 时间因果红线
- `agent/feature_spec.py` 里的 `normalize_spec` —— 特征血缘/因果语义
- `agent/evolution_runner.py` / `agent/evolution_schema.py` —— 自进化 Agent 闭环（刚加了模型选择）
- `data/task_builder.py` 里的 `build_features` —— 严格过去向特征构造器

你的工作模式 = **只新增 `ecl_*.py`，共享核心只读（import 不 modify）**。

---

## 8. 进度检查点（每个做完给负责人看一次）

1. §0 + §2 完成（能讲清项目 + ECL 数据字典 + **迁移实验设计**）→ 负责人确认「迁移协议」没跑偏
2. `ecl_loader.py` + `ecl_task_builder.py` 跑通最小迁移基线 → 负责人 review 数据/泄漏
3. E1–E6 全绿 + 基线数字 → 负责人 review 后合并

> 卡住了先别硬扛，把「卡在哪 + 报错 + 你试过什么」整理清楚再问。
