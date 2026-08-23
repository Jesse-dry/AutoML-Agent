# Solar（光伏）赛道接入任务书

> 给你的一句话任务：把 GEFCom2014 光伏赛道接进项目，**照 Wind 那套流程**，产出「数据层 + 无泄漏滚动回放基线 + 测试」，让 Solar 和 Load / Wind 并列成第三块能源。
>
> 你不是从零设计，你是「照葫芦画瓢」——所有参考代码都在仓库里，你只新建 `solar_*.py`，共享核心一个字都别改（见 §7 红线）。

---

## 0. 先花半天搞懂「这项目在干嘛」

读三份文档（都在仓库根目录）：

1. `README.md` —— 项目整体在做什么
2. `ROADMAP.md` —— 只看「一、项目定位」和「二、多数据集版图」里的 **Solar 那一段**
3. `CLAUDE.md`（如果本地有）—— 常用命令

然后跑两条命令，把数字记下来（这就是「基线」，Solar 做完要跟它们并列）：

```bash
# Load 负荷基线（Task 15 LightGBM RMSE ≈ 8.51）
.venv/Scripts/python.exe experiments/run_task_replay.py --tasks 1:15 --model lightgbm --protocol online_h1

# Wind 风电基线（LightGBM RMSE ≈ 0.0998，归一化 [0,1] 量纲）
.venv/Scripts/python.exe experiments/run_wind_replay.py --tasks 1:15 --model lightgbm
```

> 跑通后你要能回答三个问题：**①目标是什么 ②RMSE 越小越好、但 Load/Wind 量纲为什么差这么多 ③为什么「不能偷看未来数据」。** 第 ③ 个答不上来就回去重读 ROADMAP 的「时间因果」部分——这是整个项目的命门。

---

## 1. 你的「教材」：5 个要照抄的文件

| 参考文件（读 + 抄） | 它是干嘛的 |
|---|---|
| `data/wind_loader.py` | 数据加载：时间戳解析 + 真值解析 + 缺失插值 |
| `data/wind_task_builder.py` | 特征 spec + 任务构建（train/val 切分） |
| `evaluation/wind_replay.py` | 滚动回放主循环（泄漏检查 + 预测 + 指标） |
| `experiments/run_wind_replay.py` | CLI 入口 |
| `tests/test_wind_suite.py` | W1–W6 测试（你要抄成 S1–S6） |

**读的顺序**：先读 `wind_loader.py` 的文件头注释（数据语义都写在里面），再读 `wind_task_builder.py` 的「因果性约定」注释，最后看 `test_wind_suite.py` 里 W1–W6 各在测什么。

---

## 2. 第一步交付：Solar 数据字典（先做这个，别急着写代码）

数据在这里：`GEFCom2014 Data/GEFCom2014-S_V2.zip`（36MB，是个压缩包，先解压）。

**你要搞清楚并写成一份说明**（照 `wind_loader.py` 文件头注释的格式）：

1. **时间戳格式**：Solar 的 TIMESTAMP 长什么样？（Load 是 `MMDDYY H:MM`，Wind 是 `YYYYMMDD H:MM`，Solar 可能是第三种——必须实测确认，别猜）
2. **列名 + 目标列**：目标列叫什么（Load=LOAD，Wind=TARGETVAR，Solar=？）
3. **真值从哪来**：Task 1–14 的真值在哪个文件？Task 15 是不是 `Solution to Task 15` 下的文件？
4. **数据文件结构**：ROADMAP 说每 Task = `train{k}.csv` + `benchmark{k}.csv` + `predictors{k}.csv`（天气变量）——核实是不是这样，是普通 CSV 还是像 Wind 那样套了 zip。
5. **天气 predictor 字段**：`predictors{k}.csv` 里有哪些列（辐照度/温度/云量？），它们跟 `benchmark{k}.csv` 时间戳怎么对齐。
6. **光伏特有现象**：夜间出力恒 0 的占比、缺失率。

> 这一步可以用 AI 帮你读 CSV（让它写个 `pd.read_csv` + `head()` + 时间戳解析的探查脚本），但**结论必须你自己写**。这一步没做对，后面全错。

---

## 3. 第二步：照抄生成 `solar_*.py`

按这个顺序，**每写一个文件就马上跑一次测试/冒烟**（见 §6 playbook）：

1. `data/solar_loader.py` ← 抄 `wind_loader.py`
   - 大概率比 Wind 简单：Solar 可能是普通 CSV，没有 Wind 那种「zip 套 zip」。
   - 时间戳解析器要按你 §2 实测到的格式写，**别复用 Load 的**（那是有歧义的变长解析器）。
2. `data/solar_task_builder.py` ← 抄 `wind_task_builder.py`
   - 先做**最小 spec**：时间特征 + 目标列 lag/rolling（照 Load 的基础特征集），跑通再说。
   - 再把 `predictors` 里的天气变量当**外生特征**加进去（照 Wind 的 `ws10/ws100` 做法）。
   - 光伏的「夜间恒 0」是特征设计的关键：时间特征 `hour` 已经隐含了昼夜，先观察基线再决定要不要额外处理。
3. `evaluation/solar_replay.py` ← 抄 `wind_replay.py`
4. `experiments/run_solar_replay.py` ← 抄 `run_wind_replay.py`

> 核心红线提醒：`solar_*.py` 里要 `import` 共享的 `build_features` / `check_feature_leakage`，但**不要改它们**（§7）。

---

## 4. 第三步：测试 + 基线数字

1. 把 `test_wind_suite.py` 的 **W1–W6 抄成 S1–S6**（`tests/test_solar_suite.py`）：
   - W1 Loader → S1 数据加载正确性（时间连续、预测起点=历史终点+1h）
   - W2 真值一致性 → S2
   - W3 任务构建 → S3
   - W4 预测窗口特征 → S4
   - W5 persistence 冒烟 → S5
   - W6 lightgbm 冒烟 → S6
2. 跑通 S1–S6，再跑出基线：

```bash
.venv/Scripts/python.exe tests/test_solar_suite.py
.venv/Scripts/python.exe experiments/run_solar_replay.py --tasks 1:15 --model lightgbm
```

**验收**：S1–S6 全绿 + 一个合理的 Solar LightGBM baseline RMSE。合理量纲要和 Wind（≈0.0998）同口径（都是归一化出力），明显离谱（比如 >1 或 <0.001）就是时间戳解析或真值对齐错了。

---

## 5. 交付清单（做完这些才算完成）

- [ ] `data/solar_loader.py` + 文件头数据语义说明（§2 的数据字典）
- [ ] `data/solar_task_builder.py`
- [ ] `evaluation/solar_replay.py`
- [ ] `experiments/run_solar_replay.py`
- [ ] `tests/test_solar_suite.py`（S1–S6 全绿）
- [ ] Solar LightGBM baseline RMSE（和 Load 8.51 / Wind 0.0998 并列的那张表补一行）
- [ ] 没改任何共享核心文件（§7）

---

## 6. AI 协作 playbook（你最大的杠杆，但要用对）

1. **先让 AI 读范式**：把 §1 那 5 个文件喂给 AI，让它总结「接入一个能源赛道的固定套路」，你读它的总结 + 对照文件头注释，确认理解「数据从哪来、真值从哪来、怎么防泄漏」三件事。
2. **数据字典自己写**（§2），AI 可以帮你写探查脚本，但结论和判断是你的事。
3. **让 AI 生成代码，但一小步一验证**：让 AI 写 `solar_loader.py` 的一个函数 → 你马上跑测试 → 通过再让它写下一个。**绝不让它一口气吐四五个文件再一起跑**——AI 会自信地写错，你要靠「小步验证」把它逼到对为止。
4. **拿测试当裁判**：AI 说「写好了」不算数，`S1–S6` 全绿才算数。测试失败时把报错贴回给 AI 让它改。
5. **最后 sanity check**：基线数字和 Load/Wind 不在一个合理量纲 → 一定有问题，别急着提交。

---

## 7. 红线（现阶段一个字别改，除非负责人 review）

- `evaluation/leakage_checker.py` —— 时间因果红线（防偷看未来）
- `agent/feature_spec.py` 里的 `normalize_spec` —— 特征血缘/因果语义
- `agent/evolution_runner.py` / `agent/evolution_schema.py` —— 自进化 Agent 闭环
- `data/task_builder.py` 里的 `build_features` —— 严格过去向特征构造器

你的工作模式 = **只新增 `solar_*.py`，共享核心只读（import 不 modify）**。这跟 Wind 接入时的做法完全一致。

---

## 8. 进度检查点（每个做完给负责人看一次）

1. §0 + §2 完成（能讲清项目 + 数据字典）→ 负责人确认 Solar 语义没跑偏
2. `solar_loader.py` + `solar_task_builder.py` 跑通最小基线 → 负责人 review 数据/泄漏
3. S1–S6 全绿 + baseline 数字 → 负责人 review 后合并

> 卡住了先别硬扛，把「卡在哪 + 报错 + 你试过什么」整理清楚再问，比「我不懂」更好定位。
