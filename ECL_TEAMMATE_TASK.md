# ECL 后续任务书：基线归档与经验记忆迁移实验

> 负责人：miao-419  
> 当前主线：`main`  参考提交：`22d3fc2`  
> 前置修复：`d7a3793`（统一 ECL 评测协议、严格时间切分、实验产物审计）

## 0. 当前状态

ECL 的数据层、LightGBM 迁移基线、PatchTST 迁移基线和统一评测协议已经完成，不要重复实现以下内容：

- 260 train 用户 / 61 test 用户，`seed=42`；
- train `< 2014-06-01`，validation 为 2014 年 6 月，test 为 2014-07 至 2014-12；
- online one-step ahead；
- PatchTST 按目标时间切分 validation；
- LightGBM、PatchTST、persistence、seasonal naive 使用同一测试时间段和有效样本掩码。

统一协议下 README 记录的结果为：PatchTST Mean RMSE `135.7`、Median RMSE `67.5`，93.4% 的测试用户优于 persistence；LightGBM Mean RMSE `233.5`。

## 1. 第一阶段：实验产物与回归测试收尾

### 任务 1.1：确认完整实验产物

在有 ECL 数据的本地或服务器上，确认以下命令可以运行：

```bash
python experiments/run_ecl_replay.py --model persistence --n-train 260 \
  --outdir experiments/output/ecl_replay/persistence

python experiments/run_ecl_replay.py --model lightgbm --n-train 260 \
  --outdir experiments/output/ecl_replay/lightgbm

python models/PatchTST/ecl_patchtst_migration.py \
  --n-train 260 --epochs 25 --seed 42
```

确认产物目录至少包含：

```text
run_manifest.json
metrics_summary.json
per_user_metrics.csv
predictions.csv
```

PatchTST 还必须包含：

```text
training_history.csv
best_model.pt
```

### 任务 1.2：补充 ECL 协议回归测试

在 `tests/test_ecl_suite.py` 中增加最小测试，覆盖：

- train/validation 目标时间严格不重叠；
- test 预测时间只覆盖 2014-07-01 至 2014-12-31；
- 四种指标使用相同有效样本数；
- `metrics_summary.json` 和 `per_user_metrics.csv` 字段齐全；
- `predictions.csv` 的时间戳、用户和预测长度一致。

不要为了测试加载完整 PatchTST 训练；协议函数和小型合成数据即可完成这些检查。

### 第一阶段验收

- E1-E6 全部通过；
- 新增协议回归测试全部通过；
- 产物可以从 CSV/JSON 重新计算主要指标；
- 提供一次完整运行的产物目录，不把 ECL 原始数据、模型权重和 `experiments/output/` 提交到 Git。

## 2. 第二阶段：Exp 4 经验记忆对照实验

ECL 接入项目的核心目的不是只比较 LightGBM 和 PatchTST，而是验证：**带经验记忆的 Agent 是否比无记忆策略更能迁移到未见用户**。

### 任务 2.1：定义两个对照组

至少实现并保持同一 ECL 协议：

1. **No-Memory baseline**：每个实验只使用当前 train 用户数据，不读取历史实验记忆；
2. **Memory Agent**：允许读取历史用户/任务经验，根据当前用户历史统计选择或调整特征策略。

两个对照组必须保持一致：

- train/test 用户划分；
- 时间边界；
- 特征可用性；
- 模型后端；
- 随机种子；
- 评测指标和有效样本掩码。

### 任务 2.2：ECL Agent 接入边界

优先做一个最小可审计版本，不要直接改动全局 Agent 核心：

- 可以新增 `agent/ecl_memory_runner.py` 或等价的 ECL 适配器；
- 复用 `memory/memory_manager.py` 的读写接口；
- 复用已有特征构造和泄漏检查；
- 经验记录必须带 `dataset="ecl"`、用户/实验标识、动作、结果和时间边界；
- 不要把 test 用户标签写入训练阶段的经验记忆；
- 不要修改 `data/task_builder.py`、`evaluation/leakage_checker.py`、`agent/feature_spec.py` 等共享核心，除非先和负责人确认。

### 任务 2.3：实验结果

至少输出：

| 对照组 | Mean RMSE | Median RMSE | Ratio vs persistence | 胜出用户比例 |
|---|---:|---:|---:|---:|
| No-Memory |  |  |  |  |
| Memory Agent |  |  |  |  |

同时报告逐用户结果，避免只看总体平均值掩盖少数大用户的影响。

### 第二阶段验收

- No-Memory 与 Memory Agent 的协议、用户切分、时间窗口完全一致；
- 经验记忆不会读取 test 标签；
- 能从实验产物追溯每次策略决策和对应结果；
- 至少有一次固定 seed=42 的完整运行；
- 如果 Memory Agent 没有优于 No-Memory，要如实报告，不通过调指标或改变测试协议“修正”结果。

## 3. 推荐工作顺序

```text
确认最新 main
  -> 第一阶段冒烟
  -> 第一阶段完整运行
  -> 补协议回归测试
  -> 提交基线产物和 commit
  -> 负责人 review
  -> 第二阶段 No-Memory / Memory 对照实验
```

先完成第一阶段并回传结果，再开始第二阶段。这样可以把“基线协议问题”和“经验记忆算法问题”分开定位。

## 4. 回传模板

完成后请回传：

```text
分支：
commit：

运行环境：
Python/PyTorch：
数据文件：

运行命令：

测试结果：

产物目录：

主要结果：

协议或泄漏风险：

未完成事项：
```

## 5. Git 边界

- 从最新 `origin/main` 建分支，例如 `feat/ecl-memory-eval`；
- 只提交源码、测试和必要的短文档；
- 不提交 `ECL/electricity.txt`、模型权重、完整预测产物和 API Key；
- 提交前运行：

```bash
git status --short
git diff --check
```
