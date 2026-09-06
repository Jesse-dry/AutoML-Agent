# Price Task 1–15 滚动回放主循环
# ---------------------------------------------------------------
# 逐 Task：build_price_task → 严格泄漏检查（中止违规）→ fit 独立后端
# → 滚动预测 → 指标。预测窗口 = 1 天（24h），单分区无 Zone 维度。
#
# 与 Wind 版（evaluation/wind_replay.py）的差异：
#   - 无 Zone 维度：逐 Task 独立建模，Task 得分 = 该 Task 单一指标
#   - 预测窗口特征含外生负荷列（Forecasted Total/Zonal Load，决策时点可得）
#   - 目标为 Zonal Price（连续肥尾电价）
# 复用：evaluate_task / backends / protocols / check_feature_leakage / build_features。
# ---------------------------------------------------------------
import json
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from data.price_loader import PRICE_DATA_DIR, PRICE_EXOGENOUS_COLS
from data.price_task_builder import (
    PRICE_FEATURE_SPEC,
    build_price_forecast_features,
    build_price_task,
)
from data.task_builder import feature_spec_hash
from evaluation.evaluator import summarize, evaluate_task
from evaluation.forecast_protocol import ONLINE_H1, ForecastProtocol
from evaluation.leakage_checker import check_feature_leakage
from models.replay_backends import ModelBackend

_HOUR = timedelta(hours=1)


class LeakageError(RuntimeError):
    """泄漏检查未通过，中止回放。"""


def _raise_leakage(where: str, violations) -> None:
    msgs = "\n".join(f"  - [{v.kind}] {v.message}" for v in violations[:20])
    raise LeakageError(f"[LEAK] Price {where} 泄漏检查未通过:\n{msgs}")


# ---------------------------------------------------------------
# 滚动预测（online_h1 向量化 / recursive 逐点回填）
# ---------------------------------------------------------------

def _price_features_at(
    observed: pd.Series,
    exogenous: pd.DataFrame,
    t: pd.Timestamp,
    spec: List[dict],
    target_col: str,
) -> dict:
    """
    单点特征计算（recursive 协议用）。与 build_features 在相同输入下逐位一致：
      目标 lag/rolling 读取 observed；外生 lag/rolling 读取 exogenous（外生，含预报）。
    """
    row: dict = {}
    for s in spec:
        stype, name = s["type"], s["name"]
        if stype == "time":
            attr = s["attr"]
            if attr == "hour":
                row[name] = t.hour
            elif attr == "weekday":
                row[name] = t.weekday()
            elif attr == "month":
                row[name] = t.month
            elif attr == "is_weekend":
                row[name] = 1 if t.weekday() >= 5 else 0
        elif stype == "lag":
            src = s["source"]
            series = observed if src == target_col else exogenous[src]
            row[name] = series.get(t - _HOUR * s["k"], np.nan)
        elif stype == "rolling":
            src = s["source"]
            series = observed if src == target_col else exogenous[src]
            w = s["window"]
            min_periods = s.get("min_periods", w)
            vals = series.loc[t - _HOUR * w : t - _HOUR].dropna()
            if len(vals) < min_periods:
                row[name] = np.nan
            else:
                row[name] = getattr(vals, s["stat"])()
    return row


def price_rolling_predict(
    backend: ModelBackend,
    task,
    protocol: ForecastProtocol = ONLINE_H1,
    spec: List[dict] = PRICE_FEATURE_SPEC,
) -> pd.Series:
    """
    对 Task 的预测日（24h）做逐小时滚动预测，返回以 forecast_ts 为索引的预测序列。

    外生负荷列 = 历史外生 ∪ 预测日预报（train 文件预测日段），决策时点可得，不随协议回填。
    """
    history_target = task.history_df[task.target_col]
    exo_full = pd.concat(
        [task.history_df[PRICE_EXOGENOUS_COLS],
         task.exogenous_forecast_df[PRICE_EXOGENOUS_COLS]]
    )

    if not protocol.recursive:
        observed = pd.concat([history_target, task.y_true])
        X = build_price_forecast_features(observed, exo_full, task.forecast_ts, spec)
        y_hat = backend.predict(X)
        return pd.Series(y_hat, index=task.forecast_ts)

    # recursive：预测日内目标只能回填自己的预测值；外生仍取预报（外生）
    observed = history_target.copy()
    y_hats = []
    for t in task.forecast_ts:
        x = _price_features_at(observed, exo_full, t, spec, task.target_col)
        yh = backend.predict(pd.DataFrame([x]))[0]
        y_hats.append(yh)
        observed.loc[t] = protocol.backfill(yh, task.y_true.loc[t])
    return pd.Series(y_hats, index=task.forecast_ts)


# ---------------------------------------------------------------
# 泄漏检查
# ---------------------------------------------------------------

def _check_price_task_leakage(task, leak_check: str, protocol: ForecastProtocol,
                              spec: List[dict]) -> None:
    """对历史段（+ 预测窗口，非 recursive 时）做严格泄漏检查，违规即中止。

    与 Wind 版同构：Pass A 血缘静态检查 + Pass B recompute。外生负荷列
    （Forecasted Total/Zonal Load）为 spec 的非目标 source，Pass A 只校验血缘元数据；
    Pass B 在含外生列的完整序列上重算，保证与构建特征逐位一致。
    """
    extra_points = [
        task.train_df.index[-1],
        task.val_df.index[0],
        task.val_df.index[-1],
        task.forecast_ts[0],
    ]
    ok_hist, v_hist = check_feature_leakage(
        task.history_df, spec=spec, feature_cols=task.feature_cols,
        target_col=task.target_col, mode=leak_check, extra_check_points=extra_points,
    )
    if not ok_hist:
        _raise_leakage(f"Task {task.task_id} 历史段", v_hist)

    if not protocol.recursive:
        observed_full = pd.concat([task.history_df[task.target_col], task.y_true])
        exo_full = pd.concat(
            [task.history_df[PRICE_EXOGENOUS_COLS],
             task.exogenous_forecast_df[PRICE_EXOGENOUS_COLS]]
        )
        fw_feat = build_price_forecast_features(observed_full, exo_full,
                                                task.forecast_ts, spec)
        full_check = pd.concat([task.history_df, fw_feat])
        full_check[task.target_col] = observed_full.values
        for c in PRICE_EXOGENOUS_COLS:
            full_check[c] = exo_full[c].values
        extra_fw = [task.forecast_ts[0], task.forecast_ts[-1]]
        ok_fw, v_fw = check_feature_leakage(
            full_check, spec=spec, feature_cols=task.feature_cols,
            target_col=task.target_col, mode=leak_check, extra_check_points=extra_fw,
        )
        if not ok_fw:
            _raise_leakage(f"Task {task.task_id} 预测窗口段", v_fw)


# ---------------------------------------------------------------
# 主回放循环
# ---------------------------------------------------------------

def replay_price(
    task_ids: Iterable[int],
    backend: ModelBackend,
    protocol: ForecastProtocol = ONLINE_H1,
    val_hours: int = 168,
    leak_check: str = "sample",
    skip_leak_check: bool = False,
    data_dir: Path = PRICE_DATA_DIR,
    seed: int = 42,
    outdir: Path = None,
    spec: List[dict] = PRICE_FEATURE_SPEC,
) -> dict:
    """
    Price 滚动回放主循环。返回 {results, table, summary}。
    outdir 提供时写入审计输出（predictions/*.csv + run_manifest.json + 汇总文件）。
    """
    if data_dir is None:
        data_dir = PRICE_DATA_DIR

    results = []
    for tid in sorted(task_ids):
        task = build_price_task(tid, data_dir, val_hours, spec)
        if not skip_leak_check:
            _check_price_task_leakage(task, leak_check, protocol, spec)
        backend.fit(task.train_df, task.val_df, task.feature_cols,
                    task.target_col, seed)
        y_pred = price_rolling_predict(backend, task, protocol, spec)
        tr = evaluate_task(task, y_pred, backend.name, protocol.name)
        results.append(tr)
        print(
            f"  Task {tid:>2}: RMSE={tr.metrics['RMSE']:.4f} "
            f"MAE={tr.metrics['MAE']:.4f} MAPE={tr.metrics['MAPE']:.2f}% "
            f"N={len(tr.forecast_ts)}"
        )

    summary_dict = summarize(results)
    payload = {
        "results": results,
        "table": summary_dict["table"],
        "summary": summary_dict["summary"],
    }

    if outdir is not None:
        _write_outputs(results, backend.name, protocol.name, seed,
                       val_hours, leak_check, sorted(task_ids), outdir,
                       skip_leak_check, spec)
    return payload


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_outputs(results, model, protocol, seed, val_hours, leak_check,
                   task_ids, outdir, skip_leak_check, spec) -> None:
    outdir = Path(outdir)
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{model}_{protocol}_{ts}"

    summary_dict = summarize(results)
    summary_dict["table"].to_csv(outdir / f"task_summary_{base}.csv",
                                 encoding="utf-8-sig")
    with open(outdir / f"summary_{base}.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict["summary"], f, ensure_ascii=False, indent=2, default=str)

    for r in results:
        df_pred = pd.DataFrame({
            "timestamp": r.forecast_ts,
            "y_true": r.y_true,
            "y_pred": r.y_pred,
            "error": np.asarray(r.y_true) - np.asarray(r.y_pred),
        })
        df_pred.to_csv(pred_dir / f"task_{r.task_id:02d}.csv",
                       index=False, encoding="utf-8-sig")

    manifest = {
        "dataset": "price",
        "protocol": protocol,
        "feature_spec_hash": feature_spec_hash(spec),
        "feature_spec": spec,
        "seed": seed,
        "val_hours": val_hours,
        "model": model,
        "tasks": list(task_ids),
        "git_commit": _git_commit(),
        "leakage_check": "skipped" if skip_leak_check else leak_check,
        "leakage_result": "passed",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tasks": len(task_ids),
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  审计输出 -> {outdir}")
