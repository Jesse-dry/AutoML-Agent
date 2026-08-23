# Solar Task 1–15 × Zone 1–3 滚动回放主循环
# ---------------------------------------------------------------
# 逐 Task → 逐 Zone：build_solar_task → 严格泄漏检查（中止违规）→ fit 独立后端
# → 滚动预测 → 指标。Task 得分 = 3 分区指标均值（逐分区独立模型）。
#
# 与 Wind 版（evaluation/wind_replay.py）同构，差异：
#   - 3 分区（非 10）
#   - 目标为 POWER（归一化光伏出力 [0,1]）
#   - 气象外生列为 VAR169/VAR164/VAR167（predictors，决策时点可得），非 U/V 派生
# 复用：evaluate_task / backends / protocols / check_feature_leakage / build_features。
# ---------------------------------------------------------------
import json
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from data.solar_loader import SOLAR_DATA_DIR, SOLAR_ZONES
from data.solar_task_builder import (
    SOLAR_FEATURE_SPEC,
    SOLAR_WEATHER_COLS,
    build_solar_forecast_features,
    build_solar_task,
)
from data.task_builder import feature_spec_hash
from evaluation.evaluator import TaskResult, evaluate_task
from evaluation.forecast_protocol import ONLINE_H1, ForecastProtocol
from evaluation.leakage_checker import check_feature_leakage
from models.replay_backends import ModelBackend

_HOUR = timedelta(hours=1)
_METRIC_COLS = ["RMSE", "MAE", "MAPE", "SMAPE", "R2", "N"]


class LeakageError(RuntimeError):
    """泄漏检查未通过，中止回放。"""


def _raise_leakage(where: str, violations) -> None:
    msgs = "\n".join(f"  - [{v.kind}] {v.message}" for v in violations[:20])
    raise LeakageError(f"[LEAK] Solar {where} 泄漏检查未通过:\n{msgs}")


# ---------------------------------------------------------------
# 滚动预测（online_h1 向量化 / recursive 逐点回填）
# ---------------------------------------------------------------

def _solar_features_at(
    observed: pd.Series,
    weather: pd.DataFrame,
    t: pd.Timestamp,
    spec: List[dict],
    target_col: str,
) -> Dict[str, float]:
    """
    单点特征计算（recursive 协议用）。与 build_features 在相同输入下逐位一致：
      目标 lag/rolling 读取 observed；气象 lag/rolling 读取 weather（外生，含预报）。
    """
    row: Dict[str, float] = {}
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
            series = observed if src == target_col else weather[src]
            row[name] = series.get(t - _HOUR * s["k"], np.nan)
        elif stype == "rolling":
            src = s["source"]
            series = observed if src == target_col else weather[src]
            w = s["window"]
            min_periods = s.get("min_periods", w)
            vals = series.loc[t - _HOUR * w : t - _HOUR].dropna()
            if len(vals) < min_periods:
                row[name] = np.nan
            else:
                row[name] = getattr(vals, s["stat"])()
    return row


def solar_rolling_predict(
    backend: ModelBackend,
    task,
    protocol: ForecastProtocol = ONLINE_H1,
    spec: List[dict] = SOLAR_FEATURE_SPEC,
) -> pd.Series:
    """
    对 Task×Zone 的预测月做逐小时滚动预测，返回以 forecast_ts 为索引的预测序列。

    气象外生列 = 历史气象 ∪ 预测月预报（predictors），决策时点可得，不随协议回填。
    """
    history_target = task.history_df[task.target_col]
    weather_full = pd.concat(
        [task.history_df[SOLAR_WEATHER_COLS],
         task.weather_forecast_df[SOLAR_WEATHER_COLS]]
    )

    if not protocol.recursive:
        observed = pd.concat([history_target, task.y_true])
        X = build_solar_forecast_features(observed, weather_full, task.forecast_ts, spec)
        y_hat = backend.predict(X)
        return pd.Series(y_hat, index=task.forecast_ts)

    # recursive：预测月内目标只能回填自己的预测值；气象仍取预报（外生）
    observed = history_target.copy()
    y_hats = []
    for t in task.forecast_ts:
        x = _solar_features_at(observed, weather_full, t, spec, task.target_col)
        yh = backend.predict(pd.DataFrame([x]))[0]
        y_hats.append(yh)
        observed.loc[t] = protocol.backfill(yh, task.y_true.loc[t])
    return pd.Series(y_hats, index=task.forecast_ts)


# ---------------------------------------------------------------
# 泄漏检查
# ---------------------------------------------------------------

def _check_solar_task_leakage(task, leak_check: str, protocol: ForecastProtocol,
                              spec: List[dict]) -> None:
    """对历史段（+ 预测窗口，非 recursive 时）做严格泄漏检查，违规即中止。

    与 Wind 版同构：Pass A 血缘静态检查 + Pass B recompute。气象外生列
    （VAR169/VAR164/VAR167）为 spec 的非目标 source，Pass A 只校验血缘元数据；
    Pass B 在含气象列的完整序列上重算，保证与构建特征逐位一致。
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
        _raise_leakage(f"Task {task.task_id} Z{task.zone} 历史段", v_hist)

    if not protocol.recursive:
        observed_full = pd.concat([task.history_df[task.target_col], task.y_true])
        weather_full = pd.concat(
            [task.history_df[SOLAR_WEATHER_COLS],
             task.weather_forecast_df[SOLAR_WEATHER_COLS]]
        )
        fw_feat = build_solar_forecast_features(observed_full, weather_full,
                                                task.forecast_ts, spec)
        full_check = pd.concat([task.history_df, fw_feat])
        full_check[task.target_col] = observed_full.values
        for c in SOLAR_WEATHER_COLS:
            full_check[c] = weather_full[c].values
        extra_fw = [task.forecast_ts[0], task.forecast_ts[-1]]
        ok_fw, v_fw = check_feature_leakage(
            full_check, spec=spec, feature_cols=task.feature_cols,
            target_col=task.target_col, mode=leak_check, extra_check_points=extra_fw,
        )
        if not ok_fw:
            _raise_leakage(f"Task {task.task_id} Z{task.zone} 预测窗口段", v_fw)


# ---------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------

def summarize_solar(zone_results: Dict[int, List[TaskResult]]) -> Dict[str, object]:
    """
    汇总 Solar 回放结果：
      detail_table — 逐 Task×Zone 一行指标
      task_table   — 每 Task 一行 = 3 分区指标均值（zone="MEAN"）
      summary      — 跨 Task（分区均值）的 mean/std/worst/best RMSE
    """
    if not zone_results:
        raise ValueError("zone_results 为空，无法汇总")

    detail_rows, task_rows = [], []
    for tid in sorted(zone_results):
        results = zone_results[tid]
        for r in results:
            row = {"task_id": tid, "zone": r.zone}
            for c in _METRIC_COLS:
                row[c] = r.metrics.get(c)
            detail_rows.append(row)
        trow = {"task_id": tid, "zone": "MEAN"}
        for c in _METRIC_COLS:
            vals = [r.metrics.get(c) for r in results if r.metrics.get(c) is not None]
            trow[c] = float(np.mean(vals)) if vals else None
        task_rows.append(trow)

    detail_table = pd.DataFrame(detail_rows)
    task_table = pd.DataFrame(task_rows).set_index("task_id")

    rmses = task_table["RMSE"].astype(float)
    first = zone_results[sorted(zone_results)[0]][0]
    summary = {
        "model": first.model,
        "protocol": first.protocol,
        "mean_rmse": float(rmses.mean()),
        "std_rmse": float(rmses.std()),
        "worst_rmse": float(rmses.max()),
        "best_rmse": float(rmses.min()),
        "worst_task": int(rmses.idxmax()),
        "best_task": int(rmses.idxmin()),
        "n_tasks": len(task_rows),
    }
    if task_table["MAE"].notna().any():
        summary["mean_mae"] = float(task_table["MAE"].astype(float).mean())
    if task_table["MAPE"].notna().any():
        summary["mean_mape"] = float(task_table["MAPE"].astype(float).mean())
    if task_table["R2"].notna().any():
        summary["mean_r2"] = float(task_table["R2"].astype(float).mean())

    return {"detail_table": detail_table, "task_table": task_table, "summary": summary}


# ---------------------------------------------------------------
# 主回放循环
# ---------------------------------------------------------------

def replay_solar(
    task_ids: Iterable[int],
    backend: ModelBackend,
    protocol: ForecastProtocol = ONLINE_H1,
    val_hours: int = 168,
    leak_check: str = "sample",
    skip_leak_check: bool = False,
    data_dir: Path = SOLAR_DATA_DIR,
    zones: Optional[List[int]] = None,
    seed: int = 42,
    outdir: Path = None,
    spec: List[dict] = SOLAR_FEATURE_SPEC,
) -> dict:
    """
    Solar 滚动回放主循环。返回 {zone_results, detail_table, task_table, summary}。
    outdir 提供时写入审计输出（predictions/*.csv + run_manifest.json + 汇总文件）。
    """
    if data_dir is None:
        data_dir = SOLAR_DATA_DIR
    if zones is None:
        zones = list(SOLAR_ZONES)

    zone_results: Dict[int, List[TaskResult]] = {}
    for tid in sorted(task_ids):
        results: List[TaskResult] = []
        for zone in zones:
            task = build_solar_task(tid, zone, data_dir, val_hours, spec)
            if not skip_leak_check:
                _check_solar_task_leakage(task, leak_check, protocol, spec)
            backend.fit(task.train_df, task.val_df, task.feature_cols,
                        task.target_col, seed)
            y_pred = solar_rolling_predict(backend, task, protocol, spec)
            tr = evaluate_task(task, y_pred, backend.name, protocol.name)
            tr.zone = zone
            results.append(tr)
        zone_results[tid] = results
        mean_rmse = float(np.mean([r.metrics["RMSE"] for r in results]))
        n_forecast = len(results[0].forecast_ts)
        print(
            f"  Task {tid:>2}: RMSE(zone-mean)={mean_rmse:.4f} "
            f"MAE(zone-mean)={float(np.mean([r.metrics['MAE'] for r in results])):.4f} "
            f"N={n_forecast}"
        )

    summary_dict = summarize_solar(zone_results)
    payload = {
        "zone_results": zone_results,
        "detail_table": summary_dict["detail_table"],
        "task_table": summary_dict["task_table"],
        "summary": summary_dict["summary"],
    }

    if outdir is not None:
        _write_outputs(zone_results, backend.name, protocol.name, seed,
                       val_hours, leak_check, sorted(task_ids), outdir,
                       skip_leak_check, spec, zones)
    return payload


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_outputs(zone_results, model, protocol, seed, val_hours, leak_check,
                   task_ids, outdir, skip_leak_check, spec, zones) -> None:
    outdir = Path(outdir)
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{model}_{protocol}_{ts}"

    summary_dict = summarize_solar(zone_results)
    summary_dict["task_table"].to_csv(outdir / f"task_summary_{base}.csv",
                                      encoding="utf-8-sig")
    summary_dict["detail_table"].to_csv(outdir / f"detail_summary_{base}.csv",
                                        encoding="utf-8-sig")
    with open(outdir / f"summary_{base}.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict["summary"], f, ensure_ascii=False, indent=2, default=str)

    for tid, results in zone_results.items():
        for r in results:
            df_pred = pd.DataFrame({
                "timestamp": r.forecast_ts,
                "y_true": r.y_true,
                "y_pred": r.y_pred,
                "error": np.asarray(r.y_true) - np.asarray(r.y_pred),
            })
            df_pred.to_csv(pred_dir / f"task_{tid:02d}_zone_{r.zone:02d}.csv",
                           index=False, encoding="utf-8-sig")

    manifest = {
        "dataset": "solar",
        "zones": list(zones),
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
