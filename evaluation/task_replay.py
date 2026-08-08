# Task 1–15 滚动回放主循环
# ---------------------------------------------------------------
# 逐 Task：build_task → 严格泄漏检查（中止违规）→ fit → 滚动预测 → 指标。
# 产出可审计结果：汇总表 / summary JSON / 逐小时预测 / run_manifest。
# ---------------------------------------------------------------
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from data.gefcom_loader import GEFCOM_DATA_DIR
from data.task_builder import (
    TARGET_COL,
    build_task,
    feature_spec_hash,
)
from evaluation.evaluator import TaskResult, evaluate_task, summarize
from evaluation.forecast_protocol import ForecastProtocol, get_protocol
from evaluation.leakage_checker import check_feature_leakage
from evaluation.rolling_backtest import build_forecast_features, rolling_predict
from models.replay_backends import ModelBackend


class LeakageError(RuntimeError):
    """泄漏检查未通过，中止回放。"""


def _check_task_leakage(task, leak_check: str, protocol: ForecastProtocol) -> None:
    """对历史段（+ 预测窗口，非 recursive 时）做严格泄漏检查，违规即中止。"""
    extra_points = [
        task.train_df.index[-1],
        task.val_df.index[0],
        task.val_df.index[-1],
        task.forecast_ts[0],
    ]

    # 历史段（含全部特征 + 目标）
    ok_hist, v_hist = check_feature_leakage(
        task.history_df, mode=leak_check, extra_check_points=extra_points
    )
    if not ok_hist:
        _raise_leakage("历史段", v_hist)

    if not protocol.recursive:
        # 预测窗口段：必须拼接完整历史上下文，否则月首 lag/rolling 的
        # 历史前缀缺失会导致 Pass B 误报。拼接后 Pass B 在完整序列上重算，
        # 与已构建特征逐位一致（T5 parity）。
        observed_full = pd.concat([task.history_df[TARGET_COL], task.y_true])
        fw_feat = build_forecast_features(observed_full, task.forecast_ts)
        full_check = pd.concat([task.history_df, fw_feat])
        full_check[TARGET_COL] = observed_full.values
        extra_fw = [task.forecast_ts[0], task.forecast_ts[-1]]
        ok_fw, v_fw = check_feature_leakage(
            full_check, mode=leak_check, extra_check_points=extra_fw
        )
        if not ok_fw:
            _raise_leakage("预测窗口段", v_fw)


def _raise_leakage(where: str, violations) -> None:
    msgs = "\n".join(f"  - [{v.kind}] {v.message}" for v in violations[:20])
    raise LeakageError(f"[LEAK] {where} 泄漏检查未通过:\n{msgs}")


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def replay(
    task_ids: Iterable[int],
    backend: ModelBackend,
    protocol: ForecastProtocol,
    val_hours: int = 168,
    leak_check: str = "sample",
    skip_leak_check: bool = False,
    data_dir: Path = GEFCOM_DATA_DIR,
    seed: int = 42,
    outdir: Path = None,
    spec=None,
) -> dict:
    """
    滚动回放主循环。返回 {results, summary, table}。
    outdir 提供时写入审计输出（predictions/*.csv + run_manifest.json + 汇总文件）。
    """
    from data.task_builder import FEATURE_SPEC
    if spec is None:
        spec = FEATURE_SPEC

    results: List[TaskResult] = []
    for tid in sorted(task_ids):
        task = build_task(tid, data_dir, val_hours, spec=spec)
        if not skip_leak_check:
            _check_task_leakage(task, leak_check, protocol)
        backend.fit(task.train_df, task.val_df, task.feature_cols, task.target_col, seed)
        y_pred = rolling_predict(backend, task, protocol, spec=spec)
        results.append(evaluate_task(task, y_pred, backend.name, protocol.name))
        print(
            f"  Task {tid:>2}: RMSE={results[-1].metrics['RMSE']:.4f} "
            f"MAE={results[-1].metrics['MAE']:.4f} N={task.n_forecast}"
        )

    summary_dict = summarize(results)
    payload = {"results": results, "summary": summary_dict["summary"], "table": summary_dict["table"]}

    if outdir is not None:
        _write_outputs(results, summary_dict, backend.name, protocol.name, seed,
                       val_hours, leak_check, sorted(task_ids), outdir, skip_leak_check, spec)
    return payload


def _write_outputs(results, summary_dict, model, protocol, seed, val_hours,
                   leak_check, task_ids, outdir, skip_leak_check, spec) -> None:
    outdir = Path(outdir)
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"{model}_{protocol}_{ts}"

    summary_dict["table"].to_csv(outdir / f"task_replay_{base}.csv", encoding="utf-8-sig")
    with open(outdir / f"summary_{base}.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict["summary"], f, ensure_ascii=False, indent=2, default=str)

    # 逐小时预测：timestamp, y_true, y_pred, error
    for r in results:
        df_pred = pd.DataFrame(
            {
                "timestamp": r.forecast_ts,
                "y_true": r.y_true,
                "y_pred": r.y_pred,
                "error": r.y_true - r.y_pred,
            }
        )
        df_pred.to_csv(
            pred_dir / f"task_{r.task_id:02d}.csv", index=False, encoding="utf-8-sig"
        )

    # run_manifest
    manifest = {
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
        "n_tasks": len(results),
    }
    with open(outdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"  审计输出 -> {outdir}")
