"""ECL Exp4: No-Memory vs Memory Agent.

This entry point keeps the ECL protocol in :mod:`evaluation.ecl_protocol`
and the model/replay implementation in :mod:`evaluation.ecl_replay`.
The memory agent only reads/writes the public ``MemoryManager`` JSONL
interface; test-user labels are never written to memory.

Example (small smoke run)::

    python experiments/run_ecl_memory.py --model persistence --n-train 30 \
      --outdir experiments/output/ecl_memory_smoke

For a full run use ``--n-train 260`` (the default).  ECL data and model
artifacts are intentionally not generated in the repository by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecl_loader import load_ecl_matrix  # noqa: E402
from data.ecl_task_builder import ECL_FEATURE_SPEC, build_migration_task  # noqa: E402
from evaluation import ecl_protocol as ep  # noqa: E402
from evaluation.ecl_replay import replay_ecl  # noqa: E402
from memory.memory_manager import (  # noqa: E402
    MemoryManager,
    Scenario,
    StrategyRecord,
    season_from_month,
)
from models.replay_backends import make_backend  # noqa: E402


def candidate_specs() -> Dict[str, List[dict]]:
    """Return a small, deterministic candidate set for the memory bootstrap."""
    base = [dict(s) for s in ECL_FEATURE_SPEC]
    # Keep candidates protocol-safe: all features are past-only and <=168h.
    lag48 = base + [{
        "name": "lag_48", "type": "lag", "source": "load", "k": 48,
        "lookback_start": -48, "lookback_end": -48,
        "uses_current_target": False,
    }]
    no_std = [s for s in base if s["name"] != "rolling_std_24"]
    return {"base": base, "lag48": lag48, "no_std24": no_std}


def _scenario_from_series(series: pd.Series) -> Scenario:
    """Compute the memory similarity vector from observed history only."""
    y = pd.Series(series, dtype=float).dropna()
    mean = float(y.mean())
    cv = float(y.std() / mean) if mean else 0.0
    # pandas autocorrelation is deterministic and does not inspect future labels.
    acf24 = float(y.autocorr(lag=24)) if len(y) > 24 else 0.0
    acf168 = float(y.autocorr(lag=168)) if len(y) > 168 else 0.0
    # ECL's forecast origin is July 2014 in the shared protocol.
    return Scenario(
        season=season_from_month(ep.TEST_START.month),
        acf_24=float(np.nan_to_num(acf24)),
        acf_168=float(np.nan_to_num(acf168)),
        load_cv=float(np.nan_to_num(cv)),
        energy="load",
    )


def _validation_rmse(
    spec: List[dict],
    model_name: str,
    *,
    data_dir: Optional[Path],
    n_train: int,
    seed: int,
) -> float:
    """Fit on ECL train users before ``TRAIN_END`` and score June validation.

    This bootstrap score is only used to rank candidate strategies in memory;
    final reported metrics always come from ``replay_ecl`` on the untouched
    test-user/time window.
    """
    task = build_migration_task(data_dir=data_dir, n_train=n_train, seed=seed, spec=spec)
    train = task.train_df[task.train_df.index < ep.TRAIN_END]
    val = task.train_df[(task.train_df.index >= ep.VAL_START) & (task.train_df.index < ep.VAL_END)]
    if train.empty or val.empty:
        raise ValueError("ECL train/validation split is empty; check data and protocol")
    backend = make_backend(model_name)
    backend.fit(train, val, task.feature_cols, task.target_col, seed)
    pred = np.asarray(backend.predict(val[task.feature_cols]), dtype=float)
    y = val[task.target_col].to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if not mask.any():
        return float("inf")
    return float(np.sqrt(np.mean((y[mask] - pred[mask]) ** 2)))


def _bootstrap_memory(
    memory: MemoryManager,
    specs: Dict[str, List[dict]],
    model_name: str,
    *,
    data_dir: Optional[Path],
    n_train: int,
    seed: int,
) -> Tuple[Dict[str, float], Scenario]:
    """Create compact train-validation StrategyRecords (never test labels)."""
    matrix = load_ecl_matrix(data_dir) if data_dir is not None else load_ecl_matrix()
    train_users, _ = __import__("data.ecl_loader", fromlist=["split_users"]).split_users(
        n_users=matrix.shape[1], n_train=n_train, seed=seed
    )
    # Aggregate scenario from train-user histories.  This is the query vector
    # used for retrieval on the held-out test population.
    scenarios = [_scenario_from_series(matrix[u]) for u in train_users]
    query = Scenario(
        season=max((s.season for s in scenarios), key=lambda x: sum(t.season == x for t in scenarios)),
        acf_24=float(np.mean([s.acf_24 for s in scenarios])),
        acf_168=float(np.mean([s.acf_168 for s in scenarios])),
        load_cv=float(np.mean([s.load_cv for s in scenarios])),
        energy="load",
    )
    scores: Dict[str, float] = {}
    for name, spec in specs.items():
        score = _validation_rmse(spec, model_name, data_dir=data_dir, n_train=n_train, seed=seed)
        scores[name] = score
        rec = StrategyRecord(
            task_id=-(len(memory.load_strategies()) + 1),
            energy="load",
            spec=spec,
            rmse=score,
            scenario=query,
            stats={"dataset": "ecl", "scope": "train_validation", "experiment_id": f"ecl_exp4_bootstrap_{name}",
                   "n_train_users": n_train, "train_users_hash": ep.hash_list(train_users)},
            profile={"dataset": "ecl", "model": model_name, "candidate": name,
                     "time_split": {"train_until": str(ep.TRAIN_END),
                                     "val": f"{ep.VAL_START} ~ {ep.VAL_END}"}},
            policy="memory_bootstrap",
            init_max_iter=0,
        )
        memory.record_strategy(rec)
    return scores, query


def _choose_memory_spec(memory: MemoryManager, query: Scenario, specs: Dict[str, List[dict]]) -> Tuple[str, List[dict], List[dict]]:
    # MemoryManager intentionally has a generic energy-level filter.  Apply an
    # explicit dataset guard here so ECL Exp4 cannot consume unrelated LOAD
    # memories from GEFCom/outer-loop runs sharing the same JSONL file.
    recs = [r for r in memory.retrieve_strategies(query, top_k=max(3, len(specs)))
            if (r.stats or {}).get("dataset") == "ecl"
            or (r.profile or {}).get("dataset") == "ecl"]
    if not recs:
        return "base", specs["base"], []
    # Similarity retrieval is followed by validation-score selection.  This
    # makes the policy auditable and avoids depending on JSONL line order.
    rec = min(recs, key=lambda r: float(r.rmse))
    name = str((rec.profile or {}).get("candidate", "memory"))
    return name, rec.spec, [
        {"task_id": r.task_id, "candidate": (r.profile or {}).get("candidate"),
         "rmse": r.rmse, "similarity": None}
        for r in recs
    ]


def _condition_row(name: str, payload: dict) -> dict:
    s = payload["summary"]
    return {
        "condition": name,
        "model": s["model"],
        "n_users": s["n_users"],
        "mean_rmse": s["mean_rmse"],
        "median_rmse": s["median_rmse"],
        "std_rmse": s["std_rmse"],
        "ratio_vs_persistence": s["ratio_vs_persistence"]["mean"],
        "pct_better_than_persistence": s["ratio_vs_persistence"]["pct_better"],
    }


def run_experiment(
    *,
    model: str = "lightgbm",
    n_train: int = 260,
    seed: int = 42,
    data_dir: Optional[Path] = None,
    outdir: Optional[Path] = None,
    memory_file: Optional[Path] = None,
    bootstrap: bool = True,
) -> dict:
    specs = candidate_specs()
    outdir = Path(outdir) if outdir else PROJECT_ROOT / "experiments" / "output" / "ecl_memory"
    outdir.mkdir(parents=True, exist_ok=True)
    mem_path = Path(memory_file) if memory_file else outdir / "memory.jsonl"
    memory = MemoryManager(mem_path)

    scores, query = ({}, _scenario_from_series(pd.Series([1.0])))
    if bootstrap:
        scores, query = _bootstrap_memory(memory, specs, model, data_dir=data_dir,
                                          n_train=n_train, seed=seed)
    else:
        # Query is still derived from train users; no test labels are used.
        matrix = load_ecl_matrix(data_dir) if data_dir is not None else load_ecl_matrix()
        from data.ecl_loader import split_users
        train_users, _ = split_users(n_users=matrix.shape[1], n_train=n_train, seed=seed)
        query = _scenario_from_series(matrix[train_users[0]])

    mem_name, mem_spec, retrieved = _choose_memory_spec(memory, query, specs)
    no_mem = replay_ecl(make_backend(model), n_train=n_train, seed=seed, data_dir=data_dir,
                        spec=specs["base"])
    mem_run = replay_ecl(make_backend(model), n_train=n_train, seed=seed, data_dir=data_dir,
                         spec=mem_spec)

    rows = [_condition_row("No-Memory", no_mem), _condition_row("Memory Agent", mem_run)]
    summary = {"conditions": rows, "selected_candidate": mem_name,
               "bootstrap_validation_rmse": scores, "retrieved_strategies": retrieved,
               "memory_records": len(memory.load_strategies()), "memory_file": str(mem_path)}
    (outdir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(rows).to_csv(outdir / "condition_summary.csv", index=False, encoding="utf-8-sig")
    # Per-user comparison uses the exact same mask/timestamps from replay_ecl.
    a = ep.user_rows(no_mem["per_user_rmse"], no_mem["per_user_persist_rmse"], no_mem["per_user_snaive_rmse"], no_mem["user_n_pred"])
    b = ep.user_rows(mem_run["per_user_rmse"], mem_run["per_user_persist_rmse"], mem_run["per_user_snaive_rmse"], mem_run["user_n_pred"])
    a = a.rename(columns={c: f"no_memory_{c}" for c in a.columns if c != "user"})
    b = b.rename(columns={c: f"memory_{c}" for c in b.columns if c != "user"})
    a.merge(b, on="user", how="outer").to_csv(outdir / "per_user_comparison.csv", index=False, encoding="utf-8-sig")
    manifest = {"experiment": "ECL Exp4", "model": model, "n_train": n_train,
                "seed": seed, "git_commit": ep.git_commit(), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "protocol": no_mem["summary"]["protocol"], "selected_candidate": mem_name,
                "memory_file": str(mem_path), "bootstrap": bootstrap}
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"summary": summary, "no_memory": no_mem, "memory": mem_run}


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ECL Exp4 No-Memory vs Memory Agent")
    p.add_argument("--model", default="lightgbm", help="lightgbm | persistence | seasonal_naive_24 | seasonal_naive_168")
    p.add_argument("--n-train", type=int, default=260)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--memory-file", default=None)
    p.add_argument("--no-bootstrap", action="store_true", help="do not create train-validation memory records")
    args = p.parse_args(argv)
    try:
        result = run_experiment(model=args.model, n_train=args.n_train, seed=args.seed,
                                data_dir=Path(args.data_dir) if args.data_dir else None,
                                outdir=Path(args.outdir) if args.outdir else None,
                                memory_file=Path(args.memory_file) if args.memory_file else None,
                                bootstrap=not args.no_bootstrap)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    for row in result["summary"]["conditions"]:
        print(f"{row['condition']:<14} mean={row['mean_rmse']:.4f} median={row['median_rmse']:.4f} "
              f"ratio_vs_persistence={row['ratio_vs_persistence']:.4f} "
              f"pct_better={row['pct_better_than_persistence']:.1f}%")
    print(f"selected candidate: {result['summary']['selected_candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
