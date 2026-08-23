# 自进化 Agent 主循环（P1-A 状态机核心）
# ---------------------------------------------------------------
# 每轮：
#   构建上下文（场景 + 特征重要性 + 误差画像 + 相关历史经验 + 迭代历史）
#   → LLM 一次调用返回 ≤n_candidates 个候选（不同假设）
#   → 每候选: apply_actions → validate_spec_list → evaluate_spec（预测月 online_h1）
#   → Selector: 最优候选优于 best → 接受；否则 current:=best（自动回滚）
#   → memory.record → 下一轮 / 停止
#
# 核心不变量：
#   best_rmse = min(所有评测过的 spec)；best_spec 恒为达到该值的深拷贝。
#   每轮结束 current_spec ∈ {该轮最优候选 spec，或回滚后的 best_spec}。
# 自动回滚由 Selector 强制（无候选改进即 current:=best），不依赖 LLM 发 rollback；
# 显式 rollback 动作 = 候选 base 换成 best（"重开一条新路"）。
# ---------------------------------------------------------------
from typing import Any, Callable, Dict, List, Optional

from agent.evolution_schema import (
    EvolutionContext,
    build_llm_v2_messages,
    format_feature_importance,
    format_round_history,
    parse_llm_v2,
)
from agent.feature_agent import compute_acf_summary
from agent.feature_spec import apply_actions, snapshot, validate_spec_list
from data.availability import available_history
from data.gefcom_loader import GEFCOM_DATA_DIR
from data.task_builder import FEATURE_SPEC, TARGET_COL, feature_spec_hash
from evaluation.error_profiler import format_profile_for_llm
from evaluation.forecast_protocol import ForecastProtocol, ONLINE_H1
from evaluation.spec_evaluator import evaluate_spec
from memory.memory_manager import (
    ExperienceRecord,
    MemoryManager,
    Scenario,
    format_memories_for_llm,
    season_from_month,
)
from models.replay_backends import LightGBMBackend, ModelBackend, make_backend


class EvolutionRunner:
    """单 Task 自进化特征工程闭环。"""

    def __init__(
        self,
        task_id: int,
        backend_factory: Optional[Callable[[], ModelBackend]] = None,
        protocol: ForecastProtocol = ONLINE_H1,
        spec_evaluator: Optional[Callable] = None,
        llm_client: Optional[Any] = None,
        memory: Optional[MemoryManager] = None,
        n_candidates: int = 3,
        max_iter: int = 5,
        max_retries: int = 3,
        improvement_eps: float = 1e-4,
        patience: int = 2,
        val_hours: int = 168,
        seed: int = 42,
        data_dir=None,
        dataset_name: str = "",
        max_lag: int = 168,
        init_spec: Optional[List[dict]] = None,
        init_spec_label: str = "",
        energy: str = "load",
        zone: Optional[int] = None,
    ):
        self.task_id = task_id
        self.energy = energy
        self.zone = zone
        self.backend_factory = backend_factory or (lambda: LightGBMBackend())
        self.protocol = protocol
        self.llm_client = llm_client
        self.memory = memory
        self.n_candidates = max(1, int(n_candidates))
        self.max_iter = int(max_iter)
        self.max_retries = int(max_retries)
        self.improvement_eps = improvement_eps
        self.patience = int(patience)
        self.val_hours = val_hours
        self.seed = seed
        self.max_lag = int(max_lag)

        # energy/track 资源解析（Load 默认保持现状，Wind 切换并行件）
        if energy == "wind":
            from data.wind_loader import (
                WIND_DATA_DIR,
                WIND_TARGET_COL,
                wind_available_history,
            )
            from data.wind_task_builder import (
                WIND_FEATURE_SPEC,
                WIND_WEATHER_DERIVED_COLS,
            )
            from evaluation.spec_evaluator import evaluate_wind_spec

            self.target_col = WIND_TARGET_COL
            self._base_spec = snapshot(WIND_FEATURE_SPEC)
            self._default_data_dir = WIND_DATA_DIR
            self._availability_fn = wind_available_history
            self._default_spec_evaluator = evaluate_wind_spec
            self.allowed_sources = {WIND_TARGET_COL, *WIND_WEATHER_DERIVED_COLS}
            self._default_dataset_name = f"GEFCom2014-W Task {task_id} Zone {zone}"
        else:
            self.target_col = TARGET_COL
            self._base_spec = snapshot(FEATURE_SPEC)
            self._default_data_dir = GEFCOM_DATA_DIR
            self._availability_fn = available_history
            self._default_spec_evaluator = evaluate_spec
            self.allowed_sources = {TARGET_COL}
            self._default_dataset_name = f"GEFCom2014 Task {task_id}"

        self.data_dir = data_dir or self._default_data_dir
        self.dataset_name = dataset_name or self._default_dataset_name
        self.spec_evaluator = spec_evaluator or self._default_spec_evaluator

        # 状态（init_spec 提供跨 Task 迁移的 warm-start 起点；
        # Round 0 评测 init_spec → baseline_rmse = 继承策略在本 Task 的 RMSE）
        self.init_spec = snapshot(init_spec) if init_spec is not None else None
        base = self.init_spec if self.init_spec is not None else snapshot(self._base_spec)
        self.init_spec_label = (
            init_spec_label
            or ("inherited spec" if self.init_spec is not None else "FEATURE_SPEC")
        )
        self.baseline_spec: List[dict] = base
        self.current_spec: List[dict] = snapshot(base)
        self.best_spec: List[dict] = snapshot(base)
        self.best_rmse: float = float("inf")
        self.baseline_rmse: float = float("inf")
        self.best_round: int = 0
        # 模型选择状态：当前模型与达到 best 的模型（从 backend_factory 推断初始名）
        _probe = self.backend_factory()
        self.current_model: str = getattr(_probe, "name", "lightgbm")
        self.best_model: str = self.current_model
        self._current_result: Optional[Dict] = None   # 当前 best 的评测结果（画像/重要性）
        self._scenario: Optional[Scenario] = None
        self.round_records: List[Dict] = []
        self._eval_cache: Dict[tuple, Dict] = {}

    # ---------------------------------------------------------
    # 场景
    # ---------------------------------------------------------
    def _build_scenario(self) -> Scenario:
        if self.energy == "wind":
            av = self._availability_fn(self.task_id, self.zone, self.data_dir)
        else:
            av = self._availability_fn(self.task_id, self.data_dir)
        target = av.history_df[self.target_col].dropna()
        cv = float(target.std() / target.mean()) if target.mean() != 0 else 0.0
        acf = compute_acf_summary(av.history_df, self.target_col, lags=[24, 168])
        season = season_from_month(av.forecast_ts[0].month)
        return Scenario(
            season=season,
            acf_24=float(acf.get(24, 0.0)),
            acf_168=float(acf.get(168, 0.0)),
            load_cv=cv,
            energy=self.energy,
        )

    # ---------------------------------------------------------
    # 评测（带缓存）
    # ---------------------------------------------------------
    def _evaluate(self, spec: List[dict], model_name: Optional[str] = None) -> Dict:
        model = model_name or self.current_model
        key = (self.task_id, self.zone, feature_spec_hash(spec), model, self.protocol.name)
        if key in self._eval_cache:
            return self._eval_cache[key]
        factory = lambda: make_backend(model)
        if self.energy == "wind":
            res = self.spec_evaluator(
                self.task_id, self.zone, spec, self.protocol,
                val_hours=self.val_hours,
                backend_factory=factory,
                seed=self.seed,
                data_dir=self.data_dir,
            )
        else:
            res = self.spec_evaluator(
                self.task_id, spec, self.protocol,
                val_hours=self.val_hours,
                backend_factory=factory,
                seed=self.seed,
                data_dir=self.data_dir,
            )
        self._eval_cache[key] = res
        return res

    # ---------------------------------------------------------
    # 上下文
    # ---------------------------------------------------------
    def _build_context(self, rnd: int) -> EvolutionContext:
        if self._scenario is None:
            self._scenario = self._build_scenario()
        scenario = self._scenario

        profile_text = ""
        imp_text = ""
        if self._current_result is not None:
            profile_text = format_profile_for_llm(self._current_result["profile"])
            imp_text = format_feature_importance(
                self._current_result.get("feature_importance")
            )

        memories_text = ""
        if self.memory is not None:
            recs = self.memory.retrieve(scenario, top_k=5)
            memories_text = format_memories_for_llm(recs)

        hist_text = format_round_history(self.round_records)

        return EvolutionContext(
            task_id=self.task_id,
            round=rnd,
            max_iterations=self.max_iter,
            n_candidates=self.n_candidates,
            dataset_name=self.dataset_name,
            season=scenario.season,
            acf_24=scenario.acf_24,
            acf_168=scenario.acf_168,
            load_cv=scenario.load_cv,
            current_features=[s["name"] for s in self.current_spec],
            best_rmse=self.best_rmse,
            baseline_rmse=self.baseline_rmse,
            best_round=self.best_round,
            error_profile_text=profile_text,
            feature_importance_text=imp_text,
            memories_text=memories_text,
            round_history_text=hist_text,
            max_lag=self.max_lag,
            current_model=self.current_model,
            target_col=self.target_col,
            energy=self.energy,
            exogenous_sources=sorted(self.allowed_sources - {self.target_col}),
        )

    # ---------------------------------------------------------
    # LLM 调用（含解析重试）
    # ---------------------------------------------------------
    def _call_llm_with_retry(self, ctx: EvolutionContext, verbose: bool) -> Optional[Dict]:
        if self.llm_client is None:
            raise RuntimeError("EvolutionRunner 需要 llm_client（QwenClient 或 ScriptedLLM）")

        messages = build_llm_v2_messages(ctx)
        error_history = []
        for attempt in range(1, self.max_retries + 1):
            if verbose:
                print(f"  [LLM] 第 {attempt}/{self.max_retries} 次调用...")
            try:
                raw = self.llm_client.chat(messages, temperature=0.2)
                return parse_llm_v2(raw, self.n_candidates)
            except ValueError as e:
                error_history.append(str(e))
                if verbose:
                    print(f"  [RETRY] 输出格式错误: {str(e)[:150]}...")
                err_text = "\n\n".join(
                    f"错误 #{i + 1}: {err}" for i, err in enumerate(error_history)
                )
                messages[-1] = {
                    "role": "user",
                    "content": messages[-1]["content"]
                    + f"\n\n【修正】你上一次输出有格式错误，请严格按 JSON 格式重出：\n{err_text}",
                }
        if verbose:
            print(f"  [FAIL] {self.max_retries} 次重试后仍失败，本轮停止")
        return None

    # ---------------------------------------------------------
    # 候选执行
    # ---------------------------------------------------------
    def _execute_candidates(self, parsed: Dict, rnd: int) -> List[Dict]:
        candidates: List[Dict] = []
        for c in parsed["candidates"]:
            acts = c["actions"]
            model = c.get("model") or self.current_model
            base = {
                "candidate_id": c["candidate_id"],
                "hypothesis": c["hypothesis"],
                "actions": acts,
                "model": model,
            }

            if any(a.get("type") == "stop" for a in acts):
                candidates.append({**base, "state": "stop"})
                continue

            has_rollback = any(a.get("type") == "rollback" for a in acts)
            base_spec = snapshot(self.best_spec if has_rollback else self.current_spec)

            try:
                filtered_acts = [a for a in acts if a.get("type") != "rollback"]
                cspec = apply_actions(base_spec, filtered_acts,
                                      target_col=self.target_col,
                                      allowed_sources=self.allowed_sources)
            except ValueError as e:
                candidates.append({**base, "state": "invalid", "error": str(e)})
                continue

            viols = validate_spec_list(cspec, target_col=self.target_col,
                                       max_lag=self.max_lag)
            if viols:
                candidates.append({
                    **base, "state": "invalid",
                    "error": "; ".join(v.message for v in viols[:5]),
                })
                continue

            res = self._evaluate(cspec, model)
            candidates.append({
                **base, "state": "evaluated", "spec": cspec, "res": res,
                "base_spec": base_spec, "rollback_base": has_rollback,
            })
        return candidates

    # ---------------------------------------------------------
    # Selector：择优 / 自动回滚
    # ---------------------------------------------------------
    def _select(self, rnd: int, candidates: List[Dict], verbose: bool) -> str:
        evaled = [c for c in candidates if c["state"] == "evaluated"]
        if not evaled:
            self.current_spec = snapshot(self.best_spec)
            note = "; ".join(
                c.get("error", c["state"]) for c in candidates[:3]
            ) or "全部候选无效"
            self.round_records.append({
                "round": rnd, "outcome": "no_candidate",
                "best_rmse": self.best_rmse, "delta_rmse": 0.0,
                "candidate_note": f"[no_candidate] {note}",
            })
            return "no_candidate"

        best_c = min(evaled, key=lambda c: c["res"]["rmse"])
        cand_rmse = best_c["res"]["rmse"]
        before = self.best_rmse
        delta = cand_rmse - before

        if cand_rmse < self.best_rmse - self.improvement_eps:
            # 接受：更新 best 与 current（spec + model 同步）
            self.best_spec = snapshot(best_c["spec"])
            self.best_rmse = cand_rmse
            self.current_spec = snapshot(self.best_spec)
            self.best_model = best_c.get("model", self.current_model)
            self.current_model = self.best_model
            self.best_round = rnd
            self._current_result = best_c["res"]
            outcome = "improved"
            note = f"[improved:{self.best_model}] {', '.join(a.get('type','?') for a in best_c['actions'])}"
        else:
            # 自动回滚：current:=best（spec + model 同步，失败动作保留进 memory 作反例）
            self.current_spec = snapshot(self.best_spec)
            self.current_model = self.best_model
            outcome = "rolled_back"
            note = f"[rolled_back] 最优候选 ΔRMSE {delta:+.4f}，回到 best={self.best_rmse:.4f} ({self.best_model})"

        if verbose:
            rmse_list = ", ".join(f"C{c['candidate_id']}={c['res']['rmse']:.4f}" for c in evaled)
            print(f"  → {outcome}: 候选 [{rmse_list}], best={self.best_rmse:.4f} (round {self.best_round})")

        self.round_records.append({
            "round": rnd, "outcome": outcome,
            "best_rmse": self.best_rmse, "delta_rmse": delta,
            "candidate_note": note,
            "best_candidate": best_c,
        })

        # 写 memory（经验入库：成功/失败都记录，供后续 Task 检索）
        if self.memory is not None:
            self._record_memory(rnd, outcome, best_c, before, cand_rmse)

        return outcome

    def _record_memory(self, rnd: int, outcome: str, best_c: Dict,
                       before: float, after: float) -> None:
        res = best_c["res"]
        profile = res["profile"]
        worst = profile.worst_segment()
        rec = ExperienceRecord(
            task_id=self.task_id,
            round=rnd,
            energy=self.energy,
            scenario=self._scenario or self._build_scenario(),
            problem={"worst_segment": worst, "bias": profile.bias},
            actions=best_c["actions"],
            model=best_c.get("model", self.current_model),
            spec_before=best_c.get("base_spec", self.current_spec),
            spec_after=best_c.get("spec", self.current_spec),
            before_rmse=before,
            after_rmse=after,
            delta_rmse=after - before,
            outcome=outcome,
            accepted=(outcome == "improved"),
        )
        try:
            self.memory.record(rec)
        except Exception as e:
            print(f"  [WARN] memory 写入失败: {e}")

    # ---------------------------------------------------------
    # 主循环
    # ---------------------------------------------------------
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        # Round 0：baseline
        if verbose:
            print("=" * 60)
            print(f"EvolutionRunner — Task {self.task_id} 自进化特征工程")
            print(f"  协议: {self.protocol.name}  候选/轮: {self.n_candidates}  "
                  f"最大轮: {self.max_iter}  模型: {self.current_model}")
            print("=" * 60)
            print("Round 0: baseline 评测...")

        base_res = self._evaluate(self.baseline_spec)
        self.baseline_rmse = base_res["rmse"]
        self.best_rmse = base_res["rmse"]
        self.best_spec = snapshot(self.baseline_spec)
        self.current_spec = snapshot(self.baseline_spec)
        self._current_result = base_res
        if verbose:
            print(f"  Baseline RMSE = {self.baseline_rmse:.4f}  (n={base_res['profile'].n})")

        # 迭代
        no_improve_streak = 0
        for rnd in range(1, self.max_iter + 1):
            if verbose:
                print(f"\n{'─' * 40}")
                print(f"Round {rnd}/{self.max_iter}")
                print(f"{'─' * 40}")

            ctx = self._build_context(rnd)
            parsed = self._call_llm_with_retry(ctx, verbose)
            if parsed is None:
                self.round_records.append({
                    "round": rnd, "outcome": "stopped",
                    "best_rmse": self.best_rmse, "delta_rmse": 0.0,
                    "candidate_note": "[stopped] LLM 解析失败",
                })
                break

            candidates = self._execute_candidates(parsed, rnd)

            stop_count = sum(1 for c in candidates if c["state"] == "stop")
            if stop_count == len(candidates) and len(candidates) > 0:
                if verbose:
                    print("  [STOP] LLM 提议全部候选停止")
                self.round_records.append({
                    "round": rnd, "outcome": "stopped",
                    "best_rmse": self.best_rmse, "delta_rmse": 0.0,
                    "candidate_note": "[stopped] 全部候选 stop",
                })
                break

            outcome = self._select(rnd, candidates, verbose)

            # 停止判定：连续无改善
            if outcome in ("rolled_back", "no_candidate"):
                no_improve_streak += 1
            elif outcome == "improved":
                no_improve_streak = 0
            if no_improve_streak >= self.patience:
                if verbose:
                    print(f"  [STOP] 连续 {self.patience} 轮无改善")
                break

        return self._build_result(verbose)

    def _build_result(self, verbose: bool) -> Dict[str, Any]:
        summary_rows = [{
            "round": 0, "outcome": "baseline", "best_rmse": self.baseline_rmse,
            "delta_rmse": 0.0, "n_features": len(self.baseline_spec),
            "note": f"baseline {self.init_spec_label}",
        }]
        for r in self.round_records:
            summary_rows.append({
                "round": r["round"], "outcome": r["outcome"],
                "best_rmse": r["best_rmse"], "delta_rmse": r["delta_rmse"],
                "n_features": len(self.best_spec),
                "note": r.get("candidate_note", ""),
            })

        if verbose:
            print(f"\n{'=' * 60}")
            print("自进化完成!")
            print(f"  Baseline RMSE = {self.baseline_rmse:.4f} → Best RMSE = {self.best_rmse:.4f} "
                  f"({self.best_rmse - self.baseline_rmse:+.4f}, round {self.best_round})")
            print(f"  评测过的 spec 数: {len(self._eval_cache)}")
            print("  迭代历史:")
            for row in summary_rows:
                print(f"    R{row['round']:<2d} [{row['outcome']:<11s}] "
                      f"best={row['best_rmse']:.4f} Δ={row['delta_rmse']:+.4f} {row['note'][:80]}")

        return {
            "task_id": self.task_id,
            "baseline_rmse": self.baseline_rmse,
            "best_rmse": self.best_rmse,
            "best_round": self.best_round,
            "best_model": self.best_model,
            "baseline_spec": self.baseline_spec,
            "best_spec": self.best_spec,
            "current_spec": self.current_spec,
            "n_evaluations": len(self._eval_cache),
            "summary": summary_rows,
            "round_records": self.round_records,
        }
