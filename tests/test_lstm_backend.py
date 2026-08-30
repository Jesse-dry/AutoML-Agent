# LSTMBackend + 模型选择 单测
# ---------------------------------------------------------------
# 覆盖：
#   L1 make_backend("lstm") 返回 LSTMBackend
#   L2 fit/predict 形状对齐（返回长度 == len(X)，Task 1 小窗口小 epoch 冒烟）
#   L3 predict 依赖 lag_1 重建 target（缺失报错）
#   L4 recursive 单行预测未实现（报 NotImplementedError）
#   L5 parse_llm_v2 识别 candidate.model 字段 + 非法 model 拒绝
# ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from agent.evolution_schema import parse_llm_v2
from data.task_builder import FEATURE_SPEC
from evaluation.forecast_protocol import ONLINE_H1
from evaluation.spec_evaluator import evaluate_spec
from models.replay_backends import LSTMBackend, make_backend

_FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _FAILED.append(name)


# ---------------- L1 ----------------
def test_l1_make_backend():
    b = make_backend("lstm")
    check("L1 make_backend('lstm') 类型", isinstance(b, LSTMBackend))
    check("L1 name", b.name == "lstm")


# ---------------- L2 ----------------
def test_l2_fit_predict_shape():
    res = evaluate_spec(
        1, FEATURE_SPEC, ONLINE_H1,
        backend_factory=lambda: LSTMBackend(train_window=500, max_epochs=3),
        seed=42,
    )
    check("L2 fit/predict 形状对齐", len(res["y_pred"]) == len(res["y_true"]))
    check("L2 backend 名", res["backend_name"] == "lstm")
    check("L2 RMSE 合理量纲 (Load ~150MW)", 0.0 < res["rmse"] < 100.0,
          f"RMSE={res['rmse']:.2f}")


# ---------------- L3 ----------------
def test_l3_predict_requires_lag1():
    b = LSTMBackend()
    b._context = pd.DataFrame({"_t": []})  # 占位，未真正 fit
    b._target_col = "LOAD"
    X = pd.DataFrame({"hour": [0.0, 1.0]}, index=pd.date_range("2020-01-01", periods=2, freq="h"))
    try:
        b.predict(X)
        check("L3 缺失 lag_1 报错", False)
    except ValueError as e:
        check("L3 缺失 lag_1 报错", "lag_1" in str(e))


# ---------------- L4 ----------------
def test_l4_recursive_not_implemented():
    b = LSTMBackend()
    b._context = pd.DataFrame({"_t": []})
    b._target_col = "LOAD"
    X = pd.DataFrame({"lag_1": [1.0]}, index=pd.date_range("2020-01-01", periods=1, freq="h"))
    try:
        b.predict(X)
        check("L4 recursive 单行报 NotImplementedError", False)
    except NotImplementedError:
        check("L4 recursive 单行报 NotImplementedError", True)


# ---------------- L5 ----------------
def test_l5_parse_model_field():
    raw = {"round": 1, "analysis": "t", "candidates": [
        {"candidate_id": 1, "hypothesis": "h", "model": "lstm",
         "actions": [{"type": "keep"}]},
        {"candidate_id": 2, "hypothesis": "h", "actions": [{"type": "keep"}]},
    ]}
    p = parse_llm_v2(raw, 3)
    check("L5 model 字段识别", p["candidates"][0]["model"] == "lstm"
          and p["candidates"][1]["model"] is None)

    bad = {"round": 1, "analysis": "t", "candidates": [
        {"candidate_id": 1, "hypothesis": "h", "model": "xgboost",
         "actions": [{"type": "keep"}]},
    ]}
    try:
        parse_llm_v2(bad, 3)
        check("L5 非法 model 拒绝", False)
    except ValueError:
        check("L5 非法 model 拒绝", True)


def main():
    print("=" * 60)
    test_l1_make_backend()
    test_l2_fit_predict_shape()
    test_l3_predict_requires_lag1()
    test_l4_recursive_not_implemented()
    test_l5_parse_model_field()
    print("=" * 60)
    if _FAILED:
        print(f"FAILED: {_FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
