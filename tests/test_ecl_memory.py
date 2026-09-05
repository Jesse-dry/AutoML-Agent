"""Unit checks for the ECL Exp4 adapter (no ECL data required)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from experiments.run_ecl_memory import _choose_memory_spec, candidate_specs
from memory.memory_manager import MemoryManager, Scenario, StrategyRecord


def test_candidate_specs_are_past_only_and_bounded():
    specs = candidate_specs()
    assert set(specs) == {"base", "lag48", "no_std24"}
    for spec in specs.values():
        for item in spec:
            assert item.get("uses_current_target") is False
            if item["type"] == "lag":
                assert 1 <= item["k"] <= 168


def test_memory_selection_prefers_low_validation_rmse(tmp_path: Path):
    specs = candidate_specs()
    mm = MemoryManager(tmp_path / "memory.jsonl")
    scenario = Scenario("summer", 0.8, 0.7, 0.3, energy="load")
    mm.record_strategy(StrategyRecord(
        task_id=-1, energy="load", spec=specs["base"], rmse=10.0,
        scenario=scenario, profile={"dataset": "ecl", "candidate": "base"},
    ))
    mm.record_strategy(StrategyRecord(
        task_id=-2, energy="load", spec=specs["lag48"], rmse=5.0,
        scenario=scenario, profile={"dataset": "ecl", "candidate": "lag48"},
    ))
    mm.record_strategy(StrategyRecord(
        task_id=99, energy="load", spec=specs["no_std24"], rmse=0.1,
        scenario=scenario, profile={"dataset": "gefcom", "candidate": "foreign"},
    ))
    name, spec, retrieved = _choose_memory_spec(mm, scenario, specs)
    assert name == "lag48"
    assert [s["name"] for s in spec][-1] == "lag_48"
    assert len(retrieved) == 2


if __name__ == "__main__":
    import tempfile

    test_candidate_specs_are_past_only_and_bounded()
    with tempfile.TemporaryDirectory() as d:
        test_memory_selection_prefers_low_validation_rmse(Path(d))
    print("ALL ECL MEMORY TESTS PASSED")
