import copy

import pytest

from scripts.select_readout import select


def candidates():
    specifications = [("shared_lora", 1e-4, 32, 3), ("shared_lora", 3e-4, 30, 4),
                      ("replay_lora", 1e-4, 33, 2), ("full_readout", 2e-5, 35, 2),
                      ("expand", 1e-4, 40, 1), ("expand", 3e-4, 43, 4)]
    runs, jobs = [], []
    for index, (method, lr, ap, af) in enumerate(specifications):
        config = {"method": method, "lr": lr, "epochs": 5, "rank": 8, "alpha": 16,
                  "router_text": "raw",
                  "batch_size": 16, "router_modality": "joint", "eval_split": "validation",
                  "task_limit": 0, "protocol": "p", "manifest_sha256": "data", "source_sha256": {"model": "frozen"},
                  "output": f"run-{index}"}
        runs.append({"config": config, "directory": config["output"], "full_protocol": True,
                     "metrics": {"AP": ap, "AF": af}, "plasticity_pp": [20, 10, 8],
                     "expert_count": 3 if method == "expand" else 1, "largest_answer_fraction": 0.25})
        jobs.append({key: config[key] for key in ("method", "lr", "epochs", "eval_split", "output")})
    return runs, jobs


def test_select_keeps_strong_baseline_and_rejects_worse_forgetting():
    runs, jobs = candidates()
    selected = select(runs, jobs)["p"]
    assert selected["strongest_baseline"] == "full_readout"
    assert selected["primary_common_budget"]["config"]["lr"] == 1e-4
    assert selected["secondary_independent"]["metrics"] == {"AP": 40, "AF": 1}
    assert selected["secondary_independent"]["validation_joint_success"]


def test_select_never_uses_official_eval_or_partial_protocol():
    runs, jobs = candidates()
    runs[0]["config"]["eval_split"] = "eval"
    with pytest.raises(ValueError, match="full-protocol validation"):
        select(runs, jobs)
    runs, jobs = candidates()
    runs[0]["full_protocol"] = False
    with pytest.raises(ValueError, match="full-protocol validation"):
        select(runs, jobs)


def test_select_rejects_missing_conditions_and_data_changes():
    runs, jobs = candidates()
    with pytest.raises(ValueError, match="Incomplete"):
        select(runs[:-1], jobs)
    changed = copy.deepcopy(runs)
    changed[0]["config"]["manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="changed data"):
        select(changed, jobs)


def test_select_records_failure_when_low_forgetting_is_no_learning():
    runs, jobs = candidates()
    for row in runs:
        if row["config"]["method"] == "expand":
            row["plasticity_pp"] = [80, 0, 0]
    selected = select(runs, jobs)["p"]
    assert not selected["primary_common_budget"]["validation_joint_success"]
    assert not selected["secondary_independent"]["validation_joint_success"]
    assert selected["secondary_independent"]["post_first_task_plasticity_pp"] == 0
