import pytest

from scripts.summarize_vision import select
from scripts.vision_array import calibration_jobs


def calibration_fixture():
    plan = {"protocols": ["example"], "epochs_per_functional_phase": 20, "calibration": {
        "shared_head_lr_candidates": [2e-5, 2e-6], "sema_threshold_candidates": [1.0, 0.75], "selection": "declared"}}
    runs = []
    for job in calibration_jobs(plan):
        if job["lr"] == 2e-6:
            ap, af = 10., 1.
        elif job["method"] == "sequential":
            ap, af = 30., 5.
        elif job["threshold"] == 1.0:
            ap, af = 31., 4.
        else:
            ap, af = 35., 8.
        runs.append(job | {"eval_split": "validation", "metrics": {"AP": ap, "AF": af}})
    return plan, runs


def test_selection_does_not_weaken_baseline_or_ignore_forgetting():
    plan, runs = calibration_fixture()
    result = select(runs, plan)
    assert result["selected"]["example"] == {"lr": 2e-5, "threshold": 1.0, "epochs": 20}
    assert result["validation_evidence"]["example"]["joint_criterion"] is True


def test_official_eval_cannot_select_hyperparameters():
    plan, runs = calibration_fixture()
    runs[0]["eval_split"] = "eval"
    with pytest.raises(ValueError, match="prohibited"):
        select(runs, plan)


def test_selection_matches_training_budget_to_strongest_baseline():
    plan, runs = calibration_fixture()
    plan["calibration"]["epoch_candidates"] = [20, 5]
    short = [dict(row, epochs=5, metrics={"AP": 40., "AF": 3.}) for row in runs]
    for row in short:
        if row["method"] == "sema":
            row["metrics"] = {"AP": 41., "AF": 2.}
    result = select(runs + short, plan)
    assert result["selected"]["example"]["epochs"] == 5
    assert result["validation_evidence"]["example"]["baseline"]["AP"] == 40.


def test_independent_selection_keeps_strong_baseline_but_tunes_sema_budget():
    plan, runs = calibration_fixture()
    plan["calibration"]["epoch_candidates"] = [20, 5]
    short = [dict(row, epochs=5, metrics={"AP": 20., "AF": 3.}) for row in runs]
    for row in short:
        if row["method"] == "sema" and row["lr"] == 2e-6 and row["threshold"] == 1.0:
            row["metrics"] = {"AP": 40., "AF": 2.}
    shared = select(runs + short, plan)
    assert shared["selected"]["example"] == {"lr": 2e-5, "threshold": 1.0, "epochs": 20}
    plan["calibration"]["selection_scope"] = "independent"
    independent = select(runs + short, plan)
    assert independent["selected"]["example"] == {"lr": 2e-6, "threshold": 1.0, "epochs": 5}
    assert independent["validation_evidence"]["example"]["baseline"]["AP"] == 30.
    assert independent["validation_evidence"]["example"]["joint_criterion"] is True
