import copy

import pytest

from scripts.summarize_vision import compare, compare_exposure


def confirmation_fixture():
    seeds = [17, 29, 43]
    plan = {"protocols": ["example"], "confirmation": {
        "seeds": seeds, "primary_methods": ["sequential", "sema"],
        "ablation_methods_seed17": ["no_expansion", "task_expansion"]}}
    runs = []
    for cohort in ("confirmation", "confirmation_independent"):
        methods = ["sequential", "sema"] if cohort == "confirmation" else ["sema"]
        for method in methods + ["no_expansion", "task_expansion"]:
            for seed in seeds if method in methods else [17]:
                index = seeds.index(seed) + 1
                ap, af = 10. * index, 4. + index
                if method == "sema":
                    ap += index if cohort == "confirmation" else -index
                    af -= 1.
                runs.append({"cohort": cohort, "protocol": "example", "method": method,
                             "seed": seed, "eval_split": "eval", "lr": 2e-6, "epochs": 5,
                             "metrics": {"AP": ap, "AF": af, "BWT": -af, "MAP": ap + af / 2},
                             "matrix": [[(ap + af) / 100], [ap / 100, ap / 100]],
                             "final_answer_counts": {"yes": 3, "no": 1},
                             "experts": [1, 2, 1], "functional_phases": {"0": ["functional"]}})
    return plan, runs


def test_comparison_pairs_seeds_not_input_order_and_reports_uncertainty():
    plan, runs = confirmation_fixture()
    report = compare(list(reversed(runs)), plan)
    primary, secondary = report["comparisons"]
    assert primary["paired_delta"]["AP"] == {
        "mean": 2., "sample_sd": 1., "by_seed": {"17": 1., "29": 2., "43": 3.}}
    assert primary["paired_delta"]["AF"]["mean"] == -1.
    assert primary["paired_delta"]["task_end_AP"]["mean"] == pytest.approx(1.5)
    assert primary["joint_criterion"] is True
    assert secondary["paired_delta"]["AP"]["mean"] == -2.
    assert secondary["joint_criterion"] is False


@pytest.mark.parametrize("problem", ["missing", "duplicate", "validation"])
def test_comparison_rejects_incomplete_or_contaminated_confirmation(problem):
    plan, runs = confirmation_fixture()
    if problem == "missing":
        runs.pop()
    elif problem == "duplicate":
        runs.append(dict(runs[0]))
    else:
        runs[0] = dict(runs[0], eval_split="validation")
    with pytest.raises(ValueError, match="confirmation grid|official eval"):
        compare(runs, plan)


def exposure_comparison_fixture():
    plan = {"protocols": ["example"], "source_cohorts": ["confirmation"],
            "source_runner_before_extension_sha256": "old-loop",
            "confirmation": {"seeds": [17, 29, 43], "primary_methods": ["sequential", "no_expansion"]}}
    parents, controls = [], []
    for index, seed in enumerate(plan["confirmation"]["seeds"]):
        ap = 30. + index
        parent = {"cohort": "confirmation", "protocol": "example", "method": "sema", "seed": seed,
                  "eval_split": "eval", "manifest_sha256": "same-data", "lr": 1e-5, "epochs": 5,
                  "threshold": .75, "directory": f"parent/{seed}",
                  "source_sha256": {"sema_vqa/vision_run.py": "old-loop", "sema_vqa/vision_model.py": "same-model"},
                  "functional_phases": {"0": ["functional"], "1": ["layer9", "layer10"]},
                  "metrics": {"AP": ap, "AF": 5., "BWT": -5., "MAP": ap + 2.5},
                  "matrix": [[(ap + 5.) / 100], [ap / 100, ap / 100]]}
        parents.append(parent)
        for method, difference in (("sequential", -2.), ("no_expansion", 1.)):
            control = copy.deepcopy(parent)
            control["method"] = method
            control["source_sha256"]["sema_vqa/vision_run.py"] = "exposure-loop"
            control_ap, control_af = ap + difference, 5. - difference
            control["metrics"] = {"AP": control_ap, "AF": control_af, "BWT": -control_af,
                                  "MAP": control_ap + control_af / 2}
            control["matrix"] = [[(control_ap + control_af) / 100], [control_ap / 100, control_ap / 100]]
            control["exposure_match"] = {"source_run": parent["directory"], "phase_counts": [1, 2]}
            control["initial_predictions_match_parent"] = True
            controls.append(control)
    return plan, parents, controls


def test_exposure_comparison_pairs_exact_parent_and_control_method():
    plan, parents, controls = exposure_comparison_fixture()
    comparisons = compare_exposure(list(reversed(controls)), parents, plan)["comparisons"]
    sequential, frozen = comparisons
    assert sequential["paired_delta"]["AP"]["by_seed"] == {"17": 2., "29": 2., "43": 2.}
    assert sequential["paired_delta"]["AF"]["mean"] == -2.
    assert sequential["joint_criterion"] is True
    assert frozen["paired_delta"]["AP"]["mean"] == -1.
    assert frozen["joint_criterion"] is False


@pytest.mark.parametrize("problem", ["unmatched_phases", "unverified_initial"])
def test_exposure_comparison_refuses_invalid_equivalence(problem):
    plan, parents, controls = exposure_comparison_fixture()
    if problem == "unmatched_phases":
        controls[0]["functional_phases"]["1"].pop()
    else:
        controls[0]["initial_predictions_match_parent"] = False
    with pytest.raises(ValueError, match="phase counts|Initial-task"):
        compare_exposure(controls, parents, plan)
