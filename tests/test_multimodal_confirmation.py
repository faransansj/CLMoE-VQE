"""Behavior tests for the G4 confirmation audit.

Only plausible statistical, selection and boundary bugs: the exact seed-unit
sign-flip convention, Holm family separation for tuned replay, grid
completeness (no missing/duplicate/extra trials), tuned replay reuse when the
selected condition is the common one, frozen run-config contract boundaries and
generation-guard thresholds. No wiring/source-text assertions: the full audit
path needs real manifest data and the parent verifies it after integration.
"""
import collections
import hashlib
import json
from pathlib import Path
import pytest

from scripts.evaluate_multimodal_confirmation import (
    BOOTSTRAP_SEED, CONFIRMATION_SEEDS, G1_REPORT_ARTIFACT, G2_REPORT_ARTIFACT,
    G3_REPORT_ARTIFACT, NORMALIZERS_ARTIFACT, ORDERS, REQUIRED_PRIOR_ARTIFACTS,
    _family_af, _family_endpoints, _guard_failures, _load_prior_reports,
    _number_matches, _parse_lock, _per_seed_order_means, _run_name,
    _validate_gate_chain, _validate_run_config, expected_conditions, holm,
    paired_statistics, parse_job, primary_statistics, transfer_records, validate_locked_jobs,
    validate_first_task_parity, validate_stage_budgets)
from sema_multimodal.metrics import continual_metrics

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64
HEX_F = "f" * 64


def test_first_task_parity_checks_every_order_against_independent_control(tmp_path):
    trials = {}
    for name, kind, method, order in (
        ("control_v", "independent", "independent", ["V"]),
        ("control_c", "independent", "independent", ["C"]),
        ("first_order", "common_cl", "shared_lora", ["V", "C", "G", "F"]),
        ("second_order", "common_cl", "shared_lora", ["V", "G", "C", "F"]),
        ("capacity", "common_cl", "capacity_matched_shared", ["V", "C", "G", "F"]),
        ("tuned", "tuned_replay", "replay_lora", ["V", "C", "G", "F"]),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / f"predictions_after_00_{order[0]}.jsonl").write_text(
            '{"prediction":"same"}\n' if name not in ("capacity", "tuned")
            else '{"prediction":"different capacity or training budget"}\n')
        trials[name] = {"name": name, "kind": kind, "method": method,
                        "task_order": order, "seed": 137, "directory": str(directory)}
    report, matches = validate_first_task_parity(trials, {})
    assert matches == {"control_v": True, "first_order": True, "second_order": True}
    assert report["V/seed137"]["control"] == "control_v"
    (tmp_path / "first_order" / "predictions_after_00_V.jsonl").write_text(
        '{"prediction":"diverged in the otherwise overwritten order"}\n')
    with pytest.raises(ValueError, match="diverged.*first_order"):
        validate_first_task_parity(trials, {})
    del trials["control_v"]
    with pytest.raises(ValueError, match="lacks independent control"):
        validate_first_task_parity(trials, {})


def synthetic_lock(selected_replay=(0.0003, 10)):
    """Tests build the internal lock shape produced by `_parse_lock` directly."""
    common = {"lr": 0.0003, "epochs": 5, "batch_size": 16, "micro_batch_size": 8}
    jobs = []
    for order in ORDERS:
        tag = "_".join(order)
        for seed in CONFIRMATION_SEEDS:
            for method in ("shared_lora", "capacity_matched_shared", "replay_lora", "expand"):
                jobs.append({"name": _run_name(order, method, 0.0003, 5, seed), "argv": []})
            jobs.append({"name": _run_name(order, "frozen_first", 0.0003, 5, seed), "argv": []})
            if selected_replay != (0.0003, 5):
                jobs.append({"name": _run_name(order, "replay_lora", selected_replay[0],
                                               selected_replay[1], seed), "argv": []})
    for task in ("V", "C", "G", "F"):
        for seed in CONFIRMATION_SEEDS:
            jobs.append({"name": _run_name([task], "independent", 0.0003, 5, seed), "argv": []})
    lock = {"schema_version": 1, "stage": "G4",
            "manifest_sha256": HEX_A,
            "source_sha256": {"sema": HEX_B},
            "assets_sha256": {"tokenizer/universal": HEX_C},
            "tokenizer_sha256": {"tokenizer/universal": HEX_C},
            "scorer_locks": {task: {"metric": metric, "implementation_sha256": HEX_D}
                             for task, metric in zip(("V", "C", "G", "F"), ("vqav2", "cider_d", "gqa", "cider_d"))},
            "normalizers_sha256": HEX_E,
            "normalizers": {"schema_version": 1, "eval_split": "validation",
                            "manifest_sha256": HEX_A,
                            "scores": {"V": 30.0, "C": 0.5, "G": 25.0, "F": 0.4}},
            "scientific_config": {"optimizer": {"name": "AdamW"},
                                  "generation": {"strategy": "cached_greedy"},
                                  "loss": "BLIP", "torch": "2.0", "numpy": "1.24",
                                  "cuda": "12.0", "deterministic_algorithms": True,
                                  "tf32": False},
            "seeds": list(CONFIRMATION_SEEDS), "orders": [list(order) for order in ORDERS],
            "common_training": common,
            "selected_replay": {"lr": selected_replay[0], "epochs": selected_replay[1]},
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": 20000,
            "jobs": jobs,
            "prior_gate_reports": {G1_REPORT_ARTIFACT: HEX_B, G2_REPORT_ARTIFACT: HEX_C,
                                   G3_REPORT_ARTIFACT: HEX_D, NORMALIZERS_ARTIFACT: HEX_E}}
    lock["_seeds"] = tuple(CONFIRMATION_SEEDS)
    lock["_orders"] = ORDERS
    return lock


def job_argv(order, method, lr, epochs, seed):
    return ["--manifest", "/scratch/manifest.json", "--assets", "/scratch/assets",
            "--image-root", "/scratch/images", "--method", method, "--task-ids", *order,
            "--seed", str(seed), "--lr", str(lr), "--epochs", str(epochs),
            "--batch-size", "16", "--micro-batch-size", "8", "--eval-split", "eval",
            "--output", f"/runs/g4/{_run_name(order, method, lr, epochs, seed)}"]


def family_trial(name, *, method, order, seed, kind, vqa, caption, lr=0.0003, epochs=5):
    return {"directory": f"/runs/g4/{name}", "name": name, "kind": kind, "method": method,
            "task_order": list(order), "seed": seed, "lr": lr, "epochs": epochs,
            "family_final": {"vqa": vqa, "caption": caption},
            "family_native_AF": {"vqa": None, "caption": None},
            "family_normalized_AF": {"vqa": None, "caption": None},
            "family_balanced_normalized_final": None,
            "subsequent_task_gains": [], "generation_guard_failures": [],
            "audit": {"stage_resources": []}, "audited_predictions": 0}


def synthetic_family_trials(expand=10.0, baseline=5.0, expand_caption=3.0, baseline_caption=1.0):
    lock = synthetic_lock()
    trials = {}
    for order in ORDERS:
        tag = "_".join(order)
        for seed in CONFIRMATION_SEEDS:
            for method in ("shared_lora", "capacity_matched_shared", "replay_lora"):
                trials[_run_name(order, method, 0.0003, 5, seed)] = family_trial(
                    _run_name(order, method, 0.0003, 5, seed), method=method, order=order,
                    seed=seed, kind="common_cl", vqa=baseline, caption=baseline_caption)
            trials[_run_name(order, "expand", 0.0003, 5, seed)] = family_trial(
                _run_name(order, "expand", 0.0003, 5, seed), method="expand", order=order,
                seed=seed, kind="common_cl", vqa=expand, caption=expand_caption)
            trials[_run_name(order, "replay_lora", 0.0003, 10, seed)] = family_trial(
                _run_name(order, "replay_lora", 0.0003, 10, seed), method="replay_lora",
                order=order, seed=seed, kind="tuned_replay", epochs=10,
                vqa=baseline, caption=baseline_caption)
    return trials, lock


def optimal_run_config(order, seed, *, method="shared_lora", lr=0.0003, epochs=5):
    manifest = "/scratch/data/manifest_v2.json"
    config = {"method": method, "task_ids": list(order), "task_order": list(order),
              "eval_split": "eval", "seed": seed, "lr": lr, "epochs": epochs,
              "rank": 8, "alpha": 16, "batch_size": 16, "micro_batch_size": 8,
              "image_size": 224, "device": "cuda", "router_modality": "joint",
              "router_memory": 128, "router_ridge": 1.0, "replay_per_task": 64,
              "oracle_diagnostic": False, "image_shuffled_diagnostic": False,
              "smoke_output": False, "train_limit": 0, "eval_limit": 0,
              "schema_version": 1, "protocol": "multimodal_cl",
              "manifest": manifest, "assets": "/scratch/assets",
              "image_root": "/scratch/images", "manifest_sha256": HEX_A,
              "source_sha256": {"sema": HEX_B},
              "assets_sha256": {"tokenizer/universal": HEX_C},
              "tokenizer_sha256": {"tokenizer/universal": HEX_C},
              "scorer_locks": {task: {"metric": {"V": "vqav2", "C": "cider_d", "G": "gqa", "F": "cider_d"}[task],
                                      "implementation_sha256": HEX_D} for task in order},
              "optimizer": {"name": "AdamW"}, "loss": "BLIP", "generation": {"strategy": "cached_greedy"},
              "torch": "2.0", "numpy": "1.24", "cuda": "12.0",
              "deterministic_algorithms": True, "tf32": False}
    return config


def test_paired_statistics_resolves_predeclared_eight_seed_prefix():
    stats = paired_statistics([1.0] * 8, bootstrap_seed=BOOTSTRAP_SEED)
    assert stats["exact_two_sided_sign_flip_p"] == 2 / 256
    assert stats["seed_bootstrap_95_percentile"] == [1.0, 1.0]
    assert stats["sample_sd"] == 0
    assert stats["positive_seed_count"] == 8
    assert stats["negative_seed_count"] == 0
    assert stats["bootstrap"] == {"seed": BOOTSTRAP_SEED, "replicates": 20000}


def test_paired_statistics_is_reproducible_for_locked_bootstrap_seed():
    deltas = [3.0, -1.0, 2.0, -2.0, 4.0, -1.0, 3.0, 4.0]
    first = paired_statistics(deltas, bootstrap_seed=BOOTSTRAP_SEED)
    second = paired_statistics(deltas, bootstrap_seed=BOOTSTRAP_SEED)
    assert first["seed_bootstrap_95_percentile"] == second["seed_bootstrap_95_percentile"]
    assert first["exact_two_sided_sign_flip_p"] == second["exact_two_sided_sign_flip_p"]
    low, high = first["seed_bootstrap_95_percentile"]
    assert -2.0 <= low <= 4.0 and -2.0 <= high <= 4.0


def test_holm_correction_stays_within_one_family():
    assert holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert holm([0.7, 0.8]) == [1.0, 1.0]


def test_expected_conditions_locks_224_grid_when_tuned_differs_from_common():
    lock = synthetic_lock()
    grid, tuned_is_common = expected_conditions(lock)
    assert not tuned_is_common
    assert len(grid) == 224
    kinds = collections.Counter(condition["kind"] for condition in grid.values())
    assert kinds == {"common_cl": 128, "frozen_first": 32, "independent": 32, "tuned_replay": 32}
    tuned = [name for name in grid if "replay_lora_lr0.0003_ep10_" in name]
    assert len(tuned) == 32


def test_expected_conditions_reuses_common_runs_when_tuned_is_common():
    lock = synthetic_lock(selected_replay=(0.0003, 5))
    grid, tuned_is_common = expected_conditions(lock)
    assert tuned_is_common
    assert len(grid) == 192
    assert not any(condition["kind"] == "tuned_replay" for condition in grid.values())


def test_expected_conditions_rejects_missing_locked_job_name():
    lock = synthetic_lock()
    lock["jobs"] = lock["jobs"][:-1]
    with pytest.raises(ValueError, match="Locked jobs differ"):
        expected_conditions(lock)




def test_validate_locked_jobs_rejects_changed_budget_or_split():
    lock = synthetic_lock()
    grid, _ = expected_conditions(lock)
    name = "V_C_G_F_shared_lora_lr0.0003_ep5_seed137"
    argv = job_argv(("V", "C", "G", "F"), "shared_lora", 0.0003, 5, 137)
    swapped = argv[:argv.index("--batch-size") + 1] + ["32"] + argv[argv.index("--batch-size") + 2:]
    with pytest.raises(ValueError, match="predeclared condition"):
        validate_locked_jobs({"jobs": [{"name": name, "argv": swapped}]}, grid)
    with pytest.raises(ValueError, match="frozen scientific contract"):
        validate_locked_jobs({"jobs": [{"name": name, "argv": argv + ["--rank", "32"]}]}, grid)


def test_parse_job_rejects_smoke_eval_prefix_and_frozen_flags():
    argv = job_argv(("V", "C", "G", "F"), "shared_lora", 0.0003, 5, 137)
    for flag in ("--rank", "--oracle-diagnostic", "--smoke-output"):
        with pytest.raises(ValueError, match="frozen scientific contract"):
            parse_job(argv + [flag])
    with pytest.raises(ValueError, match="four-task"):
        parse_job(job_argv(("V", "C"), "shared_lora", 0.0003, 5, 137))


def _config_lock():
    return synthetic_lock()


def test_run_config_boundary_changes_fail_loudly():
    lock = _config_lock()
    order = ("V", "C", "G", "F")
    grid, _ = expected_conditions(lock)
    condition = grid["V_C_G_F_shared_lora_lr0.0003_ep5_seed137"]
    config = optimal_run_config(order, 137)
    _validate_run_config(config, condition, lock)
    altered_scorer = dict(config, scorer_locks={})
    with pytest.raises(ValueError, match="scorer locks"):
        _validate_run_config(altered_scorer, condition, lock)
    altered_eval = dict(config, eval_limit=1)
    with pytest.raises(ValueError, match="full official eval"):
        _validate_run_config(altered_eval, condition, lock)
    altered_rank = dict(config, rank=16)
    with pytest.raises(ValueError, match="frozen contract"):
        _validate_run_config(altered_rank, condition, lock)
    altered_ids = dict(config, task_ids=list(order)[::-1])
    with pytest.raises(ValueError, match="task ids/order"):
        _validate_run_config(altered_ids, condition, lock)
    altered_torch = dict(config, torch="9.9")
    with pytest.raises(ValueError, match="runtime contract"):
        _validate_run_config(altered_torch, condition, lock)
    altered_loss = dict(config, loss="cross-entropy")
    with pytest.raises(ValueError, match="runtime contract"):
        _validate_run_config(altered_loss, condition, lock)


def test_guard_failures_match_predeclared_thresholds():
    boundary = _guard_failures({"generation": [
        {"stage": 0, "task_id": "V", "family": "vqa",
         "empty_fraction": 0.01, "largest_output_fraction": 0.899999},
        {"stage": 0, "task_id": "V", "family": "vqa",
         "empty_fraction": 0.010001, "largest_output_fraction": 0.0},
        {"stage": 2, "task_id": "C", "family": "caption",
         "empty_fraction": 0.0, "largest_output_fraction": 0.9}]})
    assert len(boundary) == 2
    assert [record["task_id"] for record in boundary] == ["V", "C"]


def test_transfer_records_require_every_independent_control():
    lock = synthetic_lock()
    trials = {}
    for task in ("V", "C", "G", "F"):
        for seed in CONFIRMATION_SEEDS:
            name = _run_name([task], "independent", 0.0003, 5, seed)
            trials[name] = {"directory": f"/runs/{name}", "name": name, "kind": "independent",
                            "method": "independent", "task_order": [task], "seed": seed,
                            "lr": 0.0003, "epochs": 5,
                            "family_final": {"vqa": None, "caption": None},
                            "family_native_AF": {"vqa": None, "caption": None},
                            "family_normalized_AF": {"vqa": None, "caption": None},
                            "family_balanced_normalized_final": None,
                            "subsequent_task_gains": [], "generation_guard_failures": [],
                            "audited_predictions": 0, "independent_score": 5.0,
                            "audit": {"matrix": [[5.0]],
                                      "metrics": {"tasks": {task: {"normalizer": 1.0}}}}}
    cl = family_trial("V_C_G_F_shared_lora_lr0.0003_ep5_seed137", method="shared_lora",
                      order=("V", "C", "G", "F"), seed=137, kind="common_cl",
                      vqa=1.0, caption=1.0)
    cl["audit"] = {"matrix": [[1.0], [2.0, 4.0], [3.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
                   "metrics": {"tasks": {task: {"normalizer": 1.0}
                                         for task in ("V", "C", "G", "F")}}}
    trials[cl["name"]] = cl
    records = transfer_records(trials)
    assert len(records) == 1 and records[0]["rows"][0]["independent_score"] == 5.0
    del trials[_run_name(["G"], "independent", 0.0003, 5, 137)]
    with pytest.raises(ValueError, match="controls incomplete"):
        transfer_records(trials)


def test_primary_statistics_use_separate_holm_families_and_per_seed_units():
    trials, lock = synthetic_family_trials()
    primary, tuned = primary_statistics(trials, lock)
    assert len(primary) == 6 and len(tuned) == 2
    assert all(record["family"] in ("vqa", "caption") for record in primary)
    for family in ("vqa", "caption"):
        rows = [record for record in primary if record["family"] == family]
        assert len(rows) == 3
        for record in rows:
            assert record["statistics"]["mean"] == pytest.approx(5.0 if family == "vqa" else 2.0)
    for record in primary:
        assert record["statistics"]["exact_two_sided_sign_flip_p"] == 2 / 256
        assert record["holm_adjusted_p"] == pytest.approx(12 / 256)
    for record in tuned:
        assert record["statistics"]["exact_two_sided_sign_flip_p"] == 2 / 256
        assert record["holm_adjusted_p"] == pytest.approx(4 / 256)


def test_primary_statistics_contract_positions_deltas_four_orders_per_seed():
    trials, lock = synthetic_family_trials(expand=12.0, baseline=4.0)
    for index, order in enumerate(ORDERS):
        trials[_run_name(order, "expand", 0.0003, 5, 137)]["family_final"]["vqa"] = 12.0 + 4 * index
    primary, _ = primary_statistics(trials, lock)
    shared_row = next(record for record in primary
                      if record["baseline"] == "shared_lora" and record["family"] == "vqa")
    assert shared_row["statistics"]["per_seed"] == [14.0] + [8.0] * 7
    assert shared_row["candidate_seed_means"][137] == pytest.approx(18.0)
    assert shared_row["baseline_seed_means"][137] == pytest.approx(4.0)


def test_number_matches_tolerates_int_float_only():
    assert _number_matches(16, 16.0)
    assert _number_matches(1.0, 1)
    assert not _number_matches("16", 16)
    assert not _number_matches(True, 1)


def _stage_row(index):
    return {"task_id": "C", "steps": 640, "image_exposures": 10240,
            "current_exposures": 5120, "replay_exposures": 5120, "target_tokens": 4096,
            "training_seconds": 1.0, "router_memory_bytes": 0,
            "replay_images_stored": (index + 1) * 64, "replay_row_bytes": 1,
            "replay_image_bytes": 1, "raw_replay_bytes": 2, "peak_gpu_memory_bytes": 8,
            "parameter_counts": {"base": 1, "experts": 2, "active_trainable": 1,
                                 "total": 3, "trainable": 2}}


def _stage_row_replay_condition():
    return {"name": "V_C_G_F_replay_lora_lr0.0003_ep5_seed137",
            "kind": "common_cl", "method": "replay_lora",
            "order": ("V", "C", "G", "F"), "lr": 0.0003, "epochs": 5, "seed": 137}


def test_stage_budgets_use_end_stage_replay_storage_semantics():
    audited = {"stage_resources": [_stage_row(index) for index in range(4)]}
    audited["stage_resources"][3]["replay_images_stored"] = 256
    validate_stage_budgets(audited, _stage_row_replay_condition(), 2048)
    stale_semantics = dict(audited)
    stale_rows = []
    for index, row in enumerate(audited["stage_resources"]):
        fresh = dict(row, replay_images_stored=index * 64)
        stale_rows.append(fresh)
    stale_semantics["stage_resources"] = stale_rows
    with pytest.raises(ValueError, match="end-stage replay storage"):
        validate_stage_budgets(stale_semantics, _stage_row_replay_condition(), 2048)


def test_shared_method_must_not_consume_replay_rows():
    rows = [dict(_stage_row(index), replay_images_stored=0, replay_row_bytes=0,
                 replay_image_bytes=0, raw_replay_bytes=0)
            for index in range(4)]
    shared = dict(_stage_row_replay_condition(), method="shared_lora",
                  name="V_C_G_F_shared_lora_lr0.0003_ep5_seed137")
    with pytest.raises(ValueError, match="must not consume replay"):
        validate_stage_budgets({"stage_resources": rows}, shared, 2048)
    rows[1] = dict(rows[1], replay_exposures=0, current_exposures=10240,
                   replay_images_stored=32)
    with pytest.raises(ValueError, match="must not consume replay"):
        validate_stage_budgets({"stage_resources": rows}, shared, 2048)


def _prior_artifact_payloads(lock):
    return {
        G1_REPORT_ARTIFACT: {"stage": "G1", "input_valid": True, "gate_passed": True,
                             "pilot_allowed": True, "manifest_sha256": HEX_A,
                             "normalizers_sha256": lock["normalizers_sha256"]},
        G2_REPORT_ARTIFACT: {"stage": "G2", "input_valid": True, "gate_passed": True,
                             "g3_allowed": True, "manifest_sha256": HEX_A,
                             "normalizers_sha256": lock["normalizers_sha256"]},
        G3_REPORT_ARTIFACT: {"stage": "G3", "input_valid": True, "gate_passed": True,
                             "g4_allowed": True, "manifest_sha256": HEX_A,
                             "normalizers_sha256": lock["normalizers_sha256"],
                             "selected_replay": dict(lock["selected_replay"])},
        NORMALIZERS_ARTIFACT: lock["normalizers"],
    }


def _prior_artifacts(tmp_path, lock, payload_overrides=None):
    artifacts = {}
    artifacts_dir = Path(tmp_path) / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    payloads = _prior_artifact_payloads(lock)
    if payload_overrides:
        payloads = dict(payloads, **payload_overrides)
    for alias, payload in payloads.items():
        path = artifacts_dir / Path(alias).name
        path.write_text(json.dumps(payload))
        artifacts[alias] = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts




def test_prior_inventory_fails_on_missing_changed_or_wrong_stage(tmp_path):
    lock = synthetic_lock()
    artifacts = _prior_artifacts(tmp_path, lock)
    lock = dict(lock, prior_gate_reports=artifacts)
    incomplete = dict(artifacts)
    del incomplete[G2_REPORT_ARTIFACT]
    with pytest.raises(ValueError, match="missing required artifacts"):
        _load_prior_reports(dict(lock, prior_gate_reports=incomplete), artifacts_root=tmp_path)
    corrupted = dict(_prior_artifacts(tmp_path, lock))
    corrupted[G3_REPORT_ARTIFACT] = HEX_F
    with pytest.raises(ValueError, match="Changed prior gate report"):
        _load_prior_reports(dict(lock, prior_gate_reports=corrupted), artifacts_root=tmp_path)
    wrong_stage = _prior_artifacts(tmp_path, lock, {
        G3_REPORT_ARTIFACT: dict(_prior_artifact_payloads(lock)[G3_REPORT_ARTIFACT], stage="G2")})
    with pytest.raises(ValueError, match="must be the G3 stage"):
        _load_prior_reports(dict(lock, prior_gate_reports=wrong_stage), artifacts_root=tmp_path)
    changed_inline = _prior_artifacts(tmp_path, lock, {
        NORMALIZERS_ARTIFACT: dict(lock["normalizers"], scores={"V": 1.0, "C": 1.0, "G": 1.0, "F": 1.0})})
    with pytest.raises(ValueError, match="Inline G4 normalizers differ"):
        _load_prior_reports(dict(lock, prior_gate_reports=changed_inline), artifacts_root=tmp_path)


def test_gate_chain_rejects_failing_prior_gate(tmp_path):
    lock = synthetic_lock()
    failing_gate = _prior_artifacts(tmp_path, lock, {
        G2_REPORT_ARTIFACT: dict(_prior_artifact_payloads(lock)[G2_REPORT_ARTIFACT], gate_passed=False)})
    lock = dict(lock, prior_gate_reports=failing_gate)
    loaded, _ = _load_prior_reports(lock, artifacts_root=tmp_path)
    with pytest.raises(ValueError, match="does not pass its own gate"):
        _validate_gate_chain(lock, loaded)


def test_independent_family_fields_use_real_continual_metrics_singleton():
    metrics = continual_metrics([[0.5]],
                                [{"task_id": "C", "family": "caption",
                                  "metric": "cider_d_fixed_train_idf"}],
                                pre_task_scores=[0.02], normalizers={"C": 0.5})
    assert "vqa" not in metrics["families"]
    assert _family_endpoints("independent", metrics) == {"vqa": None, "caption": None}
    assert _family_af("independent", metrics, "native_AF") == {"vqa": None, "caption": None}
    assert _family_af("independent", metrics, "normalized_AF") == {"vqa": None, "caption": None}
    assert metrics["families"]["caption"]["native_AF"] is None
    assert metrics["tasks"]["C"]["normalized_final"] == pytest.approx(1.0)


def test_lock_accepts_structured_scorer_contracts_and_rejects_bare_hashes(tmp_path):
    lock = synthetic_lock()
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    _parse_lock(path, hashlib.sha256(path.read_bytes()).hexdigest())
    lock["scorer_locks"]["V"] = HEX_D
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        _parse_lock(path, hashlib.sha256(path.read_bytes()).hexdigest())
