"""G1 consumer regressions, including real PTB/CIDEr resample equivalence."""

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_multimodal_g1 as g1
from scripts import summarize_multimodal as summary
from sema_multimodal import metrics
from sema_multimodal.data import SOURCE_SPECS
from sema_multimodal.metrics import TaskScorer, continual_metrics


CAPTIONS = ["A red bird flies over green trees.", "Two dogs run across a grassy field.",
            "A blue boat floats on the quiet lake.", "Three people stand beside a white car."]


def make_row(source, split, identifier, reference):
    family = SOURCE_SPECS[source][0]
    return {"id": f"{source}/{split}/{identifier}", "source": source, "source_split": split,
            "source_image_id": str(identifier), "image": f"{source}/{identifier}.jpg",
            "image_sha256": hashlib.sha256(f"{source}/{identifier}".encode()).hexdigest(),
            "family": family, "input_text": "Describe the image." if family == "caption" else "What is shown?",
            "target": reference, "references": [reference] * (10 if source == "vqav2" else 2 if family == "caption" else 1)}


@pytest.fixture
def mini_grid(tmp_path, monkeypatch):
    # Only fixture sample sizes are reduced. Production has no CLI bypass and
    # still demands the entire 2048/256/512 manifest, tested separately below.
    monkeypatch.setattr(g1, "SPLIT_SIZES", {"train": 4, "validation": 4, "eval": 2})
    tasks, identifier = [], 1
    for source, (family, metric, task_id) in SOURCE_SPECS.items():
        task = {"task_id": task_id, "source": source, "family": family, "metric": metric}
        for split, count in g1.SPLIT_SIZES.items():
            official_split = ("val2014" if source in ("vqav2", "coco_captions") else "test" if source == "flickr30k" else "val") if split == "eval" else ("train2014" if source in ("vqav2", "coco_captions") else "train")
            task[split] = []
            for index in range(count):
                reference = CAPTIONS[index] if family == "caption" else ("red", "blue", "green", "yellow")[index]
                task[split].append(make_row(source, official_split, identifier, reference))
                identifier += 1
        tasks.append(task)
    manifest = {"schema_version": 1, "protocol": "multimodal_cl", "tasks": tasks,
                "task_order": [task["task_id"] for task in tasks]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_hash = summary._sha256(manifest_path)
    assets = tmp_path / "assets"
    (assets / "tokenizer").mkdir(parents=True)
    (assets / "tokenizer/vocab.txt").write_text("fixture tokenizer hash inventory")
    (assets / "model_base_capfilt_large.pth").write_bytes(b"fixture model hash inventory")
    asset_hashes = {name: summary._sha256(assets / name) for name in ("tokenizer/vocab.txt", "model_base_capfilt_large.pth")}
    root = tmp_path / "multimodal-runs"
    directories = []
    for task in tasks:
        for lr in g1.LEARNING_RATES:
            for epochs in g1.EPOCHS:
                directory = root / f"{task['task_id']}-{lr}-{epochs}"
                directory.mkdir(parents=True)
                config = {**copy.deepcopy(g1.FIXED_SETTINGS), "lr": lr, "epochs": epochs,
                    "task_order": [task["task_id"]], "task_ids": [task["task_id"]],
                    "protocol": manifest["protocol"], "manifest": str(manifest_path), "manifest_sha256": manifest_hash,
                    "assets": str(assets), "image_root": None,
                    "source_sha256": {str(Path(metrics.__file__)): summary._sha256(metrics.__file__)},
                    "assets_sha256": asset_hashes, "tokenizer_sha256": {"tokenizer/vocab.txt": asset_hashes["tokenizer/vocab.txt"]},
                    "scorer_locks": {task["task_id"]: {"metric": task["metric"]}}, "oracle_diagnostic": False}
                (directory / "config.json").write_text(json.dumps(config))
                directories.append(directory)
    return root, directories, manifest, manifest_path, manifest_hash


def materialize_runs(mini_grid, *, fail_task=None):
    """Emit genuine scorer/auditor fixtures, never mock the independent audit."""
    _, directories, manifest, _, _ = mini_grid
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    scorers = {task_id: TaskScorer(task) for task_id, task in tasks.items()}
    score_cache = {}
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        task_id = config["task_order"][0]
        task, scorer = tasks[task_id], scorers[task_id]
        config["scorer_locks"] = {task_id: scorer.lock}
        (directory / "config.json").write_text(json.dumps(config))
        perfect = [row["target"] for row in task["validation"]]
        wrong = ["unrelated zebra keyboard"] * len(perfect)
        predictions = {"before": [perfect[0], *wrong[1:]], "after": perfect,
                       "image_shuffled": perfect[1:] + perfect[:1]}
        if config["lr"] == 1e-4 and config["epochs"] == 10:
            predictions["after"] = wrong
        if task_id == fail_task and config["lr"] == 3e-4 and config["epochs"] == 5:
            predictions["after"] = [""] * len(perfect)
        if fail_task == "zero" and task_id == "G":
            predictions["after"] = [""] * len(perfect)
        evaluations = {}
        for kind, texts in predictions.items():
            expert = -1 if kind == "before" else 0
            filename = f"image_shuffled_00_{task_id}.jsonl" if kind == "image_shuffled" else f"predictions_{kind}_00_{task_id}.jsonl"
            records = []
            for row, prediction in zip(task["validation"], texts):
                records.append({"id": row["id"], "image_sha256": row["image_sha256"], "prediction": prediction,
                    "expert": expert, "generation": {"generated_tokens": len(prediction.split()),
                    "eos": True, "empty": not prediction.strip(), "repeated_trigram": False, "limit_reached": False}})
            # Reversed JSONL verifies that the consumer does NOT pair by file position.
            (directory / filename).write_text("".join(json.dumps(record) + "\n" for record in reversed(records)))
            cache_key = (task_id, tuple(texts))
            if cache_key not in score_cache:
                score_cache[cache_key] = scorer.score(texts, task["validation"])
            evaluations[kind] = {"task_id": task_id, "family": task["family"], "metric": task["metric"],
                "scoring": score_cache[cache_key], "generation": {key: sum(record["generation"][key] for record in records) / len(records)
                    for key in ("generated_tokens", "eos", "empty", "repeated_trigram", "limit_reached")},
                "prediction_counts": dict(Counter(texts)), "most_frequent_output_fraction": max(Counter(texts).values()) / len(texts),
                "route_counts": {str(expert): len(texts)}, "predictions_file": filename}
        before, after = evaluations["before"]["scoring"]["score"], evaluations["after"]["scoring"]["score"]
        epochs = config["epochs"]
        result = {"schema_version": 1, "status": "complete", "completed": 1, "task_order": [task_id],
                  "task_metadata": [{key: task[key] for key in ("task_id", "family", "metric")}],
                  "matrix": [[after]], "pre_task_scores": [before], "pre_to_post_gain": [after - before], "oracle_matrix": [[]],
                  "stage_evaluations": [[evaluations["after"], {"diagnostic": "image_shuffled", **evaluations["image_shuffled"]}]],
                  "stage_resources": [{"task_id": task_id, "steps": epochs, "image_exposures": epochs * 4,
                      "current_exposures": epochs * 4, "replay_exposures": 0, "router_memory_bytes": 0, "raw_replay_bytes": 0}]}
        result["metrics"] = continual_metrics(result["matrix"], [task], pre_task_scores=[before])
        (directory / "results.json").write_text(json.dumps(result))
        events = [{"event": "task_start", "pre_evaluation": evaluations["before"]},
                  {"event": "evaluation", **evaluations["after"]},
                  {"event": "image_shuffled", **evaluations["image_shuffled"]}]
        (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


def change_config(directory, field, value):
    path = directory / "config.json"
    config = json.loads(path.read_text())
    config[field] = value
    path.write_text(json.dumps(config))


def test_grid_requires_all_sixteen_distinct_conditions(mini_grid):
    root, directories, manifest, _, digest = mini_grid
    assert len(g1.discover_runs([root])) == 16
    assert len(g1.validate_grid(directories, manifest, digest)) == 16
    with pytest.raises(ValueError, match="exactly 16"):
        g1.discover_runs(directories[:-1])
    with pytest.raises(ValueError, match="Incomplete"):
        g1.validate_grid(directories[:-1], manifest, digest)
    with pytest.raises(ValueError, match="Duplicate run directory"):
        g1.discover_runs([root, directories[0]])
    change_config(directories[1], "epochs", 5)
    with pytest.raises(ValueError, match="Duplicate G1 grid condition"):
        g1.validate_grid(directories, manifest, digest)


@pytest.mark.parametrize("field,value", [("seed", 137), ("method", "shared_lora"), ("eval_split", "eval"),
    ("train_limit", 1), ("eval_limit", 1), ("smoke_output", True), ("image_shuffled_diagnostic", False),
    ("batch_size", 8), ("rank", 16), ("epochs", 2), ("lr", 0.003), ("task_order", ["V", "C"]),
    ("task_ids", ["C"]), ("manifest_sha256", "0" * 64), ("eval_limit", False),
    ("scorer_locks", {"wrong": {}}), ("source_sha256", {"wrong-source.py": "0" * 64}),
    ("assets_sha256", {"model_base_capfilt_large.pth": "0" * 64}),
    ("generation", {"strategy": "sampling"})])

def test_grid_rejects_wrong_configs_and_hash_contracts(mini_grid, field, value):
    _, directories, manifest, _, digest = mini_grid
    change_config(directories[0], field, value)
    with pytest.raises(ValueError):
        g1.validate_grid(directories, manifest, digest)


def test_manifest_cannot_relabel_sources_or_hide_smoke_budget(mini_grid, monkeypatch):
    _, _, manifest, _, _ = mini_grid
    changed = copy.deepcopy(manifest)
    caption = next(task for task in changed["tasks"] if task["task_id"] == "C")
    flickr = next(task for task in changed["tasks"] if task["task_id"] == "F")
    caption["task_id"], flickr["task_id"] = "F", "C"
    changed["task_order"] = [task["task_id"] for task in changed["tasks"]]
    with pytest.raises(ValueError, match="Wrong source"):
        g1.validate_g1_manifest(changed)
    monkeypatch.setattr(g1, "SPLIT_SIZES", {"train": 2048, "validation": 256, "eval": 512})
    with pytest.raises(ValueError, match="2048 unique images"):
        g1.validate_g1_manifest(manifest)


@pytest.mark.parametrize("fail_task", [None, "G", "zero"])
def test_real_audit_gate_all_trials_normalizer_consumer_and_cli(mini_grid, monkeypatch, tmp_path, fail_task):
    root, directories, manifest, manifest_path, digest = mini_grid
    materialize_runs(mini_grid, fail_task=fail_task)
    report, normalizers = g1.adjudicate([root], manifest_path)
    assert report["input_valid"] is True
    assert report["gate_passed"] is (fail_task is None)
    assert report["pilot_allowed"] is (fail_task is None)
    assert report["pilot_launched"] is False
    assert len(report["trials"]) == len(report["audits"]) == 16
    assert any(trial["gain"] < 0 for trial in report["trials"])
    assert all(set(trial["scores"]) == {"before", "after", "image_shuffled"} for trial in report["trials"])
    for task_id, task_report in report["tasks"].items():
        assert len(task_report["bootstrap"]["gain_samples"]) == 2000
        assert task_report["bootstrap"]["seed"] == 20261013
        assert task_report["bootstrap"]["images"] == 4
        assert task_report["normalizer_selection"]["lr"] == 1e-4
        assert task_report["normalizer_selection"]["epochs"] == 5
        assert task_report["image_shuffled_delta"] == pytest.approx(
            task_report["scores"]["after"]["score"] - task_report["scores"]["image_shuffled"]["score"])
    if fail_task is None:
        assert normalizers["manifest_sha256"] == digest
        assert set(normalizers["scores"]) == {task["task_id"] for task in manifest["tasks"]}
        assert all(score > 0 for score in normalizers["scores"].values())
        consumed = summary.audit(directories[0], normalizer_lock=normalizers)
        assert consumed["metrics"]["tasks"]["V"]["normalized_final"] == 1.0
        # Equal declarations are not proof: the delegated independent audit
        # must still compare the current source bytes against all saved hashes.
        original_source = json.loads((directories[0] / "config.json").read_text())["source_sha256"]
        for directory in directories:
            change_config(directory, "source_sha256", {name: "0" * 64 for name in original_source})
        with pytest.raises(ValueError, match="Changed source file"):
            g1.adjudicate([root], manifest_path)
        for directory in directories:
            change_config(directory, "source_sha256", original_source)
    else:
        assert normalizers is None
        assert report["normalizers_usable"] is False
        assert "G: positive_mean_gain" in report["failure_reasons"]
        assert "G: positive_ci_lower" in report["failure_reasons"]
        assert "G: empty_fraction_at_most_0.01" in report["failure_reasons"]
        assert "G: mode_fraction_below_0.9" in report["failure_reasons"]
        if fail_task == "zero":
            assert "G: normalizer_must_be_positive" in report["failure_reasons"]
            assert report["tasks"]["G"]["normalizer_selection"]["score"] == 0
    # Exercise CLI serialization and immutable namespaces without rerunning the audit.
    monkeypatch.setattr(g1, "adjudicate", lambda *args, **kwargs: (report, normalizers))
    output = tmp_path / "multimodal-report" / "g1"
    argv = ["--runs", str(root), "--manifest", str(manifest_path), "--output", str(output)]
    assert g1.main(argv) == (0 if fail_task is None else 1)
    saved = json.loads((output / "g1.json").read_text())
    assert saved["gate_passed"] is (fail_task is None)
    assert (output / "normalizers.json").exists() is (fail_task is None)
    if fail_task is None:
        assert saved["normalizers_sha256"] == summary._sha256(output / "normalizers.json")
    with pytest.raises(ValueError, match="new report directory"):
        g1.main(argv)


def test_bootstrap_is_paired_deterministic_and_uses_percentile_interval():
    before = np.asarray([0.0, 25.0, 75.0, 90.0])
    after = before + 3.0
    report = g1.paired_bootstrap(before, after)
    assert report == g1.paired_bootstrap(before, after)
    assert report["mean_gain"] == report["ci_lower"] == report["ci_upper"] == 3.0
    assert report["gain_samples"] == [3.0] * 2000
    varying = g1.paired_bootstrap(before, after[::-1])
    expected = np.quantile(varying["gain_samples"], [0.025, 0.975], method="linear")
    assert [varying["ci_lower"], varying["ci_upper"]] == expected.tolist()
    assert varying["ci_lower"] < 0 < varying["ci_upper"]


@pytest.mark.parametrize("before,after", [([], []), ([0], [0, 1]), ([float("nan")], [0]), ([0], [float("inf")])])
def test_bootstrap_rejects_invalid_score_vectors(before, after):
    with pytest.raises(ValueError):
        g1.paired_bootstrap(before, after)


def test_generation_guard_boundary_and_literal_whole_outputs():
    records = [{"prediction": str(index)} for index in range(100)]
    records[0]["prediction"] = " \n"
    assert g1.generation_guards(records)["passed"] is True  # exactly 1% is allowed
    records[1]["prediction"] = ""
    assert g1.generation_guards(records)["checks"]["empty_fraction_at_most_0.01"] is False
    records = [{"prediction": "same" if index < 90 else str(index)} for index in range(100)]
    assert g1.generation_guards(records)["checks"]["mode_fraction_below_0.9"] is False
    records[0]["prediction"] = "Same"  # literal complete output, not token frequency
    assert g1.generation_guards(records)["passed"] is True


@pytest.fixture(scope="module")
def caption_case():
    train = [make_row("coco_captions", "train2014", index + 1, caption) for index, caption in enumerate(CAPTIONS)]
    rows = [make_row("coco_captions", "train2014", index + 100, caption) for index, caption in enumerate(CAPTIONS)]
    rows[0]["references"].extend(["It's a red bird, flying (over trees)!", "A bird bird bird flies away."])
    rows[1]["references"].append("Dogs run; two dogs sprint on grass.")
    task = {"task_id": "C", "family": "caption", "metric": "cider_d", "train": train, "validation": rows}
    return TaskScorer(task), rows


def test_caption_fixed_idf_decomposition_matches_real_corpus_bootstrap(caption_case, monkeypatch):
    scorer, rows = caption_case
    before = ["A red bird.", "", "unrelated zebra keyboard", "Three people beside a car car car."]
    after = ["A RED bird flies over green trees!", "Two dogs run across a grassy field.",
             "A blue boat floats on the quiet lake. " * 2, "Three people stand beside a white car."]
    tokenizations = []
    original_tokenize = g1._tokenize

    def counted_tokenize(captions):
        tokenizations.append(len(captions))
        return original_tokenize(captions)

    monkeypatch.setattr(g1, "_tokenize", counted_tokenize)
    fixed_df, lock = copy.deepcopy(scorer._df), copy.deepcopy(scorer.lock)
    contributions = [g1.primary_image_scores(scorer, predictions, rows) for predictions in (before, after)]
    assert len(tokenizations) == 2  # Not one Java process per image or bootstrap draw.
    rng = np.random.default_rng(62)
    draws = [np.asarray([0, 0, 0, 0]), np.asarray([3, 1, 3, 1]), np.asarray([0, 1, 2, 3])]
    draws += [rng.integers(0, len(rows), size=len(rows)) for _ in range(5)]
    for draw in draws:
        sampled_rows = [rows[index] for index in draw]
        actual = []
        for predictions, values in zip((before, after), contributions):
            full_corpus = scorer.score([predictions[index] for index in draw], sampled_rows)
            assert float(np.mean(values[draw])) == pytest.approx(full_corpus["score"], rel=0, abs=1e-12)
            actual.append(full_corpus["score"])
        assert float(np.mean(contributions[1][draw]) - np.mean(contributions[0][draw])) == pytest.approx(actual[1] - actual[0], rel=0, abs=1e-12)
    g1.paired_bootstrap(*contributions)
    assert len(tokenizations) == 2
    assert scorer._df == fixed_df
    assert scorer.lock == lock


@pytest.mark.parametrize("source", ["vqav2", "gqa"])
def test_accuracy_decomposition_uses_image_not_reference_unit(source):
    rows = [make_row(source, "train2014" if source == "vqav2" else "train", index + 1, color)
            for index, color in enumerate(("red", "green", "blue"))]
    if source == "vqav2":
        rows[0]["references"] = ["red"] * 2 + ["blue"] * 8
    scorer = TaskScorer({"family": "vqa", "metric": SOURCE_SPECS[source][1]})
    predictions = ["red", "wrong", "blue"]
    values = g1.primary_image_scores(scorer, predictions, rows)
    draw = [0, 0, 2]
    assert float(np.mean(values[draw])) == pytest.approx(scorer.score([predictions[index] for index in draw], [rows[index] for index in draw])["score"])
    with pytest.raises(ValueError, match="unique image"):
        g1.primary_image_scores(scorer, ["red", "red"], [rows[0], rows[0]])
    duplicate_identity = copy.deepcopy(rows[1])
    duplicate_identity["source_image_id"] = rows[0]["source_image_id"]
    with pytest.raises(ValueError, match="unique image identities"):
        g1.primary_image_scores(scorer, ["red", "red"], [rows[0], duplicate_identity])


def test_prediction_pairing_rejects_duplicate_ids_and_trained_frozen_base(mini_grid):
    _, directories, manifest, _, _ = mini_grid
    rows = manifest["tasks"][0]["validation"]
    path = directories[0] / "predictions_before_00_V.jsonl"
    records = [{"id": row["id"], "prediction": row["target"], "expert": -1} for row in rows]
    path.write_text("".join(json.dumps(record) + "\n" for record in reversed(records)))
    assert [record["id"] for record in g1._ordered_predictions(path, rows, -1)] == [row["id"] for row in rows]
    records[-1] = records[0]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="duplicate prediction"):
        g1._ordered_predictions(path, rows, -1)
    records = [{"id": row["id"], "prediction": row["target"], "expert": 0} for row in rows]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="frozen-base"):
        g1._ordered_predictions(path, rows, -1)


def test_incomplete_or_replay_budget_cannot_be_independent_evidence(mini_grid):
    task = mini_grid[2]["tasks"][0]
    good = {"stage_resources": [{"task_id": task["task_id"], "steps": 5, "image_exposures": 20,
        "current_exposures": 20, "replay_exposures": 0, "router_memory_bytes": 0, "raw_replay_bytes": 0}]}
    g1._check_training_budget(good, task, 5)
    for field, value in (("steps", 4), ("image_exposures", 19), ("current_exposures", 19), ("replay_exposures", 1)):
        bad = copy.deepcopy(good)
        bad["stage_resources"][0][field] = value
        with pytest.raises(ValueError, match="training budget"):
            g1._check_training_budget(bad, task, 5)


def test_cli_invalid_evidence_cannot_emit_normalizers_or_allow_pilot(tmp_path):
    output = tmp_path / "multimodal-report"
    assert g1.main(["--runs", str(tmp_path / "missing-runs"), "--manifest", str(tmp_path / "missing.json"),
                    "--output", str(output)]) == 2
    report = json.loads((output / "g1.json").read_text())
    assert report["input_valid"] is report["gate_passed"] is report["pilot_allowed"] is False
    assert report["failure_reasons"]
    assert not (output / "normalizers.json").exists()


def test_cli_refuses_archived_and_overlapping_namespaces(mini_grid):
    root, _, _, manifest_path, _ = mini_grid
    for output in (summary.ROOT / "artifacts/readout_report/g1", root / "new-report"):
        with pytest.raises(ValueError):
            g1.main(["--runs", str(root), "--manifest", str(manifest_path), "--output", str(output)])
