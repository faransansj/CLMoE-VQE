"""Official scorer regressions; caption tests require the real pinned PTB jar/Java."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from sema_multimodal.metrics import TaskScorer, continual_metrics, gqa_accuracy, vqav2_accuracy


def row(identifier, references, family="caption"):
    return {"id": identifier, "image": identifier + ".jpg",
        "image_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
        "source_image_id": identifier, "input_text": "Describe the image." if family == "caption" else "What?",
        "target": references[0], "references": list(references),
        "source": "coco2014" if family == "caption" else "vqav2",
        "source_split": "train", "family": family}


@pytest.mark.parametrize("matches,expected", [(0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)])
def test_vqa_official_leave_one_annotator_out(matches, expected):
    references = ["yes"] * matches + ["no"] * (10 - matches)
    assert vqav2_accuracy("yes", references) == pytest.approx(expected)


def test_vqa_official_conditional_normalization_and_whitespace():
    references = ["2"] * 4 + ["three"] * 6
    original = list(references)
    assert vqav2_accuracy("\nThe two!\t", references) == 1.0
    assert references == original
    assert vqav2_accuracy("1,000", ["1000"] * 4 + ["2000"] * 6) == 1.0
    assert vqav2_accuracy("cant", ["can't"] * 4 + ["yes"] * 6) == 1.0
    # Official VQAEval does not normalize unanimous ground truth.
    assert vqav2_accuracy("The two!", ["2"] * 10) == 0.0
    assert vqav2_accuracy("\t2\n", ["2"] * 10) == 1.0


def test_vqa_requires_real_ten_annotator_references():
    with pytest.raises(ValueError, match="ten"):
        vqav2_accuracy("yes", ["yes"])


def test_gqa_exact_official_gold_after_adapter_only():
    assert gqa_accuracy("RED...", "red") == 1.0
    assert gqa_accuracy(" red.", "red") == 0.0
    assert gqa_accuracy("the red", "red") == 0.0
    assert gqa_accuracy("two", "2") == 0.0
    assert gqa_accuracy("red", "RED") == 0.0


def test_accuracy_points_and_ordered_alignment():
    task = {"task_id": "V", "family": "vqa", "metric": "vqav2"}
    scorer = TaskScorer(task)
    rows = [row("v1", ["yes"] * 10, "vqa"), row("v2", ["no"] * 10, "vqa")]
    assert scorer.score(["yes", "no"], rows) == {"score": 100.0, "accuracy": 100.0}
    assert scorer.score(["no", "yes"], rows)["score"] == 0.0
    with pytest.raises(ValueError, match="One string"):
        scorer.score(["yes"], rows)
    with pytest.raises(ValueError, match="family"):
        scorer.score(["yes"], [row("caption", ["yes"])])


@pytest.fixture(scope="module")
def caption_task():
    captions = ["A red bird flies over trees.", "Two dogs run across a grassy field.",
                "A blue boat floats on the lake."]
    return {"task_id": "C", "family": "caption", "metric": "cider_d",
        "train": [row(f"train-{i}", [caption, caption]) for i, caption in enumerate(captions)],
        "validation": [row(f"validation-{i}", [caption, caption]) for i, caption in enumerate(captions)],
        "eval": [row(f"eval-{i}", [caption, caption]) for i, caption in enumerate(captions)]}


@pytest.fixture(scope="module")
def caption_scorer(caption_task):
    return TaskScorer(caption_task)


def test_real_ptb_caption_perfect_wrong_and_empty(caption_task, caption_scorer):
    rows = caption_task["validation"]
    perfect = caption_scorer.score([r["references"][0].upper() for r in rows], rows)
    wrong = caption_scorer.score(["Zebra keyboards sleep beside purple computers."] * len(rows), rows)
    empty = caption_scorer.score([""] * len(rows), rows)
    assert perfect["score"] == pytest.approx(10.0)
    assert perfect["bleu4"] == pytest.approx(1.0, abs=1e-8)
    assert wrong["score"] == 0.0
    assert perfect["bleu4"] > wrong["bleu4"]
    assert empty["score"] == 0.0
    assert empty["bleu4"] == 0.0


def test_fixed_idf_exact_parity_with_upstream_on_train_corpus(caption_task, caption_scorer):
    from pycocoevalcap.cider.cider import Cider
    from sema_multimodal.metrics import _tokenize
    rows = caption_task["train"]
    refs = _tokenize({r["id"]: r["references"] for r in rows})
    preds = _tokenize({r["id"]: [r["target"]] for r in rows})
    official, _ = Cider(n=4, sigma=6.0).compute_score(refs, preds)
    actual = caption_scorer.score([r["target"] for r in rows], rows)
    assert actual["score"] == pytest.approx(official, abs=1e-12)


def test_heldout_references_do_not_change_idf(caption_task, caption_scorer):
    original_df = dict(caption_scorer._df)
    original_lock = copy.deepcopy(caption_scorer.lock)
    rows = caption_task["validation"]
    single = caption_scorer.score([rows[0]["target"]], rows[:1])["score"]
    other = row("unrelated-heldout", ["Rare zebras juggle unusual keyboards."])
    pair = caption_scorer.score([rows[0]["target"], other["target"]], [rows[0], other])["score"]
    alone = caption_scorer.score([other["target"]], [other])["score"]
    assert pair == pytest.approx((single + alone) / 2, abs=1e-12)
    changed = copy.deepcopy(caption_task)
    changed["validation"] = [other]
    changed["eval"] = [other]
    changed_scorer = TaskScorer(changed)
    assert changed_scorer.lock == original_lock
    assert caption_scorer._df == original_df
    assert caption_scorer.lock == original_lock
    assert caption_scorer.score([rows[0]["target"]], rows[:1])["score"] == single


def test_train_reference_change_changes_fixed_lock(caption_task, caption_scorer):
    changed = copy.deepcopy(caption_task)
    changed["train"][0]["references"] = ["Rare zebras juggle unusual keyboards."]
    other = TaskScorer(changed)
    assert other.lock["train_reference_sha256"] != caption_scorer.lock["train_reference_sha256"]
    assert other.lock["document_frequency_sha256"] != caption_scorer.lock["document_frequency_sha256"]


def test_bootstrap_repeated_images_preserve_fixed_idf(caption_task, caption_scorer):
    rows = caption_task["validation"][:1]
    score = caption_scorer.score([rows[0]["target"]], rows)["score"]
    repeated = caption_scorer.score([rows[0]["target"]] * 3, rows * 3)["score"]
    assert repeated == pytest.approx(score, abs=1e-12)


def test_caption_train_id_hash_mismatch_rejected(caption_task, caption_scorer):
    mismatch = copy.deepcopy(caption_task["train"][0])
    mismatch["image_sha256"] = caption_task["train"][1]["image_sha256"]
    with pytest.raises(ValueError, match="identity"):
        caption_scorer.score([mismatch["target"]], [mismatch])
    changed_id = copy.deepcopy(caption_task["train"][0])
    changed_id["id"] = "heldout-renamed"
    with pytest.raises(ValueError, match="identity"):
        caption_scorer.score([changed_id["target"]], [changed_id])


def test_train_idf_requires_unique_images_and_nonempty_references(caption_task):
    changed = copy.deepcopy(caption_task)
    changed["train"][1]["image_sha256"] = changed["train"][0]["image_sha256"]
    with pytest.raises(ValueError, match="unique image"):
        TaskScorer(changed)
    changed = copy.deepcopy(caption_task)
    changed["train"][0]["references"] = []
    with pytest.raises(ValueError, match="reference"):
        TaskScorer(changed)


def test_task_metric_and_family_must_agree():
    with pytest.raises(ValueError, match="family"):
        TaskScorer({"family": "vqa", "metric": "cider_d"})
    with pytest.raises(ValueError, match="Unsupported"):
        TaskScorer({"family": "vqa", "metric": "exact"})


def test_family_balanced_normalization_and_no_unlike_unit_average():
    tasks = [{"task_id": "V", "family": "vqa", "metric": "vqav2"},
             {"task_id": "C", "family": "caption", "metric": "cider_d"},
             {"task_id": "G", "family": "vqa", "metric": "gqa"}]
    matrix = [[80.0], [60.0, 4.0], [50.0, 3.0, 70.0]]
    raw = continual_metrics(matrix, tasks, pre_task_scores=[0.0, 1.0, 20.0])
    assert "AP" not in raw
    assert "family_balanced_normalized_final" not in raw
    assert raw["tasks"]["V"]["AF"] == 30.0
    assert raw["tasks"]["C"]["AF"] == 1.0
    assert raw["tasks"]["G"]["AF"] is None
    assert raw["tasks"]["G"]["pre_to_post_gain"] == 50.0
    assert raw["families"]["vqa"]["native_AF"] == 30.0
    assert raw["families"]["vqa"]["native_subsequent_gain"] == 50.0
    assert raw["families"]["caption"]["native_subsequent_gain"] == 3.0
    normalized = continual_metrics(matrix, tasks, normalizers={"V": 50.0, "C": 2.0, "G": 35.0})
    assert normalized["tasks"]["G"]["normalized_final"] == 2.0  # never clip
    assert normalized["families"]["vqa"]["normalized_final"] == 1.5
    assert normalized["families"]["caption"]["normalized_final"] == 1.5
    assert normalized["family_balanced_normalized_final"] == 1.5
    assert normalized["normalized_matrix"][-1] == [1.0, 1.5, 2.0]


@pytest.mark.parametrize("normalizers", [{}, {"V": 0.0}, {"V": math.nan}, {"V": -1.0}, {"V": True}])
def test_normalization_requires_explicit_positive_complete_constants(normalizers):
    with pytest.raises(ValueError, match="normalizers"):
        continual_metrics([[50.0]], [{"task_id": "V", "family": "vqa", "metric": "vqav2"}],
                          normalizers=normalizers)


def test_single_task_forgetting_is_not_fabricated_zero():
    metrics = continual_metrics([[50.0]], [{"task_id": "V", "family": "vqa", "metric": "vqav2"}],
                                normalizers={"V": 50.0})
    assert metrics["tasks"]["V"]["AF"] is None
    assert metrics["families"]["vqa"]["normalized_AF"] is None


def test_document_frequency_counts_images_not_annotation_duplicates(caption_task, caption_scorer):
    changed = copy.deepcopy(caption_task)
    for training_row in changed["train"]:
        training_row["references"] *= 3
    duplicated = TaskScorer(changed)
    assert duplicated._df == caption_scorer._df
    assert duplicated.lock["train_documents"] == len(caption_task["train"])
    rows = caption_task["validation"]
    predictions = [item["target"] for item in rows]
    assert duplicated.score(predictions, rows) == caption_scorer.score(predictions, rows)


def test_caption_length_and_repetition_penalties(caption_task, caption_scorer):
    rows = caption_task["validation"]
    perfect = caption_scorer.score([item["target"] for item in rows], rows)["score"]
    repeated = caption_scorer.score([item["target"] + " " + item["target"] for item in rows], rows)["score"]
    assert 0.0 < repeated < perfect


def test_prediction_rescore_aligns_ids_and_rejects_missing_duplicate_or_changed_hash(tmp_path):
    from scripts.summarize_multimodal import rescore
    scorer = TaskScorer({"task_id": "V", "family": "vqa", "metric": "vqav2"})
    rows = [row("v1", ["yes"] * 10, "vqa"), row("v2", ["no"] * 10, "vqa")]
    records = [{"id": "v2", "prediction": "no"}, {"id": "v1", "prediction": "yes"}]
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    scores, ordered = rescore(path, rows, scorer)
    assert scores["score"] == 100.0
    assert [record["id"] for record in ordered] == ["v1", "v2"]
    for invalid in (records[:1], [records[0], records[0]],
                    [{"id": "unexpected", "prediction": "no"}, records[1]]):
        path.write_text("".join(json.dumps(record) + "\n" for record in invalid))
        with pytest.raises(ValueError, match="prediction IDs"):
            rescore(path, rows, scorer)
    records[0]["image_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="hash mismatch"):
        rescore(path, rows, scorer)


def test_audit_comparison_rejects_changed_auxiliaries_nonfinite_and_missing_entries():
    from scripts.summarize_multimodal import _assert_same
    expected = {"score": 2.0, "cider_d": 2.0, "bleu4": .4}
    for invalid in ({"score": 2.0, "cider_d": 2.0, "bleu4": .9},
                    {"score": 2.0, "cider_d": 2.0, "bleu4": math.nan},
                    {"score": 2.0, "cider_d": 2.0}):
        with pytest.raises(ValueError):
            _assert_same(expected, invalid, "caption scoring")


def test_report_output_preserves_archived_artifacts_and_source(tmp_path):
    from scripts.summarize_multimodal import ROOT, _check_output
    for path in (ROOT, ROOT / "sema_vqa", ROOT / "artifacts/readout_report",
                 ROOT / "artifacts/multimodal_runs/run"):
        with pytest.raises(ValueError):
            _check_output(path)
    assert _check_output(ROOT / "artifacts/multimodal_report") == ROOT / "artifacts/multimodal_report"
    assert _check_output(tmp_path / "multimodal-report") == tmp_path / "multimodal-report"


@pytest.fixture
def audited_run(tmp_path):
    """Real decoded image bytes and real scorer outputs in a saved-run fixture."""
    from PIL import Image
    from sema_multimodal import metrics as metrics_module
    image_root = tmp_path / "images"
    assets = tmp_path / "assets"
    directory = tmp_path / "multimodal-run"
    image_root.mkdir()
    (assets / "tokenizer").mkdir(parents=True)
    directory.mkdir()
    next_image = 1

    def make_row(source, split, family, reference):
        nonlocal next_image
        identifier = str(next_image)
        next_image += 1
        filename = identifier + ".png"
        Image.new("RGB", (3, 3), (next_image * 23, next_image * 17, next_image * 11)).save(image_root / filename)
        item = row(f"{source}/{split}/{identifier}", [reference] * (10 if family == "vqa" else 1), family)
        item.update(image=filename, image_sha256=hashlib.sha256((image_root / filename).read_bytes()).hexdigest(),
                    source_image_id=identifier, source=source, source_split=split)
        return item

    vqa = {"task_id": "V", "family": "vqa", "metric": "vqav2", "source": "vqav2",
           "train": [make_row("vqav2", "train2014", "vqa", "yes")],
           "validation": [make_row("vqav2", "train2014", "vqa", "yes")],
           "eval": [make_row("vqav2", "val2014", "vqa", "yes")]}
    caption = {"task_id": "C", "family": "caption", "metric": "cider_d", "source": "coco_captions",
               "train": [make_row("coco_captions", "train2014", "caption", "A bird flies over trees."),
                         make_row("coco_captions", "train2014", "caption", "Two dogs run over grass.")],
               "validation": [make_row("coco_captions", "train2014", "caption", "A boat floats on a lake.")],
               "eval": [make_row("coco_captions", "val2014", "caption", "A cat sleeps near a window.")]}
    manifest = {"schema_version": 1, "protocol": "multimodal_development",
                "confirmation_eligible": False, "tasks": [vqa, caption]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    tokenizer_file = assets / "tokenizer/vocab.txt"
    tokenizer_file.write_text("hash inventory fixture\n")
    scorer = TaskScorer(vqa)
    config = {"schema_version": 1, "protocol": manifest["protocol"], "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "image_root": str(image_root), "assets": str(assets),
        "source_sha256": {"sema_multimodal/metrics.py": hashlib.sha256(
            Path(metrics_module.__file__).read_bytes()).hexdigest()},
        "assets_sha256": {"tokenizer/vocab.txt": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()},
        "tokenizer_sha256": {"tokenizer/vocab.txt": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()},
        "scorer_locks": {"V": scorer.lock}, "task_order": ["V"], "method": "shared_lora",
        "eval_split": "validation", "seed": 17, "eval_limit": 0, "oracle_diagnostic": False,
        "image_shuffled_diagnostic": False}
    (directory / "config.json").write_text(json.dumps(config))
    evaluations = []
    for kind, prediction, expert in (("before", "no", -1), ("after", "yes", 0)):
        filename = f"predictions_{kind}_00_V.jsonl"
        generation = {"generated_tokens": 1, "max_new_tokens": 32, "eos": True,
                      "empty": False, "repeated_trigram": False, "limit_reached": False}
        record = {"id": vqa["validation"][0]["id"], "prediction": prediction,
                  "expert": expert, "generation": generation}
        (directory / filename).write_text(json.dumps(record) + "\n")
        evaluations.append({"task_id": "V", "family": "vqa", "metric": "vqav2",
            "scoring": scorer.score([prediction], vqa["validation"]),
            "generation": {key: float(generation[key]) for key in
                           ("generated_tokens", "eos", "empty", "repeated_trigram", "limit_reached")},
            "prediction_counts": {prediction: 1}, "most_frequent_output_fraction": 1.0,
            "route_counts": {expert: 1}, "predictions_file": filename})
    result = {"schema_version": 1, "status": "complete", "completed": 1, "task_order": ["V"],
        "task_metadata": [{"task_id": "V", "family": "vqa", "metric": "vqav2"}],
        "matrix": [[100.0]], "pre_task_scores": [0.0], "pre_to_post_gain": [100.0],
        "oracle_matrix": [[]], "stage_evaluations": [[evaluations[1]]], "stage_resources": [{}]}
    result["metrics"] = continual_metrics(result["matrix"], [vqa], pre_task_scores=result["pre_task_scores"])
    (directory / "results.json").write_text(json.dumps(result))
    events = [{"event": "task_start", "task": 0, "pre_evaluation": evaluations[0]},
              {"event": "evaluation", "task": 0, **evaluations[1]}]
    (directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return directory, config, result, image_root, tokenizer_file


def test_full_run_audit_real_image_hashes_and_fixed_normalization(audited_run):
    from scripts.summarize_multimodal import audit
    directory, config, _, _, _ = audited_run
    raw = audit(directory)
    assert raw["matrix"] == [[100.0]]
    assert raw["oracle_metrics"] is None
    assert raw["audited_predictions"] == 2
    assert raw["image_bytes_verified"] is True
    normalizers = {"schema_version": 1, "eval_split": "validation",
                   "manifest_sha256": config["manifest_sha256"], "scores": {"V": 50.0, "C": 1.0}}
    normalized = audit(directory, normalizer_lock=normalizers)
    assert normalized["metrics"]["tasks"]["V"]["normalized_final"] == 2.0
    normalizers["eval_split"] = "eval"
    with pytest.raises(ValueError, match="G1 validation"):
        audit(directory, normalizer_lock=normalizers)


@pytest.mark.parametrize("field", ["matrix", "pre_to_post_gain", "auxiliary"])
def test_full_audit_rejects_recorded_score_mismatch(audited_run, field):
    from scripts.summarize_multimodal import audit
    directory, _, result, _, _ = audited_run
    if field == "auxiliary":
        result["stage_evaluations"][0][0]["scoring"]["accuracy"] = 20.0
    elif field == "matrix":
        result[field][0][0] = 20.0
    else:
        result[field][0] = 20.0
    (directory / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="score differs"):
        audit(directory)


def test_full_audit_rejects_changed_model_asset(audited_run):
    from scripts.summarize_multimodal import audit
    directory, _, _, _, tokenizer_file = audited_run
    tokenizer_file.write_text("changed tokenizer\n")
    with pytest.raises(ValueError, match="Changed asset"):
        audit(directory)


def test_full_audit_rejects_changed_manifest_and_image_bytes(audited_run):
    from PIL import Image
    from scripts.summarize_multimodal import audit
    directory, config, _, image_root, _ = audited_run
    Image.new("RGB", (3, 3), (0, 0, 0)).save(image_root / "1.png")
    with pytest.raises(ValueError, match="image SHA256 mismatch"):
        audit(directory)
    manifest_path = Path(config["manifest"])
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        audit(directory)
