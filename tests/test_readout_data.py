import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest

from scripts import prepare_readout_data as fresh


def save_image(root, name, color):
    path = root / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 3), color).save(path, format="PNG")
    return path


def make_row(identifier, image, protocol="cl4vqa", task="one", split="train"):
    return {"id": f"{protocol}/{task}/{split}/{identifier}", "source_id": str(identifier),
            "image": image, "question": "Question?", "answer": "Original"}


def test_old_eval_excludes_identity_ids_and_differently_named_identical_bytes(tmp_path):
    old_path = save_image(tmp_path, "old.png", (1, 2, 3))
    alias_path = tmp_path / "images" / "renamed.png"
    alias_path.write_bytes(old_path.read_bytes())
    save_image(tmp_path, "different.png", (4, 5, 6))
    save_image(tmp_path, "clean.png", (7, 8, 9))
    verifier = fresh.ImageVerifier(tmp_path)
    old = fresh.OldEvaluation()
    old.add({"protocol": "cl4vqa", "tasks": [{"name": "one", "eval": [
        make_row("historical", "old.png", split="eval")]}]}, verifier.verify(["old.png"]))
    candidates = [make_row("identity", "old.png"), make_row("alias", "renamed.png"),
                  make_row("historical", "different.png"), make_row("clean", "clean.png")]
    untouched = copy.deepcopy(candidates)
    for source_split, limits in (("train", {"train": 10, "validation": 0}), ("eval", {"eval": 10})):
        selected, stats = fresh.select_rows(candidates, "cl4vqa", "one", source_split,
                                             limits, 20261008, verifier, old, set())
        assert [row["source_id"] for rows in selected.values() for row in rows] == ["clean"]
        assert stats["excluded"] == {"old_eval_image_identity": 1, "old_eval_id": 1,
                                      "old_eval_image_sha256": 1}
    assert candidates == untouched


def test_global_partitions_across_tasks_and_protocols_use_actual_bytes(tmp_path, monkeypatch):
    rows = []
    for index in range(160):
        name = f"image-{index}.png"
        path = save_image(tmp_path, name, (index, 1, 2))
        (path.parent / f"alias-{index}.png").write_bytes(path.read_bytes())
        rows.append(make_row(index, name))
    # An old evaluated image appears as a renamed image in all full source pools.
    old_row = make_row("old-question", "image-0.png", split="eval")
    old_manifest = tmp_path / "old.json"
    old_manifest.write_text(json.dumps({"protocol": "cl4vqa", "image_root": "images",
                                       "tasks": [{"name": "one", "eval": [old_row]}]}))
    old_bytes = old_manifest.read_bytes()
    monkeypatch.setattr(fresh.original, "CL4_TASKS", ("one", "two"))
    monkeypatch.setattr(fresh.original, "STREAM_TASKS", (("other", "test", "clmoe_exact"),))

    def source(root, protocol, task, split, eval_split, sources):
        sources[f"{task}/{split}"] = {"sha256": "synthetic-source"}
        return [make_row(f"{split}-{index}", f"alias-{index}.png" if task != "one" else row["image"],
                         protocol, task, split) for index, row in enumerate(rows)]

    monkeypatch.setattr(fresh, "load_rows", source)
    args = SimpleNamespace(root=tmp_path, old_manifests=[old_manifest], protocol=["cl4vqa", "vqa_stream"],
                           train_limit=24, validation_limit=8, eval_limit=4, seed=20261008,
                           download_images=False, coco_archives=None)
    result = fresh.prepare(args)
    assert result == fresh.prepare(args)
    assert old_manifest.read_bytes() == old_bytes
    old_digest = hashlib.sha256((tmp_path / "images/image-0.png").read_bytes()).hexdigest()
    groups = {split: {row["image_sha256"] for manifest in result for task in manifest["tasks"]
                      for row in task[split]} for split in fresh.SPLITS}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["eval"]
    assert not groups["validation"] & groups["eval"]
    assert old_digest not in set.union(*groups.values())
    for manifest in result:
        assert manifest["vision_partition"]["scope_protocols"] == ["cl4vqa", "vqa_stream"]
        assert not any(manifest["vision_partition"]["overlap_counts"].values())
        assert manifest["images"]["ready"]
        for task in manifest["tasks"]:
            for split, count in (("train", 24), ("validation", 8), ("eval", 4)):
                assert len(task[split]) == count
                assert len({row["id"] for row in task[split]}) == count
                assert task["selection"][split]["shortage"] == 0
                assert task["selection"][split]["selected_ids_sha256"] == fresh.digest_json(
                    [row["id"] for row in task[split]])
                for row in task[split]:
                    assert row["image_sha256"] == hashlib.sha256((tmp_path / "images" / row["image"]).read_bytes()).hexdigest()
            assert all("/train/" in row["id"] for row in task["validation"])
            assert all("/eval/" in row["id"] for row in task["eval"])


def test_zero_means_empty_and_shortage_never_duplicates(tmp_path):
    verifier = fresh.ImageVerifier(tmp_path)
    old = fresh.OldEvaluation()
    missing = [make_row("missing", "does-not-exist.png")]
    selected, stats = fresh.select_rows(missing, "cl4vqa", "one", "eval", {"eval": 0},
                                         20261008, verifier, old, set())
    assert selected == {"eval": []}
    assert stats["considered_rows"] == 0
    selected, _ = fresh.select_rows(missing, "cl4vqa", "one", "train", {"train": 0, "validation": 0},
                                     20261008, verifier, old, set())
    assert selected == {"train": [], "validation": []}
    save_image(tmp_path, "available.png", (10, 11, 12))
    selected, stats = fresh.select_rows([make_row("single", "available.png")], "cl4vqa", "one", "eval",
                                         {"eval": 512}, 20261008, verifier, old, set())
    assert len(selected["eval"]) == 1
    assert stats["source_normalized_rows"] == 1


def test_missing_and_corrupt_images_abort_without_availability_resampling(tmp_path):
    verifier = fresh.ImageVerifier(tmp_path)
    save_image(tmp_path, "good.png", (5, 6, 7))
    with pytest.raises(FileNotFoundError, match="No availability-based replacement"):
        fresh.select_rows([make_row("good", "good.png"), make_row("missing", "missing.png")],
                          "cl4vqa", "one", "eval", {"eval": 1}, 20261008,
                          verifier, fresh.OldEvaluation(), set())
    (tmp_path / "images/broken.png").write_bytes(b"not an image")
    with pytest.raises(OSError):
        verifier.verify(["broken.png"])


def test_cached_hash_requires_unchanged_file_proof(tmp_path):
    save_image(tmp_path, "image.png", (0, 0, 0))
    verifier = fresh.ImageVerifier(tmp_path)
    initial = verifier.verify(["image.png"])["image.png"]
    assert verifier.verify(["image.png"])["image.png"] == initial
    save_image(tmp_path, "image.png", (255, 255, 255))
    changed = verifier.verify(["image.png"])["image.png"]
    assert changed != initial
    assert changed == hashlib.sha256((tmp_path / "images/image.png").read_bytes()).hexdigest()


def test_original_train_consensus_and_gqa_labels_preserved(tmp_path, monkeypatch):
    answers = ["stop"] * 6 + ["red"] * 3 + ["sign"]
    text_reference = {"data": [{"image_id": "text-image", "question_id": 42,
                                "question": "Question?", "answers": answers}]}
    instruction = {"question_id": "text-image", "image": "text-image.jpg", "conversations": [
        {"from": "human", "value": "Question?"}, {"from": "gpt", "value": "stop"}]}
    monkeypatch.setattr(fresh.original, "fetch_json",
                        lambda root, sources, key, *args: [instruction] if key == "TextVQA/train_identity_source" else text_reference)
    train = dict(make_row("text-image", "text-image.jpg", task="TextVQA"),
                 answer="stop", source_row_index=0, turn_index=0)
    validation = dict(train, id=train["id"] + "/validation")
    selected = {"train": [train], "validation": [validation]}
    snapshot = copy.deepcopy(selected)
    enriched = fresh.enrich_selected(tmp_path, "TextVQA", "train", [train, validation], {})
    task = {"metric": "textvqa"}
    fresh.finalize_development(task, selected, enriched)
    assert selected == snapshot
    assert task["train"][0]["answer"] == "stop"
    assert task["validation"][0]["answers"] == answers
    assert task["validation"][0]["reference_question_ids"] == ["42"]

    reference_dir = tmp_path / "cache/references"
    reference_dir.mkdir(parents=True)
    (reference_dir / "train_balanced_questions.json").write_text(json.dumps({
        "qid": {"imageId": "gqa-image", "question": "Question?", "answer": "original"}}))
    train = make_row("gqa-image", "gqa.png", task="GQA")
    validation = dict(train, id=train["id"] + "/validation")
    selected = {"train": [train], "validation": [validation]}
    enriched = fresh.enrich_selected(tmp_path, "GQA", "train", [train, validation], {})
    task = {"metric": "gqa"}
    fresh.finalize_development(task, selected, enriched)
    assert task["train"][0]["answer"] == "Original"
    assert task["validation"][0]["answer"] == "original"
    assert task["validation"][0]["instruction_answer"] == "Original"
    assert task["validation"][0]["reference_question_ids"] == ["qid"]


def test_official_eval_reference_fidelity(tmp_path, monkeypatch):
    answers = ["yes"] * 7 + ["no"] * 3
    monkeypatch.setattr(fresh.original, "fetch_json", lambda *args: {
        "data": [{"question_id": 42, "answers": answers}]})
    text_row = dict(make_row("42", "text.png", task="TextVQA", split="eval"), source_question_id="42")
    result = fresh.enrich_selected(tmp_path, "TextVQA", "eval", [text_row], {})
    assert result[0]["answers"] == answers
    assert result[0]["answer"] == "Original"
    monkeypatch.setattr(fresh.original, "gqa_reference", lambda *args: {
        "42": {"answer": "original", "isBalanced": True}})
    gqa_row = dict(make_row("42", "gqa.png", task="GQA", split="eval"), source_question_id="42")
    result = fresh.enrich_selected(tmp_path, "GQA", "eval", [gqa_row], {})
    assert result[0]["answer"] == "original"
    assert result[0]["instruction_answer"] == "Original"


def test_refuses_existing_output_without_touching_history(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    existing = output / "cl4vqa.json"
    existing.write_text("historical bytes")
    monkeypatch.setattr("sys.argv", ["prepare_readout_data", "--root", str(tmp_path),
                                    "--output-dir", str(output), "--protocol", "cl4vqa",
                                    "--old-manifests", str(existing)])
    with pytest.raises(SystemExit):
        fresh.main()
    assert existing.read_text() == "historical bytes"


def test_old_training_alias_forbidden_in_confirmation_but_allowed_in_development(tmp_path):
    for index, split in enumerate(fresh.SPLITS):
        path = save_image(tmp_path, f"old-{split}.png", (index, 33, 44))
        (path.parent / f"alias-{split}.png").write_bytes(path.read_bytes())
    save_image(tmp_path, "unseen-id-image.png", (81, 82, 83))
    save_image(tmp_path, "clean.png", (91, 92, 93))
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps({
        "protocol": "cl4vqa", "image_root": "images", "tasks": [{
            "name": "one",
            **{split: [make_row(f"old-{split}", f"old-{split}.png", split=split)]
               for split in fresh.SPLITS},
        }],
    }))
    verifier = fresh.ImageVerifier(tmp_path)
    old = fresh.load_old_evaluation([old_path], tmp_path, verifier)
    candidates = [
        make_row("new-train-alias", "alias-train.png"),
        make_row("new-train-identity", "old-train.png"),
        make_row("old-train", "unseen-id-image.png"),
        make_row("new-validation-alias", "alias-validation.png"),
        make_row("new-eval-alias", "alias-eval.png"),
        make_row("clean", "clean.png"),
    ]
    confirmation, stats = fresh.select_rows(candidates, "cl4vqa", "one", "eval", {"eval": 10},
                                            20261008, verifier, old, set())
    assert [row["source_id"] for row in confirmation["eval"]] == ["clean"]
    assert stats["excluded"] == {
        "old_observed_image_sha256": 2, "old_observed_image_identity": 1,
        "old_observed_id": 1, "old_eval_image_sha256": 1,
    }
    for split, limits in (("train", {"train": 10, "validation": 0}),
                          ("validation", {"train": 0, "validation": 10})):
        development, _ = fresh.select_rows(candidates, "cl4vqa", "one", "train", limits,
                                            20261008, verifier, old, set())
        assert {row["source_id"] for row in development[split]} == {
            "new-train-alias", "new-train-identity", "old-train", "new-validation-alias", "clean",
        }
    assert old.row_count == 1
    assert old.observed_row_count == 3


@pytest.mark.parametrize("split", ["train2014", "val2014"])
def test_bulk_coco_preserves_exact_candidate_bytes_and_refuses_missing_member(tmp_path, split):
    name = f"COCO_{split}_000000000001.jpg"
    candidate = f"source-prefix/{name}"
    source = save_image(tmp_path, candidate, (23, 57, 91))
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "archives"
    cache.mkdir()
    with zipfile.ZipFile(cache / f"{split}.zip", "w") as archive:
        archive.write(source, f"{split}/{name}")
    source.unlink()
    with fresh.ImageVerifier(tmp_path, download=True, coco_archives=cache) as verifier:
        assert verifier.verify([candidate]) == {candidate: expected}
        missing = f"COCO_{split}_000000000002.jpg"
        with pytest.raises(KeyError):
            verifier.verify([missing])
        assert not (tmp_path / "images" / missing).exists()


def test_duplicate_textvqa_questions_keep_distinct_original_consensus():
    from sema_vqa.metrics import score_prediction

    instructions = [{"question_id": "poster", "image": "poster.jpg", "conversations": [
        {"from": "human", "value": "What comes crawling?"}, {"from": "gpt", "value": "love"}]} for _ in range(2)]
    reference = {"data": [
        {"image_id": "poster", "question_id": 16138, "question": "What comes crawling?",
         "answers": ["love"] * 9 + ["love comes crawling"]},
        {"image_id": "poster", "question_id": 16139, "question": "What comes crawling?",
         "answers": ["love"] * 10},
    ]}
    rows, _ = fresh.original.normalize(instructions, "vqa_stream", "TextVQA", "train")
    snapshot = copy.deepcopy(rows)
    enriched = fresh.attach_textvqa_training_references(rows, instructions, reference)
    assert rows == snapshot
    assert score_prediction("love comes crawling", enriched[0], "textvqa") == pytest.approx(0.3)
    assert score_prediction("love comes crawling", enriched[1], "textvqa") == 0
    assert [row["reference_question_ids"] for row in enriched] == [["16138"], ["16139"]]
    assert [row["answer"] for row in enriched] == ["love", "love"]

    # Never infer positional identity from only the selected rows.
    changed = copy.deepcopy(reference)
    changed["data"][1]["question"] = "A different question?"
    with pytest.raises(ValueError, match="whole-source alignment"):
        fresh.attach_textvqa_training_references(rows[:1], instructions, changed)
    with pytest.raises(ValueError, match="different row counts"):
        fresh.attach_textvqa_training_references(rows, instructions[:1], reference)
