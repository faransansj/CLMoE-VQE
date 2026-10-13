import copy

import pytest

from scripts.prepare_vision_splits import split_manifest
from scripts.enrich_vision_references import attach_references


def test_global_image_groups_preserve_reference_data_and_eval():
    rows = [{"id": str(i), "image": f"image{i}", "question": "q", "answer": "a", "answers": ["a"] * 10} for i in range(100)]
    duplicate_rows = [dict(row, id="second-" + row["id"], image="alias" + row["id"]) for row in rows]
    hashes = {f"image{i}": f"hash{i}" for i in range(100)}
    hashes.update({f"alias{i}": f"hash{i}" for i in range(100)})
    hashes["heldout"] = "hash0"
    evaluation = [{"id": "eval", "image": "heldout", "question": "different", "answer": "b"}]
    source = {"protocol": "example", "tasks": [{"name": "one", "metric": "textvqa", "train": rows, "eval": evaluation},
                                              {"name": "two", "metric": "textvqa", "train": duplicate_rows, "eval": evaluation}]}
    original = copy.deepcopy(source)
    result = split_manifest(source, hashes)
    assert source == original
    assert result == split_manifest(source, hashes)
    assert result["vision_partition"]["excluded"] == [
        {"task": "one", "id": "0", "image": "image0", "reason": "official_eval_image_or_id_overlap"},
        {"task": "two", "id": "second-0", "image": "alias0", "reason": "official_eval_image_or_id_overlap"}]
    train = {hashes[r["image"]] for t in result["tasks"] for r in t["train"]}
    validation = {hashes[r["image"]] for t in result["tasks"] for r in t["validation"]}
    assert not train & validation
    assert train | validation == {f"hash{i}" for i in range(1, 100)}
    assert "hash0" not in train | validation
    for task in result["tasks"]:
        assert task["eval"] == evaluation
        for row in task["train"] + task["validation"]:
            assert row["answers"] == ["a"] * 10


def reference_fixture(metric):
    rows = [{"id": str(i), "image": f"image{i}", "question": "Question?", "answer": "Yes"} for i in range(100)]
    hashes = {f"image{i}": f"hash{i}" for i in range(100)}
    hashes["eval"] = "heldout"
    source = {"tasks": [{"name": "task", "metric": metric, "train": rows,
                         "eval": [{"id": "eval", "image": "eval", "answer": "yes"}]}]}
    return source, hashes


def test_missing_consensus_fails_before_training():
    source, hashes = reference_fixture("textvqa")
    with pytest.raises(ValueError, match="ten original TextVQA"):
        split_manifest(source, hashes)


def test_gqa_validation_uses_original_reference_without_relabeling_fit():
    source, hashes = reference_fixture("gqa")
    for row in source["tasks"][0]["train"]:
        row["reference_answer"] = "yes"
    result = split_manifest(source, hashes)
    assert all(r["answer"] == "Yes" for r in result["tasks"][0]["train"])
    assert all(r["answer"] == "yes" and r["instruction_answer"] == "Yes" for r in result["tasks"][0]["validation"])


def test_original_question_and_label_resolve_gqa_annotation_ambiguity():
    source = {"tasks": [{"name": "GQA", "train": [
        {"id": "row", "source_id": "image", "question": "<image>\nDevice?.", "answer": "Computer"}
    ]}]}
    references = {
        "correct": {"imageId": "image", "question": "Device?", "answer": "computer"},
        "different": {"imageId": "image", "question": "Device?", "answer": "keyboard"},
    }
    result = attach_references(source, {"data": []}, references)
    row = result["tasks"][0]["train"][0]
    assert row["answer"] == "Computer"
    assert row["reference_answer"] == "computer"
    assert row["reference_question_ids"] == ["correct"]
    source["tasks"][0]["train"][0]["answer"] = "Unknown device"
    with pytest.raises(ValueError, match="Ambiguous GQA"):
        attach_references(source, {"data": []}, references)


def test_textvqa_keeps_all_original_consensus_answers():
    original_answers = ["stop"] * 6 + ["red"] * 3 + ["sign"]
    source = {"tasks": [{"name": "TextVQA", "train": [
        {"id": "row", "source_id": "image", "question": "<image>\nSign?\nReference OCR token:STOP", "answer": "stop"}
    ]}]}
    reference = {"data": [{"image_id": "image", "question_id": 42, "question": "sign?",
                            "answers": original_answers}]}
    row = attach_references(source, reference, {})["tasks"][0]["train"][0]
    assert row["answers"] == original_answers
    assert row["reference_question_ids"] == ["42"]
