import json

import pytest

from scripts.summarize_readout import rescore, summarize


def test_rescore_requires_complete_unique_predictions(tmp_path):
    path = tmp_path / "predictions.jsonl"
    rows = [{"id": "one", "answer": "yes"}, {"id": "two", "answer": "no"}]
    path.write_text(json.dumps({"id": "one", "prediction": "YES", "expert": 0, "score": 1}) + "\n" +
                    json.dumps({"id": "two", "prediction": "yes", "expert": 1, "score": 0}) + "\n")
    score, _ = rescore(path, rows, "clmoe_exact")
    assert score == 0.5
    with path.open("a") as stream:
        stream.write(json.dumps({"id": "one", "prediction": "YES", "expert": 0, "score": 1}) + "\n")
    with pytest.raises(ValueError, match="Missing/duplicate"):
        rescore(path, rows, "clmoe_exact")


def test_rescore_rejects_inflated_stored_score(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"id": "one", "prediction": "no", "score": 1}) + "\n")
    with pytest.raises(ValueError, match="score mismatch"):
        rescore(path, [{"id": "one", "answer": "yes"}], "clmoe_exact")


def test_rescore_preserves_uniform_consensus_accuracy_without_accumulation_drift(tmp_path):
    rows = [{"id": str(index), "question": f"What is marker {index}?",
             "answers": ["yes"] + ["no"] * 9} for index in range(100)]
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps({"id": row["id"], "prediction": "yes", "score": 0.3}) + "\n"
                            for row in rows))
    score, _ = rescore(path, rows, "textvqa")
    # Every example has exactly the same official consensus credit. Naive +=
    # drift caused valid saved continual-learning matrices to be rejected.
    assert score == 0.3


def run(seed, split="validation", task_limit=0, revision="one"):
    return {"config": {"protocol": "p", "eval_split": split, "method": "expand", "lr": 0.0003,
                       "epochs": 5, "rank": 8, "router_modality": "joint", "growth_margin": 0.5,
                       "alpha": 16, "router_memory": 128, "router_ridge": 1,
                       "router_text": "raw",
                       "replay_per_task": 64, "probe_size": 128, "image_size": 224,
                       "batch_size": 16, "micro_batch_size": 8,
                       "manifest_sha256": "same", "task_limit": task_limit, "seed": seed,
                       "source_sha256": {"model": revision}},
            "metrics": {"AP": 20 + seed, "AF": 2, "BWT": -2, "MAP": 30},
            "mean_plasticity_pp": 12, "expert_count": 4, "largest_answer_fraction": 0.3,
            "oracle_metrics": {"AP": 30}}


def test_summary_separates_pilot_validation_and_confirmation():
    values = summarize([run(1), run(2), run(1, split="eval"), run(1, task_limit=3)])
    assert len(values) == 3
    full_validation = next(value for value in values if value["eval_split"] == "validation" and not value["task_limit"])
    assert full_validation["AP"] == 21.5
    assert full_validation["AP_sd"] == pytest.approx(2**-0.5)
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize([run(1), run(1)])
    with pytest.raises(ValueError, match="Mixed source"):
        summarize([run(1), run(2, revision="changed")])


def test_summary_does_not_merge_different_router_memory_budgets():
    small, large = run(1), run(2)
    large["config"]["router_memory"] = 512
    values = summarize([small, large])
    assert {value["router_memory"]: value["AP"] for value in values} == {128: 21, 512: 22}
