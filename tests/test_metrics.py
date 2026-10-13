import pytest

from sema_vqa.metrics import continual_metrics, score_prediction


@pytest.mark.parametrize("metric", ["clmoe_exact", "vqav2_exact"])
def test_exact_only_normalizes_case(metric):
    row = {"answer": "Two cats"}
    assert score_prediction("tWO CATS", row, metric) == 1.0
    for prediction in ["two cats.", " two cats", "two cats ", "2 cats", "the two cats"]:
        assert score_prediction(prediction, row, metric) == 0.0


def test_gqa_uses_literal_official_reference_and_trailing_period_conversion():
    assert score_prediction("NO...", {"answer": "no"}, "gqa") == 1.0
    for prediction in ["No. ", " no", "the no", "No!"]:
        assert score_prediction(prediction, {"answer": "no"}, "gqa") == 0.0
    assert score_prediction("NO", {"answer": "No"}, "gqa") == 0.0


@pytest.mark.parametrize("count, expected", [(0, 0.0), (1, 0.3), (2, 0.6),
                                             (3, 0.9), (4, 1.0), (10, 1.0)])
def test_textvqa_leave_one_annotator_out_consensus(count, expected):
    row = {"answers": ["two"] * count + ["other"] * (10 - count)}
    assert score_prediction("The TWO!", row, "textvqa") == pytest.approx(expected)


@pytest.mark.parametrize("prediction, answer", [
    ("The TWO!", "2"),
    ("A cat\n", "cat"),
    ("dont", "don't"),
    ("1,000", "1000"),
    ("3.14", "3.14"),
])
def test_textvqa_official_evalai_normalization(prediction, answer):
    assert score_prediction(prediction, {"answers": [answer] * 10}, "textvqa") == 1.0


def test_textvqa_preserves_decimal_point():
    assert score_prediction("3.14", {"answers": ["314"] * 10}, "textvqa") == 0.0


@pytest.mark.parametrize("answers", [["cat"], ["cat"] * 9, ["cat"] * 11,
                                     "0123456789", [None] * 10])
def test_textvqa_rejects_missing_consensus_metadata(answers):
    with pytest.raises(ValueError, match="ten official"):
        score_prediction("cat", {"answers": answers}, "textvqa")


@pytest.mark.parametrize("prediction, expected", [
    ("B", 1.0), ("b", 1.0), ("B. because", 1.0),
    ("The answer is (B).", 1.0), ("option B", 1.0),
    ("I choose B", 0.0),  # Official regex first sees I, not B.
    ("A or B", 0.0), ("2 then B", 0.0),
    ("E", 0.0), ("FAILED", 0.0), ("", 0.0), ("blue", 0.0),
])
def test_scienceqa_literal_source_parser_and_invalid_options(prediction, expected):
    row = {"choices": ["red", "blue"], "answer_index": 1}
    assert score_prediction(prediction, row, "scienceqa") == expected


@pytest.mark.parametrize("target", [-1, 2, True, "B"])
def test_scienceqa_rejects_invalid_ground_truth(target):
    with pytest.raises(ValueError, match="valid answer_index"):
        score_prediction("FAILED", {"choices": ["red", "blue"], "answer_index": target}, "scienceqa")


def test_unknown_metric_is_not_silently_exact_match():
    with pytest.raises(ValueError, match="Unknown metric"):
        score_prediction("yes", {"answer": "yes"}, "not_a_metric")


def test_continual_metrics_percentages_and_stage_weighted_map():
    result = continual_metrics([[0.8], [0.6, 0.9], [0.5, 0.7, 0.6]])
    assert result == pytest.approx({"AP": 60, "AF": 25, "BWT": -25,
                                    "MAP": 100 * (0.8 + 0.75 + 0.6) / 3})


def test_negative_af_is_improvement_not_clipped_or_max_forgetting():
    # Task 0 improves to .9 then drops to .6: max-forgetting would be .3,
    # but signed diagonal-final is -.2 and must remain negative.
    result = continual_metrics([[0.4], [0.9, 0.5], [0.6, 0.7, 0.8]])
    assert result["AF"] == pytest.approx(-20)
    assert result["BWT"] == pytest.approx(20)
    assert result["AP"] == pytest.approx(70)


def test_square_none_padding_matches_ragged_triangle():
    assert continual_metrics([[0.4, None], [0.5, 0.7]]) == continual_metrics([[0.4], [0.5, 0.7]])


def test_single_task_convention():
    assert continual_metrics([[0.25]]) == {"AP": 25, "AF": 0, "BWT": 0, "MAP": 25}


@pytest.mark.parametrize("matrix", [
    [], [[]], [[0.4], [0.5]], [[0.4], [0.5, None]],
    [[0.4, 0.0], [0.5, 0.7]], [[0.4, None], [0.5]],
    [[float("nan")]], [[float("inf")]], [[-0.1]], [[1.01]],
    [[70.0]], [[True]], [["0.5"]],
])
def test_continual_metrics_requires_complete_fractional_lower_triangle(matrix):
    with pytest.raises(ValueError):
        continual_metrics(matrix)
