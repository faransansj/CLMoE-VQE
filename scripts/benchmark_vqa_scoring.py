"""Offline scoring throughput proxy; not a model-quality experiment.

The workload and oracle are benchmark policy: freeze both during optimization.
Only production scorer execution is timed, including its input validation.
"""

import copy
import math
import random
import statistics
import time

from sema_multimodal.metrics import TaskScorer, vqav2_accuracy
from sema_vqa.metrics import _textvqa_evaluator


def official_score(prediction, references):
    """Literal leave-one-annotator-out oracle with the archived normalizer."""
    clean = lambda text: text.replace("\n", " ").replace("\t", " ").strip()
    answers = [clean(ref) for ref in references]
    prediction = clean(prediction)
    if len(set(answers)) > 1:
        processor = copy.copy(_textvqa_evaluator().answer_processor)
        processor.NUMBER_MAP = dict(processor.NUMBER_MAP)
        import re
        processor.COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
        normalize = lambda text: processor.process_digit_article(processor.process_punctuation(text))
        answers = [normalize(ref) for ref in answers]
        prediction = normalize(prediction)
    scores = []
    for index in range(10):
        matches = sum(answer == prediction for other, answer in enumerate(answers) if other != index)
        scores.append(min(1.0, matches / 3.0))
    return math.fsum(scores) / 10.0


def workload():
    # Consensus boundaries, unanimous conditional-normalization behavior,
    # number/article/punctuation/contraction normalization and wrong answers.
    cases = [("yes", ["yes"] * matches + ["no"] * (10 - matches))
             for matches in range(11)]
    cases += [
        ("\nThe two!\t", ["2"] * 4 + ["three"] * 6),
        ("1,000", ["1000"] * 4 + ["2000"] * 6),
        ("cant", ["can't"] * 4 + ["yes"] * 6),
        ("The two!", ["2"] * 10),
        ("\t2\n", ["2"] * 10),
        ("red.", ["red"] * 3 + ["blue"] * 7),
        ("", ["no"] * 10),
        ("a cat", ["cat"] * 2 + ["dog"] * 8),
        ("ten", ["10"] + ["nine"] * 9),
    ]
    rng = random.Random(20261023)
    vocabulary = ["yes", "no", "The two!", "2", "three", "1,000", "1000", "cant", "can't", "red.", "red"]
    cases += [(rng.choice(vocabulary), [rng.choice(vocabulary) for _ in range(10)])
              for _ in range(200)]
    expected = [official_score(prediction, refs) for prediction, refs in cases]
    # Check the oracle itself against manually known consensus boundaries.
    assert expected[:11] == [0.0, 0.3, 0.6, 0.9] + [1.0] * 7
    assert expected[11:20] == [1.0, 1.0, 1.0, 0.0, 1.0, 0.9, 0.0, 0.6, 0.3]
    for (prediction, refs), target in zip(cases, expected):
        actual = vqav2_accuracy(prediction, refs)
        if not math.isclose(actual, target, rel_tol=0, abs_tol=1e-12):
            raise AssertionError((prediction, refs, actual, target))
    rows = [{"id": str(index), "family": "vqa", "references": list(refs)}
            for index, (_, refs) in enumerate(cases * 16)]
    predictions = [prediction for prediction, _ in cases * 16]
    return rows, predictions, 100.0 * math.fsum(expected * 16) / len(rows)


def main():
    rows, predictions, expected = workload()
    scorer = TaskScorer({"family": "vqa", "metric": "vqav2"})
    original = copy.deepcopy(rows)

    def run():
        start = time.perf_counter_ns()
        result = scorer.score(predictions, rows)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if set(result) != {"score", "accuracy"} or any(
                not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
                for value in result.values()):
            raise AssertionError((result, expected))
        if rows != original:
            raise AssertionError("Scoring mutated the reference corpus")
        return elapsed

    run()  # Untimed warmup; exactly seven measured full-corpus passes follow.
    elapsed = statistics.median(run() for _ in range(7))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise AssertionError("Invalid elapsed time")
    print(f"METRIC scoring_ms={elapsed:.6f}")
    print(f"METRIC predictions_per_second={len(rows) * 1000 / elapsed:.6f}")
    print(f"METRIC accuracy_points={expected:.12f}")
    print(f"METRIC predictions={len(rows)}")


if __name__ == "__main__":
    main()
