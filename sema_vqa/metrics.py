"""Source-faithful per-example scores and continual-learning summaries."""

import importlib.util
import math
import re
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping, Sequence


@lru_cache(maxsize=1)
def _textvqa_evaluator():
    # Load the standalone official scorer, not llava's model-heavy package.
    path = (Path(__file__).resolve().parents[1] / "third_party" / "SMoLoRA"
            / "llava" / "eval" / "m4c_evaluator.py")
    spec = importlib.util.spec_from_file_location("_smolora_m4c_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official TextVQA scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


def score_prediction(prediction: str, row: Mapping, metric: str) -> float:
    """Return a source benchmark score in [0, 1], without generic cleanup.

    Exact metrics require ``answer``; GQA requires the *official* answer.
    TextVQA requires ten official ``answers``. ScienceQA requires official
    ``choices`` and integer ``answer_index``. Whitespace is not implicitly
    stripped: caller-side generation cleanup precedes source scoring.
    """
    if metric in {"clmoe_exact", "vqav2_exact"}:
        return float(prediction.upper() == row["answer"].upper())
    if metric == "gqa":
        return float(prediction.rstrip(".").lower() == row["answer"])
    if metric == "textvqa":
        answers = row["answers"]
        if (not isinstance(answers, (list, tuple)) or len(answers) != 10
                or not all(isinstance(answer, str) for answer in answers)):
            raise ValueError("TextVQA requires exactly ten official answer strings")
        evaluator = _textvqa_evaluator()
        scores = evaluator._compute_answer_scores(answers)
        return float(scores.get(evaluator.answer_processor(prediction), 0.0))
    if metric == "scienceqa":
        choices, target = row["choices"], row["answer_index"]
        options = "ABCDE"
        if (not isinstance(choices, (list, tuple)) or not 1 <= len(choices) <= 5
                or not all(isinstance(choice, str) for choice in choices)
                or type(target) is not int or not 0 <= target < len(choices)):
            raise ValueError("ScienceQA requires official choices and a valid answer_index")
        # Match eval_science_qa.py literally, including the first standalone
        # word character (not the first plausible option in an explanation).
        if prediction in list(options):
            answer = prediction
        elif (len(prediction) >= 3 and prediction[0] in options
              and prediction[1:3] == ". "):
            answer = prediction[0]
        else:
            match = re.search(r"\b(\w)\b", prediction)
            answer = match.group(1).upper() if match else "FAILED"
        index = options.index(answer) if answer in list(options[:len(choices)]) else -1
        return float(index == target)
    raise ValueError(f"Unknown metric: {metric}")


def continual_metrics(matrix: Sequence[Sequence[float | None]]) -> dict[str, float]:
    """Summarize complete task-by-task accuracies supplied in [0, 1].

    Row i is evaluation after learning task i. Accept a ragged lower triangle
    (row lengths 1..T) or a square matrix whose upper triangle is all None.
    All on/below-diagonal scores must be finite and present.

    Outputs are percentages (AF/BWT are percentage-point differences): AP is
    final-row mean; MAP is the mean of per-stage seen-task means. AF is signed
    mean(diagonal - final) over the first T-1 tasks, NOT maximum forgetting;
    BWT is its negative. For one task, AF and BWT are defined here as zero.
    """
    n = len(matrix)
    if n == 0:
        raise ValueError("A nonempty, complete lower-triangular matrix is required")
    ragged = all(len(row) == i + 1 for i, row in enumerate(matrix))
    square = all(len(row) == n for row in matrix)
    if not (ragged or square):
        raise ValueError("Matrix must be a complete lower triangle or None-padded square")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if j > i:
                if value is not None:
                    raise ValueError("Unevaluated upper-triangular entries must be None")
            elif (value is None or isinstance(value, bool)
                  or not isinstance(value, (int, float))
                  or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("All evaluated entries must be finite scores in [0, 1]")
    final = matrix[-1]
    ap = sum(final) / n
    map_score = sum(sum(row[:i + 1]) / (i + 1)
                    for i, row in enumerate(matrix)) / n
    af = sum(matrix[i][i] - final[i] for i in range(n - 1)) / (n - 1) if n > 1 else 0.0
    return {"AP": 100.0 * ap, "AF": 100.0 * af,
            "BWT": -100.0 * af if n > 1 else 0.0, "MAP": 100.0 * map_score}
