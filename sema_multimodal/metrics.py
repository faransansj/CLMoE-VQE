"""Source-locked VQA/GQA and train-IDF COCO caption corpus metrics.

VQA follows GT-Vision-Lab/VQA's PythonEvaluationTools/vqaEvaluation/vqaEval.py
(including its conditional normalization). GQA follows the archived official
comparison plus SMoLoRA/scripts/convert_gqa_for_eval.py's prediction adapter.
CIDEr-D is pycocoevalcap 1.2's clipped, Gaussian-penalized implementation,
with the sole protocol change that document frequency is fixed on train images.
Accuracy is in percentage points; CIDEr-D and BLEU remain native scorer units.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_ROOT = Path(__file__).resolve().parents[1]
_VQA_PROCESSOR = _ROOT / "third_party/SMoLoRA/llava/eval/m4c_evaluator.py"
_GQA_ADAPTER = _ROOT / "third_party/SMoLoRA/scripts/convert_gqa_for_eval.py"
_GQA_EVALUATOR = _ROOT / "third_party/SMoLoRA/llava/eval/SMoLoRA/Eval/eval_gqa.py"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _vqa_processor():
    # Reuse the inspected official tables and methods, NOT TextVQA's extra
    # word_tokenize preprocessing or its unconditional normalization.
    from sema_vqa.metrics import _textvqa_evaluator
    processor = copy.copy(_textvqa_evaluator().answer_processor)
    processor.NUMBER_MAP = dict(processor.NUMBER_MAP)
    # VQAEval's comma regex is intentionally kept literally.
    processor.COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")
    return processor


def vqav2_accuracy(prediction: str, references: Sequence[str]) -> float:
    """Official leave-one-annotator-out consensus, without intermediate rounding."""
    if (not isinstance(references, (list, tuple)) or len(references) != 10
            or not all(isinstance(ref, str) for ref in references)):
        raise ValueError("VQAv2 requires exactly ten official answer strings")
    clean = lambda text: text.replace("\n", " ").replace("\t", " ").strip()
    answers, prediction = [clean(ref) for ref in references], clean(prediction)
    distinct = set(answers)
    if len(distinct) > 1:
        processor = _vqa_processor()
        normalize = lambda text: processor.process_digit_article(processor.process_punctuation(text))
        normalized = {answer: normalize(answer) for answer in distinct}
        answers = [normalized[answer] for answer in answers]
        prediction = normalized[prediction] if prediction in normalized else normalize(prediction)
    matches = sum(ref == prediction for ref in answers)
    return math.fsum(min(1.0, (matches - int(ref == prediction)) / 3.0)
                     for ref in answers) / 10.0


def gqa_accuracy(prediction: str, reference: str) -> float:
    """Official exact gold comparison after the archived GQA output adapter."""
    return float(prediction.rstrip(".").lower() == reference)


@lru_cache(maxsize=1)
def _caption_dependencies():
    if importlib.metadata.version("pycocoevalcap") != "1.2":
        raise RuntimeError("Caption scoring requires pycocoevalcap==1.2")
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.tokenizer import ptbtokenizer
    jar = Path(ptbtokenizer.__file__).parent / ptbtokenizer.STANFORD_CORENLP_3_4_1_JAR
    if not jar.is_file():
        raise RuntimeError(f"Missing Stanford CoreNLP 3.4.1 tokenizer: {jar}")
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("COCO PTB tokenization requires Java on PATH")
    return CiderScorer, Bleu, ptbtokenizer, jar, java


def _tokenize(captions: Mapping) -> dict:
    """Run the package's real Stanford PTB tokenizer, failing on Java errors.

    The package wrapper silently accepts failed subprocesses/truncated outputs;
    use the identical command and punctuation list with checked I/O instead.
    """
    _, _, tokenizer, jar, java = _caption_dependencies()
    ids = [key for key, values in captions.items() for _ in values]
    sentences = [text.replace("\n", " ").replace("\r", " ")
                 for values in captions.values() for text in values]
    if not sentences:
        raise ValueError("A nonempty caption corpus is required")
    with tempfile.TemporaryDirectory(prefix="multimodal-ptb-") as directory:
        path = Path(directory) / "captions.txt"
        path.write_text("\n".join(sentences), encoding="utf-8")
        result = subprocess.run([java, "-cp", str(jar),
            "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase", str(path)],
            check=True, capture_output=True, text=True, encoding="utf-8")
    lines = result.stdout.split("\n")
    if len(lines) == len(ids) + 1 and lines[-1] == "":
        lines.pop()
    if len(lines) != len(ids):
        raise ValueError(f"PTB tokenizer returned {len(lines)} lines for {len(ids)} captions")
    output = {key: [] for key in captions}
    for key, line in zip(ids, lines):
        output[key].append(" ".join(word for word in line.rstrip().split(" ")
                                    if word not in tokenizer.PUNCTUATIONS))
    return output


def _fixed_cider_score(scorer, document_frequency, train_documents):
    """pycocoevalcap 1.2 compute_cider with fixed DF/log(train image count).

    Derived from salaniz/pycocoevalcap/cider/cider_scorer.py (Tsung-Yi Lin,
    Ramakrishna Vedantam). Preserve TF, bigram length, clipping, sigma=6,
    four-order mean, reference mean and the native factor of ten exactly.
    No evaluation reference contributes to the IDF corpus.
    """
    import numpy as np
    ref_len = np.log(float(train_documents))

    def counts2vec(counts):
        vectors = [defaultdict(float) for _ in range(scorer.n)]
        norms, length = [0.0] * scorer.n, 0
        for ngram, frequency in counts.items():
            order = len(ngram) - 1
            weight = float(frequency) * (ref_len - np.log(max(1.0, document_frequency.get(ngram, 0.0))))
            vectors[order][ngram] = weight
            norms[order] += weight ** 2
            if order == 1:
                length += frequency
        return vectors, [np.sqrt(norm) for norm in norms], length

    def similarity(hyp, ref):
        hyp_vec, hyp_norm, hyp_len = hyp
        ref_vec, ref_norm, ref_length = ref
        values = np.zeros(scorer.n)
        penalty = np.e ** (-float(hyp_len - ref_length) ** 2 / (2 * scorer.sigma ** 2))
        for order in range(scorer.n):
            for ngram, weight in hyp_vec[order].items():
                values[order] += min(weight, ref_vec[order][ngram]) * ref_vec[order][ngram]
            if hyp_norm[order] != 0 and ref_norm[order] != 0:
                values[order] /= hyp_norm[order] * ref_norm[order]
            values[order] *= penalty
        return values

    scores = []
    for prediction, references in zip(scorer.ctest, scorer.crefs):
        hyp = counts2vec(prediction)
        values = np.zeros(scorer.n)
        for reference in references:
            values += similarity(hyp, counts2vec(reference))
        scores.append(float(np.mean(values) / len(references) * 10.0))
    return float(np.mean(scores))


class TaskScorer:
    """Build once from a full task; score ordered predictions on any row corpus.

    ``lock`` pins metric implementation, dependencies, tokenizer and train-IDF.
    Bootstrap may repeat rows: positional scoring retains repeated image draws.
    Unique prediction IDs are checked by the artifact auditor, not this method.
    """

    def __init__(self, task: Mapping):
        self.metric = task["metric"]
        self.family = task["family"]
        if self.metric not in {"vqav2", "gqa", "cider_d"}:
            raise ValueError(f"Unsupported multimodal metric: {self.metric}")
        expected_family = "caption" if self.metric == "cider_d" else "vqa"
        if self.family != expected_family:
            raise ValueError("Task family and metric disagree")
        self.lock = {"metric": self.metric, "implementation_sha256": _file_hash(__file__),
                     "units": "native_cider_d" if self.metric == "cider_d" else "accuracy_points"}
        if self.metric == "vqav2":
            self.lock.update({"normalizer_sha256": _file_hash(_VQA_PROCESSOR),
                              "consensus": "official_leave_one_out_conditional_normalization"})
        elif self.metric == "gqa":
            self.lock.update({"adapter_sha256": _file_hash(_GQA_ADAPTER),
                              "official_evaluator_sha256": _file_hash(_GQA_EVALUATOR)})
        else:
            from inspect import getfile
            CiderScorer, Bleu, tokenizer, jar, _ = _caption_dependencies()
            from pycocoevalcap.bleu.bleu_scorer import BleuScorer
            train = task["train"]
            if len(train) < 2 or len({row["id"] for row in train}) != len(train):
                raise ValueError("Fixed CIDEr IDF requires at least two unique train images")
            if len({row["image_sha256"] for row in train}) != len(train):
                raise ValueError("CIDEr train corpus must contain unique image hashes")
            self._train_images = {row["id"]: row["image_sha256"] for row in train}
            self._train_hashes = set(self._train_images.values())
            references = {row["id"]: self._references(row) for row in train}
            tokenized = _tokenize(references)
            corpus = CiderScorer(n=4, sigma=6.0)
            for row in train:
                corpus += (None, tokenized[row["id"]])
            corpus.compute_doc_freq()
            self._df = dict(corpus.document_frequency)
            self._train_documents = len(train)
            self.lock.update({"pycocoevalcap": "1.2", "numpy": importlib.metadata.version("numpy"),
                "tokenizer": "Stanford_CoreNLP_3.4.1_PTB_preserveLines_lowerCase",
                "tokenizer_jar_sha256": _file_hash(jar), "tokenizer_wrapper_sha256": _file_hash(tokenizer.__file__),
                "cider_source_sha256": _file_hash(getfile(CiderScorer)),
                "bleu_source_sha256": _file_hash(getfile(Bleu)),
                "bleu_scorer_sha256": _file_hash(getfile(BleuScorer)),
                "idf_source": "task_train_references", "train_documents": len(train),
                "train_reference_sha256": _digest(sorted((row["id"], row["image_sha256"], row["references"]) for row in train)),
                "document_frequency_sha256": _digest(sorted((list(ngram), count) for ngram, count in self._df.items())),
                "n": 4, "sigma": 6.0, "scale": 10.0})

    @staticmethod
    def _references(row):
        references = row["references"]
        if (not isinstance(references, (list, tuple)) or not references
                or not all(isinstance(ref, str) and ref.strip() for ref in references)):
            raise ValueError("Scoring requires nonempty official reference strings")
        return references

    def score(self, predictions: Sequence[str], rows: Sequence[Mapping]) -> dict[str, float]:
        if (not rows or len(predictions) != len(rows)
                or not all(isinstance(prediction, str) for prediction in predictions)):
            raise ValueError("One string prediction is required per nonempty row corpus")
        if any(row["family"] != self.family for row in rows):
            raise ValueError("Scoring rows belong to the wrong task family")
        if self.metric == "vqav2":
            accuracy = 100.0 * math.fsum(vqav2_accuracy(pred, self._references(row))
                for pred, row in zip(predictions, rows)) / len(rows)
            return {"score": accuracy, "accuracy": accuracy}
        if self.metric == "gqa":
            references = [self._references(row) for row in rows]
            if any(len(refs) != 1 for refs in references):
                raise ValueError("GQA requires exactly one official answer")
            accuracy = 100.0 * math.fsum(gqa_accuracy(pred, refs[0])
                for pred, refs in zip(predictions, references)) / len(rows)
            return {"score": accuracy, "accuracy": accuracy}
        # Never allow references from a held-out row to be mistaken for train IDF.
        # Training-corpus scoring itself is allowed, but mixed identity is not.
        for row in rows:
            if row["id"] in self._train_images:
                if self._train_images[row["id"]] != row["image_sha256"]:
                    raise ValueError("Caption row ID/image hash disagrees with train corpus identity")
            elif row["image_sha256"] in self._train_hashes:
                raise ValueError("Caption row ID/image hash disagrees with train corpus identity")
        CiderScorer, Bleu, _, _, _ = _caption_dependencies()
        captions = {("reference", index): self._references(row) for index, row in enumerate(rows)}
        captions.update({("prediction", index): [pred] for index, pred in enumerate(predictions)})
        tokenized = _tokenize(captions)
        references = {index: tokenized[("reference", index)] for index in range(len(rows))}
        hypotheses = {index: tokenized[("prediction", index)] for index in range(len(rows))}
        corpus = CiderScorer(n=4, sigma=6.0)
        for index in references:
            corpus += (hypotheses[index][0], references[index])
        cider = _fixed_cider_score(corpus, self._df, self._train_documents)
        bleu, _ = Bleu(4).compute_score(references, hypotheses, verbose=0)
        return {"score": cider, "cider_d": cider, "bleu4": float(bleu[3])}


def continual_metrics(matrix, tasks, *, pre_task_scores=None, normalizers=None):
    """Task/family summaries; never average unlike native units.

    Normalizers are explicit positive G1 validation constants keyed by task_id.
    Unnormalized family AF is meaningful only within the same scorer units.
    Singleton/empty previous-task AF is None, never a fabricated zero.
    """
    n = len(tasks)
    if not n or len(matrix) != n:
        raise ValueError("A complete task matrix is required")
    ids = [task["task_id"] for task in tasks]
    if len(set(ids)) != n:
        raise ValueError("Task IDs must be unique")
    if not (all(len(row) == i + 1 for i, row in enumerate(matrix))
            or all(len(row) == n for row in matrix)):
        raise ValueError("Matrix must be lower triangular or square")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if j > i:
                if value is not None:
                    raise ValueError("Unseen task matrix entries must be None")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Seen task scores must be finite")
    if pre_task_scores is not None and (len(pre_task_scores) != n or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in pre_task_scores)):
        raise ValueError("Pre-task scores must be complete and finite")
    if normalizers is not None:
        if set(normalizers) != set(ids) or any(isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
                for value in normalizers.values()):
            raise ValueError("Explicit positive fixed normalizers are required for every task")
    task_reports = {}
    for j, task in enumerate(tasks):
        final, diagonal = matrix[-1][j], matrix[j][j]
        report = {"family": task["family"], "metric": task["metric"],
                  "units": "accuracy_points" if task["metric"] in {"vqav2", "gqa"} else "native_cider_d",
                  "final": final, "post_train": diagonal,
                  "AF": diagonal - final if j < n - 1 else None,
                  "pre_to_post_gain": diagonal - pre_task_scores[j] if pre_task_scores is not None else None}
        if normalizers is not None:
            denominator = normalizers[ids[j]]
            report.update({"normalizer": denominator, "normalized_final": final / denominator,
                "normalized_AF": report["AF"] / denominator if report["AF"] is not None else None,
                "normalized_gain": report["pre_to_post_gain"] / denominator if report["pre_to_post_gain"] is not None else None})
        task_reports[ids[j]] = report
    families = {}
    for family in sorted({task["family"] for task in tasks}):
        reports = [report for report in task_reports.values() if report["family"] == family]
        previous = [report for report in reports if report["AF"] is not None]
        subsequent = [task_reports[ids[j]] for j in range(1, n)
                      if tasks[j]["family"] == family and pre_task_scores is not None]
        same_units = len({report["units"] for report in reports}) == 1
        family_report = {"tasks": [key for key, report in task_reports.items() if report["family"] == family],
            "units": reports[0]["units"] if same_units else None,
            "native_AF": math.fsum(report["AF"] for report in previous) / len(previous) if previous and same_units else None,
            "native_subsequent_gain": math.fsum(report["pre_to_post_gain"] for report in subsequent) / len(subsequent) if subsequent and same_units else None}
        if normalizers is not None:
            family_report.update({"normalized_final": math.fsum(report["normalized_final"] for report in reports) / len(reports),
                "normalized_AF": math.fsum(report["normalized_AF"] for report in previous) / len(previous) if previous else None,
                "normalized_subsequent_gain": math.fsum(report["normalized_gain"] for report in subsequent) / len(subsequent) if subsequent else None})
        families[family] = family_report
    result = {"tasks": task_reports, "families": families}
    if normalizers is not None:
        result["family_balanced_normalized_final"] = math.fsum(
            report["normalized_final"] for report in families.values()) / len(families)
        result["normalized_matrix"] = [[value / normalizers[ids[j]] if value is not None else None
                                       for j, value in enumerate(row)] for row in matrix]
    return result
