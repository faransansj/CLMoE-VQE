"""Predeclared paired comparisons, exact seed sign-flip tests and Holm correction.

Seed-bootstrap intervals describe uncertainty under resampling the observed
training seeds, not new task orders or datasets. Exact randomization assumes
exchangeability of within-seed method labels under the null. Do not equate a
positive point estimate with significance or publication novelty.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np


MATCHED_DATA_KEYS = ("protocol", "manifest_sha256", "image_size", "eval_split", "task_limit")


def paired_statistics(deltas):
    if len(deltas) < 2 or not all(math.isfinite(x) for x in deltas):
        raise ValueError("Need at least two finite paired seed deltas")
    if len(deltas) > 16:
        raise ValueError("Exact sign-flip enumeration limited to16 predeclared seeds")
    mean = statistics.mean(deltas)
    observed = abs(sum(deltas))
    permutations = itertools.product((-1, 1), repeat=len(deltas))
    count = sum(abs(sum(sign * delta for sign, delta in zip(signs, deltas))) >= observed - 1e-12
                for signs in permutations)
    rng = np.random.default_rng(20261008)
    samples = np.asarray(deltas)[rng.integers(0, len(deltas), size=(20000, len(deltas)))].mean(1)
    return {"mean": mean, "sample_sd": statistics.stdev(deltas), "per_seed": deltas,
            "seed_bootstrap_95_percentile": np.quantile(samples, [0.025, 0.975]).tolist(),
            "exact_two_sided_sign_flip_p": count / (2 ** len(deltas)),
            "positive_seed_count": sum(value > 0 for value in deltas),
            "negative_seed_count": sum(value < 0 for value in deltas)}


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted, previous = [None] * len(pvalues), 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def pick(runs, pattern, seeds):
    selected = [run for run in runs if all(run["config"].get(key) == value for key, value in pattern.items())]
    if len(selected) != len(seeds) or {run["config"]["seed"] for run in selected} != set(seeds):
        raise ValueError(f"Missing, duplicate or unexpected seeds for {pattern}")
    return {run["config"]["seed"]: run for run in selected}


def compare(runs, specifications, seeds):
    if len(set(seeds)) != len(seeds) or any(run["config"]["eval_split"] != "eval" or not run["full_protocol"] for run in runs):
        raise ValueError("Confirmation comparison needs unique declared seeds and full official evaluation runs")
    comparisons = []
    for specification in specifications:
        candidate = pick(runs, specification["candidate"], seeds)
        baseline = pick(runs, specification["baseline"], seeds)
        for seed in seeds:
            a, b = candidate[seed]["config"], baseline[seed]["config"]
            if any(a[key] != b[key] for key in MATCHED_DATA_KEYS):
                raise ValueError("Paired comparison changes data/protocol/preprocessing")
            if a["source_sha256"] != b["source_sha256"] or a["assets_sources"] != b["assets_sources"]:
                raise ValueError("Paired comparison changes implementation or initialization assets")
        metrics = {metric: paired_statistics([candidate[seed]["metrics"][metric] - baseline[seed]["metrics"][metric]
                                             for seed in seeds]) for metric in ("AP", "AF", "BWT", "MAP")}
        comparisons.append({"label": specification["label"], "primary": specification["primary"],
            "specification": specification, "seeds": seeds, "delta_candidate_minus_baseline": metrics,
            "candidate_mean": {metric: statistics.mean(candidate[seed]["metrics"][metric] for seed in seeds) for metric in metrics},
            "baseline_mean": {metric: statistics.mean(baseline[seed]["metrics"][metric] for seed in seeds) for metric in metrics},
            "candidate_post_first_plasticity_pp": [statistics.mean(candidate[seed]["plasticity_pp"][1:]) for seed in seeds],
            "candidate_expert_counts": [candidate[seed]["expert_count"] for seed in seeds],
            "candidate_largest_answer_fractions": [candidate[seed]["largest_answer_fraction"] for seed in seeds],
            "training_examples_equal_each_seed": [candidate[seed]["training_examples"] == baseline[seed]["training_examples"] for seed in seeds],
            "optimizer_steps_equal_each_seed": [candidate[seed]["optimizer_steps"] == baseline[seed]["optimizer_steps"] for seed in seeds]})
    primary = [row for row in comparisons if row["primary"]]
    adjusted = holm([row["delta_candidate_minus_baseline"]["AP"]["exact_two_sided_sign_flip_p"] for row in primary])
    for row, pvalue in zip(primary, adjusted):
        row["AP_holm_adjusted_p"] = pvalue
        row["positive_AP_significant_at_0_05"] = row["delta_candidate_minus_baseline"]["AP"]["mean"] > 0 and pvalue < 0.05
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--specifications", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("configs/readout_research.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    audit = json.loads(args.audit.read_text())
    specifications = json.loads(args.specifications.read_text())
    result = {"comparisons": compare(audit["runs"], specifications, plan["confirmation"]["seeds"]),
              "interpretation": "Delta=candidate minus baseline. Higher AP/lower AF desirable. Primary AP sign-flip tests Holm-corrected across the entire declared primary family; other endpoints/ablations exploratory. Seed-bootstrap intervals do not cover unseen task orders or new test samples. Inspect raw task scores, learning gains, answer distributions and resource costs before a joint improvement claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
