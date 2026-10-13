"""Lock readout settings from complete train-derived validation runs only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


BASELINES = ("shared_lora", "replay_lora", "full_readout")


def rank_key(run):
    config = run["config"]
    return (-run["metrics"]["AP"], run["metrics"]["AF"], config["epochs"], config["lr"])


def gains(run):
    return statistics.mean(run["plasticity_pp"][1:] or run["plasticity_pp"])


def candidate_record(run, baseline):
    return {"source_run": run["directory"], "config": run["config"], "metrics": run["metrics"],
            "post_first_task_plasticity_pp": gains(run), "expert_count": run["expert_count"],
            "largest_answer_fraction": run["largest_answer_fraction"],
            "validation_joint_success": (run["metrics"]["AP"] > baseline["metrics"]["AP"]
                                         and run["metrics"]["AF"] <= baseline["metrics"]["AF"]
                                         and gains(run) > 0 and run["largest_answer_fraction"] < 0.9)}


def select(runs, declared_jobs):
    if not runs or any(run["config"]["eval_split"] != "validation" or not run["full_protocol"]
                       or run["config"]["task_limit"] != 0 for run in runs):
        raise ValueError("Selection requires complete full-protocol validation runs; no pilot or official eval")
    if any(job.get("eval_split") != "validation" or job.get("task_limit", 0) for job in declared_jobs):
        raise ValueError("Declared selection jobs must be full validation")
    if len(runs) != len(declared_jobs):
        raise ValueError("Incomplete or extra declared development grid")
    matched = []
    for job in declared_jobs:
        candidates = [index for index, run in enumerate(runs)
                      if all(run["config"].get(key) == value for key, value in job.items())]
        if len(candidates) != 1 or candidates[0] in matched:
            raise ValueError("Declared development condition missing, duplicated or ambiguous")
        matched.append(candidates[0])
    selections = {}
    for protocol in sorted({run["config"]["protocol"] for run in runs}):
        values = [run for run in runs if run["config"]["protocol"] == protocol]
        if len({run["config"]["manifest_sha256"] for run in values}) != 1:
            raise ValueError("Cannot select across changed data")
        if len({json.dumps(run["config"]["source_sha256"], sort_keys=True) for run in values}) != 1:
            raise ValueError("Cannot select across changed implementation")
        baseline_choices = {}
        for method in BASELINES:
            candidates = [run for run in values if run["config"]["method"] == method]
            if not candidates:
                raise ValueError(f"Missing baseline {method}")
            baseline_choices[method] = min(candidates, key=rank_key)
        strongest = min(baseline_choices.values(), key=rank_key)
        shared = baseline_choices["shared_lora"]
        expansions = [run for run in values if run["config"]["method"] == "expand"
                      and run["config"]["router_modality"] == "joint" and run["config"]["router_text"] == "raw"]
        matched_expansions = [run for run in expansions if all(run["config"][key] == shared["config"][key]
                                                              for key in ("lr", "epochs", "rank", "alpha", "batch_size"))]
        if not expansions or not matched_expansions:
            raise ValueError("Missing common-budget expansion comparison")
        eligible = [run for run in expansions if run["metrics"]["AF"] <= strongest["metrics"]["AF"]
                    and gains(run) > 0 and run["largest_answer_fraction"] < 0.9]
        independent = min(eligible or expansions, key=rank_key)
        primary = min(matched_expansions, key=rank_key)
        selections[protocol] = {
            "baselines": {name: candidate_record(run, strongest) for name, run in baseline_choices.items()},
            "strongest_baseline": strongest["config"]["method"],
            "primary_common_budget": candidate_record(primary, shared),
            "secondary_independent": candidate_record(independent, strongest),
            "selection_eval_split": "validation",
            "interpretation": "Validation selection only, not confirmation or evidence of statistical significance. No successful candidate means record failure, not suppress controls."
        }
    return selections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_raw, jobs_raw = args.audit.read_bytes(), args.jobs.read_bytes()
    result = {"selection": select(json.loads(audit_raw)["runs"], json.loads(jobs_raw)),
              "audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
              "jobs_sha256": hashlib.sha256(jobs_raw).hexdigest(), "official_eval_used": False}
    if args.output.exists():
        raise ValueError("Refusing to overwrite a locked selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
