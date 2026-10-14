"""Inference-only frozen-bank routing evaluator; no expert update or backbone mutation.

For a completed ``expand`` parent run the final expert bank never changes after its
own training phase, so every earlier stage reconstructs exactly from it. The
evaluator re-fits each stage's image/question feature-modality router from exactly
the parent's TRAIN routing-memory selection (same features, same memory subsample,
same ridge solve, same per-stage expert counts) and evaluates the five locked G5
conditions at every saved stage: native_joint, image_only, question_only,
family_restricted_joint, task_id_oracle. Native predictions and the native matrix
must byte-match the parent at every stage, including the pre-task reconstruction,
before any alternative diagnostic is accepted. No future/unseen expert is ever
routed to and expert retraining steps are always zero.

Family restriction is a candidate mask on the native joint router, never a refit:
the native router's per-stage snapshots (frozen memory and exact center/weight/
prior tensors) are captured while the native stage is replayed and the family
variant routes with those byte-exact coefficients, restricted to the
already-learned experts of the public request family (fixed caption instruction
vs VQA question format). Gold task IDs are never used. Before a family's first
task is introduced there is no eligible expert: the stage is diagnosed as
base-only and unroutable instead of pretending knowledge. The task-ID oracle is a
separate upper bound that forces the gold task's own already-trained expert for
seen tasks only; it never produces pre-task diagnostics, which would require a
future expert.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

from sema_multimodal import run as runner
from sema_multimodal.data import CAPTION_INSTRUCTION, validate_manifest
from sema_multimodal.metrics import TaskScorer, continual_metrics
from sema_multimodal.model import MultimodalReadout
from sema_vqa.readout_router import FeatureRouter

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("native_joint", "image_only", "question_only", "family_restricted_joint", "task_id_oracle")
MODALITY = {"native_joint": "joint", "image_only": "image", "question_only": "question",
            "family_restricted_joint": "joint"}
GENERATION_KEYS = ("generated_tokens", "eos", "empty", "repeated_trigram", "limit_reached")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions_path(directory, kind, stage, task_id):
    return directory / f"predictions_{kind}_{stage:02d}_{task_id}.jsonl"


def _snapshot_router(router):
    """Frozen per-stage copy of the native joint router.

    Tensors are detached clones so a snapshot can never alias later ``observe``
    mutations (weight/center/prior are replaced per fit and the memory list only
    grows); the family variant routes with these byte-exact native coefficients.
    """
    import torch
    return {"memory": [(tensor.detach().clone(), expert) for tensor, expert in router.memory],
            "memory_per_task": router.memory_per_task, "ridge": router.ridge,
            "modality": router.modality, "seed": router.seed, "num_experts": router.num_experts,
            "weight": None if router.weight is None else router.weight.detach().clone(),
            "center": None if router.center is None else router.center.detach().clone(),
            "prior": None if router.prior is None else router.prior.detach().clone()}


class MaskedRouter:
    """Native joint router snapshot restricted to eligible already-learned experts.

    The family variant never refits: the snapshot carries the byte-exact native
    center/weight/prior fitted on the same TRAIN memory at the reconstructed
    stage, and predict reuses them identically with only the candidate mask
    applied. ``memory`` stays nonempty so the inherited routed-prediction path
    treats this view as fitted.
    """

    def __init__(self, snapshot, eligible):
        import torch
        if snapshot.get("weight") is None or not snapshot.get("memory"):
            raise ValueError("Masked routing requires a fitted native joint router snapshot")
        if not eligible:
            raise ValueError("Masked routing requires at least one eligible learned expert")
        self._torch = torch
        self.snapshot = snapshot
        self.memory = list(snapshot["memory"])
        self.eligible = list(eligible)
        self.center = snapshot["center"]
        self.weight = snapshot["weight"]
        self.prior = snapshot["prior"]

    def _features(self, features):
        torch = self._torch
        if features.ndim != 2 or features.shape[1] != 1536 or not torch.isfinite(features).all():
            raise ValueError("Expected finite image/question features [N,1536]")
        return features.float()

    def predict(self, features):
        x = self._features(features)
        device = x.device
        scores = (x - self.center.to(device)) @ self.weight.to(device) + self.prior.to(device)
        mask = self._torch.full(scores.shape, -float("inf"), dtype=scores.dtype, device=scores.device)
        mask[:, self.eligible] = 0.0
        return (scores + mask).argmax(-1)


def request_family(task, rows):
    """Public request family derived only from the visible request text."""
    formats = {row["input_text"] == CAPTION_INSTRUCTION for row in rows}
    if len(formats) != 1:
        raise ValueError(f"Mixed request formats inside one task: {task['task_id']}")
    family = "caption" if formats.pop() else "vqa"
    if family != task["family"]:
        raise ValueError(f"Public request family disagrees with manifest family: {task['task_id']}")
    return family


def _assert_recorded(actual, recorded, label, tolerance=1e-10):
    """Float-shaped scientific values must reproduce the parent recording."""
    if isinstance(actual, dict):
        if not isinstance(recorded, dict) or set(actual) != set(recorded):
            raise ValueError(f"Recorded {label} keys differ")
        for key in actual:
            _assert_recorded(actual[key], recorded[key], f"{label}/{key}", tolerance)
    elif isinstance(actual, list):
        if not isinstance(recorded, list) or len(actual) != len(recorded):
            raise ValueError(f"Recorded {label} shape differs")
        for index, (left, right) in enumerate(zip(actual, recorded)):
            _assert_recorded(left, right, f"{label}/{index}", tolerance)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                or not math.isfinite(actual) or not math.isfinite(recorded)
                or not math.isclose(actual, recorded, rel_tol=tolerance, abs_tol=tolerance)):
            raise ValueError(f"Recorded {label} differs")
    elif actual != recorded:
        raise ValueError(f"Recorded {label} differs")


def _positive(config, key):
    value = config.get(key)
    return type(value) is int and value >= 1


def _finite_positive(config, key):
    value = config.get(key)
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _verify_parent(cli):
    """Load and verify the completed parent expansion run."""
    config = json.loads((cli.parent / "config.json").read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected parent schema_version=1")
    if config["method"] != "expand":
        raise ValueError("Only expansion banks reconstruct every stage from a frozen final bank")
    if config["router_modality"] != "joint":
        raise ValueError(f"Native joint routing is the fixed G5 contract; parent uses {config['router_modality']}")
    if not _positive(config, "router_memory") or not _finite_positive(config, "router_ridge"):
        raise ValueError("Parent router memory/ridge contract is malformed")
    task_ids = config["task_ids"]
    if config.get("task_order") != task_ids or not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("Parent task order must equal the declared unique task IDs")
    if len(task_ids) != 4:
        raise ValueError("Frozen-bank routing diagnostics cover the four-task confirmation stream")
    if any(not str(task_id).replace("-", "").replace("_", "").isalnum() for task_id in task_ids):
        raise ValueError("Task IDs must be safe filename components")
    for key in ("seed", "rank", "batch_size", "micro_batch_size", "image_size"):
        if not _positive(config, key):
            raise ValueError(f"Parent {key} must be a positive integer")
    for key in ("lr", "alpha"):
        if not _finite_positive(config, key):
            raise ValueError(f"Parent {key} must be finite and positive")
    if config["eval_split"] not in ("validation", "eval"):
        raise ValueError("Parent eval_split must be validation or eval")
    if config["eval_split"] == "eval" and not cli.allow_eval:
        raise ValueError("Confirmation-eval parents require --allow-eval (locked confirmation invocation)")
    if config["eval_split"] == "validation" and cli.allow_eval:
        raise ValueError("Validation parents must not pass --allow-eval")
    limits = (config.get("train_limit", 0), config.get("eval_limit", 0))
    if any(type(limit) is not int or limit < 0 for limit in limits):
        raise ValueError("Parent eval/train prefixes must be nonnegative integers")
    if any(limits) and (not config.get("smoke_output") or config["eval_split"] != "validation"):
        raise ValueError("Tiny parent subsets are only valid for explicit validation smoke")
    result = json.loads((cli.parent / "results.json").read_text())
    if (result.get("schema_version") != 1 or result.get("completed") != len(task_ids)
            or result.get("status") != "complete"):
        raise ValueError("Parent run is incomplete")
    if result.get("task_order") != task_ids:
        raise ValueError("Parent result task order differs from the configuration")
    assignments = result.get("task_expert_assignments")
    if assignments != list(range(len(task_ids))):
        raise ValueError("Expansion parents add exactly one fresh expert per task")
    matrix = result.get("matrix")
    if (not isinstance(matrix, list) or len(matrix) != len(task_ids)
            or any(not isinstance(row, list) or len(row) != stage + 1 for stage, row in enumerate(matrix))):
        raise ValueError("Parent matrix must be a complete lower triangle")
    pre_scores = result.get("pre_task_scores")
    if (not isinstance(pre_scores, list) or len(pre_scores) != len(task_ids)
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in pre_scores)):
        raise ValueError("Parent pre-task scores must be complete and finite")
    stage_evaluations = result.get("stage_evaluations")
    if (not isinstance(stage_evaluations, list) or len(stage_evaluations) != len(task_ids)
            or any(not isinstance(entries, list) for entries in stage_evaluations)):
        raise ValueError("Parent stage evaluations must cover every task stage")
    return config, result


def _verify_manifest(config):
    raw = Path(config["manifest"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Parent manifest changed since the run")
    manifest = json.loads(raw)
    if config.get("protocol") != manifest["protocol"]:
        raise ValueError("Parent protocol differs from the manifest")
    if config["eval_split"] == "eval" and manifest.get("confirmation_eligible") is False:
        raise ValueError("Development manifests are not confirmation eligible")
    return validate_manifest(manifest, image_root=Path(config["image_root"]).resolve())


def _parse_argv(argv):
    """Parse the literal runner argv stored inside a confirmation lock job."""
    values = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:].replace("-", "_")
        collected = []
        probe = index + 1
        while probe < len(argv) and not argv[probe].startswith("--"):
            collected.append(argv[probe])
            probe += 1
        if name.startswith("task_ids"):
            values[name] = collected
        elif collected:
            values[name] = collected[0]
        else:
            values[name] = True
        index = probe
    return values


def _numeric(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _job_expands_parent(argv, config):
    values = _parse_argv(argv)
    task_ids = values.get("task_ids")
    return (values.get("method") == "expand" and isinstance(task_ids, list) and task_ids == config["task_ids"]
            and values.get("seed") == str(config["seed"])
            and _numeric(values.get("lr")) == config["lr"]
            and _numeric(values.get("epochs")) == config["epochs"]
            and values.get("eval_split") == config["eval_split"])


def verify_confirmation_lock(path, expected_sha256, config):
    """Verify the parent-created G4 lock and bind this parent to its job grid."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Explicit confirmation-lock SHA256 mismatch")
    lock = json.loads(raw)
    if lock.get("schema_version") != 1 or lock.get("stage") != "G4":
        raise ValueError("Expected a locked G4 confirmation artifact")
    for field in ("manifest_sha256", "source_sha256", "assets_sha256", "tokenizer_sha256", "scorer_locks",
                  "normalizers_sha256", "normalizers", "seeds", "orders", "common_training", "selected_replay",
                  "bootstrap_seed", "bootstrap_replicates", "jobs", "prior_gate_reports"):
        if field not in lock:
            raise ValueError(f"Confirmation lock is missing {field}")
    if lock["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Locked confirmation uses a different global manifest")
    for field in ("source_sha256", "assets_sha256", "tokenizer_sha256", "scorer_locks"):
        if lock[field] != config[field]:
            raise ValueError(f"Locked confirmation {field} differs from the parent configuration")
    normalizers = lock["normalizers"]
    if (normalizers.get("schema_version") != 1 or normalizers.get("eval_split") != "validation"
            or normalizers.get("manifest_sha256") != config["manifest_sha256"]):
        raise ValueError("Locked normalizer must be a fixed matching validation artifact")
    scores = normalizers.get("scores")
    if (not isinstance(scores, dict) or set(scores) != set(config["task_ids"])
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in scores.values())):
        raise ValueError("Locked normalizer must carry finite positive constants for every task")
    if (type(lock["bootstrap_seed"]) is not int or type(lock["bootstrap_replicates"]) is not int
            or lock["bootstrap_replicates"] < 1):
        raise ValueError("Locked bootstrap constants are malformed")
    if not isinstance(lock["orders"], list) or config["task_ids"] not in [
            list(order) for order in lock["orders"] if isinstance(order, list)]:
        raise ValueError("Parent task order is not part of the locked confirmation orders")
    if not isinstance(lock["seeds"], list) or config["seed"] not in lock["seeds"]:
        raise ValueError("Parent seed is not part of the locked confirmation seeds")
    selected = lock["selected_replay"]
    if (not isinstance(selected, dict) or set(selected) != {"lr", "epochs"}
            or any(isinstance(selected.get(key), bool) or not isinstance(selected.get(key), (int, float))
                   or not math.isfinite(float(selected.get(key))) for key in ("lr", "epochs"))):
        raise ValueError("Locked tuned replay selection is malformed")
    common = lock["common_training"]
    if not isinstance(common, dict) or not all(key in common for key in ("lr", "epochs", "batch_size",
                                                                        "micro_batch_size")):
        raise ValueError("Locked common training contract is malformed")
    for key, name in (("lr", "lr"), ("epochs", "epochs"), ("batch_size", "batch_size"),
                      ("micro_batch_size", "micro_batch_size")):
        locked_value, parent_value = common[key], config[name]
        if (isinstance(locked_value, bool) or isinstance(parent_value, bool)
                or not isinstance(locked_value, (int, float)) or not isinstance(parent_value, (int, float))
                or not math.isclose(float(locked_value), float(parent_value), rel_tol=1e-12)):
            raise ValueError(f"Locked common {name} differs from the parent configuration")
    jobs = lock["jobs"]
    if not isinstance(jobs, list) or not 192 <= len(jobs) <= 224:
        raise ValueError("Locked confirmation must declare the complete 192-224 training job grid")
    matched = False
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("name"), str):
            raise ValueError("Every locked job needs a literal name")
        argv = job.get("argv")
        if isinstance(argv, str):
            argv = argv.split()
        if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
            raise ValueError("Every locked job argv must be literal strings")
        if _job_expands_parent(argv, config):
            matched = True
    if not matched:
        raise ValueError("Parent configuration is absent from the locked confirmation job grid")
    prior = lock["prior_gate_reports"]
    if not isinstance(prior, dict) or not prior:
        raise ValueError("Prior gate report inventory must be a nonempty hash map")
    for name, expected in prior.items():
        candidate = Path(name) if Path(name).is_absolute() else ROOT / name
        if not candidate.is_file():
            raise ValueError(f"Prior gate report file is missing on this machine: {candidate}")
        if digest_file(candidate) != expected:
            raise ValueError(f"Prior gate report changed: {candidate}")
    return {"path": str(Path(path).resolve()), "sha256": expected_sha256,
            "normalizers": scores, "normalizers_sha256": lock["normalizers_sha256"]}


def read_records(path):
    """Every saved record must be a complete generated prediction with diagnostics."""
    expert_values, records = [], []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                or not isinstance(record.get("prediction"), str)):
            raise ValueError(f"Malformed prediction record: {path}")
        expert = record["expert"]
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise ValueError(f"Malformed expert field: {path}")
        generation = record.get("generation", {})
        for key in GENERATION_KEYS:
            value = generation.get(key)
            valid = type(value) is int and value >= 0 if key == "generated_tokens" else type(value) is bool
            if not valid:
                raise ValueError(f"Invalid generation diagnostic {key}: {path}")
        expert_values.append(expert)
        records.append(record)
    if not records:
        raise ValueError(f"Empty prediction file: {path}")
    return records, expert_values


def _verify_routes(kind, stage, variant, expert_values, eligible, gold, routing):
    """Reject any routing contract violation, including touching a future expert."""
    values = set(expert_values)
    if kind == "after":
        if variant == "task_id_oracle":
            if values != {gold}:
                raise ValueError("Oracle routing must use only the gold task's already-trained expert")
            return
        if -1 in values:
            raise ValueError("Seen tasks are never routed base-only")
        if any(value < 0 or value >= stage + 1 for value in values):
            raise ValueError("Router touched a future/unseen expert")
        if variant == "family_restricted_joint" and not values <= set(eligible):
            raise ValueError("Family-restricted routing left the public request family")
        return
    if variant == "task_id_oracle":
        raise ValueError("The oracle has no pre-task diagnostics: its expert may not exist yet")
    if stage >= 1 and any(value < -1 or value >= stage for value in values):
        raise ValueError("Pre-task router touched a future/unseen expert")
    if stage == 0 and values != {-1}:
        raise ValueError("Stage-0 pre-task reconstruction must be exactly the parent base-only routing")
    if -1 in values:
        if routing not in ("base_only_no_expert", "base_only_no_eligible_family"):
            raise ValueError("Base-only routing must be declared for unroutable pre-task requests")
        return
    if variant == "family_restricted_joint" and routing == "family_masked" and not values <= set(eligible):
        raise ValueError("Family-restricted routing left the public request family")


def _absorb_view(state, *, kind, stage, stages, task_id, task_family, gold, eligible, routing,
                 family_available, view, path):
    """Read back the emitted prediction file, verify routing, and record the entry."""
    records, expert_values = read_records(path)
    _verify_routes(kind, stage, state["variant"], expert_values, eligible, gold, routing)
    if kind == "after" and stage == stages - 1:
        state["final_answer_counts"].update([record["prediction"] for record in records])
    entry = {"stage": stage, "kind": kind, "task_id": task_id, "family": task_family,
             "metric": view["metric"], "scoring": dict(view["scoring"]),
             "generation": dict(view["generation"]), "prediction_counts": dict(view["prediction_counts"]),
             "most_frequent_output_fraction": view["most_frequent_output_fraction"],
             "route_counts": {str(expert): count for expert, count in view["route_counts"].items()},
             "routing": routing, "family_available":
                 None if state["variant"] != "family_restricted_joint" else family_available,
             "examples": len(expert_values), "predictions_file": path.name}
    state["stage_evaluations"].append(entry)
    if kind == "after":
        confusion = state.setdefault("stage_route_gold", {}).setdefault(stage, Counter())
        for expert, count in view["route_counts"].items():
            confusion[(gold, expert)] += count
    return view["scoring"]["score"]


def _run_pass(group, *, model, tasks, scorers, args, joint_train_features, eval_features, assignments,
              expert_family, directories, parent, states, resources, config, stages, stage_snapshots=None):
    """Replay the parent stage sequence and evaluate the requested variants.

    The family-restricted variant never constructs or observes its own router:
    it consumes the byte-exact native per-stage router snapshots recorded by the
    native pass (``stage_snapshots[stage]["before"]`` for the pre-task state and
    ``["after"]`` for the freshly observed state) and applies a candidate mask.
    """
    import torch
    if "family_restricted_joint" in group and stage_snapshots is None:
        raise ValueError("The family-restricted variant requires native per-stage router snapshots")
    routers = {variant: FeatureRouter(config["router_memory"], config["router_ridge"],
                                      MODALITY[variant], config["seed"])
               for variant in group if variant not in ("task_id_oracle", "family_restricted_joint")}
    native_router = routers.get("native_joint")
    for stage in range(stages):
        task = tasks[stage]
        task_id = task["task_id"]
        available = stage + 1
        eligible_pre = [expert for expert in range(stage) if expert_family[expert] == task["own_family"]]
        scores = {variant: [] for variant in group}
        if stage_snapshots is not None and native_router is not None:
            stage_snapshots.append({"before": _snapshot_router(native_router), "after": None})
        for variant in group:
            if variant == "task_id_oracle":
                continue
            state, router = states[variant], routers.get(variant)
            path = predictions_path(directories[variant], "before", stage, task_id)
            started = time.perf_counter()
            family_restricted = variant == "family_restricted_joint"
            if stage == 0:
                # The parent itself routed this pre-task request base-only: no expert existed yet.
                evaluation = runner.evaluate(model, router, task, scorers[task_id], eval_features[stage],
                                             args, path, forced_expert=-1)
                routing, family_reachable = "base_only_no_expert", None
            elif family_restricted and not eligible_pre:
                # No learned expert of this request family exists yet; never pretend one.
                evaluation = runner.evaluate(model, router, task, scorers[task_id], eval_features[stage],
                                             args, path, forced_expert=-1)
                routing, family_reachable = "base_only_no_eligible_family", False
            elif family_restricted:
                prediction_router = MaskedRouter(stage_snapshots[stage]["before"], eligible_pre)
                evaluation = runner.evaluate(model, prediction_router, task, scorers[task_id],
                                             eval_features[stage], args, path)
                routing, family_reachable = "family_masked", True
            else:
                evaluation = runner.evaluate(model, router, task, scorers[task_id],
                                             eval_features[stage], args, path)
                routing, family_reachable = "router_memory", None
            score = _absorb_view(state, kind="before", stage=stage, stages=stages, task_id=task_id,
                                 task_family=task["own_family"], gold=None, eligible=eligible_pre,
                                 routing=routing, family_available=family_reachable,
                                 view=evaluation, path=path)
            state["pre_task_scores"].append(score)
            resources[variant]["generation_seconds"] += time.perf_counter() - started
            if parent is not None:
                _assert_parent_parity(path, parent)
                state["parent_parity_files"] = state.get("parent_parity_files", 0) + 1
        for variant in group:
            if variant not in ("task_id_oracle", "family_restricted_joint"):
                routers[variant].observe(joint_train_features[stage], assignments[stage], available)
        if stage_snapshots is not None and native_router is not None:
            stage_snapshots[stage]["after"] = _snapshot_router(native_router)
        for seen in range(stage + 1):
            previous = tasks[seen]
            previous_id = previous["task_id"]
            eligible = [expert for expert in range(available) if expert_family[expert] == previous["own_family"]]
            for variant in group:
                state = states[variant]
                path = predictions_path(directories[variant], "after", stage, previous_id)
                started = time.perf_counter()
                if variant == "task_id_oracle":
                    evaluation = runner.evaluate(model, None, previous, scorers[previous_id],
                                                 eval_features[seen], args, path,
                                                 forced_expert=assignments[seen])
                elif variant == "family_restricted_joint":
                    if not eligible:
                        raise ValueError("Seen tasks always expose their own family expert")
                    evaluation = runner.evaluate(model, MaskedRouter(stage_snapshots[stage]["after"],
                                                                     eligible), previous,
                                                 scorers[previous_id], eval_features[seen], args, path)
                else:
                    evaluation = runner.evaluate(model, routers[variant], previous, scorers[previous_id],
                                                 eval_features[seen], args, path)
                routing = ("gold_task_expert" if variant == "task_id_oracle"
                           else "family_masked" if variant == "family_restricted_joint" else "router_memory")
                family_reachable = True if variant == "family_restricted_joint" else None
                score = _absorb_view(state, kind="after", stage=stage, stages=stages, task_id=previous_id,
                                     task_family=previous["own_family"], gold=assignments[seen],
                                     eligible=eligible, routing=routing, family_available=family_reachable,
                                     view=evaluation, path=path)
                scores[variant].append(score)
                resources[variant]["generation_seconds"] += time.perf_counter() - started
                if parent is not None:
                    _assert_parent_parity(path, parent)
                    state["parent_parity_files"] = state.get("parent_parity_files", 0) + 1
        for variant in group:
            row = scores[variant]
            if len(row) != stage + 1:
                raise ValueError("Stage evaluation lost a seen-task score")
            states[variant]["matrix"].append(row)
        if model.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated()
            for variant in group:
                values = resources[variant]
                values["peak_gpu_memory_bytes"] = max(values["peak_gpu_memory_bytes"], peak)
    return routers


def _assert_parent_parity(path, parent):
    parent_path = parent / path.name
    if not parent_path.is_file():
        raise AssertionError(f"Parent prediction file missing: {parent_path}")
    if path.read_bytes() != parent_path.read_bytes():
        raise AssertionError(f"Native reconstruction does not byte-match the parent at {path.name}")


def _verify_replayed_router(router, saved):
    """The replayed native router must equal the parent checkpoint's own router."""
    import torch
    if saved.get("num_experts") != router.num_experts:
        raise ValueError("Replayed router expert count differs from the parent checkpoint")
    memory = saved.get("memory")
    if not isinstance(memory, list) or len(memory) != len(router.memory):
        raise ValueError("Replayed routing memory differs from the parent checkpoint")
    for (saved_tensor, saved_expert), (replay_tensor, replay_expert) in zip(memory, router.memory):
        if saved_expert != replay_expert or not torch.equal(saved_tensor, torch.as_tensor(replay_tensor)):
            raise ValueError("Replayed routing memory tensors differ from the parent checkpoint")
    for key in ("weight", "center", "prior"):
        saved_value, replayed = saved.get(key), getattr(router, key)
        if saved_value is None or replayed is None or not torch.equal(replayed, saved_value):
            raise ValueError(f"Replayed router {key} differs from the parent checkpoint")
    return True


def _check_bank_unchanged(model, bank_state):
    """Prove the frozen bank tensors survived the entire inference ablation."""
    import torch
    if model.rank != bank_state["rank"] or not math.isclose(model.alpha, bank_state["alpha"], rel_tol=1e-12):
        raise ValueError("LoRA capacity changed during the routing ablation")
    for name, reference in bank_state["projections"].items():
        projection = model._projections.get(name)
        if projection is None:
            raise ValueError(f"Frozen bank projection went missing: {name}")
        current = {key: value.detach().cpu() for key, value in projection.experts.state_dict().items()}
        if set(current) != set(reference):
            raise ValueError(f"Frozen bank projection keys changed: {name}")
        for key, expected in reference.items():
            if not torch.equal(current[key], expected):
                raise ValueError(f"Frozen bank projection tensor changed: {name}")
    return True


def _assert_native_support_matches(state, result, depth):
    """Native stage auxiliaries must also reproduce the parent's recorded after-evaluations."""
    recorded_by_file = {}
    for recorded_stage in result["stage_evaluations"]:
        for entry in recorded_stage:
            recorded_by_file[entry["predictions_file"]] = entry
    common = ("task_id", "family", "metric", "scoring", "generation", "prediction_counts",
              "most_frequent_output_fraction", "route_counts", "predictions_file")
    for entry in state["stage_evaluations"]:
        if entry["kind"] != "after":
            continue
        recorded = recorded_by_file.get(entry["predictions_file"])
        if recorded is None:
            raise ValueError(f"Parent stage evaluation is missing: {entry['predictions_file']}")
        if not all(key in recorded for key in common):
            raise ValueError(f"Parent stage evaluation keyset differs: {entry['predictions_file']}")
        _assert_recorded({key: entry[key] for key in common},
                         {key: recorded[key] for key in common},
                         f"stage_evaluations/{entry['predictions_file']}")


def _finish_states(states, tasks, expert_family, resources, feature_seconds, stages):
    for variant, state in states.items():
        if variant == "task_id_oracle":
            state["metrics"] = continual_metrics(state["matrix"], tasks)
            state["pre_task_scores"] = None
            state["pre_to_post_gain"] = None
        else:
            state["metrics"] = continual_metrics(state["matrix"], tasks, pre_task_scores=state["pre_task_scores"])
            state["pre_to_post_gain"] = [row[index] - state["pre_task_scores"][index]
                                         for index, row in enumerate(state["matrix"])]
        final_confusion = state.get("stage_route_gold", {}).get(stages - 1, {})
        confusion = {"vqa": {}, "caption": {}, "all": {}}
        for (gold, routed), count in final_confusion.items():
            family = expert_family[gold]
            confusion["all"].setdefault(str(gold), {})[str(routed)] = count
            confusion[family].setdefault(str(gold), {})[str(routed)] = count
        state["final_stage_confusion"] = confusion
    return {
        "variants": {variant: {"matrix": state["matrix"], "pre_task_scores": state["pre_task_scores"],
                               "pre_to_post_gain": state["pre_to_post_gain"], "metrics": state["metrics"],
                               "final_answer_counts": dict(state["final_answer_counts"]),
                               "largest_answer_fraction": (max(state["final_answer_counts"].values())
                                                           / sum(state["final_answer_counts"].values()))
                               if state["final_answer_counts"] else 0.0,
                               "stage_evaluations": state["stage_evaluations"],
                               "final_stage_confusion": state["final_stage_confusion"],
                               "parent_parity_files": state.get("parent_parity_files", 0)}
                     for variant, state in states.items()},
        "resources": {variant: {**values, "feature_seconds_shared": feature_seconds}
                      for variant, values in resources.items()},
    }


def _safe_output(path, parent):
    resolved = Path(path).expanduser().resolve()
    forbidden = ("readout", "vision", "legacy", "sema_runs", "vqa_runs")
    if any(part.casefold().startswith(forbidden) for part in resolved.parts):
        raise ValueError("Archived artifact namespaces are read-only")
    runs = (ROOT / "artifacts" / "multimodal_runs").resolve()
    if not resolved.is_relative_to(runs) or resolved == runs:
        raise ValueError("Routing ablation outputs stay under artifacts/multimodal_runs")
    parent_resolved = Path(parent).resolve()
    if resolved == parent_resolved or resolved.is_relative_to(parent_resolved):
        raise ValueError("The ablation output must not live inside the parent run")
    if resolved.exists():
        raise ValueError("Inference ablation requires an unused output path")
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--parent", type=Path, required=True, help="Completed expansion parent run directory")
    parser.add_argument("--output", type=Path, required=True, help="New ablation output directory")
    parser.add_argument("--allow-eval", action="store_true",
                        help="Permit a confirmation-eval parent; requires an explicit locked confirmation artifact")
    parser.add_argument("--confirmation-lock", type=Path, help="Parent-created G4 confirmation lock JSON")
    parser.add_argument("--confirmation-lock-sha256", help="Explicit SHA256 of --confirmation-lock")
    cli = parser.parse_args(argv)
    if bool(cli.confirmation_lock) != bool(cli.confirmation_lock_sha256):
        parser.error("--confirmation-lock and --confirmation-lock-sha256 must be supplied together")
    import numpy as np
    import torch
    run_started = time.perf_counter()
    config, result = _verify_parent(cli)
    for library, current in (("torch", torch.__version__), ("numpy", np.__version__)):
        recorded = config.get(library)
        if recorded is not None and str(current) != str(recorded):
            raise ValueError(f"Parent recorded a different {library} version; numerical parity is not on this stack")
    if config["eval_split"] == "eval":
        if cli.confirmation_lock is None:
            raise ValueError("Confirmation-eval --allow-eval requires an explicit locked confirmation artifact")
        confirmation_lock = verify_confirmation_lock(cli.confirmation_lock, cli.confirmation_lock_sha256, config)
    else:
        confirmation_lock = None
    manifest = _verify_manifest(config)
    by_id = {task["task_id"]: task for task in manifest["tasks"]}
    if set(config["task_ids"]) != set(by_id):
        raise ValueError("Parent task IDs must cover the complete manifest")
    tasks = []
    for task_id in config["task_ids"]:
        task = dict(by_id[task_id])
        task["own_family"] = request_family(task, task[config["eval_split"]])
        tasks.append(task)
    scorers = {task["task_id"]: TaskScorer(task) for task in tasks}
    _assert_recorded({task["task_id"]: scorer.lock for task, scorer in zip(tasks, scorers.values())},
                     config["scorer_locks"], "scorer_locks")
    _assert_recorded(runner.source_hashes(), config["source_sha256"], "source_sha256")
    asset_root = Path(config["assets"]).resolve()
    for name, expected_sha256 in config["assets_sha256"].items():
        asset_path = Path(name)
        candidate = asset_path if asset_path.is_absolute() else asset_root / name
        if digest_file(candidate) != expected_sha256:
            raise ValueError(f"Changed model asset file: {candidate}")
    _assert_recorded({key: value for key, value in config["assets_sha256"].items() if key.startswith("tokenizer/")},
                     config["tokenizer_sha256"], "tokenizer_sha256")
    assignments = result["task_expert_assignments"]
    expert_family = {expert: task["own_family"] for expert, task in zip(assignments, tasks)}
    checkpoint = torch.load(cli.parent / "latest.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("settings") != config:
        raise ValueError("Parent checkpoint settings differ from the recorded configuration")
    if checkpoint.get("state", {}).get("assignments") != assignments:
        raise ValueError("Parent checkpoint assignments differ from the recorded assignments")
    bank_state = checkpoint["model"]
    if (bank_state.get("num_experts") != len(tasks) or bank_state.get("rank") != config["rank"]
            or not math.isclose(bank_state.get("alpha"), config["alpha"], rel_tol=1e-12)):
        raise ValueError("Parent checkpoint bank does not match the declared capacity")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = config.get("device", "cuda")
    model = MultimodalReadout(config["assets"], image_size=config["image_size"], rank=config["rank"],
                              alpha=config["alpha"], device=device)
    model.load_checkpoint_state(bank_state)
    if model.num_experts != len(tasks):
        raise ValueError("The reconstructed bank size disagrees with the parent stream")
    model._freeze_all()
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    args = SimpleNamespace(**config)
    output = _safe_output(cli.output, cli.parent)
    stages = len(tasks)
    eval_limit = config.get("eval_limit", 0)
    eval_features, joint_train_features, feature_started = [], [], time.perf_counter()
    for stage, task in enumerate(tasks):
        joint_train_features.append(runner.features(model, task["train"][:config.get("train_limit", 0) or None],
                                                    args))
        eval_features.append(runner.features(model, task[config["eval_split"]][:eval_limit or None], args))
    feature_seconds = time.perf_counter() - feature_started
    directories = {variant: output / variant for variant in VARIANTS}
    output.mkdir(parents=True, exist_ok=False)
    for variant in VARIANTS:
        directories[variant].mkdir()
    states = {variant: {"variant": variant, "matrix": [], "pre_task_scores": [],
                        "final_answer_counts": Counter(), "stage_evaluations": [], "stage_route_gold": {}}
              for variant in VARIANTS}
    resources = {variant: {"generation_seconds": 0.0, "peak_gpu_memory_bytes": 0} for variant in VARIANTS}
    stage_snapshots = []
    # Pass 1 reproduces the parent's native routing byte-for-byte and records the
    # per-stage native router snapshots; alternatives only run afterwards and reuse
    # those snapshots for the family restriction, never refitting anything.
    native_routers = _run_pass(("native_joint",), model=model, tasks=tasks, scorers=scorers, args=args,
                               joint_train_features=joint_train_features, eval_features=eval_features,
                               assignments=assignments, expert_family=expert_family, directories=directories,
                               parent=cli.parent, states=states, resources=resources, config=config,
                               stages=stages, stage_snapshots=stage_snapshots)
    _assert_recorded(states["native_joint"]["matrix"], result["matrix"], "native matrix")
    _assert_recorded(states["native_joint"]["pre_task_scores"], result["pre_task_scores"], "native pre-task scores")
    router_verified = _verify_replayed_router(native_routers["native_joint"], checkpoint["router"])
    _assert_native_support_matches(states["native_joint"], result, stages)
    native_parity_expected = stages + sum(stage + 1 for stage in range(stages))
    if states["native_joint"].get("parent_parity_files", 0) != native_parity_expected:
        raise ValueError("Native parity coverage is incomplete")
    if len(stage_snapshots) != stages or any(
            "before" not in snapshot or "after" not in snapshot for snapshot in stage_snapshots):
        raise ValueError("Missing native per-stage router snapshots for the family restriction")
    _run_pass(("image_only", "question_only", "family_restricted_joint", "task_id_oracle"), model=model,
              tasks=tasks, scorers=scorers, args=args, joint_train_features=joint_train_features,
              eval_features=eval_features, assignments=assignments, expert_family=expert_family,
              directories=directories, parent=None, states=states, resources=resources, config=config,
              stages=stages, stage_snapshots=stage_snapshots)
    bank_verified = _check_bank_unchanged(model, bank_state)
    selection = _finish_states(states, tasks, expert_family, resources, feature_seconds, stages)
    unroutable = [{"variant": variant, "stage": entry["stage"], "task_id": entry["task_id"],
                   "family": entry["family"]}
                  for variant, state in states.items() for entry in state["stage_evaluations"]
                  if entry["routing"] == "base_only_no_eligible_family"]
    report = {
        "schema_version": 1, "stage": "G5_frozen_bank_routing", "parent": str(Path(cli.parent).resolve()),
        "parent_config": config, "manifest_sha256": config["manifest_sha256"],
        "parent_checkpoint_sha256": digest_file(cli.parent / "latest.pt"),
        "parent_results_sha256": digest_file(cli.parent / "results.json"),
        "script_sha256": digest_file(Path(__file__)), "confirmation_lock": confirmation_lock,
        "allow_eval": bool(cli.allow_eval), "eval_split": config["eval_split"],
        "assignments": assignments, "expert_family": {str(expert): family for expert, family in expert_family.items()},
        "native_variant": "native_joint", "native_prediction_files_byte_identical": native_parity_expected,
        "native_prediction_files_expected": native_parity_expected,
        "expert_retraining_steps": 0, "model_projection_tensors_byte_identical_to_checkpoint": bank_verified,
        "router_checkpoint_tensors_identical": router_verified,
        "variants": selection["variants"], "resources": selection["resources"],
        "unroutable_stage_tasks": unroutable,
        "pre_task_semantics": ("Pre-task diagnostics reproduce the parent's base-only behavior when router memory "
                               "is absent; family-restricted pre-task requests without an eligible learned expert "
                               "are also base-only and flagged unroutable instead of pretending knowledge."),
        "scope": "Inference-only frozen-bank routing diagnostics. Native predictions and matrices byte-match the "
                 "parent before alternatives are reported; no expert update or backbone mutation; no future or "
                 "unseen expert routing; the task-ID oracle is a separate upper bound over already-trained experts.",
    }
    runner.atomic_json(output / "routing_ablation.json", report)
    print(json.dumps({"parent": str(Path(cli.parent).resolve()), "output": str(output),
                      "native_prediction_files_byte_identical": native_parity_expected,
                      "expert_retraining_steps": 0, "unroutable_stage_tasks": len(unroutable),
                      "variants": {variant: state["metrics"] for variant, state in states.items()},
                      "wall_seconds": time.perf_counter() - run_started}, allow_nan=False))


if __name__ == "__main__":
    main()
