"""Reproducible routing-review analysis for MULTIMODAL_RESEARCH_SUMMARY.md items 2 and 7.

Scope (delegated sub-review; the parent aggregates findings into the document):

  item 2 - verify code initialization, freezing, and independent vs expansion training
           conditions; determine whether fresh-expert training is truly
           independent-equivalent (RNG, batch/reference selection, dropout), and check
           the router-only cross-task knowledge path.
  item 7 - use existing G4/G5 predictions with previously saved corpus scores; at each
           task's immediately-post-training (diagonal) stage partition VQA predictions
           into correctly-routed-current-expert vs misrouted rows, pair IDs with the
           independent runs' predictions, reproduce the reported normalized VQA
           transfer -0.0432397 under the original aggregation weighting, and use the G5
           task_id_oracle after-training predictions to separate the routing
           counterfactual from training differences; caption final-stage native vs
           oracle partition into same-family-other-expert vs own-expert.

Guarantee honored here: NO model inference and NO scorer executions.  Only existing,
previously-rescored corpus scores are aggregated plus deterministic text-identity
predicates and expert-assignment counts read from saved prediction files.

Inputs (read-only):
  artifacts/multimodal-g4-report/audit.json         existing rescored G4 corpus scores + transfer records
  artifacts/multimodal-g4-report/statistics.json    reported transfer statistics to reproduce
  artifacts/multimodal-g5-report/routing_audit.json existing rescored G5 per-bank variant scores
  artifacts/multimodal-g5-summary.json              reported G5 aggregates to reconcile
  artifacts/multimodal-g1-normalizers.json          fixed G1 validation normalizers
  artifacts/multimodal-g4-raw.tar.gz                G4 per-row predictions (id, prediction, expert)
  artifacts/multimodal-g5-raw.tar.gz                G5 per-row predictions for all routing variants
  sema_multimodal/*.py, sema_vqa/{readout_model,readout_router,vision_model,expansion}.py
                                                    code under review (SHA256 parity is checked)

Outputs:
  artifacts/routing-review-code-conditions.json     item-2 verification evidence
  artifacts/routing-review-routing-partition.json   item-7 partition/decomposition numbers
"""
from __future__ import annotations

import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ("V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V")
SEEDS = (137, 149, 163, 179, 191, 211, 227, 241)
TASKS = ("V", "C", "G", "F")
FAMILY_OF = {"V": "vqa", "G": "vqa", "C": "caption", "F": "caption"}
FAMILY_MEMBERS = {"vqa": ("V", "G"), "caption": ("C", "F")}
EVAL_ROWS = 512  # confirmation eval size per task (asserted below)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


class TarReader:
    """Preload selected prediction members from the raw archives in one linear pass.

    gzip archives do not support cheap random access, so members are read exactly once
    into memory. The wanted member set is a pure function of the fixed experiment grid,
    so the preselection keeps the run deterministic and stream-safe.
    """

    _G4_ROOT = "artifacts/multimodal_runs/g4"
    _G5_ROOT = "artifacts/multimodal_runs/routing_g4"

    def __init__(self):
        self._members: dict[str, bytes] = {}
        self._wanted_g4: set[str] = set()
        self._wanted_g5: set[str] = set()

    def want_g4(self, member: str) -> "TarReader":
        self._wanted_g4.add(f"{self._G4_ROOT}/{member}")
        return self

    def want_g5(self, member: str) -> "TarReader":
        self._wanted_g5.add(f"{self._G5_ROOT}/{member}")
        return self

    def preload(self):
        for relname, wanted in (("artifacts/multimodal-g4-raw.tar.gz", self._wanted_g4),
                                ("artifacts/multimodal-g5-raw.tar.gz", self._wanted_g5)):
            if not wanted:
                continue
            with tarfile.open(ROOT / relname, "r:gz") as tar:
                found = 0
                for tarinfo in tar:
                    if tarinfo.name in wanted:
                        stream = tar.extractfile(tarinfo)
                        self._members[tarinfo.name] = stream.read() if stream else b""
                        found += 1
                if found != len(wanted):
                    # Stage files for tasks not yet seen at that stream position do not
                    # exist by construction; missing members surface as read errors only.
                    print(f"note: {relname}: wanted {len(wanted)} members, found {found}")

    def read(self, archive: str, member: str) -> bytes:
        # `member` already includes the archive's internal run root (G4_ROOT/G5_ROOT);
        # `archive` only picks the file for error messages.
        key = member
        if key not in self._members:
            raise FileNotFoundError(f"{archive}: {key}")
        return self._members[key]

    def jsonl(self, archive_root: str, member: str) -> list[dict]:
        return jsonl(self.read(archive_root, member))

    def close(self):
        self._members.clear()


def declare_wanted_members() -> None:
    """Register every member the fixed 32-bank grid needs."""
    for order_name in ORDERS:
        order = order_name.split("_")
        for seed in SEEDS:
            name = bank_name(order_name, seed)
            for p, task_id in enumerate(order):
                TAR.want_g4(f"{name}/predictions_after_{p:02d}_{task_id}.jsonl")
                TAR.want_g5(f"{name}/native_joint/predictions_after_{p:02d}_{task_id}.jsonl")
                TAR.want_g5(f"{name}/task_id_oracle/predictions_after_{p:02d}_{task_id}.jsonl")
            # caption final-stage files for every stream position (C/F may not be at stage 3)
            for stage in range(4):
                for task_id in ("C", "F"):
                    TAR.want_g5(f"{name}/native_joint/predictions_after_{stage:02d}_{task_id}.jsonl")
                    TAR.want_g5(f"{name}/task_id_oracle/predictions_after_{stage:02d}_{task_id}.jsonl")
            TAR.want_g4(f"{name}/results.json")
            TAR.want_g4(f"{name}/config.json")
    for task_id in TASKS:
        for seed in SEEDS:
            TAR.want_g4(f"{task_id}_independent_lr0.0003_ep5_seed{seed}/results.json")
            TAR.want_g4(f"{task_id}_independent_lr0.0003_ep5_seed{seed}/config.json")
            TAR.want_g4(f"{task_id}_independent_lr0.0003_ep5_seed{seed}/predictions_after_00_{task_id}.jsonl")


TAR = TarReader()
G4 = "artifacts/multimodal-g4-raw.tar.gz"
G5 = "artifacts/multimodal-g5-raw.tar.gz"
G4_ROOT = "artifacts/multimodal_runs/g4"
G5_ROOT = "artifacts/multimodal_runs/routing_g4"


def bank_name(order_name: str, seed: int) -> str:
    return f"{order_name}_expand_lr0.0003_ep5_seed{seed}"


declare_wanted_members()
TAR.preload()


def one(value) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return float(value)


def mean(values) -> float:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("empty mean")
    return math.fsum(values) / len(values)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


failures: list[str] = []


def check(condition: bool, message: str):
    if not condition:
        failures.append(message)


# ------------------------------------------------------------------ load reports
AUDIT_PATH = ROOT / "artifacts/multimodal-g4-report/audit.json"
STATS_PATH = ROOT / "artifacts/multimodal-g4-report/statistics.json"
G5_AUDIT_PATH = ROOT / "artifacts/multimodal-g5-report/routing_audit.json"
G5_SUMMARY_PATH = ROOT / "artifacts/multimodal-g5-summary.json"
NORM_PATH = ROOT / "artifacts/multimodal-g1-normalizers.json"
INPUTS = [(AUDIT_PATH, "existing G4 rescored corpus scores and transfer records"),
          (STATS_PATH, "reported G4 transfer statistics to reproduce"),
          (G5_AUDIT_PATH, "existing G5 rescored per-bank variant corpus scores"),
          (G5_SUMMARY_PATH, "reported G5 aggregate for reconciliation"),
          (NORM_PATH, "fixed G1 validation normalizer lock"),
          (ROOT / G4, "G4 per-row predictions (id, prediction, expert, generation)"),
          (ROOT / G5, "G5 per-row predictions for all routing variants")]

g4_audit = json.loads(AUDIT_PATH.read_text())
g4_stats = json.loads(STATS_PATH.read_text())
g5_audit = json.loads(G5_AUDIT_PATH.read_text())
g5_summary = json.loads(G5_SUMMARY_PATH.read_text())
NORMALIZER = json.loads(NORM_PATH.read_text())["scores"]

# transfer rows: (order, seed, task) -> {post_train, independent_score, normalized_delta, ...}
# transfer_records covers common CL, frozen_first and tuned_replay kinds; the reported
# expansion transfer uses only the common-CL (lr3e-4, 5 epochs) records.
trial_rows: dict[tuple, dict] = {}
for record in g4_audit["transfer_records"]:
    if record["kind"] != "common_cl" or record["method"] != "expand":
        continue
    order = tuple(record["order"])
    for row in record["rows"]:
        trial_rows[(order, record["seed"], row["task_id"])] = row
check(len(trial_rows) == 128, f"expected 128 common-CL transfer rows, got {len(trial_rows)}")

independent_metrics: dict[tuple, dict] = {}
for trial in g4_audit["trials"]:
    if trial["kind"] == "independent":
        task_id = trial["task_order"][0]
        metrics = trial["audit"]["metrics"]["tasks"][task_id]
        independent_metrics[(trial["seed"], task_id)] = {
            "score": one(metrics["post_train"]),
            "normalized": one(metrics["post_train"]) / one(metrics["normalizer"]),
            "directory": trial["directory"].split("/")[-1],
        }

banks: dict[str, dict] = {}
for run in g5_audit["runs"]:
    name = run["directory"].rstrip("/").split("/")[-1]
    order: tuple = tuple(run["parent_config"]["task_ids"])
    banks[name] = {"order": order, "seed": int(name.rsplit("seed", 1)[1]), "run": run,
                   "pos": {task_id: index for index, task_id in enumerate(order)}}
check(len(banks) == 32, f"expected 32 banks, got {len(banks)}")

# ------------------------------------------------------------------ 1. reproduce transfer
reported = {}
for family in ("vqa", "caption"):
    entry = next(item for item in g4_stats["exploratory_transfer"]
                 if item["family"] == family and item["method"] == "expand")
    reported[family] = {"mean": entry["statistics"]["mean"], "per_seed": entry["statistics"]["per_seed"]}


def slot(order: tuple, seed: int, family: str) -> float:
    return mean(one(trial_rows[(order, seed, task_id)]["normalized_delta"])
                for task_id in FAMILY_MEMBERS[family])


recomputed: dict[str, dict] = {}
for family in ("vqa", "caption"):
    per_seed = [mean(slot(tuple(name.split("_")), seed, family) for name in ORDERS) for seed in SEEDS]
    recomputed[family] = {"mean": mean(per_seed), "per_seed": per_seed}
    check(close(recomputed[family]["mean"], reported[family]["mean"], 1e-10),
          f"{family} transfer mean reproduction failed")
    check(all(close(a, b, 1e-10) for a, b in zip(recomputed[family]["per_seed"], reported[family]["per_seed"])),
          f"{family} transfer per-seed reproduction failed")

# ------------------------------------------------- 2. G4 audit parity vs raw results.json
G4_RESULTS: dict[tuple, dict] = {}
for order_name in ORDERS:
    for seed in SEEDS:
        G4_RESULTS[(order_name, seed, "expand")] = json.loads(TAR.read(
            G4, f"{G4_ROOT}/{bank_name(order_name, seed)}/results.json"))
        for task_id in TASKS:
            G4_RESULTS[(order_name, seed, f"independent:{task_id}")] = json.loads(TAR.read(
                G4, f"{G4_ROOT}/{task_id}_independent_lr0.0003_ep5_seed{seed}/results.json"))

parity_checked = 0
for order_name in ORDERS:
    order = tuple(order_name.split("_"))
    for seed in SEEDS:
        expand_trial = next(t for t in g4_audit["trials"]
                            if tuple(t["task_order"]) == order and t["seed"] == seed
                            and t["method"] == "expand")
        audit_matrix = expand_trial["audit"]["matrix"]
        raw_matrix = G4_RESULTS[(order_name, seed, "expand")]["matrix"]
        check(all(close(one(a), one(b), 1e-12) for row_a, row_b in zip(audit_matrix, raw_matrix)
                  for a, b in zip(row_a, row_b)), f"audit/raw matrix mismatch {order_name} {seed}")
        parity_checked += 1
        for task_id in TASKS:
            raw_ind = G4_RESULTS[(order_name, seed, f"independent:{task_id}")]["matrix"][0][0]
            check(close(one(raw_ind), one(independent_metrics[(seed, task_id)]["score"]), 1e-12),
                  f"independent audit/raw mismatch {task_id} {seed}")
            parity_checked += 1

# ---------------------------------------------- 3. diagonal cells: native/oracle/independent
diag: dict[tuple, dict] = {}
native_diag_byte_checks = []
for name, bank in sorted(banks.items()):
    order, seed, pos = bank["order"], bank["seed"], bank["pos"]
    run = bank["run"]
    native_norm = run["variants"]["native_joint"]["matrix_normalized"]
    oracle_norm = run["variants"]["task_id_oracle"]["matrix_normalized"]
    native_raw = run["variants"]["native_joint"]["matrix"]
    oracle_raw = run["variants"]["task_id_oracle"]["matrix"]
    for p, task_id in enumerate(order):
        rows_g4 = TAR.jsonl(G4, f"{G4_ROOT}/{name}/predictions_after_{p:02d}_{task_id}.jsonl")
        rows_g5 = TAR.jsonl(G5, f"{G5_ROOT}/{name}/native_joint/predictions_after_{p:02d}_{task_id}.jsonl")
        rows_orc = TAR.jsonl(G5, f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_{p:02d}_{task_id}.jsonl")
        check(len(rows_g4) == len(rows_g5) == len(rows_orc) == EVAL_ROWS,
              f"diag row-count mismatch {name} {task_id}")
        check([r["id"] for r in rows_g4] == [r["id"] for r in rows_g5] == [r["id"] for r in rows_orc],
              f"diag id order mismatch {name} {task_id}")
        check(all(a["prediction"] == b["prediction"] for a, b in zip(rows_g4, rows_g5)),
              f"G5 native diag not byte-identical to G4 diag {name} {task_id}")
        native_diag_byte_checks.append(True)
        check(close(one(native_raw[p][p]), one(trial_rows[(order, seed, task_id)]["post_train"]), 1e-12),
              f"G5 native diag raw != G4 post_train {name} {task_id}")
        own_expert = pos[task_id]
        own_mask = [row["expert"] == own_expert for row in rows_g4]
        # Same expert weights on the same row are not bitwise guaranteed across two
        # generation runs: micro-batch composition changes bf16 rounding of near-tied
        # greedy tokens. These are counted, never silently assumed zero.
        own_route_text_mismatch = sum(1 for a, b, own in zip(rows_g4, rows_orc, own_mask) if own
                                      and a["prediction"] != b["prediction"])
        rows_ind = TAR.jsonl(G4, f"{G4_ROOT}/{task_id}_independent_lr0.0003_ep5_seed{seed}"
                                 f"/predictions_after_00_{task_id}.jsonl")
        check(len(rows_ind) == EVAL_ROWS and [r["id"] for r in rows_ind] == [r["id"] for r in rows_g4],
              f"independent pairing mismatch {name} {task_id}")
        eq_na_ind = [a["prediction"] == b["prediction"] for a, b in zip(rows_g4, rows_ind)]
        eq_orc_ind = [a["prediction"] == b["prediction"] for a, b in zip(rows_orc, rows_ind)]
        eq_na_orc = [row["prediction"] == rowo["prediction"]
                     for row, rowo, own in zip(rows_g4, rows_orc, own_mask) if not own]
        # expert k present at a stream stage == position k; a misrouted row's target task
        # is therefore order[row["expert"]]
        diag[(name, task_id)] = {
            "position": p, "family": FAMILY_OF[task_id],
            "native_diag_normalized": one(native_norm[p][p]),
            "oracle_diag_normalized": one(oracle_norm[p][p]),
            "independent_normalized": independent_metrics[(seed, task_id)]["normalized"],
            "mis_targets": dict(Counter(order[row["expert"]] for row in rows_g4
                                        if row["expert"] != own_expert)),
            "identity": {
                "n": EVAL_ROWS,
                "own_rows": sum(own_mask),
                "mis_routed_rows": EVAL_ROWS - sum(own_mask),
                "own_text_eq_independent": sum(eq for eq, own in zip(eq_na_ind, own_mask) if own),
                "mis_text_eq_independent": sum(eq for eq, own in zip(eq_na_ind, own_mask) if not own),
                "mis_text_eq_oracle": sum(eq_na_orc),
                "own_route_text_mismatched_vs_oracle": own_route_text_mismatch,
            },
        }
check(len(native_diag_byte_checks) == 128, "diag parity coverage incomplete")
total_own_route_text_mismatch = sum(cell["identity"]["own_route_text_mismatched_vs_oracle"]
                                    for cell in diag.values())

# ---------------------------------------------- 4. T = R + K transfer decomposition
def decomposition_for(family: str) -> dict:
    members = FAMILY_MEMBERS[family]
    cells = {(order_name, seed): {}
             for order_name in ORDERS for seed in SEEDS}
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        for seed in SEEDS:
            for task_id in members:
                d = diag[(bank_name(order_name, seed), task_id)]
                T = d["native_diag_normalized"] - d["independent_normalized"]
                R = d["native_diag_normalized"] - d["oracle_diag_normalized"]
                K = d["oracle_diag_normalized"] - d["independent_normalized"]
                check(close(T, R + K, 1e-12), f"T != R + K for {order_name}/{seed}/{task_id}")
                cells[(order_name, seed)][task_id] = {
                    "T": T, "R": R, "K": K, "position": d["position"],
                    "native": d["native_diag_normalized"], "oracle": d["oracle_diag_normalized"],
                    "independent": d["independent_normalized"]}
    by_order = {}
    for order_name in ORDERS:
        slots = [{"T": mean(c["T"] for c in cells[(order_name, seed)].values()),
                  "R": mean(c["R"] for c in cells[(order_name, seed)].values()),
                  "K": mean(c["K"] for c in cells[(order_name, seed)].values())} for seed in SEEDS]
        by_order[order_name] = {"T": mean(s["T"] for s in slots), "R": mean(s["R"] for s in slots),
                                "K": mean(s["K"] for s in slots), "per_seed": [slots[i] for i in range(8)]}
    per_task = {}
    for task_id in members:
        values = [cells[(order_name, seed)][task_id] for order_name in ORDERS for seed in SEEDS]
        per_task[task_id] = {"T": mean(v["T"] for v in values), "R": mean(v["R"] for v in values),
                             "K": mean(v["K"] for v in values),
                             "positions_seen": sorted({v["position"] for v in values})}
    total = {"T": mean(c["T"] for cell in cells.values() for c in cell.values()),
             "R": mean(c["R"] for cell in cells.values() for c in cell.values()),
             "K": mean(c["K"] for cell in cells.values() for c in cell.values())}
    check(close(total["T"], recomputed[family]["mean"], 1e-9),
          f"decomposed T != reported transfer mean for {family}")
    check(close(total["T"], total["R"] + total["K"], 1e-9), f"aggregate T != R + K for {family}")
    per_task_cells = {task_id: [{"order": order_name, "seed": seed,
                                 **{key: round(cells[(order_name, seed)][task_id][key], 12)
                                    for key in ("T", "R", "K")},
                                 **{key: cells[(order_name, seed)][task_id][key]
                                    for key in ("position", "native", "oracle", "independent")}}
                                for order_name in ORDERS for seed in SEEDS]
                      for task_id in members}
    return {"total": total, "by_order": by_order, "per_task": per_task,
            "per_task_cells": per_task_cells}


decomposition = {family: decomposition_for(family) for family in ("vqa", "caption")}

# Exact worst-case bound on how much R (native minus oracle at the diagonal stage) could
# come from own-routed rows whose two variant texts differ. For a row the per-example
# accuracy point delta is bounded by 100 points; its weight in the reported aggregation is
# 1/(EVAL_ROWS * normalizer) inside the task, task averaged over the family's 2 tasks,
# then averaged over the 32 (order, seed) cells. Counts come from the actual files.
own_mismatch_by_task = {task_id: sum(
    diag[(bank_name(order_name, seed), task_id)]["identity"]["own_route_text_mismatched_vs_oracle"]
    for order_name in ORDERS for seed in SEEDS) for task_id in ("V", "G")}
own_mismatch_bound = (1.0 / (32 * 2 * EVAL_ROWS)) * math.fsum(
    100.0 * own_mismatch_by_task[task_id] / one(NORMALIZER[task_id]) for task_id in ("V", "G"))
vqa_R_bound = {
    "what": "worst-case max |own-route contribution to R| over all 32 diagonal cells (no scorer execution)",
    "own_route_text_mismatch_rows_by_task": own_mismatch_by_task,
    "normalizers": {task_id: one(NORMALIZER[task_id]) for task_id in ("V", "G")},
    "rows_per_stage_task_eval": EVAL_ROWS,
    "aggregation_cells": 32,
    "bound": own_mismatch_bound,
    "R_interval_under_bound": [decomposition["vqa"]["total"]["R"] - own_mismatch_bound,
                               decomposition["vqa"]["total"]["R"] + own_mismatch_bound],
    "ratio_bound_to_absolute_transfer": own_mismatch_bound / abs(reported["vqa"]["mean"]),
    "caveats": [
        "not a confidence interval and carries no causal claim; per-row error signs and magnitudes stay unknown",
        "applies only to R (native minus oracle on the trained bank); the native-vs-independent subgroup "
        "contribution of the own group is not recoverable because the K side is a different comparison",
        "per-group absolute scores remain unavailable without per-example accuracy (never persisted)",
        "text-identical own rows contribute exactly 0 (deterministic scorer), so the bound uses only "
        "actually mismatching rows from the files, not a hardcoded count",
    ],
}

# ---------------------------------------------- 5. VQA diagonal partition per (task, order)
vqa_partition: dict[str, dict] = {}
for task_id in ("V", "G"):
    per_order = {}
    for order_name in ORDERS:
        order = tuple(order_name.split("_"))
        row_sets = [diag[(bank_name(order_name, seed), task_id)]["identity"] for seed in SEEDS]
        total_rows = sum(r["n"] for r in row_sets)
        aggregated = {key: sum(r[key] for r in row_sets) for key in
                      ("own_rows", "mis_routed_rows", "own_text_eq_independent",
                       "mis_text_eq_independent", "mis_text_eq_oracle")}
        mis_targets = Counter()
        own_route_mismatches = sum(diag[(bank_name(order_name, seed), task_id)]
                                   ["identity"]["own_route_text_mismatched_vs_oracle"] for seed in SEEDS)
        for seed in SEEDS:
            mis_targets.update(diag[(bank_name(order_name, seed), task_id)]["mis_targets"])
        normalizer = one(NORMALIZER[task_id])
        per_order[order_name] = {
            "position": order.index(task_id),
            "rows_observed": total_rows,
            "own_routed_rows": aggregated["own_rows"],
            "misrouted_rows": aggregated["mis_routed_rows"],
            "own_route_text_mismatch_rows": own_route_mismatches,
            "mis_target_rows": dict(mis_targets),
            "own_share": aggregated["own_rows"] / total_rows,
            "mis_share": aggregated["mis_routed_rows"] / total_rows,
            "own_text_eq_independent_rows": aggregated["own_text_eq_independent"],
            "mis_text_eq_independent_rows": aggregated["mis_text_eq_independent"],
            "mis_text_eq_oracle_rows": aggregated["mis_text_eq_oracle"],
            "per_row_max_delta_accuracy_points": 100.0,
            "own_contribution_bound_normalized": 100.0 * (aggregated["own_rows"]
                                                          - aggregated["own_text_eq_independent"]) / (total_rows and 512 * 8) / normalizer,
            "mis_contribution_bound_normalized": 100.0 * (aggregated["mis_routed_rows"]
                                                          - aggregated["mis_text_eq_independent"]) / (total_rows and 512 * 8) / normalizer,
            "native_diag_normalized_mean": mean([diag[(bank_name(order_name, s), task_id)]["native_diag_normalized"]
                                                 for s in SEEDS]),
            "oracle_diag_normalized_mean": mean([diag[(bank_name(order_name, s), task_id)]["oracle_diag_normalized"]
                                                 for s in SEEDS]),
            "independent_normalized_mean": mean([independent_metrics[(s, task_id)]["normalized"] for s in SEEDS]),
        }
    vqa_partition[task_id] = per_order

# ---------------------------------------------- 6. caption final-stage native vs oracle
caption_final: dict[tuple, dict] = {}
cross_family_rows_total = 0
for name, bank in sorted(banks.items()):
    order, seed, pos = bank["order"], bank["seed"], bank["pos"]
    native_raw = bank["run"]["variants"]["native_joint"]["matrix"][3]
    oracle_raw = bank["run"]["variants"]["task_id_oracle"]["matrix"][3]
    for task_id in ("C", "F"):
        p, p_other = pos[task_id], pos["F" if task_id == "C" else "C"]
        rows_native = TAR.jsonl(G5, f"{G5_ROOT}/{name}/native_joint/predictions_after_03_{task_id}.jsonl")
        rows_oracle = TAR.jsonl(G5, f"{G5_ROOT}/{name}/task_id_oracle/predictions_after_03_{task_id}.jsonl")
        check(len(rows_native) == len(rows_oracle) == EVAL_ROWS
              and [r["id"] for r in rows_native] == [r["id"] for r in rows_oracle],
              f"caption final pairing mismatch {name} {task_id}")
        n = len(rows_native)
        own = sum(1 for row in rows_native if row["expert"] == p)
        same_other = sum(1 for row in rows_native if row["expert"] == p_other)
        cross = sum(1 for row in rows_native if row["expert"] not in (p, p_other))
        cross_family_rows_total += cross
        own_text_mismatch = sum(1 for a, b in zip(rows_native, rows_oracle)
                                if a["expert"] == p and a["prediction"] != b["prediction"])
        identity_differ = same_other + cross
        # The selected-group mean paired delta is identified for a bank/task cell ONLY when
        # every own-route row has identical native/oracle text (0 mismatching rows). Cells
        # with any same-expert text difference keep counts and the whole-corpus delta but
        # their subgroup absolute scores and paired differences stay unidentifiable.
        identified = own_text_mismatch == 0 and identity_differ > 0
        s_native, s_oracle = one(native_raw[p]), one(oracle_raw[p])
        caption_final[(name, task_id)] = {
            "n": n, "own_rows": own, "same_family_other_rows": same_other, "cross_family_rows": cross,
            "own_route_text_mismatched_vs_oracle": own_text_mismatch,
            "group_paired_delta_identified": identified,
            "native_cider": s_native, "oracle_cider": s_oracle,
            "corpus_delta_native_minus_oracle": s_native - s_oracle,
            "fraction_route_differ": identity_differ / n,
            # per-image mean paired difference over the route-differ group, identified ONLY
            # when no own-route row differs textually between the variants; otherwise None.
            "selected_group_mean_paired_delta":
                (s_native - s_oracle) * n / identity_differ if identified else None,
            "route_differ_rows_text_equal_to_oracle_prediction":
                sum(1 for row, rowo in zip(rows_native, rows_oracle)
                    if row["expert"] != p and row["prediction"] == rowo["prediction"]),
        }

caption_task_orders: dict[str, dict] = {}
for task_id in ("C", "F"):
    normalizer = one(NORMALIZER[task_id])
    per_order = {}
    for order_name in ORDERS:
        cells = [caption_final[(bank_name(order_name, seed), task_id)] for seed in SEEDS]
        key_join = lambda key: [cell[key] for cell in cells]  # noqa: E731
        per_order[order_name] = {
            "position": order_name.split("_").index(task_id),
            "rows_observed": sum(key_join("n")),
            "own_routed_rows": sum(key_join("own_rows")),
            "same_family_other_rows": sum(key_join("same_family_other_rows")),
            "cross_family_rows": sum(key_join("cross_family_rows")),
            "own_route_text_mismatch_rows": sum(key_join("own_route_text_mismatched_vs_oracle")),
            "route_differ_text_eq_oracle_rows": sum(key_join("route_differ_rows_text_equal_to_oracle_prediction")),
            "eligible_seed_count": sum(1 for cell in cells if cell["group_paired_delta_identified"]),
            "unidentifiable_seed_count": sum(1 for cell in cells if not cell["group_paired_delta_identified"]),
            "identified_mean_paired_delta_eligible_only":
                mean([cell["selected_group_mean_paired_delta"] for cell in cells
                      if cell["group_paired_delta_identified"]])
                if any(cell["group_paired_delta_identified"] for cell in cells) else None,
            "identified_rows_eligible": sum(cell["n"] for cell in cells
                                            if cell["group_paired_delta_identified"]),
            "own_share_mean": mean(c["own_rows"] / c["n"] for c in cells),
            "route_differ_share_mean": mean(
                (c["same_family_other_rows"] + c["cross_family_rows"]) / c["n"] for c in cells),
            "native_cider_mean_over_seeds": mean(key_join("native_cider")),
            "oracle_cider_mean_over_seeds": mean(key_join("oracle_cider")),
            "corpus_delta_native_minus_oracle_mean": mean(key_join("corpus_delta_native_minus_oracle")),
            "per_seed": [{"seed": seed,
                          **{key: caption_final[(bank_name(order_name, seed), task_id)][key] for key in
                             ("n", "own_rows", "same_family_other_rows", "cross_family_rows",
                              "own_route_text_mismatched_vs_oracle", "group_paired_delta_identified",
                              "native_cider", "oracle_cider", "corpus_delta_native_minus_oracle",
                              "selected_group_mean_paired_delta",
                              "route_differ_rows_text_equal_to_oracle_prediction")}}
                         for seed in SEEDS],
        }
    caption_task_orders[task_id] = {
        "normalizer": normalizer,
        "per_order": per_order,
        "all_orders": {
            "native_cider_mean_over_orders_and_seeds": mean(
                [v["native_cider_mean_over_seeds"] for v in per_order.values()]),
            "oracle_cider_mean_over_orders_and_seeds": mean(
                [v["oracle_cider_mean_over_seeds"] for v in per_order.values()]),
            "corpus_delta_native_minus_oracle_mean": mean(
                [v["corpus_delta_native_minus_oracle_mean"] for v in per_order.values()]),
            "eligible_cells_total": sum(v["eligible_seed_count"] for v in per_order.values()),
            "unidentifiable_cells_total": sum(v["unidentifiable_seed_count"] for v in per_order.values()),
            "identified_mean_paired_delta_eligible_only":
                mean([cell["selected_group_mean_paired_delta"] for cell in
                      [caption_final[(bank_name(order_name, seed), task_id)]
                       for order_name in ORDERS for seed in SEEDS
                       if caption_final[(bank_name(order_name, seed), task_id)]["group_paired_delta_identified"]]])
                if any(caption_final[(bank_name(order_name, seed), task_id)]["group_paired_delta_identified"]
                       for order_name in ORDERS for seed in SEEDS) else None,
            "ownership_note": "see per-order eligible coverage; pool-level subgroup stats are unidentifiable",
            "route_differ_share_mean": mean([v["route_differ_share_mean"] for v in per_order.values()]),
            "cross_family_rows_total": sum(v["cross_family_rows"] for v in per_order.values()),
        },
    }

caption_family_norm: dict[str, dict] = {}
for variant in ("native_joint", "task_id_oracle"):
    per_order = {}
    for order_name in ORDERS:
        cells = []
        for seed in SEEDS:
            bank = banks[bank_name(order_name, seed)]
            matrix_norm = bank["run"]["variants"][variant]["matrix_normalized"]
            cells.append(mean(one(matrix_norm[3][bank["pos"][task_id]]) for task_id in ("C", "F")))
        per_order[order_name] = mean(cells)
    caption_family_norm[variant] = {"per_order": per_order, "overall": mean(per_order.values())}
check(close(caption_family_norm["native_joint"]["overall"],
            one(g5_summary["variants"]["native_joint"]["family_normalized_final"]["caption"]), 1e-9),
      "caption family native final does not reconcile with shipped G5 summary")
check(close(caption_family_norm["task_id_oracle"]["overall"],
            one(g5_summary["variants"]["task_id_oracle"]["family_normalized_final"]["caption"]), 1e-9),
      "caption family oracle final does not reconcile with shipped G5 summary")

# ---------------------------------------------- 7. own-expert position invariance (empirical)
position_invariance: dict[str, dict] = {}
for task_id in TASKS:
    slots: dict[tuple, list] = defaultdict(list)
    slots_native: dict[tuple, list] = defaultdict(list)
    for name, bank in banks.items():
        p = bank["pos"][task_id]
        slots[(bank["seed"], p)].append(one(bank["run"]["variants"]["task_id_oracle"]["matrix"][p][p]))
        slots_native[(bank["seed"], p)].append(one(bank["run"]["variants"]["native_joint"]["matrix"][p][p]))
    detail = {f"seed_{seed}_position_{p}": {
        "n_banks": len(values),
        "oracle_diag_distinct_values": len(set(values)),
        "oracle_diag_value": values[0] if len(set(values)) == 1 else None,
        "native_diag_distinct_values": len(set(slots_native[(seed, p)])),
    } for (seed, p), values in sorted(slots.items())}
    position_invariance[task_id] = {
        "oracle_diag_bitwise_position_invariant": all(len(set(v)) == 1 for v in slots.values()),
        "native_diag_varies_within_slot_sometimes": any(len(set(v)) > 1 for v in slots_native.values()),
        "slots": detail,
    }

# ---------------------------------------------- 8. first task equals independent run (byte check)
first_task = []
for name, bank in sorted(banks.items()):
    order, seed = bank["order"], bank["seed"]
    task_id = order[0]
    rows_expand = TAR.jsonl(G4, f"{G4_ROOT}/{name}/predictions_after_00_{task_id}.jsonl")
    rows_ind = TAR.jsonl(G4, f"{G4_ROOT}/{task_id}_independent_lr0.0003_ep5_seed{seed}"
                             f"/predictions_after_00_{task_id}.jsonl")
    identical = len(rows_expand) == len(rows_ind) and all(
        a["prediction"] == b["prediction"] and a["expert"] == b["expert"]
        for a, b in zip(rows_expand, rows_ind))
    row = trial_rows[(order, seed, task_id)]
    first_task.append({"bank": name, "task_id": task_id,
                       "predictions_and_experts_identical": identical,
                       "transfer_delta_exactly_zero": one(row["native_delta"]) == 0.0})
check(all(entry["transfer_delta_exactly_zero"] for entry in first_task),
      "first-task transfer delta is not exactly zero")
check(all(entry["predictions_and_experts_identical"] for entry in first_task),
      "first-task predictions are not byte-identical to the independent run")

# ---------------------------------------------- 9. code verification (item 2)
CODE_FILES = ["sema_multimodal/run.py", "sema_multimodal/data.py", "sema_multimodal/model.py",
              "sema_multimodal/metrics.py", "sema_vqa/readout_model.py", "sema_vqa/readout_router.py",
              "sema_vqa/vision_model.py", "sema_vqa/expansion.py"]


def cite(relpath: str, pattern: str) -> str:
    path = ROOT / relpath
    found = None
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if pattern in line:
            found = index
    if found is None:
        raise ValueError(f"pattern {pattern!r} not in {relpath}")
    code_assertions.append({"file": relpath, "line": found, "text": path.read_text().splitlines()[found - 1].strip()[:160]})
    return f"{relpath}:{found}"


code_assertions: list[dict] = []

citations = {
    "zero_delta": [cite("sema_vqa/readout_model.py", "self.B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))"),
                   cite("sema_vqa/readout_model.py", "nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))"),
                   cite("sema_vqa/readout_model.py", "def add_expert(self, copy_from=None):"),
                   cite("sema_multimodal/run.py", "expert = model.add_expert()")],
    "freezing": [cite("sema_vqa/readout_model.py", "def _freeze_all(self):"),
                 cite("sema_vqa/readout_model.py", "def train_expert(self, index):"),
                 cite("sema_vqa/readout_model.py", 'raise ValueError("Old experts are permanently frozen after expansion")'),
                 cite("sema_multimodal/run.py", "if any(parameter.grad is not None for parameter in frozen):")],
    "no_training_dropout": [cite("sema_vqa/readout_model.py", "self.visual_encoder.eval()"),
                            cite("sema_vqa/readout_model.py", "self.text_encoder.eval()"),
                            cite("sema_vqa/readout_model.py", "self.text_decoder.eval()")],
    "rng_streams": [cite("sema_multimodal/run.py", 'phase_seed = args.seed if args.method == "independent" else args.seed + index * 10007'),
                    cite("sema_multimodal/run.py", "rng = random.Random(seed + task * 1000 + epoch)"),
                    cite("sema_multimodal/run.py", "batch = training_rows(batch, seed=args.seed, epoch=epoch)"),
                    cite("sema_multimodal/data.py", 'offset = int.from_bytes(hashlib.sha256(f"{seed}/{image_identity(row)}".encode()).digest(), "big")')],
    "router_path": [cite("sema_multimodal/run.py", "router.observe(features(model, rows, args), state[\"assignments\"][index], model.num_experts)"),
                    cite("sema_vqa/readout_router.py", "def fit(self, num_experts, device):"),
                    cite("sema_vqa/readout_router.py", "torch.randperm(len(features), generator=generator)[:self.memory_per_task]"),
                    cite("sema_multimodal/data.py", "def public_rows(rows):")],
    "optimizer_reset": [cite("sema_multimodal/run.py", "def make_optimizer():"),
                        cite("sema_multimodal/run.py", "optimizer, scheduler = make_optimizer()")],
}

sample_config = json.loads(TAR.read(G4, f"{G4_ROOT}/{bank_name('V_C_G_F', 137)}/config.json"))
source_parity = [{"file": relpath, "match": sha256_file(ROOT / relpath) == recorded}
                 for relpath, recorded in sample_config["source_sha256"].items()]
check(all(entry["match"] for entry in source_parity),
      "current repo source files differ from recorded run-time hashes")

config_fields = ("eval_split", "lr", "epochs", "batch_size", "micro_batch_size", "image_size",
                 "rank", "alpha", "router_modality", "router_memory", "router_ridge", "replay_per_task",
                 "manifest_sha256")
expand_config_values = {field: {json.loads(TAR.read(G4, f"{G4_ROOT}/{bank_name(order_name, seed)}/config.json"))[field]
                                for order_name in ORDERS for seed in SEEDS}
                        for field in config_fields}
independent_config_values = {field: {json.loads(TAR.read(G4, f"{G4_ROOT}/{task_id}_independent_lr0.0003_ep5_seed{seed}/config.json"))[field]
                                     for task_id in TASKS for seed in SEEDS}
                             for field in config_fields}
for field in config_fields:
    check(expand_config_values[field] == independent_config_values[field] == {sample_config[field]},
          f"config drift on {field}: expand={expand_config_values[field]} independent={independent_config_values[field]}")
check({json.loads(TAR.read(G4, f"{G4_ROOT}/{bank_name(order_name, seed)}/config.json"))["seed"]
       for seed in SEEDS for order_name in ORDERS} == set(SEEDS), "expand runs do not cover the confirmation seeds")

trainable_parity = []
for order_name in ORDERS:
    for seed in SEEDS:
        result = G4_RESULTS[(order_name, seed, "expand")]
        for stage, resources in enumerate(result["stage_resources"]):
            counts = resources["parameter_counts"]
            trainable_parity.append((stage, counts["active_trainable"], counts["trainable"]))
for stage, active, trainable in trainable_parity:
    check(active == trainable, f"trainable != active_trainable at stage {stage}")

# ---------------------------------------------- 10. writes
now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def input_record(path: Path, role: str) -> dict:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "role": role}


code_conditions = {
    "schema_version": 1,
    "generated_by": "scripts/routing_review_items27.py",
    "generated_at_utc": now,
    "status": "pass" if not failures else "fail",
    "failures": failures,
    "claims": [
        {
            "claim": "New experts initialize zero-delta and are never copied from prior experts.",
            "verdict": "verified_with_rng_caveat",
            "evidence": citations["zero_delta"],
            "detail": ("LoRAExpert sets B=zeros and A=kaiming_uniform_(a=sqrt(5)); forward is x -> A -> B so the "
                       "appended expert's delta output is exactly zero at init; add_expert() is called with no "
                       "copy_from in both methods, and the independent method rebuilds the model from base weights. "
                       "Caveat: the A matrices are drawn from the global torch RNG, so bitwise-identical experts "
                       "require the same (seed, stream position); distribution and protocol are identical. "
                       "Empirically confirmed below: position-matched banks produce bitwise-equal own-expert "
                       "diagonal scores, and first-task (position 0) streams are byte-identical to independent runs."),
        },
        {
            "claim": "Backbone and old experts are frozen; only the newest expert is trainable; optimizer state is per-phase.",
            "verdict": "verified",
            "evidence": citations["freezing"] + citations["optimizer_reset"],
            "detail": ("add_expert/train_expert call _freeze_all (requires_grad_(False) plus grad clears); "
                       "train_expert refuses any index except the newest; optimizer_step aborts if any frozen "
                       "parameter acquires a gradient; make_optimizer receives only requires_grad parameters and is "
                       "rebuilt fresh for every phase, so optimizer moments never carry across tasks or methods. "
                       "Run-time stage_resources show trainable == active_trainable at every stage (newest expert only)."),
        },
        {
            "claim": "Training is dropout-free in both methods.",
            "verdict": "verified",
            "evidence": citations["no_training_dropout"],
            "detail": ("ReadoutMoEVQA.train() keeps visual_encoder/text_encoder/text_decoder in eval() unconditionally; "
                       "LoRA experts contain no dropout; the frozen zero-output vision adapters are not grown by "
                       "add_expert (only text query/value projections). So phase training consumes torch RNG only "
                       "through the expert A-init draws."),
        },
        {
            "claim": "Fresh-expert training is independent-equivalent in the sense used for the transfer endpoint.",
            "verdict": "protocol_equivalent_not_bitwise; two seeded differences remain possible",
            "evidence": citations["rng_streams"] + citations["no_training_dropout"],
            "differences": [
                {"id": "expert_A_init_draw_stream",
                 "independent": "torch.manual_seed(args.seed) before add_expert at every phase",
                 "expansion": "torch.manual_seed(args.seed + index*10007) before add_expert",
                 "consequence": "different kaiming draws (bitwise) for stream index > 0; same distribution",
                 "citations": ["sema_multimodal/run.py phase_seed", "sema_vqa/readout_model.py LoRAExpert.__init__"]},
                {"id": "batch_shuffling_seed",
                 "independent": "epoch_batches rng seed = args.seed + 0*1000 + epoch",
                 "expansion": "epoch_batches rng seed = args.seed + index*1000 + epoch",
                 "consequence": "different shuffle of the same 2048-image corpus for index > 0",
                 "citations": ["sema_multimodal/run.py epoch_batches call"]},
                {"id": "caption_reference_cycling",
                 "independent": "hash(seed/image)-anchored cycle; identical formula/inputs",
                 "expansion": "identical",
                 "consequence": "none: per-image reference selection identical per epoch",
                 "citations": ["sema_multimodal/data.py training_rows"]},
                {"id": "frozen_sibling_experts_and_router",
                 "independent": "single-expert model, no router",
                 "expansion": "frozen siblings numerically inert and no router in phase training "
                              "(observe happens at phase END, features from LoRA-disabled frozen backbone)",
                 "consequence": "no numerical effect on the phase's own gradients",
                 "citations": ["sema_multimodal/run.py observe site", "sema_multimodal/citations router_path"]},
            ],
            "empirical_support": {
                "first_task_rows_byte_identical_to_independent": sum(p["predictions_and_experts_identical"] for p in first_task),
                "first_task_banks_checked": len(first_task),
                "first_task_transfer_delta_exactly_zero": sum(p["transfer_delta_exactly_zero"] for p in first_task),
                "own_expert_diag_position_invariance": {t: position_invariance[t]["oracle_diag_bitwise_position_invariant"]
                                                        for t in TASKS},
            },
        },
        {
            "claim": "The router is the only cross-task knowledge path; it never influences phase training.",
            "verdict": "verified_at_code_level",
            "evidence": citations["router_path"] + citations["optimizer_reset"],
            "detail": ("Experts occupy disjoint ModuleList slots and only the active expert enters the forward; the "
                       "gradient guard allows updates only for the current expert. The FeatureRouter refits its ridge "
                       "boundary at the END of each phase from per-task frozen-feature samples (128 per task, "
                       "generator-seeded randperm; equal per-task weighting) and at inference only selects an expert "
                       "per row from image/question features; routing_features disables LoRA and reads no "
                       "answer/task/family metadata. Hence the new task's expert is trained without router or prior "
                       "expert influence: cross-task influence is limited to the shared frozen backbone (pretrained) "
                       "and to eval-time routing of previously-learned and current tasks."),
        },
    ],
    "claimed_code_parity": {"files_checked": len(source_parity),
                            "all_match": all(entry["match"] for entry in source_parity),
                            "mismatches": [entry for entry in source_parity if not entry["match"]]},
    "config_drift_check": {"fields": list(config_fields),
                           "expand_vs_independent_identical":
                               all(expand_config_values[f] == independent_config_values[f] for f in config_fields)},
    "per_phase_trainable_parity": {"stage_entries": len(trainable_parity),
                                   "trainable_equals_active_everywhere": True},
    "code_assertions_quoted": code_assertions,
    "limitations": [
        "Code quotes refer to the current repo; SHA256 parity against the recorded run-time source_sha256 confirms "
        "the quoted code is the code that produced G4/G5.",
        "[INFERENCE] Because the two seeded differences (A-init draws, batch shuffle seed) are the only remaining "
        "divergences and first-task streams are byte-identical to independent runs, the transfer endpoint's "
        "oracle-vs-independent residual (K) reflects these seeded differences plus any non-bitwise training-path "
        "effects, not parameter-level cross-task knowledge (older experts are frozen and inert).",
    ],
}

routing_partition = {
    "schema_version": 1,
    "generated_by": "scripts/routing_review_items27.py",
    "generated_at_utc": now,
    "status": "pass" if not failures else "fail",
    "failures": failures,
    "aggregation_weighting": {
        "definition": ("original: per (order, seed, task) normalized_delta = (post_train - independent)/normalizer; "
                       "family slot = mean over the family's two tasks; per-seed = mean over the four orders; "
                       "reported mean = mean over the 8 confirmation seeds "
                       "(scripts/evaluate_multimodal_confirmation.py, transfer_records/transfer_statistics)"),
        "reproduction": {
            "vqa_reported_mean": reported["vqa"]["mean"],
            "vqa_recomputed_mean": recomputed["vqa"]["mean"],
            "vqa_reported_per_seed": reported["vqa"]["per_seed"],
            "vqa_recomputed_per_seed": recomputed["vqa"]["per_seed"],
            "caption_reported_mean": reported["caption"]["mean"],
            "caption_recomputed_mean": recomputed["caption"]["mean"],
            "caption_reported_per_seed": reported["caption"]["per_seed"],
            "caption_recomputed_per_seed": recomputed["caption"]["per_seed"],
        },
    },
    "decomposition_definition": {
        "T": "native_diag_normalized - independent_normalized (the reported normalized transfer endpoint)",
        "R": "native_diag_normalized - oracle_diag_normalized (routing counterfactual: cost of native routing vs own task_id_oracle routing at the diagonal stage, same trained bank)",
        "K": "oracle_diag_normalized - independent_normalized (training-path difference when routing is held at the own expert)",
        "identity": "T == R + K holds per bank/task (all three scores from the same eval corpus; verified numerically)",
        "score_sources": ("native/oracle: artifacts/multimodal-g5-report/routing_audit.json variant matrices "
                          "(previously rescored); independent: artifacts/multimodal-g4-report/audit.json "
                          "(independent trials; post_train/normalizer); NO scorer executed here"),
    },
    "vqa_transfer": {"reported_mean": reported["vqa"]["mean"], "decomposed": decomposition["vqa"],
                     "R_own_route_worstcase_bound": vqa_R_bound},
    "caption_transfer": decomposition["caption"],
    "routing_counterfactual_extent": {
        "definition": ("share of misrouted rows whose own-expert (task_id_oracle) prediction text is byte-equal to "
                       "the native prediction text; those rows' routing choice is inconsequential for the answer"),
        "vqa_diag_cells_route_differ_rows": sum(
            cell["identity"]["mis_routed_rows"] for cell in diag.values() if cell["family"] == "vqa"),
        "vqa_diag_cells_route_differ_text_equal_rows": sum(
            cell["identity"]["mis_text_eq_oracle"] for cell in diag.values() if cell["family"] == "vqa"),
        "caption_diag_route_differ_rows": sum(
            cell["identity"]["mis_routed_rows"] for cell in diag.values() if cell["family"] == "caption"),
        "caption_diag_route_differ_text_equal_rows": sum(
            cell["identity"]["mis_text_eq_oracle"] for cell in diag.values() if cell["family"] == "caption"),
        "caption_final_route_differ_rows": sum(
            cell["same_family_other_rows"] + cell["cross_family_rows"] for cell in caption_final.values()),
        "caption_final_route_differ_text_equal_rows": sum(
            cell["route_differ_rows_text_equal_to_oracle_prediction"] for cell in caption_final.values()),
    },
    "independent_baseline_raw": {
        task_id: {"mean_over_seeds": mean([independent_metrics[(seed, task_id)]["score"] for seed in SEEDS]),
                  "normalizer_g1_validation": one(NORMALIZER[task_id]),
                  "per_seed": [independent_metrics[(seed, task_id)]["score"] for seed in SEEDS]}
        for task_id in TASKS
    },
    "interpretation": {
        "vqa_transfer": ("T = -0.04324 splits exactly into R = -0.02087 (native routing vs own-expert routing at the "
                         "diagonal stage, the routing counterfactual) and K = -0.02237 (the stream-trained expert, "
                         "perfectly routed, still below the same-seed independent run). Code review rules out any "
                         "parameter-level cross-task source for K (frozen siblings and router are inert during the "
                         "phase; position-0 tasks have T == R == K == 0 exactly because the first-task stream is "
                         "byte-identical to the independent run). The remaining cause(s) of nonzero K at positions "
                         "> 0 are unverified; the seeded protocol difference (different A-init draw streams and "
                         "batch-shuffle seeds) is one unproven hypothesis, and no causal RNG attribution or "
                         "'learned cross-task knowledge' reading is established."),
        "caption_transfer": ("T = +0.02015 = R +0.01586 + K +0.00429. Per task the decomposition is strongly "
                             "heterogeneous: C is T=-0.0521 (routing deficit R=-0.0511 dominates) while F is "
                             "T=+0.0924 (R=+0.0828: the oracle-forced F expert differs from native routing and the "
                             "corpus favors native). The caption aggregate hides this sign structure. [INFERENCE, "
                             "unverified] the F component is measured against an independent F reference that "
                             "scores close to its fixed normalizer (0.0999 vs 0.1526 mean CIDEr over seeds), so "
                             "the positive F transfer may partly reflect a weak independent reference rather than "
                             "new-task learning."),
        "caption_final_stage": ("Whole-corpus: task_id_oracle caption family final (0.7114) is below native joint "
                                "(0.7434), and per task oracle beats native for C while losing for F. Per-cell "
                                "subgroup paired deltas are identified only for eligible cells (0 same-expert text "
                                "mismatches): per-order eligible seed counts, corpus deltas and counts are in "
                                "per_task/per_order; pool-level subgroup statistics are unidentifiable whenever "
                                "any own-route row differs. No cross-family caption routes exist (0 rows across all "
                                "32 banks), so the route-differ set equals the same-family-other-expert partition "
                                "exactly."),
    },
    "vqa_diagonal_partition": {
        "definition": ("all rows of the diagonal-stage VQA prediction files; 'own' = row routed to the task's own "
                       "expert (assignment index == position in stream); 'mis' = any other expert; text identity "
                       "against the same-seed independent run and the G5 own-expert counterfactual"),
        "identification_note": ("rows whose native prediction text equals the independent prediction contribute "
                                "exactly 0 to the transfer delta (deterministic scorer on identical text). Cautions: "
                                "(a) rows routed to their own expert are not guaranteed to produce identical text in "
                                "the native and oracle variant files - same-expert text mismatches are counted per "
                                "task x order below and their source is unknown (one unverified hypothesis is "
                                "micro-batch composition effects on bf16 greedy near-ties); (b) per-example accuracy "
                                "is not persisted anywhere, so the magnitude split inside each group is not "
                                "identifiable without re-running the scorer - only the max-magnitude bounds below "
                                "and the exact corpus-level R/K split above are available"),
        "per_row_contribution_bound_points": "abs per-example accuracy delta <= 100 points",
        "per_task": {task_id: {"per_order": vqa_partition[task_id],
                               "overall_own_share": mean(o["own_share"] for o in vqa_partition[task_id].values()),
                               "overall_mis_share": mean(o["mis_share"] for o in vqa_partition[task_id].values())}
                     for task_id in ("V", "G")},
    },
    "caption_final_stage": {
        "definition": ("final stream stage; per C/F row the native expert is own-task expert, same-family other "
                       "caption expert, or cross-family (validated); native/oracle corpus CIDEr-D come from the G5 "
                       "audit; whole-corpus native-minus-oracle deltas and counts are reported for every C/F x "
                       "order x seed. The selected-group mean paired delta is IDENTIFIED ONLY for bank/task cells "
                       "with zero own-route text mismatches (reported 'group_paired_delta_identified'); for any "
                       "cell with a same-expert text difference the whole-cohort subgroup absolute scores and "
                       "paired differences stay unidentifiable. Own-route rows are predictions of the same expert "
                       "weights in both variants, but equal outputs are an observation made per bank, not a "
                       "guaranteed identity (source of observed mismatches unknown)."),
        "cross_family_rows_total": cross_family_rows_total,
        "per_task": caption_task_orders,
        "own_route_generation_numerics": {
            "caption_final_own_route_text_mismatch_rows": sum(
                cell["own_route_text_mismatched_vs_oracle"] for cell in caption_final.values()),
            "caption_final_own_route_rows": sum(cell["own_rows"] for cell in caption_final.values()),
            "all_diag_cells_own_route_text_mismatch_rows": total_own_route_text_mismatch,
            "all_diag_cells_own_route_rows": sum(c["identity"]["own_rows"] for c in diag.values()),
            "vqa_diag_cells_own_route_text_mismatch_rows": sum(
                c["identity"]["own_route_text_mismatched_vs_oracle"] for key, c in diag.items()
                if key[1] in ("V", "G")),
            "caption_diag_cells_own_route_text_mismatch_rows": sum(
                c["identity"]["own_route_text_mismatched_vs_oracle"] for key, c in diag.items()
                if key[1] in ("C", "F")),
            "explanation": ("two generation runs of the same trained bank on the same expert and the same rows are "
                            "not bitwise-guaranteed; the source of the observed same-expert text differences is "
                            "unknown. One unverified hypothesis is micro-batch-composition-dependent bf16 rounding "
                            "of near-tied greedy tokens (readout_model.py generate(), run.py routed_predictions "
                            "grouping); counts are exact, causal attribution is not established."),
            "consequence": "same-expert rows may contribute nonzero (c_native - c_oracle) even when they are routed "
                           "to their own expert in both variants; pool-level subgroup statistics are therefore "
                           "unidentifiable whenever any same-expert row differs, and the exact selected-group mean "
                           "is retained only for eligible cells (0 own-route text mismatches).",
        },
        "family_normalized_final": {
            "native_joint_per_order": caption_family_norm["native_joint"]["per_order"],
            "native_joint_overall": caption_family_norm["native_joint"]["overall"],
            "task_id_oracle_per_order": caption_family_norm["task_id_oracle"]["per_order"],
            "task_id_oracle_overall": caption_family_norm["task_id_oracle"]["overall"],
            "not_equal_to_g4_final_stage": "final-stage values are diagonal-free summaries of the shipped G5 audit",
        },
        "unidentifiable_without_new_scoring": [
            "per-group absolute native CIDEr-D mean (mean of per-image native contributions over the group)",
            "per-group absolute oracle CIDEr-D mean",
            "per-image CIDEr-D contributions",
            "paired-difference distributions within the route-differ group",
            "pool-level (whole-task) subgroup means and paired differences whenever any own-route row of the pool "
            "differs textually between the variants",
            "decomposition of a same-expert text mismatch into native-vs-oracle score gaps on that row",
        ],
        "reachable_substitute": ("exact selected-group mean paired difference ONLY for eligible cells (0 own-route "
                                 "text mismatches; eligible seed count reported per task x order), plus group "
                                 "counts, mismatch counts and prediction text-identity counts everywhere"),
        "selection_bias_note": ("the route-differ subset is chosen endogenously by the router from item features and "
                                "caption-instruction form; native would have routed these items elsewhere anyway; "
                                "the reported paired difference (eligible cells only) is a descriptive counterfactual "
                                "swap statistic for the selected items, not a treatment effect over a random "
                                "sample; wording such as 'other-expert benefit' is not implied."),
    },
    "cider_per_image_statement": {
        "per_image_contributions_saved": False,
        "what_exists": ("TaskScorer.score (sema_multimodal/metrics.py) returns corpus-level {score, accuracy} for "
                        "VQA tasks and {score, cider_d, bleu4} for caption tasks; _fixed_cider_score computes a "
                        "per-image score list internally but returns only its mean; the audit archives store "
                        "per-file corpus scores only; prediction JSONL rows contain id/prediction/expert/generation"),
        "missing_fields_for_absolutegroup_means": ["per-image CIDEr-D contribution per saved prediction",
                                                   "per-example VQAv2/GQA accuracy"],
        "consequence": ("additive fixed-IDF structure is confirmed in code, so caption paired-difference group means "
                        "are exactly computable from corpus scores at fixed IDF; absolute per-group means are not"),
    },
    "parity_checks": {
        "g4_audit_vs_raw_results_matrixes": parity_checked,
        "g5_native_diag_byte_parity_vs_g4_pairs": len(native_diag_byte_checks),
        "g5_native_diag_raw_equals_g4_post_train": "verified per bank/task",
        "caption_eval_size_and_id_order": "verified per bank/task (512 rows)",
        "caption_own_route_rows_byte_equal_native_vs_oracle": "observed per bank/task; mismatches counted "
                                                              "(41 of 26,276 caption-final own-route rows)",
        "caption_family_final_reconciles_with_g5_summary": caption_family_norm["native_joint"]["overall"] ==
                                                            one(g5_summary["variants"]["native_joint"]["family_normalized_final"]["caption"]),
    },
    "first_task_detail": first_task,
    "position_invariance": position_invariance,
    "inputs": [input_record(path, role) for path, role in INPUTS],
    "note_caption_units": ("caption rows are CIDEr-D points of the given task (fixed train-IDF); caption family "
                           "normalized values use the fixed G1 normalizers, matching the document's aggregation"),
    "relation_to_document": {
        "sections_inspected": ["1.2", "3.3", "3.5", "4.4", "4.5", "5.1"],
        "confirmations": [
            "zero-delta / no-copy initialization claim of section 1.2 (verified at code level, RNG caveat recorded)",
            "freezing and per-phase optimizer reordering claims of section 1.2/3.3",
            "the exact reported transfer values of section 4.4 (-0.0432397 VQA, +0.0201529 caption) reproduce from the "
            "shipped audits under the documented weighting",
            "section 4.5 observation that task-ID-oracle routing is not an upper bound for caption is confirmed "
            "quantitatively (family final native 0.7434 > oracle 0.7114; oracle loses on F, wins on C)",
        ],
        "conflicts_or_nuances": [
            ("The caption family positive transfer (+0.0202) in section 4.4 is driven by F only (+0.0924) while C "
             "is negative (-0.0521). The independent F reference (0.0999 CIDEr mean over seeds vs normalizer "
             "0.1526) is low in absolute terms; [INFERENCE, unverified] part of the positive F transfer may reflect "
             "this weak reference rather than new-task learning. No characterization beyond these numbers is "
             "claimed."),
            ("The VQA transfer deficit (-0.0432) is roughly half a routing counterfactual (-0.0209 native-vs-own "
             "misrouting at the diagonal stage) and half a training-path difference with routing held perfect "
             "(-0.0224). The latter has no parameter-level cross-task source per code review; its remaining cause "
             "is unverified, so section 4.4's transfer number should be quoted alongside this decomposition rather "
             "than attributed."),
            ("At the earliest stages the diagonal routing is fully own-expert (position-0 and position-1 VQA rows: "
             "100% own); the visible misrouting appears from stage 2 onward and targets same-family experts almost "
             "exclusively (2 cross-family VQA rows across all diagonal cells; 0 for caption)."),
        ],
    },
    "limitations": [
        "No scorer or model execution was performed; every delta aggregates the previously rescored corpus scores "
        "already stored in the G4/G5 audit artifacts plus deterministic text/assignment predicates.",
        "Per-example VQA accuracy and per-image CIDEr-D contributions were never persisted, so within-group "
        "magnitude splits of the transfer are not identifiable; only the exact corpus-level decomposition "
        "(T = R + K), counts, identities and max-magnitude bounds are reported.",
        "R (routing counterfactual) and K (training-path difference) are defined against the same-seed independent "
        "runs of the confirmation grid; the 4 fixed orders x 8 seeds only, same as the reported endpoint.",
        "Same-expert rows can produce different text in the native and oracle variant files (counts reported; "
        "source unknown), so pool-level subgroup absolute scores and paired differences are unidentifiable; "
        "selected-group paired deltas are reported only for eligible cells (0 same-expert text mismatches) with "
        "their eligible seed counts per task x order.",
        "Text-identity zero-delta logic is exact for a deterministic scorer regardless (identical text always scores "
        "identically); it cannot recover the sign of score changes for text-differing rows.",
    ],

    "generated_note": "reproducible: python3 scripts/routing_review_items27.py",
}

for path, payload in ((ROOT / "artifacts/routing-review-code-conditions.json", code_conditions),
                      (ROOT / "artifacts/routing-review-routing-partition.json", routing_partition),):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)

TAR.close()
print("wrote artifacts/routing-review-code-conditions.json")
print("wrote artifacts/routing-review-routing-partition.json")
print(json.dumps({
    "failures": failures,
    "vqa_reproduced": recomputed["vqa"]["mean"],
    "caption_reproduced": recomputed["caption"]["mean"],
    "vqa_decomposed": decomposition["vqa"]["total"],
    "caption_decomposed": decomposition["caption"]["total"],
    "position_invariance": {t: position_invariance[t]["oracle_diag_bitwise_position_invariant"] for t in TASKS},
    "first_task_identical_pairs": sum(p["predictions_and_experts_identical"] for p in first_task),
}, indent=1))
