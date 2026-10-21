"""Local verification harness for week-1 1-C outputs (run after fetching from DC1).

Checks the rescore envelope against the recorded review numbers and the strict
tolerance contract, and prints the numbers the parent needs for integration.
"""

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = json.loads((ROOT / "artifacts/week1-per-example-rescore.json").read_text())
REVIEW = json.loads((ROOT / "artifacts/routing-review-routing-partition.json").read_text())

fail = []
def need(condition, message):
    if not condition:
        fail.append(message)

need(ENV["status"] == "pass", f"envelope status {ENV['status']}")
need(not ENV["failures"], f"envelope failures {ENV['failures'][:6]}")
need(ENV["stored_parity_maxima"]["fresh_vs_stored_max"] <= 1e-9,
     f"stored parity max {ENV['stored_parity_maxima']['fresh_vs_stored_max']}")
need(ENV["stored_parity_maxima"]["stored_mirror_max"] <= 1e-12,
     f"mirror parity max {ENV['stored_parity_maxima']['stored_mirror_max']}")
need(ENV["native_parity_g5_vs_g4"]["files_row_identical"] ==
     ENV["native_parity_g5_vs_g4"]["files_checked"] == 192,
     "g5/g4 native parity files")
need(all(not rep["drift"] for rep in ENV["scorer_locks_parity"].values()), "lock drift")
need(ENV["static_import_scan"]["torch_free"], "torch import scan")
need(ENV["grid_coverage"]["unique_corpora"] == 416, "unique corpus count")

rows = 0
with gzip.open(ROOT / "artifacts/week1-per-example-scores.jsonl.gz", "rt", encoding="utf-8") as handle:
    for _ in handle:
        rows += 1
need(rows == ENV["rows_written"] == 416 * 512, f"rows {rows}")

diag = ENV["diag_cells"]
own_v = sum(cell["counts"]["own"] for cell in diag if cell["task"] == "V")
mis_v = sum(cell["counts"]["misrouted"] for cell in diag if cell["task"] == "V")
own_g = sum(cell["counts"]["own"] for cell in diag if cell["task"] == "G")
mis_g = sum(cell["counts"]["misrouted"] for cell in diag if cell["task"] == "G")
need((own_v, mis_v, own_g, mis_g) == (14137, 2247, 13989, 2395),
     f"VQA own/mis counts {(own_v, mis_v, own_g, mis_g)}")
vqa_mm = sum(cell["own_route_text_mismatch_rows"] for cell in diag if cell["task"] in ("V", "G"))
need(vqa_mm == 21, f"vqa mismatch rows {vqa_mm}")
cap_mm = sum(cell["own_route_text_mismatch_rows"] for cell in ENV["caption_final_cells"])
need(cap_mm == 41, f"caption final mismatch rows {cap_mm}")
totals = ENV["mismatch_totals"]
need(totals["caption_diagonal"] == 20, f"caption diag mismatches {totals['caption_diagonal']}")
need(totals["caption_diagonal_shared_with_final"] == 16, "shared F p3 mismatch rows")
need(totals["unique_occurrences"] == 66, f"unique mismatch occurrences {totals['unique_occurrences']}")
need(len([e for e in ENV["mismatch_registry"] if e["in_batch1_62_scope"]]) == 62,
     "batch1-scope registry rows")
cc = ENV["batch1_manifest_crosscheck"]
need(cc.get("available") and cc["matched_keys"] == 62 and cc["prediction_text_agreements"] == 62,
     f"batch1 crosscheck {cc}")

# review identified-cell reproduction
for task in ("C", "F"):
    for order in ("V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V"):
        mine = ENV["review_cross_checks"]["caption_final"][task][order]
        expected = REVIEW["caption_final_stage"]["per_task"][task]["per_order"][order]
        if expected["eligible_seed_count"]:
            got = mine["identified_mean_paired_delta_eligible_only"]
            want = expected["identified_mean_paired_delta_eligible_only"]
            need(got is not None and abs(got - want) <= 1e-6,
                 f"identified delta {task}/{order}: {got} vs {want}")

# T = R + K identity on every diag cell
for cell in diag:
    t_r_k = cell["t_r_k"]
    need(abs(t_r_k["T"] - (t_r_k["R"] + t_r_k["K"])) <= 1e-12, f"T!=R+K {cell['order']}/{cell['task']}")

print("VERIFICATION:", "PASS" if not fail else "FAIL")
for message in fail:
    print("  FAIL:", message)
print("\n== headline numbers ==")
print("stored parity maxima:", ENV["stored_parity_maxima"])
print("runtime:", ENV["runtimes"])
for task, values in ENV["diag_occurrence_weighted_task"].items():
    print(f"diag {task}: units={ENV['task_units'][task]} N_all={values['native_all_mean']:.6f} "
          f"O_all={values['oracle_all_mean']:.6f} I_all={values['independent_all_mean']:.6f} "
          f"own_rows={values['own_occurrence_rows']} mis_rows={values['mis_occurrence_rows']}")
flickr = ENV["flickr_pre_post_cells"]
print("flickr 16 cells: wins", sum(c["per_image_wins"] for c in flickr),
      "losses", sum(c["per_image_losses"] for c in flickr),
      "ties", sum(c["per_image_ties"] for c in flickr),
      "corpus delta mean", sum(c["corpus_delta_native_minus_pre"] for c in flickr) / len(flickr))
sys.exit(1 if fail else 0)
