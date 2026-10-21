"""Render the figures of RESEARCH_OVERVIEW.md from committed artifacts.

Read-only: no training, inference or scorer call. Every value comes from a
committed artifact under ``artifacts/``; the source path is printed next to
each figure in the overview document.

    uv run --with matplotlib --with numpy python scripts/make_research_figures.py
"""

from __future__ import annotations

import csv
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "figures"

METHOD_LABEL = {
    "shared_lora": "Shared LoRA",
    "capacity_matched_shared": "Capacity-matched",
    "replay_lora": "Common replay",
    "expand": "Expansion",
    "frozen_first": "Frozen-first",
    "tuned_replay": "Tuned replay (10ep)",
}
METHOD_COLOR = {
    "shared_lora": "#9e9e9e",
    "capacity_matched_shared": "#6d8fb3",
    "replay_lora": "#e08a3c",
    "expand": "#2e7d32",
    "frozen_first": "#c9c9c9",
    "tuned_replay": "#b5562a",
}
TASK_LABEL = {"V": "VQAv2", "C": "COCO", "G": "GQA", "F": "Flickr30k"}


def load_json(name: str):
    return json.loads((ART / name).read_text())


def g4_tables():
    with tarfile.open(ART / "multimodal-g4-report.tar.gz") as tar:
        summary = list(csv.DictReader(io.StringIO(tar.extractfile("g4/summary.csv").read().decode())))
        tasks = list(csv.DictReader(io.StringIO(tar.extractfile("g4/tasks.csv").read().decode())))
    return summary, tasks


def method_key(row) -> str:
    if row["kind"] == "tuned_replay":
        return "tuned_replay"
    return row["method"]


def seed_means(summary, column: str):
    """Per method: list of per-seed values (mean over the four orders)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in summary:
        if row["kind"] == "independent" or row[column] == "NA":
            continue
        acc[method_key(row)][row["seed"]].append(float(row[column]))
    return {m: [float(np.mean(v)) for _, v in sorted(seeds.items())] for m, seeds in acc.items()}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def fig_final_scores(summary) -> None:
    order = ["frozen_first", "shared_lora", "capacity_matched_shared", "expand", "replay_lora", "tuned_replay"]
    columns = [
        ("family_vqa_normalized_final", "VQA family"),
        ("family_caption_normalized_final", "Caption family"),
        ("family_balanced_normalized_final", "Balanced"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (column, title) in zip(axes, columns):
        data = seed_means(summary, column)
        for i, m in enumerate(order):
            vals = np.array(data[m])
            ax.bar(i, vals.mean(), color=METHOD_COLOR[m], edgecolor="black", linewidth=0.6)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=10, color="black", zorder=3)
            ax.text(i, vals.max() + 0.02, f"{vals.mean():.3f}", ha="center", fontsize=8)
        ax.axhline(1.0, ls="--", lw=0.8, color="gray")
        ax.set_xticks(range(len(order)), [METHOD_LABEL[m] for m in order], rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
    axes[0].set_ylabel("Normalized final score\n(÷ G1 independent validation ref.)")
    fig.suptitle("G4 confirmation: final normalized score (8 seeds × 4 orders; dots = seed means)")
    save(fig, "fig1_g4_final_scores.png")


def fig_main_comparisons(stats) -> None:
    rows = []
    for item in stats["primary_comparisons"] + stats["tuned_replay_comparisons"]:
        cand = np.array(list(item["candidate_seed_means"].values()))
        base = np.array(list(item["baseline_seed_means"].values()))
        diff = cand - base
        st = item.get("statistics", {})
        ci = st.get("seed_bootstrap_95_percentile") or item.get("CI_95") or [np.nan, np.nan]
        holm = item["holm_adjusted_p"]
        rows.append((item["family"], item["baseline"], diff.mean(), ci, holm))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = []
    for y, (family, base, mean, ci, holm) in enumerate(reversed(rows)):
        significant = holm < 0.05
        color = "#757575" if not significant else ("#2e7d32" if mean > 0 else "#c62828")
        ax.errorbar(mean, y, xerr=[[mean - ci[0]], [ci[1] - mean]], fmt="o", color=color, capsize=4)
        holm_txt = f"Holm p={holm:.4g}" + ("" if significant else " (n.s.)")
        ax.text(max(ci[1], mean) + 0.015, y, f"{mean:+.3f}  {holm_txt}", va="center", fontsize=8)
        labels.append(f"{family.upper()} vs {METHOD_LABEL.get(base, base)}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Expansion − baseline (normalized final, 95% seed-bootstrap CI)")
    ax.set_xlim(-0.45, 0.75)
    ax.set_title("Pre-registered comparisons (gray = not significant after Holm)\n"
                 "tuned replay = separate Holm family of 2; min attainable p: 0.046875 / 0.015625")
    save(fig, "fig2_main_comparisons.png")


def fig_forgetting_gain(summary, tasks) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    order = ["shared_lora", "capacity_matched_shared", "replay_lora", "expand"]
    ax = axes[0]
    width = 0.2
    for j, (column, fam) in enumerate(
        [("family_vqa_normalized_AF", "VQA"), ("family_caption_normalized_AF", "Caption")]
    ):
        data = seed_means(summary, column)
        for i, m in enumerate(order):
            vals = np.array(data[m])
            x = j + (i - 1.5) * width
            ax.bar(x, vals.mean(), width, color=METHOD_COLOR[m], edgecolor="black", linewidth=0.6,
                   label=METHOD_LABEL[m] if j == 0 else None)
            ax.text(x, vals.mean() + 0.01, f"{vals.mean():.3f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1], ["VQA family", "Caption family"])
    ax.set_ylabel("Normalized average forgetting (AF, lower = better)")
    ax.set_title("(a) Forgetting")
    ax.legend(fontsize=8)

    ax = axes[1]
    gains: dict[str, list[float]] = defaultdict(list)
    for row in tasks:
        if row["method"] != "expand" or row["kind"] != "common_cl" or row["stage_index"] == "0":
            continue
        gains[row["task_id"]].append(float(row["normalized_pre_to_post_gain"]))
    ids = ["V", "G", "C", "F"]
    for i, t in enumerate(ids):
        vals = np.array(gains[t])
        color = "#c62828" if vals.mean() < 0 else "#2e7d32"
        ax.bar(i, vals.mean(), color=color, edgecolor="black", linewidth=0.6)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.2, 0.2, len(vals)), vals, s=8, color="black", zorder=3)
        neg = int((vals < 0).sum())
        ax.text(i, max(vals.max(), 0) + 0.04, f"{vals.mean():+.3f}\n{neg}/{len(vals)} negative", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(4), [TASK_LABEL[t] for t in ids])
    ax.set_ylabel("Normalized pre→post gain (later tasks only)")
    ax.set_title("(b) Expansion: learning of each later task (dots = order×seed)")
    ax.set_ylim(-0.35, 1.4)
    save(fig, "fig3_forgetting_and_gain.png")


def fig_transfer_decomposition(partition) -> None:
    fam = {"VQA family": partition["vqa_transfer"]["decomposed"]["total"],
           "Caption family": partition["caption_transfer"]["total"]}
    fig, ax = plt.subplots(figsize=(7, 4))
    keys = [("T", "T = native − independent"), ("R", "R = native − oracle (routing)"),
            ("K", "K = oracle − independent (residual)")]
    colors = ["#455a64", "#ef6c00", "#7b1fa2"]
    for i, (name, vals) in enumerate(fam.items()):
        for j, (k, label) in enumerate(keys):
            x = i + (j - 1) * 0.25
            ax.bar(x, vals[k], 0.25, color=colors[j], label=label if i == 0 else None)
            ax.text(x, vals[k] + (0.002 if vals[k] >= 0 else -0.004), f"{vals[k]:+.4f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1], list(fam))
    ax.set_ylabel("Normalized score difference")
    ax.set_title("Diagonal transfer decomposition T = R + K (G4/G5)")
    ax.set_ylim(-0.052, 0.03)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig4_transfer_decomposition.png")


def fig_routing(g5) -> None:
    variants = ["native_joint", "family_restricted_joint", "question_only", "image_only", "task_id_oracle"]
    names = ["Native joint", "Family-restricted", "Question-only", "Image-only", "Task-ID oracle"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    for j, (fam, color) in enumerate([("vqa", "#1565c0"), ("caption", "#f9a825"), ("balanced", "#424242")]):
        vals = [g5["variants"][v]["family_normalized_final"][fam] for v in variants]
        xs = np.arange(len(variants)) + (j - 1) * 0.26
        ax.bar(xs, vals, 0.26, color=color, label=fam)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(variants)), names, rotation=20, fontsize=8)
    ax.set_ylabel("Normalized final score")
    ax.set_ylim(0, 1.3)
    ax.set_title("(a) Same 32 expert banks, different routers (G5)")
    ax.legend(fontsize=8, loc="upper center", ncol=3)

    ax = axes[1]
    other_expert = [g5["variants"][v]["other_expert_fraction"]["all"] for v in variants]
    other_family = [g5["variants"][v]["other_family_fraction"]["all"] for v in variants]
    xs = np.arange(len(variants))
    ax.bar(xs - 0.18, other_expert, 0.36, color="#8d6e63", label="routed to another task's expert")
    ax.bar(xs + 0.18, other_family, 0.36, color="#d32f2f", label="routed to another family")
    for x, a, b in zip(xs, other_expert, other_family):
        ax.text(x - 0.18, a + 0.01, f"{a:.1%}", ha="center", fontsize=7)
        ax.text(x + 0.18, b + 0.01, f"{b:.2%}", ha="center", fontsize=7)
    ax.set_xticks(xs, names, rotation=20, fontsize=8)
    ax.set_ylabel("Share of eval rows")
    ax.set_title("(b) Routing errors are within-family, not cross-family")
    ax.legend(fontsize=8)
    save(fig, "fig5_routing_variants.png")


def fig_caption_native_vs_oracle(partition) -> None:
    per_task = partition["caption_final_stage"]["per_task"]
    orders = ["V_C_G_F", "F_G_C_V", "V_G_C_F", "F_C_G_V"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (task, color) in enumerate([("C", "#1565c0"), ("F", "#f9a825")]):
        vals = [per_task[task]["per_order"][o]["corpus_delta_native_minus_oracle_mean"] for o in orders]
        xs = np.arange(len(orders)) + (j - 0.5) * 0.36
        ax.bar(xs, vals, 0.36, color=color, label=f"{TASK_LABEL[task]} (CIDEr-D ×1)")
        for x, v in zip(xs, vals):
            ax.text(x, v + (0.002 if v >= 0 else -0.006), f"{v:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(orders)), [o.replace("_", "→") for o in orders])
    ax.set_ylabel("Native − task-ID oracle (final stage)")
    ax.set_title("Caption: oracle wins on COCO, native wins on Flickr (sign reversal)")
    ax.set_ylim(-0.088, 0.038)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig6_caption_native_vs_oracle.png")


def fig_flickr_termination(target_format, eos, diagnosis) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    ax = axes[0]
    for j, (task, color) in enumerate([("C", "#1565c0"), ("F", "#f9a825")]):
        splits = ["train", "validation", "eval"]
        vals = [target_format["reference_formats"][task][s]["ending_share"]["period"]
                + target_format["reference_formats"][task][s]["ending_share"].get("space_period", 0.0)
                for s in splits]
        xs = np.arange(3) + (j - 0.5) * 0.36
        ax.bar(xs, vals, 0.36, color=color, label=TASK_LABEL[task])
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    ax.set_xticks(range(3), ["train", "validation", "eval"])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Share of references ending with '.'")
    ax.set_title("(a) Flickr targets end with a period far more often")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    ax = axes[1]
    groups = ["independent", "expand/F_first", "expand/F_last", "replay_lora/F_first", "replay_lora/F_last",
              "shared_lora/F_last", "capacity_matched_shared/F_last"]
    agg = eos["aggregate"]
    vals = [agg[g]["eos_rate"]["mean"] for g in groups]
    toks = [agg[g]["mean_generated_tokens"]["mean"] for g in groups]
    xs = np.arange(len(groups))
    ax.bar(xs, vals, color=["#f9a825" if "F_first" in g or g == "independent" else "#8d6e63" for g in groups])
    for x, v, t in zip(xs, vals, toks):
        ax.text(x, v + 0.01, f"{v:.2f}\n{t:.0f} tok", ha="center", fontsize=7)
    ax.set_xticks(xs, [g.replace("capacity_matched_shared", "capacity").replace("_lora", "") for g in groups],
                  rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("EOS rate on Flickr eval (G4 final stage)")
    ax.set_title("(b) Flickr-trained adapters often fail to stop")

    ax = axes[2]
    split = diagnosis["old_c_vs_new_f_by_new_f_termination"]
    cats = [("new_f_emitted_eos", "new F emitted EOS"), ("new_f_no_eos", "new F hit 64-token limit")]
    for i, (k, label) in enumerate(cats):
        old_c, new_f = split[k]["old_c_score_mean"], split[k]["new_f_score_mean"]
        ax.bar(i - 0.18, old_c, 0.36, color="#1565c0", label="old COCO expert" if i == 0 else None)
        ax.bar(i + 0.18, new_f, 0.36, color="#f9a825", label="new Flickr expert" if i == 0 else None)
        ax.text(i, max(old_c, new_f) + 0.008, f"n={split[k]['occurrences']}\nΔ={new_f - old_c:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(2), [c[1] for c in cats])
    ax.set_ylabel("Per-image CIDEr-D (×1), same Flickr images")
    ax.set_title("(c) The Flickr deficit is concentrated in non-EOS rows")
    ax.set_ylim(0, 0.21)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig7_flickr_termination.png")


def fig_decoding_and_strip(probe, strip) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
    ax = axes[0]
    variants = ["D0", "D1_b1", "D1_b2", "D1_b4", "D2_r1.2", "D2_r1.5", "D3_n32", "D3_n24", "D4_period"]
    for j, (task, color) in enumerate([("C", "#1565c0"), ("F", "#f9a825")]):
        means = []
        for v in variants:
            means.append(np.mean([probe["runs"][f"{task}/{s}"]["variants"][v]["cider_d"] for s in (137, 149, 163)]))
        xs = np.arange(len(variants)) + (j - 0.5) * 0.38
        ax.bar(xs, means, 0.38, color=color, label=f"{TASK_LABEL[task]}-trained independent adapter")
        for x, m in zip(xs, means):
            ax.text(x, m + 0.004, f"{m:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(variants)), variants, rotation=30, fontsize=8)
    ax.set_ylabel("Flickr validation CIDEr-D (×1), mean of 3 seeds")
    ax.set_title("(a) Stage 0.5 decoding probe on Flickr val (diagnostic only)")
    ax.set_ylim(0, 0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    seeds = [137, 149, 163]
    base = [probe["runs"][f"F/{s}"]["variants"]["D0"]["cider_d"] for s in seeds]
    coco = [probe["runs"][f"C/{s}"]["variants"]["D0"]["cider_d"] for s in seeds]
    stripped = [strip["runs"][f"STRIP/{s}"]["results"]["D0"]["summary"]["cider_d"] for s in seeds]
    eos_base = [probe["runs"][f"F/{s}"]["variants"]["D0"]["eos_rate"] for s in seeds]
    eos_strip = [strip["runs"][f"STRIP/{s}"]["results"]["D0"]["summary"]["eos_rate"] for s in seeds]
    xs = np.arange(3)
    ax.bar(xs - 0.27, base, 0.27, color="#f9a825", label="Flickr, original target (D0)")
    ax.bar(xs, stripped, 0.27, color="#2e7d32", label="Flickr, terminal '.' stripped (D0)")
    ax.bar(xs + 0.27, coco, 0.27, color="#1565c0", label="COCO adapter on Flickr (D0)")
    for x, b, s_, eb, es in zip(xs, base, stripped, eos_base, eos_strip):
        ax.text(x - 0.27, b + 0.005, f"{b:.3f}\nEOS {eb:.2f}", ha="center", fontsize=7)
        ax.text(x, s_ + 0.005, f"{s_:.3f}\nEOS {es:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xs, [f"seed {s}" for s in seeds])
    ax.set_ylim(0, 0.38)
    ax.set_ylabel("Flickr validation CIDEr-D (×1)")
    ax.set_title("(b) Stage 1 single-factor fix: strip terminal period from targets")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig8_decoding_probe_and_period_strip.png")


def fig_truncation(trunc) -> None:
    rules = ["none", "R1", "R2", "R3_12", "R3_16", "R3_20", "R4"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, (cohort, label, color) in enumerate([
        ("g4_independent_F", "G4 independent Flickr (8 cells)", "#f9a825"),
        ("flickr_last_cells", "Flickr-last oracle F expert (16 cells)", "#8d6e63"),
    ]):
        vals = [trunc["results"][cohort]["variants"][r]["pooled_corpus_cider_d"] for r in rules]
        xs = np.arange(len(rules)) + (j - 0.5) * 0.38
        ax.bar(xs, vals, 0.38, color=color, label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.003, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(range(len(rules)), rules)
    ax.set_ylabel("Eval CIDEr-D (×1), post-hoc truncation")
    ax.set_title("Post-hoc truncation of saved non-EOS Flickr outputs (diagnostic)")
    ax.set_ylim(0, 0.215)
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig9_truncation_rescore.png")


def main() -> None:
    summary, tasks = g4_tables()
    stats = load_json("multimodal-g4-statistics.json")
    partition = load_json("routing-review-routing-partition.json")
    g5 = load_json("multimodal-g5-summary.json")
    fig_final_scores(summary)
    fig_main_comparisons(stats)
    fig_forgetting_gain(summary, tasks)
    fig_transfer_decomposition(partition)
    fig_routing(g5)
    fig_caption_native_vs_oracle(partition)
    fig_flickr_termination(load_json("stage05-target-format.json"), load_json("g4-final-flickr-eos.json"),
                           load_json("stage0-flickr-expert-diagnosis.json"))
    fig_decoding_and_strip(load_json("stage05-decoding-probe.json"), load_json("stage1-period-strip.json"))
    fig_truncation(load_json("stage05-truncation-rescore.json"))


if __name__ == "__main__":
    main()
