"""Attach real TextVQA/GQA TRAIN references before making train-only validation.

Public instruction train rows have one answer, unlike original evaluation rows.
Never fabricate ten consensus answers or score title-cased GQA targets as if they
were official references. Match source image/question and, when needed, the original instruction label; never model predictions.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from scripts.prepare_data import fetch_json, extract_archive_members, record_source, GQA_QUESTIONS


def canonical_question(question):
    question = question.replace("<image>", "").strip().split("\n")[0]
    return re.sub(r"\s+", " ", question).strip().rstrip(".?!").casefold()


def attach_references(manifest, textvqa, gqa):
    result = copy.deepcopy(manifest)
    text_index, gqa_index = {}, {}
    for row in textvqa["data"]:
        key = (str(row["image_id"]), canonical_question(row["question"]))
        text_index.setdefault(key, []).append(row)
    for question_id, row in gqa.items():
        key = (str(row["imageId"]), canonical_question(row["question"]))
        gqa_index.setdefault(key, []).append((question_id, row))
    failures = []
    for task in result["tasks"]:
        if task["name"] not in ("TextVQA", "GQA"):
            continue
        for row in task["train"]:
            key = (str(row["source_id"]), canonical_question(row["question"]))
            candidates = (text_index if task["name"] == "TextVQA" else gqa_index).get(key, [])
            if not candidates:
                failures.append({"task": task["name"], "id": row["id"], "key": key})
                continue
            if task["name"] == "TextVQA":
                answer_sets = {tuple(sorted(candidate["answers"])) for candidate in candidates}
                if len(answer_sets) != 1 or any(len(candidate["answers"]) != 10 for candidate in candidates):
                    raise ValueError(f"Ambiguous/malformed TextVQA reference: {key}")
                row["answers"] = candidates[0]["answers"]
                row["reference_question_ids"] = sorted(str(candidate["question_id"]) for candidate in candidates)
            else:
                answers = {candidate["answer"] for _, candidate in candidates}
                if len(answers) != 1:
                    # Instruction conversion discarded the original question ID.
                    # Disambiguate using its EXISTING annotation, not a model output.
                    candidates = [(qid, candidate) for qid, candidate in candidates
                                  if candidate["answer"].casefold() == row["answer"].strip().casefold()]
                    if len({candidate["answer"] for _, candidate in candidates}) != 1:
                        raise ValueError(f"Ambiguous GQA reference for original label {row['answer']!r}: {key}")
                row["reference_answer"] = candidates[0][1]["answer"]
                row["reference_question_ids"] = sorted(str(qid) for qid, _ in candidates)
    if failures:
        raise ValueError(f"Official training questions unmatched ({len(failures)}): {json.dumps(failures[:20])}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    sources = {}
    textvqa = fetch_json(root, sources, "TextVQA/train_reference",
                        "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json",
                        Path("references/TextVQA_0.5.1_train.json"))
    destination = root / "cache/references/train_balanced_questions.json"
    if not destination.exists():
        extract_archive_members(GQA_QUESTIONS, root / "cache/archives", [("train_balanced_questions.json", destination)])
    record_source(sources, "GQA/train_reference", destination, root, GQA_QUESTIONS,
                  archive_member_suffix="train_balanced_questions.json")
    manifest_bytes = Path(args.manifest).read_bytes()
    result = attach_references(json.loads(manifest_bytes), textvqa, json.loads(destination.read_text()))
    result["training_reference_sources"] = sources
    result["reference_base_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    result["reference_matching_policy"] = "Original image ID and canonical question; conflicting GQA annotations disambiguated only by the existing instruction target. No model predictions or official eval answers used."
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for item in sources.values():
        (root / item["path"]).touch()
    print(json.dumps({"reference_sources": sources, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
