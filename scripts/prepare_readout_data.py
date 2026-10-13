#!/usr/bin/env python3
"""Prepare new, source-backed readout development and untouched confirmation data.

Run as ``python -m scripts.prepare_readout_data``. All limits count QA rows;
zero means an EMPTY split, never the full source. Pass both protocols in ONE
invocation for cross-protocol byte-disjoint partitions. Separate invocations
only guarantee disjointness within their own output, even with the same seed.
All historical manifests must be supplied: unlisted runs cannot be excluded.
Fresh evaluation excludes ALL old train/validation/eval images and IDs; new
development excludes old eval only and may reuse old training images. No model
scores, predictions, or image availability influence sampling. Missing/corrupt
images abort; --download-images acquires historical images and 64-row candidate
batches, with actual SHA256 hashing and full decoding required in either mode.
Archive servers without range support may require full archive downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import zipfile

from PIL import Image

from scripts import prepare_data as original
from scripts.enrich_vision_references import attach_references, canonical_question


SPLITS = ("train", "validation", "eval")
BATCH_SIZE = 64


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class ImageVerifier:
    """Hash and fully decode actual files; only reuse a digest for unchanged files."""

    def __init__(self, root, download=False, coco_archives=None):
        self.root = Path(root)
        self.download = download
        self.cache = {}
        self.acquisition = {}
        self.coco_archives = Path(coco_archives) if coco_archives is not None else None
        self.archives = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for archive in self.archives.values():
            archive.close()

    def acquire(self, images):
        remaining = []
        for image in images:
            name = Path(image).name
            match = original.COCO_NAME.fullmatch(name)
            if self.coco_archives is None or match is None:
                remaining.append(image)
                continue
            split = match[1]
            if split not in self.archives:
                path = self.coco_archives / f"{split}.zip"
                self.archives[split] = zipfile.ZipFile(path)
                self.acquisition[f"COCO2014/{split}"] = {
                    "url": f"http://images.cocodataset.org/zips/{split}.zip",
                    "method": "official ZIP member extraction",
                    "archive_sha256": original.sha256(path),
                }
            # The same official bytes and exact candidate names as individual
            # downloads; missing/corrupt members abort rather than change samples.
            original.extract_member(self.archives[split], f"{split}/{name}", self.root / "images" / image)
        if remaining:
            self.acquisition.update(original.acquire_images(self.root, remaining))

    def verify(self, images, image_root=None):
        image_root = Path(image_root) if image_root is not None else self.root / "images"
        images = sorted({original.relative_image(image) for image in images})
        missing = [image for image in images if not (image_root / image).is_file()]
        if missing and self.download and image_root.resolve() == (self.root / "images").resolve():
            self.acquire(missing)
        result = {}
        for image in images:
            path = image_root / image
            if not path.is_file():
                raise FileNotFoundError(f"Required image {path}; acquire it or use --download-images. "
                                        "No availability-based replacement is allowed.")
            before = path.stat()
            proof = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            cached = self.cache.get(path.resolve())
            if cached is not None and cached[0] == proof:
                result[image] = cached[1]
                continue
            digest = original.sha256(path)
            with Image.open(path) as decoded:
                decoded.load()
                if min(decoded.size) <= 0:
                    raise ValueError(f"Invalid decoded image size: {path}")
            after = path.stat()
            if proof != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise RuntimeError(f"Image changed while hashing/decoding: {path}")
            self.cache[path.resolve()] = (proof, digest)
            result[image] = digest
        return result


class OldEvaluation:
    def __init__(self):
        self.identities = set()
        self.hashes = set()
        self.ids = set()
        self.source_ids = set()
        self.sources = []
        self.row_count = 0
        self.observed_identities = set()
        self.observed_hashes = set()
        self.observed_ids = set()
        self.observed_source_ids = set()
        self.observed_row_count = 0

    def add(self, manifest, image_hashes, path=None, sha256=None):
        protocol = manifest["protocol"]
        counts = Counter()
        for task in manifest["tasks"]:
            for split in SPLITS:
                for row in task.get(split, []):
                    identity = original.image_identity(row["image"])
                    digest = image_hashes[row["image"]]
                    source_ids = {(protocol, task["name"], str(row[field]))
                                  for field in ("source_id", "source_question_id") if field in row}
                    self.observed_identities.add(identity)
                    self.observed_hashes.add(digest)
                    self.observed_ids.add(row["id"])
                    self.observed_source_ids.update(source_ids)
                    if split == "eval":
                        self.identities.add(identity)
                        self.hashes.add(digest)
                        self.ids.add(row["id"])
                        self.source_ids.update(source_ids)
                    counts[split] += 1
        self.row_count += counts["eval"]
        self.observed_row_count += sum(counts.values())
        self.sources.append({"path": str(path) if path is not None else None,
                             "sha256": sha256, "protocol": protocol, "eval_rows": counts["eval"],
                             "observed_rows_by_split": dict(counts),
                             "verified_image_hashes_sha256": digest_json(sorted(image_hashes.items()))})

    def reason(self, protocol, task, row, confirmation=False):
        if original.image_identity(row["image"]) in self.identities:
            return "old_eval_image_identity"
        if row["id"] in self.ids or any(
            (protocol, task, str(row[field])) in self.source_ids
            for field in ("source_id", "source_question_id") if field in row
        ):
            return "old_eval_id"
        if confirmation:
            if original.image_identity(row["image"]) in self.observed_identities:
                return "old_observed_image_identity"
            if row["id"] in self.observed_ids or any(
                (protocol, task, str(row[field])) in self.observed_source_ids
                for field in ("source_id", "source_question_id") if field in row
            ):
                return "old_observed_id"
        return None


def load_old_evaluation(paths, root, verifier):
    excluded = OldEvaluation()
    for path in paths:
        raw = Path(path).read_bytes()
        manifest = json.loads(raw)
        image_root = Path(manifest.get("image_root", "images"))
        if not image_root.is_absolute():
            image_root = root / image_root
        images = {row["image"] for task in manifest["tasks"] for split in SPLITS for row in task.get(split, [])}
        hashes = verifier.verify(images, image_root)
        excluded.add(manifest, hashes, Path(path).resolve(), hashlib.sha256(raw).hexdigest())
    return excluded


def training_partition(digest, seed, train_limit, validation_limit):
    total = train_limit + validation_limit
    if not total:
        raise ValueError("Cannot partition an empty development request")
    value = int(hashlib.sha256(f"{seed}/development-image/{digest}".encode()).hexdigest(), 16)
    return "validation" if value * total < validation_limit * 2**256 else "train"


def select_rows(rows, protocol, task, source_split, limits, seed, verifier, old, eval_hashes):
    """Sample deterministic candidates; fail on missing bytes, refill on overlap only."""
    selected = {split: [] for split in limits}
    stats = {"source_normalized_rows": len(rows), "considered_rows": 0,
             "excluded": Counter(), "unselected_full_partition": 0}
    if not any(limits.values()):
        return selected, stats
    indices = list(range(len(rows)))
    derived = hashlib.sha256(f"{seed}/{protocol}/{task}/{source_split}/candidates".encode()).digest()
    random.Random(int.from_bytes(derived, "big")).shuffle(indices)
    seen_ids = set()
    for start in range(0, len(indices), BATCH_SIZE):
        if all(len(selected[split]) >= limit for split, limit in limits.items()):
            break
        candidates = []
        for index in indices[start:start + BATCH_SIZE]:
            row = rows[index]
            stats["considered_rows"] += 1
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate normalized candidate ID: {row['id']}")
            seen_ids.add(row["id"])
            reason = old.reason(protocol, task, row, confirmation=source_split == "eval")
            if reason:
                stats["excluded"][reason] += 1
            else:
                candidates.append(row)
        hashes = verifier.verify(row["image"] for row in candidates)
        for row in candidates:
            digest = hashes[row["image"]]
            if digest in old.hashes:
                stats["excluded"]["old_eval_image_sha256"] += 1
                continue
            if source_split == "eval" and digest in old.observed_hashes:
                stats["excluded"]["old_observed_image_sha256"] += 1
                continue
            if source_split == "train" and digest in eval_hashes:
                stats["excluded"]["fresh_eval_image_sha256"] += 1
                continue
            split = "eval" if source_split == "eval" else training_partition(
                digest, seed, limits["train"], limits["validation"])
            if len(selected[split]) >= limits[split]:
                stats["unselected_full_partition"] += 1
                continue
            selected[split].append({**row, "image_sha256": digest})
    return selected, stats


def load_rows(root, protocol, task, split, eval_split, sources):
    key = f"{task}/{split}"
    source_split = "train" if split == "train" else eval_split
    if protocol == "cl4vqa":
        path = original.cl4_source(root, task, split)
        records = original.load_json(path)
        original.record_source(sources, key, path, root, official_partition=path.name)
    else:
        url = f"{original.HF_BASE}/{task}/{source_split}.json"
        records = original.fetch_json(root, sources, key, url,
                                      Path("smolora") / original.HF_REVISION / task / f"{source_split}.json")
        sources[key]["revision"] = original.HF_REVISION
    rows, counts = original.normalize(records, protocol, task, split)
    sources[key].update(counts, source_split=source_split)
    return rows


def attach_textvqa_training_references(rows, instructions, reference):
    """Recover discarded question IDs only after validating whole-source order.

    The pinned instruction conversion retains all official training rows in
    order, but puts image IDs in its question_id field. Image/question matching
    alone loses distinct annotation sets for repeated questions.
    """
    originals = reference["data"]
    if len(instructions) != len(originals):
        raise ValueError("TextVQA source order cannot be verified: different row counts")
    question_ids = set()
    for index, (instruction, original_row) in enumerate(zip(instructions, originals)):
        turns = instruction["conversations"]
        answers = original_row["answers"]
        question_id = str(original_row["question_id"])
        if (len(turns) != 2 or turns[0]["from"] != "human" or turns[1]["from"] != "gpt"
                or str(instruction["question_id"]) != str(original_row["image_id"])
                or Path(instruction["image"]).stem != str(original_row["image_id"])
                or canonical_question(turns[0]["value"]) != canonical_question(original_row["question"])
                or not isinstance(answers, list) or len(answers) != 10
                or not all(isinstance(answer, str) for answer in answers)
                or turns[1]["value"].strip().casefold() not in {answer.strip().casefold() for answer in answers}
                or question_id in question_ids):
            raise ValueError(f"TextVQA whole-source alignment failed at row {index}")
        question_ids.add(question_id)
    result = []
    for row in rows:
        index = row["source_row_index"]
        if not 0 <= index < len(instructions):
            raise ValueError("TextVQA selected row is outside the verified source")
        instruction, original_row = instructions[index], originals[index]
        if (row["turn_index"] != 0 or str(row["source_id"]) != str(instruction["question_id"])
                or row["image"] != original.relative_image(instruction["image"])
                or row["question"] != instruction["conversations"][0]["value"]
                or row["answer"] != instruction["conversations"][1]["value"]):
            raise ValueError(f"TextVQA selected row disagrees with its source position: {row['id']}")
        result.append(dict(row, answers=list(original_row["answers"]),
                           reference_question_ids=[str(original_row["question_id"])],
                           reference_identity_method="whole-source-verified row position"))
    return result


def enrich_selected(root, task, split, rows, sources):
    """Only copy/enrich selected rows; fit labels stay byte-for-byte unchanged."""
    if not rows:
        return rows
    if task == "ScienceQA":
        reference = original.fetch_json(root, sources, "ScienceQA/reference", original.SCIENCEQA_REFERENCE,
                                        Path("references/scienceqa_problems.json"))
        original.enrich_references(task, split, rows, reference)
    elif split == "eval" and task == "TextVQA":
        reference = original.fetch_json(root, sources, "TextVQA/reference", original.TEXTVQA_REFERENCE,
                                        Path("references/TextVQA_0.5.1_val.json"))
        by_id = {}
        for item in reference["data"]:
            key = str(item["question_id"])
            if key in by_id:
                raise ValueError(f"Duplicate original TextVQA question ID: {key}")
            by_id[key] = item
        original.enrich_references(task, split, rows, by_id)
    elif split == "eval" and task == "GQA":
        original.enrich_references(task, split, rows, original.gqa_reference(root, sources))
    elif split == "train" and task in ("TextVQA", "GQA"):
        textvqa, gqa = {"data": []}, {}
        if task == "TextVQA":
            textvqa = original.fetch_json(root, sources, "TextVQA/train_reference",
                                          "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json",
                                          Path("references/TextVQA_0.5.1_train.json"))
            instructions = original.fetch_json(
                root, sources, "TextVQA/train_identity_source", f"{original.HF_BASE}/TextVQA/train.json",
                Path("smolora") / original.HF_REVISION / "TextVQA/train.json")
            rows = attach_textvqa_training_references(rows, instructions, textvqa)
        else:
            path = root / "cache/references/train_balanced_questions.json"
            if not path.is_file():
                original.extract_archive_members(original.GQA_QUESTIONS, root / "cache/archives",
                                                  [("train_balanced_questions.json", path)])
            original.record_source(sources, "GQA/train_reference", path, root, original.GQA_QUESTIONS,
                                   archive_member_suffix="train_balanced_questions.json")
            gqa = original.load_json(path)
            rows = attach_references({"tasks": [{"name": task, "train": rows}]}, textvqa, gqa)["tasks"][0]["train"]
    return rows


def finalize_development(task, selected, enriched):
    by_id = {row["id"]: row for row in enriched}
    for split in ("train", "validation"):
        task[split] = [by_id[row["id"]] for row in selected[split]]
    for row in task["validation"]:
        if task["metric"] == "textvqa":
            answers = row.get("answers")
            if not isinstance(answers, list) or len(answers) != 10 or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"Validation requires ten original TextVQA answers: {row['id']}")
        elif task["metric"] == "gqa":
            if not isinstance(row.get("reference_answer"), str):
                raise ValueError(f"Validation requires original GQA training reference: {row['id']}")
            row["instruction_answer"] = row["answer"]
            row["answer"] = row["reference_answer"]


def partition_report(manifests, old):
    groups = {split: {row["image_sha256"] for manifest in manifests for task in manifest["tasks"]
                      for row in task[split]} for split in SPLITS}
    overlaps = {f"{a}/{b}": len(groups[a] & groups[b])
                for a, b in (("train", "validation"), ("train", "eval"), ("validation", "eval"))}
    overlaps.update({f"{split}/old_eval": len(groups[split] & old.hashes) for split in SPLITS})
    overlaps["eval/old_observed"] = len(groups["eval"] & old.observed_hashes)
    if any(overlaps.values()):
        raise AssertionError(f"Image-byte partition overlap: {overlaps}")
    return {"scope_protocols": [manifest["protocol"] for manifest in manifests],
            "scope": "All tasks and protocols produced together in this invocation; NOT independent invocations.",
            "image_groups": {split: len(values) for split, values in groups.items()},
            "image_groups_sha256": {split: digest_json(sorted(values)) for split, values in groups.items()},
            "overlap_counts": overlaps,
            "old_observed": {"rows_including_repeated_manifests": old.observed_row_count,
                             "unique_image_identities": len(old.observed_identities),
                             "unique_image_sha256": len(old.observed_hashes),
                             "image_sha256_set_digest": digest_json(sorted(old.observed_hashes))},
            "old_eval": {"supplied_manifests": old.sources, "rows_including_repeated_manifests": old.row_count,
                         "unique_image_identities": len(old.identities), "unique_image_sha256": len(old.hashes),
                         "image_sha256_set_digest": digest_json(sorted(old.hashes))}}


def prepare(args):
    with ImageVerifier(args.root, args.download_images, args.coco_archives) as verifier:
        return prepare_verified(args, verifier)


def prepare_verified(args, verifier):
    root = args.root
    old = load_old_evaluation(args.old_manifests, root, verifier)
    specifications = {protocol: ([(name, "test", "clmoe_exact") for name in original.CL4_TASKS]
                                 if protocol == "cl4vqa" else original.STREAM_TASKS)
                      for protocol in args.protocol}
    manifests = []
    eval_hashes = set()
    # Reserve confirmation images across EVERY requested protocol before reading train sources.
    # Reference answers are attached only after selection and never enter candidate ranking.
    for protocol, specs in specifications.items():
        manifest = {"protocol": protocol, "image_root": str((root / "images").resolve()),
                    "subset_seed": args.seed, "sources": {}, "tasks": []}
        for name, eval_split, metric in specs:
            rows = load_rows(root, protocol, name, "eval", eval_split, manifest["sources"])
            selected, stats = select_rows(rows, protocol, name, "eval", {"eval": args.eval_limit},
                                          args.seed, verifier, old, eval_hashes)
            del rows
            evaluation = selected["eval"]
            if protocol == "vqa_stream":
                evaluation = enrich_selected(root, name, "eval", evaluation, manifest["sources"])
            eval_hashes.update(row["image_sha256"] for row in evaluation)
            manifest["tasks"].append({"name": name, "metric": metric, "eval": evaluation,
                                      "selection": {"eval": stats}})
        manifests.append(manifest)
    for manifest in manifests:
        protocol = manifest["protocol"]
        for task, (_, eval_split, _) in zip(manifest["tasks"], specifications[protocol]):
            rows = load_rows(root, protocol, task["name"], "train", eval_split, manifest["sources"])
            selected, stats = select_rows(rows, protocol, task["name"], "train",
                                          {"train": args.train_limit, "validation": args.validation_limit},
                                          args.seed, verifier, old, eval_hashes)
            del rows
            development = selected["train"] + selected["validation"]
            if protocol == "vqa_stream":
                development = enrich_selected(root, task["name"], "train", development, manifest["sources"])
            finalize_development(task, selected, development)
            task["selection"]["development"] = stats
            for split, limit in (("train", args.train_limit), ("validation", args.validation_limit), ("eval", args.eval_limit)):
                task["selection"].setdefault(split, {}).update(
                    requested_limit=limit, selected_count=len(task[split]), shortage=limit - len(task[split]),
                    selected_ids_sha256=digest_json([row["id"] for row in task[split]]))
                original.announce(f"{protocol}/{task['name']}/{split}: {len(task[split])}/{limit} QA rows"
                                  + (" (SHORTAGE; no replacement)" if len(task[split]) < limit else ""))
    report = partition_report(manifests, old)
    for manifest in manifests:
        manifest["vision_partition"] = report
        manifest["subset_policy"] = (
            "Seeded candidate permutation of full official sources, no replacement; zero limits mean empty. "
            "Old eval identities, namespaced source IDs, and verified image bytes excluded from every split. "
            "Fresh official eval additionally excludes ALL old train/validation image identities, IDs and bytes. "
            "New fit/validation may reuse historical train/validation examples. "
            "Fresh official eval reserved first, never used for fitting or validation selection. "
            "Train-only fit/validation use global seeded SHA256 image buckets with requested-count ratio. "
            "Deterministic 64-row candidate batches refilled on overlap; missing/corrupt images abort. "
            "Selection/exclusion counts describe considered candidates, not an audit of unused source images. "
            "Sources can be exhausted, including a hash bucket, so actual counts/shortages are explicit."
        )
        manifest["reference_policy"] = (
            "Original training targets unchanged. Selected TextVQA/GQA training rows enriched from official TRAIN "
            "references before validation scoring conversion. TextVQA question IDs recovered from source row "
            "positions only after verifying all original instruction/reference rows align by image, question and "
            "existing target; each original ten-answer set is preserved, never merged or chosen by prediction. "
            "GQA validation uses reference_answer and retains instruction_answer. Official eval references use existing scorers."
        )
        hashes = {row["image"]: row["image_sha256"] for task in manifest["tasks"] for split in SPLITS for row in task[split]}
        manifest["image_sha256"] = hashes
        manifest["images"] = {"ready": True, "download_requested": args.download_images,
                              "selected_unique_paths": len(hashes), "present_count": len(hashes), "missing_count": 0,
                              "missing_paths": [], "decoded": True}
        manifest["image_acquisition_sources"] = verifier.acquisition
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Shared original data/cache/image root")
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory; existing paths are refused")
    parser.add_argument("--protocol", nargs="+", choices=("cl4vqa", "vqa_stream"), required=True,
                        help="Supply both together for cross-protocol partition disjointness")
    parser.add_argument("--old-manifests", nargs="+", type=Path, required=True,
                        help="ALL historical train/validation/eval manifests from both protocols, including previous new runs")
    parser.add_argument("--train-limit", type=int, default=2048, help="Fit QA/task; 0 means empty")
    parser.add_argument("--validation-limit", type=int, default=256, help="Train-only validation QA/task; 0 means empty")
    parser.add_argument("--eval-limit", type=int, default=512, help="Fresh official eval QA/task; 0 means empty")
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--coco-archives", type=Path,
                        help="Optional official train2014.zip/val2014.zip cache; transport only, never resampling")
    args = parser.parse_args()
    if min(args.train_limit, args.validation_limit, args.eval_limit) < 0:
        parser.error("Limits must be nonnegative; zero means empty")
    if len(set(args.protocol)) != len(args.protocol):
        parser.error("Do not repeat a protocol")
    args.root = args.root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.old_manifests = [path.expanduser().resolve() for path in args.old_manifests]
    if args.output_dir.exists():
        parser.error("--output-dir must be a new path; no existing output is overwritten")
    manifests = prepare(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for manifest in manifests:
        destination = args.output_dir / f"{manifest['protocol']}.json"
        with destination.open("x", encoding="utf-8") as target:
            json.dump(manifest, target, ensure_ascii=False, indent=2)
            target.write("\n")
        original.announce(f"Wrote {destination}")


if __name__ == "__main__":
    main()
