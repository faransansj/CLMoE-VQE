#!/usr/bin/env python3
"""Prepare local official sources without downloads or availability-based sampling.

Source JSON schema: schema_version=1, sources=[{type, task_id, revision, license,
train, eval}]. VQAv2 split objects contain questions, annotations, images and an
optional image_template. COCO split objects contain annotations and images.
GQA split objects contain questions, images and optional image_template.
Flickr30k uses dataset, images, split_recipe={train:["train"],eval:["test"]}.
Annotation paths are relative to the config file; image directories are relative
to --image-root. Source types: vqav2, coco_captions, gqa, flickr30k.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from sema_multimodal.data import (CAPTION_INSTRUCTION, ImageVerifier, SOURCE_SPECS,
                                  SPLITS, image_identity, nonempty, relative_image,
                                  validate_manifest)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def rank(seed, *parts):
    return hashlib.sha256(json.dumps([seed, *parts], separators=(",", ":")).encode()).hexdigest()


def read_source(path, records, label, revision):
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    records[label] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "revision": revision}
    return json.loads(raw)


def source_path(base, value):
    nonempty(value, "source path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def image_path(directory, filename):
    nonempty(directory, "images directory")
    return relative_image((PurePosixPath(directory) / relative_image(filename)).as_posix())


def unique_index(rows, key, label):
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"Malformed {label} entry")
        identifier = str(row[key])
        if identifier in result:
            raise ValueError(f"Duplicate {label} ID: {identifier}")
        result[identifier] = row
    return result


def text(value, label):
    # Keep official text byte-for-byte; stripping is only a validity check.
    return nonempty(value, label)


def make_row(source, split, identifier, image_id, image, input_text, references, target=None):
    if not isinstance(references, list) or not references:
        raise ValueError(f"Missing official references: {source}/{identifier}")
    refs = [text(value, "reference") for value in references]
    image_id = str(image_id)
    nonempty(image_id, "source_image_id")
    return {"id": f"{source}/{split}/{identifier}", "image": image,
            "source_image_id": image_id, "input_text": text(input_text, "input_text"),
            "target": text(target if target is not None else refs[0], "target"),
            "references": refs, "source": source, "source_split": split,
            "family": SOURCE_SPECS[source][0]}


def normalize_source(spec, base, sources, seed):
    source = spec["type"]
    revision = nonempty(spec.get("revision"), f"{source} revision")
    nonempty(spec.get("license"), f"{source} license/use permission")
    candidates = {"train": [], "eval": []}
    if source == "flickr30k":
        recipe = spec.get("split_recipe")
        if not isinstance(recipe, dict) or recipe.get("eval") != ["test"] or recipe.get("train") not in (["train"], ["train", "restval"]):
            raise ValueError("Flickr30k requires pinned train[/restval] -> development, test -> eval recipe")
        records = read_source(source_path(base, spec["dataset"]), sources, f"{source}/dataset", revision)
        by_image = unique_index(records.get("images"), "imgid", "Karpathy images")
        filenames = set()
        for identifier, record in by_image.items():
            split = record.get("split")
            if split not in ("train", "val", "test", "restval"):
                raise ValueError(f"Unknown Flickr30k Karpathy split: {split}")
            filename = record["filename"]
            if filename in filenames:
                raise ValueError(f"Duplicate Flickr30k filename: {filename}")
            filenames.add(filename)
            sentences = record.get("sentences")
            if not isinstance(sentences, list) or not sentences:
                raise ValueError(f"Missing Flickr30k sentences: {identifier}")
            refs = [text(sentence.get("raw"), "Karpathy raw caption") for sentence in sentences]
            filename_id = Path(filename).stem
            if not filename_id.isdigit():
                raise ValueError(f"Flickr30k filename must carry original numeric image identity: {filename}")
            role = "train" if split in recipe["train"] else "eval" if split == "test" else None
            if role:
                candidates[role].append(make_row(source, split, filename_id, filename_id,
                    image_path(spec["images"], filename), CAPTION_INSTRUCTION, refs))
        sources[f"{source}/recipe"] = {"split_recipe": recipe, "image_ids_by_role": {
            role: sorted(row["source_image_id"] for row in rows) for role, rows in candidates.items()}}
    else:
        for role, official_split in (("train", "train2014" if source in ("vqav2", "coco_captions") else "train"),
                                     ("eval", "val2014" if source in ("vqav2", "coco_captions") else "val")):
            config = spec.get(role)
            if not isinstance(config, dict):
                raise ValueError(f"Missing {source}/{role} source configuration")
            if source == "vqav2":
                questions = read_source(source_path(base, config["questions"]), sources, f"{source}/{role}/questions", revision)
                annotations = read_source(source_path(base, config["annotations"]), sources, f"{source}/{role}/annotations", revision)
                for document in (questions, annotations):
                    if "data_subtype" in document and document["data_subtype"] != official_split:
                        raise ValueError(f"VQAv2 data_subtype mismatch for {role}")
                qindex = unique_index(questions.get("questions"), "question_id", "VQAv2 questions")
                aindex = unique_index(annotations.get("annotations"), "question_id", "VQAv2 annotations")
                if qindex.keys() != aindex.keys():
                    raise ValueError(f"VQAv2 question/annotation ID mismatch for {role}")
                template = config.get("image_template", f"COCO_{official_split}_{{image_id:012d}}.jpg")
                for qid, question in qindex.items():
                    annotation = aindex[qid]
                    if str(question["image_id"]) != str(annotation["image_id"]):
                        raise ValueError(f"VQAv2 question/annotation image mismatch: {qid}")
                    answers = unique_index(annotation.get("answers"), "answer_id", "VQAv2 answers")
                    if set(answers) != {str(i) for i in range(1, 11)}:
                        raise ValueError(f"VQAv2 requires answer IDs 1..10: {qid}")
                    refs = [answers[str(i)]["answer"] for i in range(1, 11)]
                    target = annotation.get("multiple_choice_answer")
                    if target not in refs:
                        raise ValueError(f"VQAv2 official target is absent from answers: {qid}")
                    image_id = int(question["image_id"])
                    candidates[role].append(make_row(source, official_split, qid, image_id,
                        image_path(config["images"], template.format(image_id=image_id)), question["question"], refs, target))
            elif source == "coco_captions":
                document = read_source(source_path(base, config["annotations"]), sources, f"{source}/{role}/annotations", revision)
                images = unique_index(document.get("images"), "id", "COCO images")
                annotations = unique_index(document.get("annotations"), "id", "COCO captions")
                refs = defaultdict(list)
                for aid in sorted(annotations, key=lambda value: int(value)):
                    annotation = annotations[aid]
                    image_id = str(annotation["image_id"])
                    if image_id not in images:
                        raise ValueError(f"COCO caption refers to missing image: {aid}")
                    refs[image_id].append(annotation["caption"])
                for image_id, image in images.items():
                    candidates[role].append(make_row(source, official_split, image_id, int(image_id),
                        image_path(config["images"], image["file_name"]), CAPTION_INSTRUCTION, refs[image_id]))
            elif source == "gqa":
                questions = read_source(source_path(base, config["questions"]), sources, f"{source}/{role}/questions", revision)
                if not isinstance(questions, dict):
                    raise ValueError("GQA balanced question JSON must be an ID-keyed object")
                template = config.get("image_template", "{image_id}.jpg")
                for qid, question in questions.items():
                    if not isinstance(question, dict) or question.get("isBalanced", True) is not True:
                        raise ValueError(f"GQA source contains unbalanced/malformed question: {qid}")
                    image_id = str(question["imageId"])
                    candidates[role].append(make_row(source, official_split, qid, image_id,
                        image_path(config["images"], template.format(image_id=image_id)), question["question"], [question["answer"]]))
    # Choose one QA per image by a locked answer-independent rank; preserve caption refs.
    for role, rows in candidates.items():
        by_image = defaultdict(list)
        ids = set()
        for row in rows:
            if row["id"] in ids:
                raise ValueError(f"Duplicate normalized source ID: {row['id']}")
            ids.add(row["id"])
            by_image[image_identity(row)].append(row)
        if SOURCE_SPECS[source][0] == "caption" and any(len(group) != 1 for group in by_image.values()):
            raise ValueError(f"Repeated caption image identity in {source}/{role}; references cannot be discarded")
        candidates[role] = [min(group, key=lambda row: rank(seed, "qa", row["id"]))
                            for _, group in sorted(by_image.items())]
    return candidates


class LocalImages(ImageVerifier):
    def __init__(self, root, coco_archives=None, gqa_archive=None, gqa_directories=()):
        super().__init__(root)
        self.archive_root = Path(coco_archives).resolve() if coco_archives is not None else None
        self.gqa_archive = Path(gqa_archive).resolve() if gqa_archive is not None else None
        self.gqa_directories = {PurePosixPath(directory).as_posix() for directory in gqa_directories}
        self.archives = {}
        self.archive_records = {}

    def verify(self, image):
        relative_image(image)
        path = self.root / image
        match = re.fullmatch(r"COCO_(train2014|val2014)_\d{12}\.jpg", Path(image).name)
        archive_path, member, key = None, None, None
        if match and self.archive_root is not None:
            key = match[1]
            archive_path = self.archive_root / f"{key}.zip"
            member = f"{key}/{Path(image).name}"
        elif self.gqa_archive is not None and PurePosixPath(image).parent.as_posix() in self.gqa_directories:
            key, archive_path, member = "gqa", self.gqa_archive, f"images/{Path(image).name}"
        if archive_path is not None and key not in self.archives:
            self.archives[key] = zipfile.ZipFile(archive_path)
            digest = hashlib.sha256()
            with archive_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.archive_records[key] = {"path": str(archive_path), "sha256": digest.hexdigest(),
                                         "transport": "explicit local official ZIP exact member"}
        if not path.is_file() and archive_path is not None:
            if not path.resolve().is_relative_to(self.root):
                raise ValueError(f"Archive destination escapes image root: {image}")
            path.parent.mkdir(parents=True, exist_ok=True)
            # Open the exact member first; never remove a pre-existing/racing file.
            with self.archives[key].open(member) as source:
                with path.open("xb") as destination:
                    try:
                        while block := source.read(1024 * 1024):
                            destination.write(block)
                    except Exception:
                        path.unlink()
                        raise
        return super().verify(image)

    def close(self):
        for archive in self.archives.values():
            archive.close()


def legacy_identity(row, task):
    if "source" in row and "source_image_id" in row and row["source"] in SOURCE_SPECS:
        return image_identity(row)
    name = Path(row["image"]).name
    match = re.fullmatch(r"COCO_(?:train2014|val2014)_(\d+)\.jpg", name)
    if match:
        return f"coco:{int(match[1])}"
    task_name = str(task.get("name", task.get("task_id", ""))).lower()
    if task_name in ("gqa", "g"):
        return f"gqa:{Path(name).stem}"
    if task_name in ("flickr30k", "f"):
        return f"flickr30k:{Path(name).stem}"
    return f"legacy-path:{row['image']}"


def load_history(paths, root_override=None):
    exclusion = {scope: {field: set() for field in ("identities", "hashes", "ids")} for scope in ("all", "eval")}
    records = []
    verifiers = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        root = Path(root_override).resolve() if root_override is not None else Path(manifest.get("image_root", "images")).expanduser()
        if not root.is_absolute():
            root = path.parent / root
        root = root.resolve()
        verifier = verifiers.setdefault(root, ImageVerifier(root))
        counts = Counter()
        verified = []
        for task in manifest["tasks"]:
            for split in SPLITS:
                for row in task.get(split, []):
                    digest = verifier.verify(row["image"])
                    if "image_sha256" in row and row["image_sha256"] != digest:
                        raise ValueError(f"Historical recorded image hash mismatch: {path}/{row['id']}")
                    identity = legacy_identity(row, task)
                    row_id = nonempty(row.get("id"), "historical row ID")
                    for scope in (("all", "eval") if split == "eval" else ("all",)):
                        exclusion[scope]["identities"].add(identity)
                        exclusion[scope]["hashes"].add(digest)
                        exclusion[scope]["ids"].add(row_id)
                    counts[split] += 1
                    verified.append((row["image"], digest))
        records.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "image_root": str(root),
                        "counts": dict(counts), "verified_images_sha256": digest_json(sorted(set(verified)))})
    return exclusion, records


def targets_for(specs, targets, train, validation, evaluation):
    result = {}
    if targets is not None and (not isinstance(targets, dict) or set(targets) != {spec["task_id"] for spec in specs}):
        raise ValueError("--targets must map every configured task_id, and no others, to train/validation/eval counts")
    for spec in specs:
        task_id = spec["task_id"]
        counts = targets[task_id] if targets is not None else dict(zip(SPLITS, (train, validation, evaluation)))
        if not isinstance(counts, dict) or set(counts) != set(SPLITS) or any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError(f"Targets must be nonnegative integer image counts: {task_id}")
        result[task_id] = dict(counts)
    return result


def prepare(config_path, image_root, history_paths, *, seed=20261013, train_target=2048,
            validation_target=256, eval_target=512, targets=None, development_subset=False,
            allow_empty_history=False, history_image_root=None, coco_archives=None, gqa_archive=None):
    config_path = Path(config_path).expanduser().resolve()
    raw_config = config_path.read_bytes()
    config = json.loads(raw_config)
    if not isinstance(config, dict) or type(config.get("schema_version")) is not int or config["schema_version"] != 1 or not isinstance(config.get("sources"), list):
        raise ValueError("Source config requires schema_version=1 and sources list")
    if type(seed) is not int:
        raise ValueError("Sampling seed must be an integer")
    specs = [dict(spec) for spec in config["sources"]]
    seen_sources, seen_tasks = set(), set()
    for spec in specs:
        source = spec.get("type")
        if source not in SOURCE_SPECS or source in seen_sources:
            raise ValueError(f"Unknown or repeated source type: {source}")
        seen_sources.add(source)
        spec.setdefault("task_id", SOURCE_SPECS[source][2])
        task_id = nonempty(spec["task_id"], "task_id")
        if task_id in seen_tasks:
            raise ValueError(f"Repeated task_id: {task_id}")
        seen_tasks.add(task_id)
    if development_subset:
        if not (2 <= len(seen_sources) < 4 and
                {SOURCE_SPECS[source][0] for source in seen_sources} == {"vqa", "caption"}):
            raise ValueError("Development requires an explicit proper subset containing VQA and caption sources")
    elif seen_sources != set(SOURCE_SPECS):
        raise ValueError("Provide all four sources; proper G0 subsets require --development-subset")
    if not history_paths and not (development_subset and allow_empty_history):
        raise ValueError("Supply ALL historical manifests; only explicit development --allow-empty-history may omit them")
    if allow_empty_history and not development_subset:
        raise ValueError("--allow-empty-history is development-only")
    if (coco_archives is not None or gqa_archive is not None) and history_image_root is not None and Path(image_root).resolve() == Path(history_image_root).resolve():
        raise ValueError("Archive extraction must use a NEW image root, not the historical image root")
    counts = targets_for(specs, targets, train_target, validation_target, eval_target)
    history, history_records = load_history(history_paths, history_image_root)
    if (coco_archives is not None or gqa_archive is not None) and any(
            Path(record["image_root"]).resolve() == Path(image_root).resolve() for record in history_records):
        raise ValueError("Archive extraction must not write to any historical image root")
    provenance = {}
    pools = {spec["task_id"]: normalize_source(spec, config_path.parent, provenance, seed) for spec in specs}
    groups = {role: defaultdict(list) for role in ("train", "eval")}
    for task_id, pool in pools.items():
        for role, rows in pool.items():
            for row in rows:
                groups[role][image_identity(row)].append((task_id, row))
    selection = {task_id: {split: [] for split in SPLITS} for task_id in counts}
    stats = {task_id: {"normalized_images": {role: len(rows) for role, rows in pools[task_id].items()},
                        "considered_images": Counter(), "excluded": Counter(), "owned_available": Counter()}
             for task_id in counts}
    used_hashes, used_identities = set(), set()
    gqa_directories = [spec[role]["images"] for spec in specs if spec["type"] == "gqa" for role in ("train", "eval")]
    images = LocalImages(image_root, coco_archives, gqa_archive, gqa_directories)
    total_train = sum(value["train"] for value in counts.values())
    total_validation = sum(value["validation"] for value in counts.values())
    try:
        # Official held-out groups are handled first. Entire held-out identity pool
        # is unavailable to train, not just the selected evaluation sample.
        for role in ("eval", "train"):
            role_splits = ("eval",) if role == "eval" else ("train", "validation")
            for identity in sorted(groups[role], key=lambda value: rank(seed, "candidate", role, value)):
                if all(len(selection[task][split]) == counts[task][split] for task in counts for split in role_splits):
                    break
                entries = groups[role][identity]
                for task_id, row in entries:
                    stats[task_id]["considered_images"][role] += 1
                reason = None
                if role == "train" and identity in groups["eval"]:
                    reason = "official_heldout_identity"
                scope = "all" if role == "eval" else "eval"
                if identity in history[scope]["identities"]:
                    reason = f"historical_{scope}_identity"
                if reason:
                    for task_id, _ in entries:
                        stats[task_id]["excluded"][reason] += 1
                    continue
                # Decode before assignment. Missing/corrupt required candidates abort;
                # availability cannot be used as a filter or replacement reason.
                verified = [(task_id, dict(row, image_sha256=images.verify(row["image"]))) for task_id, row in entries]
                digests = {row["image_sha256"] for _, row in verified}
                reason = f"historical_{scope}_bytes" if digests & history[scope]["hashes"] else None
                if digests & used_hashes or identity in used_identities:
                    reason = "global_image_bytes_or_identity"
                if any(row["id"] in history[scope]["ids"] for _, row in verified):
                    reason = f"historical_{scope}_id"
                if reason:
                    for task_id, _ in entries:
                        stats[task_id]["excluded"][reason] += 1
                    continue
                group_digest = digest_json(sorted(digests))
                if role == "eval":
                    split = "eval"
                else:
                    total = total_train + total_validation
                    value = int(rank(seed, "partition", group_digest), 16)
                    split = "validation" if value * total < total_validation * 2**256 else "train"
                # Ownership is fixed before checking whether a task quota is full.
                # Never pass a filled task's images to another task to hide shortages.
                task_id, row = min(verified, key=lambda entry: rank(seed, "owner", group_digest, entry[0]))
                stats[task_id]["owned_available"][split] += 1
                used_hashes.update(digests)
                used_identities.add(identity)
                if len(selection[task_id][split]) < counts[task_id][split]:
                    selection[task_id][split].append(row)
            # A selected eval byte is excluded from every later train/validation task.
        shortages = []
        for task_id in counts:
            for split in SPLITS:
                actual = len(selection[task_id][split])
                if actual != counts[task_id][split]:
                    shortages.append({"task_id": task_id, "split": split, "requested": counts[task_id][split],
                                      "selected": actual, "shortage": counts[task_id][split] - actual,
                                      "candidate_counts": stats[task_id]})
        if shortages:
            raise ValueError("Insufficient image-disjoint source candidates; no replacement or split substitution. "
                             "Acquire missing official sources or re-lock explicit targets BEFORE validation/confirmation. "
                             + json.dumps(shortages, sort_keys=True))
        tasks = []
        for spec in specs:
            task_id = spec["task_id"]
            family, metric, _ = SOURCE_SPECS[spec["type"]]
            tasks.append({"task_id": task_id, "source": spec["type"], "family": family, "metric": metric,
                          **selection[task_id], "selection": stats[task_id]})
        manifest = {"schema_version": 1,
                    "protocol": "multimodal_development" if development_subset else "multimodal_cl",
                    "confirmation_eligible": not development_subset and all(counts[task] == {"train": 2048, "validation": 256, "eval": 512} for task in counts),
                    "image_root": str(images.root), "sampling_seed": seed, "task_order": [task["task_id"] for task in tasks],
                    "targets": counts, "tasks": tasks, "sources": provenance,
                    "source_config": {"path": str(config_path), "sha256": hashlib.sha256(raw_config).hexdigest(),
                                      "sources": specs}, "history": history_records,
                    "historical_exclusions": {scope: {field: sorted(values) for field, values in fields.items()}
                                              for scope, fields in history.items()},
                    "local_archives": images.archive_records,
                    "split_policy": {"revision": "multimodal-image-groups-v1", "candidate_rank": "seeded identity SHA256",
                                     "partition": "verified byte-group SHA256; global requested train/validation ratio",
                                     "task_assignment": "locked verified byte-group/task SHA256 rank before quotas",
                                     "official_eval_reserved_first": True, "one_qa_per_image": True,
                                     "no_replacement": True, "all_caption_references_preserved": True,
                                     "history_scope": "all historical splits excluded from eval; historical eval excluded from development",
                                     "scope": "all configured tasks in this invocation; unlisted history cannot be excluded"}}
        validate_manifest(manifest)
        return manifest
    finally:
        images.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New manifest path; existing output is refused")
    parser.add_argument("--history", nargs="+", type=Path, default=[])
    parser.add_argument("--history-image-root", type=Path)
    parser.add_argument("--coco-archives", type=Path, help="Explicit local train2014.zip/val2014.zip transport, no downloads")
    parser.add_argument("--gqa-archive", type=Path, help="Explicit local GQA ZIP with images/{image_id}.jpg members")
    parser.add_argument("--train-target", type=int, default=2048)
    parser.add_argument("--validation-target", type=int, default=256)
    parser.add_argument("--eval-target", type=int, default=512)
    parser.add_argument("--targets", type=Path, help="JSON task_id -> {train,validation,eval} exact unique-image counts")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--development-subset", action="store_true", help="Explicit proper G0 source subset; never confirmation")
    parser.add_argument("--allow-empty-history", action="store_true", help="Explicit development-only opt-out; cannot certify historical exclusions")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist; archived manifests are never overwritten")
    targets = json.loads(args.targets.read_bytes()) if args.targets else None
    manifest = prepare(args.source_config, args.image_root, args.history, seed=args.seed,
                       train_target=args.train_target, validation_target=args.validation_target, eval_target=args.eval_target,
                       targets=targets, development_subset=args.development_subset, allow_empty_history=args.allow_empty_history,
                       history_image_root=args.history_image_root, coco_archives=args.coco_archives, gqa_archive=args.gqa_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(f"Wrote {args.output}: " + ", ".join(f"{task['task_id']}=" + "/".join(str(len(task[split])) for split in SPLITS) for task in manifest["tasks"]))


if __name__ == "__main__":
    main()
