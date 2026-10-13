#!/usr/bin/env python3
"""Prepare fixed, source-backed VQA subsets; image acquisition is opt-in.

Run on the Yonsei login node. CL4VQA annotations must already be downloaded
under ROOT/CL4VQA. No dataset split or image is synthesized by this script.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


HF_REVISION = "8acf4f538c372af7193e296d05ce37bb46eb6968"
HF_BASE = (
    "https://huggingface.co/datasets/zackie29/SMoLoRA/resolve/"
    f"{HF_REVISION}/Instructions_Single"
)
CL4_TASKS = (
    "recognition", "location", "judge", "commonsense", "count",
    "action", "color", "type", "subcategory", "causal",
)
STREAM_TASKS = (
    ("ScienceQA", "test", "scienceqa"),
    ("TextVQA", "val", "textvqa"),
    ("GQA", "test", "gqa"),
    ("VQAv2", "val", "vqav2_exact"),
)
TEXTVQA_REFERENCE = "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json"
SCIENCEQA_REFERENCE = (
    "https://raw.githubusercontent.com/lupantech/ScienceQA/main/data/scienceqa/problems.json"
)
GQA_QUESTIONS = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_IMAGES = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
TEXTVQA_IMAGES = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"
COCO_NAME = re.compile(r"COCO_(train2014|val2014)_\d{12}\.jpg\Z")


def announce(message):
    print(message, file=sys.stderr, flush=True)


def request(url, headers=None):
    return urlopen(Request(url, headers={
        "User-Agent": "sema-vqa-data-preparation/1", "Accept-Encoding": "identity",
        **(headers or {}),
    }), timeout=120)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination):
    """Resume an interrupted download, never treating a partial file as complete."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    announce(f"Downloading {url} -> {destination} (resume offset {offset})")
    with request(url, headers) as response:
        if response.status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or int(match[1]) != offset:
                raise RuntimeError(f"Invalid resume response for {url}")
            expected = int(match[3])
            mode = "ab" if offset else "wb"
        elif response.status == 200:
            # A server without range support must restart, not append a full file.
            offset = 0
            size = response.headers.get("Content-Length")
            expected = int(size) if size is not None else None
            mode = "wb"
        else:
            raise RuntimeError(f"Unexpected HTTP status {response.status}: {url}")
        with partial.open(mode) as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
    if not partial.stat().st_size or (expected is not None and partial.stat().st_size != expected):
        raise RuntimeError(f"Incomplete download retained for resume: {partial}")
    partial.replace(destination)


class RangeUnavailable(Exception):
    pass


def range_request(url, headers):
    try:
        return request(url, headers)
    except HTTPError as error:
        if error.code in (400, 403, 405, 416, 501):
            error.close()
            raise RangeUnavailable(f"server rejected range request with HTTP {error.code}") from error
        raise


class HTTPRangeReader(io.RawIOBase):
    """Seekable, bounded-cache HTTP reader for zipfile's central-directory access."""

    BLOCK_SIZE = 1024 * 1024

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.position = 0
        self.blocks = OrderedDict()
        with range_request(url, {"Range": "bytes=0-0"}) as response:
            if response.status != 206:
                raise RangeUnavailable(f"server returned HTTP {response.status} instead of 206")
            match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if not match or len(response.read(2)) != 1:
                raise RangeUnavailable("server did not return a valid byte-range response")
            self.size = int(match[1])
            self.etag = response.headers.get("ETag")
            self.modified = response.headers.get("Last-Modified")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        position = offset + {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[whence]
        if position < 0:
            raise ValueError("Negative archive seek")
        self.position = position
        return position

    def _block(self, index):
        if index in self.blocks:
            self.blocks.move_to_end(index)
            return self.blocks[index]
        start = index * self.BLOCK_SIZE
        end = min(start + self.BLOCK_SIZE, self.size) - 1
        headers = {"Range": f"bytes={start}-{end}"}
        validator = self.etag or self.modified
        if validator:
            headers["If-Range"] = validator
        with range_request(self.url, headers) as response:
            if response.status != 206:
                raise RangeUnavailable("byte ranges stopped working or archive changed")
            expected_range = f"bytes {start}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected_range:
                raise RuntimeError(f"Archive range mismatch: {self.url}")
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError(f"Truncated archive range: {self.url}")
        self.blocks[index] = data
        if len(self.blocks) > 4:
            self.blocks.popitem(last=False)
        return data

    def read(self, size=-1):
        if size is None or size < 0:
            size = max(0, self.size - self.position)
        size = min(size, max(0, self.size - self.position))
        result = bytearray()
        while size:
            index, offset = divmod(self.position, self.BLOCK_SIZE)
            block = self._block(index)
            count = min(size, len(block) - offset)
            result.extend(block[offset:offset + count])
            self.position += count
            size -= count
        return bytes(result)


@contextmanager
def open_archive(url, cache):
    """Prefer ranges; fall back explicitly to a resumable full archive download."""
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".zip")
    if local.is_file() and local.stat().st_size:
        with zipfile.ZipFile(local) as archive:
            yield archive
        return
    reader = None
    try:
        reader = HTTPRangeReader(url)
        archive = zipfile.ZipFile(reader)
    except RangeUnavailable as error:
        if reader is not None:
            reader.close()
        announce(f"HTTP range access unavailable ({error}); FULL ZIP FALLBACK: {url}")
        download(url, local)
        with zipfile.ZipFile(local) as archive:
            yield archive
        return
    try:
        with archive:
            yield archive
    finally:
        reader.close()


def member_index(archive):
    """Index basenames once; ambiguous suffixes are errors, not guessed mappings."""
    result = defaultdict(list)
    for info in archive.infolist():
        if not info.is_dir():
            result[PurePosixPath(info.filename).name].append(info.filename)
    return result


def match_member(index, suffix):
    candidates = [name for name in index[PurePosixPath(suffix).name]
                  if name == suffix or name.endswith("/" + suffix)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix!r}; found {candidates[:10]}")
    return candidates[0]


def extract_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with archive.open(member) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if not partial.stat().st_size:
        raise ValueError(f"Empty archive member: {member}")
    partial.replace(destination)


def extract_archive_members(url, cache, members):
    try:
        with open_archive(url, cache) as archive:
            index = member_index(archive)
            for suffix, destination in members:
                extract_member(archive, match_member(index, suffix), destination)
    except RangeUnavailable as error:
        # A server can stop honoring ranges after the central directory request.
        announce(f"HTTP range access failed ({error}); FULL ZIP FALLBACK: {url}")
        local = cache / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".zip")
        download(url, local)
        with zipfile.ZipFile(local) as archive:
            index = member_index(archive)
            for suffix, destination in members:
                if not destination.is_file() or not destination.stat().st_size:
                    extract_member(archive, match_member(index, suffix), destination)


def load_json(path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def record_source(sources, key, path, root, url=None, **metadata):
    sources[key] = {"path": str(path.relative_to(root)), "sha256": sha256(path),
                    "bytes": path.stat().st_size, **metadata}
    if url is not None:
        sources[key]["url"] = url


def fetch_json(root, sources, key, url, relative_path):
    path = root / "cache" / relative_path
    download(url, path)
    data = load_json(path)
    record_source(sources, key, path, root, url)
    return data


def gqa_reference(root, sources):
    destination = root / "cache" / "references" / "testdev_balanced_questions.json"
    if not destination.is_file():
        extract_archive_members(GQA_QUESTIONS, root / "cache" / "archives",
                                [("testdev_balanced_questions.json", destination)])
    record_source(sources, "GQA/reference", destination, root, GQA_QUESTIONS,
                  archive_member_suffix="testdev_balanced_questions.json")
    data = load_json(destination)
    if not isinstance(data, dict):
        raise ValueError("GQA reference must be a question-ID-keyed object")
    return data


def relative_image(value):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid image path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe or unknown image path schema: {value!r}")
    return str(path)


def source_identifier(record, index, evaluation):
    if evaluation:
        if "question_id" not in record:
            raise ValueError(f"Evaluation row {index} has no question_id")
        identifier = record["question_id"]
    else:
        identifier = record.get("id", record.get("question_id"))
    if identifier is None or isinstance(identifier, (dict, list, bool)):
        raise ValueError(f"Row {index} has no supported source identifier")
    return str(identifier)


def normalize(records, protocol, task, split):
    if not isinstance(records, list):
        raise ValueError(f"{task}/{split}: official instructions must be a JSON array")
    rows = []
    filtered = 0
    seen_eval = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{task}/{split} row {index} is not an object")
        source_id = source_identifier(record, index, split == "eval")
        if split == "eval":
            if source_id in seen_eval:
                raise ValueError(f"Duplicate evaluation question_id in {task}: {source_id}")
            seen_eval.add(source_id)
        image = record.get("image")
        if not image:
            if task == "ScienceQA":
                filtered += 1
                continue
            raise ValueError(f"{task}/{split} source row {index} has no image")
        image = relative_image(image)
        base = {"image": image, "source_id": source_id, "source_row_index": index}
        prefix = f"{protocol}/{task}/{split}/{quote(source_id, safe='')}"
        if split == "eval":
            question, answer = record.get("text"), record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(f"{task}/eval row {index}: expected text and answer strings")
            rows.append({**base, "id": prefix, "source_question_id": source_id,
                         "question": question, "answer": answer})
            continue
        turns = record.get("conversations")
        if not isinstance(turns, list) or not turns or len(turns) % 2:
            raise ValueError(f"{task}/train row {index}: expected complete human/gpt pairs")
        for turn in range(0, len(turns), 2):
            human, assistant = turns[turn:turn + 2]
            if (not isinstance(human, dict) or not isinstance(assistant, dict)
                    or human.get("from") != "human" or assistant.get("from") != "gpt"
                    or not isinstance(human.get("value"), str)
                    or not isinstance(assistant.get("value"), str)):
                raise ValueError(f"{task}/train row {index}, turn {turn}: unsupported conversation schema")
            rows.append({**base, "id": f"{prefix}/row-{index}/turn-{turn // 2}",
                         "turn_index": turn // 2, "question": human["value"],
                         "answer": assistant["value"]})
    return rows, {"source_records": len(records), "excluded_no_image_records": filtered,
                  "image_bearing_records": len(records) - filtered,
                  "normalized_turns": len(rows)}


def select_subset(rows, limit, seed, protocol, task, split):
    derived = hashlib.sha256(f"{seed}/{protocol}/{task}/{split}".encode()).hexdigest()
    if limit and limit < len(rows):
        indices = sorted(random.Random(int(derived, 16)).sample(range(len(rows)), limit))
        selected = [rows[index] for index in indices]
    else:
        selected = rows
    if not selected:
        raise ValueError(f"No eligible rows in {task}/{split}")
    identifiers = [row["id"] for row in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Duplicate normalized IDs in {task}/{split}")
    return selected, {"requested_limit": limit, "selected_count": len(selected),
                      "selected_ids_sha256": hashlib.sha256(
                          json.dumps(identifiers, ensure_ascii=False, separators=(",", ":")).encode()
                      ).hexdigest()}


def enrich_references(task, split, rows, reference):
    if task == "ScienceQA":
        for row in rows:
            problem = reference[row["source_id"]]
            choices, answer = problem["choices"], problem["answer"]
            if (not isinstance(choices, list) or not 2 <= len(choices) <= 5
                    or not all(isinstance(choice, str) for choice in choices)
                    or type(answer) is not int or not 0 <= answer < len(choices)):
                raise ValueError(f"Invalid original ScienceQA choices/answer: {row['source_id']}")
            if row["answer"].strip().upper() != chr(ord("A") + answer):
                raise ValueError(f"ScienceQA target/reference disagreement: {row['source_id']}")
            row.update(choices=choices, answer_index=answer)
    elif task == "TextVQA" and split == "eval":
        for row in rows:
            answers = reference[row["source_question_id"]]["answers"]
            if not isinstance(answers, list) or len(answers) != 10 or not all(isinstance(a, str) for a in answers):
                raise ValueError(f"TextVQA requires ten original answers: {row['source_question_id']}")
            row["answers"] = answers
    elif task == "GQA" and split == "eval":
        for row in rows:
            question = reference[row["source_question_id"]]
            if question.get("isBalanced") is not True or not isinstance(question.get("answer"), str):
                raise ValueError(f"GQA eval question lacks balanced official reference: {row['source_question_id']}")
            row["instruction_answer"] = row["answer"]
            row["answer"] = question["answer"]


def image_identity(image):
    # CL4VQA and SMoLoRA may use different directory prefixes for the same COCO file.
    name = PurePosixPath(image).name
    return name if COCO_NAME.fullmatch(name) else image


def overlap_report(train, evaluation):
    train_images = {image_identity(row["image"]) for row in train}
    eval_images = {image_identity(row["image"]) for row in evaluation}
    common = sorted(train_images & eval_images)
    return {"train_unique_images": len(train_images), "eval_unique_images": len(eval_images),
            "overlap_count": len(common), "overlap_images": common,
            "split_independence_claimed": False}


def cl4_source(root, task, split):
    filename = f"{'train' if split == 'train' else 'test'}_q_{task}.json"
    matches = sorted((root / "CL4VQA").rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Need exactly one official {filename} under {root / 'CL4VQA'}; found {matches}. "
            "Download the official Drive train/test folders first; no partitions are generated."
        )
    return matches[0]


def prepare(args):
    root = args.root
    sources = {}
    tasks = []
    specifications = [(name, "test", "clmoe_exact") for name in CL4_TASKS] if args.protocol == "cl4vqa" else STREAM_TASKS
    for task, eval_split, metric in specifications:
        reference = None
        if args.protocol == "vqa_stream":
            if task == "ScienceQA":
                reference = fetch_json(root, sources, "ScienceQA/reference", SCIENCEQA_REFERENCE,
                                       Path("references/scienceqa_problems.json"))
            elif task == "TextVQA":
                original = fetch_json(root, sources, "TextVQA/reference", TEXTVQA_REFERENCE,
                                      Path("references/TextVQA_0.5.1_val.json"))
                reference = {}
                for item in original["data"]:
                    key = str(item["question_id"])
                    if key in reference:
                        raise ValueError(f"Duplicate original TextVQA question ID: {key}")
                    reference[key] = item
            elif task == "GQA":
                reference = gqa_reference(root, sources)
        entry = {"name": task, "metric": metric, "selection": {}}
        for split, source_split, limit in (("train", "train", args.train_limit), ("eval", eval_split, args.eval_limit)):
            key = f"{task}/{split}"
            if args.protocol == "cl4vqa":
                path = cl4_source(root, task, split)
                records = load_json(path)
                record_source(sources, key, path, root, official_partition=path.name)
            else:
                url = f"{HF_BASE}/{task}/{source_split}.json"
                records = fetch_json(root, sources, key, url,
                                     Path("smolora") / HF_REVISION / task / f"{source_split}.json")
                sources[key]["revision"] = HF_REVISION
            rows, counts = normalize(records, args.protocol, task, split)
            del records
            sources[key]["source_records"] = counts["source_records"]
            selected, selection = select_subset(rows, limit, args.subset_seed, args.protocol, task, split)
            del rows
            if reference is not None:
                enrich_references(task, split, selected, reference)
            entry[split] = selected
            entry["selection"][split] = {**counts, **selection, "source_split": source_split}
            announce(f"{task}/{split}: {counts['source_records']} source records -> "
                     f"{counts['normalized_turns']} image-bearing QA turns -> {len(selected)} selected")
        entry["selected_image_overlap"] = overlap_report(entry["train"], entry["eval"])
        tasks.append(entry)
    eval_ids = [row["id"] for task in tasks for row in task["eval"]]
    if len(eval_ids) != len(set(eval_ids)):
        raise ValueError("Duplicate evaluation IDs across manifest")
    return {
        "protocol": args.protocol,
        "protocol_description": (
            "Official CL4VQA ten-task order and existing train/test partitions; fixed POC subsets."
            if args.protocol == "cl4vqa" else
            "SMoLoRA Instructions_Single VQA-only filtered stream: ScienceQA, TextVQA, GQA, VQAv2; "
            "not the full published six-task protocol. ScienceQA is restricted to image-bearing records."
        ),
        "image_root": "images", "subset_seed": args.subset_seed,
        "subset_policy": "Per-task/split deterministic sampling without replacement after all-turn conversion; "
                         "independent of optimizer/model seed; limit 0 means all eligible rows.",
        "conversion": "Every alternating human/gpt pair becomes a single-turn image-conditioned row. "
                      "All pairs are preserved before sampling; prior-turn context is not replayed. "
                      "Question/answer strings, including OCR and image markers, are preserved. "
                      "Training IDs include source row index and turn index; evaluation IDs are task-namespaced. "
                      "ScienceQA records without images are excluded and counted, not assigned fake images.",
        "sources": sources, "tasks": tasks,
        "selected_image_overlap_all_tasks": overlap_report(
            [row for task in tasks for row in task["train"]],
            [row for task in tasks for row in task["eval"]],
        ),
    }


def image_archive(image):
    path = PurePosixPath(image)
    parts = path.parts
    if len(parts) == 5 and parts[:2] == ("ScienceQA", "images") and parts[2] in ("train", "test"):
        return (f"https://scienceqa.s3.us-west-1.amazonaws.com/images/{parts[2]}.zip",
                "/".join(parts[3:]))
    if len(parts) == 3 and parts[:2] == ("TextVQA", "train_images"):
        return TEXTVQA_IMAGES, path.name
    if len(parts) == 3 and parts[:2] == ("GQA", "images"):
        return GQA_IMAGES, path.name
    raise ValueError(f"Unknown image acquisition schema: {image!r}; inspect original source, do not guess")


def acquire_images(root, images):
    archives = defaultdict(list)
    acquisition = {}
    direct = []
    for image in images:
        destination = root / "images" / image
        name = PurePosixPath(image).name
        match = COCO_NAME.fullmatch(name)
        if match:
            # The official COCO host advertises HTTP; its HTTPS certificate does
            # not cover this hostname. Do not disable TLS verification globally.
            url = f"http://images.cocodataset.org/{match[1]}/{name}"
            acquisition.setdefault("COCO2014", {"url_template": "http://images.cocodataset.org/{split}/{filename}"})
            direct.append((url, destination))
        else:
            url, suffix = image_archive(image)
            acquisition.setdefault(url, {"url": url, "method": "HTTP ranges or explicit resumable full ZIP fallback"})
            if not destination.is_file() or not destination.stat().st_size:
                archives[url].append((suffix, destination))
    # Bounded I/O concurrency; no GPU/CPU-node download or unbounded workers.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, url, destination) for url, destination in direct]
        for future in futures:
            future.result()
    for url, members in archives.items():
        extract_archive_members(url, root / "cache" / "archives", members)
    return acquisition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Shared data root (not the CL4VQA child directory)")
    parser.add_argument("--protocol", choices=("cl4vqa", "vqa_stream"), required=True)
    parser.add_argument("--train-limit", type=int, default=512, help="QA turns per task; 0 selects all")
    parser.add_argument("--eval-limit", type=int, default=128, help="Questions per task; 0 selects all")
    parser.add_argument("--subset-seed", type=int, default=2026)
    parser.add_argument("--download-images", action="store_true", help="Acquire selected real images; safe to rerun after manifest-only preparation")
    args = parser.parse_args()
    if args.train_limit < 0 or args.eval_limit < 0:
        parser.error("Subset limits must be nonnegative (zero means full)")
    args.root = args.root.expanduser().resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args)
    images = sorted({row["image"] for task in manifest["tasks"] for split in ("train", "eval") for row in task[split]})
    (args.root / "images").mkdir(exist_ok=True)
    if args.download_images:
        manifest["image_acquisition_sources"] = acquire_images(args.root, images)
    missing = [image for image in images if not (args.root / "images" / image).is_file()
               or not (args.root / "images" / image).stat().st_size]
    manifest["images"] = {"download_requested": args.download_images, "selected_unique_paths": len(images),
                          "present_count": len(images) - len(missing), "missing_count": len(missing),
                          "missing_paths": missing, "ready": not missing}
    if args.download_images and missing:
        raise RuntimeError(f"Selected images are missing after acquisition: {missing[:10]}")
    if missing:
        announce(f"MANIFEST ONLY: {len(missing)} selected images are missing. Rerun with --download-images before training.")
    destination = args.root / "manifests" / f"{args.protocol}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.part")
    with partial.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    partial.replace(destination)
    announce(f"Wrote {destination}")


if __name__ == "__main__":
    main()
