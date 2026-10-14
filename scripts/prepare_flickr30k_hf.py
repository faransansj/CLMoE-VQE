#!/usr/bin/env python3
"""Convert the pinned nlphuji/flickr30k mirror subset, without network access.

Usage: python -m scripts.prepare_flickr30k_hf --csv CSV --archive ZIP
       --output NEW_JSON --image-root NEW_DIR

NEW_DIR contains filename.jpg directly. The annotation JSON is accepted by the
existing Flickr30k adapter; NEW_JSON's sibling .sources.json records provenance.
This mirror has 31,014 images, not the original 31,783-image collection. Preserve
CSV row splits (29,000 train / 1,014 val / 1,000 test), not HF's top-level TEST.
Outputs must be new. Interrupted extraction can resume only over identical image
bytes. Downloaded CSV/ZIP files are never modified. Fixture helpers deliberately
do not claim the production source hashes; the CLI always checks both pins.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from PIL import Image

REVISION = "2b239befc81b6e3f035ce6bd52f5f4d60f5625f7"
CSV_SHA256 = "395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137"
ARCHIVE_SHA256 = "2ce2420c0d17f0531deaa89ac657b4d5067ec519da16ff1ea12acbf5619c7391"
SPLIT_COUNTS = {"train": 29000, "val": 1014, "test": 1000}
FIELDS = {"raw", "sentids", "split", "filename", "img_id"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Source SHA256 mismatch for {path}: expected {expected}, got {actual}")
    return {"path": str(Path(path).resolve()), "sha256": actual,
            "bytes": Path(path).stat().st_size}


def read_annotations(path, *, expected_counts=None):
    """Validate CSV records, retaining caption text, reference order and row split."""
    images = []
    filenames, imgids, sentids = set(), set(), set()
    counts = Counter()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, strict=True)
        if rows.fieldnames is None or len(rows.fieldnames) != len(FIELDS) or set(rows.fieldnames) != FIELDS:
            raise ValueError("CSV fields must be raw,sentids,split,filename,img_id")
        for number, row in enumerate(rows, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV field count at row {number}")
            filename = row["filename"]
            if not re.fullmatch(r"[0-9]+\.jpg", filename):
                raise ValueError(f"Unsafe or malformed filename at row {number}: {filename!r}")
            if not re.fullmatch(r"0|[1-9][0-9]*", row["img_id"]):
                raise ValueError(f"Invalid img_id at row {number}")
            imgid = int(row["img_id"])
            if filename in filenames or imgid in imgids:
                raise ValueError(f"Duplicate filename or imgid at row {number}")
            if row["split"] not in SPLIT_COUNTS:
                raise ValueError(f"Invalid split at row {number}: {row['split']!r}")
            try:
                raw, ids = json.loads(row["raw"]), json.loads(row["sentids"])
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid caption/sentids JSON at row {number}") from error
            if not isinstance(raw, list) or len(raw) != 5 or not all(isinstance(value, str) and value.strip() for value in raw):
                raise ValueError(f"Expected five nonempty raw captions at row {number}")
            if not isinstance(ids, list) or len(ids) != 5 or not all(type(value) is int and value >= 0 for value in ids):
                raise ValueError(f"Expected five nonnegative integer sentids at row {number}")
            if len(set(ids)) != 5 or sentids.intersection(ids):
                raise ValueError(f"Duplicate sentence ID at row {number}")
            filenames.add(filename)
            imgids.add(imgid)
            sentids.update(ids)
            counts[row["split"]] += 1
            images.append({"imgid": imgid, "filename": filename, "split": row["split"],
                           "sentences": [{"raw": text, "sentid": sid} for text, sid in zip(raw, ids)]})
    if not images:
        raise ValueError("CSV contains no images")
    if expected_counts is not None and dict(counts) != expected_counts:
        raise ValueError(f"CSV split counts mismatch: expected {expected_counts}, got {dict(counts)}")
    return images


def extract_images(archive, images, image_root):
    """Extract exact CSV-named members; helpers also support genuinely small fixtures."""
    root = Path(image_root)
    if root.is_symlink():
        raise ValueError(f"Image root must not be a symlink: {root}")
    expected = {}
    for image in images:
        filename = image["filename"]
        if not isinstance(filename, str) or not re.fullmatch(r"[0-9]+\.jpg", filename):
            raise ValueError(f"Unsafe image filename: {filename!r}")
        if filename in expected:
            raise ValueError(f"Duplicate image filename: {filename}")
        expected[filename] = f"flickr30k-images/{filename}"
    hashes = {}
    with zipfile.ZipFile(archive) as source:
        members = {}
        for info in source.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if (not name or name.startswith("/") or "\\" in name or "\x00" in name
                    or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
                    or not parts or ":" in parts[0]):
                raise ValueError(f"Unsafe archive member: {name!r}")
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Non-regular archive member: {name}")
            if name in members:
                raise ValueError(f"Duplicate archive member: {name}")
            members[name] = info
        for filename, member in expected.items():
            if member not in members or members[member].is_dir():
                raise ValueError(f"Missing exact image archive member: {member}")
            destination = root / filename
            if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                raise ValueError(f"Unsafe image destination: {destination}")
        root.mkdir(parents=True, exist_ok=True)
        for filename in sorted(expected):
            data = source.read(members[expected[filename]])
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValueError(f"Expected JPEG image: {filename}")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
            digest = hashlib.sha256(data).hexdigest()
            destination = root / filename
            try:
                descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file() or sha256_file(destination) != digest:
                    raise ValueError(f"Conflicting existing image bytes: {destination}") from None
            else:
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(data)
                except BaseException:
                    destination.unlink()
                    raise
            # Scratch NAS expires files by atime; do not inherit old ZIP timestamps.
            os.utime(destination, None, follow_symlinks=False)
            hashes[filename] = digest
    return hashes


def write_new_json(path, value):
    """Never truncate a pre-existing output, including a dangling symlink."""
    with Path(path).open("x", encoding="utf-8") as stream:
        try:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        except BaseException:
            Path(path).unlink()
            raise


def convert(csv_path, archive_path, output, image_root):
    """Production entry point: fixed source hashes and full mirror counts mandatory."""
    csv_path, archive_path, output = Path(csv_path), Path(archive_path), Path(output)
    provenance_path = output.with_suffix(".sources.json")
    if output == provenance_path:
        raise ValueError("Annotation and provenance output paths must differ")
    for path in (output, provenance_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output must be new: {path}")
    sources = {"csv": verify_source(csv_path, CSV_SHA256),
               "archive": verify_source(archive_path, ARCHIVE_SHA256)}
    images = read_annotations(csv_path, expected_counts=SPLIT_COUNTS)
    hashes = extract_images(archive_path, images, image_root)
    aggregate = hashlib.sha256(json.dumps(sorted(hashes.items()), ensure_ascii=False,
                                        separators=(",", ":")).encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(output, {"images": images})
    provenance = {"schema_version": 1, "dataset": "nlphuji/flickr30k",
                  "revision": REVISION, "scope": "31,014-image HF mirror subset; not original 31,783-image collection",
                  "sources": sources, "image_count": len(images), "caption_count": 5 * len(images),
                  "split_counts": dict(Counter(image["split"] for image in images)),
                  "split_policy": "CSV row split preserved; HF top-level TEST ignored",
                  "annotations": {"path": str(output.resolve()), "sha256": sha256_file(output)},
                  "image_root": str(Path(image_root).resolve()),
                  "image_sha256": hashes, "extracted_sha256": aggregate,
                  "extracted_sha256_recipe": "SHA256 of UTF-8 compact JSON sorted [filename, SHA256] pairs"}
    write_new_json(provenance_path, provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        provenance = convert(args.csv, args.archive, args.output, args.image_root)
    except (ValueError, OSError, csv.Error, zipfile.BadZipFile) as error:
        parser.exit(1, f"Flickr30k preparation failed: {error}\n")
    print(json.dumps({"output": str(args.output), "sources": str(args.output.with_suffix('.sources.json')),
                      "image_count": provenance["image_count"], "split_counts": provenance["split_counts"],
                      "extracted_sha256": provenance["extracted_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
