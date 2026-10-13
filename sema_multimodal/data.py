"""Validated image-disjoint multimodal manifests and metadata-free model views."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re

from PIL import Image

CAPTION_INSTRUCTION = "Describe the image."
SPLITS = ("train", "validation", "eval")
SOURCE_SPECS = {
    "vqav2": ("vqa", "vqav2", "V"),
    "coco_captions": ("caption", "cider_d", "C"),
    "gqa": ("vqa", "gqa", "G"),
    "flickr30k": ("caption", "cider_d", "F"),
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def relative_image(value):
    nonempty(value, "image")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value != path.as_posix():
        raise ValueError(f"image must be a normalized relative path: {value!r}")
    return value


def image_identity(row):
    """COCO question/caption sources share one identity namespace."""
    source = row["source"]
    identifier = str(row["source_image_id"])
    if source in ("vqav2", "coco_captions"):
        if not identifier.isdigit():
            raise ValueError(f"COCO image ID must be numeric: {identifier!r}")
        return f"coco:{int(identifier)}"
    return f"{source}:{identifier}"


class ImageVerifier:
    """Hash and decode current local bytes, without trusting filesystem timestamps."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def verify(self, image):
        relative_image(image)
        path = (self.root / image).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"Image escapes root through a symlink: {image}")
        if not path.is_file():
            raise FileNotFoundError(f"Required local image missing: {path}. Acquire the official image; "
                                    "no download or availability-based replacement is performed.")
        before = path.stat()
        proof = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        try:
            with Image.open(path) as decoded:
                decoded.load()
                if min(decoded.size) <= 0:
                    raise ValueError("empty decoded image")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot decode required local image {path}: {exc}") from exc
        after = path.stat()
        if proof != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise RuntimeError(f"Image changed during decode/hash: {path}")
        result = digest.hexdigest()
        return result


def validate_manifest(manifest, image_root=None):
    """Reject malformed rows, leakage and mismatched real image hashes."""
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected multimodal manifest schema_version=1")
    protocol = manifest.get("protocol")
    if protocol not in ("multimodal_cl", "multimodal_development"):
        raise ValueError("Unknown multimodal protocol")
    if protocol == "multimodal_development" and manifest.get("confirmation_eligible") is not False:
        raise ValueError("Development subset must explicitly be non-confirmation")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty ordered list")
    task_ids, sources, row_ids, identities, hashes, paths = set(), set(), set(), set(), set(), set()
    verifier = ImageVerifier(image_root) if image_root is not None else None
    history = manifest.get("historical_exclusions", {})
    if not isinstance(history, dict):
        raise ValueError("historical_exclusions must be an object")
    history_sets = {}
    for scope in ("eval", "all"):
        exclusion = history.get(scope, {})
        if not isinstance(exclusion, dict):
            raise ValueError(f"historical_exclusions/{scope} must be an object")
        history_sets[scope] = {}
        for field in ("identities", "hashes", "ids"):
            values = exclusion.get(field, [])
            if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"historical_exclusions/{scope}/{field} must be a string list")
            if field == "hashes" and not all(SHA256.fullmatch(v) for v in values):
                raise ValueError("Malformed historical image SHA256")
            history_sets[scope][field] = set(values)
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be an object")
        task_id = nonempty(task.get("task_id"), "task_id")
        if task_id in task_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        task_ids.add(task_id)
        task_source = None
        for split in SPLITS:
            rows = task.get(split)
            if not isinstance(rows, list):
                raise ValueError(f"{task_id}/{split} must be a row list")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"{task_id}/{split}: row must be an object")
                for field in ("id", "source", "source_split", "family", "source_image_id", "input_text", "target", "image_sha256"):
                    nonempty(row.get(field), field)
                source = row["source"]
                if source not in SOURCE_SPECS:
                    raise ValueError(f"Unsupported source: {source}")
                family, metric, _ = SOURCE_SPECS[source]
                if row["family"] != family or task.get("family") != family or task.get("metric") != metric:
                    raise ValueError(f"Source/family/metric mismatch: {row['id']}")
                if task_source is not None and task_source != source:
                    raise ValueError(f"Mixed sources in task {task_id}")
                task_source = source
                if not row["id"].startswith(f"{source}/{row['source_split']}/") or row["id"].endswith("/"):
                    raise ValueError(f"ID must include source/split namespace: {row['id']}")
                allowed = {"vqav2": ("train2014", "val2014"), "coco_captions": ("train2014", "val2014"),
                           "gqa": ("train", "val"), "flickr30k": ("train", "test")}[source]
                # Flickr's explicitly pinned recipe may include restval; never use its val as eval.
                if source == "flickr30k" and split != "eval":
                    allowed = ("train", "restval")
                expected = allowed[-1] if split == "eval" else allowed[0]
                if row["source_split"] != expected and not (source == "flickr30k" and split != "eval" and row["source_split"] == "restval"):
                    raise ValueError(f"Nonofficial source split for {split}: {row['id']}")
                refs = row.get("references")
                if not isinstance(refs, list) or not refs or not all(isinstance(v, str) and v.strip() for v in refs):
                    raise ValueError(f"references must contain all nonempty source strings: {row['id']}")
                if source == "vqav2" and len(refs) != 10:
                    raise ValueError(f"VQAv2 requires ten official answers: {row['id']}")
                if source == "gqa" and len(refs) != 1:
                    raise ValueError(f"GQA requires its single official answer: {row['id']}")
                if row["target"] not in refs:
                    raise ValueError(f"Target is absent from source references: {row['id']}")
                if family == "caption" and row["input_text"] != CAPTION_INSTRUCTION:
                    raise ValueError(f"Caption instruction is not fixed: {row['id']}")
                if family == "vqa" and row["input_text"] == CAPTION_INSTRUCTION:
                    raise ValueError(f"VQA question collides with caption request format: {row['id']}")
                image = relative_image(row.get("image"))
                digest = row["image_sha256"]
                if not SHA256.fullmatch(digest):
                    raise ValueError(f"Invalid image_sha256: {row['id']}")
                identity = image_identity(row)
                for label, value, seen in (("row ID", row["id"], row_ids), ("image identity", identity, identities),
                                           ("image bytes", digest, hashes), ("image path", image, paths)):
                    if value in seen:
                        raise ValueError(f"Duplicate {label} across tasks/splits: {value}")
                    seen.add(value)
                scopes = ("eval", "all") if split == "eval" else ("eval",)
                for scope in scopes:
                    old = history_sets[scope]
                    if identity in old["identities"] or digest in old["hashes"] or row["id"] in old["ids"]:
                        raise ValueError(f"Historical {scope} leakage: {row['id']}")
                if verifier is not None and verifier.verify(image) != digest:
                    raise ValueError(f"Actual image SHA256 mismatch: {row['id']}")
        source = task.get("source", task_source)
        if source not in SOURCE_SPECS:
            raise ValueError(f"Task source is missing/unsupported: {task_id}")
        if task_source is not None and source != task_source:
            raise ValueError(f"Task source mismatch: {task_id}")
        if (task.get("family"), task.get("metric")) != SOURCE_SPECS[source][:2]:
            raise ValueError(f"Task family/metric mismatch: {task_id}")
        if source in sources:
            raise ValueError(f"Duplicate source task: {source}")
        sources.add(source)
    if protocol == "multimodal_cl" and sources != set(SOURCE_SPECS):
        raise ValueError("Full multimodal_cl requires all four official sources")
    if protocol == "multimodal_development" and not (2 <= len(sources) < 4 and
            {SOURCE_SPECS[source][0] for source in sources} == {"vqa", "caption"}):
        raise ValueError("Development requires an explicit proper subset containing VQA and caption sources")
    if "task_order" in manifest and manifest["task_order"] != [task["task_id"] for task in tasks]:
        raise ValueError("task_order disagrees with ordered tasks")
    return manifest


def public_rows(rows):
    """Only legal request data reaches the backbone and router."""
    return [{"image": row["image"], "question": row["input_text"]} for row in rows]


def training_rows(rows, seed, epoch):
    """Cycle all caption references without adding image exposures."""
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("seed and nonnegative epoch must be integers")
    result = []
    for row in rows:
        target = row["target"]
        if row["family"] == "caption":
            refs = row["references"]
            offset = int.from_bytes(hashlib.sha256(f"{seed}/{image_identity(row)}".encode()).digest(), "big")
            target = refs[(offset + epoch) % len(refs)]
        result.append({"image": row["image"], "question": row["input_text"], "answer": target})
    return result
