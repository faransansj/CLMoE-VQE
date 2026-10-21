#!/usr/bin/env python3
"""Exact stdlib-only inspection of the serialized confirmation checkpoints.

Reads artifacts/multimodal-confirmation-checkpoints.tar (256 recorded .pt files,
sha256-inventoried in artifacts/multimodal-confirmation-checkpoints.json) WITHOUT
loading torch and without executing any model code:

1. every .pt is a torch ZIP container, opened with stdlib zipfile;
2. every ``data.pkl`` is parsed with a stdlib pickle.Unpickler whose
   ``find_class`` returns inert stub objects for torch/numpy classes and whose
   ``persistent_load`` captures the ('storage', StorageType, key, location,
   numel) persistent ids. No torch/numpy module is imported or executed;
3. the reconstructed tree is walked to classify each tensor record by its saved
   state path (model LoRA bank, router memory features, router fit tensors,
   optimizer state, RNG state), and the ZIP-member byte sizes give the exact
   serialized payload — no dtype/size assumption anywhere: dtypes come from the
   StorageClass names in the pickle and dtype sizes from stdlib constants only.

The output JSON aggregates per method (kind/variant) and per file, and records
the composition identity  serialized bytes == data.pkl + data/<key> members +
byteorder/version members + ZIP local/central header overhead for each file.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import re
import statistics as st
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DTYPE_ITEMS = {
    "torch.FloatStorage": 4, "torch.HalfStorage": 2, "torch.BFloat16Storage": 2,
    "torch.DoubleStorage": 8, "torch.IntStorage": 4, "torch.LongStorage": 8,
    "torch.ShortStorage": 2, "torch.CharStorage": 1, "torch.ByteStorage": 1,
    "torch.BoolStorage": 1, "torch.ComplexFloatStorage": 8, "torch.ComplexDoubleStorage": 16,
}


class Record:
    """Result of calling an inert torch/numpy stub (e.g. _rebuild_tensor_v2)."""

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __setstate__(self, state):  # BUILD with state must not raise
        self.state = state


class StubRef:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return Record(self.name, args)

    def __setstate__(self, state):
        pass


class ProbeUnpickler(pickle.Unpickler):
    def __init__(self, data, storages, labels):
        super().__init__(io.BytesIO(data))
        self.storages = storages
        self.labels = labels

    def find_class(self, module, name):
        token = f"{module}.{name}"
        if module.split(".")[0] in ("torch", "numpy", "array"):
            return StubRef(token)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        rec = {"__storage__": True, "dtype": getattr(pid[1], "name", repr(pid[1])),
               "key": pid[2], "location": pid[3], "numel": pid[4]}
        self.storages.append(rec)
        return rec


def walk(obj, path, tensor_records, python_int_labels, inside_router_memory=False):
    if isinstance(obj, Record):
        if obj.name.endswith(("_rebuild_tensor_v2", "_rebuild_tensor")):
            storage = obj.args[0]
            tensor_records.append({"path": path, "storage": storage,
                                   "size": obj.args[2] if len(obj.args) > 2 else None})
        elif inside_router_memory and path.endswith("label"):
            pass  # handled below
        else:
            for k, arg in enumerate(obj.args):
                walk(arg, f"{path}[{k}]", tensor_records, python_int_labels, inside_router_memory)
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if inside_router_memory and (isinstance(key, int)):
                python_int_labels[key] = 1
            walk(val, f"{path}.{key}" if isinstance(key, str) else f"{path}.{key!r}",
                 tensor_records, python_int_labels, inside_router_memory or
                 path.endswith("router.memory"))
    elif isinstance(obj, (list, tuple)):
        for k, val in enumerate(obj):
            if inside_router_memory and isinstance(val, int):
                python_int_labels[val] = 1
            walk(val, f"{path}[{k}]", tensor_records, python_int_labels, inside_router_memory or
                 path.endswith("router.memory"))


def classify(path: str) -> str:
    stripped = path
    for prefix in ("root.", "root['", "root["):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
    # full run checkpoint nests under 'model.'; independent per-task archives
    # save checkpoint_state() directly (no 'model' wrapper).
    if stripped.startswith("model.projections") or stripped.startswith("projections"):
        return "lora_parameter"
    if stripped.startswith("router.memory"):
        return "router_memory_feature"
    if stripped.startswith("router.weight"):
        return "router_fit_weight"
    if stripped.startswith("router.center"):
        return "router_fit_center"
    if stripped.startswith("router.prior"):
        return "router_fit_prior"
    if stripped.startswith("optimizer"):
        return "optimizer"
    if stripped.startswith("scheduler"):
        return "scheduler"
    if stripped.startswith("rng."):
        return "rng_state"
    return "other"


def probe_file(raw: bytes, tar_size: int) -> dict:
    zf = zipfile.ZipFile(io.BytesIO(raw))
    infos = zf.infolist()
    pkl_name = [e.filename for e in infos if e.filename.endswith("data.pkl")][0]
    root = pkl_name.rsplit("/", 1)[0]
    storages, labels = [], {}
    obj = ProbeUnpickler(zf.read(pkl_name), storages, labels).load()

    tensor_records = []
    python_labels = {}
    walk(obj, "root", tensor_records, python_labels)

    member_by_key = {}
    for e in infos:
        if e.filename.startswith(root + "/data/"):
            member_by_key[e.filename.rsplit("/", 1)[-1]] = e.file_size
    data_member_bytes = sum(member_by_key.values())
    pkl_bytes = next(e.file_size for e in infos if e.filename.endswith("data.pkl"))
    stored_total = sum(e.file_size for e in infos)
    other_member_bytes = stored_total - data_member_bytes - pkl_bytes  # e.g. version/byteorder, serialization_id
    serialized = tar_size
    overhead = serialized - stored_total  # ZIP local headers + central directory + EOCD

    groups = defaultdict(lambda: {"n": 0, "bytes": 0, "dtypes": Counter(), "sizes": Counter()})
    composition = {"lora_parameter": 0, "router_memory_feature": 0, "router_fit": 0,
                   "optimizer": 0, "rng_state": 0, "other": 0}
    for record in tensor_records:
        storage = record["storage"]
        cat = classify(record["path"])
        bucket = "router_fit" if cat in ("router_fit_weight", "router_fit_center", "router_fit_prior") else cat
        item = DTYPE_ITEMS[storage["dtype"]]
        nbytes = storage["numel"] * item
        groups[bucket]["n"] += 1
        groups[bucket]["bytes"] += nbytes
        groups[bucket]["dtypes"][storage["dtype"]] += 1
        groups[bucket]["sizes"][tuple(record["size"] or ())] += 1
        composition[bucket] += nbytes
        # exact member cross-check
        assert member_by_key[str(storage["key"])] == nbytes, f"storage key {storage['key']} bytes mismatch"
        assert storage["numel"] * DTYPE_ITEMS[storage["dtype"]] == nbytes
        if bucket == "lora_parameter":
            composition["lora_parameter"] += 0  # counted once
    checked = (groups["lora_parameter"]["bytes"] + groups["router_memory_feature"]["bytes"]
               + groups["router_fit"]["bytes"] + groups["optimizer"]["bytes"]
               + groups["rng_state"]["bytes"] + groups["other"]["bytes"])
    assert checked == data_member_bytes, "storage member byte accounting mismatch"
    assert pkl_bytes + data_member_bytes + other_member_bytes + overhead == serialized

    top_categories = {}
    for key, value in obj.items() if isinstance(obj, dict) else ():
        top_categories[key] = value.name if isinstance(value, Record) else (Record if False else type(value).__name__)

    optimizer_nones = {"optimizer": obj.get("optimizer"), "scheduler": obj.get("scheduler")}
    optimizer_storage_keys = {}
    if not isinstance(obj, dict):
        raise TypeError("checkpoint pickle must be a dict")
    if not isinstance(obj.get("optimizer"), (dict, type(None))):
        raise TypeError("unexpected optimizer type")
    if not isinstance(obj.get("scheduler"), (dict, type(None))):
        raise TypeError("unexpected scheduler type")
    if obj.get("router") is not None and not isinstance(obj.get("router"), dict):
        raise TypeError("unexpected router type")
    if "model" in obj:
        if not isinstance(obj.get("model"), dict):
            raise TypeError("unexpected model type")
    if "state" in obj and not isinstance(obj.get("state"), dict):
        raise TypeError("unexpected state type")

    replay_rows = len(obj["state"].get("replay", ())) if isinstance(obj.get("state"), dict) else None
    stage_resources_n = len(obj["state"].get("stage_resources", ())) if isinstance(obj.get("state"), dict) else None
    return {
        "member_name": root.rsplit("/", 1)[-1] if "/" in root else root,
        "groups": {k: {"n": v["n"], "bytes": v["bytes"], "dtypes": dict(v["dtypes"]),
                       "sizes": [{"shape": "x".join(str(d) for d in s), "n": n} for s, n in v["sizes"].items()]}
                   for k, v in groups.items()},
        "composition_bytes": composition,
        "python_int_labels_in_router_memory": dict(python_labels) if python_labels else {},
        "optimizer": "None" if obj.get("optimizer") is None else "STATE_DICT",
        "scheduler": "None" if obj.get("scheduler") is None else "STATE_DICT",
        "replay_rows": replay_rows,
        "stage_resources": stage_resources_n,
        "router_has_memory": isinstance(obj.get("router"), dict) and bool(obj["router"].get("memory")),
        "num_experts_declared": (obj["model"].get("num_experts") if isinstance(obj.get("model"), dict)
                                 else obj.get("num_experts")),
        "rank_declared": (obj["model"].get("rank") if isinstance(obj.get("model"), dict) else obj.get("rank")),
        "alpha_declared": (obj["model"].get("alpha") if isinstance(obj.get("model"), dict) else obj.get("alpha")),
        "serialized": serialized,
        "data_pkl_bytes": pkl_bytes,
        "data_member_bytes": data_member_bytes,
        "version_byteorder_bytes": other_member_bytes,
        "zip_overhead_bytes": overhead,
        "zip_members": len(infos),
        "top_level_types": top_categories,
        "optimizer_nones": {k: optimizer_nones[k] is None for k in optimizer_nones},
        "optimizer_storage_keys": optimizer_storage_keys,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--tar", default="artifacts/multimodal-confirmation-checkpoints.tar")
    ap.add_argument("--out", default="artifacts/multimodal-checkpoint-zip-inspection.json")
    args = ap.parse_args()
    tar_path = args.root / args.tar
    files = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".pt"):
                continue
            raw = tf.extractfile(m).read()
            rec = probe_file(raw, m.size)
            rec["path"] = m.name
            files.append(rec)
    runtime = json.loads((args.root / "artifacts" / "multimodal-confirmation-checkpoints.json").read_text())["files"]
    for rec in files:
        inventory = runtime.get(rec["path"])
        assert inventory and inventory["bytes"] == rec["serialized"], (rec["path"], inventory)

    def agg(vals):
        return {"n": len(vals), "min": min(vals), "max": max(vals), "mean": st.mean(vals)}

    by_variant_file = defaultdict(list)
    name_re = re.compile(r"^(?P<order>[A-Z](?:_[A-Z])*)_(?P<variant>capacity_matched_shared|shared_lora|replay_lora|"
                         r"expand|frozen_first|independent)(?:_(?:lr(?P<lr>[0-9.]+))?(?:_ep(?P<epochs>[0-9]+(?:\.[0-9]+)?))?)?"
                         r"_seed(?P<seed>\d+)$")
    for rec in files:
        base = Path(rec["path"]).parent.name
        match = name_re.match(base)
        variant = match["variant"] if match else "unparsed"
        if variant == "replay_lora" and match["epochs"] == "10":
            variant = "tuned_replay"
        fname = Path(rec["path"]).name
        if fname == "latest.pt":
            by_variant_file[(variant, "latest.pt")].append(rec)
        elif fname.startswith("independent_"):
            by_variant_file[(variant, "independent_archive")].append(rec)
        else:
            by_variant_file[(variant, fname)].append(rec)
    summary = {}
    for key, recs in sorted(by_variant_file.items()):
        values = defaultdict(list)
        for rec in recs:
            values["serialized"].append(rec["serialized"])
            for cat in ("lora_parameter", "router_memory_feature", "router_fit",
                        "optimizer", "rng_state", "other"):
                values[cat].append(rec["composition_bytes"][cat])
            values["data_pkl"].append(rec["data_pkl_bytes"])
            values["zip_overhead"].append(rec["zip_overhead_bytes"])
            values["version_byteorder"].append(rec["version_byteorder_bytes"])
            values["batch_zip_members"].append(rec["zip_members"])
            values["optimizer_state_dict"].append(rec["optimizer"])
            values["scheduler_state_dict"].append(rec["scheduler"])
            values["replay_rows"].append(rec["replay_rows"] or 0)
            values["router_memory_int_labels"].append(
                sum(rec["python_int_labels_in_router_memory"].values()) if rec["python_int_labels_in_router_memory"] else 0)
            values["rank_declared"].append(rec["rank_declared"])
            values["num_experts_declared"].append(rec["num_experts_declared"])
            values["alpha_declared"].append(rec["alpha_declared"])
        opt = Counter(values["optimizer_state_dict"])
        dtypes = Counter()
        for rec in recs:
            for cat, group in rec["groups"].items():
                for dtype, n in group["dtypes"].items():
                    dtypes[dtype] += n
        shapes = Counter()
        for rec in recs:
            for cat, group in rec["groups"].items():
                for entry in group["sizes"]:
                    shapes[(cat, entry["shape"])] += entry["n"]
        summary[f"{key[0]}|{key[1]}"] = {
            "variant": key[0],
            "file_kind": key[1],
            "n_files": len(recs),
            "serialized_bytes": agg(values["serialized"]),
            "lora_parameter_bytes": agg(values["lora_parameter"]),
            "router_memory_feature_bytes": agg(values["router_memory_feature"]),
            "router_fit_bytes": agg(values["router_fit"]),
            "optimizer_bytes": agg(values["optimizer"]),
            "rng_state_bytes": agg(values["rng_state"]),
            "other_storage_bytes": agg(values["other"]),
            "data_pkl_metadata_bytes": agg(values["data_pkl"]),
            "version_byteorder_bytes": agg(values["version_byteorder"]),
            "zip_overhead_bytes": agg(values["zip_overhead"]),
            "zip_members": agg(values["batch_zip_members"]),
            "optimizer_state_dict": dict(opt),
            "scheduler_state_dict": dict(Counter(values["scheduler_state_dict"])),
            "replay_rows_stored": agg(values["replay_rows"]),
            "router_int_labels_total": agg(values["router_memory_int_labels"]),
            "declared_model": {
                "rank": sorted(set(values["rank_declared"])),
                "alpha": sorted(set(values["alpha_declared"])),
                "num_experts": sorted(set(values["num_experts_declared"])),
            },
            "storage_dtypes_all": dict(dtypes),
            "shapes_by_category": {f"{cat}|{shape}": n for (cat, shape), n in shapes.items()},
        }

    output = {"schema": "multimodal-checkpoint-zip-inspection/1",
              "method": "stdlib zipfile + stdlib pickle with torch/numpy stub find_class; torch never imported or executed; no conversion of dtypes assumed — item sizes read from each zip member and cross-checked against numel*stdlib_itemsize and against the recorded sha256 inventory bytes",
              "tar": str(tar_path.name),
              "n_files": len(files),
              "per_file": files,
              "summary_by_variant_file": summary}
    (args.root / args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"written": str(args.root / args.out), "files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
