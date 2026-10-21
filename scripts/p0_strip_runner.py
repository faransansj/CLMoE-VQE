#!/usr/bin/env python3
"""G4-strip P0 runner: one G4 cell through the UNTOUCHED sema_multimodal.run path.

Only deployment paths may differ from the recorded G4 scientific argv. This runner
changes --eval-split to validation and --output to the selected run namespace.
Cell --strip applies the hash-verified Stage 1 period-strip hook unchanged.
A runtime hook snapshots model+router at every task end (task_end_<index>.pt).
No sema_* source is edited.

By default the run is not lock-gated: it always records the Git HEAD commit (or the
explicit reason it is unavailable), the working-tree preregistration SHA256 and
per-file execution hashes. --require-prereg-lock enforces a separately committed,
complete locked preregistration and a verified clean Git checkout of its execution
code before any model import or output. --validate-only checks the contract without torch.

gate collects an unpatched run for an explicitly invoked external comparator;
it never opens historical checkpoints or declares cross-GPU bitwise fidelity.
cell collects one P0 cell (strip ON with --strip) on RTX 4090 or L40S, never L40.

Note: the untouched run.main validates the whole manifest, which parses eval rows and
hashes eval image bytes for leakage/integrity checks (as in G4 and Stage 1); the eval
split is never generated or scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = "scripts/p0_strip_runner.py"
JOBS = ROOT / "configs/multimodal_g4_jobs.json"
OUT_ROOT = ROOT / "artifacts/multimodal_runs/p0_strip"
REPORT_ROOT = ROOT / "artifacts/p0_strip"
STAGE1 = ROOT / "scripts/stage1_period_strip_train.py"
STAGE1_SHA256 = "3cc2bbbc4b91045d2d9e695cb777145dbe2cce12924b111f9e36b1b3445c66db"
STRIP_FUNCTION_SHA256 = "073229d3f22e2b2688943ef06a43609d6dc42991e51743aef76cfaf75c2daac7"
PREREG = "preregistration-p0-strip.md"
EXECUTION_PATHS = (
    TOOL, "scripts/stage1_period_strip_train.py", "scripts/p0_strip.sbatch",
    "sema_multimodal", "sema_vqa", "third_party/BLIP/models",
    "third_party/BLIP/configs/med_config.json",
    "third_party/SMoLoRA/llava/eval/m4c_evaluator.py",
    "third_party/SMoLoRA/scripts/convert_gqa_for_eval.py",
    "third_party/SMoLoRA/llava/eval/SMoLoRA/Eval/eval_gqa.py",
    "requirements.txt", "requirements-vision.txt", "requirements-multimodal.txt",
    "sitecustomize.py", "usercustomize.py",
)
SCHEMA = 2


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage1():
    raw = STAGE1.read_bytes()
    if sha256_bytes(raw) != STAGE1_SHA256:
        raise ValueError("stage1 script differs from the recorded hash")
    spec = importlib.util.spec_from_file_location("stage1_strip", STAGE1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if sha256_bytes(inspect.getsource(module.install_strip_patch).encode()) != STRIP_FUNCTION_SHA256:
        raise ValueError("install_strip_patch source differs from the recorded hash")
    return module


def g4_argv(name: str, jobs: Path = JOBS) -> list[str]:
    payload = json.loads(jobs.read_bytes())
    matches = [job["argv"] for job in payload["jobs"] if job["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"G4 job not found exactly once: {name}")
    argv = matches[0]
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise ValueError(f"Invalid argv for G4 job: {name}")
    return list(argv)


def replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    index = argv.index(flag)
    return argv[:index + 1] + [value] + argv[index + 2:]


def environment() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("A CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = getattr(properties, "uuid", None)
    if not gpu_uuid:
        raise ValueError("PyTorch must expose the active GPU UUID for card provenance")
    gpu_uuid = str(gpu_uuid)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "-i", gpu_uuid, "--query-gpu=name,uuid,pci.bus_id,driver_version",
             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot verify active GPU identity with nvidia-smi: {exc}") from exc
    packages = sorted(
        (re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(), distribution.version)
        for distribution in importlib.metadata.distributions() if distribution.metadata["Name"])
    env = {
        "python": platform.python_version(), "python_implementation": platform.python_implementation(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "packages": packages,
        "platform": platform.platform(), "gpu_name": properties.name, "gpu_uuid": gpu_uuid,
        "gpu_capability": f"{properties.major}.{properties.minor}",
        "nvidia_smi": smi.stdout.strip().splitlines(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"), "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    }
    # No clock, hostname, device UUID, bus address, or output path enters this hash.
    hashed = {key: env[key] for key in (
        "python", "python_implementation", "torch", "cuda", "cudnn", "packages",
        "gpu_name", "gpu_capability", "cublas_workspace_config", "omp_threads")}
    hashed["driver"] = sorted({line.rsplit(",", 1)[-1].strip() for line in env["nvidia_smi"]})
    env["environment_sha256"] = sha256_bytes(json.dumps(hashed, sort_keys=True).encode())
    return env


def scheduler() -> dict:
    job_id = os.environ.get("SLURM_JOB_ID")
    info = {"slurm_job_id": job_id, "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
            "partition": os.environ.get("SLURM_JOB_PARTITION"), "scontrol": None}
    if job_id:
        try:
            result = subprocess.run(["scontrol", "show", "job", job_id], capture_output=True, text=True, check=False)
        except OSError:
            return info
        fields = {}
        for token in result.stdout.split():
            key, _, value = token.partition("=")
            if value:
                fields.setdefault(key, value)
        info["scontrol"] = {key: fields[key] for key in ("JobId", "Partition", "NodeList", "TresPerNode", "QOS",
                                                          "SubmitTime", "StartTime", "Timelimit") if key in fields}
    return info


def install_snapshot_hook(runner) -> None:
    original = runner.evaluate
    pattern = re.compile(r"predictions_after_(\d{2})_")

    def snapshot_evaluate(model, router, task, scorer, cached, args, path, *rest, **kw):
        match = pattern.match(path.name)
        if match and args.method != "independent":
            import torch
            snap = path.parent / f"task_end_{match[1]}.pt"
            if not snap.exists():
                temporary = snap.with_suffix(".pt.tmp")
                torch.save({"schema_version": 1, "task_index": int(match[1]),
                            "model": model.checkpoint_state(), "router": runner.router_state(router)}, temporary)
                temporary.replace(snap)
        return original(model, router, task, scorer, cached, args, path, *rest, **kw)

    runner.evaluate = snapshot_evaluate


# ---------------------------------------------------------------- provenance
def git(*args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(ROOT), *args],
            capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        raise ValueError(
            "Verified local Git history is required (restore an offline Git bundle); "
            f"git {' '.join(args)} failed: {detail.decode(errors='replace').strip() or exc}"
        ) from exc
    return result.stdout


def full_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError("Commit IDs must be full lowercase Git object IDs, not refs or abbreviations")
    resolved = git("rev-parse", "--verify", f"{value}^{{commit}}").decode().strip()
    if resolved != value:
        raise ValueError(f"Not a commit object: {value}")
    return resolved


def verify_preregistration(commit: str, jobs: Path) -> dict:
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
        raise ValueError("Runner must execute inside its own verified Git checkout")
    commit = full_commit(commit)
    prereg = git("show", f"{commit}:{PREREG}")
    text = prereg.decode("utf-8")
    if re.search(r"TBD", text, re.IGNORECASE):
        raise ValueError("Committed preregistration still contains TBD; training is blocked")
    frontmatter = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.DOTALL)
    if frontmatter is None:
        raise ValueError("Committed preregistration requires explicit YAML front matter")
    fields = frontmatter.group(1)
    status = re.findall(r"^status:\s*([^\r\n]+)", fields, re.MULTILINE)
    locked = re.findall(r"^locked_code_commit:\s*([^\r\n]+)", fields, re.MULTILINE)
    if status != ["locked"] or len(locked) != 1:
        raise ValueError("Committed preregistration must have status: locked and one locked_code_commit")
    source = full_commit(locked[0].strip())
    if source == commit:
        raise ValueError("Preregistration lock must be a separate commit after the code commit")
    git("merge-base", "--is-ancestor", source, commit)
    head = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    git("merge-base", "--is-ancestor", commit, head)
    try:
        selected_jobs = jobs.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("--jobs must be a tracked config inside the verified checkout") from exc
    scopes = [*EXECUTION_PATHS, JOBS.relative_to(ROOT).as_posix(), selected_jobs]
    # Include ignored files: an ignored dependency is not verified Git provenance.
    untracked = set()
    for flags in ((), ("--ignored",)):
        names = git("ls-files", "-z", "--others", "--exclude-standard", *flags, "--", *scopes)
        for name in names.decode().split("\0"):
            if name and "__pycache__" not in Path(name).parts and not name.endswith((".pyc", ".pyo")):
                untracked.add(name)
    if untracked:
        raise ValueError("Untracked execution/config files must be reviewed and committed: "
                         + ", ".join(sorted(untracked)))
    tree = git("ls-tree", "-rz", source, "--", *scopes)
    if tree != git("ls-tree", "-rz", head, "--", *scopes):
        raise ValueError("HEAD execution/config tree differs from locked_code_commit")
    if git("diff", "--cached", "--name-only", source, "--", *scopes).strip():
        raise ValueError("Index execution/config tree differs from locked_code_commit")
    tracked = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.decode().split()
        name = raw_name.decode()
        path = ROOT / name
        if kind != "blob" or mode not in ("100644", "100755"):
            raise ValueError(f"Execution provenance requires regular tracked files, not links/submodules: {name}")
        if not path.is_file() or any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError(f"Missing or symlinked execution file: {name}")
        raw = path.read_bytes()
        algorithm = "sha1" if len(object_id) == 40 else "sha256"
        actual = hashlib.new(algorithm, f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != object_id:
            raise ValueError(f"Dirty execution/config file differs from locked_code_commit: {name}")
        if bool(path.stat().st_mode & 0o111) != (mode == "100755"):
            raise ValueError(f"Dirty executable mode on execution/config file: {name}")
        tracked[name] = sha256_bytes(raw)
    required = required_execution_files(selected_jobs)
    missing = sorted(set(required) - tracked.keys())
    if missing:
        raise ValueError("Required execution/config files absent from locked source tree: " + ", ".join(missing))
    return {"lock_verified": True, "prereg_commit": commit, "prereg_sha256": sha256_bytes(prereg),
            "prereg_sha256_source": f"git:{commit}:{PREREG}",
            "locked_code_commit": source, "source_commit": head, "source_commit_status": "verified_clean",
            "execution_sha256": tracked}


def required_execution_files(selected_jobs: str) -> list[str]:
    return [TOOL, STAGE1.relative_to(ROOT).as_posix(), JOBS.relative_to(ROOT).as_posix(), selected_jobs,
            *[f"sema_multimodal/{name}.py" for name in ("__init__", "run", "model", "data", "metrics")],
            *[f"sema_vqa/{name}.py" for name in (
                "__init__", "readout_model", "readout_router", "vision_model", "expansion", "metrics")],
            "third_party/BLIP/models/med.py", "third_party/BLIP/models/vit.py",
            "third_party/BLIP/configs/med_config.json",
            *[name for name in EXECUTION_PATHS if name.startswith("third_party/SMoLoRA/")]]


def record_provenance(claimed_prereg: str | None, jobs: Path) -> dict:
    """Unlocked default: record, never fabricate. No Git => commit fields are null with a reason."""
    try:
        selected_jobs = jobs.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("--jobs must be a config inside this checkout") from exc
    hashes = {}
    for scope in (*EXECUTION_PATHS, JOBS.relative_to(ROOT).as_posix(), selected_jobs):
        path = ROOT / scope
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
        for file in files:
            if "__pycache__" in file.parts or file.suffix in (".pyc", ".pyo"):
                continue
            hashes[file.relative_to(ROOT).as_posix()] = sha256_file(file)
    missing = sorted(set(required_execution_files(selected_jobs)) - hashes.keys())
    if missing:
        raise ValueError("Required execution/config files are missing: " + ", ".join(missing))
    prereg = ROOT / PREREG
    provenance = {"lock_verified": False, "prereg_commit": claimed_prereg, "prereg_commit_verified": False,
                  "prereg_sha256": sha256_file(prereg) if prereg.is_file() else None,
                  "prereg_sha256_source": f"working_tree:{PREREG}" if prereg.is_file() else "file_absent",
                  "locked_code_commit": None, "source_commit": None, "source_commit_status": None,
                  "dirty_execution_paths": None, "execution_sha256": hashes}
    try:
        if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != ROOT:
            raise ValueError("runner is not at the root of its Git checkout")
        provenance["source_commit"] = git("rev-parse", "--verify", "HEAD^{commit}").decode().strip()
        status = git("status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching", "--",
                     *EXECUTION_PATHS, JOBS.relative_to(ROOT).as_posix(), selected_jobs, PREREG)
        dirty = sorted({entry[3:] for entry in status.decode().split("\0") if len(entry) > 3
                        and "__pycache__" not in entry and not entry.endswith((".pyc", ".pyo"))})
        provenance["dirty_execution_paths"] = dirty
        provenance["source_commit_status"] = "dirty_or_untracked" if dirty else "clean"
    except ValueError as exc:
        provenance["source_commit_status"] = f"unavailable: {str(exc).splitlines()[0][:300]}"
    return provenance


def safe_identity(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value) or ".." in value:
        raise argparse.ArgumentTypeError("Use a safe single path component (letters/digits/_.-, no '..')")
    return value


def selected_argv(name: str, jobs: Path) -> tuple[list[str], list[str]]:
    original, selected = g4_argv(name), g4_argv(name, jobs)
    # Deployment configs relocate inputs, never retune G4 scientific arguments.
    deployment_flags = ("--manifest", "--assets", "--image-root", "--output")
    normalized = []
    for argv in (original, selected):
        copy = list(argv)
        for flag in deployment_flags:
            if copy.count(flag) != 1 or copy.index(flag) + 1 == len(copy):
                raise ValueError(f"Expected exactly one value for {flag}")
            copy = replace_flag(copy, flag, "<deployment-path>")
        normalized.append(copy)
    if normalized[0] != normalized[1]:
        raise ValueError("Selected jobs change G4 scientific argv; only deployment paths may differ")
    return original, selected


# ---------------------------------------------------------------- main
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("gate", "cell"))
    parser.add_argument("--g4-job", required=True, type=safe_identity, help="Exact recorded G4 job name")
    parser.add_argument("--strip", action="store_true", help="cell mode: apply the Stage 1 period strip")
    parser.add_argument("--prereg-commit", default=None,
                        help="Preregistration commit; recorded as an unverified claim unless --require-prereg-lock")
    parser.add_argument("--require-prereg-lock", action="store_true",
                        help="Enforce the separately committed locked preregistration and clean verified source")
    parser.add_argument("--jobs", type=Path, default=JOBS, help="Tracked deployment jobs config (scientific argv unchanged)")
    parser.add_argument("--run-id", required=True, type=safe_identity, help="Unique card/run namespace; never reused")
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT,
                        help="Output root under this checkout's artifacts/multimodal_runs")
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate config/paths (and the lock if required) only; no torch import, outputs, or training")
    args = parser.parse_args(argv)
    if args.mode == "gate" and args.strip:
        parser.error("Gate collection runs with strip OFF")
    try:
        jobs = args.jobs.resolve()
        if args.require_prereg_lock:
            if not args.prereg_commit:
                raise ValueError("--require-prereg-lock needs --prereg-commit")
            provenance = verify_preregistration(args.prereg_commit, jobs)
        else:
            provenance = record_provenance(args.prereg_commit, jobs)
        original_argv, deployment_argv = selected_argv(args.g4_job, jobs)
        variant = "gate" if args.mode == "gate" else ("strip" if args.strip else "nostrip")
        output = (args.output_root / args.run_id / variant / args.g4_job).resolve()
        report_path = (args.report_root / args.run_id / variant / f"{args.g4_job}.json").resolve()
        if (ROOT / "artifacts/multimodal_runs").resolve() not in output.parents:
            raise ValueError("--output-root must remain under this checkout's artifacts/multimodal_runs")
        if output.exists() or report_path.exists():
            raise ValueError(f"Refusing existing output/report: {output} / {report_path}")
        if output in report_path.parents:
            raise ValueError("Report root must not be inside the training output directory")
        run_argv = replace_flag(replace_flag(deployment_argv, "--eval-split", "validation"), "--output", str(output))
        if sha256_file(STAGE1) != STAGE1_SHA256:
            raise ValueError("stage1 script differs from the recorded hash")
    except (ValueError, OSError, KeyError, UnicodeError) as exc:
        parser.error(str(exc))
    if args.validate_only:
        print(json.dumps({"validated": True, "training_started": False, "gpu_checked": False,
                          "run_id": args.run_id, "provenance": provenance,
                          "output": str(output), "report": str(report_path),
                          "g4_jobs_sha256": sha256_file(JOBS), "selected_jobs_sha256": sha256_file(jobs)}))
        return

    started, started_utc = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    env_before = environment()
    allowed = {"NVIDIA L40S", "NVIDIA GeForce RTX 4090"}
    if args.mode == "gate":
        allowed.add("NVIDIA L40")
    if env_before["gpu_name"] not in allowed:
        parser.error(f"{args.mode} GPU must be one of {sorted(allowed)}; got {env_before['gpu_name']}")
    # Force imports to resolve to the verified checkout, never a different installed sema package.
    sys.path.insert(0, str(ROOT))
    stage1 = load_stage1()
    import sema_multimodal.run as runner

    runner.safe_output(output)
    # Exclusive reservation prevents a second card from silently resuming the same cell.
    output.mkdir(parents=True, exist_ok=False)
    install_snapshot_hook(runner)
    restore = stage1.install_strip_patch() if args.strip else None
    wall = time.perf_counter()
    try:
        result = runner.main(run_argv)
    finally:
        if restore is not None:
            import sema_multimodal.data as mmdata
            mmdata.training_rows = restore
    wall = round(time.perf_counter() - wall, 3)

    files = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    config = json.loads((output / "config.json").read_text())
    report = {"schema_version": SCHEMA, "stage": f"p0-strip-{args.mode}", "g4_job": args.g4_job,
              "started_utc": started_utc, "run_id": args.run_id, "provenance": provenance,
              "labels": {"strip": bool(args.strip), "eval_split_generated_or_scored": False,
                         "internal_evaluation_split": "validation",
                         "prereg_commit": args.prereg_commit,
                         "prereg_lock_required": bool(args.require_prereg_lock),
                         "argv_changes_vs_g4": ["--eval-split eval->validation", "--output",
                                                 "deployment input paths from selected jobs"]},
              "argv": run_argv, "g4_argv": original_argv, "selected_argv": deployment_argv,
              "tool": {"path": TOOL, "sha256": sha256_file(Path(__file__).resolve()),
                       "stage1_sha256": STAGE1_SHA256, "strip_function_sha256": STRIP_FUNCTION_SHA256,
                       "g4_jobs_sha256": sha256_file(JOBS), "selected_jobs_sha256": sha256_file(jobs),
                       "selected_jobs": str(jobs)},
              "environment": env_before, "scheduler": scheduler(), "scientific_config": config,
              "input_hashes": {key: config[key] for key in (
                  "manifest_sha256", "assets_sha256", "tokenizer_sha256")},
              "run": {"output": str(output), "wall_seconds": wall,
                      "status": result.get("status"), "completed_tasks": result.get("completed"),
                      "resources": result.get("resources"), "files_sha256": files}}

    if args.mode == "gate":
        report["gate"] = {"collection_only": True, "comparison_performed": False}

    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["runtime_seconds_total"] = round(time.perf_counter() - started, 3)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stage1.exclusive_create_text(report_path, json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(report_path), "mode": args.mode, "run_id": args.run_id,
                      "comparison_performed": False}), flush=True)


if __name__ == "__main__":
    main()
