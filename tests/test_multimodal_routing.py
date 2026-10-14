"""Consumer-visible boundary tests for the G5 routing evaluator and rescorer.

No BLIP weights/GPU are needed: these cover the pure routing contracts (public
request family derivation, masked native-coefficient routing, no-future-expert
and base-only pre-task semantics), the output-safety refusal, and the locked
normalizer binding required by the independent rescorer.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH,
                                reason="torch is required to import the multimodal routing modules")


def test_public_request_family_is_derived_from_visible_requests():
    from sema_multimodal.data import CAPTION_INSTRUCTION
    from scripts.ablate_multimodal_routing import request_family
    caption_task = {"task_id": "C", "family": "caption"}
    caption_rows = [{"input_text": CAPTION_INSTRUCTION}, {"input_text": CAPTION_INSTRUCTION}]
    assert request_family(caption_task, caption_rows) == "caption"
    vqa_task = {"task_id": "V", "family": "vqa"}
    vqa_rows = [{"input_text": "What color is the dog?"}, {"input_text": "How many people?"}]
    assert request_family(vqa_task, vqa_rows) == "vqa"
    with pytest.raises(ValueError):
        request_family(vqa_task, [{"input_text": "What?"}, {"input_text": CAPTION_INSTRUCTION}])
    with pytest.raises(ValueError):
        request_family({"task_id": "V", "family": "caption"}, vqa_rows)


def test_masked_router_uses_native_joint_coefficients_with_candidate_mask():
    torch = pytest.importorskip("torch")
    from sema_vqa.readout_router import FeatureRouter
    from scripts.ablate_multimodal_routing import MaskedRouter, _snapshot_router
    generator = torch.Generator().manual_seed(5)
    memory_a = torch.randn(24, 1536, generator=generator)
    memory_b = torch.randn(24, 1536, generator=generator)
    router = FeatureRouter(8, 1.0, "joint", 17)
    router.observe(memory_a, 0, 1)
    router.observe(memory_b, 1, 2)
    probe = torch.randn(16, 1536, generator=generator)
    snapshot = _snapshot_router(router)
    native = router.predict(probe)
    assert torch.equal(native, MaskedRouter(snapshot, [0, 1]).predict(probe))
    assert {int(value) for value in MaskedRouter(snapshot, [1]).predict(probe).tolist()} == {1}
    assert {int(value) for value in MaskedRouter(snapshot, [0]).predict(probe).tolist()} == {0}
    with pytest.raises(ValueError):
        MaskedRouter(_snapshot_router(FeatureRouter(8, 1.0, "joint", 17)), [0])
    with pytest.raises(ValueError):
        MaskedRouter(snapshot, [])


def test_family_masked_router_uses_frozen_native_stage_snapshot():
    """The snapshot must stay frozen while the native router advances to later stages."""
    torch = pytest.importorskip("torch")
    from sema_vqa.readout_router import FeatureRouter
    from scripts.ablate_multimodal_routing import MaskedRouter, _snapshot_router
    generator = torch.Generator().manual_seed(11)
    router = FeatureRouter(8, 1.0, "joint", 17)
    router.observe(torch.randn(24, 1536, generator=generator), 0, 1)
    router.observe(torch.randn(24, 1536, generator=generator), 1, 2)
    snapshot = _snapshot_router(router)
    reference = {key: snapshot[key].clone() for key in ("weight", "center", "prior")}
    memory_reference = [(tensor.clone(), expert) for tensor, expert in snapshot["memory"]]
    # Advancing the native router must not leak into the frozen snapshot.
    router.observe(torch.randn(24, 1536, generator=generator), 2, 3)
    assert len(router.memory) == 3
    assert len(snapshot["memory"]) == 2
    for key, expected in reference.items():
        assert torch.equal(snapshot[key], expected)
    for (snapshot_tensor, snapshot_expert), (reference_tensor, reference_expert) in zip(
            [(tensor, expert) for tensor, expert in snapshot["memory"]], memory_reference):
        assert snapshot_expert == reference_expert
        assert torch.equal(snapshot_tensor, reference_tensor)
    probe = torch.randn(8, 1536, generator=generator)
    scores = (probe.float() - reference["center"]) @ reference["weight"] + reference["prior"]
    restricted = torch.full(scores.shape, -float("inf"), dtype=scores.dtype)
    restricted[:, [0, 1]] = 0.0
    assert torch.equal(MaskedRouter(snapshot, [0, 1]).predict(probe), (scores + restricted).argmax(-1))
    only_caption = torch.full(scores.shape, -float("inf"), dtype=scores.dtype)
    only_caption[:, [1]] = 0.0
    assert {int(value) for value in MaskedRouter(snapshot, [1]).predict(probe).tolist()} == {1}


def test_routing_contract_rejects_future_experts_and_oracle_pre_task():
    from scripts.ablate_multimodal_routing import _verify_routes as verify
    # Seen tasks may route anywhere below the stage boundary; never at or past it.
    verify("after", 1, "native_joint", [0, 1], None, None, "router_memory")
    with pytest.raises(ValueError):
        verify("after", 1, "native_joint", [0, 2], None, None, "router_memory")
    with pytest.raises(ValueError):
        verify("after", 1, "family_restricted_joint", [0, 1], [1], None, "family_masked")
    # The oracle is the gold task's own expert only, and never a future expert.
    verify("after", 2, "task_id_oracle", [1], None, 1, "gold_task_expert")
    with pytest.raises(ValueError):
        verify("after", 2, "task_id_oracle", [2], None, 1, "gold_task_expert")
    # Stage-0 pre-task reconstruction is exactly the parent base-only behavior.
    verify("before", 0, "family_restricted_joint", [-1], [], None, "base_only_no_expert")
    with pytest.raises(ValueError):
        verify("before", 0, "family_restricted_joint", [0], [], None, "family_masked")
    with pytest.raises(ValueError):
        verify("before", 1, "task_id_oracle", [1], [], None, "gold_task_expert")
    # An unroutable pre-task request must be declared as such.
    verify("before", 1, "family_restricted_joint", [-1], [], None, "base_only_no_eligible_family")
    with pytest.raises(ValueError):
        verify("before", 1, "native_joint", [-1], None, None, "router_memory")


def test_rescorer_refuses_conflicting_and_existing_report_outputs(tmp_path):
    from scripts.summarize_multimodal_routing import _check_output, _positive_constants, _routing_prepare
    report_root = ROOT / "artifacts" / "multimodal_report"
    export = report_root / f"routing_report_test_{hashlib.sha256(str(tmp_path).encode()).hexdigest()[:12]}"
    export.mkdir(parents=True, exist_ok=True)
    try:
        with pytest.raises(ValueError):
            _check_output(export)
    finally:
        export.rmdir()
    forbidden = ROOT / "legacy_routing_report"
    with pytest.raises(ValueError):
        _check_output(forbidden)
    with pytest.raises(ValueError):
        _positive_constants({"V": 10.0}, {"V": 10.0, "C": 1.0})
    with pytest.raises(ValueError):
        _positive_constants({"V": -1.0}, {"V": -1.0})
    report = {"schema_version": 1, "stage": "G5_frozen_bank_routing", "native_variant": "native_joint",
              "variants": {name: {} for name in ("native_joint", "image_only", "question_only",
                                                 "family_restricted_joint", "task_id_oracle")},
              "expert_retraining_steps": 0,
              "model_projection_tensors_byte_identical_to_checkpoint": True,
              "router_checkpoint_tensors_identical": True,
              "parent_config": {"schema_version": 1, "method": "expand", "router_modality": "joint",
                                "eval_split": "validation"},
              "eval_split": "validation", "allow_eval": False,
              "script_sha256": hashlib.sha256(
                  (ROOT / "scripts/ablate_multimodal_routing.py").read_bytes()).hexdigest()}
    config = _routing_prepare(report)
    assert config["method"] == "expand"
    for mutation in ("method",):
        broken = json.loads(json.dumps(report))
        broken["parent_config"][mutation] = "shared_lora"
        with pytest.raises(ValueError):
            _routing_prepare(broken)
    broken = json.loads(json.dumps(report))
    broken["variants"].pop("task_id_oracle")
    with pytest.raises(ValueError):
        _routing_prepare(broken)


def test_rescorer_rejects_unroutable_and_future_expert_records():
    pytest.importorskip("torch")
    from scripts.summarize_multimodal_routing import _check_routing_records
    report = {"variants": {"family_restricted_joint": {"stage_evaluations": [
        {"kind": "before", "stage": 1, "task_id": "V", "routing": "base_only_no_eligible_family",
         "family_available": False}]}, "native_joint": {"stage_evaluations": []}}}
    expert_family = {0: "vqa", 1: "caption", 2: "vqa", 3: "caption"}
    task = {"task_id": "V", "own_family": "vqa"}
    records = [{"expert": -1}] * 3
    _check_routing_records(report, "family_restricted_joint", "before", 1, None, task, records,
                           expert_family)
    with pytest.raises(ValueError):
        _check_routing_records(report, "native_joint", "before", 1, None, task, records, expert_family)
    with pytest.raises(ValueError):
        _check_routing_records(report, "family_restricted_joint", "after", 2, 0, task,
                               [{"expert": 1}], expert_family, [0, 1, 2, 3])
    with pytest.raises(ValueError):
        _check_routing_records(report, "family_restricted_joint", "after", 2, 0, task,
                               [{"expert": 4}], expert_family, [0, 1, 2, 3])


def test_prediction_reader_preserves_boolean_generation_diagnostics(tmp_path):
    from scripts.ablate_multimodal_routing import read_records

    record = {"id": "image-1", "prediction": "a person", "expert": -1,
              "generation": {"generated_tokens": 3, "max_new_tokens": 32,
                             "eos": True, "empty": False, "repeated_trigram": False,
                             "limit_reached": False}}
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(record) + "\n")
    records, experts = read_records(path)
    assert experts == [-1]
    assert records[0]["generation"]["eos"] is True
    assert records[0]["generation"]["empty"] is False

    record["generation"]["generated_tokens"] = True
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        read_records(path)
    record["generation"]["generated_tokens"] = 3
    record["generation"]["eos"] = 1
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError):
        read_records(path)


def test_final_routing_error_rates_use_all_final_images():
    from scripts.summarize_multimodal_routing import final_route_fractions

    vqa = {"0": {"0": 6, "2": 3, "1": 1}, "2": {"2": 8, "3": 2}}
    caption = {"1": {"1": 7, "3": 2, "0": 1}, "3": {"3": 10}}
    observed = final_route_fractions(
        {"all": {**vqa, **caption}, "vqa": vqa, "caption": caption},
        {0: "vqa", 1: "caption", 2: "vqa", 3: "caption"})
    assert observed["other_expert_route_fraction_final"] == pytest.approx(
        {"all": 9 / 40, "vqa": 6 / 20, "caption": 3 / 20})
    assert observed["other_family_route_fraction_final"] == pytest.approx(
        {"all": 4 / 40, "vqa": 3 / 20, "caption": 1 / 20})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
