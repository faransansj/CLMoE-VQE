import copy
import hashlib
import json
import zipfile

from PIL import Image
import pytest

from scripts import prepare_multimodal_data as prep
from sema_multimodal.data import (CAPTION_INSTRUCTION, ImageVerifier, SPLITS,
                                  image_identity, public_rows, training_rows,
                                  validate_manifest)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def save_image(root, name, identifier):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), (identifier % 256, identifier // 256, 31)).save(path, format="PNG")
    return path


def official_sources(tmp_path, size=160, include_all=False):
    root = tmp_path / "images"
    sources = [{"type": "vqav2", "task_id": "V", "revision": "official-v2", "license": "research permission"},
               {"type": "coco_captions", "task_id": "C", "revision": "official-2014", "license": "research permission"}]
    for role, split, start in (("train", "train2014", 1), ("eval", "val2014", 1001)):
        questions, annotations, images, captions = [], [], [], []
        for image_id in range(start, start + size):
            filename = f"COCO_{split}_{image_id:012d}.jpg"
            save_image(root, f"coco/{split}/{filename}", image_id)
            images.append({"id": image_id, "file_name": filename})
            for offset in range(2):
                qid = image_id * 10 + offset
                questions.append({"question_id": qid, "image_id": image_id, "question": f"Question {offset}?"})
                annotations.append({"question_id": qid, "image_id": image_id, "multiple_choice_answer": "yes",
                                    "answers": [{"answer_id": index, "answer": "yes" if index <= 7 else "no"} for index in range(1, 11)]})
            for index in range(5):
                captions.append({"id": image_id * 10 + index, "image_id": image_id, "caption": f"Image {image_id}, caption {index}."})
        qpath = dump(tmp_path / f"{role}_questions.json", {"data_subtype": split, "questions": questions})
        apath = dump(tmp_path / f"{role}_answers.json", {"data_subtype": split, "annotations": annotations})
        cpath = dump(tmp_path / f"{role}_captions.json", {"images": images, "annotations": captions})
        sources[0][role] = {"questions": qpath.name, "annotations": apath.name, "images": f"coco/{split}"}
        sources[1][role] = {"annotations": cpath.name, "images": f"coco/{split}"}
    if include_all:
        gqa = {"type": "gqa", "task_id": "G", "revision": "balanced-v1", "license": "research permission"}
        flickr = {"type": "flickr30k", "task_id": "F", "revision": "karpathy-flickr30k", "license": "research permission",
                  "dataset": "dataset_flickr30k.json", "images": "flickr", "split_recipe": {"train": ["train"], "eval": ["test"]}}
        records = []
        for role, split, start in (("train", "train", 2001), ("eval", "val", 3001)):
            questions = {}
            for image_id in range(start, start + size):
                save_image(root, f"gqa/{image_id}.jpg", image_id)
                questions[str(image_id * 10)] = {"imageId": str(image_id), "question": "What color?", "answer": "blue", "isBalanced": True}
                flickr_id = image_id + 2000
                save_image(root, f"flickr/{flickr_id}.jpg", flickr_id)
                records.append({"imgid": image_id, "filename": f"{flickr_id}.jpg", "split": "train" if role == "train" else "test",
                                "sentences": [{"raw": f"Flickr image {flickr_id}, reference {i}."} for i in range(5)]})
            gqa[role] = {"questions": dump(tmp_path / f"{split}_balanced_questions.json", questions).name, "images": "gqa"}
        dump(tmp_path / flickr["dataset"], {"dataset": "flickr30k", "images": records})
        sources.extend([gqa, flickr])
    config = dump(tmp_path / "sources.json", {"schema_version": 1, "sources": sources})
    return config, root


def prepare_development(config, root, **kwargs):
    return prep.prepare(config, root, [], development_subset=True, allow_empty_history=True,
                        train_target=8, validation_target=3, eval_target=4, **kwargs)


def all_rows(manifest):
    return [row for task in manifest["tasks"] for split in SPLITS for row in task[split]]


def test_real_official_normalization_preserves_references_and_one_question_per_image(tmp_path):
    config, root = official_sources(tmp_path, include_all=True)
    document = json.loads(config.read_text())
    provenance = {}
    for spec in document["sources"]:
        pools = prep.normalize_source(spec, tmp_path, provenance, 20261013)
        assert len(pools["train"]) == 160
        assert len({image_identity(row) for row in pools["train"]}) == 160
        for rows in pools.values():
            for row in rows:
                assert row["target"] in row["references"]
                if row["source"] == "vqav2":
                    assert row["references"] == ["yes"] * 7 + ["no"] * 3
                elif row["source"] == "gqa":
                    assert row["references"] == ["blue"]
                else:
                    assert len(row["references"]) == 5
                    assert row["input_text"] == CAPTION_INSTRUCTION
    assert all("sha256" in record and "revision" in record for key, record in provenance.items() if key != "flickr30k/recipe")
    assert len(provenance["flickr30k/recipe"]["image_ids_by_role"]["eval"]) == 160


def test_deterministic_byte_and_identity_disjoint_selection_and_actual_decoding(tmp_path):
    config, root = official_sources(tmp_path)
    manifest = prepare_development(config, root)
    assert manifest == prepare_development(config, root)
    assert validate_manifest(manifest, root) is manifest
    rows = all_rows(manifest)
    assert len({image_identity(row) for row in rows}) == len(rows)
    assert len({row["image_sha256"] for row in rows}) == len(rows)
    assert manifest["confirmation_eligible"] is False
    for task in manifest["tasks"]:
        assert [len(task[split]) for split in SPLITS] == [8, 3, 4]
        for row in all_rows({"tasks": [task]}):
            assert row["image_sha256"] == hashlib.sha256((root / row["image"]).read_bytes()).hexdigest()


def test_source_order_and_annotation_order_do_not_change_assignment(tmp_path):
    config, root = official_sources(tmp_path)
    first = prepare_development(config, root)
    document = json.loads(config.read_text())
    document["sources"].reverse()
    dump(config, document)
    for path in tmp_path.glob("*_questions.json"):
        value = json.loads(path.read_text())
        value["questions"].reverse()
        dump(path, value)
    second = prepare_development(config, root)
    selected = lambda manifest: {task["task_id"]: {split: [row["id"] for row in task[split]] for split in SPLITS} for task in manifest["tasks"]}
    assert selected(first) == selected(second)


def test_public_views_remove_every_label_and_metadata_and_caption_cycle_covers_refs(tmp_path):
    config, root = official_sources(tmp_path)
    manifest = prepare_development(config, root)
    rows = all_rows(manifest)
    snapshot = copy.deepcopy(rows)
    assert all(set(row) == {"image", "question"} for row in public_rows(rows))
    assert all(set(row) == {"image", "question", "answer"} for row in training_rows(rows, 17, 0))
    caption = next(row for row in rows if row["family"] == "caption")
    cycle = [training_rows([caption], 17, epoch)[0]["answer"] for epoch in range(len(caption["references"]))]
    assert sorted(cycle) == sorted(caption["references"])
    assert training_rows([caption], 17, 5) == training_rows([caption], 17, 0)
    qa = next(row for row in rows if row["family"] == "vqa")
    assert training_rows([qa], 17, 0) == training_rows([qa], 17, 11)
    assert rows == snapshot


def test_history_verifies_bytes_and_excludes_old_eval_from_all_and_old_train_from_eval(tmp_path):
    config, root = official_sources(tmp_path)
    original = prepare_development(config, root)
    old_eval = original["tasks"][0]["train"][0]
    old_train = original["tasks"][0]["eval"][0]
    history = dump(tmp_path / "old.json", {"image_root": str(root), "protocol": "vqa_stream", "tasks": [{"name": "VQAv2",
        "train": [{"id": "legacy-train", "image": old_train["image"]}], "eval": [{"id": "legacy-eval", "image": old_eval["image"]}]}]})
    before = history.read_bytes()
    result = prep.prepare(config, root, [history], development_subset=True, train_target=8, validation_target=3, eval_target=4)
    rows = all_rows(result)
    assert old_eval["image_sha256"] not in {row["image_sha256"] for row in rows}
    assert old_train["image_sha256"] not in {row["image_sha256"] for task in result["tasks"] for row in task["eval"]}
    assert history.read_bytes() == before
    validate_manifest(result, root)


def test_renamed_identical_bytes_are_excluded_globally_and_in_history(tmp_path):
    config, root = official_sources(tmp_path)
    source = root / "coco/train2014/COCO_train2014_000000000001.jpg"
    alias = save_image(root, "history/alias.png", 9999)
    alias.write_bytes(source.read_bytes())
    history = dump(tmp_path / "old.json", {"image_root": str(root), "tasks": [{"name": "other", "eval": [{"id": "old", "image": "history/alias.png"}]}]})
    result = prep.prepare(config, root, [history], development_subset=True, train_target=8, validation_target=3, eval_target=4)
    assert hashlib.sha256(source.read_bytes()).hexdigest() not in {row["image_sha256"] for row in all_rows(result)}
    # All candidate bytes equal: quotas cannot be filled by distinct names/IDs.
    for path in root.rglob("*.jpg"):
        path.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Insufficient image-disjoint"):
        prepare_development(config, root)


@pytest.mark.parametrize("field,value", [("input_text", " "), ("target", "not referenced"),
                                          ("references", []), ("image_sha256", "bad"),
                                          ("source_split", "test"), ("id", "unqualified")])
def test_validation_rejects_invalid_row_contract(tmp_path, field, value):
    config, root = official_sources(tmp_path)
    manifest = prepare_development(config, root)
    manifest["tasks"][0]["train"][0][field] = value
    with pytest.raises(ValueError):
        validate_manifest(manifest)


@pytest.mark.parametrize("duplicate", ["identity", "hash", "row_id"])
def test_validation_rejects_cross_task_split_duplicates(tmp_path, duplicate):
    config, root = official_sources(tmp_path)
    manifest = prepare_development(config, root)
    first, second = manifest["tasks"][0]["train"][0], manifest["tasks"][1]["validation"][0]
    if duplicate == "identity":
        second["source_image_id"] = first["source_image_id"]
    elif duplicate == "hash":
        second["image_sha256"] = first["image_sha256"]
    else:
        # Same-source duplicate keeps namespace valid so duplicate check is exercised.
        second = copy.deepcopy(first)
        manifest["tasks"][0]["validation"].append(second)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(manifest)


def test_validation_rejects_actual_hash_mismatch_and_history_leakage(tmp_path):
    config, root = official_sources(tmp_path)
    manifest = prepare_development(config, root)
    row = manifest["tasks"][0]["train"][0]
    row["image_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="Actual image SHA256 mismatch"):
        validate_manifest(manifest, root)
    row["image_sha256"] = hashlib.sha256((root / row["image"]).read_bytes()).hexdigest()
    manifest["historical_exclusions"]["eval"]["hashes"].append(row["image_sha256"])
    with pytest.raises(ValueError, match="Historical eval leakage"):
        validate_manifest(manifest)


def test_missing_corrupt_and_mutated_images_do_not_trigger_replacement(tmp_path):
    verifier = ImageVerifier(tmp_path)
    with pytest.raises(FileNotFoundError, match="no download"):
        verifier.verify("missing.jpg")
    (tmp_path / "bad.jpg").write_bytes(b"not decoded image bytes")
    with pytest.raises(ValueError, match="Cannot decode"):
        verifier.verify("bad.jpg")
    save_image(tmp_path, "real.jpg", 3)
    first = verifier.verify("real.jpg")
    save_image(tmp_path, "real.jpg", 4)
    assert verifier.verify("real.jpg") != first
    with pytest.raises(ValueError, match="relative path"):
        verifier.verify("../outside.jpg")


def test_shortages_are_fatal_actionable_and_zero_targets_are_empty(tmp_path):
    config, root = official_sources(tmp_path, size=2)
    with pytest.raises(ValueError, match='"shortage"') as error:
        prepare_development(config, root)
    assert '"requested"' in str(error.value) and '"selected"' in str(error.value)
    result = prep.prepare(config, root, [], development_subset=True, allow_empty_history=True,
                          train_target=0, validation_target=0, eval_target=0)
    assert not all_rows(result)
    validate_manifest(result)


def test_explicit_development_subset_and_history_opt_out_are_required(tmp_path):
    config, root = official_sources(tmp_path)
    with pytest.raises(ValueError, match="all four"):
        prep.prepare(config, root, [])
    with pytest.raises(ValueError, match="ALL historical"):
        prep.prepare(config, root, [], development_subset=True)
    document = json.loads(config.read_text())
    document["sources"][0]["task_id"] = "custom-vqa"
    dump(config, document)
    result = prepare_development(config, root)
    assert result["task_order"] == ["custom-vqa", "C"]
    result["confirmation_eligible"] = True
    with pytest.raises(ValueError, match="non-confirmation"):
        validate_manifest(result)


def test_question_annotation_mismatch_and_unbalanced_gqa_fail(tmp_path):
    config, root = official_sources(tmp_path, include_all=True)
    specs = json.loads(config.read_text())["sources"]
    annotations_path = tmp_path / specs[0]["train"]["annotations"]
    annotations = json.loads(annotations_path.read_text())
    annotations["annotations"][0]["image_id"] += 1
    dump(annotations_path, annotations)
    with pytest.raises(ValueError, match="image mismatch"):
        prep.normalize_source(specs[0], tmp_path, {}, 1)
    questions_path = tmp_path / specs[2]["train"]["questions"]
    questions = json.loads(questions_path.read_text())
    next(iter(questions.values()))["isBalanced"] = False
    dump(questions_path, questions)
    with pytest.raises(ValueError, match="unbalanced"):
        prep.normalize_source(specs[2], tmp_path, {}, 1)


def test_explicit_local_archives_extract_only_exact_needed_members(tmp_path):
    original = tmp_path / "original"
    save_image(original, "one.png", 2)
    archives = tmp_path / "archives"
    archives.mkdir()
    filename = "COCO_train2014_000000000001.jpg"
    with zipfile.ZipFile(archives / "train2014.zip", "w") as archive:
        archive.writestr(f"train2014/{filename}", (original / "one.png").read_bytes())
        archive.writestr("unrelated.txt", b"not extracted")
    gqa_archive = tmp_path / "gqa.zip"
    with zipfile.ZipFile(gqa_archive, "w") as archive:
        archive.writestr("images/42.jpg", (original / "one.png").read_bytes())
    root = tmp_path / "new_images"
    verifier = prep.LocalImages(root, archives, gqa_archive, ["gqa"])
    try:
        assert verifier.verify(f"coco/{filename}") == verifier.verify("gqa/42.jpg")
        assert not (root / "unrelated.txt").exists()
        assert set(verifier.archive_records) == {"train2014", "gqa"}
        with pytest.raises(KeyError):
            verifier.verify("gqa/999.jpg")
        assert not (root / "gqa/999.jpg").exists()
    finally:
        verifier.close()


def test_full_four_source_manifest_and_explicit_three_source_development(tmp_path):
    config, root = official_sources(tmp_path, include_all=True)
    save_image(root, "old/excluded.png", 9900)
    history = dump(tmp_path / "old.json", {"image_root": str(root), "tasks": [
        {"name": "old", "eval": [{"id": "old-eval", "image": "old/excluded.png"}]}]})
    manifest = prep.prepare(config, root, [history], train_target=8, validation_target=3, eval_target=4)
    assert manifest["protocol"] == "multimodal_cl"
    assert manifest["confirmation_eligible"] is False
    assert manifest["task_order"] == ["V", "C", "G", "F"]
    validate_manifest(manifest, root)
    document = json.loads(config.read_text())
    document["sources"].pop()
    dump(config, document)
    development = prep.prepare(config, root, [history], development_subset=True,
                               train_target=8, validation_target=3, eval_target=4)
    assert development["task_order"] == ["V", "C", "G"]
    assert development["confirmation_eligible"] is False
    validate_manifest(development, root)
