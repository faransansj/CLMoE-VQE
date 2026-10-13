import pytest
import torch

from sema_vqa.readout_router import FeatureRouter, question_stem


def samples(axis, count=24):
    x = torch.zeros(count, 1536)
    x[:, axis] = 1
    x[:, 768 + axis] = 1
    return x


def test_router_learns_new_expert_and_restores_old_routes():
    router = FeatureRouter(memory_per_task=8)
    router.observe(samples(0), 0, 1)
    assert router.predict(samples(1)).tolist() == [0] * 24
    router.observe(samples(1), 1, 2)
    assert router.predict(torch.cat((samples(0, 2), samples(1, 3)))).tolist() == [0, 0, 1, 1, 1]
    restored = FeatureRouter.from_state_dict(router.state_dict())
    assert torch.equal(restored.predict(samples(1)), router.predict(samples(1)))
    assert [len(x) for x, _ in router.memory] == [8, 8]


def test_question_routing_ignores_image_change():
    router = FeatureRouter(modality="question")
    router.observe(samples(0), 0, 1)
    router.observe(samples(1), 1, 2)
    changed = samples(0)
    changed[:, :768] = samples(1)[:, :768] * 99
    assert router.predict(changed).tolist() == [0] * len(changed)


def test_multiple_tasks_may_reuse_same_expert():
    router = FeatureRouter()
    router.observe(samples(0), 0, 1)
    router.observe(samples(1), 0, 1)
    router.observe(samples(2), 1, 2)
    assert router.predict(torch.cat((samples(0, 1), samples(1, 1), samples(2, 1)))).tolist() == [0, 0, 1]


def test_router_rejects_unfitted_or_invalid_features():
    router = FeatureRouter()
    with pytest.raises(ValueError, match="not been fitted"):
        router.predict(samples(0))
    with pytest.raises(ValueError, match="Invalid training expert"):
        router.observe(samples(0), 1, 1)
    with pytest.raises(ValueError, match="finite"):
        router.observe(torch.full((2, 1536), float("nan")), 0, 1)


def test_router_stem_excludes_source_directives_but_keeps_question():
    plain = "What color is the floor?"
    decorated = "<image>\nWhat   color is the floor?.\nReference OCR token: BLUE\nAnswer with one word."
    assert question_stem(plain) == question_stem(decorated) == "what color is the floor"
    assert question_stem("<image>\nHow many cats?\nA. One\nB. Two") == "how many cats"
    with pytest.raises(ValueError, match="Empty"):
        question_stem("<image>\n ")
