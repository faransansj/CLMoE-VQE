import pytest

from scripts.compare_readout import holm, paired_statistics, pick


def test_exact_seed_test_resolution_is_not_overclaimed():
    five = paired_statistics([1.0] * 5)
    eight = paired_statistics([1.0] * 8)
    assert five["exact_two_sided_sign_flip_p"] == 2 / 32
    assert eight["exact_two_sided_sign_flip_p"] == 2 / 256
    assert eight["seed_bootstrap_95_percentile"] == [1, 1]
    assert eight["sample_sd"] == 0


def test_signed_deltas_and_null_symmetry():
    positive = paired_statistics([1, 2, 3, 4])
    negative = paired_statistics([-1, -2, -3, -4])
    assert positive["mean"] == -negative["mean"] == 2.5
    assert positive["exact_two_sided_sign_flip_p"] == negative["exact_two_sided_sign_flip_p"]
    assert paired_statistics([1, -1, 2, -2])["exact_two_sided_sign_flip_p"] == 1
    assert paired_statistics([0, 0, 0])["exact_two_sided_sign_flip_p"] == 1


def test_holm_corrects_whole_family_and_restores_order():
    assert holm([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    assert holm([0.7, 0.8]) == [1, 1]


def test_seed_pairing_never_drops_failed_or_inconvenient_seeds():
    runs = [{"config": {"method": "expand", "seed": seed}} for seed in (17, 29, 43)]
    assert set(pick(runs, {"method": "expand"}, [17, 29, 43])) == {17, 29, 43}
    with pytest.raises(ValueError, match="Missing, duplicate"):
        pick(runs, {"method": "expand"}, [17, 29, 43, 59])
    with pytest.raises(ValueError, match="Missing, duplicate"):
        pick(runs + [runs[0]], {"method": "expand"}, [17, 29, 43])
