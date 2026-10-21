"""Aggregate saved G4 statistics/gains only; no scoring, inference, or new tests."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def aggregate(root):
    paths = [root / 'artifacts/multimodal-g4-report.tar.gz',
             root / 'artifacts/multimodal-g4-statistics.json',
             root / 'artifacts/multimodal-g1-normalizers.json']
    with tarfile.open(paths[0]) as archive:
        member = next(n for n in archive.getnames() if n.endswith('/audit.json'))
        audit = json.load(archive.extractfile(member))
    saved = json.loads(paths[1].read_text())
    normalizers = json.loads(paths[2].read_text())['scores']
    trials = [t for t in audit['trials'] if t['kind'] == 'common_cl' and t['method'] == 'expand']
    gains = [g for t in trials for g in t['subsequent_task_gains']]
    assert len(trials) == 32 and len(gains) == 96
    for g in gains:
        assert math.isclose(g['normalized_gain'], g['gain'] / normalizers[g['task_id']], abs_tol=1e-12)
    summaries = {}
    for group in ('vqa', 'caption', 'V', 'C', 'G', 'F'):
        selected = [g for g in gains if g['family'] == group or g['task_id'] == group]
        summaries[group] = {
            'transitions': len(selected), 'negative_transitions': sum(g['gain'] < 0 for g in selected),
            'raw_transition_mean': statistics.mean(g['gain'] for g in selected),
            'normalized_transition_mean': statistics.mean(g['normalized_gain'] for g in selected)}
        if group in ('vqa', 'caption'):
            slots = [[g for g in t['subsequent_task_gains'] if g['family'] == group] for t in trials]
            summaries[group]['raw_equal_order_seed_mean'] = statistics.mean(
                statistics.mean(g['gain'] for g in slot) for slot in slots)
            summaries[group]['normalized_equal_order_seed_mean'] = statistics.mean(
                statistics.mean(g['normalized_gain'] for g in slot) for slot in slots)
            original = next(r for r in saved['subsequent_gain_statistics'] if r['family'] == group)
            for computed, recorded in [('raw_transition_mean', original['mean_native_gain']),
                                       ('normalized_transition_mean', original['mean_normalized_gain']),
                                       ('normalized_equal_order_seed_mean', original['statistics']['mean'])]:
                assert math.isclose(summaries[group][computed], recorded, abs_tol=1e-12)
    comparisons = []
    for family, key in [('primary_six', 'primary_comparisons'), ('tuned_two', 'tuned_replay_comparisons')]:
        for row in saved[key]:
            comparisons.append({'family': family, 'label': row['label'],
                                'raw_p': row['statistics']['exact_two_sided_sign_flip_p'],
                                'holm_p': row['holm_adjusted_p'],
                                'mean_difference': row['statistics']['mean']})
    n = len(audit['lock']['seeds'])
    raw_floor = 2 / 2 ** n
    inputs = []
    for path in paths:
        with path.open('rb') as handle:
            inputs.append({'path': str(path.relative_to(root)),
                           'sha256': hashlib.file_digest(handle, 'sha256').hexdigest()})
    return {'scope': 'Existing saved scores only; no model inference, rescoring or new significance tests.',
            'inputs': inputs, 'audit_archive_member': member, 'seed_count': n,
            'exact_two_sided_p_floor': raw_floor,
            'holm_primary_six_floor': 6 * raw_floor, 'holm_tuned_two_floor': 2 * raw_floor,
            'comparisons': comparisons, 'gain_summaries': summaries,
            'gain_weighting': 'Transition means weight each subsequent task equally. Equal-order/seed means first average eligible tasks within each trial, then average 32 equally weighted trials; this matches the saved gain inference statistic.',
            'normalizers': normalizers,
            'caption_score_units': {task: {key: audit['lock']['scorer_locks'][task][key]
                                          for key in ('units', 'scale', 'idf_source')}
                                    for task in ('C', 'F')},
            'scale_note': 'Native pycocoevalcap CIDEr-D output is reported x1, not presentation x100. Its internal CIDEr scale factor is 10.0; do not confuse internal scale with external display multiplier.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('artifacts/multimodal-review-statistics.json'))
    args = parser.parse_args()
    report = aggregate(args.root.resolve())
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output), 'gains': report['gain_summaries'],
                      'comparisons': report['comparisons']}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
