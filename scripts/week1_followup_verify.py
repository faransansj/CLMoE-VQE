"""Verify Week-1 C/D outputs and aggregate saved expert pairs; no scoring/inference."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/week1-followup-verification.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(name):
    return json.loads((ROOT / 'artifacts' / name).read_text())


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    baseline = load('week1-followup-baseline.json')
    for path, expected in baseline['sha256'].items():
        assert digest(ROOT / path) == expected, path
    score = load('week1-per-example-rescore.json')
    cases = load('week1-batch1-cases.json')
    regen = load('week1-batch1-regeneration.json')
    assert score['status'] == 'pass' and not score['failures']
    for entry in score['inputs']:
        assert digest(ROOT / entry['path']) == entry['sha256'], entry['path']
    for path, expected in regen['local_verifications']['inputs'].items():
        assert digest(ROOT / path) == expected, path
    assert digest(ROOT / 'artifacts/week1-per-example-scores.jsonl.gz') == score['output_rows_sha256']
    assert digest(ROOT / 'artifacts/week1-batch1-cases.json') == regen['cases_sha256']
    assert digest(ROOT / score['postprocessing']['script']) == score['postprocessing']['script_sha256']

    groups = defaultdict(list)
    with gzip.open(ROOT / 'artifacts/week1-per-example-scores.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            order = '_'.join(row['order']) if row['order'] else None
            bank = (f"{order}_expand_lr0.0003_ep5_seed{row['seed']}" if order
                    else f"{row['task']}_independent_lr0.0003_ep5_seed{row['seed']}")
            key = f"{row['kind']}|{bank}|{row['task']}_p{row['stage']:02d}|{row['phase']}"
            groups[key].append(row)
    assert len(groups) == 416
    max_delta = 0.0
    for corpus in score['per_corpus']:
        rows = groups[corpus['key']]
        assert len(rows) == len({r['id'] for r in rows}) == 512
        value = math.fsum(r['score'] for r in rows) / len(rows)
        for ref in corpus['sources']:
            max_delta = max(max_delta, abs(value - ref['fresh']))
            assert abs(value - ref['stored']) <= 1e-9
    assert max_delta <= 1e-9

    pairs = []
    for cell in score['flickr_pre_post_cells']:
        bank = f"{cell['order']}_expand_lr0.0003_ep5_seed{cell['seed']}"
        pre = groups[cell['pre_native_corpus']]
        post = groups[f'oracle|{bank}|F_p03|after']
        assert [r['id'] for r in pre] == [r['id'] for r in post]
        assert all(r['expert_name'] == 'C' for r in pre)
        assert all(r['expert_name'] == 'F' for r in post)
        deltas = [f['score'] - c['score'] for c, f in zip(pre, post)]
        pairs.append({'order': cell['order'], 'seed': cell['seed'], 'n': len(deltas),
                      'mean_new_f_minus_old_c': math.fsum(deltas) / len(deltas),
                      'new_f_wins': sum(d > 0 for d in deltas),
                      'old_c_wins': sum(d < 0 for d in deltas),
                      'ties': sum(d == 0 for d in deltas),
                      'pre_c_corpus': cell['pre_native_corpus'],
                      'post_f_corpus': f'oracle|{bank}|F_p03|after'})
    assert len(pairs) == 16

    by_id = {c['occurrence_id']: c for c in cases['occurrences']}
    seen, matches = set(), Counter()
    for line in (ROOT / 'artifacts/week1-batch1-predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        for oid in row['occurrence_ids']:
            assert oid not in seen
            seen.add(oid)
            case = by_id[oid]
            assert row['bank'] == case['bank'] and row['row_id'] == case['row_id']
            n = row['regenerated_prediction'] == case['native_prediction']
            o = row['regenerated_prediction'] == case['oracle_prediction']
            label = 'both' if n and o else 'native' if n else 'oracle' if o else 'neither'
            assert label == row['match_class']
            matches[label] += 1
    assert seen == set(by_id) and len(seen) == 62
    assert all(v['expert_tensors_identical_pre_post'] and v['whole_model_identical_pre_post_bank']
               for v in regen['banks'].values())

    input_names = ['week1-followup-baseline.json', 'week1-per-example-rescore.json',
                   'week1-per-example-scores.jsonl.gz', 'week1-batch1-cases.json',
                   'week1-batch1-predictions.jsonl', 'week1-batch1-regeneration.json',
                   'week1-batch1-execution-provenance.zip', 'week1-per-example-execution-provenance.zip']
    payload = {
        'status': 'pass',
        'script': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': digest(Path(__file__))},
        'inputs': [{'path': 'artifacts/' + name, 'sha256': digest(ROOT / 'artifacts' / name)}
                   for name in input_names],
        'protected_files_unchanged': len(baseline['sha256']),
        'per_example_corpora': len(groups), 'per_example_rows': sum(map(len, groups.values())),
        'max_row_mean_vs_fresh': max_delta,
        'batch1_occurrences': len(seen), 'batch1_matches_recomputed': dict(matches),
        'flickr_old_c_vs_new_f': {
            'units': 'native CIDEr-D x1; frozen task train-IDF',
            'cells': pairs, 'occurrences': sum(p['n'] for p in pairs),
            'cells_new_f_lower': sum(p['mean_new_f_minus_old_c'] < 0 for p in pairs),
            'mean_new_f_minus_old_c': math.fsum(p['mean_new_f_minus_old_c'] for p in pairs) / len(pairs),
            'new_f_wins': sum(p['new_f_wins'] for p in pairs),
            'old_c_wins': sum(p['old_c_wins'] for p in pairs),
            'ties': sum(p['ties'] for p in pairs),
            'interpretation': 'Saved pre-stage native outputs all use C; saved post-stage oracle outputs use F. Same images paired across stages; observed output-quality comparison, not a controlled causal intervention. Original batch composition may differ. Repeated images/orders/seeds are occurrences, not independent images.'},
        'new_scoring_or_inference_by_this_script': False,
        'failed_batch1_attempts': [
            {'job': '2557282', 'failure': 'pre-inference model digest on dict extra-state',
             'generation_calls': 0, 'persisted_rows': 0, 'allocation_seconds': None},
            {'job': '2557297', 'failure': 'record construction after first generation, missing occurrence_ids key',
             'generation_calls': 1, 'persisted_rows': 0, 'allocation_seconds': None}],
        'cost_limit': 'Failed GPU attempts have no surviving scheduler allocation records; successful-run allocation is not total campaign cost.'
    }
    with OUTPUT.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'pass', 'rows': payload['per_example_rows'],
                      'max_row_mean_delta': max_delta, 'batch1_matches': dict(matches),
                      'new_f_minus_old_c': payload['flickr_old_c_vs_new_f']['mean_new_f_minus_old_c']}))


if __name__ == '__main__':
    main()
