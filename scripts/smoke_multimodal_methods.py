"""Exercise all non-replay method transitions on real validation-only tiny data."""
import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'assets', 'image-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from sema_multimodal.run import main as run, safe_output, atomic_json
    import torch
    output = safe_output(args.output, smoke=True)
    if output.exists():
        raise ValueError('Acceptance requires an unused output directory')
    manifest = json.loads(args.manifest.read_text())
    vqa = next(task['task_id'] for task in manifest['tasks'] if task['family'] == 'vqa')
    caption = next(task['task_id'] for task in manifest['tasks'] if task['family'] == 'caption')
    common = ['--manifest', str(args.manifest), '--assets', str(args.assets),
              '--image-root', str(args.image_root), '--task-ids', vqa, caption,
              '--epochs', '1', '--batch-size', '3', '--micro-batch-size', '2',
              '--train-limit', '5', '--eval-limit', '2', '--smoke-output']
    observed = {}
    for method in ('shared_lora', 'capacity_matched_shared', 'expand', 'frozen_first', 'independent'):
        command = common + ['--method', method, '--output', str(output / method)]
        if method == 'expand':
            command += ['--oracle-diagnostic']
        result = run(command)
        assert result['status'] == 'complete' and result['completed'] == 2
        assert [len(row) for row in result['matrix']] == [1, 2]
        assert result['resources']['steps'] == (2 if method == 'frozen_first' else 4)
        assert result['resources']['image_exposures'] == (5 if method == 'frozen_first' else 10)
        assert result['resources']['replay_exposures'] == 0
        stages = result['stage_resources']
        if method in ('frozen_first', 'independent'):
            assert result['matrix'][1][0] == result['matrix'][0][0]
        if method == 'frozen_first':
            assert result['matrix'][1][1] == result['pre_task_scores'][1]
            assert stages[1]['parameter_counts']['trainable'] == 0
        if method == 'expand':
            assert [row['parameter_counts']['experts'] for row in stages] == [1179648, 2359296]
            assert all(row['router_memory_bytes'] > 0 for row in stages)
            assert [len(row) for row in result['oracle_matrix']] == [1, 2]
        else:
            assert all(row['router_memory_bytes'] == 0 for row in stages)
        if method == 'independent':
            assert result['resources']['independent_storage_bytes'] > 0
            assert [row['parameter_counts']['experts'] for row in stages] == [1179648, 1179648]
        observed[method] = {'matrix': result['matrix'], 'pre_task_scores': result['pre_task_scores'],
                            'stage_resources': stages, 'resources': result['resources']}
        gc.collect()
        torch.cuda.empty_cache()
    matched = observed['capacity_matched_shared']['stage_resources'][-1]['parameter_counts']
    expanded = observed['expand']['stage_resources'][-1]['parameter_counts']
    assert matched['experts'] == expanded['experts']
    assert matched['total'] == expanded['total']
    atomic_json(output / 'smoke.json', {'scope': 'Real method-transition acceptance, not learnability',
                                      'actual_gpu': torch.cuda.get_device_name(),
                                      'capacity_total_parameter_equivalence': True,
                                      'methods': observed})
    print('PASS: shared/capacity/expand/frozen-first/independent real transitions and matched total parameters')


if __name__ == '__main__':
    main()
