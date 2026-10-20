"""Recompute current legacy results and the descriptive AutoTrust comparison, offline.

Run without flags to verify. --write updates only the three derived output files.
Measured expanded observations and the published reference are never changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(reference, measured):
    assert set(reference) == set(measured)
    improved = sum(measured[k] and not reference[k] for k in reference)
    regressed = sum(reference[k] and not measured[k] for k in reference)
    discordant = improved + regressed
    p = min(1.0, 2 * sum(math.comb(discordant, j) for j in range(min(improved, regressed) + 1)) / 2 ** discordant)
    return {'first_correct': sum(reference.values()), 'second_correct': sum(measured.values()),
            'second_minus_first_items': improved - regressed, 'improved': improved, 'regressed': regressed,
            'mcnemar_exact_two_sided_p': p}


def build(root):
    root = Path(root)
    task_meta = json.loads((root / 'legacy231-task-metadata.json').read_text())
    tasks = task_meta['items']
    assert len(tasks) == 231 and len({t['family'] for t in tasks.values()}) == 18
    source = read_rows(root / 'expanded/observations.jsonl')
    legacy = [r for r in source if r['lane'] in ('legacy213strict', 'legacy18score')]
    assert len(source) == 3300 and len(legacy) == 693
    assert len({(r['repeat'], r['lane'], r['item_id']) for r in legacy}) == 693
    assert {r['repeat'] for r in legacy} == {1, 2, 3}
    reference = [r for r in read_rows(root / 'legacy213-observations.jsonl') if r['system'] == 'jev-1.13.0-published']
    assert len(reference) == 213 and len({r['task_id'] for r in reference}) == 213
    assert sum(r['correct'] for r in reference) == 186
    ref = {r['task_id']: r['correct'] for r in reference}
    strict_rows, runs, repeats, comparisons = [], [], [], []
    by_repeat = {}
    for repeat in (1, 2, 3):
        rs = [r for r in legacy if r['repeat'] == repeat]
        assert len(rs) == 231 and {r['item_id'] for r in rs} == set(tasks)
        families = {}
        strict = []
        for r in rs:
            task = tasks[r['item_id']]
            mapping = {'false': 'no', 'true': 'yes'} if task['gold'] in ('yes', 'no') else {}
            rename = lambda key: mapping.get(key, key)
            assert rename(r['gold']) == task['gold']
            if r['valid']:
                p = r['probabilities']
                assert set(p) == set(r['option_order'])
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
                assert abs(sum(p.values()) - 1) < 1e-6
                prediction = min(p, key=lambda key: (-p[key], key))
                serving = max(r['option_order'], key=p.get)
                assert r['predicted'] == prediction and r['predicted_serving'] == serving
                assert r['correct'] == (prediction == r['gold'])
                assert r['correct_serving'] == (serving == r['gold'])
            else:
                assert r['probabilities'] is None and not r['correct'] and not r['correct_serving']
            family = families.setdefault(task['family'], {'n': 0, 'correct': 0, 'correct_serving': 0})
            family['n'] += 1
            family['correct'] += r['correct']
            family['correct_serving'] += r['correct_serving']
            if r['lane'] == 'legacy213strict':
                assert r['item_id'] in ref
                strict.append({'system': 'B0', 'repeat': repeat, 'task_id': r['item_id'], 'gold': task['gold'],
                               'valid': r['valid'], 'probabilities': {rename(k): v for k, v in r['probabilities'].items()} if r['valid'] else None,
                               'predicted': rename(r['predicted']) if r['valid'] else None, 'correct': r['correct']})
        assert len(strict) == 213 and len(families) == 18
        by_repeat[repeat] = {r['task_id']: r for r in strict}
        strict_rows.extend(strict)
        valid = [r for r in strict if r['valid']]
        ece = 0.0
        for i in range(10):
            bucket = [r for r in valid if min(9, int(max(r['probabilities'].values()) * 10)) == i]
            if bucket:
                ece += len(bucket) / len(valid) * abs(statistics.mean(max(r['probabilities'].values()) for r in bucket) - statistics.mean(r['correct'] for r in bucket))
        correct = sum(r['correct'] for r in strict)
        runs.append({'system': 'B0', 'repeat': repeat, 'n': 213, 'correct': correct, 'accuracy': correct / 213,
                     'valid_probability_n': len(valid), 'ece_10_equal_width': ece})
        comparisons.append(paired(ref, {r['task_id']: r['correct'] for r in strict}))
        repeats.append({'repeat': repeat, 'n': 231, 'correct': sum(r['correct'] for r in rs),
                        'micro_accuracy': statistics.mean(r['correct'] for r in rs), 'n_families': 18,
                        'family_macro_accuracy': statistics.mean(f['correct'] / f['n'] for f in families.values()),
                        'family_macro_serving_rule': statistics.mean(f['correct_serving'] / f['n'] for f in families.values()),
                        'families': dict(sorted(families.items()))})
    all_rows = reference + strict_rows
    text = ''.join(json.dumps(r, sort_keys=True, separators=(',', ':')) + '\n' for r in all_rows)
    scope = ['Three repeats reuse public items that were used during development; not fresh holdout evidence.',
             'Jev was not rerun; the reference is its saved published per-item output.',
             'This is not an official modern JevBench leaderboard score.',
             'Failed attempts count wrong; confidence metrics use valid distributions only.']
    summary = {'schema': 'qfn.legacy213-results/2', 'date_utc': '2026-10-20', 'items': 213,
               'benchmark': 'JevBench public legacy strict subset', 'base_arms': 'unchanged B0',
               'runs': runs, 'mean_accuracy': {'B0': statistics.mean(r['accuracy'] for r in runs)},
               'descriptive_B0_vs_saved_Jev': comparisons,
               'same_arm_repeat_agreement': {'B0': {f'{a}-{b}': sum(by_repeat[a][k]['predicted'] == by_repeat[b][k]['predicted'] for k in ref) for a, b in ((1, 2), (1, 3), (2, 3))}},
               'observations_sha256': hashlib.sha256(text.encode()).hexdigest(),
               'expanded_observations_sha256': digest(root / 'expanded/observations.jsonl'),
               'expanded_panel': {'measured_calls': 3300, 'errors': 16, 'summary': 'expanded/summary.json'}, 'scope': scope}
    macro = statistics.mean(r['family_macro_accuracy'] for r in repeats)
    comparison = {'schema': 'qfn.autotrust-comparison/2', 'date_utc': '2026-10-20',
                  'derivative_analysis': 'Family-macro re-aggregation of all three repeats from the current 3,300-call run; no new inference.',
                  'observations_sha256': digest(root / 'expanded/observations.jsonl'),
                  'task_metadata_sha256': digest(root / 'legacy231-task-metadata.json'),
                  'task_revision': task_meta['revision'], 'tasks': task_meta['sources'], 'repeats': repeats,
                  'mean_family_macro_accuracy': macro,
                  'mean_family_macro_accuracy_serving_rule': statistics.mean(r['family_macro_serving_rule'] for r in repeats),
                  'published_autotrust_family_macro': 0.887, 'difference_points_descriptive': (macro - 0.887) * 100,
                  'autotrust_source': {'url': 'https://huggingface.co/autotrust/JEV-27B/blob/14b425269ed24452e0a5ad5a8f3e51d139647a88/README.md',
                                      'readme_sha256': 'ddff9a781adf07fef7a5985390e3ee18e46b0cd75aefc1886bd102a8bdfe9985'},
                  'caveats': scope + ['The 18 score tasks use choice surrogates; AutoTrust uses native score.',
                                     'No AutoTrust per-item outcomes or exact benchmark-input hashes are available here; this is descriptive, not a controlled paired comparison.']}
    return {'legacy213-observations.jsonl': text, 'results.json': json.dumps(summary, indent=2) + '\n',
            'autotrust-comparison.json': json.dumps(comparison, indent=2) + '\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    outputs = build(root)
    for name, text in outputs.items():
        if args.write:
            (root / name).write_text(text, encoding='utf-8', newline='\n')
        else:
            assert (root / name).read_text(encoding='utf-8') == text, name + ' is stale'
    print(json.dumps({'verified_files': list(outputs), 'repeats': 3, 'legacy231_observations': 693}))


if __name__ == '__main__':
    main()
