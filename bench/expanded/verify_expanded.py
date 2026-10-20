"""Recompute the expanded 3,300-call panel tables from the included observations.

Run with no arguments to check everything the release contains: row coverage, probability
validity, both tie rules, every count in summary.json, and the legacy213 paired comparison
against the published Jev outcomes in ../legacy213-observations.jsonl.

ChaosNLI gold labels and Jev's ChaosNLI predictions are not redistributed. To recompute the
full ChaosNLI accuracy and the Jev comparison, download the two pinned files listed in
summary.json["sources"] and pass them with --chaosnli-rows and --jev-predictions.
"""
from math import comb, isfinite
from pathlib import Path
import argparse, hashlib, json

HERE = Path(__file__).resolve().parent
EXPECTED_ROWS = {('legacy213strict', 'primary'): 213, ('legacy18score', 'primary'): 18,
                 ('dmup', 'primary'): 200, ('dmup', 'shuffle_controls'): 40, ('chaosnli96', 'primary'): 96}


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def lexi(p):
    return min(p, key=lambda k: (-p[k], k))


def first_slot(p, order):
    m = max(p[k] for k in order)
    return next(k for k in order if p[k] == m)


def mcnemar(ref, ours, ids):
    plus = sum(1 for i in ids if not ref[i] and ours[i])
    minus = sum(1 for i in ids if ref[i] and not ours[i])
    n = plus + minus
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, j) for j in range(min(plus, minus) + 1)) / 2 ** n)
    return {'reference_correct': sum(ref[i] for i in ids), 'ours_correct': sum(ours[i] for i in ids),
            'ours_only_correct': plus, 'reference_only_correct': minus, 'exact_two_sided_p': p}


def key(o):
    k = o['lane'] + ('' if o['panel'] == 'full_chaos' else '/repeat' + str(o['repeat']))
    return k + ('' if o['block'] == 'primary' else '/' + o['block'])


def verify(root=HERE, chaosnli_rows=None, jev_predictions=None):
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    obs = read_jsonl(root / 'observations.jsonl')
    assert len(obs) == 3300 and summary['measured_calls'] == 3300
    assert sum(o['panel'] == 'pb1' for o in obs) == 1701 and sum(o['panel'] == 'full_chaos' for o in obs) == 1599
    seen = set()
    for o in obs:
        ident = (o['panel'], o['lane'], o['repeat'], o['block'], o['item_id'])
        assert ident not in seen, ident
        seen.add(ident)
        p, order = o['probabilities'], o['option_order']
        assert len(set(order)) == len(order)
        if not o['valid']:
            # failed call: wrong under both rules, no probabilities (2026-10-20 run: 16 socket-timeout 502s)
            assert o['panel'] == 'pb1' and o['http_status'] != 200 and p is None
            assert o['predicted'] is None and o['predicted_serving'] is None
            assert o['correct'] is False and o['correct_serving'] is False
            continue
        assert sorted(p) == sorted(order)
        assert all(isfinite(v) and 0 <= v <= 1 for v in p.values()) and abs(sum(p.values()) - 1) < 1e-6
        assert o['predicted'] == lexi(p) and o['predicted_serving'] == first_slot(p, order)
        if 'gold' in o:
            assert o['correct'] == (o['predicted'] == o['gold'])
            assert o['correct_serving'] == (o['predicted_serving'] == o['gold'])
        elif o['predicted'] == o['predicted_serving']:
            assert o['correct'] == o['correct_serving']
    for (lane, block), n in EXPECTED_ROWS.items():
        for rep in (1, 2, 3):
            got = sum(1 for o in obs if o['lane'] == lane and o['block'] == block and o['repeat'] == rep)
            assert got == n, (lane, block, rep, got)
    for rule, name in (('correct', 'counts_benchmark_rule'), ('correct_serving', 'counts_serving_rule')):
        counts = {}
        for o in obs:
            c = counts.setdefault(key(o), {'correct': 0, 'n': 0})
            c['correct'] += o[rule]
            c['n'] += 1
        assert counts == summary[name], name
    full_ids = {o['item_id'] for o in obs if o['panel'] == 'full_chaos'}
    overlap = {o['item_id'] for o in obs if o['panel'] == 'full_chaos' and o['also_in_repeated_96_lane']}
    assert len(full_ids) == 1599 and overlap == {o['item_id'] for o in obs if o['lane'] == 'chaosnli96'} and len(overlap) == 96

    legacy = read_jsonl(root.parent / 'legacy213-observations.jsonl')
    jev = {r['task_id']: r['correct'] for r in legacy if r['system'] == 'jev-1.13.0-published'}
    assert len(jev) == 213 and sum(jev.values()) == 186
    cmp_ = summary['comparisons']['legacy213strict_vs_published_jev_1_13_0']['by_rule_and_repeat']
    for rule in ('correct', 'correct_serving'):
        for rep in (1, 2, 3):
            ours = {o['item_id']: o[rule] for o in obs if o['lane'] == 'legacy213strict' and o['repeat'] == rep}
            assert set(ours) == set(jev)
            assert mcnemar(jev, ours, sorted(ours)) == cmp_[rule][str(rep)], (rule, rep)

    report = {'observations': len(obs), 'counts_match_summary': True, 'legacy213_comparison_match': True,
              'chaosnli_full_recomputed_from_source': False}
    if chaosnli_rows and jev_predictions:
        src = summary['sources']
        for path, want in ((chaosnli_rows, src['chaosnli_rows']['sha256']), (jev_predictions, src['jev_chaosnli_predictions']['sha256'])):
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == want, path
        gold = {r['id']: r['label'] for r in read_jsonl(chaosnli_rows)}
        jev_c = {}
        for r in read_jsonl(jev_predictions):
            if r['id'].startswith('chaosnli/'):
                jev_c[r['id']] = r['answer'] == gold[r['id']]
        full = [o for o in obs if o['panel'] == 'full_chaos']
        for o in full:
            assert o['correct'] == (o['predicted'] == gold[o['item_id']])
            assert o['correct_serving'] == (o['predicted_serving'] == gold[o['item_id']])
        ref = summary['comparisons']['chaosnli_full1599_vs_published_jev_1_13_0']['by_rule']
        for rule in ('correct', 'correct_serving'):
            ours = {o['item_id']: o[rule] for o in full}
            assert mcnemar(jev_c, ours, sorted(ours)) == ref[rule], rule
        report['chaosnli_full_recomputed_from_source'] = True
    report['legacy213_by_repeat'] = [summary['counts_benchmark_rule']['legacy213strict/repeat%d' % r]['correct'] for r in (1, 2, 3)]
    report['chaosnli_full_correct'] = summary['counts_benchmark_rule']['chaosnli_full1599']['correct']
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--chaosnli-rows', help='pinned data/chaosnli/test.jsonl from summary.json sources')
    ap.add_argument('--jev-predictions', help='pinned results/jev-1.13.0/test_predictions.jsonl from summary.json sources')
    a = ap.parse_args()
    print(json.dumps(verify(chaosnli_rows=a.chaosnli_rows, jev_predictions=a.jev_predictions), indent=2))
