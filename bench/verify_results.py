"""Verify current legacy213 and family-macro derivatives from all preserved repeats."""
from pathlib import Path
import importlib.util
import json


def verify(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('qfn_benchmark_recompute', root / 'recompute_legacy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, expected in module.build(root).items():
        assert (root / name).read_text(encoding='utf-8') == expected, name + ' is stale'
    result = json.loads((root / 'results.json').read_text())
    assert [r['correct'] for r in result['runs']] == [198, 198, 199]
    assert result['expanded_panel']['errors'] == 16
    return {'items': 213, 'repeats': 3, 'correct_by_repeat': [198, 198, 199], 'legacy231_repeats_verified': 3}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
