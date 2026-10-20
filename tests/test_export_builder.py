"""Guarded patch construction over a generated, non-vLLM source fixture."""
import hashlib
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_patcher', ROOT / 'engine/build_capture_patch.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


def build(tmp_path, text=None, **kwargs):
    text = text or ('from vllm.logger import init_logger\n'
                    'class GPUModelRunner:\n'
                    '    def sample(self, hidden_states, input_batch):\n'
                    '        sample_hidden_states = hidden_states[input_batch.logits_indices]\n'
                    '        logits = self.model.compute_logits(sample_hidden_states)\n'
                    '        if grammar_output is not None:\n'
                    '            pass\n')
    src = tmp_path / 'input.py'
    src.write_text(text, encoding='utf-8', newline='\n')
    options = {'src': src, 'out': tmp_path / 'output.py',
               'expect_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
               'expect_bytes': src.stat().st_size}
    options.update(kwargs)
    return options


def test_builder_refuses_in_place_without_changing_source(tmp_path):
    args = build(tmp_path)
    before = args['src'].read_bytes()
    args['out'] = args['src']
    with pytest.raises(patcher.PatchError):
        patcher.build_patch(**args)
    assert args['src'].read_bytes() == before


def test_builder_applies_only_after_logits_and_keeps_source(tmp_path):
    args = build(tmp_path)
    before = args['src'].read_bytes()
    meta = patcher.build_patch(**args)
    text = args['out'].read_text()
    assert text.index('logits = self.model.compute_logits') < text.index('_qfn_capture_after_logits(self, hidden_states, logits, input_batch)') < text.index('if grammar_output is not None:')
    assert args['src'].read_bytes() == before
    assert meta['patched_function_statement_deltas'] == {'sample': 1}


@pytest.mark.parametrize('guard,value', [
    ('expect_sha256', '0' * 64), ('expect_bytes', 0), ('source_version', 'unknown'),
])
def test_builder_refuses_wrong_identity_without_output(tmp_path, guard, value):
    args = build(tmp_path, **{guard: value})
    with pytest.raises(patcher.PatchError):
        patcher.build_patch(**args)
    assert not args['out'].exists()


def test_builder_refuses_marker_and_ambiguous_anchor(tmp_path):
    args = build(tmp_path)
    text = args['src'].read_text()
    for altered in (text + '\n' + patcher.MARKER, text.replace(patcher.ANCHOR_CALL, patcher.ANCHOR_CALL * 2)):
        args = build(tmp_path, altered)
        with pytest.raises(patcher.PatchError):
            patcher.build_patch(**args)
        assert not args['out'].exists()
