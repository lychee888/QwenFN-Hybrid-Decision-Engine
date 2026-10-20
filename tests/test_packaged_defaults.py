"""Default dictionaries come from package data, not a checkout or caller's cwd."""
import importlib.resources
import json
from qfn_decision import backend


def test_default_dictionary_is_available_without_a_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend, '__file__', str(tmp_path / 'installed' / 'qfn_decision' / 'backend.py'))
    expected = json.loads(importlib.resources.files('qfn_decision').joinpath('default_api.json').read_text(encoding='utf-8'))
    assert backend.default_api_config_dict() == expected


def test_default_dictionary_does_not_read_foreign_cwd_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend, '__file__', str(tmp_path / 'installed' / 'qfn_decision' / 'backend.py'))
    foreign = tmp_path / 'configs'
    foreign.mkdir()
    (foreign / 'decision_api.json').write_text('{"foreign_config": true}', encoding='utf-8')
    assert 'foreign_config' not in backend.default_api_config_dict()
