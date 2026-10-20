"""Public gateway startup and retained synthetic loopback contracts."""
import dataclasses
import http.client
import json
from pathlib import Path
import pytest
from qfn_decision import api, backend, configure, protocol
from qfn_decision.hybrid import HybridBackend, ROWS


def config(tmp_path, **env):
    payload, _ = configure.build_configs({'CAPTURE_DIR': str(tmp_path / 'captures'), **env})
    return backend.ApiConfig.from_dict(payload)


def test_hybrid_launcher_checks_config_without_starting_engine(tmp_path, capsys):
    from qfn_decision import hybrid
    cfg = config(tmp_path)
    path = tmp_path / 'gateway.json'
    path.write_text(json.dumps(cfg.as_dict()))
    assert hybrid.main(['--config', str(path), '--check']) == 0
    assert 'trained_adapter_loaded' in capsys.readouterr().out


@pytest.mark.parametrize('mutation', ['arm', 'h1', 'template', 'thinking'])
def test_launcher_refuses_unsupported_runtime_settings(tmp_path, mutation):
    from qfn_decision import hybrid
    cfg = config(tmp_path)
    if mutation == 'arm': cfg.default_arm = 'H1'
    elif mutation == 'h1': cfg.h1['artifact_path'] = 'head.npz'
    elif mutation == 'template': cfg.template_id = 'autotrust_raw_bare-v1'
    elif mutation == 'thinking': cfg.chat_template_kwargs = {'enable_thinking': True}
    with pytest.raises(backend.ConfigError):
        hybrid.make_server(cfg)


def test_hybrid_boolean_choice_and_systemone_on_loopback(tmp_path, monkeypatch):
    """Use a clearly synthetic engine; never a model quality or hardware test."""
    original = backend.FixtureEngineServer._record
    def pinned_record(self, *args, **kwargs):
        return dataclasses.replace(original(self, *args, **kwargs), verbalizer_ids=tuple(ROWS))
    monkeypatch.setattr(backend.FixtureEngineServer, '_record', pinned_record)
    with backend.FixtureEngineServer(capture_dir=tmp_path / 'captures').start() as engine:
        cfg = config(tmp_path, ENGINE_URL=engine.base_url, ENGINE_MODEL=engine.model)
        with api.GatewayServer(cfg, backend_instance=HybridBackend(cfg), port=0, log_sink=[].append) as gateway:
            def post(route, body):
                conn = http.client.HTTPConnection('127.0.0.1', gateway.port, timeout=5)
                try:
                    conn.request('POST', route, json.dumps(body), {'Content-Type': 'application/json'})
                    response = conn.getresponse()
                    return response.status, json.loads(response.read())
                finally: conn.close()
            status, result = post('/v1/decide', {'kind': 'noul', 'state': 'Synthetic state', 'question': 'Synthetic question', 'thinking': 'off'})
            assert status == 200, result
            assert result['kind'] == 'noul' and set(result['probabilities']) == {'false', 'true'}
            assert result['usage']['typed_output_tokens'] == 0
            assert result['usage']['internal_generated_tokens'] == 1
            assert result['capture']['wire_kind'] == 'choice'
            assert all(value == 0 for value in result['slot_probabilities'][2:])
            status, result = post('/v1/systemone', {'state': 'Synthetic state', 'questions': {'check': {'type': 'noul', 'instructions': 'Synthetic question'}}})
            assert status == 200, result
            assert 0 <= result['answers']['check']['noul'] <= 1
            for arm in ['H1', 'L1O']:
                status, _ = post('/v1/decide', {'kind': 'noul', 'state': 'S', 'question': 'Q', 'arm': arm})
                assert status == 422
            status, _ = post('/v1/decide', {'kind': 'score', 'state': 'S', 'question': 'Q', 'score_rubrics': ['level'] * 6})
            assert status == 422
