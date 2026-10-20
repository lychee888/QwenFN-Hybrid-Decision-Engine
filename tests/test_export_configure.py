"""Export configuration contract. All paths are test-owned and local."""
import json
from pathlib import Path
import pytest
from qfn_decision import backend, templates


def test_configure_builds_chat_b0_and_capture_config(tmp_path):
    from qfn_decision import configure
    caps = tmp_path / 'captures'
    gateway, capture = configure.build_configs({
        'CAPTURE_DIR': str(caps), 'MODEL_DIR': str(tmp_path / 'model'),
        'ENGINE_URL': 'http://127.0.0.1:8897',
    })
    cfg = backend.ApiConfig.from_dict(gateway)
    assert cfg.default_arm == 'B0'
    assert cfg.h1['artifact_path'] is None and cfg.decision_adapter is None
    assert cfg.template_id == templates.TEMPLATE_QWEN_CHAT_LETTERS
    assert cfg.template_version == 'letters-v1'
    assert cfg.chat_template_kwargs == {'enable_thinking': False}
    assert cfg.capture_dir == str(caps)
    assert cfg.capture_dir_allowlist == [str(caps)]
    assert capture['capture_dir'] == str(caps)
    assert capture['production_storage']['retention_root'] == str(caps)
    assert capture['production_storage']['ttl_seconds'] == 86400
    assert capture['verbalizers']['ids'] == [3721, 1802, 15, 16, 17, 18, 19, 20, *range(32, 48)]


@pytest.mark.parametrize('env', [
    {}, {'CAPTURE_DIR': 'relative'}, {'CAPTURE_DIR': '/captures', 'ENGINE_URL': 'http://example.com'},
    {'CAPTURE_DIR': '/captures', 'ENGINE_URL': 'http://user:secret@127.0.0.1:8897'},
    {'CAPTURE_DIR': '/captures', 'GATEWAY_PORT': '0'},
])
def test_configure_refuses_ambiguous_or_nonlocal_settings(env, tmp_path):
    from qfn_decision import configure
    env = dict(env)
    if env.get('CAPTURE_DIR') == '/captures':
        env['CAPTURE_DIR'] = str(tmp_path / 'captures')
    with pytest.raises(ValueError):
        configure.build_configs(env)


def test_nginx_router_uses_exact_typed_routes_only():
    from qfn_decision import configure
    text = configure.render_router({})
    assert 'listen 127.0.0.1:8888;' in text
    assert 'location = /v1/decide {' in text
    assert 'location = /v1/systemone {' in text
    assert 'location = /v1/decide/ {' in text
    assert 'location = /v1/systemone/ {' in text
    assert 'location / {' in text and 'proxy_pass http://127.0.0.1:8897;' in text
    assert 'proxy_buffering off;' in text and 'proxy_request_buffering off;' in text
    assert 'Authorization $http_authorization' in text and 'X-Request-Id $http_x_request_id' in text
    assert 'location ~' not in text
