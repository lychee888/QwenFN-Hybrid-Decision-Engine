"""Generate loopback gateway, capture and nginx configs without starting an engine."""
from __future__ import annotations

import argparse
import importlib.resources
import json
import os
from pathlib import Path
import urllib.parse
from typing import Mapping


def _port(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key, str(default))
    if not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise ValueError(f'{key} must be an integer in 1..65535')
    return int(value)


def _engine_url(env: Mapping[str, str]) -> str:
    value = env.get('ENGINE_URL', 'http://127.0.0.1:8897')
    url = urllib.parse.urlsplit(value)
    if (url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password
            or url.path not in ('', '/') or url.query or url.fragment):
        raise ValueError('ENGINE_URL must be a credential-free http://127.0.0.1:port URL')
    port = url.port or 80
    if not 1 <= port <= 65535:
        raise ValueError('ENGINE_URL port is out of range')
    return f'http://127.0.0.1:{port}'


def build_configs(env: Mapping[str, str]) -> tuple[dict, dict]:
    """Return explicit B0 configs. CAPTURE_DIR is the engine-visible shared path."""
    path = env.get('CAPTURE_DIR')
    if not path or not Path(path).is_absolute():
        raise ValueError('CAPTURE_DIR must be an absolute shared directory path')
    captures = str(Path(path))
    gateway = json.loads(importlib.resources.files('qfn_decision').joinpath('default_api.json').read_text())
    gateway['service_version'] = 'qfn-decision/0.1.0'
    gateway['stage'] = 'Release 0.1.0; B0 readout live-validated on a two-node GB10 deployment, choice 2..256 options'
    gateway['host'] = '127.0.0.1'
    gateway['port'] = _port(env, 'GATEWAY_PORT', 8890)
    gateway['engine']['base_url'] = _engine_url(env)
    gateway['engine']['model'] = env.get('ENGINE_MODEL', 'qwen3.8-flash-next-uncensored')
    gateway['engine']['note'] = 'One separately launched engine; no lifecycle management by the gateway.'
    gateway['template'] = {
        'template_id': 'qwen_chat_letters-v1', 'version': 'letters-v1',
        'require_version_match': True, 'chat_template_kwargs': {'enable_thinking': False},
    }
    gateway['tokenizer']['revision'] = env.get('TOKENIZER_REVISION') or None
    gateway['head'] = {'default_arm': 'B0', 'h1': {'artifact_path': None, 'verify_checksum': True}}
    gateway.pop('decision_adapter', None)
    gateway['capture']['dir'] = env.get('GATEWAY_CAPTURE_DIR', captures)
    if not Path(gateway['capture']['dir']).is_absolute():
        raise ValueError('GATEWAY_CAPTURE_DIR must be absolute')
    gateway['capture']['dir_allowlist'] = [gateway['capture']['dir']]
    gateway['capture']['note'] = 'Same committed capture pairs as the engine, readable locally by the gateway.'
    gateway['fixture']['note'] = 'Synthetic fixture provider is for CPU tests only.'
    capture = {
        'schema': 'qfn.capture-config/1', 'capture_dir': captures,
        'marker_prefix': 'cmpl-qfn-', 'export_dir': str(Path(captures) / 'artifacts'),
        'limits': {'max_bytes': 4294967296, 'max_records': 4096},
        'capture': {'capture_fp32_ref': True, 'capture_full_logits_row': False,
                    'capture_ranks': 'rank0', 'debug_hidden_hash_all_ranks': False,
                    'token_hash_scope': 'prompt', 'tp_export_rows': True},
        'verbalizers': {'names': ['false', 'true', *map(str, range(6)), *'ABCDEFGHIJKLMNOP'],
                        'ids': [3721, 1802, 15, 16, 17, 18, 19, 20, *range(32, 48)]},
        'production_storage': {
            'retention_root': captures, 'max_retained_records': 4096,
            'max_retained_bytes': 34359738368, 'min_free_disk_bytes': 10737418240,
            'max_record_bytes': 262144, 'grace_seconds': 3600, 'ttl_seconds': 86400,
            'snapshot_ttl_seconds': 5.0, 'pending_ttl_seconds': 300,
        },
    }
    # Opt-in wide-label lane. Disabled unless WIDE_CHOICE is explicitly truthy; enabling it pins the
    # bundled validated catalog (byte hash + content digest + source tokenizer revision) and the
    # frozen prompt digest, and validates them here so a drift fails at configure time.
    enabled = str(env.get('WIDE_CHOICE', '')).strip().lower() in ('1', 'true', 'yes', 'on')
    if enabled:
        from . import wide as wide_mod

        catalog_path = env.get('WIDE_CATALOG_PATH')
        if not catalog_path or not Path(catalog_path).is_absolute():
            raise ValueError('WIDE_CHOICE requires WIDE_CATALOG_PATH as an absolute path')
        wide_mod.WideCatalog.from_file(
            catalog_path,
            tokenizer_revision=wide_mod.PINNED_CATALOG_TOKENIZER_REVISION,
            expected_digest=wide_mod.PINNED_CATALOG_DIGEST,
            expected_file_sha256=wide_mod.PINNED_CATALOG_FILE_SHA256,
        )
        if gateway['tokenizer']['revision'] not in (None, wide_mod.PINNED_CATALOG_TOKENIZER_REVISION):
            raise ValueError(
                'TOKENIZER_REVISION disagrees with the pinned wide catalog tokenizer revision'
            )
        wide_block = {
            'enabled': True,
            'catalog_path': str(Path(catalog_path)),
            'catalog_file_sha256': wide_mod.PINNED_CATALOG_FILE_SHA256,
            'catalog_digest': wide_mod.PINNED_CATALOG_DIGEST,
            'tokenizer_revision': wide_mod.PINNED_CATALOG_TOKENIZER_REVISION,
            'template_digest': wide_mod.WIDE_TEMPLATE_CONTENT_SHA256,
            'max_options': wide_mod.MAX_WIDE_CHOICE_OPTIONS,
        }
        gateway['wide_choice'] = wide_block
        gateway['tokenizer']['revision'] = wide_mod.PINNED_CATALOG_TOKENIZER_REVISION
        # The engine block carries the same catalog/template pins but NOT the gateway-only option
        # cap: max_options bounds what the gateway will accept, while the helper always captures all
        # 256 labels and refuses unknown keys, so copying it verbatim would make the generated
        # capture config unloadable by its real consumer.
        capture['wide_choice'] = {
            key: value for key, value in wide_block.items() if key != 'max_options'
        }
    else:
        gateway.pop('wide_choice', None)
    return gateway, capture


def render_router(env: Mapping[str, str]) -> str:
    """Exact typed routes, all other requests and streaming bytes go to the engine."""
    front = _port(env, 'FRONTEND_PORT', 8888)
    gateway = _port(env, 'GATEWAY_PORT', 8890)
    engine = _engine_url(env)
    locations = []
    for route in ('/v1/decide', '/v1/systemone', '/v1/decide/', '/v1/systemone/'):
        locations.append(f'    location = {route} {{ proxy_pass http://127.0.0.1:{gateway}; }}')
    return '\n'.join([
        'worker_processes 1;', 'pid nginx.pid;', 'error_log stderr warn;',
        'events { worker_connections 1024; }', 'http {', '  access_log off;',
        '  server_tokens off;', '  client_max_body_size 0;', '  client_body_timeout 3600s;',
        '  send_timeout 3600s;',
        "  map $http_upgrade $connection_upgrade { default upgrade; '' close; }",
        '  server {', f'    listen 127.0.0.1:{front};', '    proxy_http_version 1.1;',
        '    proxy_set_header Host $http_host;',
        '    proxy_set_header Authorization $http_authorization;',
        '    proxy_set_header X-Api-Key $http_x_api_key;',
        '    proxy_set_header X-Request-Id $http_x_request_id;',
        '    proxy_set_header Upgrade $http_upgrade;',
        '    proxy_set_header Connection $connection_upgrade;',
        '    proxy_buffering off;', '    proxy_request_buffering off;', '    proxy_cache off;',
        '    proxy_intercept_errors off;', '    proxy_read_timeout 3600s;',
        '    proxy_send_timeout 3600s;', '    proxy_connect_timeout 10s;',
        *locations, f'    location / {{ proxy_pass {engine}; }}', '  }', '}', '',
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New local configuration directory')
    args = parser.parse_args(argv)
    try:
        gateway, capture = build_configs(os.environ)
        router = render_router(os.environ)
        files = {'gateway.json': json.dumps(gateway, indent=2) + '\n',
                 'capture.json': json.dumps(capture, indent=2) + '\n', 'nginx.conf': router}
        if any((args.out / name).exists() for name in files):
            raise ValueError('Refusing to overwrite an existing configuration')
        args.out.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            with (args.out / name).open('x', encoding='utf-8', newline='\n') as output:
                output.write(data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Wrote gateway.json, capture.json and nginx.conf to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
