#!/usr/bin/env bash
set -euo pipefail
QFN_URL="${QFN_URL:-http://127.0.0.1:8888}"
ENGINE_MODEL="${ENGINE_MODEL:-qwen3.8-flash-next-uncensored}"

curl --fail-with-body -sS "$QFN_URL/v1/decide" \
  -H 'Content-Type: application/json' \
  --data '{"kind":"noul","state":"The service is responding normally.","question":"Is the service responding normally?","thinking":"off","arm":"B0"}'

curl --fail-with-body -sS "$QFN_URL/v1/systemone" \
  -H 'Content-Type: application/json' \
  --data '{"state":"The service is responding normally.","questions":{"stable":{"type":"noul","instructions":"Is the service responding normally?"}}}'

# Ordinary generation goes directly to the matching engine through the router.
curl --fail-with-body -N -sS "$QFN_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data "{\"model\":\"$ENGINE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"stream\":true,\"max_tokens\":32}"
