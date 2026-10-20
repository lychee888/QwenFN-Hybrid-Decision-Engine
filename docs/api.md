# API

The generated frontend binds `127.0.0.1:8888`. Only the two typed paths below and their trailing-slash variants are handled by the CPU gateway. Other frontend paths go directly to the engine. The gateway's own readiness endpoint is `http://127.0.0.1:8890/health` by default; frontend `/health` is the engine's endpoint.

## Typed decision

`POST /v1/decide`, JSON body:

```json
{
  "kind": "choice",
  "state": "The service is responding normally after a restart.",
  "question": "Which action fits the current state?",
  "options": [
    {"key": "observe", "description": "Continue observation"},
    {"key": "restart", "description": "Restart the service again"}
  ],
  "thinking": "off",
  "arm": "B0"
}
```

`kind` is `choice` or `noul` in the supported launcher. Choice accepts 2 to 256 ordered, unique
caller keys. Two internal prompt surfaces back this one API:

- 2 to 16 options use the 24-slot lettered path. The model sees lettered options A through P;
  the response maps probabilities back to the original keys. Ties use the first active option.
  The frozen template is `qwen_chat_letters-v1`, version `letters-v1`, with `enable_thinking=false`.
  The 24-slot internal representation (`noul` 0:2, `score` 2:8, `choice` 8:24) is a fixed wire
  layout for small choices and booleans, not the maximum choice count and not a trained head.
- 17 to 256 options are served by the wide-label lane: separate frozen prompt surface
  `qwen_chat_wide-v1`, the 256-entry label catalog, request-bound terminal raw-logit capture and
  a masked softmax over the `n` active labels. The response schema is `qfn.decide.wide/1`.
  See [wide-label decisions](api-wide-labels.md) for the wide contract, pins and capture records.

For `noul`, omit `options`. The gateway sends A=false and B=true as a two-option choice, with descriptions equal to the keys. It then projects wire slots 8 and 9 onto public slots 0 and 1. The response contains `p_true` and a `false`/`true` probability map. The capture/template metadata describes the actual choice wire request, not an invented native boolean engine head.

The successful payload includes:

- `kind`, `probabilities`, selection and primitive-specific fields. Small-choice and boolean responses also include `slot_probabilities`; wide responses do not.
- `arm`, with training/calibration status; the launcher only serves unchanged-base B0.
- `template`, `tokenizer`, `engine` and `capture` provenance, including request IDs and token hashes.
- `usage.typed_output_tokens=0`, the actual internal generated-token count and a false completion-text-returned flag.
- Timing and cache evidence. Unknown cache-hit counts remain unknown, rather than being inferred from prefill progress.

The distribution is a masked softmax of selected raw engine logits. Inactive slots are exactly zero. This is not a guaranteed calibrated estimate of correctness. A request schedules one internal completion token, which may be EOS; its text is discarded. Zero returned text tokens does not mean the engine performed no sampling. Ordinary chat, tool calls and streaming share the same engine and are not routed through the typed gateway.

Native score is unsupported by this launcher. Lower-level protocol modules still contain legacy score and optional head interfaces for compatibility tests, but those interfaces are not enabled serving features. Requests for H1 or L1O fail instead of silently falling back to B0. No learned artifact is shipped.

## Jev-shaped endpoint

`POST /v1/systemone`, JSON body:

```json
{
  "state": "The service is responding normally after a restart.",
  "questions": {
    "stable": {"type": "noul", "instructions": "Is the service currently responding normally?"},
    "action": {
      "type": "choice",
      "instructions": "Which action fits the current state?",
      "criteria": {
        "observe": "Continue observation",
        "restart": "Restart the service again"
      }
    }
  }
}
```

The response has an `answers` map keyed by question name. A boolean answer has `type` and scalar `noul` fields. A choice answer has `type`, `choice`, keyed `probabilities` (the Jev choice confidence formula `(p_max - 1/n) / (1 - 1/n)`, unchanged) and `confidence`. The compatibility choice confidence follows the documented TypeSafe primitive formula; it is not a calibration guarantee. Boolean answers do not gain an invented confidence field. A choice question may carry up to 256 `criteria` keys per question; 17..256-option questions are executed on the wide-label lane and add an explicit wide provenance block to the answer (see [wide-label decisions](api-wide-labels.md)).

An optional request `model` field is accepted for request-shape compatibility; it does not switch the configured engine or claim that this service is Jev. Structured state/instructions use deterministic serialization. Optional boolean criteria are included in the question text. Choice criteria preserve insertion order. Unsupported fields or limits are refused.

The launcher supports choice questions with 2 to 256 criteria when the wide lane is enabled, plus booleans; native score is refused. This is not full Jev feature equivalence. Multiple questions share the bounded concurrency limit. A failed question returns an error naming it without partial answers.

## Limits and errors

Generated defaults cap the body at 262,144 bytes, state at 32,768 characters, question at 4,096 characters, option description at 2,048 characters, key at 64 characters and total typed content at 96,000 characters. The default typed concurrency limit is 8. Oversized inputs are rejected, not truncated.

- `400`: malformed JSON or invalid framing.
- `422`: schema/primitive limits, unavailable arm or unsupported readout.
- `429`: typed concurrency is full.
- `502`: engine transport or capture-integrity failure.
- `503`: unavailable backend or explicit per-request `capture_refused`.
- `504`: the bounded request/capture deadline elapsed.

Capture decoding checks the caller/engine request binding, prompt-token digest, terminal position, file safety and row identity. It refuses ambiguity, stale data and corrupt captures. Probabilities are not returned on failure.

## Ordinary engine routes

Use ordinary `/v1/chat/completions`, `/v1/completions`, `/v1/models`, tokenization and other engine endpoints through the frontend. Tool-call and SSE semantics belong to the separately launched matching engine. The router does not send these requests through the typed gateway's stricter body or concurrency contract.

The frontend forwards authentication headers; the CPU gateway itself is not an authentication boundary. Keep both services on loopback and supply an independently reviewed authentication/TLS layer before external use.

## Wide-label lane (17 to 256 options, built-in)

Options 17 to 256 are part of the same release runtime, on the same `POST /v1/decide` and
`/v1/systemone` paths. The lane is disabled unless both the gateway and the engine capture config
declare an enabled `wide_choice` block, and generation via `configure` requires an explicit
`WIDE_CHOICE=1` plus an absolute catalog path. It uses its own request/result/record types, its own
frozen prompt surface (one or two letters), the `qfn.decide.wide/1` response schema and a 256-label
capture extension. See
[wide-label decisions](api-wide-labels.md) for the request/response shapes, the catalog pins, the
capture contract and the integration steps.
