# Wide-label decisions (17 to 256 options)

Status: **in production since 20 October 2026.** The maintainer-host deployment accepted typed
wide calls at 17, 26, 27, 32, 64, 128 and 256 options with per-call NPZ audits, refused a
257-option request, and kept legacy chat, tools, SSE and MTP behaviour unchanged. An 84-call
UI-selection panel passed 84/84 as a separate smoke; it is not part of the accuracy or
calibration evidence. See [benchmarks](benchmarks.md) for the accuracy measurements and their
limits. No learned artifact is involved: the wide lane reads raw option logits of unchanged
weights only.

## Readout structure

One public choice API selects its internal prompt surface by option count. Small choices and
booleans use a fixed 24-slot representation (`noul` 0:2, `score` 2:8, `choice` 8:24 with
labels `A..P`). Wide choices use the versioned 256-label catalog. Neither path loads a trained
head or adapter:

| | legacy `<=16` | wide `17..256` |
|---|---|---|
| request type | `protocol.DecisionRequest` | `wide.WideChoiceRequest` |
| prompt surface | `qwen_chat_letters-v1` (unchanged, byte for byte) | `qwen_chat_wide-v1` (frozen, one or two letters) |
| labels | `A..P` | the 256-label catalog (`A..Z`, then two-letter labels up to `JU`) |
| readout | masked softmax over 24 slots | masked softmax over the first `n` raw label logits |
| response schema | `qfn.decide/1` | `qfn.decide.wide/1` |
| trained arms | B0 | **B0 only** (no head, no adapter, no score) |

The gateway dispatches requests by option count. A choice with at most 16 options uses
`qfn.decide/1`; a choice with 17 to 256 options uses `qfn.decide.wide/1`.

## Request (`POST /v1/decide`)

```json
{
  "kind": "choice",
  "state": "...",
  "question": "which option?",
  "options": [{"key": "custom-key", "description": "one line of text"}, "... 17 to 256 items"],
  "thinking": "off",
  "arm": "B0",
  "seed": 0,
  "deadline_ms": 30000
}
```

Rules (enforced before any tokenization or engine call):

* `kind` must be `choice`; `score` and `noul` stay legacy primitives.
* 17 to 256 options, caller keys are arbitrary unique strings (Unicode allowed), descriptions are
  single-line and bounded by the same protocol limits as the legacy path.
* `257` options is refused with `422 choice_too_many_options`. Unknown fields, unknown option
  fields, duplicate keys, empty/oversized state or question, `thinking != "off"` and any arm other
  than `B0` are refused.
* No caller-supplied request id is accepted; the gateway mints `qfn-wide-<n>-<uuid>` itself.
* `deadline_ms` is an **absolute budget for the whole acquisition**: tokenization, the completion
  request and its response body, the capture poll and the readout all draw on the same monotonic
  deadline. A request whose budget expires gets `504 deadline_exceeded` with no probabilities, and
  no completion is sent once tokenization has already exhausted the budget. A response (or a capture
  record) that only arrives at/after the deadline is never accepted. A configured
  `engine.socket_timeout_s` that is shorter than the budget keeps its existing meaning: a
  transport error (`502 backend_error`), not a deadline expiry.
* The ordinary caller limits (body bytes, total typed characters, state and question lengths) still
  apply unchanged. With 256 options, maximal descriptions can exceed the body limit; keep
  descriptions short when the option list is long.

## Response (`qfn.decide.wide/1`)

```json
{
  "schema": "qfn.decide.wide/1",
  "kind": "choice",
  "adaptation": {
    "mode": "wide-labels", "n_options": 26, "native_choice_slots": 16,
    "slot_vector_used": false, "calibrated_confidence": false, "trained_arms": false
  },
  "probabilities": {"key": 0.0, "...": 1.0},
  "selected": "key",
  "selected_label": "Q",
  "raw_max_probability": 0.71,
  "jev_confidence": null,
  "confidence_status": "absent_unverified_for_wide_labels",
  "catalog": {"schema": "qfn.wide-label-catalog/1", "digest": "...", "n_labels": 256,
              "active_labels": 26, "labels": ["A", "..."]},
  "template": {"template_id": "qwen_chat_wide-v1", "version": "wide-v1",
               "content_sha256": "...", "messages_sha256": "...", "prompt_token_sha256": "..."},
  "capture": {"wide": {"catalog_digest": "...", "raw_logits_sha256": "...", "...": "..."}, "...": "..."},
  "engine": {"engine_request_id": "cmpl-qfn-wide-26-<uuid>-0", "...": "..."},
  "usage": {"typed_output_tokens": 0, "internal_generated_tokens": 1, "...": "..."}
}
```

There is **no** `slot_probabilities` key: `n` probabilities are never presented as a 24-slot
vector. `probabilities` carries exactly the caller keys, in request order.

One internal completion token is scheduled and its text is discarded; `typed_output_tokens` is
always 0. Zero returned text tokens does not mean the engine performed no sampling. The returned
probabilities are a masked softmax of raw captured logits; they are not calibrated confidence.

## `/v1/systemone`

A choice question with 17..256 `criteria` keys is translated to a wide request. The answer keeps
Jev's shape (`choice`, `confidence`, `probabilities`) and adds an explicit provenance block:

```json
{"answers": {"pick": {
  "type": "choice", "choice": "k12", "confidence": 0.34,
  "probabilities": {"k0": 0.01, "...": 0.02},
  "wide": {"adaptation": "wide-labels", "n_options": 26, "native_choice_slots": 16,
           "calibrated_confidence": false, "catalog_digest": "...", "template_digest": "...",
           "capture_uuid": "...", "prompt_token_sha256": "..."}}}}
```

`confidence` is Jev's documented choice formula `(p_max - 1/n) / (1 - 1/n)` over the `n` wide
options. The formula is unchanged; no probability calibration is claimed for the wide adaptation.
Mixed question maps
(wide + small choice + noul) work on one request and share the gateway's inflight permit.

`/v1/systemone` has no caller deadline parameter. A wide question executed there uses the gateway's
configured `timeouts.capture_deadline_s` as its acquisition budget (the same absolute,
phase-spanning budget described above). No new public field is defined for this stage.

## Configuration

Gateway (`decision_api.json`), opt-in and disabled when absent:

```json
{
  "wide_choice": {
    "enabled": true,
    "catalog_path": "/abs/path/configs/wide-label-catalog.json",
    "catalog_file_sha256": "71a861ff84956974dd3294e20d09f86a49e323d4eff03299c30184e5517fb9bf",
    "catalog_digest": "d3aba3f02f6af23c32303e4b6617cde13da8e3c82ca7876d2c1b635450aa71d1",
    "tokenizer_revision": "1b3c3cedc14fb3bee4294dea5e3515af74b8268a",
    "template_digest": "ed5cfb299d0d4ad9789a7b6b267108027af9fd07e474fe9164b22fc41ad58e52",
    "max_options": 256
  }
}
```

`configure.build_configs` does not enable the lane by default. To enable it, set
`WIDE_CHOICE=1` and `WIDE_CATALOG_PATH="$PWD/configs/wide-label-catalog.json"` (the path must be
absolute) before running the generator; the resulting blocks use the pinned constants above.

Engine capture config: the same block under the top-level `wide_choice` key, **without**
`max_options`. The generator (`configure.build_configs`) writes the two blocks separately for exactly
this reason: `max_options` bounds what the gateway will accept, while the helper always captures all
256 labels and refuses unknown keys. Copying the gateway block verbatim makes the generated capture
config unloadable by its real consumer.

Both sides validate the catalog length, labels, ID order/uniqueness, source tokenizer revision,
file-byte hash and content digest **before** serving or capturing. The configured `catalog_path`
must point to a file the gateway and every engine worker can read at that exact absolute path, with
matching bytes. The engine can only echo the
template digest it was configured with; the gateway is what pins the frozen prompt text, so a
mismatched engine configuration produces records the gateway refuses. The engine capture
configuration is read once at startup, so enabling the wide lane needs a coordinated engine
restart; the gateway reads its own config at gateway startup.

`tokenizer_revision` here is an **operator pin**: it records which tokenizer the label catalog was
built with and is checked against the catalog, but the gateway does not independently fetch or verify
the identity of the tokenizer running on the engine host. Likewise the configured model alias is a
configuration value, not served identity. Verify the live model/tokenizer on the host before enabling
the wide lane; configuration validation alone cannot detect a host-side mismatch.

### Catalog digest vs file hash

`catalog_file_sha256` is the byte identity of the shipped JSON file. `catalog_digest` is the
semantic identity:

```
sha256(json.dumps({"schema": ..., "tokenizer_revision": ..., "labels": [...], "token_ids": [...]},
                  sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
```

A reformatted or key-reordered file keeps the content digest; a changed label, order or id never
does. The engine helper derives the same digest independently and a CPU test asserts they agree.

## Capture contract

A wide request mints `request_id = qfn-wide-<n>-<uuid>`, which the engine sees as
`cmpl-qfn-wide-<n>-<uuid>-<index>`. On the proven terminal row only, the helper index-selects all
256 catalog rows from the **same** validated raw `logits[j]` row and writes them as real NPZ
arrays:

```
wide_ids           int64[256]    exactly the catalog ids, in catalog order
wide_logits_fp32   float32[256]  finite raw label logits from that one row
meta_json          scalar wide_* block (schema, catalog/template digests, request uuid,
                   prompt hash/len, label count, option count)
```

The arrays are never serialised into the scalar metadata. The legacy reader refuses a record that
carries the extension (even with `tolerate_unknown_meta_fields`), and the wide reader refuses a
record that lacks it: there is no fallback from a missing wide capture to a plausible 16-option
result. The reader also re-derives and re-checks the catalog content, the ordered ids, the request
uuid, the prompt hash, the terminal position, the branch rank, `token_hash_from_engine_state`, the
capture age and the vocab bound.

`/v1/decide` captures all 256 labels and slices the first `n` **in the reader** before the softmax,
so unused catalog labels can never carry probability mass.

## Operational notes for integration

1. Copy `configs/wide-label-catalog.json` next to the deployed configs (or anywhere absolute) and
   apply the `wide_choice` block to **both** nodes' gateway config and engine capture config.
2. The engine capture configuration is read once at startup: the wide lane needs a **coordinated
   engine restart**. The gateway config is read at its own startup.
3. Without the block (or with `enabled: false`), the deployment accepts only small choices and
   booleans. A choice above 16 options is refused; the gateway never silently truncates it.
4. `record_max_bytes` applies to the extended record. Measured overhead on one synthetic n=32
   fixture (receipt: `correction-r2/worker/receipts/overhead-measurement.txt`): the extension arrays
   are 3072 raw bytes
   (256 x int64 ids + 256 x float32 logits) plus a scalar `wide_*` block; the stored NPZ grew by
   4202 bytes and the published NPZ+JSON pair by 4868 bytes. The independent review measured 4201 and
   4910 bytes on its own fixture, so container and JSON overhead vary; treat these as example
   measurements, not a compression or size promise. Budget roughly 4 to 5 KB per wide record on top
   of an unchanged legacy record.

## Known limits

* Wide requests are choice-only. There is no native score, no H1 head, no adapter and no trained
  arm for wide labels.
* `tokenizer_revision` and the gateway model alias are configuration pins, not independently fetched
  engine identity. Verify the live host's model/tokenizer before enabling
  the lane; configuration validation alone cannot detect a host-side mismatch.
* If the 24 legacy verbalizer ids cannot be resolved in the engine config, the helper writes no
  wide extension and the request fails closed (502) rather than publishing a partial record.
* The `logprobs` API path is not used at all: earlier probes showed it cannot return all
  allowed labels. Wide support depends only on selected-logit capture.
