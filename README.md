# QwenFN Hybrid Decision Engine

**One model, one engine, two interfaces: System 1 + System 2.** Conventional LLM and decision model (jev-style) output on one vLLM instance.

**Model package:** [lychee888/Qwen3.8-Flash-Next-Hybrid-Decision](https://huggingface.co/lychee888/Qwen3.8-Flash-Next-Hybrid-Decision) carries the matching complete weight tree, model card and decision configuration.

A typed decision gateway for Qwen3.8-Flash-Next. You send a question with a fixed set of options (2 to 256); the gateway reads the option-token logits at the end of the prompt through a request-bound vLLM capture hook and returns a probability for each option. Ordinary chat, tool calls and streaming keep running on the same engine.

The default path (B0) uses the unchanged base model. This repository contains no learned head or LoRA weights, and the launcher never loads one.

## Lineage and acknowledgments

[AutoTrust AI's JEV-27B](https://huggingface.co/autotrust/JEV-27B) was the architectural reference for this project: one frozen backbone serving generation and typed decisions, a 24-slot decision layout, and the `false`/`true`, score-level and option-letter verbalizers. Our reference protocol and raw-template control use their pinned revision `ff9b67b9bcb2dc5a8499d540e0c4cd10b254f3a2`. The production prompt is instead the chat-letter surface from [blockbrain-ai/swanone-recipe](https://github.com/blockbrain-ai/swanone-recipe). The typed primitives and System One API format originate with [TypeSafe AI](https://docs.typesafe.ai/).

AutoTrust trained a decision LoRA and head; this package uses unchanged Flash-Next weights and their raw option logits. It includes no AutoTrust model or adapter weights. See [NOTICE](NOTICE) for attribution and [the performance comparison](docs/benchmarks.md#comparison-with-autotrustjev-27b) for the published-result scope.

## How choices work

One public choice API, two internal prompt surfaces. Options 2 to 16 use the frozen `qwen_chat_letters-v1` surface and a fixed 24-slot internal representation (`noul` 0:2, `score` 2:8, `choice` 8:24); that layout is a wire format for small choices and booleans, not the maximum choice count and not a trained head. Options 17 to 256 use the wide-label lane: the frozen `qwen_chat_wide-v1` surface, a 256-entry label catalog, request-bound terminal raw-logit capture and a masked softmax over the active labels. Both lanes run unchanged weights only. See the [API](docs/api.md) and [wide-label](docs/api-wide-labels.md) docs for the contracts.

A typed call schedules one internal completion token and discards it, so it returns zero text tokens while the engine still runs one sampling step. Probabilities come straight from the model's option logits and are not guaranteed to be calibrated. Native score questions and the H1 and L1O arms are refused by the launcher. See [API](docs/api.md) for the full request rules and limits.

## Status

This runtime serves our production deployment on two DGX Spark (GB10) nodes. On 20 October 2026 the maintainer-host deployment accepted typed 256-option wide calls alongside the 2 to 16 option path, with per-call NPZ audits and legacy chat, tools, SSE and MTP behaviour unchanged; that acceptance history also includes a real rollback to the previous engine. The evaluation results in [benchmarks](docs/benchmarks.md) come from that deployment on 20 October 2026.

Those launches were on our own hosts; nobody has installed it from scratch on a fresh machine yet. The tested engine is a model-specific vLLM build, `0.1.dev20073+g8e685d198`, and its full source commit and build recipe are not recovered (the suffix is not a commit in `vllm-project/vllm`). It also needs the inherited PLE and QSA overlays, the runner call-site patch and the capture helper in `engine/`. Do not expect current vLLM main to run this path. [Installation](docs/installation.md) and [runtime-dependencies.json](runtime-dependencies.json) list the pins and what is still unknown.

## Interfaces

- `POST /v1/decide`: one question per call. A `choice` takes 2 to 256 keyed options; `noul` is a yes/no question. Returns `probabilities`, `selected` and `raw_max_probability`, plus audit fields (engine request id, prompt token hash, template hash, timing).
- `POST /v1/systemone`: the same readout in TypeSafe Jev's request and response shape. Several named questions per call, each choice question carrying up to 256 criteria, and a `confidence` field computed with Jev's published formula.
- Every other path goes straight to the engine, so `/v1/chat/completions`, `/v1/completions`, `/v1/models` and SSE streaming behave like the underlying vLLM server.

## Results

![Accuracy advantage over published references](docs/charts/head-to-head.png)

Summary of the 20 October 2026 measurements on the production deployment (details, caveats, both tie rules and the wide scaling panel in [Benchmarks](docs/benchmarks.md)):

- Legacy lanes (option counts up to 16): the unchanged base scored 198, 198 and 199 of 213 on the public JevBench strict legacy tasks in three repeats, against 186 for Jev 1.13.0's published outputs (paired exact McNemar p between 0.0023 and 0.0075). On all 1,599 ChaosNLI test rows under the frozen benchmark tie rule, plurality accuracy was 64.29% against Jev's published 61.35%, but our probabilities were further from the human vote spread (mean total variation distance 0.362 against 0.332). 16 calls failed and count as wrong; the benchmarks doc describes them with no causal attribution.
- Wide scaling (option sizes 16 to 256, the same 100 texts at each size): primary accuracy fell from 73.67% at 16 options to 57.00% at 256 options. The paired 256-versus-16 difference is -16.67 points, clustered 95% CI [-26.00, -7.33]. Option count and prompt/presentation change together, so this decrease is not attributed to a single architectural mechanism. The 100 texts are reused, so this is paired scaling evidence, not 3,000 independent examples.

All items are public, and the legacy tasks were used during development, so this is not fresh holdout evidence. Every probability is an uncalibrated option-logit readout.

## CPU setup

Python 3.12 or newer and NumPy are required. From this checkout:

```sh
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -e '.[dev]'
python -m pytest
```

The current source suite and a wheel installed into a fresh isolated environment each passed 690 tests with 5 skips (torch absent and Windows symlink permissions). Imports were checked against the installed site-packages, and the installed package generated 256-option configuration accepted by both actual consumers. These checks cover the gateway, protocol, capture records and retention. Only the production launch above exercises the GPU path.

## Launch

The launcher never starts or stops an engine. Prepare the patched engine and a capture directory shared with it, then:

```sh
export MODEL_DIR=/srv/models/qwen-flash-next
export CAPTURE_DIR=/srv/qfn/captures
export ENGINE_URL=http://127.0.0.1:8897
export WIDE_CHOICE=1
export WIDE_CATALOG_PATH="$PWD/configs/wide-label-catalog.json"
python -m qfn_decision.configure --out runtime
python -m qfn_decision.hybrid --config runtime/gateway.json --check
python -m qfn_decision.hybrid --config runtime/gateway.json
```

This recipe explicitly enables all 256 choice options. `WIDE_CATALOG_PATH` must be absolute. Omitting `WIDE_CHOICE=1` generates a small-choice-only configuration.

The configured path must be readable at that exact absolute path inside every engine worker and by the gateway, with matching bytes. See the [wide-label docs](docs/api-wide-labels.md) for the two separate config blocks and the engine restart requirement.

`MODEL_DIR` belongs to the separately launched engine; generating the configuration does not read or change weights. The generated `nginx.conf` sends the two typed paths (with and without a trailing slash) to the gateway and everything else to the engine. It listens on loopback by default. Change the `listen` line for LAN access and put authentication in front of anything you expose. Check it with `nginx -t` and start nginx yourself.

Examples: [Python](examples/decide.py) and [curl](examples/requests.sh). Both print real responses only.

## Agent skill

A public agent skill, `skills/qfn-decide/SKILL.md`, describes the same choice API for agents that drive an already-running endpoint. It uses the `QFN_ENDPOINT` environment variable and the optional `QFN_API_KEY`; the operator installs their own endpoint. It contains no helper script and no private addresses.

## Verification

[`VERIFICATION.json`](VERIFICATION.json) records the current source and isolated installed-wheel checks, the 256-option configuration seam, and the benchmark-export coverage. It also lists the unresolved engine-provenance and model-redistribution gates.

## Model and licenses

The matching model package is [lychee888/Qwen3.8-Flash-Next-Hybrid-Decision](https://huggingface.co/lychee888/Qwen3.8-Flash-Next-Hybrid-Decision), which carries the complete weight tree this runtime expects plus the card, decision configuration and catalog. It derives from [lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE](https://huggingface.co/lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE) at revision `1b3c3cedc14fb3bee4294dea5e3515af74b8268a`. The weights are licensed under the Qwen Community License 1.0, not by this repository.

Project code is Apache-2.0. The inherited overlays keep vLLM's Apache headers and the MIT recipe attribution. Model licensing is separate: the quantized derivative's card declares Apache-2.0, but the official Qwen base carries the Qwen Community License 1.0, which includes conditions for commercial Model as a Service and AI Work Assistant businesses. An Apache software license does not remove those model-license conditions. We have not pinned which official base revision the derivative was made from, so do not read this repository as granting unrestricted commercial rights to the weights. See [licenses](docs/licenses.md) and [NOTICE](NOTICE).

The source model has reduced safety alignment, and this runtime adds no safety filter. Check its decisions before using them for anything that affects people, money or access to services.
