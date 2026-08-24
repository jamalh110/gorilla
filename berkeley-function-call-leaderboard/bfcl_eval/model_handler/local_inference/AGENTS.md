# Doc-to-LoRA BFCL Handler

This directory contains the BFCL handler for Doc-to-LoRA (D2L) models. Unlike
standard handlers that inject tool schemas into the prompt, D2L internalizes
tool definitions into LoRA weights via a hypernetwork.

## Current v6 Late-Schema Handoff (July 23, 2026)

The detailed authoritative handoff is
`/home/jah649/tool-lora/doc-to-lora/AGENTS.md` under
**Sakana v6 Late-Schema Result**. Current evaluation artifacts:

- Sakana v6 checkpoint:
  `train_outputs/runs/Jul23_17-24-52_lambda1.cs.cornell.edu_f6fe4f09/checkpoint-300/pytorch_model.bin`
- Primary exact Phase A result:
  `result_d2l/sakana_late_checkpoint300_phase_a` — **183/200 (91.5%)**,
  route **198/200**, conditional bind **183/198**, zero malformed, and zero
  thinking leakage.
- Oracle-route LoRA-on: **184/200 (92.0%)**.
- Exact late-transcript oracle-base: **182/200 (91.0%)**.
- Prior v5 checkpoint under the exact evaluator: **177/200 (88.5%)**.
- Full report:
  `result_d2l/sakana_late_checkpoint300_phase_a/FULL_FINDINGS.md`.
- The 94% Phase A gate was not met. Do not promote Phase B as the final path
  until late-schema binding quality improves.
- Follow-up plain-schema report:
  `result_d2l/sakana_checkpoint300_plain_schema_bind/FULL_FINDINGS.md`.
  Base Qwen produced zero valid selector calls on the 10-case same-meta smoke
  but emitted the gold candidate name in all 10. The LoRA-on router-free bind
  (`system, user query, user raw schema`) scored **181/200 (90.5%)**, with
  route **198/200**, conditional bind **181/198**, and zero malformed outputs.

Historical v5 artifacts:

- A router:
  `train_outputs/runs/Jul15_19-44-44_lambda1.cs.cornell.edu_3a7ecc9a/pytorch_model.bin`
- A binder:
  `train_outputs/qwen3_0_6b_binder_v5_500/checkpoint-500`
- B joint:
  `train_outputs/runs/Jul15_19-44-48_lambda1.cs.cornell.edu_187fbb32/pytorch_model.bin`
- A smoke: `staged_smoke/variant_a_v5_final_atomic` — 10/10 pipeline, 8/10 AST
- B smoke: `staged_smoke/variant_b_v5_final_atomic` — 10/10 pipeline, 9/10 AST
- B official provisional: `score_d2l/variant_b_v5_final_official` — 83.0%.
  Rerun because that score predates the free-form-object validator fix.
- A official: `variant_a_v5_final_official` was running when this was written.
- Existing one-pass Apr08 official score: 78.0%.

Important Variant B behavior:

- the all-tools generated LoRA remains attached for routing and binding;
- routing and binding are greedy and unconstrained: no XGrammar, lexical
  processor, assistant prefix, repair retry, or thinking;
- the selected original BFCL schema is appended after the untouched
  `select_tool` assistant call as a Qwen `<tool_response>`;
- Phase A fully recomputes the exact `P || R || S` transcript and asserts token
  prefix identity; Phase B retains KV state and processes only `S`;
- traces record P/R/S/full token hashes, raw and normalized schemas, active
  adapter identity, constraint mode, and malformed output;
- free-form propertyless objects permit arbitrary keys;
- property-declared objects remain strict;
- Variant A binder never receives the original query;
- official runs must use full GPU permissions, not the shell sandbox.

Do not use older v2-v4 smoke/score artifacts for final comparisons.

## Files

| File | Purpose |
|------|---------|
| `doc_to_lora.py` | `DocToLoraHandler` — BFCL handler class. Spawns a D2L worker subprocess, normalizes BFCL tool schemas, internalizes tools, queries the model, and parses `<tool_call>` output |
| `d2l_worker.py` | Subprocess worker with exact transcript rendering/hashes, full-recompute late-schema generation, and cache-retaining `start_late_session` / `append_late_schema_session` / `close_session` commands |
| `doc_to_lora_staged.py` | Variant A legacy path plus Variant B's unconstrained LoRA-on `select_tool` → raw late-schema → real-call protocol. Includes a frozen-base oracle-route ceiling handler |
| `binder_worker.py` | Plain-HF Qwen3-0.6B worker for Variant A. Receives only selected schema plus intent and always calls `apply_chat_template(enable_thinking=False)` |

## Schema Normalization

BFCL tool definitions use non-standard types that differ from the OpenAI-style
JSON Schema format used in training data (Nemotron, Toucan). The handler's
`_normalize_bfcl_function` rewrites these before internalization so the model
sees the same format it was trained on:

| BFCL type | Normalized to |
|-----------|---------------|
| `dict` | `object` |
| `tuple` | `array` |
| `float` | `number` |
| `String` | `string` |
| `Boolean` | `boolean` |
| `HashMap` | `object` |
| `ArrayList` | `array` |
| `any` | `string` |

The non-standard `"optional"` key is also stripped from properties and from the
top-level parameters object.

This normalization is mirrored in the D2L training data preparation
(`prepare_bfcl_data.py`) so training and inference see identical schemas.

## Constrained Decoding

When `D2L_RESTRICT_TOOLGEN=1` is set, the worker creates a
`ToolConstrainedLogitsProcessor` that restricts generation to valid function
names, parameter keys, and enum values. The processor uses character-level tries
and a JSON state machine. It correctly handles BFCL naming patterns including:

- Dotted names (`math.factorial`, `triangle_properties.get`)
- CamelCase (`GorillaFileSystem.cat`, `GeometryPresentation.createPresentation`)

Dots are treated as ordinary characters in the trie; the JSON state machine
extracts the full dotted name for parameter trie resolution.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `D2L_CHECKPOINT_PATH` | (required) | Path to D2L checkpoint (`pytorch_model.bin`) |
| `D2L_CHUNK_SIZE` | `1024` | Max tokens per context chunk for internalization |
| `D2L_MAX_NEW_TOKENS` | `1024` | Max tokens to generate |
| `D2L_SOURCE_PATH` | `~/tool-lora/doc-to-lora/src` | Path to D2L source directory |
| `D2L_PYTHON` | `~/tool-lora/doc-to-lora/.venv/bin/python` | Python interpreter in D2L's virtualenv |
| `D2L_RESTRICT_TOOLGEN` | `0` | Set to `1` to enable constrained decoding |
| `D2L_RAW_LOG` | (disabled) | Path to write raw model outputs for debugging |
| `D2L_BINDER_CHECKPOINT_PATH` | (required for A) | Standard Hugging Face checkpoint for the fully fine-tuned Qwen3-0.6B binder |
| `D2L_BINDER_PYTHON` | `D2L_PYTHON` | Python executable used for `binder_worker.py` |
| `D2L_STAGED_MAIN_GPU` | first visible GPU | GPU used by the staged D2L 4B worker |
| `D2L_BINDER_GPU` | second visible GPU | GPU used by the Variant A binder worker |
| `D2L_BINDER_MAX_NEW_TOKENS` | `256` | Maximum Variant A binder completion length |
| `D2L_STATEFUL_CONTINUATION` | `0` | `1` to retain `P || R` KV state and append only late-schema `S`; default Phase A exactly recomputes the same transcript |
| `D2L_ORACLE_ROUTE` | `0` | Inject the BFCL gold `select_tool` call for oracle-route bind diagnostics |

## Staged Contracts

- Variant A internalizes all normalized tools in Qwen3-4B, emits one
  `route_and_plan` call whose `tool_name` is dynamically constrained to the real
  names, then looks up the selected normalized schema. The binder prompt contains
  only that schema and the parsed schema-neutral intent.
- Variant B emits one unconstrained `select_tool` call from the meta schema and
  original query. It then appends the original/raw selected BFCL function in a
  `role: tool` response and decodes the real call from the same conversation
  with the same generated LoRA. There is no second system message or repeated
  query.
- Primary Variant B has one deterministic attempt per stage. A stop criterion
  may end decoding after `</tool_call>` but does not alter logits. Outputs are
  parsed without content repair; schema validation is diagnostic and the
  unchanged BFCL AST checker determines correctness. Text
  outside the call, extra calls, unknown arguments, missing required arguments,
  invalid nested types, and invalid enum values are rejected.
- `run_staged_smoke.py` selects a deterministic seed-42 set of ten `multiple`
  cases covering 2/3/4 tools, deep-copies every case, runs the normal BFCL AST
  checker, and writes separate raw and score JSONL artifacts.

## Maintenance Rule

When modifying these files, also update the D2L project's `AGENTS.md` (at
`~/tool-lora/doc-to-lora/AGENTS.md`) which documents the full BFCL integration.
Keep the schema normalization table in sync between the two locations.
