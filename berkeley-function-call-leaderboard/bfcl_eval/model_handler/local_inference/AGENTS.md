# Doc-to-LoRA BFCL Handler

This directory contains the BFCL handler for Doc-to-LoRA (D2L) models. Unlike
standard handlers that inject tool schemas into the prompt, D2L internalizes
tool definitions into LoRA weights via a hypernetwork.

## Files

| File | Purpose |
|------|---------|
| `doc_to_lora.py` | `DocToLoraHandler` — BFCL handler class. Spawns a D2L worker subprocess, normalizes BFCL tool schemas, internalizes tools, queries the model, and parses `<tool_call>` output |
| `d2l_worker.py` | Subprocess worker running inside D2L's virtualenv. Handles model loading, tool internalization (including multi-chunk LoRA merging), generation (with optional constrained decoding), and PEFT adapter export. Communicates via JSON-lines over stdin/stdout |

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

## Maintenance Rule

When modifying these files, also update the D2L project's `AGENTS.md` (at
`~/tool-lora/doc-to-lora/AGENTS.md`) which documents the full BFCL integration.
Keep the schema normalization table in sync between the two locations.
