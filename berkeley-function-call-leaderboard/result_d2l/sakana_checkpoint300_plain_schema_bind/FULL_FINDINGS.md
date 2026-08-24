# Meta-Name and Router-Free Plain-Schema Experiments

## Shared setup

- Checkpoint:
  `train_outputs/runs/Jul23_17-24-52_lambda1.cs.cornell.edu_f6fe4f09/checkpoint-300/pytorch_model.bin`
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Greedy, unconstrained decoding with thinking disabled.
- Dataset: BFCL `multiple`.

## Experiment 1: base model with the meta-tool

The generated LoRA was disabled, while the exact D2L meta-tool prompt and its
candidate-name enum remained visible.

The fixed 10-case smoke produced zero structurally valid `select_tool` calls.
Base Qwen emitted literal `\n` escape sequences in the call wrapper, and in
five cases it called the real candidate tool directly instead of the
`select_tool` meta-tool. The other five attempted `select_tool` semantically
but were still malformed.

Despite failing the routing protocol, the real or nested selected name matched
the BFCL gold tool in all 10 cases. This indicates that candidate names alone
carry substantial routing signal on the smoke set, while the generated D2L
LoRA is important for consistently producing the trained selector protocol.

Because the smoke was structurally invalid in all 10 cases, it was not promoted
to a full end-to-end BFCL run.

Artifacts:

- `staged_smoke/base_meta_names_only/variant_b_raw.jsonl`
- `staged_smoke/base_meta_names_only/variant_b_score.jsonl`

## Experiment 2: LoRA-on router-free plain-schema bind

Stage 1 used the unchanged trained D2L router prompt. After tool selection,
stage 2 fully recomputed this transcript:

```text
system(meta-tool) -> user(original query)
-> user(out-of-band routing-complete instruction + raw BFCL schema)
-> assistant(real call)
```

The selector assistant turn is absent from stage 2. The selected schema is a
plain user message, not a `tool` role or `<tool_response>`. The original BFCL
schema is unnormalized, the query appears once, and the generated all-tools
LoRA remains attached. A future implementation could retain the pre-router
cache, route on a branch, discard that branch, and append the schema message;
that cache optimization was intentionally not implemented.

The first smoke showed that changing the trained meta-system instruction broke
selector syntax in all 10 cases. Keeping the exact trained router prompt fixed
that issue. The corrected smoke scored 9/10.

Full BFCL result:

| Metric | Result |
|---|---:|
| End-to-end | 181/200 (90.5%) |
| Route | 198/200 (99.0%) |
| Conditional bind | 181/198 (91.41%) |
| Malformed | 0 |
| Thinking leakage | 0 |

All stage-1 and stage-2 attempts used `constraint_mode: none`.

Compared with the current late `<tool_response>` protocol at 183/200, the plain
schema treatment had two paired wins and four paired losses, for a net
two-example regression. Its 19 failures comprise ten string-value errors, four
other value errors, two missing-optional errors, two wrong-function/routing
errors, and one list/tuple value error.

Artifacts:

- `result_d2l/sakana_checkpoint300_plain_schema_bind`
- `score_d2l/sakana_checkpoint300_plain_schema_bind`
- `staged_smoke/sakana_plain_schema_bind`

## Conclusion

Tool names alone were sufficient to infer the correct tool on all 10 base-model
smoke cases, but not to reliably obey the `select_tool` wire format. Removing
the selector call and replacing the tool response with a plain appended schema
did not improve binding: it scored 90.5%, versus 91.5% for the late
`<tool_response>` transcript.
