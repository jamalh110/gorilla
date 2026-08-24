# Post-Prefill LoRA Attachment Pilot

## Protocol

- Dataset: the existing 10-case `live_simple` smoke subset.
- Adapters: `live_simple_schema_sft` (the adapter family used by the 60.47%
  full evaluation).
- Decode budget: 1,024 output tokens, greedy.
- Hybrid condition:
  1. Prefill every prompt token except the final token with the base model.
  2. Retain that base-model KV cache.
  3. Activate the correct LoRA.
  4. Replay the final prompt token with the correct LoRA.
  5. Select output token zero from the resulting LoRA-conditioned logits.
  6. Continue decoding with the correct LoRA and hybrid cache.

## Results

| Condition | Correct | Accuracy | Malformed | Mean latency |
|---|---:|---:|---:|---:|
| Base only | 0/10 | 0% | 10% | 1.54 s |
| Base prefill, replay last token with correct LoRA | 3/10 | 30% | 0% | 1.42 s |
| Correct LoRA for full prefill and decode | 5/10 | 50% | 0% | 9.41 s |

The hybrid condition was 20 percentage points below the always-correct LoRA:

- Paired bootstrap 95% CI: [-60, +20] points.
- Exact McNemar p-value: 0.625.
- Pair outcomes: one hybrid win, three hybrid losses, six ties.

The sample is too small for a definitive accuracy conclusion. The direction is
negative, but the confidence interval is wide.

## Behavioral signal

- The switch fired in all 10 cases and took 12.28 ms on average.
- The hybrid condition produced a syntactically valid tool call in all 10
  cases and usually recovered the correct function name.
- Its dominant failures were schema details: five missing-required-parameter
  errors, one wrong function name, and one wrong value.
- All hybrid outputs diverged from the always-correct trajectory at output
  token zero.
- Hybrid outputs usually skipped long reasoning and emitted `<tool_call>`
  immediately, explaining the large latency reduction.

This suggests that replaying one prompt token under the correct LoRA is often
enough to recover tool-call structure and function identity, but not enough to
reconstruct all schema details that full LoRA-prefill encoded into the cache.

A subsequent 258-case run was authorized; see `FULL_FINDINGS.md`.
