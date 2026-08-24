# Full Post-Prefill LoRA Attachment Results

## Protocol

- Dataset: all 258 BFCL `live_simple` cases.
- Adapters: `live_simple_schema_sft`.
- Decode budget: 1,024 output tokens, greedy.
- Baseline: correct LoRA active for the complete prompt prefill and decode.
- Treatment:
  1. Prefill `prompt[:-1]` with the base model.
  2. Retain the base-model KV cache.
  3. Activate the correct LoRA.
  4. Replay the final prompt token under the correct LoRA.
  5. Select output token zero from the resulting LoRA-conditioned logits.
  6. Continue decoding with the correct LoRA and hybrid cache.

## Results

| Condition | Correct | Accuracy | Malformed | Mean latency |
|---|---:|---:|---:|---:|
| Correct LoRA for full prefill and decode | 164/258 | 63.57% | 0.39% | 8.80 s |
| Base prefill, replay last token with correct LoRA | 89/258 | 34.50% | 5.81% | 1.64 s |

Paired treatment effect:

- Accuracy delta: -29.07 percentage points.
- Paired bootstrap 95% CI: [-35.66, -22.09] points.
- Exact McNemar p-value: 7.52e-15.
- Pair outcomes: 13 treatment wins, 88 treatment losses, 157 ties.

The degradation is large and statistically decisive. Standard schema-SFT D2L
LoRAs are not compatible with a base-prefilled prompt when only the final
prompt token is replayed under the adapter.

## Failure pattern

The treatment activated successfully in all 258 cases and always selected
output token zero under the correct LoRA. Nevertheless, every treatment output
diverged from the always-correct trajectory at output token zero.

The hybrid cache often retained enough signal to emit a plausible tool call,
but schema fidelity degraded:

- 58 missing-required-parameter errors.
- 38 string-value errors.
- 30 wrong-function-name errors.
- 15 wrong-call-count errors.
- 8 unexpected-parameter errors.
- 7 simple type errors.

This is consistent with the pilot: late attachment often recovers tool-call
format or broad function intent, but not the exact schema and argument mapping.

## Runtime

- Adapter activation after prefill: 52.98 ms on average.
- Hybrid generation: 1.64 s on average.
- Always-correct generation: 8.80 s on average.

The hybrid path is faster primarily because it usually skips the long
LoRA-conditioned reasoning trajectory and emits a short call directly. The
speedup does not compensate for the 29-point accuracy loss.

## Conclusion

For these standard D2L/schema-SFT adapters, the correct LoRA must influence a
substantial portion of prompt processing. Replaying only the final prompt token
does not make a base-produced cache equivalent to a LoRA-produced cache.

Potential follow-up experiments should vary the replayed suffix length
(for example 8, 16, 32, or 64 prompt tokens) to measure how much LoRA-prefill
is required, or train invocation-aware adapters such as Activated LoRA.
