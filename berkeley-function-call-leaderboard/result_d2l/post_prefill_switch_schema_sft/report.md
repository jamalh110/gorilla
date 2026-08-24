# Mid-Decode D2L LoRA Switching Results

Total generation records: 556

## Mechanical gates

- Custom-loop vs `model.generate()` exact parity: 10/10
- Correct→correct no-op exact parity: 0/0
- Token-zero switches match their always-on destination: 0/0
- Execution errors: 0

## Accuracy and runtime

| Backend | Subset | Condition | Boundary | Cache | N | Accuracy | Malformed | Switch applied | Mean latency | Switch latency |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| peft | full | base_prefill_replay_last_to_correct | - | preserve | 258/258 | 34.50% | 5.81% | 100.00% | 1.644s | 52.98ms |
| peft | full | correct_only | - | preserve | 258/258 | 63.57% | 0.39% | N/A | 8.797s | N/A |
| peft | smoke | base_only | - | preserve | 10/10 | 0.00% | 10.00% | N/A | 1.544s | N/A |
| peft | smoke | base_prefill_replay_last_to_correct | - | preserve | 10/10 | 30.00% | 0.00% | 100.00% | 1.423s | 12.28ms |
| peft | smoke | correct_only | - | preserve | 10/10 | 50.00% | 0.00% | N/A | 9.411s | N/A |
| peft | smoke | model_generate_correct | - | generate | 10/10 | 50.00% | 0.00% | N/A | 9.402s | N/A |

## Paired deltas from always-correct LoRA

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p | First divergence |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| peft | full | base_prefill_replay_last_to_correct | None | preserve | 258 | -29.07% | [-35.66%, -22.09%] | 0.0000 | 0.0 |
| peft | smoke | base_prefill_replay_last_to_correct | None | preserve | 10 | -20.00% | [-60.00%, 20.00%] | 0.6250 | 0.0 |

## Paired deltas where the switch actually fired

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p |
|---|---|---|---|---|---:|---:|---:|---:|
| peft | full | base_prefill_replay_last_to_correct | None | preserve | 258 | -29.07% | [-35.66%, -22.09%] | 0.0000 |
| peft | smoke | base_prefill_replay_last_to_correct | None | preserve | 10 | -20.00% | [-60.00%, 20.00%] | 0.6250 |

## Decision gates

- Mechanically feasible: FAIL
- Operationally useful: PASS
- Best full schedule: base_prefill_replay_last_to_correct at None (preserve)
- Behavioral 5-point margin, point estimate: FAIL
- Behavioral 5-point margin, 95% CI: INCONCLUSIVE

## Architecture recommendation

- Awaiting the full phase-switch result.
- Do not treat the current D2L adapters as freely interchangeable token-level experts without joint switch-aware training.
