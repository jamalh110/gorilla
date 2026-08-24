# Mid-Decode D2L LoRA Switching Results

Total generation records: 40

## Mechanical gates

- Custom-loop vs `model.generate()` exact parity: 10/10
- Correct→correct no-op exact parity: 0/0
- Token-zero switches match their always-on destination: 0/0
- Execution errors: 0

## Accuracy and runtime

| Backend | Subset | Condition | Boundary | Cache | N | Accuracy | Malformed | Switch applied | Mean latency | Switch latency |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| peft | smoke | base_only | - | preserve | 10/10 | 0.00% | 10.00% | N/A | 1.538s | N/A |
| peft | smoke | base_prefill_replay_8_to_correct | - | preserve | 10/10 | 10.00% | 0.00% | 100.00% | 1.454s | 12.14ms |
| peft | smoke | correct_only | - | preserve | 10/10 | 50.00% | 0.00% | N/A | 9.377s | N/A |
| peft | smoke | model_generate_correct | - | generate | 10/10 | 50.00% | 0.00% | N/A | 9.376s | N/A |

## Paired deltas from always-correct LoRA

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p | First divergence |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| peft | smoke | base_prefill_replay_8_to_correct | None | preserve | 10 | -40.00% | [-70.00%, -10.00%] | 0.1250 | 0.0 |

## Paired deltas where the switch actually fired

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p |
|---|---|---|---|---|---:|---:|---:|---:|
| peft | smoke | base_prefill_replay_8_to_correct | None | preserve | 10 | -40.00% | [-70.00%, -10.00%] | 0.1250 |

## Decision gates

- Mechanically feasible: FAIL
- Operationally useful: PASS
- Behavioral viability: pending a full-set result

## Architecture recommendation

- Awaiting the full phase-switch result.
- Do not treat the current D2L adapters as freely interchangeable token-level experts without joint switch-aware training.
