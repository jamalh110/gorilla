# Mid-Decode D2L LoRA Switching Results

Total generation records: 80

## Mechanical gates

- Custom-loop vs `model.generate()` exact parity: 0/0
- Correct→correct no-op exact parity: 0/0
- Token-zero switches match their always-on destination: 0/0
- Execution errors: 0

## Accuracy and runtime

| Backend | Subset | Condition | Boundary | Cache | N | Accuracy | Malformed | Switch applied | Mean latency | Switch latency |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| peft | pilot | correct_only | - | preserve | 40/40 | 20.00% | 0.00% | N/A | 9.232s | N/A |
| peft | pilot | correct_to_base | after_tool_call | preserve | 40/40 | 20.00% | 0.00% | 100.00% | 9.128s | 7.86ms |

## Paired deltas from always-correct LoRA

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p | First divergence |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| peft | pilot | correct_to_base | after_tool_call | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 | 227.5 |

## Paired deltas where the switch actually fired

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p |
|---|---|---|---|---|---:|---:|---:|---:|
| peft | pilot | correct_to_base | after_tool_call | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 |

## Decision gates

- Mechanically feasible: FAIL
- Operationally useful: PASS
- Behavioral viability: pending a full-set result

## Architecture recommendation

- Awaiting the full phase-switch result.
- Do not treat the current D2L adapters as freely interchangeable token-level experts without joint switch-aware training.
