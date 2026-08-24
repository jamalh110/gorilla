# Mid-Decode D2L LoRA Switching Results

Total generation records: 162

## Mechanical gates

- Custom-loop vs `model.generate()` exact parity: 0/0
- Correct→correct no-op exact parity: 0/0
- Token-zero switches match their always-on destination: 0/0
- Execution errors: 0

## Accuracy and runtime

| Backend | Subset | Condition | Boundary | Cache | N | Accuracy | Malformed | Switch applied | Mean latency | Switch latency |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| peft | pilot | correct_only | - | preserve | 40/40 | 35.00% | 20.00% | N/A | 2.158s | N/A |
| peft | pilot | correct_to_base | after_arguments_key | preserve | 40/40 | 35.00% | 20.00% | 100.00% | 1.990s | 6.91ms |
| peft | pilot | correct_to_wrong | after_arguments_key | preserve | 40/40 | 32.50% | 20.00% | 100.00% | 2.176s | 17.73ms |
| peft | pilot | wrong_to_correct | token_16 | recompute | 40/40 | 30.00% | 20.00% | 100.00% | 2.185s | 17.90ms |
| peft | smoke | correct_only | - | preserve | 1/1 | 100.00% | 0.00% | N/A | 3.548s | N/A |
| peft | smoke | correct_to_base | after_tool_call | preserve | 1/1 | 100.00% | 0.00% | 0.00% | 2.767s | N/A |

## Paired deltas from always-correct LoRA

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p | First divergence |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| peft | pilot | correct_to_base | after_arguments_key | preserve | 40 | 0.00% | [-7.50%, 7.50%] | 1.0000 | 20.7 |
| peft | pilot | correct_to_wrong | after_arguments_key | preserve | 40 | -2.50% | [-10.00%, 5.00%] | 1.0000 | 20.1 |
| peft | pilot | wrong_to_correct | token_16 | recompute | 40 | -5.00% | [-12.50%, 0.00%] | 0.5000 | 13.7 |
| peft | smoke | correct_to_base | after_tool_call | preserve | 1 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |

## Paired deltas where the switch actually fired

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p |
|---|---|---|---|---|---:|---:|---:|---:|
| peft | pilot | correct_to_base | after_arguments_key | preserve | 40 | 0.00% | [-7.50%, 7.50%] | 1.0000 |
| peft | pilot | correct_to_wrong | after_arguments_key | preserve | 40 | -2.50% | [-10.00%, 5.00%] | 1.0000 |
| peft | pilot | wrong_to_correct | token_16 | recompute | 40 | -5.00% | [-12.50%, 0.00%] | 0.5000 |

## Decision gates

- Mechanically feasible: FAIL
- Operationally useful: PASS
- Behavioral viability: pending a full-set result

## Architecture recommendation

- Awaiting the full phase-switch result.
- Do not treat the current D2L adapters as freely interchangeable token-level experts without joint switch-aware training.
