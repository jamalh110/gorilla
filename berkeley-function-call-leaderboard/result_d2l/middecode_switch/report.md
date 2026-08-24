# Mid-Decode D2L LoRA Switching Results

Total generation records: 2882

## Mechanical gates

- Custom-loop vs `model.generate()` exact parity: 30/30
- Correct→correct no-op exact parity: 100/100
- Token-zero switches match their always-on destination: 119/120
- Execution errors: 0

## Accuracy and runtime

| Backend | Subset | Condition | Boundary | Cache | N | Accuracy | Malformed | Switch applied | Mean latency | Switch latency |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| peft | full | correct_only | - | preserve | 258/258 | 34.11% | 24.03% | N/A | 8.257s | N/A |
| peft | full | correct_to_base | after_tool_call | preserve | 258/258 | 33.33% | 23.64% | 82.17% | 8.194s | 23.90ms |
| peft | full | correct_to_wrong | after_tool_call | preserve | 258/258 | 31.01% | 23.64% | 82.17% | 8.292s | 57.26ms |
| peft | full | wrong_to_correct | token_16 | recompute | 258/258 | 31.78% | 25.19% | 100.00% | 8.292s | 53.54ms |
| peft | pilot | base_only | - | preserve | 40/40 | 0.00% | 17.50% | N/A | 1.707s | N/A |
| peft | pilot | base_to_correct | token_0 | preserve | 40/40 | 15.00% | 35.00% | 100.00% | 7.872s | 26.62ms |
| peft | pilot | base_to_correct | token_8 | preserve | 40/40 | 0.00% | 17.50% | 100.00% | 1.634s | 27.05ms |
| peft | pilot | correct_only | - | preserve | 40/40 | 15.00% | 35.00% | N/A | 7.820s | N/A |
| peft | pilot | correct_to_base | after_tool_call | preserve | 29/29 | 20.69% | 10.34% | 100.00% | 7.100s | 12.03ms |
| peft | pilot | correct_to_base | after_tool_call | recompute | 29/29 | 17.24% | 10.34% | 100.00% | 7.117s | 12.06ms |
| peft | pilot | correct_to_correct | token_16 | preserve | 40/40 | 15.00% | 35.00% | 100.00% | 7.867s | 27.22ms |
| peft | pilot | correct_to_wrong | after_tool_call | preserve | 29/29 | 24.14% | 10.34% | 100.00% | 7.164s | 29.02ms |
| peft | pilot | correct_to_wrong | after_tool_call | recompute | 29/29 | 20.69% | 10.34% | 100.00% | 7.159s | 28.92ms |
| peft | pilot | correct_to_wrong | eos | preserve | 40/40 | 15.00% | 35.00% | 35.00% | 7.871s | 29.23ms |
| peft | pilot | model_generate_correct | - | generate | 10/10 | 20.00% | 30.00% | N/A | 7.622s | N/A |
| peft | pilot | wrong_only | - | preserve | 40/40 | 10.00% | 35.00% | N/A | 7.963s | N/A |
| peft | pilot | wrong_to_correct | token_16 | preserve | 40/40 | 10.00% | 30.00% | 100.00% | 7.927s | 27.24ms |
| peft | pilot | wrong_to_correct | token_16 | recompute | 40/40 | 17.50% | 30.00% | 100.00% | 7.818s | 27.02ms |
| peft | smoke | base_only | - | preserve | 10/10 | 0.00% | 10.00% | N/A | 1.614s | N/A |
| peft | smoke | base_to_correct | after_think | preserve | 7/7 | 0.00% | 14.29% | 0.00% | 1.561s | N/A |
| peft | smoke | base_to_correct | after_think | recompute | 7/7 | 0.00% | 14.29% | 0.00% | 1.562s | N/A |
| peft | smoke | base_to_correct | after_tool_call | preserve | 7/7 | 0.00% | 14.29% | 0.00% | 1.563s | N/A |
| peft | smoke | base_to_correct | after_tool_call | recompute | 7/7 | 0.00% | 14.29% | 0.00% | 1.562s | N/A |
| peft | smoke | base_to_correct | token_0 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 7.961s | 12.36ms |
| peft | smoke | base_to_correct | token_16 | preserve | 10/10 | 0.00% | 10.00% | 100.00% | 1.722s | 12.67ms |
| peft | smoke | base_to_correct | token_16 | recompute | 10/10 | 0.00% | 0.00% | 100.00% | 1.752s | 12.62ms |
| peft | smoke | base_to_correct | token_32 | preserve | 10/10 | 0.00% | 10.00% | 50.00% | 1.842s | 12.91ms |
| peft | smoke | base_to_correct | token_32 | recompute | 10/10 | 0.00% | 0.00% | 50.00% | 1.785s | 12.84ms |
| peft | smoke | base_to_correct | token_8 | preserve | 10/10 | 0.00% | 10.00% | 100.00% | 1.757s | 11.67ms |
| peft | smoke | base_to_correct | token_8 | recompute | 10/10 | 0.00% | 0.00% | 100.00% | 2.140s | 12.61ms |
| peft | smoke | correct_only | - | preserve | 10/10 | 20.00% | 30.00% | N/A | 7.667s | N/A |
| peft | smoke | correct_to_base | after_think | preserve | 7/7 | 28.57% | 0.00% | 100.00% | 6.585s | 5.63ms |
| peft | smoke | correct_to_base | after_think | recompute | 7/7 | 28.57% | 0.00% | 100.00% | 6.657s | 5.54ms |
| peft | smoke | correct_to_base | after_tool_call | preserve | 7/7 | 28.57% | 0.00% | 100.00% | 6.734s | 5.55ms |
| peft | smoke | correct_to_base | after_tool_call | recompute | 7/7 | 28.57% | 0.00% | 100.00% | 6.704s | 5.65ms |
| peft | smoke | correct_to_base | token_0 | preserve | 10/10 | 0.00% | 10.00% | 100.00% | 1.690s | 4.55ms |
| peft | smoke | correct_to_base | token_16 | preserve | 10/10 | 0.00% | 70.00% | 100.00% | 8.034s | 4.90ms |
| peft | smoke | correct_to_base | token_16 | recompute | 10/10 | 0.00% | 70.00% | 100.00% | 5.457s | 4.91ms |
| peft | smoke | correct_to_base | token_32 | preserve | 10/10 | 0.00% | 50.00% | 100.00% | 7.701s | 5.11ms |
| peft | smoke | correct_to_base | token_32 | recompute | 10/10 | 0.00% | 10.00% | 100.00% | 4.056s | 5.13ms |
| peft | smoke | correct_to_base | token_8 | preserve | 10/10 | 0.00% | 60.00% | 100.00% | 7.906s | 4.85ms |
| peft | smoke | correct_to_base | token_8 | recompute | 10/10 | 0.00% | 70.00% | 100.00% | 5.456s | 4.82ms |
| peft | smoke | correct_to_correct | token_16 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 7.649s | 11.80ms |
| peft | smoke | correct_to_wrong | after_think | preserve | 7/7 | 42.86% | 0.00% | 100.00% | 6.802s | 13.40ms |
| peft | smoke | correct_to_wrong | after_think | recompute | 7/7 | 28.57% | 0.00% | 100.00% | 6.848s | 13.36ms |
| peft | smoke | correct_to_wrong | after_tool_call | preserve | 7/7 | 42.86% | 0.00% | 100.00% | 6.802s | 13.37ms |
| peft | smoke | correct_to_wrong | after_tool_call | recompute | 7/7 | 28.57% | 0.00% | 100.00% | 6.873s | 13.33ms |
| peft | smoke | correct_to_wrong | eos | preserve | 10/10 | 20.00% | 30.00% | 30.00% | 7.647s | 13.45ms |
| peft | smoke | correct_to_wrong | token_0 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 7.862s | 12.15ms |
| peft | smoke | correct_to_wrong | token_16 | preserve | 10/10 | 30.00% | 30.00% | 100.00% | 7.904s | 12.59ms |
| peft | smoke | correct_to_wrong | token_16 | recompute | 10/10 | 20.00% | 40.00% | 100.00% | 7.721s | 12.59ms |
| peft | smoke | correct_to_wrong | token_32 | preserve | 10/10 | 10.00% | 30.00% | 100.00% | 7.793s | 12.83ms |
| peft | smoke | correct_to_wrong | token_32 | recompute | 10/10 | 20.00% | 40.00% | 100.00% | 7.639s | 12.86ms |
| peft | smoke | correct_to_wrong | token_8 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 7.699s | 12.54ms |
| peft | smoke | correct_to_wrong | token_8 | recompute | 10/10 | 10.00% | 30.00% | 100.00% | 7.325s | 12.55ms |
| peft | smoke | model_generate_correct | - | generate | 10/10 | 20.00% | 30.00% | N/A | 7.659s | N/A |
| peft | smoke | wrong_only | - | preserve | 10/10 | 20.00% | 30.00% | N/A | 7.581s | N/A |
| peft | smoke | wrong_to_correct | after_think | preserve | 7/7 | 28.57% | 14.29% | 85.71% | 6.756s | 13.26ms |
| peft | smoke | wrong_to_correct | after_think | recompute | 7/7 | 28.57% | 14.29% | 85.71% | 6.746s | 13.41ms |
| peft | smoke | wrong_to_correct | after_tool_call | preserve | 7/7 | 28.57% | 14.29% | 85.71% | 6.758s | 13.47ms |
| peft | smoke | wrong_to_correct | after_tool_call | recompute | 7/7 | 14.29% | 28.57% | 85.71% | 6.776s | 13.35ms |
| peft | smoke | wrong_to_correct | token_0 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 7.926s | 12.19ms |
| peft | smoke | wrong_to_correct | token_16 | preserve | 10/10 | 20.00% | 40.00% | 100.00% | 7.479s | 12.67ms |
| peft | smoke | wrong_to_correct | token_16 | recompute | 10/10 | 30.00% | 30.00% | 100.00% | 7.448s | 12.61ms |
| peft | smoke | wrong_to_correct | token_32 | preserve | 10/10 | 20.00% | 40.00% | 100.00% | 7.541s | 12.86ms |
| peft | smoke | wrong_to_correct | token_32 | recompute | 10/10 | 30.00% | 40.00% | 100.00% | 7.767s | 12.84ms |
| peft | smoke | wrong_to_correct | token_8 | preserve | 10/10 | 20.00% | 40.00% | 100.00% | 7.452s | 12.56ms |
| peft | smoke | wrong_to_correct | token_8 | recompute | 10/10 | 30.00% | 40.00% | 100.00% | 7.612s | 12.51ms |
| raw | pilot | base_only | - | preserve | 40/40 | 0.00% | 20.00% | N/A | 1.749s | N/A |
| raw | pilot | correct_only | - | preserve | 40/40 | 17.50% | 27.50% | N/A | 9.319s | N/A |
| raw | pilot | correct_to_base | after_tool_call | preserve | 40/40 | 17.50% | 27.50% | 72.50% | 9.098s | 1.48ms |
| raw | pilot | correct_to_correct | token_16 | preserve | 40/40 | 17.50% | 27.50% | 100.00% | 9.330s | 5.16ms |
| raw | pilot | correct_to_wrong | after_tool_call | preserve | 40/40 | 17.50% | 27.50% | 72.50% | 9.317s | 5.40ms |
| raw | pilot | correct_to_wrong | eos | preserve | 40/40 | 17.50% | 27.50% | 30.00% | 9.330s | 5.58ms |
| raw | pilot | wrong_only | - | preserve | 40/40 | 10.00% | 35.00% | N/A | 9.542s | N/A |
| raw | pilot | wrong_to_correct | token_16 | recompute | 40/40 | 20.00% | 30.00% | 100.00% | 9.206s | 5.11ms |
| raw | smoke | base_only | - | preserve | 10/10 | 0.00% | 10.00% | N/A | 1.654s | N/A |
| raw | smoke | base_to_correct | after_think | preserve | 8/8 | 0.00% | 12.50% | 12.50% | 1.691s | 3.59ms |
| raw | smoke | base_to_correct | after_think | recompute | 8/8 | 0.00% | 12.50% | 12.50% | 1.693s | 3.52ms |
| raw | smoke | base_to_correct | after_tool_call | preserve | 8/8 | 0.00% | 12.50% | 0.00% | 1.692s | N/A |
| raw | smoke | base_to_correct | after_tool_call | recompute | 8/8 | 0.00% | 12.50% | 0.00% | 1.692s | N/A |
| raw | smoke | base_to_correct | token_0 | preserve | 10/10 | 30.00% | 20.00% | 100.00% | 8.882s | 3.35ms |
| raw | smoke | base_to_correct | token_16 | preserve | 10/10 | 0.00% | 10.00% | 100.00% | 1.920s | 3.39ms |
| raw | smoke | base_to_correct | token_16 | recompute | 10/10 | 0.00% | 0.00% | 100.00% | 2.070s | 3.22ms |
| raw | smoke | base_to_correct | token_32 | preserve | 10/10 | 0.00% | 10.00% | 50.00% | 1.679s | 3.26ms |
| raw | smoke | base_to_correct | token_32 | recompute | 10/10 | 0.00% | 0.00% | 50.00% | 2.008s | 3.37ms |
| raw | smoke | base_to_correct | token_8 | preserve | 10/10 | 0.00% | 10.00% | 100.00% | 2.014s | 3.43ms |
| raw | smoke | base_to_correct | token_8 | recompute | 10/10 | 0.00% | 0.00% | 100.00% | 2.590s | 3.49ms |
| raw | smoke | correct_only | - | preserve | 10/10 | 30.00% | 20.00% | N/A | 8.657s | N/A |
| raw | smoke | correct_to_base | after_think | preserve | 8/8 | 37.50% | 0.00% | 100.00% | 7.544s | 1.46ms |
| raw | smoke | correct_to_base | after_think | recompute | 8/8 | 37.50% | 12.50% | 100.00% | 7.706s | 1.47ms |
| raw | smoke | correct_to_base | after_tool_call | preserve | 8/8 | 37.50% | 0.00% | 100.00% | 7.638s | 1.45ms |
| raw | smoke | correct_to_base | after_tool_call | recompute | 8/8 | 37.50% | 12.50% | 100.00% | 7.778s | 1.48ms |
| raw | smoke | correct_to_base | token_0 | preserve | 10/10 | 0.00% | 10.00% | 100.00% | 1.696s | 1.22ms |
| raw | smoke | correct_to_base | token_16 | preserve | 10/10 | 0.00% | 70.00% | 100.00% | 8.374s | 1.29ms |
| raw | smoke | correct_to_base | token_16 | recompute | 10/10 | 0.00% | 80.00% | 100.00% | 5.732s | 1.32ms |
| raw | smoke | correct_to_base | token_32 | preserve | 10/10 | 10.00% | 60.00% | 100.00% | 8.431s | 1.33ms |
| raw | smoke | correct_to_base | token_32 | recompute | 10/10 | 0.00% | 10.00% | 100.00% | 4.264s | 1.32ms |
| raw | smoke | correct_to_base | token_8 | preserve | 10/10 | 0.00% | 60.00% | 100.00% | 8.002s | 1.28ms |
| raw | smoke | correct_to_base | token_8 | recompute | 10/10 | 0.00% | 60.00% | 100.00% | 5.799s | 1.27ms |
| raw | smoke | correct_to_correct | token_16 | preserve | 10/10 | 30.00% | 20.00% | 100.00% | 8.638s | 3.51ms |
| raw | smoke | correct_to_wrong | after_think | preserve | 8/8 | 37.50% | 0.00% | 100.00% | 8.012s | 3.76ms |
| raw | smoke | correct_to_wrong | after_think | recompute | 8/8 | 37.50% | 0.00% | 100.00% | 7.981s | 3.85ms |
| raw | smoke | correct_to_wrong | after_tool_call | preserve | 8/8 | 37.50% | 0.00% | 100.00% | 8.025s | 3.77ms |
| raw | smoke | correct_to_wrong | after_tool_call | recompute | 8/8 | 37.50% | 0.00% | 100.00% | 8.016s | 4.03ms |
| raw | smoke | correct_to_wrong | eos | preserve | 10/10 | 30.00% | 20.00% | 30.00% | 8.639s | 4.62ms |
| raw | smoke | correct_to_wrong | token_0 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 8.938s | 3.47ms |
| raw | smoke | correct_to_wrong | token_16 | preserve | 10/10 | 10.00% | 30.00% | 100.00% | 9.034s | 3.53ms |
| raw | smoke | correct_to_wrong | token_16 | recompute | 10/10 | 20.00% | 30.00% | 100.00% | 8.795s | 3.60ms |
| raw | smoke | correct_to_wrong | token_32 | preserve | 10/10 | 20.00% | 40.00% | 100.00% | 9.178s | 3.72ms |
| raw | smoke | correct_to_wrong | token_32 | recompute | 10/10 | 20.00% | 40.00% | 100.00% | 8.892s | 3.65ms |
| raw | smoke | correct_to_wrong | token_8 | preserve | 10/10 | 20.00% | 30.00% | 100.00% | 9.180s | 3.64ms |
| raw | smoke | correct_to_wrong | token_8 | recompute | 10/10 | 20.00% | 30.00% | 100.00% | 8.755s | 3.46ms |
| raw | smoke | model_generate_correct | - | generate | 10/10 | 30.00% | 20.00% | N/A | 8.601s | N/A |
| raw | smoke | wrong_only | - | preserve | 10/10 | 20.00% | 30.00% | N/A | 8.746s | N/A |
| raw | smoke | wrong_to_correct | after_think | preserve | 8/8 | 25.00% | 25.00% | 87.50% | 8.459s | 3.79ms |
| raw | smoke | wrong_to_correct | after_think | recompute | 8/8 | 25.00% | 25.00% | 87.50% | 8.421s | 3.80ms |
| raw | smoke | wrong_to_correct | after_tool_call | preserve | 8/8 | 25.00% | 25.00% | 87.50% | 8.457s | 3.75ms |
| raw | smoke | wrong_to_correct | after_tool_call | recompute | 8/8 | 25.00% | 25.00% | 87.50% | 8.370s | 3.84ms |
| raw | smoke | wrong_to_correct | token_0 | preserve | 10/10 | 30.00% | 20.00% | 100.00% | 8.860s | 3.49ms |
| raw | smoke | wrong_to_correct | token_16 | preserve | 10/10 | 10.00% | 50.00% | 100.00% | 9.395s | 3.51ms |
| raw | smoke | wrong_to_correct | token_16 | recompute | 10/10 | 30.00% | 30.00% | 100.00% | 8.607s | 3.69ms |
| raw | smoke | wrong_to_correct | token_32 | preserve | 10/10 | 20.00% | 40.00% | 100.00% | 8.720s | 3.57ms |
| raw | smoke | wrong_to_correct | token_32 | recompute | 10/10 | 30.00% | 40.00% | 100.00% | 8.737s | 3.54ms |
| raw | smoke | wrong_to_correct | token_8 | preserve | 10/10 | 20.00% | 40.00% | 100.00% | 8.624s | 3.65ms |
| raw | smoke | wrong_to_correct | token_8 | recompute | 10/10 | 30.00% | 40.00% | 100.00% | 8.777s | 3.45ms |

## Paired deltas from always-correct LoRA

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p | First divergence |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| peft | full | correct_to_base | after_tool_call | preserve | 258 | -0.78% | [-2.71%, 1.16%] | 0.6875 | 171.7 |
| peft | full | correct_to_wrong | after_tool_call | preserve | 258 | -3.10% | [-5.81%, -0.39%] | 0.0574 | 177.8 |
| peft | full | wrong_to_correct | token_16 | recompute | 258 | -2.33% | [-5.81%, 0.78%] | 0.2632 | 43.3 |
| peft | pilot | base_to_correct | token_0 | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| peft | pilot | base_to_correct | token_8 | preserve | 40 | -15.00% | [-27.50%, -5.00%] | 0.0312 | 0.0 |
| peft | pilot | correct_to_base | after_tool_call | preserve | 29 | 0.00% | [0.00%, 0.00%] | 1.0000 | 179.4 |
| peft | pilot | correct_to_base | after_tool_call | recompute | 29 | -3.45% | [-10.34%, 0.00%] | 1.0000 | 171.3 |
| peft | pilot | correct_to_wrong | after_tool_call | preserve | 29 | 3.45% | [0.00%, 10.34%] | 1.0000 | 190.2 |
| peft | pilot | correct_to_wrong | after_tool_call | recompute | 29 | 0.00% | [0.00%, 0.00%] | 1.0000 | 183.5 |
| peft | pilot | correct_to_wrong | eos | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| peft | pilot | wrong_to_correct | token_16 | preserve | 40 | -5.00% | [-12.50%, 0.00%] | 0.5000 | 32.3 |
| peft | pilot | wrong_to_correct | token_16 | recompute | 40 | 2.50% | [-5.00%, 10.00%] | 1.0000 | 68.4 |
| peft | smoke | base_to_correct | after_think | preserve | 7 | -28.57% | [-57.14%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | after_think | recompute | 7 | -28.57% | [-57.14%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | after_tool_call | preserve | 7 | -28.57% | [-57.14%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | after_tool_call | recompute | 7 | -28.57% | [-57.14%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| peft | smoke | base_to_correct | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | token_16 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | token_32 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | token_32 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | token_8 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | base_to_correct | token_8 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | correct_to_base | after_think | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 150.4 |
| peft | smoke | correct_to_base | after_think | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 150.4 |
| peft | smoke | correct_to_base | after_tool_call | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 178.4 |
| peft | smoke | correct_to_base | after_tool_call | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 169.7 |
| peft | smoke | correct_to_base | token_0 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 0.0 |
| peft | smoke | correct_to_base | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 22.3 |
| peft | smoke | correct_to_base | token_16 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 19.8 |
| peft | smoke | correct_to_base | token_32 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 34.9 |
| peft | smoke | correct_to_base | token_32 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 34.5 |
| peft | smoke | correct_to_base | token_8 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 15.2 |
| peft | smoke | correct_to_base | token_8 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 13.4 |
| peft | smoke | correct_to_wrong | after_think | preserve | 7 | 14.29% | [0.00%, 42.86%] | 1.0000 | 161.0 |
| peft | smoke | correct_to_wrong | after_think | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 167.5 |
| peft | smoke | correct_to_wrong | after_tool_call | preserve | 7 | 14.29% | [0.00%, 42.86%] | 1.0000 | 198.0 |
| peft | smoke | correct_to_wrong | after_tool_call | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 169.0 |
| peft | smoke | correct_to_wrong | eos | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| peft | smoke | correct_to_wrong | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 33.9 |
| peft | smoke | correct_to_wrong | token_16 | preserve | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 | 58.1 |
| peft | smoke | correct_to_wrong | token_16 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 41.1 |
| peft | smoke | correct_to_wrong | token_32 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 48.4 |
| peft | smoke | correct_to_wrong | token_32 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 46.2 |
| peft | smoke | correct_to_wrong | token_8 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 50.4 |
| peft | smoke | correct_to_wrong | token_8 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 35.2 |
| peft | smoke | wrong_to_correct | after_think | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 38.4 |
| peft | smoke | wrong_to_correct | after_think | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 38.4 |
| peft | smoke | wrong_to_correct | after_tool_call | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 | 38.4 |
| peft | smoke | wrong_to_correct | after_tool_call | recompute | 7 | -14.29% | [-42.86%, 0.00%] | 1.0000 | 38.4 |
| peft | smoke | wrong_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| peft | smoke | wrong_to_correct | token_16 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 33.6 |
| peft | smoke | wrong_to_correct | token_16 | recompute | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 | 57.7 |
| peft | smoke | wrong_to_correct | token_32 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 35.1 |
| peft | smoke | wrong_to_correct | token_32 | recompute | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 | 49.7 |
| peft | smoke | wrong_to_correct | token_8 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 33.7 |
| peft | smoke | wrong_to_correct | token_8 | recompute | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 | 62.9 |
| raw | pilot | correct_to_base | after_tool_call | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 | 175.5 |
| raw | pilot | correct_to_wrong | after_tool_call | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 | 199.8 |
| raw | pilot | correct_to_wrong | eos | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| raw | pilot | wrong_to_correct | token_16 | recompute | 40 | 2.50% | [-5.00%, 10.00%] | 1.0000 | 59.8 |
| raw | smoke | base_to_correct | after_think | preserve | 8 | -37.50% | [-75.00%, -12.50%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | after_think | recompute | 8 | -37.50% | [-75.00%, -12.50%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | after_tool_call | preserve | 8 | -37.50% | [-75.00%, -12.50%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | after_tool_call | recompute | 8 | -37.50% | [-75.00%, -12.50%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| raw | smoke | base_to_correct | token_16 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | token_16 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | token_32 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | token_32 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | token_8 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | base_to_correct | token_8 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | correct_to_base | after_think | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 152.2 |
| raw | smoke | correct_to_base | after_think | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 152.2 |
| raw | smoke | correct_to_base | after_tool_call | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 178.4 |
| raw | smoke | correct_to_base | after_tool_call | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 175.1 |
| raw | smoke | correct_to_base | token_0 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 0.0 |
| raw | smoke | correct_to_base | token_16 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 22.3 |
| raw | smoke | correct_to_base | token_16 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 19.8 |
| raw | smoke | correct_to_base | token_32 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 34.6 |
| raw | smoke | correct_to_base | token_32 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 34.5 |
| raw | smoke | correct_to_base | token_8 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 16.1 |
| raw | smoke | correct_to_base | token_8 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 | 13.4 |
| raw | smoke | correct_to_wrong | after_think | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 233.0 |
| raw | smoke | correct_to_wrong | after_think | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 200.7 |
| raw | smoke | correct_to_wrong | after_tool_call | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 233.0 |
| raw | smoke | correct_to_wrong | after_tool_call | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 | 218.5 |
| raw | smoke | correct_to_wrong | eos | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| raw | smoke | correct_to_wrong | token_0 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 27.2 |
| raw | smoke | correct_to_wrong | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 55.9 |
| raw | smoke | correct_to_wrong | token_16 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 38.9 |
| raw | smoke | correct_to_wrong | token_32 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 52.6 |
| raw | smoke | correct_to_wrong | token_32 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 45.5 |
| raw | smoke | correct_to_wrong | token_8 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 38.4 |
| raw | smoke | correct_to_wrong | token_8 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 29.1 |
| raw | smoke | wrong_to_correct | after_think | preserve | 8 | -12.50% | [-37.50%, 0.00%] | 1.0000 | 27.1 |
| raw | smoke | wrong_to_correct | after_think | recompute | 8 | -12.50% | [-37.50%, 0.00%] | 1.0000 | 27.1 |
| raw | smoke | wrong_to_correct | after_tool_call | preserve | 8 | -12.50% | [-37.50%, 0.00%] | 1.0000 | 27.1 |
| raw | smoke | wrong_to_correct | after_tool_call | recompute | 8 | -12.50% | [-37.50%, 0.00%] | 1.0000 | 27.1 |
| raw | smoke | wrong_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | N/A |
| raw | smoke | wrong_to_correct | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 | 26.6 |
| raw | smoke | wrong_to_correct | token_16 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 40.2 |
| raw | smoke | wrong_to_correct | token_32 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 28.1 |
| raw | smoke | wrong_to_correct | token_32 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 41.0 |
| raw | smoke | wrong_to_correct | token_8 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 | 35.3 |
| raw | smoke | wrong_to_correct | token_8 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 | 71.9 |

## Paired deltas where the switch actually fired

| Backend | Subset | Condition | Boundary | Cache | N | Delta | 95% CI | McNemar p |
|---|---|---|---|---|---:|---:|---:|---:|
| peft | full | correct_to_base | after_tool_call | preserve | 212 | -0.94% | [-3.30%, 1.42%] | 0.6875 |
| peft | full | correct_to_wrong | after_tool_call | preserve | 212 | -3.77% | [-7.08%, -0.47%] | 0.0574 |
| peft | full | wrong_to_correct | token_16 | recompute | 258 | -2.33% | [-5.81%, 0.78%] | 0.2632 |
| peft | pilot | base_to_correct | token_0 | preserve | 40 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | pilot | base_to_correct | token_8 | preserve | 40 | -15.00% | [-27.50%, -5.00%] | 0.0312 |
| peft | pilot | correct_to_base | after_tool_call | preserve | 29 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | pilot | correct_to_base | after_tool_call | recompute | 29 | -3.45% | [-10.34%, 0.00%] | 1.0000 |
| peft | pilot | correct_to_wrong | after_tool_call | preserve | 29 | 3.45% | [0.00%, 10.34%] | 1.0000 |
| peft | pilot | correct_to_wrong | after_tool_call | recompute | 29 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | pilot | correct_to_wrong | eos | preserve | 14 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | pilot | wrong_to_correct | token_16 | preserve | 40 | -5.00% | [-12.50%, 0.00%] | 0.5000 |
| peft | pilot | wrong_to_correct | token_16 | recompute | 40 | 2.50% | [-5.00%, 10.00%] | 1.0000 |
| peft | smoke | base_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | base_to_correct | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | base_to_correct | token_16 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | base_to_correct | token_32 | preserve | 5 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | base_to_correct | token_32 | recompute | 5 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | base_to_correct | token_8 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | base_to_correct | token_8 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | after_think | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_base | after_think | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_base | after_tool_call | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_base | after_tool_call | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_base | token_0 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | token_16 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | token_32 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | token_32 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | token_8 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_base | token_8 | recompute | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| peft | smoke | correct_to_wrong | after_think | preserve | 7 | 14.29% | [0.00%, 42.86%] | 1.0000 |
| peft | smoke | correct_to_wrong | after_think | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | after_tool_call | preserve | 7 | 14.29% | [0.00%, 42.86%] | 1.0000 |
| peft | smoke | correct_to_wrong | after_tool_call | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | eos | preserve | 3 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_16 | preserve | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_16 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_32 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_32 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_8 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | correct_to_wrong | token_8 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | after_think | preserve | 6 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | after_think | recompute | 6 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | after_tool_call | preserve | 6 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | after_tool_call | recompute | 6 | -16.67% | [-50.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_16 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_16 | recompute | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_32 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_32 | recompute | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_8 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| peft | smoke | wrong_to_correct | token_8 | recompute | 10 | 10.00% | [0.00%, 30.00%] | 1.0000 |
| raw | pilot | correct_to_base | after_tool_call | preserve | 29 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | pilot | correct_to_wrong | after_tool_call | preserve | 29 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | pilot | correct_to_wrong | eos | preserve | 12 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | pilot | wrong_to_correct | token_16 | recompute | 40 | 2.50% | [-5.00%, 10.00%] | 1.0000 |
| raw | smoke | base_to_correct | after_think | preserve | 1 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | base_to_correct | after_think | recompute | 1 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | base_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | base_to_correct | token_16 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | base_to_correct | token_16 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | base_to_correct | token_32 | preserve | 5 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | base_to_correct | token_32 | recompute | 5 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | base_to_correct | token_8 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | base_to_correct | token_8 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_base | after_think | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_base | after_think | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_base | after_tool_call | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_base | after_tool_call | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_base | token_0 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_base | token_16 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_base | token_16 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_base | token_32 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| raw | smoke | correct_to_base | token_32 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_base | token_8 | preserve | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_base | token_8 | recompute | 10 | -30.00% | [-60.00%, 0.00%] | 0.2500 |
| raw | smoke | correct_to_wrong | after_think | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | after_think | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | after_tool_call | preserve | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | after_tool_call | recompute | 8 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | eos | preserve | 3 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | token_0 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| raw | smoke | correct_to_wrong | token_16 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | token_32 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | token_32 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | token_8 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | correct_to_wrong | token_8 | recompute | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | after_think | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | after_think | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | after_tool_call | preserve | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | after_tool_call | recompute | 7 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | token_0 | preserve | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | token_16 | preserve | 10 | -20.00% | [-50.00%, 0.00%] | 0.5000 |
| raw | smoke | wrong_to_correct | token_16 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | token_32 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | token_32 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | token_8 | preserve | 10 | -10.00% | [-30.00%, 0.00%] | 1.0000 |
| raw | smoke | wrong_to_correct | token_8 | recompute | 10 | 0.00% | [0.00%, 0.00%] | 1.0000 |

## Decision gates

- Mechanically feasible: PASS
- Operationally useful: PASS
- Best full schedule: correct_to_base at after_tool_call (preserve)
- Behavioral 5-point margin, point estimate: PASS
- Behavioral 5-point margin, 95% CI: PASS

## Architecture recommendation

- Use pre-decode expert selection, with optional switch-off or handoff only after the tool call has been semantically committed.
- Do not treat the current D2L adapters as freely interchangeable token-level experts without joint switch-aware training.
