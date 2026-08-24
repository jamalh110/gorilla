# Final Findings: Mid-Decode D2L LoRA Switching

## Bottom line

The current D2L LoRAs are mechanically switchable during autoregressive
decoding, but they are not freely interchangeable token-level experts.

- **Go:** select the LoRA before decoding, and optionally switch it off after
  the tool call has been semantically committed.
- **Conditional:** hand off from one LoRA to another after commitment; the
  measured penalty is modest, but equivalence was not established.
- **No-go without additional training:** begin on the base or wrong expert and
  expect a later token-level switch to recover the correct trajectory.

## Mechanical result

- 30/30 custom greedy decodes exactly matched `model.generate()`.
- 100/100 correct-to-correct no-op switches were token exact.
- 0 execution errors, crashes, or invalid-logit failures occurred.
- 119/120 token-zero switches matched an always-on destination exactly; the
  one mismatch first diverged at token 164.
- PEFT switching took 23.9 ms for correct-to-base and 57.3 ms for
  correct-to-wrong in the full run. The largest measured switch-time fraction
  was 1.66% of generation latency.
- Raw D2L state switching took roughly 1.5-5.4 ms in the pilot.

## Behavioral results

### Early activation fails

On the 40-case PEFT pilot, base-to-correct at output token 8 lost 15.0 accuracy
points relative to always-correct LoRA:

- 95% paired bootstrap CI: [-27.5, -5.0] points
- Exact McNemar p-value: 0.03125

Switching at token zero reproduced the always-on destination. The failure is
therefore behavioral, not an adapter-loading artifact: the D2L LoRA changes
the decode trajectory from the start.

### Switching off after commitment works

On all 258 `live_simple` cases with a 256-token cap:

| Schedule | Accuracy | Delta vs always-correct | 95% CI |
|---|---:|---:|---:|
| Always-correct LoRA | 34.11% | — | — |
| Correct LoRA to base after `<tool_call>` | 33.33% | -0.78 pt | [-2.71, 1.16] |
| Correct LoRA to wrong LoRA after `<tool_call>` | 31.01% | -3.10 pt | [-5.81, -0.39] |
| Wrong LoRA to correct LoRA at token 16, recompute | 31.78% | -2.33 pt | [-5.81, 0.78] |

The `<tool_call>` switch fired on 212/258 cases. Restricted to those 212
actual switches:

- Correct-to-base: -0.94 point, 95% CI [-3.30, 1.42].
- Correct-to-wrong: -3.77 points, 95% CI [-7.08, -0.47].

The correct-to-base schedule satisfies the pre-registered 5-point practical
equivalence margin, including its 95% confidence interval. The arbitrary
LoRA-to-LoRA handoff does not satisfy that stricter confidence-interval test.

### Longer and constrained validations agree

The 256-token cap prevented some reference decodes from reaching
`<tool_call>`. A separate 40-case run with 1,024 output tokens reached the
boundary in all 40 cases:

- Always-correct accuracy: 20.0%.
- Correct-to-base accuracy: 20.0%.
- Paired delta: exactly 0.0 points.

With XGrammar constrained decoding on 40 cases:

- Always-correct: 35.0%.
- Correct-to-base after the JSON `"arguments"` key: 35.0% (0.0-point delta).
- Correct-to-wrong at that boundary: 32.5% (-2.5 points).
- Wrong-to-correct at token 16 with recomputation: 30.0% (-5.0 points).

## Raw D2L replication

The raw hypernetwork-generated LoRAs reproduced the qualitative PEFT result
on the 40-case pilot:

- Always-correct: 17.5%.
- Correct-to-base after `<tool_call>`: 17.5% (0.0-point delta).
- Correct-to-wrong after `<tool_call>`: 17.5% (0.0-point delta).
- Wrong-to-correct at token 16 with recomputation: 20.0% (+2.5 points,
  statistically inconclusive).

Raw and exported PEFT adapters had the same first-token argmax on all three
logit-parity cases. Their maximum first-step logit differences were
0.156-0.250, and small numerical differences accumulated into later token
divergence. Because full greedy sequences were not token exact, the raw backend
was not expanded to 258 cases; its behavioral pilot nevertheless matched the
PEFT conclusions.

## Recommendation

Use a **phase-gated architecture**:

1. Route to a D2L expert before the first generated token.
2. Keep that expert active through tool selection and structural commitment.
3. Optionally switch to the base model after `<tool_call>` or after the
   constrained JSON function name/`"arguments"` boundary.

Do not use the current independently generated D2L LoRAs as rapidly switching
token-level experts. To pursue that design, jointly train with randomized
expert transitions and explicit cache semantics, then repeat this matrix.

Primary artifacts:

- `report.md` and `summary.json`: combined main evaluation
- `accuracy_by_boundary.png` and `logprob_jump_by_boundary.png`: plots
- `../middecode_switch_constrained/report.md`: constrained pilot
- `../middecode_switch_long/report.md`: 1,024-token validation
