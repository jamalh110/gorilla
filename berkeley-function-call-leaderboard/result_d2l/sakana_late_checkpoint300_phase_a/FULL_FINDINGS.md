# Sakana LoRA-On Late-Schema Results

## Protocol

- Model: `Qwen/Qwen3-4B-Instruct-2507`, with thinking disabled.
- Hypernetwork initialization: Sakana QA checkpoint
  `trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin`.
- Retuned checkpoint:
  `train_outputs/runs/Jul23_17-24-52_lambda1.cs.cornell.edu_f6fe4f09/checkpoint-300/pytorch_model.bin`.
- Dataset: all 200 BFCL `multiple` examples.
- Routing and binding are greedy and unconstrained.
- The generated all-tools LoRA remains attached for both turns.
- Binding uses the selected original BFCL schema in a Qwen tool response after
  the router call.
- Phase A recomputes the exact `P || R || S` transcript and asserts token-prefix
  identity. It does not use a fresh schema-plus-query bind prompt.

## Data and training

- v6 late-schema rows: 150,799 train and 3,113 validation.
- Structural validation: zero malformed rows.
- Over-length rows skipped by training: six train and one validation.
- Objectives: router CE `1.5`, bind CE `1.0`, bind-only base-teacher KL `0.5`,
  generated-LoRA L1 `0.05`.
- The initial run reached step 300 but ended during its final validation before
  saving. Training was resumed from checkpoint 250 with final validation
  disabled, and checkpoint 300 was saved successfully.

## BFCL results

| Condition | Correct | Accuracy | Route | Conditional bind | Malformed |
|---|---:|---:|---:|---:|---:|
| Retuned, generated route, LoRA on | 183/200 | 91.5% | 198/200 | 183/198 (92.42%) | 0 |
| Retuned, oracle route, LoRA on | 184/200 | 92.0% | 200/200 | 184/200 | 0 |
| Oracle route, base/LoRA off | 182/200 | 91.0% | 200/200 | 182/200 | 1 |
| Prior Variant B, exact Phase A | 177/200 | 88.5% | 199/200 | 177/199 (88.94%) | 1 |

Every retuned generated-route attempt recorded `constraint_mode: none`.
No router or binder output contained thinking tags.

Historical, protocol-different comparators:

- Prior Variant B with its old fresh bind prompt: 166/200 (83.0%).
- Native raw all-tools Qwen FC: 189/200 (94.5%).
- Native raw oracle-single-tool Qwen FC: 188/200 (94.0%).

The retune improves the prior checkpoint by 3.0 points under the same exact
Phase A evaluator and by 8.5 points over the historical old-prompt score.

## Non-interference

With identical oracle router calls and raw late-schema transcripts:

- LoRA on: 184/200.
- LoRA off: 182/200.
- Paired outcomes: seven LoRA-on wins, five LoRA-on losses, 177 correct ties,
  and 11 wrong ties.
- Exact final output matches: 120/200.

The attached LoRA does not produce a material correctness regression under this
protocol; it improves net correctness by two examples. The remaining shortfall
is primarily the late-schema binder ceiling, not routing.

## Failure classes

Retuned generated-route failures:

- 9 string-value errors.
- 3 other value errors.
- 2 missing-optional errors.
- 2 wrong-function errors; these are the two routing misses.
- 1 list/tuple value error.

Oracle-base failures are similar: nine string-value errors, three
missing-optional errors, four other value errors, one list/tuple error, and one
malformed JSON output.

## Gate decision

The required Phase A gate is 188/200 (94.0%). The retuned model reached 183/200,
and the exact-transcript oracle-base ceiling reached only 182/200. Therefore the
quality gate was not met. Per the approved plan, the stateful KV continuation
evaluation is not promoted or run as the final path. The next iteration must
improve the late-schema binding wrapper/curriculum before Phase B parity and
full stateful evaluation are justified.
