import unittest
from types import SimpleNamespace

import torch

from bfcl_eval.model_handler.local_inference.middecode_decode import (
    greedy_decode_with_switch,
)
from bfcl_eval.model_handler.local_inference.bfcl_tool_schema import tools_hash
from run_middecode_switch_eval import choose_alternates


class FakeCausalModel:
    """Tiny state-sensitive model used to exercise cache semantics."""

    def __init__(self, active_state):
        self.active_state = active_state

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        return_dict,
        past_key_values=None,
    ):
        del attention_mask, use_cache, return_dict
        offset = {"a": 0, "b": 3, None: 7}[self.active_state["value"]]
        total = 0 if past_key_values is None else past_key_values
        logits = []
        for token_id in input_ids[0].tolist():
            total += token_id + offset
            row = torch.full((11,), -10.0)
            row[total % 11] = 10.0
            logits.append(row)
        return SimpleNamespace(
            logits=torch.stack(logits)[None, :, :],
            past_key_values=total,
        )


def decode(active_state, start, end, switch_at, cache_policy):
    def activate(value):
        active_state["value"] = value
        return {"fake": True}

    return greedy_decode_with_switch(
        model=FakeCausalModel(active_state),
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        activate_adapter=activate,
        start_adapter=start,
        end_adapter=end,
        switch_at=switch_at,
        cache_policy=cache_policy,
        max_new_tokens=5,
        eos_token_id=None,
        top_k=3,
    )


class MidDecodeLoopTests(unittest.TestCase):
    def test_no_op_switch_preserves_exact_tokens(self):
        baseline = decode({"value": None}, "a", "a", None, "preserve")
        no_op = decode({"value": None}, "a", "a", 2, "preserve")
        self.assertEqual(baseline["token_ids"], no_op["token_ids"])
        self.assertTrue(no_op["switch_applied"])

    def test_cache_policies_are_intentionally_distinct(self):
        preserved = decode({"value": None}, "a", "b", 2, "preserve")
        recomputed = decode({"value": None}, "a", "b", 2, "recompute")
        self.assertNotEqual(preserved["token_ids"], recomputed["token_ids"])

    def test_zero_boundary_uses_end_adapter_for_prefill(self):
        switched = decode({"value": None}, "a", "b", 0, "preserve")
        always_b = decode({"value": None}, "b", "b", None, "preserve")
        self.assertEqual(switched["token_ids"], always_b["token_ids"])

    def test_invalid_cache_policy_fails_fast(self):
        with self.assertRaises(ValueError):
            decode({"value": None}, "a", "b", 2, "invalid")

    def test_logits_processor_may_mask_with_negative_infinity(self):
        active_state = {"value": None}

        def activate(value):
            active_state["value"] = value

        def force_four(input_ids, scores):
            del input_ids
            masked = torch.full_like(scores, -torch.inf)
            masked[:, 4] = 0
            return masked

        result = greedy_decode_with_switch(
            model=FakeCausalModel(active_state),
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            activate_adapter=activate,
            start_adapter="a",
            end_adapter="a",
            switch_at=None,
            cache_policy="preserve",
            max_new_tokens=3,
            eos_token_id=None,
            logits_processors=[force_four],
        )
        self.assertEqual(result["token_ids"], [4, 4, 4])

    def test_replayed_last_prompt_token_uses_hybrid_cache(self):
        active_state = {"value": None}

        def activate(value):
            active_state["value"] = value

        result = greedy_decode_with_switch(
            model=FakeCausalModel(active_state),
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            activate_adapter=activate,
            start_adapter="b",
            end_adapter="b",
            switch_at=None,
            cache_policy="preserve",
            max_new_tokens=1,
            eos_token_id=None,
            prefill_adapter=None,
            replay_last_prompt_token=True,
        )
        # Prefix token 1 under base(None): 1 + 7 = 8. Final prompt token 2
        # under adapter b: 2 + 3, producing argmax (8 + 5) % 11 == 2.
        self.assertEqual(result["token_ids"], [2])
        self.assertEqual(
            [event["adapter"] for event in result["activation_events"]],
            [None, "b"],
        )
        self.assertTrue(result["switch_applied"])

    def test_replayed_prompt_suffix_uses_requested_token_count(self):
        active_state = {"value": None}

        def activate(value):
            active_state["value"] = value

        result = greedy_decode_with_switch(
            model=FakeCausalModel(active_state),
            input_ids=torch.tensor([list(range(1, 11))]),
            attention_mask=torch.ones((1, 10), dtype=torch.long),
            activate_adapter=activate,
            start_adapter="b",
            end_adapter="b",
            switch_at=None,
            cache_policy="preserve",
            max_new_tokens=1,
            eos_token_id=None,
            prefill_adapter=None,
            replay_prompt_tokens=8,
        )
        # Tokens 1..2 use base offset 7; tokens 3..10 use adapter-b offset 3:
        # (1 + 7) + (2 + 7) + sum(3..10) + 8 * 3 = 93, and 93 % 11 == 5.
        self.assertEqual(result["token_ids"], [5])
        self.assertEqual(result["replay_prompt_tokens"], 8)
        self.assertFalse(result["replay_last_prompt_token"])


class AdapterPairingTests(unittest.TestCase):
    def test_same_name_different_schema_is_preferred(self):
        first = [
            {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ]
        second = [
            {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"zip": {"type": "integer"}},
                },
            }
        ]
        entries = [
            {"id": "one", "function": first},
            {"id": "two", "function": second},
        ]
        first_hash = tools_hash(first)
        second_hash = tools_hash(second)
        alternates = choose_alternates(entries, {first_hash, second_hash})
        self.assertEqual(
            alternates[first_hash],
            (second_hash, "same_name_different_schema"),
        )


if __name__ == "__main__":
    unittest.main()
