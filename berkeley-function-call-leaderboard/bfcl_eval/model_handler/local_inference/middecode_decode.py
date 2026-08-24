"""Instrumented greedy decoding with one adapter transition.

This module intentionally has no BFCL dependencies so the local inference
workers can import it from the separate D2L virtual environment.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

AdapterState = str | None
CachePolicy = Literal["preserve", "recompute"]
ActivateAdapter = Callable[[AdapterState], dict[str, Any] | None]


def _synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _activate_timed(torch, device, activate: ActivateAdapter, state: AdapterState):
    _synchronize(torch, device)
    started = time.perf_counter()
    metadata = activate(state) or {}
    _synchronize(torch, device)
    return {
        "adapter": state,
        "latency": time.perf_counter() - started,
        **metadata,
    }


def _forward(model, input_ids, attention_mask, past_key_values=None):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "return_dict": True,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    return model(**kwargs)


def greedy_decode_with_switch(
    *,
    model,
    input_ids,
    attention_mask,
    activate_adapter: ActivateAdapter,
    start_adapter: AdapterState,
    end_adapter: AdapterState,
    switch_at: int | None,
    cache_policy: CachePolicy,
    max_new_tokens: int,
    eos_token_id: int | list[int] | tuple[int, ...] | None,
    tokenizer=None,
    stop_text: str | None = None,
    top_k: int = 5,
    return_first_logits: bool = False,
    logits_processors=None,
    prefill_adapter: AdapterState = None,
    replay_last_prompt_token: bool = False,
    replay_prompt_tokens: int = 0,
) -> dict[str, Any]:
    """Greedily decode a batch of one and optionally switch adapters.

    ``switch_at`` is an output-token boundary.  For example, ``switch_at=8``
    means output tokens 0..7 are selected under ``start_adapter`` and token 8
    is selected after activating ``end_adapter``.

    With ``cache_policy="preserve"``, K/V entries for earlier positions remain
    as they were computed while the old adapter was active.  The final emitted
    prefix token is processed under the new adapter to produce logits for the
    first post-switch token.  With ``"recompute"``, the complete prompt and
    emitted prefix are re-forwarded under the new adapter at the boundary.
    """

    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("mid-decode switching currently supports batch size 1")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if switch_at is not None and switch_at < 0:
        raise ValueError("switch_at must be non-negative or None")
    replay_count = int(replay_prompt_tokens)
    if replay_last_prompt_token and replay_count == 0:
        replay_count = 1
    replay_enabled = replay_count > 0
    if replay_count < 0:
        raise ValueError("replay_prompt_tokens must be non-negative")
    if replay_enabled and switch_at is not None:
        raise ValueError("prompt replay cannot be combined with a decode switch")
    if replay_enabled and replay_count >= input_ids.shape[1]:
        raise ValueError("prompt replay must leave at least one base-prefilled token")
    if cache_policy not in ("preserve", "recompute"):
        raise ValueError(f"unsupported cache policy: {cache_policy}")

    device = input_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    activation_events = []
    initial_adapter = prefill_adapter if replay_enabled else start_adapter
    initial_event = _activate_timed(
        torch, device, activate_adapter, initial_adapter
    )
    initial_event.update({"kind": "initial", "output_token_index": 0})
    activation_events.append(initial_event)
    active_adapter = initial_adapter

    switch_applied = False
    if switch_at == 0 and not replay_enabled:
        switch_event = _activate_timed(
            torch, device, activate_adapter, end_adapter
        )
        switch_event.update({"kind": "switch", "output_token_index": 0})
        activation_events.append(switch_event)
        active_adapter = end_adapter
        switch_applied = True

    started = time.perf_counter()
    if replay_enabled:
        prefix_ids = input_ids[:, :-replay_count]
        prefix_attention_mask = attention_mask[:, :-replay_count]
        with torch.inference_mode():
            prefix_outputs = _forward(
                model,
                prefix_ids,
                prefix_attention_mask,
            )
        replay_event = _activate_timed(
            torch, device, activate_adapter, start_adapter
        )
        replay_event.update(
            {"kind": "post_prefill_switch", "output_token_index": 0}
        )
        activation_events.append(replay_event)
        active_adapter = start_adapter
        switch_applied = True
        with torch.inference_mode():
            outputs = _forward(
                model,
                input_ids[:, -replay_count:],
                attention_mask,
                past_key_values=prefix_outputs.past_key_values,
            )
    else:
        with torch.inference_mode():
            outputs = _forward(model, input_ids, attention_mask)
    prefill_latency = time.perf_counter() - started
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    first_logits = (
        next_token_logits[0].float().cpu().tolist()
        if return_first_logits
        else None
    )

    full_ids = input_ids
    full_attention_mask = attention_mask
    generated_ids: list[int] = []
    selected_logprobs: list[float] = []
    boundary_topk: list[dict[str, Any]] = []
    decode_started = time.perf_counter()
    stopped_reason = "max_new_tokens"

    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = {int(token_id) for token_id in eos_token_id}

    for step in range(max_new_tokens):
        processed_logits = next_token_logits
        for processor in logits_processors or ():
            processed_logits = processor(full_ids, processed_logits)

        invalid = torch.isnan(processed_logits) | torch.isposinf(processed_logits)
        if bool(invalid.any()) or not bool(torch.isfinite(processed_logits).any()):
            bad_count = int(invalid.sum().item())
            raise FloatingPointError(
                f"invalid logits at output token {step}: {bad_count}"
            )

        logprobs = torch.log_softmax(processed_logits.float(), dim=-1)
        next_token = torch.argmax(processed_logits, dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        selected_logprobs.append(float(logprobs[0, token_id].item()))

        if (
            step == 0
            or switch_at is not None
            and abs(step - switch_at) <= 1
        ):
            k = min(top_k, logprobs.shape[-1])
            values, indices = torch.topk(logprobs[0], k=k)
            boundary_topk.append(
                {
                    "output_token_index": step,
                    "active_adapter": active_adapter,
                    "token_ids": [int(item) for item in indices.cpu().tolist()],
                    "logprobs": [float(item) for item in values.cpu().tolist()],
                }
            )

        next_token_column = next_token[:, None]
        full_ids = torch.cat((full_ids, next_token_column), dim=1)
        full_attention_mask = torch.cat(
            (
                full_attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=full_attention_mask.dtype,
                    device=device,
                ),
            ),
            dim=1,
        )

        if token_id in eos_ids:
            stopped_reason = "eos"
            break
        if stop_text and tokenizer is not None:
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
            if stop_text in generated_text:
                stopped_reason = "stop_text"
                break

        next_step = step + 1
        switch_now = switch_at is not None and next_step == switch_at
        if switch_now:
            switch_event = _activate_timed(
                torch, device, activate_adapter, end_adapter
            )
            switch_event.update(
                {"kind": "switch", "output_token_index": next_step}
            )
            activation_events.append(switch_event)
            active_adapter = end_adapter
            switch_applied = True

        with torch.inference_mode():
            if switch_now and cache_policy == "recompute":
                outputs = _forward(model, full_ids, full_attention_mask)
            else:
                outputs = _forward(
                    model,
                    next_token_column,
                    full_attention_mask,
                    past_key_values=past_key_values,
                )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    _synchronize(torch, device)
    decode_latency = time.perf_counter() - decode_started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    result = {
        "token_ids": generated_ids,
        "selected_logprobs": selected_logprobs,
        "boundary_topk": boundary_topk,
        "activation_events": activation_events,
        "switch_at": switch_at,
        "switch_applied": switch_applied,
        "start_adapter": start_adapter,
        "end_adapter": end_adapter,
        "final_adapter": active_adapter,
        "cache_policy": cache_policy,
        "prefill_adapter": prefill_adapter if replay_enabled else None,
        "replay_last_prompt_token": replay_count == 1,
        "replay_prompt_tokens": replay_count,
        "prefill_latency": prefill_latency,
        "decode_latency": decode_latency,
        "peak_memory_bytes": peak_memory,
        "stopped_reason": stopped_reason,
    }
    if first_logits is not None:
        result["first_logits"] = first_logits
    return result
