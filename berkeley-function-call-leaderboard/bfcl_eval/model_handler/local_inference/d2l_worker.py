"""
Subprocess worker for Doc-to-LoRA inference.

Runs inside D2L's own virtualenv so that D2L and BFCL dependencies never
coexist in the same process.  Communicates with the parent BFCL handler
via a JSON-lines protocol over stdin (requests) and a saved copy of the
original stdout file descriptor (responses).  fd 1 is redirected to stderr
so that any print output from D2L / PyTorch / CUDA does not corrupt the
protocol channel.

Usage (called by DocToLoraHandler, not invoked manually):
    <d2l-venv-python> d2l_worker.py <d2l-src-path>
"""

import json
import os
import sys
import uuid
from hashlib import sha256
from math import ceil

# ---------------------------------------------------------------------------
# Redirect stdout → stderr so all print() / C-level stdout goes to the
# terminal, not the protocol pipe.  Save the real stdout fd for JSON IPC.
# ---------------------------------------------------------------------------
_real_stdout_fd = os.dup(1)
os.dup2(2, 1)
_proto = os.fdopen(_real_stdout_fd, "w", buffering=1)  # line-buffered
sys.stdout = sys.stderr


def _send(obj: dict):
    _proto.write(json.dumps(obj) + "\n")
    _proto.flush()


# ---------------------------------------------------------------------------
# D2L imports (deferred until main() so sys.path is set first)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_current_tools: list[dict] | None = None
_cached_loras: dict[str, dict] = {}
_active_lora_name: str | None = None
_sessions: dict[str, dict] = {}


def _load_model(checkpoint_path: str):
    global _model, _tokenizer, _cached_loras, _active_lora_name, _sessions
    import torch
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    print(f"[d2l_worker] Loading model from {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, weights_only=False)
    # Must match the chunk_scaling the checkpoint was trained with, since it
    # sets how strongly a stacked multi-chunk adapter perturbs the base model.
    chunk_scaling = os.environ.get("D2L_CHUNK_SCALING", "none")
    _model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_sequence_packing=False,
        chunk_scaling=chunk_scaling,
    )
    print(f"[d2l_worker] chunk_scaling={chunk_scaling}")
    _model.reset()
    _tokenizer = get_tokenizer(_model.base_model.name_or_path)
    _cached_loras = {}
    _active_lora_name = None
    _sessions = {}
    name = _model.base_model.name_or_path
    print(f"[d2l_worker] Model loaded (base: {name})")
    return {"base_model_name": name}


def _internalize(
    tool_defs: str,
    chunk_size: int = 1024,
    ctx_chunk_mode: str = "none",
    tools_per_chunk: int = 1,
):
    global _current_tools, _active_lora_name, _sessions
    import torch
    from ctx_to_lora.data.definitions import CTX_AFFIXES
    from ctx_to_lora.data.processing import (
        tokenize_ctx_per_tool,
        tokenize_ctx_text,
    )
    from ctx_to_lora.model_loading import get_tokenizer

    _current_tools = json.loads(tool_defs)
    _active_lora_name = None
    _sessions = {}
    _model.reset()

    ctx_tokenizer = get_tokenizer(_model.ctx_encoder.base_model.name_or_path)

    # D2L_ORDER_ENSEMBLE=K: internalise the WHOLE catalogue K times under K
    # different orderings and rank-concatenate, so the ensemble costs one scoring
    # pass per query instead of K. Use with D2L_CHUNK_SCALING=mean -- rank
    # concatenation SUMS the deltas, so K full-strength adapters would perturb
    # the base model K times too hard.
    _order_k = int(os.environ.get("D2L_ORDER_ENSEMBLE", "0") or 0)
    if _order_k > 1:
        chunks = _order_ensemble_chunks(tool_defs, _order_k, ctx_tokenizer)
        print(f"[d2l] order-ensemble k={_order_k} "
              f"ctx_tokens={[len(c) for c in chunks]}", file=sys.stderr, flush=True)
        return _generate_stacked_loras(tool_defs, chunks, _order_k)

    if ctx_chunk_mode == "per_tool":
        # Whole schemas per chunk, matching how the checkpoint was trained. Each
        # chunk already carries the full chat template, so the affix stitching
        # used by the token-boundary path does not apply.
        chunks = tokenize_ctx_per_tool(
            dict(context=[tool_defs]),
            ctx_tokenizer,
            tools_per_chunk=tools_per_chunk,
        )["ctx_ids"][0]
        n_chunks = len(chunks)
        if n_chunks > 1:
            return _generate_stacked_loras(tool_defs, chunks, n_chunks)
        _model.internalize(tool_defs)
        _model._n_ctx_chunks = 1
        _active_lora_name = (
            "internalized:" + sha256(tool_defs.encode("utf-8")).hexdigest()[:16]
        )
        return {}

    ctx_ids_nested = tokenize_ctx_text(
        dict(context=[tool_defs]), ctx_tokenizer
    )["ctx_ids"]
    ctx_ids = ctx_ids_nested[0]

    n_chunks = max(1, ceil(len(ctx_ids) / chunk_size))
    # ctx_chunk_mode='none' still splits and SUMS above chunk_size, so a run can
    # be silently multi-chunk. Log it: whether the catalogue was one chunk is
    # the difference between matching training and not.
    print(
        f"[d2l] internalize ctx_tokens={len(ctx_ids)} chunk_size={chunk_size} "
        f"n_chunks={n_chunks}",
        file=sys.stderr,
        flush=True,
    )

    if n_chunks == 1:
        _model.internalize(tool_defs)
        _model._n_ctx_chunks = 1
        _active_lora_name = (
            "internalized:" + sha256(tool_defs.encode("utf-8")).hexdigest()[:16]
        )
        return {}

    avg_len = ceil(len(ctx_ids) / n_chunks)
    chunks = [ctx_ids[i : i + avg_len] for i in range(0, len(ctx_ids), avg_len)]

    model_name = _model.base_model.config.name_or_path
    if model_name not in CTX_AFFIXES:
        _model.internalize(tool_defs)
        _model._n_ctx_chunks = 1
        return {}

    affixes = CTX_AFFIXES[model_name]
    chunks[0] = chunks[0] + affixes["suffix"]
    for i in range(1, len(chunks) - 1):
        chunks[i] = affixes["prefix"] + chunks[i] + affixes["suffix"]
    chunks[-1] = affixes["prefix"] + chunks[-1]

    return _generate_stacked_loras(tool_defs, chunks, n_chunks)


def _order_ensemble_chunks(tool_defs: str, k: int, ctx_tokenizer):
    """K reorderings of the WHOLE catalogue, tokenised for rank-concatenation.

    Score ensembling averages in OUTPUT space -- K forward passes, each with ctx
    and enum in the same order, then sum the candidate scores. That costs K
    scoring passes on EVERY query. Concatenating K adapters instead averages in
    WEIGHT space and costs one scoring pass per query, with the K hypernetwork
    passes cached per catalogue. Not the same operation: the model is nonlinear
    in its weights.

    Known risk, specific to anon: the composed adapter then encodes each tool at
    K DIFFERENT slots while the prompt's enum lists it at one. The ctx-index to
    enum-index correspondence that anon routing depends on holds for at most one
    component. Ordering 0 is left as the identity so that correspondence survives
    for one of them; the rest contribute order-independent content.
    """
    import json as _json
    import random as _random

    # tokenize_ctx_text is imported function-locally by the internalize path, so
    # it is NOT visible at module scope. Importing it here too -- omitting this
    # raised NameError on every call, which is why every order-concat run
    # produced 0 rows.
    from ctx_to_lora.data.processing import tokenize_ctx_text

    tools = _json.loads(tool_defs)
    out = []
    for i in range(k):
        if i == 0:
            perm = list(tools)
        else:
            perm = list(tools)
            _random.Random(f"orderens:{i}").shuffle(perm)
        ids = tokenize_ctx_text(
            dict(context=[_json.dumps(perm, ensure_ascii=False)]), ctx_tokenizer
        )["ctx_ids"][0]
        out.append(ids)
    return out


def _generate_stacked_loras(tool_defs: str, chunks: list, n_chunks: int):
    """Run the hypernetwork per chunk and concatenate along the rank axis."""
    global _active_lora_name
    import torch

    all_loras = []
    for chunk in chunks:
        ids = torch.tensor([chunk], device=_model.device)
        mask = torch.ones_like(ids)
        loras, _ = _model.generate_weights(ids, mask)
        all_loras.append(loras)

    merged = {}
    for module in all_loras[0]:
        merged[module] = {
            "A": torch.cat([l[module]["A"] for l in all_loras], dim=0),
            "B": torch.cat([l[module]["B"] for l in all_loras], dim=0),
        }

    # Store raw concatenated weights — combine_lora is deferred to
    # generate() (which always combines with bias) or _export_peft.
    _model.generated_loras = merged
    _model._n_ctx_chunks = n_chunks
    _model.patch_lora_forward()
    _active_lora_name = (
        "internalized:" + sha256(tool_defs.encode("utf-8")).hexdigest()[:16]
    )
    return {}


def _internalize_no_grad(tool_defs, chunk_size=1024, ctx_chunk_mode="none",
                         tools_per_chunk=1):
    """_internalize under inference_mode. See the dispatch-table comment."""
    import torch

    with torch.inference_mode():
        return _internalize(tool_defs, chunk_size, ctx_chunk_mode, tools_per_chunk)


def _preload_loras(adapters: dict[str, str], chunk_size: int = 1024):
    """Generate and retain multiple D2L LoRA states for cheap activation."""
    global _cached_loras, _current_tools, _active_lora_name
    import torch

    if _model is None:
        raise RuntimeError("Call load_model first")
    for name, tool_defs in sorted(adapters.items()):
        if not name:
            raise ValueError("adapter names must be non-empty")
        if name in _cached_loras:
            continue
        print(f"[d2l_worker] Internalizing cached LoRA {name}")
        # Hypernet outputs otherwise retain a full autograd graph per cached
        # adapter, which quickly exhausts GPU memory when preloading experts.
        with torch.inference_mode():
            _internalize(tool_defs, chunk_size)
        detached_loras = {
            module_name: {
                matrix_name: matrix.detach()
                for matrix_name, matrix in matrices.items()
            }
            for module_name, matrices in _model.generated_loras.items()
        }
        _model.generated_loras = detached_loras
        _cached_loras[name] = {
            "generated_loras": detached_loras,
            "n_ctx_chunks": int(getattr(_model, "_n_ctx_chunks", 1)),
            "tools": _current_tools,
        }

    _model.activate_generated_lora(None)
    _current_tools = None
    _active_lora_name = None
    return {"loaded": sorted(_cached_loras)}


def _activate_lora(adapter_name: str | None):
    """Activate a cached raw D2L state, or the frozen base model."""
    global _current_tools, _active_lora_name

    if _model is None:
        raise RuntimeError("Call load_model first")
    if adapter_name is None:
        _model.activate_generated_lora(None)
        _current_tools = None
        _active_lora_name = None
        return {"n_ctx_chunks": 0}
    if adapter_name not in _cached_loras:
        raise KeyError(f"LoRA {adapter_name!r} has not been preloaded")

    state = _cached_loras[adapter_name]
    _model.activate_generated_lora(
        state["generated_loras"],
        state["n_ctx_chunks"],
    )
    _current_tools = state["tools"]
    _active_lora_name = adapter_name
    return {"n_ctx_chunks": state["n_ctx_chunks"]}


def _get_model_info():
    """Return model metadata needed for PEFT adapter conversion."""
    return {
        "base_model_name": _model.base_model.name_or_path,
        "lora_alpha": float(_model.peft_config.lora_alpha),
        "lora_r": int(_model.peft_config.r),
        "target_modules": list(_model.hypernet.target_modules),
        "layer_indices": [int(i) for i in _model.hypernet.layer_indices],
        "n_layers": len(_model.hypernet.layer_indices),
        "use_bias": bool(_model.hypernet.config.use_bias),
    }


def _export_peft(output_dir: str):
    """Save the currently internalized LoRA as a PEFT adapter on disk.

    Replicates the combine_lora step that ``generate()`` applies before
    decoding so the exported weights exactly match inference behaviour.
    """
    import torch
    from ctx_to_lora.modeling.lora_merger import combine_lora
    from safetensors.torch import save_file

    if _model.generated_loras is None:
        raise RuntimeError("No internalized LoRA to export. Call internalize first.")

    info = _get_model_info()

    n_ctx = getattr(_model, "_n_ctx_chunks", 1)
    n_ctx_chunks = torch.tensor([n_ctx], device=_model.device)
    lora_bias = (
        _model.hypernet.get_head_bias()
        if _model.hypernet.config.use_bias
        else None
    )
    loras = combine_lora(_model.generated_loras, n_ctx_chunks, lora_bias=lora_bias)

    state_dict = {}
    for module_name in loras:
        A = loras[module_name]["A"]  # [1, n_layers, merged_rank, d_in]
        B = loras[module_name]["B"]  # [1, n_layers, merged_rank, d_out]
        merged_rank = A.shape[2]

        if module_name in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"):
            long_mname = f"self_attn.{module_name}"
        elif module_name in ("down_proj", "up_proj", "gate_proj"):
            long_mname = f"mlp.{module_name}"
        else:
            long_mname = module_name

        for layer_idx in info["layer_indices"]:
            pfx = f"base_model.model.model.layers.{layer_idx}.{long_mname}"
            state_dict[f"{pfx}.lora_A.weight"] = (
                A[0, layer_idx].contiguous().to(torch.float16)
            )
            state_dict[f"{pfx}.lora_B.weight"] = (
                B[0, layer_idx].T.contiguous().to(torch.float16)
            )

    # D2L scaling: delta *= lora_alpha
    # PEFT scaling: delta *= (lora_alpha / r)
    peft_alpha = info["lora_alpha"] * merged_rank

    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": info["base_model_name"],
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": peft_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": int(merged_rank),
        "revision": None,
        "target_modules": list(info["target_modules"]),
        "task_type": "CAUSAL_LM",
    }

    os.makedirs(output_dir, exist_ok=True)
    save_file(state_dict, os.path.join(output_dir, "adapter_model.safetensors"))
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    return {"output_dir": output_dir, "merged_rank": int(merged_rank)}


def _internalize_and_export(tool_defs: str, output_dir: str, chunk_size: int = 1024):
    """Internalize tools and save the resulting LoRA as a PEFT adapter."""
    _internalize(tool_defs, chunk_size)
    return _export_peft(output_dir)


def _render_chat_ids(
    messages: list,
    enable_thinking: bool | None,
    *,
    add_generation_prompt: bool = True,
):
    template_kwargs = dict(
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
    )
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    try:
        chat_ids = _tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        chat_ids = _tokenizer.apply_chat_template(messages, **template_kwargs)
    return chat_ids.to(_model.device)


def _token_hash(token_ids) -> str:
    values = [int(token_id) for token_id in token_ids]
    return sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _late_schema_payload(raw_function: dict) -> str:
    return json.dumps(
        {
            "selected_tool_schema": {
                "type": "function",
                "function": raw_function,
            }
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _render_late_transcript(
    stage1_messages: list,
    router_token_ids: list[int],
    raw_function: dict,
    enable_thinking: bool | None = False,
):
    """Render P || R || S and prove the full transcript preserves P and R."""
    if not router_token_ids:
        raise ValueError("router_token_ids must be non-empty")
    p_ids = _render_chat_ids(
        stage1_messages,
        enable_thinking,
        add_generation_prompt=True,
    )[0].tolist()
    r_ids = [int(token_id) for token_id in router_token_ids]
    router_text = _tokenizer.decode(r_ids, skip_special_tokens=False)
    forbidden = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
    if any(marker in router_text for marker in forbidden):
        raise ValueError("router output contains a conversation-boundary token")

    payload = _late_schema_payload(raw_function)
    transcript_messages = [
        *stage1_messages,
        {"role": "assistant", "content": router_text},
        {"role": "tool", "content": payload},
    ]
    full_ids = _render_chat_ids(
        transcript_messages,
        enable_thinking,
        add_generation_prompt=True,
    )[0].tolist()
    prefix_ids = p_ids + r_ids
    if full_ids[: len(prefix_ids)] != prefix_ids:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(full_ids, prefix_ids)
                )
                if actual != expected
            ),
            min(len(full_ids), len(prefix_ids)),
        )
        raise ValueError(
            "late transcript is not token-prefix equivalent at token "
            f"{mismatch}: full={full_ids[mismatch:mismatch + 4]!r}, "
            f"expected={prefix_ids[mismatch:mismatch + 4]!r}"
        )
    s_ids = full_ids[len(prefix_ids) :]
    if p_ids + r_ids + s_ids != full_ids:
        raise AssertionError("P || R || S token identity failed")
    return {
        "messages": transcript_messages,
        "router_text": router_text,
        "schema_payload": payload,
        "p_ids": p_ids,
        "r_ids": r_ids,
        "s_ids": s_ids,
        "full_ids": full_ids,
        "p_hash": _token_hash(p_ids),
        "r_hash": _token_hash(r_ids),
        "s_hash": _token_hash(s_ids),
        "full_hash": _token_hash(full_ids),
    }


def _decode_response(response_ids) -> str:
    response_text = _tokenizer.decode(response_ids, skip_special_tokens=False)
    for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        response_text = response_text.replace(tag, "")
    return response_text.strip()


def _semantic_boundaries(response_ids) -> dict[str, int | None]:
    token_ids = [int(token_id) for token_id in response_ids]
    result = {}
    for name, marker in (
        ("after_think", "</think>"),
        ("after_tool_call", "<tool_call>"),
        ("after_arguments_key", '"arguments"'),
    ):
        result[name] = next(
            (
                index
                for index in range(1, len(token_ids) + 1)
                if marker
                in _tokenizer.decode(
                    token_ids[:index],
                    skip_special_tokens=False,
                )
            ),
            None,
        )
    result["eos"] = len(token_ids)
    return result


def _generate(
    messages: list,
    max_new_tokens: int = 1024,
    temperature: float = 0,
    restrict_toolgen: bool = False,
    constraint_tools: list[dict] | None = None,
    assistant_prefix: str = "",
    enable_thinking: bool | None = None,
    stop_after_first_tool_call: bool = False,
    strict_json_schema: bool = False,
):
    import torch
    from transformers import StoppingCriteria

    template_kwargs = dict(
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors="pt",
    )
    prompt_messages = list(messages)
    if assistant_prefix:
        prompt_messages.append({"role": "assistant", "content": assistant_prefix})
        template_kwargs["add_generation_prompt"] = False
        template_kwargs["continue_final_message"] = True
    else:
        template_kwargs["add_generation_prompt"] = True
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    chat_ids = _tokenizer.apply_chat_template(
        prompt_messages,
        **template_kwargs,
    ).to(_model.device)

    input_token_count = chat_ids.shape[1]

    class _StopAfterFirstToolCall(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, input_token_count:]
            text = _tokenizer.decode(generated, skip_special_tokens=False)
            return "</tool_call>" in text

    n_ctx = getattr(_model, "_n_ctx_chunks", 1)
    generate_kwargs = {
        "n_ctx_chunks": torch.tensor([n_ctx], device=_model.device),
        "input_ids": chat_ids,
        "attention_mask": torch.ones_like(chat_ids),
        "max_new_tokens": max_new_tokens,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if stop_after_first_tool_call:
        generate_kwargs["stopping_criteria"] = [_StopAfterFirstToolCall()]
    if temperature <= 1e-6:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature

    tools_for_constraints = constraint_tools or _current_tools
    constraint_mode = "none"
    if restrict_toolgen and tools_for_constraints:
        from ctx_to_lora.modeling.constrained_decoding import (
            ToolConstrainedLogitsProcessor,
            build_tool_call_json_logits_processor,
        )

        if strict_json_schema and len(tools_for_constraints) == 1:
            constraint_mode = "xgrammar"
            processors = [
                build_tool_call_json_logits_processor(
                    _tokenizer,
                    tools_for_constraints,
                    vocab_size=_model.config.vocab_size,
                )
            ]
        else:
            constraint_mode = "lexical"
            processors = [
                ToolConstrainedLogitsProcessor(
                    _tokenizer,
                    tools_for_constraints,
                    # With assistant prefill, include the prefixed <tool_call>
                    # so constraints activate immediately.
                    0 if assistant_prefix else input_token_count,
                )
            ]
        generate_kwargs["logits_processor"] = processors

    outputs = _model.generate(**generate_kwargs)
    response_ids = outputs[0][input_token_count:]
    decoded_text = _tokenizer.decode(
        response_ids, skip_special_tokens=False
    )
    response_text = assistant_prefix + decoded_text

    for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        response_text = response_text.replace(tag, "")
    response_text = response_text.strip()

    return {
        "text": response_text,
        "decoded_text": decoded_text,
        "input_tokens": int(input_token_count),
        "output_tokens": len(response_ids),
        "token_ids": [int(token_id) for token_id in response_ids.cpu().tolist()],
        "semantic_boundaries": _semantic_boundaries(response_ids),
        "enable_thinking": enable_thinking,
        "constraint_mode": constraint_mode,
        "adapter_name": _active_lora_name,
    }


def _canonical_selector_call(tool_name: str) -> str:
    return (
        "<tool_call>\n"
        + json.dumps(
            {
                "name": "select_tool",
                "arguments": {"tool_name": tool_name},
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n</tool_call>"
    )


def _arrow_top_right_singular(A, B):
    """Top right-singular vector of dW = B^T A, without ever forming dW.

    A: [r, d_in]  B: [r, d_out]  ->  unit vector in R^{d_in}

    dW is d_out x d_in per layer per tool, so forming it is not an option. It
    has rank <= r, so QR on A^T reduces the problem to an r x r SVD:
    A^T = QR gives dW = (B^T R^T) Q^T, whose right-singular vectors are Q @ W.

    The prototype lives in the module's INPUT space. For down_proj that is the
    9,728-dim MLP intermediate, NOT the 2,560-dim residual stream -- matching a
    prototype against hidden states would be a meaningless cosine.
    """
    import torch

    A32 = A.float()
    Q, R = torch.linalg.qr(A32.T, mode="reduced")
    K = B.float().T @ R.T
    _, _, Wh = torch.linalg.svd(K, full_matrices=False)
    return torch.nn.functional.normalize(Q @ Wh[0], dim=0)


def _arrow_scalers(prompt_text, raw_loras, n_chunks, module_name="down_proj"):
    """Arrow-style query-dependent weight per chunk, or None if gating is off.

    alpha_i = mean_l |cos(query activation at layer l, prototype of chunk i)|.

    combine_lora applies scalers to A only, and since Delta_i = B_i^T(a_i A_i)
    = a_i B_i^T A_i, a weighted sum drops straight into the existing path.

      D2L_GATE_MODE=arrow_soft  -> normalised so mean(alpha) == 1, holding the
                                   composed delta's magnitude equal to the
                                   unweighted sum. Accuracy only, NO cost win:
                                   every chunk is still generated and stacked.
      D2L_GATE_MODE=arrow_topk  -> alpha in {0,1}, keeping D2L_GATE_TOPK chunks.
                                   THIS is the one that buys anything: rank 8k
                                   instead of 8N.

    Prior worth stating: the retrieval-only probe scored 17.3% against 25%
    chance on the warm start, i.e. the query-alignment geometry Arrow assumes
    may simply not exist here. A null is informative -- it argues the gate has
    to be learned, or applied to a silence-trained checkpoint.
    """
    import torch

    mode = os.getenv("D2L_GATE_MODE", "none")
    if mode == "none" or n_chunks < 2:
        return None
    from ctx_to_lora.utils import get_layers

    layer_indices = list(_model.hypernet.layer_indices)
    layers = get_layers(_model.base_model)
    captured = {}

    def make_hook(order):
        def hook(_module, inputs, _output):
            captured[order] = inputs[0][0, -1].detach().float()

        return hook

    handles = [
        getattr(layers[index].mlp, module_name).register_forward_hook(make_hook(order))
        for order, index in enumerate(layer_indices)
    ]
    try:
        # The adapter is detached here (_restore_lora_forwards was just called),
        # so this is the BASE model's activation -- the query representation the
        # gate is allowed to see before any tool weights are applied.
        ids = torch.tensor(
            [_tokenizer(prompt_text, add_special_tokens=False)["input_ids"]],
            device=_model.device,
        )
        with torch.inference_mode():
            _model.base_model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    module_key = next(k for k in raw_loras if module_name in k)
    A_all = raw_loras[module_key]["A"]  # [n_chunks, n_layers, r, d_in]
    B_all = raw_loras[module_key]["B"]  # [n_chunks, n_layers, r, d_out]
    alphas = torch.zeros(n_chunks, device=_model.device, dtype=torch.float32)
    with torch.inference_mode():
        for chunk in range(n_chunks):
            sims = []
            for order in range(A_all.shape[1]):
                act = captured.get(order)
                if act is None:
                    continue
                proto = _arrow_top_right_singular(
                    A_all[chunk, order], B_all[chunk, order]
                )
                sims.append(
                    torch.abs(
                        torch.dot(torch.nn.functional.normalize(act, dim=0), proto)
                    )
                )
            alphas[chunk] = torch.stack(sims).mean() if sims else 0.0

    if mode == "arrow_topk":
        k = max(1, int(os.getenv("D2L_GATE_TOPK", "1")))
        keep = torch.topk(alphas, min(k, n_chunks)).indices
        hard = torch.zeros_like(alphas)
        hard[keep] = 1.0
        return hard.to(A_all.dtype)
    total = float(alphas.sum())
    if total <= 0:
        return None
    return (alphas * (n_chunks / total)).to(A_all.dtype)


def _score_router_candidates(
    stage1_messages: list,
    candidate_names: list[str],
    enable_thinking: bool | None = False,
):
    """Select a tool by forced candidate likelihood, not wrapper generation.

    The infrastructure owns the canonical ``select_tool`` serialization.  The
    model only scores the candidate-name span under the currently active D2L
    adapter (or the frozen base when no adapter is active).
    """
    import torch
    from ctx_to_lora.modeling.lora_merger import combine_lora

    if not candidate_names or any(
        not isinstance(name, str) or not name for name in candidate_names
    ):
        raise ValueError("candidate_names must contain non-empty strings")
    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("candidate_names must be unique")

    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    prompt_text = _tokenizer.apply_chat_template(
        stage1_messages,
        **template_kwargs,
    )

    encoded_rows = []
    candidate_positions = []
    canonical_calls = []
    for name in candidate_names:
        call = _canonical_selector_call(name)
        serialized_name = json.dumps(name, ensure_ascii=False)
        marker = '"tool_name":' + serialized_name
        marker_start = call.index(marker)
        name_start = len(prompt_text) + marker_start + len('"tool_name":')
        name_end = name_start + len(serialized_name)
        full_text = prompt_text + call
        encoded = _tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = [tuple(offset) for offset in encoded["offset_mapping"]]
        positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > name_start and start < name_end
        ]
        if not positions or positions[0] == 0:
            raise AssertionError(f"could not locate routed name tokens for {name!r}")
        encoded_rows.append(ids)
        candidate_positions.append(positions)
        canonical_calls.append(call)

    max_length = max(len(ids) for ids in encoded_rows)
    pad_id = _tokenizer.pad_token_id
    input_ids = torch.full(
        (len(encoded_rows), max_length),
        pad_id,
        dtype=torch.long,
        device=_model.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, ids in enumerate(encoded_rows):
        row = torch.tensor(ids, dtype=torch.long, device=_model.device)
        input_ids[row_index, : len(ids)] = row
        attention_mask[row_index, : len(ids)] = 1

    raw_loras = _model.generated_loras
    _model._restore_lora_forwards()
    combined = None
    if raw_loras is not None:
        n_chunks = int(getattr(_model, "_n_ctx_chunks", 1))
        # WS2: query-dependent per-chunk weights. Runs inside the worker, on
        # the composition path the harness already validates, precisely because
        # the standalone version of this (gate_reserved.py) got n_qs wrong and
        # scored a live adapter BELOW an ablated one without anything failing.
        scalers = _arrow_scalers(prompt_text, raw_loras, n_chunks)
        combined = combine_lora(
            raw_loras,
            torch.tensor([n_chunks], device=_model.device),
            lora_bias=(
                _model.hypernet.get_head_bias()
                if _model.hypernet.config.use_bias
                else None
            ),
            scalers=scalers,
        )

    _model.base_model.eval()

    # Score candidates in sub-batches.
    #
    # One forward over all N candidates materialises logits of
    # (N, seq_len, vocab) and then log_softmax(logits.float()) allocates the
    # same thing again in fp32. At N=32 with ~1.2k tokens and a 152k vocab that
    # is ~47 GiB of logits alone, which OOMed 294 times on rand_32 and every
    # item of reserved_50 -- silently, because the exception was being
    # overwritten by a bug in the caller's `finally` block. It looked like a
    # capacity cliff between 24 and 50 tools; it was allocator arithmetic.
    #
    # n_qs MUST equal the number of query sequences in the sub-batch, because
    # lora_forward does A.repeat_interleave(n_qs) to expand one context's
    # adapter across the rows. Passing the full candidate count while running a
    # smaller batch leaves the adapter mis-expanded -- the same mistake that
    # made gate_reserved.py score a live adapter WORSE than no adapter at all.
    # apply_lora_to_layers MUST run inside inference_mode along with the
    # forward. generate_weights retains a full autograd graph on its outputs
    # (see the detach in _preload_loras, added for exactly this reason), so
    # re-attaching the adapter outside no-grad once per sub-batch -- ~7x per
    # item at N=50 instead of once -- accumulates graph until the card is full.
    # That showed up as 24 OOMs by item 33 whose failing allocations were only
    # 144 MiB: exhaustion by accumulation, not by one oversized tensor.
    score_batch = max(1, int(os.getenv("D2L_SCORE_BATCH", "8")))
    scores = []
    token_scores = []
    for start in range(0, len(encoded_rows), score_batch):
        stop = min(start + score_batch, len(encoded_rows))
        with torch.inference_mode():
            if combined is not None:
                _model._restore_lora_forwards()
                _model.patch_lora_forward()
                _model.apply_lora_to_layers(
                    _model.base_model,
                    _model.hypernet.layer_indices,
                    combined,
                    torch.tensor(
                        [stop - start], dtype=torch.int32, device=_model.device
                    ),
                )
            logits = _model.base_model(
                input_ids=input_ids[start:stop],
                attention_mask=attention_mask[start:stop],
                use_cache=False,
            ).logits
            for offset in range(stop - start):
                row_index = start + offset
                values = []
                for position in candidate_positions[row_index]:
                    # log_softmax(x)[i] == x[i] - logsumexp(x); doing it one
                    # position at a time keeps the fp32 copy to a single vocab
                    # vector instead of the whole (N, seq_len, vocab) tensor.
                    row_logits = logits[offset, position - 1].float()
                    target = input_ids[row_index, position]
                    values.append(
                        float(
                            (
                                row_logits[target]
                                - torch.logsumexp(row_logits, dim=-1)
                            ).item()
                        )
                    )
                token_scores.append(values)
                scores.append(sum(values) / len(values))
        del logits

    best_index = max(range(len(scores)), key=scores.__getitem__)
    selected_name = candidate_names[best_index]
    selected_call = canonical_calls[best_index]
    token_ids = _tokenize_text(selected_call)["token_ids"]
    sorted_scores = sorted(scores, reverse=True)
    margin = (
        sorted_scores[0] - sorted_scores[1]
        if len(sorted_scores) > 1
        else float("inf")
    )
    return {
        "text": selected_call,
        "decoded_text": selected_call,
        "input_tokens": len(encoded_rows[best_index]),
        "output_tokens": len(token_ids),
        "token_ids": token_ids,
        "enable_thinking": enable_thinking,
        "constraint_mode": "candidate_likelihood",
        "adapter_name": _active_lora_name,
        "selected_name": selected_name,
        "selected_index": best_index,
        "candidate_scores": [
            {
                "name": name,
                "score": score,
                "token_logprobs": per_token,
            }
            for name, score, per_token in zip(
                candidate_names, scores, token_scores
            )
        ],
        "choice_margin": margin,
    }


def _generate_late_schema(
    stage1_messages: list,
    router_token_ids: list[int],
    raw_function: dict,
    max_new_tokens: int = 1024,
    temperature: float = 0,
    enable_thinking: bool | None = False,
):
    """Full-recompute Phase A for the exact late-schema continuation."""
    transcript = _render_late_transcript(
        stage1_messages,
        router_token_ids,
        raw_function,
        enable_thinking,
    )
    # D2L_BIND_RESTRICT=1: the orchestrator just fetched this schema, so the
    # bind turn is constrained to be a call to it (lexical constraints). Off
    # by default -- free generation, historical behavior.
    bind_restrict = os.environ.get("D2L_BIND_RESTRICT", "") == "1"
    result = _generate(
        transcript["messages"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        restrict_toolgen=bind_restrict,
        constraint_tools=(
            [{"type": "function", "function": raw_function.get("function", raw_function)}]
            if bind_restrict else None
        ),
        # Prefill the opening tag: lexical constraints only steer INSIDE a
        # tool_call; without the prefill the model can still answer in prose.
        assistant_prefix="<tool_call>\n" if bind_restrict else "",
        enable_thinking=enable_thinking,
        stop_after_first_tool_call=True,
        strict_json_schema=False,
    )
    if not bind_restrict and result["input_tokens"] != len(transcript["full_ids"]):
        raise AssertionError(
            "full transcript render changed between identity check and generation"
        )
    result["transcript"] = {
        key: value
        for key, value in transcript.items()
        if key not in {"messages", "full_ids"}
    }
    result["messages"] = transcript["messages"]
    result["full_token_ids"] = transcript["full_ids"]
    return result


def _cache_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, (tuple, list)) and cache:
        layer = cache[0]
        key = layer[0] if isinstance(layer, (tuple, list)) else layer
        return int(key.shape[-2])
    raise TypeError(f"Unsupported cache type: {type(cache).__name__}")


def _start_late_session(
    stage1_messages: list,
    max_new_tokens: int = 1024,
    temperature: float = 0,
    enable_thinking: bool | None = False,
):
    """Decode R and retain an exact cache through its final generated token."""
    import torch
    from transformers import StoppingCriteria

    if temperature > 1e-6:
        raise ValueError("stateful late-schema sessions require greedy decoding")
    p_ids = _render_chat_ids(
        stage1_messages,
        enable_thinking,
        add_generation_prompt=True,
    )
    input_token_count = int(p_ids.shape[1])

    class _StopAfterRouterCall(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, input_token_count:]
            return "</tool_call>" in _tokenizer.decode(
                generated, skip_special_tokens=False
            )

    n_ctx = getattr(_model, "_n_ctx_chunks", 1)
    outputs = _model.generate(
        n_ctx_chunks=torch.tensor([n_ctx], device=_model.device),
        input_ids=p_ids,
        attention_mask=torch.ones_like(p_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
        stopping_criteria=[_StopAfterRouterCall()],
        use_cache=True,
        return_dict_in_generate=True,
    )
    sequences = outputs.sequences
    r_ids = sequences[0, input_token_count:].tolist()
    cache = outputs.past_key_values
    expected_length = int(sequences.shape[1])
    cache_length = _cache_length(cache)
    if cache_length == expected_length - 1:
        final_token = sequences[:, -1:]
        attention_mask = torch.ones(
            (1, expected_length), dtype=torch.long, device=_model.device
        )
        position_ids = torch.tensor(
            [[expected_length - 1]], dtype=torch.long, device=_model.device
        )
        consumed = _model.base_model(
            input_ids=final_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache = consumed.past_key_values
        cache_length = _cache_length(cache)
    if cache_length != expected_length:
        raise AssertionError(
            f"router cache length {cache_length} != token length {expected_length}"
        )

    session_id = uuid.uuid4().hex
    _sessions[session_id] = {
        "stage1_messages": stage1_messages,
        "p_ids": p_ids[0].tolist(),
        "r_ids": r_ids,
        "cache": cache,
        "adapter_name": _active_lora_name,
    }
    return {
        "session_id": session_id,
        "text": _decode_response(r_ids),
        "decoded_text": _tokenizer.decode(r_ids, skip_special_tokens=False),
        "input_tokens": input_token_count,
        "output_tokens": len(r_ids),
        "token_ids": r_ids,
        "enable_thinking": enable_thinking,
        "constraint_mode": "none",
        "adapter_name": _active_lora_name,
        "p_hash": _token_hash(p_ids[0].tolist()),
        "r_hash": _token_hash(r_ids),
    }


def _append_late_schema_session(
    session_id: str,
    raw_function: dict,
    max_new_tokens: int = 1024,
):
    """Append S only to the retained P || R cache, then greedily decode C."""
    import torch

    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Unknown late-schema session {session_id!r}")
    if session["adapter_name"] != _active_lora_name:
        raise RuntimeError("Generated LoRA identity changed during session")

    transcript = _render_late_transcript(
        session["stage1_messages"],
        session["r_ids"],
        raw_function,
        False,
    )
    if transcript["p_ids"] != session["p_ids"]:
        raise AssertionError("Session P tokens changed")
    s_ids = transcript["s_ids"]
    old_length = len(session["p_ids"]) + len(session["r_ids"])
    s_tensor = torch.tensor([s_ids], dtype=torch.long, device=_model.device)
    total_length = old_length + len(s_ids)
    attention_mask = torch.ones(
        (1, total_length), dtype=torch.long, device=_model.device
    )
    position_ids = torch.arange(
        old_length, total_length, dtype=torch.long, device=_model.device
    ).unsqueeze(0)
    output = _model.base_model(
        input_ids=s_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=session["cache"],
        use_cache=True,
    )
    cache = output.past_key_values
    next_logits = output.logits[:, -1, :]
    generated: list[int] = []
    eos_id = _tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        token = int(torch.argmax(next_logits, dim=-1).item())
        generated.append(token)
        text = _tokenizer.decode(generated, skip_special_tokens=False)
        if "</tool_call>" in text or token == eos_id:
            break
        current_length = total_length + len(generated)
        token_tensor = torch.tensor(
            [[token]], dtype=torch.long, device=_model.device
        )
        step = _model.base_model(
            input_ids=token_tensor,
            attention_mask=torch.ones(
                (1, current_length), dtype=torch.long, device=_model.device
            ),
            position_ids=torch.tensor(
                [[current_length - 1]],
                dtype=torch.long,
                device=_model.device,
            ),
            past_key_values=cache,
            use_cache=True,
        )
        cache = step.past_key_values
        next_logits = step.logits[:, -1, :]

    result = {
        "text": _decode_response(generated),
        "decoded_text": _tokenizer.decode(
            generated, skip_special_tokens=False
        ),
        "input_tokens": len(s_ids),
        "output_tokens": len(generated),
        "token_ids": generated,
        "enable_thinking": False,
        "constraint_mode": "none",
        "adapter_name": _active_lora_name,
        "processed_append_tokens": len(s_ids),
        "transcript": {
            key: value
            for key, value in transcript.items()
            if key not in {"messages", "full_ids"}
        },
        "messages": transcript["messages"],
        "full_token_ids": transcript["full_ids"],
    }
    del _sessions[session_id]
    return result


def _close_session(session_id: str):
    existed = _sessions.pop(session_id, None) is not None
    return {"closed": existed}


def _tokenize_text(text: str):
    token_ids = _tokenizer.encode(text, add_special_tokens=False)
    return {"token_ids": [int(token_id) for token_id in token_ids]}


def _generate_with_switch(
    messages: list,
    start_adapter: str | None,
    end_adapter: str | None,
    switch_at: int | None,
    cache_policy: str = "preserve",
    max_new_tokens: int = 1024,
    temperature: float = 0,
    enable_thinking: bool | None = False,
    return_first_logits: bool = False,
    restrict_toolgen: bool = False,
    constraint_tools: list[dict] | None = None,
    strict_json_schema: bool = True,
):
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")
    if temperature > 1e-6:
        raise ValueError("mid-decode experiment supports greedy decoding only")

    try:
        from bfcl_eval.model_handler.local_inference.middecode_decode import (
            greedy_decode_with_switch,
        )
    except ModuleNotFoundError:
        from middecode_decode import greedy_decode_with_switch

    chat_ids = _render_chat_ids(messages, enable_thinking)
    processors = []
    constraint_mode = "none"
    if restrict_toolgen and constraint_tools:
        from ctx_to_lora.modeling.constrained_decoding import (
            ToolConstrainedLogitsProcessor,
            build_tool_call_json_logits_processor,
        )

        if strict_json_schema and len(constraint_tools) == 1:
            constraint_mode = "xgrammar"
            processors.append(
                build_tool_call_json_logits_processor(
                    _tokenizer,
                    constraint_tools,
                    vocab_size=_model.config.vocab_size,
                )
            )
        else:
            constraint_mode = "lexical"
            processors.append(
                ToolConstrainedLogitsProcessor(
                    _tokenizer,
                    constraint_tools,
                    chat_ids.shape[1],
                )
            )
    trace = greedy_decode_with_switch(
        model=_model.base_model,
        input_ids=chat_ids,
        attention_mask=torch.ones_like(chat_ids),
        activate_adapter=_activate_lora,
        start_adapter=start_adapter,
        end_adapter=end_adapter,
        switch_at=switch_at,
        cache_policy=cache_policy,
        max_new_tokens=max_new_tokens,
        eos_token_id=_tokenizer.eos_token_id,
        tokenizer=_tokenizer,
        top_k=5,
        return_first_logits=return_first_logits,
        logits_processors=processors,
    )
    return {
        "text": _decode_response(trace["token_ids"]),
        "input_tokens": int(chat_ids.shape[1]),
        "output_tokens": len(trace["token_ids"]),
        "semantic_boundaries": _semantic_boundaries(trace["token_ids"]),
        "enable_thinking": enable_thinking,
        "constraint_mode": constraint_mode,
        "adapter_name": _active_lora_name,
        **trace,
    }


_DISPATCH = {
    "load_model": lambda args: _load_model(args["checkpoint_path"]),
    # inference_mode is REQUIRED here, not an optimisation. generate_weights
    # retains a full autograd graph on its outputs; _preload_loras already
    # wraps this call for exactly that reason ("quickly exhausts GPU memory"),
    # but the ordinary per-item path did not. At N=50 the whole catalogue is one
    # ~8k-token context through a 36-layer per-layer-activation encoder, so the
    # retained graph is enormous and the card fills within ~30 items -- which
    # presented as OOMs on 144 MiB allocations, i.e. exhaustion by accumulation.
    "internalize": lambda args: _internalize_no_grad(
        args["tool_defs"],
        args.get("chunk_size", 1024),
        args.get("ctx_chunk_mode", "none"),
        args.get("tools_per_chunk", 1),
    ),
    "preload_loras": lambda args: _preload_loras(
        args["adapters"], args.get("chunk_size", 1024)
    ),
    "activate_lora": lambda args: _activate_lora(args.get("adapter_name")),
    "generate": lambda args: _generate(
        args["messages"],
        args.get("max_new_tokens", 1024),
        args.get("temperature", 0),
        args.get("restrict_toolgen", False),
        args.get("constraint_tools"),
        args.get("assistant_prefix", ""),
        args.get("enable_thinking"),
        args.get("stop_after_first_tool_call", False),
        args.get("strict_json_schema", False),
    ),
    "score_router_candidates": lambda args: _score_router_candidates(
        args["stage1_messages"],
        args["candidate_names"],
        args.get("enable_thinking", False),
    ),
    "render_late_transcript": lambda args: _render_late_transcript(
        args["stage1_messages"],
        args["router_token_ids"],
        args["raw_function"],
        args.get("enable_thinking", False),
    ),
    "generate_late_schema": lambda args: _generate_late_schema(
        args["stage1_messages"],
        args["router_token_ids"],
        args["raw_function"],
        args.get("max_new_tokens", 1024),
        args.get("temperature", 0),
        args.get("enable_thinking", False),
    ),
    "start_late_session": lambda args: _start_late_session(
        args["stage1_messages"],
        args.get("max_new_tokens", 1024),
        args.get("temperature", 0),
        args.get("enable_thinking", False),
    ),
    "append_late_schema_session": lambda args: _append_late_schema_session(
        args["session_id"],
        args["raw_function"],
        args.get("max_new_tokens", 1024),
    ),
    "close_session": lambda args: _close_session(args["session_id"]),
    "tokenize_text": lambda args: _tokenize_text(args["text"]),
    "generate_with_switch": lambda args: _generate_with_switch(
        args["messages"],
        args.get("start_adapter"),
        args.get("end_adapter"),
        args.get("switch_at"),
        args.get("cache_policy", "preserve"),
        args.get("max_new_tokens", 1024),
        args.get("temperature", 0),
        args.get("enable_thinking", False),
        args.get("return_first_logits", False),
        args.get("restrict_toolgen", False),
        args.get("constraint_tools"),
        args.get("strict_json_schema", True),
    ),
    "get_model_info": lambda _: _get_model_info(),
    "export_peft": lambda args: _export_peft(args["output_dir"]),
    "internalize_and_export": lambda args: _internalize_and_export(
        args["tool_defs"], args["output_dir"], args.get("chunk_size", 1024)
    ),
    "reset": lambda _: (_model.reset() or {}),
    "ping": lambda _: {"status": "alive"},
}


def main():
    if len(sys.argv) < 2:
        print("Usage: d2l_worker.py <d2l-src-path>", file=sys.stderr)
        sys.exit(1)

    d2l_src = sys.argv[1]
    if d2l_src not in sys.path:
        sys.path.insert(0, d2l_src)

    _send({"ok": True, "result": {"status": "ready"}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _send({"ok": False, "error": f"Invalid JSON: {exc}"})
            continue

        cmd = req.get("cmd")
        args = req.get("args", {})
        handler = _DISPATCH.get(cmd)
        if handler is None:
            _send({"ok": False, "error": f"Unknown command: {cmd}"})
            continue

        try:
            result = handler(args)
            _send({"ok": True, "result": result if result is not None else {}})
        except Exception as exc:
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
