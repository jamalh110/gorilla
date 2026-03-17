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


def _load_model(checkpoint_path: str):
    global _model, _tokenizer
    import torch
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    print(f"[d2l_worker] Loading model from {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, weights_only=False)
    _model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    _model.reset()
    _tokenizer = get_tokenizer(_model.base_model.name_or_path)
    name = _model.base_model.name_or_path
    print(f"[d2l_worker] Model loaded (base: {name})")
    return {"base_model_name": name}


def _internalize(tool_defs: str, chunk_size: int = 1024):
    global _current_tools
    import torch
    from ctx_to_lora.data.definitions import CTX_AFFIXES
    from ctx_to_lora.data.processing import tokenize_ctx_text
    from ctx_to_lora.model_loading import get_tokenizer
    from ctx_to_lora.modeling.lora_merger import combine_lora

    _current_tools = json.loads(tool_defs)
    _model.reset()

    ctx_tokenizer = get_tokenizer(_model.ctx_encoder.base_model.name_or_path)
    ctx_ids_nested = tokenize_ctx_text(
        dict(context=[tool_defs]), ctx_tokenizer
    )["ctx_ids"]
    ctx_ids = ctx_ids_nested[0]

    n_chunks = max(1, ceil(len(ctx_ids) / chunk_size))

    if n_chunks == 1:
        _model.internalize(tool_defs)
        return {}

    avg_len = ceil(len(ctx_ids) / n_chunks)
    chunks = [ctx_ids[i : i + avg_len] for i in range(0, len(ctx_ids), avg_len)]

    model_name = _model.base_model.config.name_or_path
    if model_name not in CTX_AFFIXES:
        _model.internalize(tool_defs)
        return {}

    affixes = CTX_AFFIXES[model_name]
    chunks[0] = chunks[0] + affixes["suffix"]
    for i in range(1, len(chunks) - 1):
        chunks[i] = affixes["prefix"] + chunks[i] + affixes["suffix"]
    chunks[-1] = affixes["prefix"] + chunks[-1]

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

    n_ctx_chunks = torch.tensor([n_chunks], device=_model.device)
    merged = combine_lora(
        merged,
        n_ctx_chunks,
        lora_bias=(
            _model.hypernet.get_head_bias()
            if _model.hypernet.config.use_bias
            else None
        ),
    )
    _model.generated_loras = merged
    _model.patch_lora_forward()
    return {}


def _generate(
    messages: list,
    max_new_tokens: int = 1024,
    temperature: float = 0,
    restrict_toolgen: bool = False,
):
    template_kwargs = dict(
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    chat_ids = _tokenizer.apply_chat_template(
        messages,
        **template_kwargs,
    ).to(_model.device)

    input_token_count = chat_ids.shape[1]

    generate_kwargs = {"input_ids": chat_ids, "max_new_tokens": max_new_tokens}
    if temperature == 0:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature

    if restrict_toolgen and _current_tools:
        from ctx_to_lora.modeling.constrained_decoding import (
            ToolConstrainedLogitsProcessor,
        )

        processor = ToolConstrainedLogitsProcessor(
            _tokenizer, _current_tools, input_token_count
        )
        generate_kwargs["logits_processor"] = [processor]

    outputs = _model.generate(**generate_kwargs)
    response_ids = outputs[0][input_token_count:]
    response_text = _tokenizer.decode(response_ids, skip_special_tokens=False)

    for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        response_text = response_text.replace(tag, "")
    response_text = response_text.strip()

    return {
        "text": response_text,
        "input_tokens": int(input_token_count),
        "output_tokens": len(response_ids),
    }


_DISPATCH = {
    "load_model": lambda args: _load_model(args["checkpoint_path"]),
    "internalize": lambda args: _internalize(
        args["tool_defs"], args.get("chunk_size", 1024)
    ),
    "generate": lambda args: _generate(
        args["messages"],
        args.get("max_new_tokens", 1024),
        args.get("temperature", 0),
        args.get("restrict_toolgen", False),
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
