"""JSON-lines worker for the non-thinking Qwen3-0.6B staged binder.

The worker runs in the Doc-to-LoRA environment, but loads an ordinary fully
fine-tuned Hugging Face checkpoint.  It intentionally has no command that
accepts the original BFCL query; the parent sends only intent chat messages
and passes the selected schema through Qwen's native ``tools`` slot plus
XGrammar constraints.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_model = None
_tokenizer = None
_proto = None


def apply_binder_chat_template(
    tokenizer,
    messages: list[dict],
    constraint_tools: list[dict],
    assistant_prefix: str,
):
    """Apply Qwen's chat template with native tools and thinking disabled."""
    prompt_messages = list(messages)
    kwargs: dict[str, Any] = {
        "add_special_tokens": False,
        "return_attention_mask": False,
        "return_tensors": "pt",
        "enable_thinking": False,
        "tools": constraint_tools,
    }
    if assistant_prefix:
        prompt_messages.append({"role": "assistant", "content": assistant_prefix})
        kwargs["add_generation_prompt"] = False
        kwargs["continue_final_message"] = True
    else:
        kwargs["add_generation_prompt"] = True
    return tokenizer.apply_chat_template(prompt_messages, **kwargs)


def _send(obj: dict):
    _proto.write(json.dumps(obj) + "\n")
    _proto.flush()


def _load_model(checkpoint_path: str):
    global _model, _tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not checkpoint_path:
        raise ValueError("binder checkpoint path is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[binder_worker] Loading non-thinking binder from {checkpoint_path} ...")
    _tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    _model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=dtype,
    ).to(device)
    _model.eval()
    print(f"[binder_worker] Binder loaded on {device}")
    return {
        "model_name": getattr(_model.config, "name_or_path", checkpoint_path),
        "enable_thinking": False,
    }


def _generate(
    messages: list[dict],
    constraint_tools: list[dict],
    max_new_tokens: int = 256,
    temperature: float = 0,
    assistant_prefix: str = '<tool_call>\n{"name":"',
    strict_json_schema: bool = True,
):
    import torch
    from transformers import StoppingCriteria

    if _model is None or _tokenizer is None:
        raise RuntimeError("binder model is not loaded")
    if not constraint_tools:
        raise ValueError("constraint_tools must contain the selected tool")

    chat_ids = apply_binder_chat_template(
        _tokenizer, messages, constraint_tools, assistant_prefix
    ).to(_model.device)
    input_token_count = chat_ids.shape[1]

    class _StopAfterFirstToolCall(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[0, input_token_count:]
            text = _tokenizer.decode(generated, skip_special_tokens=False)
            return "</tool_call>" in text

    generate_kwargs = {
        "input_ids": chat_ids,
        "attention_mask": torch.ones_like(chat_ids),
        "max_new_tokens": max_new_tokens,
        "stopping_criteria": [_StopAfterFirstToolCall()],
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if temperature <= 1e-6:
        generate_kwargs["do_sample"] = False
    else:
        generate_kwargs.update(do_sample=True, temperature=temperature)

    from ctx_to_lora.modeling.constrained_decoding import (
        ToolConstrainedLogitsProcessor,
        build_tool_call_json_logits_processor,
    )

    if strict_json_schema:
        constraint_mode = "xgrammar"
        processor = build_tool_call_json_logits_processor(
            _tokenizer,
            constraint_tools,
            vocab_size=_model.config.vocab_size,
        )
    else:
        constraint_mode = "lexical"
        processor = ToolConstrainedLogitsProcessor(
            _tokenizer, constraint_tools, 0
        )
    generate_kwargs["logits_processor"] = [processor]

    with torch.inference_mode():
        outputs = _model.generate(**generate_kwargs)
    response_ids = outputs[0][input_token_count:]
    text = assistant_prefix + _tokenizer.decode(
        response_ids, skip_special_tokens=False
    )
    for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(tag, "")
    return {
        "text": text.strip(),
        "input_tokens": int(input_token_count),
        "output_tokens": len(response_ids),
        "enable_thinking": False,
        "constraint_mode": constraint_mode,
    }


_DISPATCH = {
    "load_model": lambda args: _load_model(args["checkpoint_path"]),
    "generate": lambda args: _generate(
        args["messages"],
        args["constraint_tools"],
        args.get("max_new_tokens", 256),
        args.get("temperature", 0),
        args.get("assistant_prefix", '<tool_call>\n{"name":"'),
        args.get("strict_json_schema", True),
    ),
    "ping": lambda _: {"status": "alive"},
}


def main():
    global _proto
    if len(sys.argv) < 2:
        print("Usage: binder_worker.py <d2l-src-path>", file=sys.stderr)
        raise SystemExit(2)

    d2l_src = sys.argv[1]
    if d2l_src not in sys.path:
        sys.path.insert(0, d2l_src)

    real_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    _proto = os.fdopen(real_stdout_fd, "w", buffering=1)
    sys.stdout = sys.stderr
    _send({"ok": True, "result": {"status": "ready"}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            command = request.get("cmd")
            handler = _DISPATCH.get(command)
            if handler is None:
                raise ValueError(f"Unknown command: {command}")
            result = handler(request.get("args", {}))
            _send({"ok": True, "result": result or {}})
        except Exception as exc:
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
