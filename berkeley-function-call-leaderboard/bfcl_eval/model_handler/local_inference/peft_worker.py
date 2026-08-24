"""
Subprocess worker for HF PEFT adapter inference (D2L-exported / schema-SFT adapters).

Runs inside the D2L virtualenv. Protocol matches d2l_worker.py (JSON-lines).
"""

from __future__ import annotations

import json
import os
import sys

_real_stdout_fd = os.dup(1)
os.dup2(2, 1)
_proto = os.fdopen(_real_stdout_fd, "w", buffering=1)
sys.stdout = sys.stderr


def _send(obj: dict):
    _proto.write(json.dumps(obj) + "\n")
    _proto.flush()


_model = None
_tokenizer = None
_base = None
_current_adapter: str | None = None
_active_adapter_name: str | None = None
_loaded_adapters: dict[str, str] = {}
_adapters_enabled = False
_device = "cuda:0"


def _load_chat_template(tokenizer, base_model_name: str, d2l_root: str):
    from pathlib import Path

    template_path = (
        Path(d2l_root) / "chat_templates" / f"{base_model_name}.jinja"
    )
    if template_path.is_file():
        text = template_path.read_text().replace("    ", "").replace("\n", "")
        tokenizer.chat_template = text
        print(f"[peft_worker] Using chat template from {template_path}")
    else:
        print(f"[peft_worker] WARNING: chat template missing at {template_path}")


def _load_base(base_model: str, d2l_root: str = ""):
    global _model, _tokenizer, _base, _current_adapter, _active_adapter_name
    global _loaded_adapters, _adapters_enabled, _device
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[peft_worker] Loading base {base_model} on {_device} ...")
    _tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if d2l_root:
        _load_chat_template(_tokenizer, base_model, d2l_root)
    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token_id = _tokenizer.eos_token_id

    _base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=_device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    _base.eval()
    _model = _base
    _current_adapter = None
    _active_adapter_name = None
    _loaded_adapters = {}
    _adapters_enabled = False
    return {"base_model_name": base_model, "device": _device}


def _unload_adapter():
    global _model, _base, _current_adapter, _active_adapter_name
    global _loaded_adapters, _adapters_enabled
    from peft import PeftModel

    if not isinstance(_model, PeftModel):
        _current_adapter = None
        _active_adapter_name = None
        _loaded_adapters = {}
        _adapters_enabled = False
        return {}
    _model = _model.unload()
    _model.eval()
    _base = _model
    _current_adapter = None
    _active_adapter_name = None
    _loaded_adapters = {}
    _adapters_enabled = False
    import torch

    torch.cuda.empty_cache()
    return {}


def _load_adapter(adapter_path: str):
    global _model, _current_adapter, _active_adapter_name
    global _loaded_adapters, _adapters_enabled
    from peft import PeftModel

    if _base is None:
        raise RuntimeError("Call load_base first")
    if _current_adapter == adapter_path:
        return {"adapter_path": adapter_path, "reused": True}

    _unload_adapter()
    print(f"[peft_worker] Loading adapter {adapter_path}")
    _model = PeftModel.from_pretrained(_base, adapter_path)
    _model.eval()
    _current_adapter = adapter_path
    _active_adapter_name = "default"
    _loaded_adapters = {"default": adapter_path}
    _adapters_enabled = True
    return {"adapter_path": adapter_path, "reused": False}


def _preload_adapters(adapters: dict[str, str]):
    """Load named adapters once so switching does not re-wrap the model."""
    global _model, _current_adapter, _active_adapter_name
    global _loaded_adapters, _adapters_enabled
    from peft import PeftModel

    if _base is None:
        raise RuntimeError("Call load_base first")
    if not adapters:
        return {"loaded": sorted(_loaded_adapters)}

    for name, path in sorted(adapters.items()):
        if not name:
            raise ValueError("adapter names must be non-empty")
        existing = _loaded_adapters.get(name)
        if existing is not None:
            if existing != path:
                raise ValueError(
                    f"adapter {name!r} already maps to {existing}, not {path}"
                )
            continue
        if not os.path.isfile(os.path.join(path, "adapter_model.safetensors")):
            raise FileNotFoundError(f"missing adapter weights under {path}")

        print(f"[peft_worker] Preloading adapter {name} from {path}")
        if not isinstance(_model, PeftModel):
            _model = PeftModel.from_pretrained(
                _base,
                path,
                adapter_name=name,
            )
        else:
            _model.load_adapter(path, adapter_name=name)
        _loaded_adapters[name] = path

    _model.eval()
    # Loading an adapter activates one as a side effect. Make that state explicit.
    first_name = next(iter(sorted(adapters)))
    _model.set_adapter(first_name)
    _model.base_model.enable_adapter_layers()
    _active_adapter_name = first_name
    _current_adapter = _loaded_adapters[first_name]
    _adapters_enabled = True
    return {
        "loaded": sorted(_loaded_adapters),
        "active_adapter": _active_adapter_name,
    }


def _activate_adapter(adapter_name: str | None):
    """Activate a named PEFT adapter, or disable all LoRAs for base inference."""
    global _current_adapter, _active_adapter_name, _adapters_enabled
    from peft import PeftModel

    if adapter_name is None:
        if isinstance(_model, PeftModel):
            _model.base_model.disable_adapter_layers()
        _active_adapter_name = None
        _current_adapter = None
        _adapters_enabled = False
        return {"adapter_path": None, "adapters_enabled": False}

    if not isinstance(_model, PeftModel):
        raise RuntimeError("No PEFT adapters loaded")
    if adapter_name not in _loaded_adapters:
        raise KeyError(f"Adapter {adapter_name!r} has not been preloaded")
    _model.set_adapter(adapter_name)
    _model.base_model.enable_adapter_layers()
    _active_adapter_name = adapter_name
    _current_adapter = _loaded_adapters[adapter_name]
    _adapters_enabled = True
    return {
        "adapter_path": _current_adapter,
        "adapters_enabled": True,
    }


def _render_chat_ids(messages: list, enable_thinking: bool | None):
    template_kwargs = dict(
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    try:
        chat_ids = _tokenizer.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        # Older templates may not accept enable_thinking.
        template_kwargs.pop("enable_thinking", None)
        chat_ids = _tokenizer.apply_chat_template(messages, **template_kwargs)
    return chat_ids.to(_model.device)


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
    enable_thinking: bool | None = False,
):
    import torch

    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded")

    chat_ids = _render_chat_ids(messages, enable_thinking)
    input_token_count = chat_ids.shape[1]

    gen_kwargs = {
        "input_ids": chat_ids,
        "attention_mask": torch.ones_like(chat_ids),
        "max_new_tokens": max_new_tokens,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
    }
    if temperature <= 1e-6:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = _model.generate(**gen_kwargs)

    response_ids = outputs[0][input_token_count:]
    response_text = _decode_response(response_ids)

    return {
        "text": response_text,
        "input_tokens": int(input_token_count),
        "output_tokens": len(response_ids),
        "token_ids": [int(token_id) for token_id in response_ids.cpu().tolist()],
        "semantic_boundaries": _semantic_boundaries(response_ids),
        "enable_thinking": enable_thinking,
        "adapter_path": _current_adapter,
        "adapter_name": _active_adapter_name,
    }


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
    prefill_adapter: str | None = None,
    replay_last_prompt_token: bool = False,
    replay_prompt_tokens: int = 0,
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
        model=_model,
        input_ids=chat_ids,
        attention_mask=torch.ones_like(chat_ids),
        activate_adapter=_activate_adapter,
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
        prefill_adapter=prefill_adapter,
        replay_last_prompt_token=replay_last_prompt_token,
        replay_prompt_tokens=replay_prompt_tokens,
    )
    return {
        "text": _decode_response(trace["token_ids"]),
        "input_tokens": int(chat_ids.shape[1]),
        "output_tokens": len(trace["token_ids"]),
        "semantic_boundaries": _semantic_boundaries(trace["token_ids"]),
        "enable_thinking": enable_thinking,
        "constraint_mode": constraint_mode,
        "adapter_path": _current_adapter,
        "adapter_name": _active_adapter_name,
        **trace,
    }


_DISPATCH = {
    "load_base": lambda args: _load_base(
        args["base_model"], args.get("d2l_root", "")
    ),
    "load_adapter": lambda args: _load_adapter(args["adapter_path"]),
    "preload_adapters": lambda args: _preload_adapters(args["adapters"]),
    "activate_adapter": lambda args: _activate_adapter(args.get("adapter_name")),
    "unload_adapter": lambda _: _unload_adapter(),
    "generate": lambda args: _generate(
        args["messages"],
        args.get("max_new_tokens", 1024),
        args.get("temperature", 0),
        args.get("enable_thinking", False),
    ),
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
        args.get("prefill_adapter"),
        args.get("replay_last_prompt_token", False),
        args.get("replay_prompt_tokens", 0),
    ),
    "ping": lambda _: {"status": "alive"},
}


def main():
    # Optional: path to doc-to-lora root for chat templates (argv[1])
    d2l_root_default = ""
    if len(sys.argv) > 1:
        d2l_root_default = sys.argv[1]
        # Stash for load_base if not passed each time
        os.environ.setdefault("D2L_ROOT", d2l_root_default)

    _send({"ok": True, "result": {"status": "ready"}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req["cmd"]
            args = req.get("args") or {}
            if cmd == "load_base" and "d2l_root" not in args and d2l_root_default:
                args = {**args, "d2l_root": d2l_root_default}
            if cmd not in _DISPATCH:
                raise KeyError(f"Unknown command: {cmd}")
            result = _DISPATCH[cmd](args)
            _send({"ok": True, "result": result})
        except Exception as e:
            import traceback

            traceback.print_exc()
            _send({"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
