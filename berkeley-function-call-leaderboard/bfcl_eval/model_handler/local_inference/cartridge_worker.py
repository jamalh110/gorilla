"""Subprocess worker for self-study Cartridge inference on BFCL.

Runs inside the cartridges repo's own virtualenv (FlexAttention / torch build is
incompatible with the BFCL env). Communicates with the parent
``CartridgeHandler`` via a JSON-lines protocol over a saved copy of stdout
(fd 1 is redirected to stderr so model/CUDA prints don't corrupt the channel).

For each test entry, the worker:
  1. selects the pretrained cartridge for the entry's tool (matched by schema
     hash, falling back to tool name),
  2. builds a schema-less prompt (cartridge replaces the tool context),
  3. generates with ``flex_generate`` using the cartridge as the KV prefix.

No reader LoRA is used — this is the pure self-study cartridge baseline.

Usage (called by CartridgeHandler, not manually):
    <cartridges-venv-python> cartridge_worker.py <cartridges-repo-path> <cartridge-dir>
"""
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


# Qwen3 chat-frame token ids (must match cartridges.datasets.qwen_messages_to_element)
USER_START = [151644, 872, 198]       # <|im_start|>user\n
ASSISTANT_START = [151644, 77091, 198]  # <|im_start|>assistant\n
MSG_END = [151645, 198]               # <|im_end|>\n

_model = None
_tokenizer = None
_attn_config = None
_cart_dir = None
_hash_to_path: dict = {}
_name_to_paths: dict = {}
_cart_cache: dict = {}
_device = "cuda"


def _build_manifest(cart_dir: str):
    global _hash_to_path, _name_to_paths
    import glob
    from collections import defaultdict

    _hash_to_path = {}
    _name_to_paths = defaultdict(list)
    for path in glob.glob(os.path.join(cart_dir, "*.pt")):
        stem = os.path.splitext(os.path.basename(path))[0]  # name_slug__hash
        if "__" in stem:
            name_slug, h = stem.rsplit("__", 1)
        else:
            name_slug, h = stem, stem
        _hash_to_path[h] = path
        _name_to_paths[name_slug].append((h, path))


def _load_model(cart_dir: str):
    global _model, _tokenizer, _attn_config, _cart_dir
    import torch
    from transformers import AutoTokenizer

    from cartridges.cache import AttnConfig
    from cartridges.models import FlexQwen3ForCausalLM, HFModelConfig
    from examples.bfcl_tools.common import MODEL_NAME, register_model

    register_model()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = (
        HFModelConfig(
            pretrained_model_name_or_path=MODEL_NAME,
            model_cls=FlexQwen3ForCausalLM,
        )
        .instantiate()
        .to(_device)
        .to(torch.bfloat16)
    )
    _model.eval()
    head_dim = (
        _model.config.head_dim
        if hasattr(_model.config, "head_dim")
        else _model.config.hidden_size // _model.config.num_attention_heads
    )
    _attn_config = AttnConfig(
        n_layers=_model.config.num_hidden_layers,
        n_heads=_model.config.num_key_value_heads,
        head_dim=head_dim,
    )
    _cart_dir = cart_dir
    _build_manifest(cart_dir)
    print(f"[cartridge_worker] model loaded; {len(_hash_to_path)} cartridges in {cart_dir}")
    return {"base_model_name": MODEL_NAME, "num_cartridges": len(_hash_to_path)}


def _resolve_cartridge_path(function: dict):
    from examples.bfcl_tools.common import schema_hash, slug

    h = schema_hash(function)
    if h in _hash_to_path:
        return _hash_to_path[h]
    name_slug = slug(function.get("name", ""))
    candidates = _name_to_paths.get(name_slug, [])
    if len(candidates) == 1:
        return candidates[0][1]
    if candidates:
        # multiple schemas share this name; hash didn't match -> take first
        return candidates[0][1]
    return None


def _get_cartridge(path: str):
    if path in _cart_cache:
        return _cart_cache[path]
    import torch
    from cartridges.cache import TrainableCache

    cart = TrainableCache.from_pretrained(path, device=_device).to(_device).to(torch.bfloat16)
    _cart_cache[path] = cart
    return cart


def _generate(messages: list, function: dict, max_new_tokens: int = 256, temperature: float = 0.0):
    import torch
    from cartridges.generation import flex_generate

    path = _resolve_cartridge_path(function)
    if path is None:
        return {"text": "", "input_tokens": 0, "output_tokens": 0, "missing": True}

    cache = _get_cartridge(path)

    # Build a schema-less prompt: only the user turn(s); the cartridge supplies
    # the tool context. This mirrors the training element format exactly.
    ids: list = []
    for m in messages:
        if m["role"] == "system":
            continue
        start = USER_START if m["role"] == "user" else ASSISTANT_START
        content_ids = _tokenizer.encode(m["content"], add_special_tokens=False)
        ids += start + content_ids + MSG_END
    ids += ASSISTANT_START

    input_ids = torch.tensor(ids, dtype=torch.long, device=_device)
    seq_ids = torch.zeros(len(ids), dtype=torch.long, device=_device)
    position_ids = torch.arange(len(ids), dtype=torch.long, device=_device)

    out = flex_generate(
        model=_model,
        tokenizer=_tokenizer,
        input_ids=input_ids,
        seq_ids=seq_ids,
        position_ids=position_ids,
        cache=cache,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    gen_ids = out.get(0, [])
    text = _tokenizer.decode(gen_ids, skip_special_tokens=False)
    for tag in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        text = text.replace(tag, "")
    text = text.strip()
    return {
        "text": text,
        "input_tokens": int(len(ids)),
        "output_tokens": int(len(gen_ids)),
        "missing": False,
    }


_DISPATCH = {
    "load_model": lambda args: _load_model(args["cartridge_dir"]),
    "generate": lambda args: _generate(
        args["messages"],
        args["function"],
        args.get("max_new_tokens", 256),
        args.get("temperature", 0.0),
    ),
    "ping": lambda _: {"status": "alive"},
}


def main():
    if len(sys.argv) < 3:
        print("Usage: cartridge_worker.py <cartridges-repo-path> <cartridge-dir>", file=sys.stderr)
        sys.exit(1)

    repo_path = sys.argv[1]
    cart_dir = sys.argv[2]
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    os.environ.setdefault("CARTRIDGES_DIR", repo_path)
    # cartridges/__init__.py requires CARTRIDGES_OUTPUT_DIR to be set on import.
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.join(repo_path, "bfcl_runs"))
    os.chdir(repo_path)

    # eager-load the model so the first generate isn't charged the load cost
    try:
        _load_model(cart_dir)
        _send({"ok": True, "result": {"status": "ready"}})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return

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
            import traceback
            traceback.print_exc()
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
