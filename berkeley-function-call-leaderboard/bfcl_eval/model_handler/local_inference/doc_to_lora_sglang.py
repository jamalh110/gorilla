"""
Handler for Doc-to-LoRA models on BFCL using SGLang for fast batched inference.

Architecture (two-phase approach):
  Phase 1 – LoRA generation (``prepare``):
      For each unique set of tool definitions in the benchmark, the D2L
      hypernetwork generates LoRA weights and exports them as standard PEFT
      adapters to disk.  The D2L worker subprocess is shut down after this
      phase to free GPU memory.

  Phase 2 – Batched SGLang serving (``prepare``):
      Adapters are grouped into batches (default 8, set via
      ``D2L_ADAPTER_BATCH_SIZE``).  For each batch, an SGLang server is
      spun up with only that batch's adapters, inference is run for all
      corresponding test cases, and the server is shut down before the
      next batch.  Results are cached so that subsequent ``inference()``
      calls are instant lookups.

Environment variables (all also accepted by the original DocToLoraHandler):
    D2L_CHECKPOINT_PATH  : Path to the D2L checkpoint (pytorch_model.bin)
    D2L_CHUNK_SIZE       : Max tokens per context chunk (default: 1024)
    D2L_MAX_NEW_TOKENS   : Max tokens to generate (default: 1024)
    D2L_SOURCE_PATH      : Path to the doc-to-lora/src directory
    D2L_PYTHON           : D2L virtualenv Python interpreter
    D2L_RESTRICT_TOOLGEN : "1" to enable constrained decoding via structural tags
    D2L_ADAPTER_DIR      : Directory to cache PEFT adapters (default: /tmp/d2l_adapters)
    D2L_ADAPTER_BATCH_SIZE : Max adapters loaded per SGLang server cycle (default: 8)
"""

import atexit
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
from tqdm import tqdm

from bfcl_eval.constants.eval_config import LOCAL_SERVER_PORT
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from bfcl_eval.utils import contain_multi_turn_interaction

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "d2l_worker.py")

TOOL_CALL_SYSTEM_MSG = (
    "You may call one or more functions to assist with the user query.\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


def _parse_tool_calls(text: str) -> list[dict]:
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m))
        except json.JSONDecodeError:
            pass

    if not calls:
        bare_pattern = (
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        )
        bare_matches = re.findall(bare_pattern, text, re.DOTALL)
        for m in bare_matches:
            try:
                obj = json.loads(m)
                if "name" in obj and "arguments" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
    return calls


# ---------------------------------------------------------------------------
# Tool hash
# ---------------------------------------------------------------------------

def _tools_hash(functions: list[dict]) -> str:
    """Deterministic hash of a tool set; used as the adapter name."""
    blob = json.dumps(functions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Structural tag builder for constrained decoding
# ---------------------------------------------------------------------------

def _bfcl_param_to_json_schema(pdef: dict) -> dict:
    """Convert a single BFCL parameter definition to a JSON-Schema property."""
    t = pdef.get("type", "string")

    type_map = {"dict": "object", "tuple": "array", "float": "number"}
    json_type = type_map.get(t, t)

    prop: dict[str, Any] = {"type": json_type}

    if "enum" in pdef:
        prop["enum"] = pdef["enum"]

    if json_type == "array" and "items" in pdef:
        prop["items"] = _bfcl_param_to_json_schema(pdef["items"])

    if json_type == "object" and "properties" in pdef:
        inner = {}
        for k, v in pdef["properties"].items():
            inner[k] = _bfcl_param_to_json_schema(v)
        prop["properties"] = inner

    return prop


def _build_args_schema(func: dict) -> dict:
    """Build a JSON schema for a function's arguments object."""
    params = func.get("parameters", {})
    props_raw = params.get("properties", {})

    properties = {}
    for pname, pdef in props_raw.items():
        properties[pname] = _bfcl_param_to_json_schema(pdef)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    required = params.get("required")
    if required:
        schema["required"] = list(required)

    return schema


def build_structural_tag(functions: list[dict]) -> dict | None:
    """Build an SGLang structural_tag for tool-call constrained decoding.

    Uses ``triggered_tags`` so that:
      - Free-form text (including ``<think>`` blocks) is allowed.
      - When ``<tool_call>`` is encountered, the grammar switches to a
        ``json_schema`` constraint for the tool call JSON, restricted to
        the valid functions and their argument schemas.
    """
    if not functions:
        return None

    tool_call_schema = _build_tool_call_schema(functions)

    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<tool_call>"],
            "tags": [
                {
                    "begin": "<tool_call>\n",
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool_call_schema,
                    },
                    "end": "\n</tool_call>",
                }
            ],
            "at_least_one": False,
            "stop_after_first": False,
        },
    }
    return structural_tag


def _build_tool_call_schema(functions: list[dict]) -> dict:
    """Build a JSON schema for the full tool call object.

    If there is only one function, the schema uses ``const`` for the name.
    Otherwise, it uses ``anyOf`` to represent the union of all functions
    with their specific argument schemas.
    """
    if len(functions) == 1:
        func = functions[0]
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": func["name"]},
                "arguments": _build_args_schema(func),
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }

    variants = []
    for func in functions:
        variants.append(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": func["name"]},
                    "arguments": _build_args_schema(func),
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    return {"anyOf": variants}


# ---------------------------------------------------------------------------
# D2L Worker Proxy (same as in doc_to_lora.py)
# ---------------------------------------------------------------------------

class _D2LWorkerProxy:
    def __init__(self, python_path: str, d2l_source_path: str):
        self._proc: subprocess.Popen | None = None
        self._python = python_path
        self._src = d2l_source_path

    def _ensure_alive(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._start()

    def _start(self):
        self._stop()
        self._proc = subprocess.Popen(
            [self._python, _WORKER_SCRIPT, self._src],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        atexit.register(self._stop)
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise RuntimeError(f"D2L worker failed to start. Response: {ready}")

    def _stop(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def _read_response(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.wait()
            raise RuntimeError(
                f"D2L worker exited unexpectedly (return code {rc})"
            )
        resp = json.loads(line)
        if not resp["ok"]:
            raise RuntimeError(f"D2L worker error: {resp['error']}")
        return resp.get("result", {})

    def send(self, cmd: str, args: dict | None = None) -> dict:
        self._ensure_alive()
        payload = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()
        return self._read_response()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class DocToLoraSGLangHandler(BaseHandler):
    """Two-phase D2L handler: generate LoRA adapters, then serve via SGLang.

    Supports optional structural-tag constrained decoding for tool calls.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

        self.checkpoint_path = kwargs.get(
            "checkpoint_path", os.environ.get("D2L_CHECKPOINT_PATH")
        )
        self.chunk_size = int(
            kwargs.get("chunk_size", os.environ.get("D2L_CHUNK_SIZE", "1024"))
        )
        self.max_new_tokens = int(
            kwargs.get(
                "max_new_tokens", os.environ.get("D2L_MAX_NEW_TOKENS", "1024")
            )
        )
        self.restrict_toolgen = str(
            kwargs.get(
                "restrict_toolgen",
                os.environ.get("D2L_RESTRICT_TOOLGEN", "0"),
            )
        ) not in ("0", "", "false", "False")
        d2l_source_path = kwargs.get(
            "d2l_source_path",
            os.environ.get(
                "D2L_SOURCE_PATH",
                os.path.expanduser("~/tool-lora/doc-to-lora/src"),
            ),
        )
        d2l_python = kwargs.get(
            "d2l_python",
            os.environ.get(
                "D2L_PYTHON",
                os.path.expanduser("~/tool-lora/doc-to-lora/.venv/bin/python"),
            ),
        )
        self._adapter_dir = kwargs.get(
            "adapter_dir",
            os.environ.get("D2L_ADAPTER_DIR", "/tmp/d2l_adapters"),
        )

        self._worker = _D2LWorkerProxy(d2l_python, d2l_source_path)
        self._model_loaded = False
        self._base_model_name: str | None = None
        self._adapter_map: dict[str, str] = {}  # adapter_name -> disk path
        self._adapter_ranks: dict[str, int] = {}  # adapter_name -> merged rank
        self._cached_results: dict[str, tuple] = {}
        self._sglang_proc: subprocess.Popen | None = None
        self.client: openai.OpenAI | None = None
        self._port = int(os.environ.get("D2L_SGLANG_PORT", str(LOCAL_SERVER_PORT)))
        self._batch_size = int(os.environ.get("D2L_ADAPTER_BATCH_SIZE", "8"))

    # ------------------------------------------------------------------
    # Phase 1: LoRA generation
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        if self._model_loaded:
            return
        if self.checkpoint_path is None:
            raise ValueError(
                "D2L checkpoint path not set. "
                "Set D2L_CHECKPOINT_PATH or pass checkpoint_path."
            )
        result = self._worker.send(
            "load_model", {"checkpoint_path": self.checkpoint_path}
        )
        self._base_model_name = result["base_model_name"]
        print(f"[D2L-SGLang] Model loaded (base: {self._base_model_name})")
        self._model_loaded = True

    def prepare(self, test_cases: list[dict]):
        """Generate all PEFT adapters, then run batched inference.

        Cycles through adapter batches (default 8 at a time), spinning up a
        fresh SGLang server for each batch, running inference for every test
        case that uses one of the batch's adapters, and caching the results
        so that the later ``inference()`` calls are instant lookups.
        """
        self._ensure_model_loaded()
        os.makedirs(self._adapter_dir, exist_ok=True)

        # -- Phase 1: generate all adapters --
        unique_tool_sets: dict[str, list[dict]] = {}
        for tc in test_cases:
            funcs = tc["function"]
            h = _tools_hash(funcs)
            if h not in unique_tool_sets:
                unique_tool_sets[h] = funcs

        print(
            f"[D2L-SGLang] Generating {len(unique_tool_sets)} unique LoRA adapters..."
        )

        for adapter_name, funcs in tqdm(
            unique_tool_sets.items(), desc="Generating LoRAs"
        ):
            adapter_path = os.path.join(self._adapter_dir, adapter_name)
            if os.path.isfile(
                os.path.join(adapter_path, "adapter_model.safetensors")
            ):
                cfg_path = os.path.join(adapter_path, "adapter_config.json")
                with open(cfg_path) as f:
                    rank = json.load(f)["r"]
                self._adapter_ranks[adapter_name] = rank
                self._adapter_map[adapter_name] = adapter_path
                continue

            tools = [{"type": "function", "function": f} for f in funcs]
            tool_defs = json.dumps(tools, indent=2)
            result = self._worker.send(
                "internalize_and_export",
                {
                    "tool_defs": tool_defs,
                    "output_dir": adapter_path,
                    "chunk_size": self.chunk_size,
                },
            )
            self._adapter_ranks[adapter_name] = result["merged_rank"]
            self._adapter_map[adapter_name] = adapter_path

        print(f"[D2L-SGLang] All adapters generated.")
        self._worker._stop()

        # -- Phase 2: batched inference --
        adapter_to_cases: dict[str, list[dict]] = defaultdict(list)
        for tc in test_cases:
            adapter_to_cases[_tools_hash(tc["function"])].append(tc)

        adapter_names = list(self._adapter_map.keys())
        total_batches = -(-len(adapter_names) // self._batch_size)  # ceil div
        print(
            f"[D2L-SGLang] Running batched inference: "
            f"{len(adapter_names)} adapters in {total_batches} batches "
            f"of up to {self._batch_size}"
        )

        for batch_start in range(0, len(adapter_names), self._batch_size):
            batch_names = adapter_names[batch_start:batch_start + self._batch_size]
            batch_map = {n: self._adapter_map[n] for n in batch_names}
            max_rank = max(self._adapter_ranks[n] for n in batch_names)

            batch_cases = []
            for name in batch_names:
                batch_cases.extend(adapter_to_cases.get(name, []))

            batch_num = batch_start // self._batch_size + 1
            print(
                f"\n[D2L-SGLang] Batch {batch_num}/{total_batches}: "
                f"{len(batch_names)} adapters, {len(batch_cases)} test cases"
            )

            self._spin_up_sglang(batch_map, max_rank)

            with ThreadPoolExecutor(max_workers=len(batch_names)) as pool:
                futures = {
                    pool.submit(
                        self._run_single_inference, tc
                    ): tc["id"]
                    for tc in batch_cases
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Batch {batch_num}/{total_batches}",
                ):
                    tc_id = futures[future]
                    try:
                        self._cached_results[tc_id] = future.result()
                    except Exception as e:
                        tqdm.write(
                            f"❗️ Error for {tc_id}: {e}\n"
                            + traceback.format_exc(limit=5)
                        )
                        self._cached_results[tc_id] = (
                            f"Error during inference: {e}",
                            {},
                        )

            self.shutdown()

        print(
            f"\n[D2L-SGLang] All {len(self._cached_results)}/{len(test_cases)} "
            f"test cases processed."
        )

    def _run_single_inference(self, test_entry: dict) -> tuple:
        """Run inference for one test case (called from batched prepare)."""
        return self.inference(
            test_entry, include_input_log=False, exclude_state_log=False
        )

    # ------------------------------------------------------------------
    # SGLang server management
    # ------------------------------------------------------------------

    def _spin_up_sglang(
        self,
        adapter_map: dict[str, str] | None = None,
        max_lora_rank: int | None = None,
    ):
        if adapter_map is None:
            adapter_map = self._adapter_map
        if max_lora_rank is None:
            max_lora_rank = max(self._adapter_ranks.values(), default=8)

        lora_path_specs = [
            f"{name}={path}" for name, path in adapter_map.items()
        ]

        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self._base_model_name,
            "--port", str(self._port),
            "--host", "0.0.0.0",
            "--max-loras-per-batch", str(len(adapter_map)),
            "--max-lora-rank", str(max_lora_rank),
        ]
        if lora_path_specs:
            cmd.append("--lora-paths")
            cmd.extend(lora_path_specs)

        print(
            f"[D2L-SGLang] Starting SGLang server with "
            f"{len(adapter_map)} adapters (max rank {max_lora_rank})..."
        )
        sglang_log = os.path.join(self._adapter_dir, "sglang_server.log")
        self._sglang_log_file = open(sglang_log, "w")
        print(f"[D2L-SGLang] Server log: {sglang_log}")
        self._sglang_proc = subprocess.Popen(
            cmd,
            stdout=self._sglang_log_file,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.shutdown)

        base_url = f"http://localhost:{self._port}"
        self.client = openai.OpenAI(base_url=f"{base_url}/v1", api_key="EMPTY")

        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                self.client.models.list()
                print(
                    f"[D2L-SGLang] Server is ready with "
                    f"{len(adapter_map)} adapters."
                )
                return
            except Exception:
                if self._sglang_proc.poll() is not None:
                    self._sglang_log_file.flush()
                    sglang_log = os.path.join(self._adapter_dir, "sglang_server.log")
                    with open(sglang_log) as f:
                        out = f.read()
                    raise RuntimeError(
                        f"SGLang server exited prematurely:\n{out}"
                    )
                time.sleep(3)

        raise TimeoutError("SGLang server did not become ready within 600s")

    def shutdown(self):
        if self._sglang_proc is not None and self._sglang_proc.poll() is None:
            print("[D2L-SGLang] Shutting down SGLang server...")
            os.kill(self._sglang_proc.pid, signal.SIGTERM)
            try:
                self._sglang_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._sglang_proc.kill()
                self._sglang_proc.wait()
            self._sglang_proc = None
        if hasattr(self, "_sglang_log_file") and self._sglang_log_file:
            self._sglang_log_file.close()
            self._sglang_log_file = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        cached = self._cached_results.pop(test_entry["id"], None)
        if cached is not None:
            return cached
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(
                test_entry, include_input_log
            )

    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = _parse_tool_calls(result)
        if tool_calls:
            return [{tc["name"]: tc.get("arguments", {})} for tc in tool_calls]
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = _parse_tool_calls(result)
        if tool_calls:
            execution_list = []
            for tc in tool_calls:
                name = tc["name"]
                params = tc.get("arguments", {})
                args_str = ",".join(f"{k}={repr(v)}" for k, v in params.items())
                execution_list.append(f"{name}({args_str})")
            return execution_list
        return default_decode_execute_prompting(result, has_tool_call_tag)

    # ------------------------------------------------------------------
    # Prompting-mode methods
    # ------------------------------------------------------------------

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        adapter_name = _tools_hash(functions)
        return {
            "message": [{"role": "system", "content": TOOL_CALL_SYSTEM_MSG}],
            "function": functions,
            "adapter_name": adapter_name,
        }

    def _query_prompting(self, inference_data: dict):
        messages: list[dict] = inference_data["message"]
        adapter_name = inference_data["adapter_name"]
        functions = inference_data["function"]

        model_str = f"{self._base_model_name}:{adapter_name}"

        extra_body: dict[str, Any] = {}
        if self.restrict_toolgen:
            tag = build_structural_tag(functions)
            if tag is not None:
                extra_body["response_format"] = tag

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=model_str,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            extra_body=extra_body if extra_body else None,
        )
        end_time = time.time()

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        inference_data["inference_input_log"] = {
            "messages": messages,
            "input_token_count": usage.prompt_tokens if usage else 0,
        }

        api_response = {
            "text": text,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }
        return api_response, end_time - start_time

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw = api_response["text"]
        reasoning_content = ""
        cleaned = raw
        if "</think>" in raw:
            parts = raw.split("</think>")
            reasoning_content = (
                parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            )
            cleaned = parts[-1].lstrip("\n")
        return {
            "model_responses": cleaned,
            "reasoning_content": reasoning_content,
            "input_token": api_response["input_tokens"],
            "output_token": api_response["output_tokens"],
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        msg = {
            "role": "assistant",
            "content": model_response_data["model_responses"],
        }
        reasoning = model_response_data.get("reasoning_content", "")
        if reasoning:
            msg["reasoning_content"] = reasoning
        inference_data["message"].append(msg)
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded_model_response,
                    "content": execution_result,
                }
            )
        return inference_data
