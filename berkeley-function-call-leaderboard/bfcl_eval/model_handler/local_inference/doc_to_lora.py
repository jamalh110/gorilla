"""
Handler for Doc-to-LoRA (D2L) models on BFCL.

D2L models internalize tool definitions into dynamically generated LoRA weights
via a hypernetwork, so tools are NOT placed in the prompt context. This is
fundamentally different from standard prompting or function-calling approaches
where tool schemas appear in the system prompt or as structured tool parameters.

Because the D2L and BFCL dependency trees are incompatible, all D2L-specific
work (model loading, internalization, generation) runs in a **separate
subprocess** (``d2l_worker.py``) under D2L's own virtualenv.  The handler
communicates with the worker via a JSON-lines protocol over stdin/stdout.

Environment variables:
    D2L_CHECKPOINT_PATH  : Path to the D2L checkpoint (pytorch_model.bin)
    D2L_CHUNK_SIZE       : Max tokens per context chunk for internalization (default: 1024)
    D2L_MAX_NEW_TOKENS   : Max tokens to generate (default: 1024)
    D2L_SOURCE_PATH      : Path to the doc-to-lora/src directory
    D2L_PYTHON           : Path to the Python interpreter in D2L's virtualenv
                           (default: ~/tool-lora/doc-to-lora/.venv/bin/python)
    D2L_RESTRICT_TOOLGEN : Set to "1" to enable constrained decoding that
                           restricts function names, parameter names, and enum
                           values to those defined by the tool schema.
"""

import atexit
import json
import os
import re
import subprocess
import time
from typing import Any

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
    """Extract tool calls from <tool_call>...</tool_call> tags in model output.

    Also attempts to parse bare JSON objects with "name" and "arguments" keys
    when tags are missing, as a fallback.
    """
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m))
        except json.JSONDecodeError:
            pass

    if not calls:
        bare_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}'
        bare_matches = re.findall(bare_pattern, text, re.DOTALL)
        for m in bare_matches:
            try:
                obj = json.loads(m)
                if "name" in obj and "arguments" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

    return calls


class _D2LWorkerProxy:
    """Manages the lifecycle of and communication with the d2l_worker subprocess."""

    def __init__(self, python_path: str, d2l_source_path: str):
        self._proc: subprocess.Popen | None = None
        self._python = python_path
        self._src = d2l_source_path

    # ------------------------------------------------------------------

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
            stderr=None,  # inherit → D2L prints appear in terminal
            text=True,
            bufsize=1,  # line-buffered
        )
        atexit.register(self._stop)
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise RuntimeError(
                f"D2L worker failed to start. Response: {ready}"
            )

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

    # ------------------------------------------------------------------

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


class DocToLoraHandler(BaseHandler):
    """BFCL handler for Doc-to-LoRA models.

    Unlike standard OSS handlers that serve models via vLLM and inject tool
    definitions into the prompt, this handler:

    1. Spawns a sandboxed D2L worker process under D2L's own virtualenv.
    2. For each test entry, converts BFCL function docs → JSON → hypernetwork →
       LoRA weights that are applied to the base model (all inside the worker).
    3. Queries the model with ONLY the user message (no tools in context).
    4. Parses ``<tool_call>`` tags from the output.
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
            kwargs.get("max_new_tokens", os.environ.get("D2L_MAX_NEW_TOKENS", "1024"))
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

        self._worker = _D2LWorkerProxy(d2l_python, d2l_source_path)
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        if self._model_loaded:
            return

        if self.checkpoint_path is None:
            raise ValueError(
                "D2L checkpoint path not set. "
                "Set the D2L_CHECKPOINT_PATH environment variable or pass "
                "checkpoint_path as a kwarg."
            )

        result = self._worker.send(
            "load_model", {"checkpoint_path": self.checkpoint_path}
        )
        print(f"D2L model loaded via worker (base: {result.get('base_model_name')})")
        self._model_loaded = True

    def _internalize_tools(self, functions: list[dict]):
        """Convert BFCL function docs to OpenAI-style tool JSON and internalize."""
        tools = [{"type": "function", "function": func} for func in functions]
        tool_defs = json.dumps(tools, indent=2)
        self._worker.send(
            "internalize",
            {"tool_defs": tool_defs, "chunk_size": self.chunk_size},
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        self._ensure_model_loaded()
        if contain_multi_turn_interaction(test_entry["id"]):
            return self.inference_multi_turn_prompting(
                test_entry, include_input_log, exclude_state_log
            )
        else:
            return self.inference_single_turn_prompting(test_entry, include_input_log)

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
    # Prompting-mode methods (called by the @final base-class flows)
    # ------------------------------------------------------------------

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        self._internalize_tools(functions)
        return {
            "message": [{"role": "system", "content": TOOL_CALL_SYSTEM_MSG}],
            "function": functions,
        }

    def _query_prompting(self, inference_data: dict):
        messages: list[dict] = inference_data["message"]

        start_time = time.time()
        result = self._worker.send(
            "generate",
            {
                "messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "restrict_toolgen": self.restrict_toolgen,
            },
        )
        end_time = time.time()

        inference_data["inference_input_log"] = {
            "messages": messages,
            "input_token_count": result["input_tokens"],
        }

        api_response = {
            "text": result["text"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        }
        return api_response, end_time - start_time

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response["text"],
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
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
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
