"""BFCL handler for self-study Cartridge models (no reader LoRA).

This is the baseline from the original Cartridges paper (HazyResearch/cartridges)
adapted to BFCL function calling: each tool schema is compressed into a trainable
KV-cache prefix ("cartridge") via self-study + context distillation, offline.
At eval time the tool schema is NOT placed in the prompt — the pretrained
cartridge for the entry's tool is injected as the KV prefix and the model is
queried with only the user message.

Like the Doc-to-LoRA handler, all cartridge-specific work (FlexAttention model,
TrainableCache) runs in a separate subprocess (``cartridge_worker.py``) under the
cartridges repo's own virtualenv, since its torch/FlexAttention build is
incompatible with the BFCL environment. Communication is via JSON-lines over
stdin/stdout.

Environment variables:
    CART_REPO_PATH : Path to the cloned cartridges repo
                     (default: ~/cartridges)
    CART_PYTHON    : Python interpreter in the cartridges venv
                     (default: ~/cartridges/.venv/bin/python)
    CART_DIR       : Directory containing the pretrained cartridges (*.pt named
                     ``<name_slug>__<hash>.pt``)
                     (default: ~/cartridges/bfcl_runs/cartridges/p128)
    CART_MAX_NEW_TOKENS : Max tokens to generate (default: 256)
"""
import atexit
import json
import os
import queue
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from bfcl_eval.utils import contain_multi_turn_interaction

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "cartridge_worker.py")

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
        bare_pattern = r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
        for m in re.findall(bare_pattern, text, re.DOTALL):
            try:
                obj = json.loads(m)
                if "name" in obj and "arguments" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
    return calls


class _CartridgeWorkerProxy:
    def __init__(self, python_path: str, repo_path: str, cart_dir: str, gpu_device: str | None = None):
        self._proc: subprocess.Popen | None = None
        self._python = python_path
        self._repo = repo_path
        self._cart_dir = cart_dir
        self.gpu_device = gpu_device
        self.model_loaded = False

    def _ensure_alive(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._start()

    def _start(self):
        self._stop()
        env = os.environ.copy()
        if self.gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_device
        env["CARTRIDGES_DIR"] = self._repo
        self._proc = subprocess.Popen(
            [self._python, _WORKER_SCRIPT, self._repo, self._cart_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        atexit.register(self._stop)
        ready = self._read_response()
        if ready.get("status") != "ready":
            raise RuntimeError(f"Cartridge worker failed to start. Response: {ready}")
        self.model_loaded = True

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
            raise RuntimeError(f"Cartridge worker exited unexpectedly (return code {rc})")
        resp = json.loads(line)
        if not resp["ok"]:
            raise RuntimeError(f"Cartridge worker error: {resp['error']}")
        return resp.get("result", {})

    def send(self, cmd: str, args: dict | None = None) -> dict:
        self._ensure_alive()
        payload = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()
        return self._read_response()


class CartridgeHandler(BaseHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

        self.max_new_tokens = int(
            kwargs.get("max_new_tokens", os.environ.get("CART_MAX_NEW_TOKENS", "256"))
        )
        repo_path = kwargs.get(
            "repo_path", os.environ.get("CART_REPO_PATH", os.path.expanduser("~/cartridges"))
        )
        cart_python = kwargs.get(
            "cart_python",
            os.environ.get("CART_PYTHON", os.path.expanduser("~/cartridges/.venv/bin/python")),
        )
        cart_dir = kwargs.get(
            "cart_dir",
            os.environ.get(
                "CART_DIR",
                os.path.expanduser("~/cartridges/bfcl_runs/cartridges/p128"),
            ),
        )

        gpu_ids = self._detect_gpu_ids()
        self._worker_pool: queue.Queue[_CartridgeWorkerProxy] = queue.Queue()
        self._all_workers: list[_CartridgeWorkerProxy] = []
        for gpu_dev in gpu_ids:
            worker = _CartridgeWorkerProxy(cart_python, repo_path, cart_dir, gpu_device=gpu_dev)
            self._all_workers.append(worker)
            self._worker_pool.put(worker)
        self._thread_local = threading.local()
        self._log_lock = threading.Lock()
        self.num_gpus = len(gpu_ids)
        print(f"CartridgeHandler: {self.num_gpus} GPU(s) → {gpu_ids}; cart_dir={cart_dir}")

        raw_log_path = os.environ.get("CART_RAW_LOG", "")
        if raw_log_path:
            os.makedirs(os.path.dirname(raw_log_path) or ".", exist_ok=True)
            self._raw_log = open(raw_log_path, "w")
        else:
            self._raw_log = None

    @staticmethod
    def _detect_gpu_ids() -> list[str]:
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_vis:
            ids = [g.strip() for g in cuda_vis.split(",") if g.strip()]
            if ids:
                return ids
        try:
            import torch

            n = torch.cuda.device_count()
            if n > 0:
                return [str(i) for i in range(n)]
        except Exception:
            pass
        return ["0"]

    @contextmanager
    def _checkout_worker(self):
        worker = self._worker_pool.get()
        self._thread_local.worker = worker
        try:
            yield worker
        finally:
            self._thread_local.worker = None
            self._worker_pool.put(worker)

    @property
    def _active_worker(self) -> _CartridgeWorkerProxy:
        w = getattr(self._thread_local, "worker", None)
        if w is None:
            raise RuntimeError("No cartridge worker checked out for the current thread.")
        return w

    # ------------------------------------------------------------------

    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        with self._checkout_worker():
            self._active_worker._ensure_alive()
            if contain_multi_turn_interaction(test_entry["id"]):
                return self.inference_multi_turn_prompting(
                    test_entry, include_input_log, exclude_state_log
                )
            return self.inference_single_turn_prompting(test_entry, include_input_log)

    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = _parse_tool_calls(result)
        if tool_calls:
            return [{tc["name"]: tc.get("arguments", {})} for tc in tool_calls]
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = _parse_tool_calls(result)
        if tool_calls:
            out = []
            for tc in tool_calls:
                params = tc.get("arguments", {})
                args_str = ",".join(f"{k}={repr(v)}" for k, v in params.items())
                out.append(f"{tc['name']}({args_str})")
            return out
        return default_decode_execute_prompting(result, has_tool_call_tag)

    # ------------------------------------------------------------------

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        return {
            "id": test_entry.get("id"),
            "message": [{"role": "system", "content": TOOL_CALL_SYSTEM_MSG}],
            "function": functions,
        }

    def _query_prompting(self, inference_data: dict):
        messages: list[dict] = inference_data["message"]
        functions = inference_data.get("function", [])
        function = functions[0] if functions else {}

        start_time = time.time()
        result = self._active_worker.send(
            "generate",
            {
                "messages": messages,
                "function": function,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
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
        if self._raw_log is not None:
            with self._log_lock:
                self._raw_log.write(
                    json.dumps(
                        {
                            "id": inference_data.get("id"),
                            "messages": messages,
                            "function": function,
                            "raw_output": result["text"],
                            "missing_cartridge": result.get("missing", False),
                            "latency": end_time - start_time,
                        }
                    )
                    + "\n"
                )
                self._raw_log.flush()
        return api_response, end_time - start_time

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw = api_response["text"]
        reasoning_content = ""
        cleaned = raw
        if "</think>" in raw:
            parts = raw.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned = parts[-1].lstrip("\n")
        return {
            "model_responses": cleaned,
            "reasoning_content": reasoning_content,
            "input_token": api_response["input_tokens"],
            "output_token": api_response["output_tokens"],
        }

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(
            {"role": "assistant", "content": model_response_data["model_responses"]}
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        for execution_result, decoded in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {"role": "tool", "name": decoded, "content": execution_result}
            )
        return inference_data
