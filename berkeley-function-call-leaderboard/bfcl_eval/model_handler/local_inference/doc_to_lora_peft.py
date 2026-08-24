"""
BFCL handler for pre-exported / schema-SFT PEFT adapters (HF, one adapter at a time).

Does NOT use SGLang. Loads Qwen3-4B-Instruct + one PEFT adapter per toolset via
a D2L-venv subprocess worker, groups live_simple cases by normalized tools hash,
and caches generations for subsequent inference() lookups.

Environment variables:
    D2L_ADAPTER_DIR   : Root directory of adapters keyed by normalized tools hash
    D2L_BASE_MODEL    : Base HF model (default: Qwen/Qwen3-4B-Instruct-2507)
    D2L_PYTHON        : D2L venv python
    D2L_ROOT          : doc-to-lora repo root (for chat templates)
    D2L_MAX_NEW_TOKENS: Max generation tokens (default 1024)
    D2L_RAW_LOG       : Optional raw JSONL log path
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any

from tqdm import tqdm

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.bfcl_tool_schema import tools_hash
from bfcl_eval.model_handler.local_inference.doc_to_lora import (
    TOOL_CALL_SYSTEM_MSG,
    _D2LWorkerProxy,
    _parse_tool_calls,
)
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
)
from bfcl_eval.utils import contain_multi_turn_interaction

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "peft_worker.py")


class DocToLoraPeftHandler(BaseHandler):
    """Serve per-toolset PEFT adapters with HF generate (no hypernet, no SGLang)."""

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

        self.adapter_dir = kwargs.get(
            "adapter_dir",
            os.environ.get("D2L_ADAPTER_DIR"),
        )
        if not self.adapter_dir:
            raise ValueError(
                "D2L_ADAPTER_DIR must point to the post-SFT (or exported) adapter root"
            )
        self.base_model = kwargs.get(
            "base_model",
            os.environ.get("D2L_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        )
        self.max_new_tokens = int(
            kwargs.get("max_new_tokens", os.environ.get("D2L_MAX_NEW_TOKENS", "1024"))
        )
        d2l_root = kwargs.get(
            "d2l_root",
            os.environ.get(
                "D2L_ROOT",
                os.path.expanduser("~/tool-lora/doc-to-lora"),
            ),
        )
        d2l_python = kwargs.get(
            "d2l_python",
            os.environ.get(
                "D2L_PYTHON",
                os.path.expanduser("~/tool-lora/doc-to-lora/.venv/bin/python"),
            ),
        )

        gpu_ids = self._detect_gpu_ids()
        self._worker = _D2LWorkerProxy(
            d2l_python,
            d2l_root,
            gpu_device=gpu_ids[0],
            worker_script=_WORKER_SCRIPT,
            worker_args=[d2l_root],
        )
        self._base_loaded = False
        self._cached_results: dict[str, tuple] = {}
        self._active_adapter_path: str | None = None
        raw_log_path = os.environ.get("D2L_RAW_LOG", "")
        if raw_log_path:
            os.makedirs(os.path.dirname(raw_log_path) or ".", exist_ok=True)
            self._raw_log = open(raw_log_path, "w")
        else:
            self._raw_log = None

        print(
            f"DocToLoraPeftHandler: adapters={self.adapter_dir} "
            f"base={self.base_model} gpu={gpu_ids[0]}"
        )

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

    def _ensure_base(self):
        if self._base_loaded:
            return
        self._worker.send("load_base", {"base_model": self.base_model})
        self._base_loaded = True

    def _adapter_path_for(self, functions: list[dict]) -> str:
        h = tools_hash(functions)
        path = os.path.join(self.adapter_dir, h)
        if not os.path.isfile(os.path.join(path, "adapter_model.safetensors")):
            raise FileNotFoundError(
                f"Missing PEFT adapter for tools hash {h} under {self.adapter_dir}"
            )
        return path

    def _set_adapter(self, adapter_path: str):
        if self._active_adapter_path == adapter_path:
            return
        self._worker.send("load_adapter", {"adapter_path": adapter_path})
        self._active_adapter_path = adapter_path

    def prepare(self, test_cases: list[dict]):
        """Group by adapter, load one at a time, cache all generations."""
        self._ensure_base()

        by_hash: dict[str, list[dict]] = defaultdict(list)
        hash_to_path: dict[str, str] = {}
        for tc in test_cases:
            h = tools_hash(tc["function"])
            by_hash[h].append(tc)
            if h not in hash_to_path:
                hash_to_path[h] = self._adapter_path_for(tc["function"])

        print(
            f"[D2L-PEFT] Preparing {len(test_cases)} cases across "
            f"{len(by_hash)} adapters (one at a time)..."
        )

        for i, (h, cases) in enumerate(sorted(by_hash.items()), 1):
            adapter_path = hash_to_path[h]
            print(
                f"[D2L-PEFT] [{i}/{len(by_hash)}] adapter {h} "
                f"({len(cases)} cases)"
            )
            self._set_adapter(adapter_path)
            for tc in tqdm(cases, desc=f"adapter {h}", leave=False):
                result = self.inference_single_turn_prompting(
                    tc, include_input_log=False
                )
                self._cached_results[tc["id"]] = result

        self._worker.send("unload_adapter", {})
        self._active_adapter_path = None
        print(f"[D2L-PEFT] Cached {len(self._cached_results)} generations.")

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        if contain_multi_turn_interaction(test_entry["id"]):
            raise NotImplementedError(
                "DocToLoraPeftHandler only supports single-turn categories"
            )
        cached = self._cached_results.pop(test_entry["id"], None)
        if cached is not None:
            return cached

        self._ensure_base()
        self._set_adapter(self._adapter_path_for(test_entry["function"]))
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

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        return {
            "id": test_entry.get("id"),
            "message": [{"role": "system", "content": TOOL_CALL_SYSTEM_MSG}],
            "function": test_entry["function"],
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _query_prompting(self, inference_data: dict):
        messages: list[dict] = inference_data["message"]
        start_time = time.time()
        result = self._worker.send(
            "generate",
            {
                "messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "enable_thinking": False,
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
            self._raw_log.write(
                json.dumps(
                    {
                        "id": inference_data.get("id"),
                        "messages": messages,
                        "functions": inference_data.get("function", []),
                        "raw_output": result["text"],
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "latency": end_time - start_time,
                        "adapter_path": result.get("adapter_path"),
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
        for execution_result, decoded in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            inference_data["message"].append(
                {
                    "role": "tool",
                    "name": decoded,
                    "content": execution_result,
                }
            )
        return inference_data
