import json
import os
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.local_inference.bfcl_tool_schema import (
    normalize_functions,
)
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


def _qwen_disable_thinking() -> bool:
    return os.environ.get("QWEN_DISABLE_THINKING", "0") not in ("0", "", "false", "False")


def _qwen_fc_normalize_tools() -> bool:
    """Use the same BFCL schema normalization as the D2L pipelines."""
    return os.environ.get("QWEN_FC_NORMALIZE_TOOLS", "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def _qwen_fc_extra_system_message() -> str | None:
    """Optional system text prepended into the FC tools system block.

    Set ``QWEN_FC_SYSTEM_MESSAGE_FILE`` to a path, or ``QWEN_FC_SYSTEM_MESSAGE``
    to inline text. Used for sparse-bind / always-call ablations.
    """
    path = os.environ.get("QWEN_FC_SYSTEM_MESSAGE_FILE")
    if path:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
        if text:
            return text
    override = os.environ.get("QWEN_FC_SYSTEM_MESSAGE")
    if override and override.strip():
        return override.strip()
    return None


_FORCE_TOOL_CALL_PREFIX = '<tool_call>\n{"name": "'


def _qwen_fc_force_tool_call_prefix() -> bool:
    """If set, prefill assistant with a tool-call opener (binder-style)."""
    return os.environ.get("QWEN_FC_FORCE_TOOL_CALL_PREFIX", "0") not in (
        "0",
        "",
        "false",
        "False",
    )


def _qwen_fc_tool_call_prefix() -> str:
    return _FORCE_TOOL_CALL_PREFIX if _qwen_fc_force_tool_call_prefix() else ""


class QwenFCHandler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        "chat_template":
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].role == 'system' %}
                {{- messages[0].content + '\n\n' }}
            {%- endif %}
            {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
        {%- else %}
            {%- if messages[0].role == 'system' %}
                {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- if message.content is string %}
                {%- set content = message.content %}
            {%- else %}
                {%- set content = '' %}
            {%- endif %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if loop.last or (not loop.last and reasoning_content) %}
                        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                    {%- else %}
                        {{- '<|im_start|>' + message.role + '\n' + content }}
                    {%- endif %}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if (loop.first and content) or (not loop.first) %}
                            {{- '\n' }}
                        {%- endif %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
                        {%- if tool_call.arguments is string %}
                            {{- tool_call.arguments }}
                        {%- else %}
                            {{- tool_call.arguments | tojson }}
                        {%- endif %}
                        {{- '}\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}
        """
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
                    
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(tool_call["arguments"], str):
                            formatted_prompt += tool_call["arguments"]
                        else:
                            formatted_prompt += json.dumps(tool_call["arguments"])
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        # Hard-disable thinking (empty think block), matching
        # tokenizer.apply_chat_template(..., enable_thinking=False).
        if _qwen_disable_thinking():
            formatted_prompt += "<think>\n\n</think>\n\n"
        # Binder-style force: completion must continue a tool call.
        formatted_prompt += _qwen_fc_tool_call_prefix()
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        if _qwen_fc_normalize_tools():
            functions = normalize_functions(functions)

        # FC models use its own tools system prompt. Optional env text is
        # prepended into that block when present (sparse-bind ablations).
        messages: list[dict] = []
        extra = _qwen_fc_extra_system_message()
        if extra:
            messages.append({"role": "system", "content": extra})

        return {"message": messages, "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        # Completions API returns only new tokens; restore forced prefix.
        prefix = _qwen_fc_tool_call_prefix()
        if prefix and not model_response.lstrip().startswith("<tool_call>"):
            model_response = prefix + model_response

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        # Tolerate missing closing tag when a forced prefix was used.
        if prefix and "</tool_call>" not in cleaned_response and "<tool_call>" in cleaned_response:
            cleaned_response = cleaned_response.rstrip() + "\n</tool_call>"

        extracted_tool_calls = self._extract_tool_calls(cleaned_response)

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }

        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }

        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception as e:
                pass
        return result


def _oracle_tool_name_from_ground_truth(test_entry_id: str) -> str:
    """Return the single gold tool name for a BFCL entry id."""
    from bfcl_eval.constants.category_mapping import VERSION_PREFIX
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH
    from bfcl_eval.eval_checker.eval_runner_helper import load_file

    # Cache across calls so SGLang batching does not reload 200 rows each time.
    cache = getattr(_oracle_tool_name_from_ground_truth, "_cache", None)
    if cache is None:
        cache = {}
        for category in ("multiple", "live_simple", "simple"):
            path = POSSIBLE_ANSWER_PATH / f"{VERSION_PREFIX}_{category}.json"
            if not path.exists():
                continue
            for entry in load_file(path):
                names: list[str] = []
                for call in entry.get("ground_truth", []):
                    if isinstance(call, dict):
                        names.extend(call.keys())
                # Keep first name if multiple; BFCL multiple has exactly one.
                if names:
                    cache[entry["id"]] = names[0]
        _oracle_tool_name_from_ground_truth._cache = cache

    if test_entry_id not in cache:
        raise KeyError(f"No oracle tool name for test entry {test_entry_id!r}")
    return cache[test_entry_id]


class QwenFCOracleSingleToolHandler(QwenFCHandler):
    """ICL FC with oracle routing: only the gold tool schema is put in context.

    Intended ablation for staged D2L select→bind: assume routing is perfect,
    then bind with the original (non-normalized) schema via native Qwen FC /
    SGLang. No XGrammar and no forced tool-call prefix unless those env flags
    are set independently.
    """

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        oracle_name = _oracle_tool_name_from_ground_truth(test_entry["id"])
        matches = [f for f in functions if f.get("name") == oracle_name]
        if len(matches) != 1:
            raise ValueError(
                f"{test_entry['id']}: expected exactly one oracle tool "
                f"{oracle_name!r}, found {len(matches)}"
            )
        # Keep the original BFCL schema; do not normalize.
        functions = matches

        messages: list[dict] = []
        extra = _qwen_fc_extra_system_message()
        if extra:
            messages.append({"role": "system", "content": extra})

        return {"message": messages, "function": functions}
