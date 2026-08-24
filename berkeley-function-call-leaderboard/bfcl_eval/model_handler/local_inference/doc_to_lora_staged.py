"""Strict two-stage Doc-to-LoRA handlers for BFCL single-turn categories.

Variant A routes and writes a schema-neutral intent with the D2L 4B model, then
binds that intent with a separate non-thinking Qwen3-0.6B worker.  Variant B
selects a tool with D2L 4B and performs a fresh schema-plus-query pass through
the same 4B model while retaining the all-tools generated LoRA.
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from copy import deepcopy
from typing import Any

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.doc_to_lora import (
    DocToLoraHandler,
    _D2LWorkerProxy,
)

_BINDER_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "binder_worker.py")
def _assistant_prefix(
    function: dict, *, force_first_required: bool = False
) -> str:
    """Use strict full-JSON generation first and a bounded legacy prefill fallback."""
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise StagedValidationError("constraint function needs a non-empty name")
    if not force_first_required:
        return "<tool_call>\n"
    prefix = (
        "<tool_call>\n"
        + '{"name":'
        + json.dumps(name, ensure_ascii=False)
        + ',"arguments":{'
    )
    required = function.get("parameters", {}).get("required", [])
    if required and isinstance(required[0], str):
        prefix += json.dumps(required[0], ensure_ascii=False) + ":"
    return prefix


def _append_missing_end_tag(text: str) -> str:
    """Repair a complete JSON payload's wrapper/boundary artifacts."""
    if (
        not text.startswith("<tool_call>")
        or text.count("<tool_call>") != 1
        or text.count("</tool_call>") > 1
    ):
        return text
    payload = text[len("<tool_call>") :].lstrip().replace("：", ":").replace("，", ",")
    # Identifier-trie tokens can straddle the closing quote of the prefixed
    # meta-tool enum value, producing `"tool_name":"value""}`. Remove exactly
    # that duplicate boundary quote; retain the untouched worker text in traces.
    repaired_payload = re.sub(
        r'("tool_name"\s*:\s*"[^"\r\n]+)""(?=\s*})',
        r'\1"',
        payload,
        count=1,
    )
    if repaired_payload == payload and "</tool_call>" in text:
        return text
    payload = repaired_payload
    try:
        _, end = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError:
        return text
    trailing = payload[end:].strip()
    if trailing == "<tool_call>":
        trailing = ""
    elif "<tool_call>" in trailing:
        return text
    # The identifier trie can permit a token that straddles a constrained
    # string boundary. Keep the first complete JSON object and discard only
    # trailing wrapper/prose artifacts; schema validation still runs below.
    payload = payload[:end]
    repaired = "<tool_call>\n" + payload
    return repaired.rstrip() + "\n</tool_call>"


def _repair_schema_key_syntax(text: str, function: dict) -> str:
    """Repair bounded punctuation errors around keys known to the schema."""
    keys: set[str] = set()

    def collect(schema: Any) -> None:
        if not isinstance(schema, dict):
            return
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            keys.update(key for key in properties if isinstance(key, str))
            for child in properties.values():
                collect(child)
        collect(schema.get("items"))

    collect(function.get("parameters", {}))
    repaired = text
    for key in sorted(keys, key=len, reverse=True):
        if not key:
            continue
        quoted = json.dumps(key, ensure_ascii=False)
        # `{key:` / `, key:` -> `{"key":` / `, "key":`
        repaired = re.sub(
            rf"([{{,]\s*){re.escape(key)}\s*:",
            lambda match, quoted=quoted: match.group(1) + quoted + ":",
            repaired,
        )
        # `{"key" value}` -> `{"key": value}` for JSON value starts.
        repaired = re.sub(
            rf"([{{,]\s*){re.escape(quoted)}\s*"
            rf'(?=(?:"|\[|\{{|-?\d|true\b|false\b|null\b))',
            lambda match, quoted=quoted: match.group(1) + quoted + ":",
            repaired,
        )
        # `{"key"=value}` -> `{"key":value}`. Some heavily format-trained
        # checkpoints substitute assignment punctuation while preserving the
        # exact schema key and value; no value content is changed here.
        repaired = re.sub(
            rf"([{{,]\s*{re.escape(quoted)}\s*)=",
            lambda match: match.group(1) + ":",
            repaired,
        )
    return repaired
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
META_CALL_SYSTEM_PREFIX = (
    "This conversation uses two-stage tool calling. First, call exactly the "
    "meta-tool below; do not call a real tool before the tool response arrives. "
    "Return only one JSON call inside <tool_call></tool_call> tags, with "
    "tool_name set to one enum value. After that call, you will receive a "
    "<tool_response> containing the selected real tool schema. Then call that "
    "real tool exactly once for the original user request, using its exact "
    "function name and parameter keys. Your second assistant turn must have "
    "exactly this complete form, not a bare arguments object: "
    "<tool_call>\n{\"name\":\"<selected real name>\",\"arguments\":{...}}"
    "\n</tool_call>. Do not repeat the meta-tool after the tool response and "
    "do not emit prose.\n\n"
    "Meta-tool schema:\n"
)
# Baseline (no Doc-to-LoRA): stage-1 only exposes candidate names via the meta
# tool. The model must pick a name; full schemas arrive only in stage-2.
BASELINE_SELECT_SYSTEM_PREFIX = (
    "You are running a two-stage tool-calling protocol without internalized "
    "tool knowledge. In this first stage you only see candidate tool names "
    "(no parameter schemas yet). Choose the single best tool name for the "
    "user query by calling the meta-tool below exactly once. Do not invent "
    "tool names outside the enum, and do not emit a real tool call yet. "
    "Return only one JSON tool call inside <tool_call></tool_call> tags.\n\n"
    "Meta-tool schema:\n"
)
BINDER_SYSTEM_MESSAGE = (
    "Bind the schema-neutral intent into exactly one call to the provided "
    "tool. Map fact values onto the tool's parameter types, enums, formats, "
    "and description conventions (canonical spellings, codes, and units). "
    "Use the exact function name and parameter keys from the tool schema. "
    "Omit parameters that are not supported by the intent facts; do not "
    "invent optional arguments. Return only one call in "
    "<tool_call></tool_call> tags."
)
B_JOINT_BIND_SYSTEM_MESSAGE = (
    "Call exactly the supplied real tool for the original user query. Use the "
    "exact function name and parameter keys in the selected schema. Return "
    "only one call in <tool_call></tool_call> tags."
)
# Baseline stage-2: selected schema is newly provided in the user payload.
BASELINE_BIND_SYSTEM_MESSAGE = (
    "You previously selected one tool by name. The user message now provides "
    "that tool's full schema plus the original query. Call exactly that tool "
    "once. Use the exact function name and parameter keys from the schema. "
    "Fill required parameters from the query; follow enums, types, and "
    "description conventions. Omit optional parameters the user did not "
    "clearly request; do not invent values. Return only one call in "
    "<tool_call></tool_call> tags."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class StagedValidationError(ValueError):
    """Raised when either stage violates its exact one-call contract."""


# Positional probe, default off. The router's accuracy on rand_8_anon runs
# 100% when the gold is the first tool presented and 0% when it is sixth, and
# BM25 on the same rows is flat across slots -- so the effect is the model's,
# not the benchmark's. The benchmark hands the schemas to the hypernetwork and
# the names to the meta-tool enum in the SAME order, which is why the two
# cannot be told apart without reordering exactly one of them.
#   D2L_PROBE_CTX_ORDER  = none|reverse|shuffle   order schemas are internalized
#   D2L_PROBE_ENUM_ORDER = none|reverse|shuffle   order names appear in the enum
# `shuffle` is seeded per row id, so a run is reproducible.
_PROBE_CTX_ORDER = os.getenv("D2L_PROBE_CTX_ORDER", "").strip().lower()
_PROBE_ENUM_ORDER = os.getenv("D2L_PROBE_ENUM_ORDER", "").strip().lower()
# shuffle2/shuffle3/... are additional independent permutations for order
# ensembling; _probe_reorder mixes the mode name into the seed for those while
# keeping plain "shuffle" byte-identical to earlier recorded runs.
_PROBE_MODES = {"", "none", "reverse", "shuffle"} | {f"shuffle{i}" for i in range(2, 9)}
for _mode in (_PROBE_CTX_ORDER, _PROBE_ENUM_ORDER):
    if _mode not in _PROBE_MODES:
        raise ValueError(
            f"unknown positional-probe order {_mode!r}; expected one of "
            f"{sorted(_PROBE_MODES - {''})}"
        )


def _probe_reorder(items: list, mode: str, key: str) -> list:
    if mode in ("", "none") or len(items) < 2:
        return list(items)
    if mode == "reverse":
        return list(items)[::-1]
    reordered = list(items)
    # The seed is the row id alone, so EVERY non-reverse mode produced the same
    # permutation -- only two orderings were reachable (aligned, reverse, shuffle).
    # Order-ensembling needs several. Modes named shuffle2/shuffle3/... mix the
    # mode into the seed to get distinct permutations; plain "shuffle" keeps the
    # original seed so previously recorded probe runs stay reproducible.
    seed = key if mode == "shuffle" else f"{key}:{mode}"
    random.Random(seed).shuffle(reordered)
    return reordered


def _function_tool(function: dict) -> dict:
    return {"type": "function", "function": function}


def build_route_and_plan_tool(real_tool_names: list[str]) -> dict:
    """Build Variant A's meta-tool with a dynamic enum of exact real names."""
    names = _validated_real_names(real_tool_names)
    return {
        "name": "route_and_plan",
        "description": (
            "Choose the real tool and preserve the user's executable intent "
            "without using real parameter-key labels."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "enum": names},
                "intent": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "facts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "meaning": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["meaning", "value"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["action", "facts"],
                    "additionalProperties": False,
                },
            },
            "required": ["tool_name", "intent"],
            "additionalProperties": False,
        },
    }


def build_select_tool(real_tool_names: list[str]) -> dict:
    """Build Variant B's selector with a dynamic enum of exact real names."""
    names = _validated_real_names(real_tool_names)
    return {
        "name": "select_tool",
        "description": "Choose the one real tool that should handle the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "enum": names},
            },
            "required": ["tool_name"],
            "additionalProperties": False,
        },
    }


def _validated_real_names(names: list[str]) -> list[str]:
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise StagedValidationError("tool names must be non-empty strings")
    if len(set(names)) != len(names):
        raise StagedValidationError("tool names must be unique")
    return list(names)


def parse_exactly_one_tool_call(text: str) -> dict:
    """Parse one tagged call and reject prose, malformed JSON, or extra calls."""
    if not isinstance(text, str):
        raise StagedValidationError("model output must be text")
    matches = list(_TOOL_CALL_RE.finditer(text))
    if len(matches) != 1:
        raise StagedValidationError(
            f"expected exactly one <tool_call>, found {len(matches)}"
        )
    outside = text[: matches[0].start()] + text[matches[0].end() :]
    if outside.strip():
        raise StagedValidationError("text outside the single tool call is not allowed")
    try:
        call = json.loads(matches[0].group(1))
    except json.JSONDecodeError as exc:
        raise StagedValidationError(f"tool call is not valid JSON: {exc}") from exc
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise StagedValidationError(
            "tool call must contain exactly 'name' and 'arguments'"
        )
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        raise StagedValidationError("tool call name/arguments have invalid types")
    return call


def validate_call_against_function(call: dict, function: dict) -> None:
    """Strictly validate a final call against one normalized function schema."""
    if call["name"] != function.get("name"):
        raise StagedValidationError(
            f"expected function {function.get('name')!r}, got {call['name']!r}"
        )
    _validate_json_value(
        call["arguments"],
        function.get("parameters", {"type": "object", "properties": {}}),
        "$.arguments",
    )


def _validate_json_value(value: Any, schema: dict, path: str) -> None:
    if "enum" in schema and value not in schema["enum"]:
        raise StagedValidationError(f"{path} is not in enum {schema['enum']!r}")

    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        errors = []
        for variant in variants:
            try:
                _validate_json_value(value, variant, path)
                return
            except StagedValidationError as exc:
                errors.append(str(exc))
        raise StagedValidationError(f"{path} matches no schema variant: {errors}")

    expected = schema.get("type")
    expected = {
        "dict": "object",
        "HashMap": "object",
        "tuple": "array",
        "ArrayList": "array",
        "float": "number",
        "String": "string",
        "Boolean": "boolean",
        "any": None,
    }.get(expected, expected)
    if isinstance(expected, list):
        errors = []
        for item_type in expected:
            try:
                _validate_json_value(value, {**schema, "type": item_type}, path)
                return
            except StagedValidationError as exc:
                errors.append(str(exc))
        raise StagedValidationError(f"{path} has none of the allowed types: {errors}")

    type_checks = {
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "array": lambda x: isinstance(x, list),
        "object": lambda x: isinstance(x, dict),
        "null": lambda x: x is None,
    }
    if expected in type_checks and not type_checks[expected](value):
        raise StagedValidationError(
            f"{path} must be {expected}, got {type(value).__name__}"
        )

    if expected == "object" or (expected is None and "properties" in schema):
        if not isinstance(value, dict):
            raise StagedValidationError(f"{path} must be object")
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise StagedValidationError(f"{path} is missing required keys {missing!r}")
        unexpected = [key for key in value if key not in properties]
        additional = schema.get("additionalProperties")
        reject_unexpected = additional is False or (
            additional is None and bool(properties)
        )
        if unexpected and reject_unexpected:
            raise StagedValidationError(f"{path} has unexpected keys {unexpected!r}")
        for key, child in value.items():
            if key in properties:
                _validate_json_value(child, properties[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                _validate_json_value(child, additional, f"{path}.{key}")

    if expected == "array":
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_json_value(item, item_schema, f"{path}[{index}]")


def _validate_meta_call(call: dict, meta_function: dict, real_functions: list[dict]):
    validate_call_against_function(call, meta_function)
    selected = call["arguments"]["tool_name"]
    selected_schema = _lookup_selected_schema(selected, real_functions)
    if call["name"] == "route_and_plan":
        parameter_names = {
            name.casefold()
            for name in selected_schema.get("parameters", {}).get("properties", {})
        }
        for fact in call["arguments"]["intent"]["facts"]:
            if fact["meaning"].casefold() in parameter_names:
                raise StagedValidationError(
                    "intent fact meanings must not copy exact parameter-key labels"
                )
    return selected_schema


def _lookup_selected_schema(name: str, functions: list[dict]) -> dict:
    matches = [function for function in functions if function.get("name") == name]
    if len(matches) != 1:
        raise StagedValidationError(
            f"selected tool {name!r} resolved to {len(matches)} schemas"
        )
    return matches[0]


def _binder_system_message() -> str:
    """Return binder system text; optional file/env overrides for prompt ablations."""
    path = os.environ.get("D2L_BINDER_SYSTEM_MESSAGE_FILE")
    if path:
        with open(path, encoding="utf-8") as handle:
            text = handle.read().strip()
        if text:
            return text
    override = os.environ.get("D2L_BINDER_SYSTEM_MESSAGE")
    if override and override.strip():
        return override.strip()
    return BINDER_SYSTEM_MESSAGE


def _stage_one_system_prefix(*, use_baseline_prompts: bool) -> str:
    """Return stage-1 system prefix; optional file/env overrides for ablations."""

    def _with_schema_anchor(text: str) -> str:
        text = text.strip()
        if not text:
            return text
        # _build_stage_one_messages appends the meta-tool JSON after this prefix.
        if text.rstrip().endswith("Meta-tool schema:"):
            return text.rstrip() + "\n"
        return text + "\n\nMeta-tool schema:\n"

    path = os.environ.get("D2L_ROUTER_SYSTEM_MESSAGE_FILE")
    if path:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        anchored = _with_schema_anchor(text)
        if anchored:
            return anchored
    override = os.environ.get("D2L_ROUTER_SYSTEM_MESSAGE")
    if override and override.strip():
        return _with_schema_anchor(override)
    return (
        BASELINE_SELECT_SYSTEM_PREFIX
        if use_baseline_prompts
        else META_CALL_SYSTEM_PREFIX
    )


def build_binder_messages(selected_schema: dict, intent: dict) -> list[dict]:
    """Build Variant A boundary; schema goes through native tools=, not user JSON."""
    del selected_schema  # provided separately via constraint_tools / tools=
    payload = {"intent": intent}
    return [
        {
            "role": "system",
            "content": _binder_system_message(),
        },
        {
            "role": "user",
            "content": _canonical_json(payload),
        },
    ]


def build_native_fc_binder_messages(original_messages: list[dict]) -> list[dict]:
    """Native Qwen FC bind: original user query only; schema via tools=."""
    messages = [
        {"role": message["role"], "content": message.get("content", "")}
        for message in original_messages
        if message.get("role") == "user"
    ]
    if not messages:
        raise StagedValidationError(
            "native FC binder requires at least one user message"
        )
    return messages


def binder_prompt_contains_original_query(
    binder_messages: list[dict], original_messages: list[dict]
) -> bool:
    del original_messages  # Query values may legitimately reappear as intent facts.
    forbidden = {"query", "original_query", "user_query", "prompt"}
    for message in binder_messages:
        if message.get("role") != "user":
            continue
        try:
            payload = json.loads(str(message.get("content", "")))
        except json.JSONDecodeError:
            return True
        if not isinstance(payload, dict):
            return True
        if forbidden.intersection(payload):
            return True
        if set(payload) != {"intent"}:
            return True
        if "selected_tool_schema" in payload:
            return True
    return False


def _build_stage_one_messages(
    meta_function: dict,
    original_messages: list[dict],
    *,
    system_prefix: str = META_CALL_SYSTEM_PREFIX,
) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                system_prefix
                + json.dumps(
                    _function_tool(meta_function),
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        },
        *deepcopy(original_messages),
    ]


def _build_variant_b_messages(
    selected_schema: dict,
    original_messages: list[dict],
    *,
    system_message: str = B_JOINT_BIND_SYSTEM_MESSAGE,
) -> list[dict]:
    query = "\n".join(
        str(message.get("content", ""))
        for message in original_messages
        if message.get("role") == "user" and message.get("content")
    )
    payload = {
        "selected_tool_schema": _function_tool(selected_schema),
        "original_query": query,
    }
    return [
        {
            "role": "system",
            "content": system_message,
        },
        {"role": "user", "content": _canonical_json(payload)},
    ]


def build_plain_schema_bind_messages(
    stage1_messages: list[dict],
    raw_selected_schema: dict,
) -> list[dict]:
    """Build a router-free bind transcript that can resume from the P cache."""
    payload = (
        "Routing is complete out of band. Do not call select_tool again. "
        "Call exactly the following original BFCL function for the original "
        "user request. Preserve its exact function name and parameter keys, "
        "and follow its raw types, enums, optional fields, and descriptions. "
        "Return only one JSON call inside <tool_call></tool_call> tags.\n\n"
        "Original BFCL function schema:\n"
        + _canonical_json(raw_selected_schema)
    )
    return [
        *deepcopy(stage1_messages),
        {"role": "user", "content": payload},
    ]


def _canonical_tool_call(call: dict) -> str:
    return (
        "<tool_call>\n"
        + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def _is_supported_staged_entry_id(entry_id: str) -> bool:
    return entry_id.startswith(("multiple_", "live_simple_"))


class DocToLoraStagedHandler(BaseHandler):
    """Configurable implementation shared by the two registered variants."""

    variant: str = ""

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        if self.variant not in {"a", "b"}:
            raise ValueError("staged handler variant must be 'a' or 'b'")

        self.checkpoint_path = kwargs.get(
            "checkpoint_path", os.environ.get("D2L_CHECKPOINT_PATH")
        )
        self.binder_checkpoint_path = kwargs.get(
            "binder_checkpoint_path", os.environ.get("D2L_BINDER_CHECKPOINT_PATH")
        )
        self.skip_internalize = str(
            kwargs.get(
                "skip_internalize",
                os.environ.get("D2L_SKIP_INTERNALIZE", "0"),
            )
        ) not in ("0", "", "false", "False")
        # When True, use baseline prompts that assume no Doc-to-LoRA knowledge.
        self.use_baseline_prompts = str(
            kwargs.get(
                "use_baseline_prompts",
                os.environ.get(
                    "D2L_BASELINE_PROMPTS",
                    "1" if self.skip_internalize else "0",
                ),
            )
        ) not in ("0", "", "false", "False")
        # When True, reset the internalized all-tools LoRA before the stage-2
        # bind pass so binding runs on the frozen base model. Stage-1 still
        # uses D2L. This isolates whether the all-tools adapter helps or hurts
        # bind; it recovers baseline-level bind while keeping D2L routing.
        self.bind_on_base = str(
            kwargs.get(
                "bind_on_base",
                os.environ.get("D2L_BIND_ON_BASE", "0"),
            )
        ) not in ("0", "", "false", "False")
        # Variant B select_tool routing + separate binder worker that sees the
        # original query and the selected schema via native Qwen tools= (same
        # contract as Qwen3-*-FC single-tool binding).
        self.native_fc_binder = str(
            kwargs.get(
                "native_fc_binder",
                os.environ.get("D2L_NATIVE_FC_BINDER", "0"),
            )
        ) not in ("0", "", "false", "False")
        self.stateful_continuation = str(
            kwargs.get(
                "stateful_continuation",
                os.environ.get("D2L_STATEFUL_CONTINUATION", "0"),
            )
        ) not in ("0", "", "false", "False")
        self.oracle_route = str(
            kwargs.get(
                "oracle_route",
                os.environ.get("D2L_ORACLE_ROUTE", "0"),
            )
        ) not in ("0", "", "false", "False")
        self.plain_schema_bind = str(
            kwargs.get(
                "plain_schema_bind",
                os.environ.get("D2L_PLAIN_SCHEMA_BIND", "0"),
            )
        ) not in ("0", "", "false", "False")
        self.router_score_candidates = str(
            kwargs.get(
                "router_score_candidates",
                os.environ.get("D2L_ROUTER_SCORE_CANDIDATES", "0"),
            )
        ) not in ("0", "", "false", "False")
        # ICL ceiling. The native binder never reads stage 1's output (its
        # messages are the user query alone, schemas arrive via tools=), so
        # handing it EVERY schema instead of the routed one turns this
        # pipeline into plain in-context function calling on the frozen base
        # model, with the routing stage made free by oracle_route. That keeps
        # the ICL number on exactly the same model, template and decoding
        # machinery as the Variant B runs it is meant to bound.
        self.icl_all_tools = str(
            kwargs.get(
                "icl_all_tools",
                os.environ.get("D2L_ICL_ALL_TOOLS", "0"),
            )
        ) not in ("0", "", "false", "False")
        if self.router_score_candidates and self.stateful_continuation:
            raise ValueError(
                "candidate-scored routing currently supports exact Phase A "
                "recompute, not stateful continuation"
            )
        self.router_schema_ablation = str(
            kwargs.get(
                "router_schema_ablation",
                os.environ.get("D2L_ROUTER_SCHEMA_ABLATION", "full"),
            )
        ).strip()
        if self.router_schema_ablation not in {
            "full",
            "names_only",
            "descriptions_only",
            "parameters_only",
            "shuffled",
        }:
            raise ValueError(
                "router_schema_ablation must be full, names_only, "
                "descriptions_only, parameters_only, or shuffled"
            )
        self.routing_only = str(
            kwargs.get(
                "routing_only",
                os.environ.get("D2L_ROUTING_ONLY", "0"),
            )
        ) not in ("0", "", "false", "False")
        self.chunk_size = int(
            kwargs.get("chunk_size", os.environ.get("D2L_CHUNK_SIZE", "1024"))
        )
        self.ctx_chunk_mode = str(
            kwargs.get(
                "ctx_chunk_mode",
                os.environ.get("D2L_CTX_CHUNK_MODE", "none"),
            )
        )
        if self.ctx_chunk_mode not in ("none", "per_tool"):
            raise ValueError("ctx_chunk_mode must be none or per_tool")
        self.tools_per_chunk = int(
            kwargs.get(
                "tools_per_chunk",
                os.environ.get("D2L_TOOLS_PER_CHUNK", "1"),
            )
        )
        self.max_new_tokens = int(
            kwargs.get(
                "max_new_tokens", os.environ.get("D2L_MAX_NEW_TOKENS", "512")
            )
        )
        self.binder_max_new_tokens = int(
            kwargs.get(
                "binder_max_new_tokens",
                os.environ.get("D2L_BINDER_MAX_NEW_TOKENS", "256"),
            )
        )
        d2l_source = kwargs.get(
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
        available_gpus = DocToLoraHandler._detect_gpu_ids()
        main_gpu = str(
            kwargs.get(
                "main_gpu",
                os.environ.get("D2L_STAGED_MAIN_GPU", available_gpus[0]),
            )
        )
        self._main_worker = _D2LWorkerProxy(
            d2l_python, d2l_source, gpu_device=main_gpu
        )

        self._binder_worker = None
        if self.variant == "a" or self.native_fc_binder:
            binder_python = kwargs.get(
                "binder_python",
                os.environ.get("D2L_BINDER_PYTHON", d2l_python),
            )
            binder_default_gpu = (
                available_gpus[1] if len(available_gpus) > 1 else available_gpus[0]
            )
            binder_gpu = str(
                kwargs.get(
                    "binder_gpu",
                    os.environ.get("D2L_BINDER_GPU", binder_default_gpu),
                )
            )
            self._binder_worker = _D2LWorkerProxy(
                binder_python,
                d2l_source,
                gpu_device=binder_gpu,
                worker_script=_BINDER_WORKER_SCRIPT,
            )

        self._inference_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.last_trace: dict | None = None
        raw_log_path = kwargs.get("raw_log_path", os.environ.get("D2L_RAW_LOG", ""))
        if raw_log_path:
            os.makedirs(os.path.dirname(raw_log_path) or ".", exist_ok=True)
            self._raw_log_fd = os.open(
                raw_log_path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND,
                0o644,
            )
        else:
            self._raw_log_fd = None

    def _ensure_models_loaded(self) -> None:
        allow_shared = os.environ.get("D2L_ALLOW_SHARED_GPU", "").strip() in {
            "1",
            "true",
            "True",
            "yes",
            "YES",
        }
        uses_binder = self.variant == "a" or self.native_fc_binder
        if (
            uses_binder
            and not allow_shared
            and self._main_worker.gpu_device == self._binder_worker.gpu_device
        ):
            raise ValueError(
                "Staged select/intent + binder requires distinct main and binder "
                "GPUs; set D2L_STAGED_MAIN_GPU and D2L_BINDER_GPU "
                "(or D2L_ALLOW_SHARED_GPU=1 to colocate)"
            )
        if (
            not self._main_worker.model_loaded
            or not self._main_worker.is_alive
        ):
            if not self.checkpoint_path:
                raise ValueError("D2L_CHECKPOINT_PATH is required")
            self._main_worker.send(
                "load_model", {"checkpoint_path": self.checkpoint_path}
            )
            self._main_worker.model_loaded = True
        if uses_binder and (
            not self._binder_worker.model_loaded
            or not self._binder_worker.is_alive
        ):
            if not self.binder_checkpoint_path:
                raise ValueError(
                    "D2L_BINDER_CHECKPOINT_PATH is required for staged binders"
                )
            self._binder_worker.send(
                "load_model", {"checkpoint_path": self.binder_checkpoint_path}
            )
            self._binder_worker.model_loaded = True

    def prepare(self, test_cases: list[dict]) -> None:
        del test_cases
        self._ensure_models_loaded()

    def _internalize(self, normalized_functions: list[dict]) -> None:
        if self.skip_internalize:
            # Clear any residual generated LoRA so generation uses the base model.
            self._main_worker.send("reset", {})
            return
        internalized_functions = deepcopy(normalized_functions)
        schema_ablation = getattr(self, "router_schema_ablation", "full")
        if schema_ablation == "names_only":
            for function in internalized_functions:
                function["description"] = ""
                function["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
        elif schema_ablation == "descriptions_only":
            for function in internalized_functions:
                function["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
        elif schema_ablation == "parameters_only":
            for function in internalized_functions:
                function["description"] = ""
        elif (
            schema_ablation == "shuffled"
            and len(internalized_functions) > 1
        ):
            payloads = [
                {
                    "description": function.get("description", ""),
                    "parameters": deepcopy(function.get("parameters", {})),
                }
                for function in internalized_functions
            ]
            payloads = payloads[1:] + payloads[:1]
            for function, payload in zip(internalized_functions, payloads):
                function.update(payload)
        tool_defs = json.dumps(
            [_function_tool(function) for function in internalized_functions],
            indent=2,
            ensure_ascii=False,
        )
        self._main_worker.send(
            "internalize",
            {
                "tool_defs": tool_defs,
                "chunk_size": self.chunk_size,
                "ctx_chunk_mode": self.ctx_chunk_mode,
                "tools_per_chunk": self.tools_per_chunk,
            },
        )

    def _generate_main(
        self,
        messages: list[dict],
        constraint_function: dict,
        *,
        force_first_required: bool = False,
    ) -> tuple[dict, float]:
        started = time.perf_counter()
        result = self._main_worker.send(
            "generate",
            {
                "messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "restrict_toolgen": True,
                "constraint_tools": [_function_tool(constraint_function)],
                "assistant_prefix": _assistant_prefix(
                    constraint_function,
                    force_first_required=force_first_required,
                ),
                "enable_thinking": False,
                "stop_after_first_tool_call": True,
                "strict_json_schema": not force_first_required,
            },
        )
        result["raw_text"] = result["text"]
        result["text"] = _append_missing_end_tag(
            _repair_schema_key_syntax(result["text"], constraint_function)
        )
        return result, time.perf_counter() - started

    def _generate_main_unconstrained(
        self,
        messages: list[dict],
    ) -> tuple[dict, float]:
        """Generate one complete call without prefixes or logits constraints."""
        started = time.perf_counter()
        result = self._main_worker.send(
            "generate",
            {
                "messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "restrict_toolgen": False,
                "constraint_tools": None,
                "assistant_prefix": "",
                "enable_thinking": False,
                "stop_after_first_tool_call": True,
                "strict_json_schema": False,
            },
        )
        result["raw_text"] = result["text"]
        return result, time.perf_counter() - started

    def _score_main_candidates(
        self,
        messages: list[dict],
        candidate_names: list[str],
    ) -> tuple[dict, float]:
        """Score only candidate-name spans; infrastructure owns call syntax."""
        started = time.perf_counter()
        result = self._main_worker.send(
            "score_router_candidates",
            {
                "stage1_messages": messages,
                "candidate_names": candidate_names,
                "enable_thinking": False,
            },
        )
        result["raw_text"] = result["text"]
        return result, time.perf_counter() - started

    def _start_main_session(
        self,
        messages: list[dict],
    ) -> tuple[dict, float]:
        started = time.perf_counter()
        result = self._main_worker.send(
            "start_late_session",
            {
                "stage1_messages": messages,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "enable_thinking": False,
            },
        )
        result["raw_text"] = result["text"]
        return result, time.perf_counter() - started

    def _generate_late_schema(
        self,
        stage1_messages: list[dict],
        router_token_ids: list[int],
        raw_selected_schema: dict,
    ) -> tuple[dict, float]:
        """Recompute the exact P || R || S transcript with the LoRA attached."""
        started = time.perf_counter()
        result = self._main_worker.send(
            "generate_late_schema",
            {
                "stage1_messages": stage1_messages,
                "router_token_ids": router_token_ids,
                "raw_function": raw_selected_schema,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "enable_thinking": False,
            },
        )
        result["raw_text"] = result["text"]
        return result, time.perf_counter() - started

    def _append_late_schema_session(
        self,
        session_id: str,
        raw_selected_schema: dict,
    ) -> tuple[dict, float]:
        started = time.perf_counter()
        result = self._main_worker.send(
            "append_late_schema_session",
            {
                "session_id": session_id,
                "raw_function": raw_selected_schema,
                "max_new_tokens": self.max_new_tokens,
            },
        )
        result["raw_text"] = result["text"]
        return result, time.perf_counter() - started

    def _generate_binder(
        self,
        messages: list[dict],
        selected_schema: dict,
        *,
        force_first_required: bool = False,
    ) -> tuple[dict, float]:
        started = time.perf_counter()
        result = self._binder_worker.send(
            "generate",
            {
                "messages": messages,
                "constraint_tools": [_function_tool(selected_schema)],
                "max_new_tokens": self.binder_max_new_tokens,
                "temperature": self.temperature,
                "assistant_prefix": _assistant_prefix(
                    selected_schema,
                    force_first_required=force_first_required,
                ),
                "strict_json_schema": not force_first_required,
            },
        )
        result["raw_text"] = result["text"]
        result["text"] = _append_missing_end_tag(
            _repair_schema_key_syntax(result["text"], selected_schema)
        )
        return result, time.perf_counter() - started

    def _generate_binder_icl(
        self, messages: list[dict], all_tools: list[dict]
    ) -> tuple[dict, float]:
        """ICL bind: every candidate schema in tools=, model picks and fills.

        Uses the strict union grammar, which now accepts N tools, so ICL gets
        exactly the guarantee the Doc-to-LoRA binder gets: output is a single
        well-formed call, and the only freedom is which tool. The earlier
        lexical fallback masked only name/param-key/enum positions, which let
        the model close the name string and then loop re-emitting it rather
        than moving on to "arguments"; the unparseable result was then scored
        as a routing error, at a rate that grew with catalogue size (2% on the
        named categories, 52% on rand_32_anon). The assistant prefix stays the
        bare ``<tool_call>`` opener -- naming a tool would make the choice for
        the model.
        """
        started = time.perf_counter()
        result = self._binder_worker.send(
            "generate",
            {
                "messages": messages,
                "constraint_tools": [
                    _function_tool(function) for function in all_tools
                ],
                "max_new_tokens": self.binder_max_new_tokens,
                "temperature": self.temperature,
                "assistant_prefix": "<tool_call>\n",
                "strict_json_schema": True,
            },
        )
        result["raw_text"] = result["text"]
        # No _repair_schema_key_syntax: which schema to repair against is the
        # very thing the model is choosing here.
        result["text"] = _append_missing_end_tag(result["text"])
        return result, time.perf_counter() - started

    def _execute_stages(
        self,
        test_entry: dict,
        original_messages: list[dict],
    ) -> tuple[str, dict]:
        normalized = [
            DocToLoraHandler._normalize_bfcl_function(function)
            for function in test_entry["function"]
        ]
        # Positional probe. The benchmark presents the schemas to the
        # hypernetwork and the names to the prompt in the SAME order, so a slot
        # effect cannot be attributed to either one. Reordering exactly one of
        # them separates them. `normalized` keeps the benchmark order, so the
        # trace's gold-slot bookkeeping is unaffected.
        probe_key = str(test_entry.get("id"))
        ctx_functions = _probe_reorder(normalized, _PROBE_CTX_ORDER, probe_key)
        names = _probe_reorder(
            [function["name"] for function in normalized],
            _PROBE_ENUM_ORDER,
            probe_key,
        )
        meta_function = (
            build_route_and_plan_tool(names)
            if self.variant == "a"
            else build_select_tool(names)
        )
        router_score_candidates = getattr(
            self, "router_score_candidates", False
        )
        router_schema_ablation = getattr(
            self, "router_schema_ablation", "full"
        )
        routing_only = getattr(self, "routing_only", False)
        trace = {
            "id": test_entry.get("id"),
            "variant": self.variant,
            "skip_internalize": self.skip_internalize,
            "use_baseline_prompts": self.use_baseline_prompts,
            "bind_on_base": self.bind_on_base,
            "native_fc_binder": self.native_fc_binder,
            "stateful_continuation": self.stateful_continuation,
            "oracle_route": self.oracle_route,
            "plain_schema_bind": self.plain_schema_bind,
            "router_score_candidates": router_score_candidates,
            "router_schema_ablation": router_schema_ablation,
            "routing_only": routing_only,
            "functions": deepcopy(test_entry["function"]),
            "normalized_functions": normalized,
            "probe_ctx_order": _PROBE_CTX_ORDER or "none",
            "probe_enum_order": _PROBE_ENUM_ORDER or "none",
            "probe_ctx_names": [f["name"] for f in ctx_functions],
            "probe_enum_names": list(names),
            "stage1": {},
            "stage2": {},
            "validation_errors": [],
            "protocol": (
                "router-free-plain-schema-bind"
                if self.plain_schema_bind
                else "late-schema-exact-transcript"
            ),
        }
        total_started = time.perf_counter()
        try:
            self._internalize(ctx_functions)

            if self.icl_all_tools:
                # Plain in-context function calling: there is no routing stage
                # at all. Bypassing it here rather than making it free via
                # oracle_route keeps ground truth out of the ICL path entirely
                # (the oracle map does not cover every category, and a lookup
                # miss is fatal) and removes a stage the baseline should not be
                # paying for. The binder sees the user query plus every schema.
                icl_messages = build_native_fc_binder_messages(original_messages)
                icl_result, icl_latency = self._generate_binder_icl(
                    icl_messages, normalized
                )
                icl_call = parse_exactly_one_tool_call(icl_result["text"])
                trace["stage1"] = {"skipped": True, "reason": "icl_all_tools"}
                trace["stage2"] = {
                    "worker": "icl-all-tools-native-fc",
                    "messages": icl_messages,
                    "n_tools_in_context": len(normalized),
                    "raw_output": icl_result["text"],
                    "parsed_call": icl_call,
                    "input_tokens": icl_result["input_tokens"],
                    "output_tokens": icl_result["output_tokens"],
                    "latency_seconds": icl_latency,
                    "constraint_mode": icl_result.get("constraint_mode"),
                }
                trace["final_call"] = icl_call
                return icl_result["text"], {
                    "input_tokens": icl_result["input_tokens"],
                    "output_tokens": icl_result["output_tokens"],
                    "latency": time.perf_counter() - total_started,
                }

            stage1_prefix = _stage_one_system_prefix(
                use_baseline_prompts=self.use_baseline_prompts
            )
            stage1_messages = _build_stage_one_messages(
                meta_function,
                original_messages,
                system_prefix=stage1_prefix,
            )
            trace["stage1"] = {
                "messages": stage1_messages,
                "meta_schema": meta_function,
                "attempts": [],
            }
            stage1_input_tokens = 0
            stage1_output_tokens = 0
            stage1_attempt_modes = (False,) if self.variant == "b" else (False, True)
            for attempt_index, force_first_required in enumerate(stage1_attempt_modes):
                stage1_result = None
                started = time.perf_counter()
                try:
                    if self.variant == "b":
                        if self.oracle_route:
                            from bfcl_eval.model_handler.local_inference.qwen_fc import (
                                _oracle_tool_name_from_ground_truth,
                            )

                            oracle_name = _oracle_tool_name_from_ground_truth(
                                test_entry["id"]
                            )
                            oracle_call = {
                                "name": "select_tool",
                                "arguments": {"tool_name": oracle_name},
                            }
                            oracle_text = _canonical_tool_call(oracle_call)
                            encoded = self._main_worker.send(
                                "tokenize_text", {"text": oracle_text}
                            )
                            stage1_result = {
                                "text": oracle_text,
                                "decoded_text": oracle_text,
                                "raw_text": oracle_text,
                                "input_tokens": 0,
                                "output_tokens": len(encoded["token_ids"]),
                                "token_ids": encoded["token_ids"],
                                "constraint_mode": "none",
                                "enable_thinking": False,
                            }
                            stage1_latency = time.perf_counter() - started
                        elif router_score_candidates:
                            stage1_result, stage1_latency = (
                                self._score_main_candidates(
                                    stage1_messages,
                                    names,
                                )
                            )
                        elif self.stateful_continuation:
                            stage1_result, stage1_latency = (
                                self._start_main_session(stage1_messages)
                            )
                        else:
                            stage1_result, stage1_latency = (
                                self._generate_main_unconstrained(stage1_messages)
                            )
                    else:
                        stage1_result, stage1_latency = self._generate_main(
                            stage1_messages,
                            meta_function,
                            force_first_required=force_first_required,
                        )
                    stage1_input_tokens += stage1_result["input_tokens"]
                    stage1_output_tokens += stage1_result["output_tokens"]
                    attempt = {
                        "force_first_required": force_first_required,
                        "raw_worker_output": stage1_result.get("raw_text"),
                        "raw_output": stage1_result["text"],
                        "input_tokens": stage1_result["input_tokens"],
                        "output_tokens": stage1_result["output_tokens"],
                        "latency_seconds": stage1_latency,
                        "constraint_mode": stage1_result.get("constraint_mode"),
                    }
                    trace["stage1"]["attempts"].append(attempt)
                    meta_call = parse_exactly_one_tool_call(stage1_result["text"])
                    selected_schema = _validate_meta_call(
                        meta_call, meta_function, normalized
                    )
                    break
                except (RuntimeError, StagedValidationError) as exc:
                    if stage1_result is None:
                        attempt = {
                            "force_first_required": force_first_required,
                            "latency_seconds": time.perf_counter() - started,
                            "constraint_mode": (
                                "xgrammar"
                                if not force_first_required
                                else "lexical"
                            ),
                        }
                        trace["stage1"]["attempts"].append(attempt)
                    attempt["validation_error"] = str(exc)
                    if (
                        attempt_index == len(stage1_attempt_modes) - 1
                        or not self._main_worker.is_alive
                    ):
                        raise
            trace["stage1"].update(
                {
                    "raw_worker_output": stage1_result.get("raw_text"),
                    "raw_output": stage1_result["text"],
                    "input_tokens": stage1_input_tokens,
                    "output_tokens": stage1_output_tokens,
                    "latency_seconds": sum(
                        attempt["latency_seconds"]
                        for attempt in trace["stage1"]["attempts"]
                    ),
                    "constraint_mode": stage1_result.get("constraint_mode"),
                    "candidate_scores": stage1_result.get("candidate_scores"),
                    "choice_margin": stage1_result.get("choice_margin"),
                }
            )
            trace["stage1"]["parsed_call"] = meta_call
            raw_selected_schema = _lookup_selected_schema(
                meta_call["arguments"]["tool_name"],
                test_entry["function"],
            )
            trace["selected_schema"] = raw_selected_schema
            trace["normalized_selected_schema"] = selected_schema

            if routing_only:
                selected_call = {
                    "name": raw_selected_schema["name"],
                    "arguments": {},
                }
                trace["stage2"] = {
                    "skipped": True,
                    "reason": "routing_only",
                }
                trace["final_call"] = selected_call
                return _canonical_tool_call(selected_call), {
                    "input_tokens": stage1_input_tokens,
                    "output_tokens": stage1_output_tokens,
                    "latency": time.perf_counter() - total_started,
                }

            if self.native_fc_binder:
                if self.variant != "b":
                    raise StagedValidationError(
                        "native_fc_binder requires Variant B select_tool routing"
                    )
                stage2_messages = build_native_fc_binder_messages(original_messages)
                stage2_worker = "native-fc-separate-binder"

                def run_stage2(force_first_required):
                    return self._generate_binder(
                        stage2_messages,
                        selected_schema,
                        force_first_required=force_first_required,
                    )

            elif self.variant == "a":
                stage2_messages = build_binder_messages(
                    selected_schema, meta_call["arguments"]["intent"]
                )
                if binder_prompt_contains_original_query(
                    stage2_messages, original_messages
                ):
                    raise StagedValidationError(
                        "Variant A binder prompt leaked the original query"
                    )
                stage2_worker = "qwen3-0.6b-non-thinking-binder"

                def run_stage2(force_first_required):
                    return self._generate_binder(
                        stage2_messages,
                        selected_schema,
                        force_first_required=force_first_required,
                    )

            else:
                if self.plain_schema_bind:
                    if self.stateful_continuation:
                        raise StagedValidationError(
                            "plain_schema_bind intentionally uses full "
                            "recomputation; stateful cache branching is not "
                            "implemented"
                        )
                    if self.bind_on_base:
                        raise StagedValidationError(
                            "plain_schema_bind requires the D2L LoRA to remain "
                            "attached"
                        )
                    if self.skip_internalize:
                        raise StagedValidationError(
                            "plain_schema_bind requires D2L internalization"
                        )
                    stage2_messages = build_plain_schema_bind_messages(
                        stage1_messages,
                        raw_selected_schema,
                    )
                    stage2_worker = (
                        "same-d2l-4b-router-free-plain-schema-adapter-active"
                    )

                    def run_stage2(force_first_required):
                        del force_first_required
                        return self._generate_main_unconstrained(stage2_messages)

                    unconstrained_late_schema = True
                else:
                    if self.bind_on_base:
                        # Drop the all-tools LoRA so the bind pass runs on the
                        # frozen base, matching the no-D2L baseline's bind condition
                        # while keeping D2L for stage-1 selection.
                        self._main_worker.send("reset", {})
                    if self.bind_on_base:
                        raise StagedValidationError(
                            "bind_on_base is incompatible with the LoRA-on "
                            "late-schema primary protocol"
                        )
                    if not stage1_result.get("token_ids"):
                        raise StagedValidationError(
                            "late-schema continuation requires exact router token_ids"
                        )
                    stage2_messages = [
                        *stage1_messages,
                        {
                            "role": "assistant",
                            "content": stage1_result.get(
                                "decoded_text", stage1_result["text"]
                            ),
                        },
                        {
                            "role": "tool",
                            "content": _canonical_json(
                                {
                                    "selected_tool_schema": _function_tool(
                                        raw_selected_schema
                                    )
                                }
                            ),
                        },
                    ]
                    stage2_worker = (
                        "base-qwen3-4b-no-d2l"
                        if self.skip_internalize
                        else "same-d2l-4b-late-schema-adapter-active"
                    )

                    def run_stage2(force_first_required):
                        del force_first_required
                        if self.stateful_continuation:
                            return self._append_late_schema_session(
                                stage1_result["session_id"],
                                raw_selected_schema,
                            )
                        else:
                            return self._generate_late_schema(
                                stage1_messages,
                                stage1_result["token_ids"],
                                raw_selected_schema,
                            )

                    unconstrained_late_schema = True

            if "unconstrained_late_schema" not in locals():
                unconstrained_late_schema = False

            attempts = []
            stage2_input_tokens = 0
            stage2_output_tokens = 0
            trace["stage2"] = {
                "worker": stage2_worker,
                "messages": stage2_messages,
                "attempts": attempts,
            }
            # First use strict JSON-schema generation. If a lightly trained
            # checkpoint loops until its token budget, retry with the first
            # required key (when present) prefilled and lexical constraints.
            attempt_modes = (False,) if unconstrained_late_schema else (False, True)
            for attempt_index, force_first_required in enumerate(attempt_modes):
                stage2_result = None
                started = time.perf_counter()
                try:
                    stage2_result, stage2_latency = run_stage2(
                        force_first_required
                    )
                    stage2_input_tokens += stage2_result["input_tokens"]
                    stage2_output_tokens += stage2_result["output_tokens"]
                    attempt = {
                        "force_first_required": force_first_required,
                        "raw_worker_output": stage2_result.get("raw_text"),
                        "raw_output": stage2_result["text"],
                        "input_tokens": stage2_result["input_tokens"],
                        "output_tokens": stage2_result["output_tokens"],
                        "latency_seconds": stage2_latency,
                        "constraint_mode": stage2_result.get("constraint_mode"),
                    }
                    attempts.append(attempt)
                    final_call = parse_exactly_one_tool_call(stage2_result["text"])
                    validation_schema = (
                        raw_selected_schema
                        if unconstrained_late_schema
                        else selected_schema
                    )
                    try:
                        validate_call_against_function(
                            final_call, validation_schema
                        )
                    except StagedValidationError as exc:
                        if not unconstrained_late_schema:
                            raise
                        attempt["schema_validation_error"] = str(exc)
                    break
                except (RuntimeError, StagedValidationError) as exc:
                    if stage2_result is None:
                        attempt = {
                            "force_first_required": force_first_required,
                            "latency_seconds": time.perf_counter() - started,
                            "constraint_mode": (
                                "xgrammar"
                                if not force_first_required
                                else "lexical"
                            ),
                        }
                        attempts.append(attempt)
                    attempt["validation_error"] = str(exc)
                    worker = (
                        self._binder_worker
                        if (self.variant == "a" or self.native_fc_binder)
                        else self._main_worker
                    )
                    if (
                        attempt_index == len(attempt_modes) - 1
                        or not worker.is_alive
                    ):
                        raise

            trace["stage2"].update(
                {
                    "raw_output": stage2_result["text"],
                    "input_tokens": stage2_input_tokens,
                    "output_tokens": stage2_output_tokens,
                    "latency_seconds": sum(
                        attempt["latency_seconds"] for attempt in attempts
                    ),
                    "enable_thinking": stage2_result.get("enable_thinking"),
                    "constraint_mode": stage2_result.get("constraint_mode"),
                    "transcript": stage2_result.get("transcript"),
                    "full_token_ids": stage2_result.get("full_token_ids"),
                }
            )
            if unconstrained_late_schema:
                trace["stage2"]["messages"] = stage2_result.get(
                    "messages", stage2_messages
                )
            trace["stage2"]["parsed_call"] = final_call
            trace["final_call"] = final_call
            final_text = (
                stage2_result["text"]
                if unconstrained_late_schema
                else _canonical_tool_call(final_call)
            )
            return final_text, {
                "input_tokens": stage1_input_tokens + stage2_input_tokens,
                "output_tokens": stage1_output_tokens + stage2_output_tokens,
                "latency": time.perf_counter() - total_started,
            }
        except Exception as exc:
            trace["validation_errors"].append(
                {"type": type(exc).__name__, "message": str(exc)}
            )
            raise
        finally:
            # stage1_result is initialised to None (line 1094), so the dict
            # default never fires -- locals().get(...) returns None and .get()
            # on it raises inside `finally`, REPLACING whatever the real stage-1
            # exception was and skipping _write_trace(). That is why high-N
            # failures were unexplainable: the diagnosis was being overwritten
            # by its own cleanup path.
            session_id = (locals().get("stage1_result") or {}).get("session_id")
            if session_id and not locals().get("stage2_result"):
                try:
                    self._main_worker.send(
                        "close_session", {"session_id": session_id}
                    )
                except Exception:
                    pass
            trace["total_latency_seconds"] = time.perf_counter() - total_started
            self.last_trace = trace
            self._write_trace(trace)

    def _write_trace(self, trace: dict) -> None:
        raw_log_fd = getattr(self, "_raw_log_fd", None)
        if raw_log_fd is None:
            return
        record = (json.dumps(trace, ensure_ascii=False) + "\n").encode("utf-8")
        with self._log_lock:
            os.write(raw_log_fd, record)

    def inference(
        self,
        test_entry: dict,
        include_input_log: bool,
        exclude_state_log: bool,
    ):
        if not _is_supported_staged_entry_id(str(test_entry.get("id", ""))):
            raise ValueError(
                "staged Doc-to-LoRA handlers support only BFCL multiple and "
                "live_simple"
            )
        original_messages = deepcopy(test_entry["question"][0])
        with self._inference_lock:
            self._ensure_models_loaded()
            text, usage = self._execute_stages(test_entry, original_messages)
        metadata = {
            "input_token_count": usage["input_tokens"],
            "output_token_count": usage["output_tokens"],
            "latency": usage["latency"],
        }
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": {
                        "stage1": self.last_trace.get("stage1", {}).get("messages"),
                        "stage2": self.last_trace.get("stage2", {}).get("messages"),
                    },
                }
            ]
        return text, metadata

    def shutdown(self) -> None:
        self._main_worker._stop()
        if self._binder_worker is not None:
            self._binder_worker._stop()
        raw_log_fd = getattr(self, "_raw_log_fd", None)
        if raw_log_fd is not None:
            os.close(raw_log_fd)
            self._raw_log_fd = None

    def decode_ast(self, result, language, has_tool_call_tag):
        call = parse_exactly_one_tool_call(result)
        return [{call["name"]: call["arguments"]}]

    def decode_execute(self, result, has_tool_call_tag):
        call = parse_exactly_one_tool_call(result)
        arguments = ",".join(
            f"{key}={value!r}" for key, value in call["arguments"].items()
        )
        return [f"{call['name']}({arguments})"]


class DocToLoraMetaIntentBinderHandler(DocToLoraStagedHandler):
    """Variant A: D2L route+intent followed by a separate 0.6B binder."""

    variant = "a"


class DocToLoraMetaSelectBindHandler(DocToLoraStagedHandler):
    """Variant B: unconstrained selection then LoRA-on late-schema binding."""

    variant = "b"


class DocToLoraMetaSelectBindBaseHandler(DocToLoraMetaSelectBindHandler):
    """Variant B protocol on the frozen base model (no Doc-to-LoRA LoRA).

    Same select_tool → selected-schema+query bind flow as Variant B, but
    skips hypernetwork internalization so generation uses plain
    Qwen3-4B-Instruct. Useful as an ablation: if this scores near D2L B,
    tool-name enums + mid-context schemas carry most of the signal.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "skip_internalize",
            os.environ.get("D2L_SKIP_INTERNALIZE", "1"),
        )
        kwargs.setdefault(
            "use_baseline_prompts",
            os.environ.get("D2L_BASELINE_PROMPTS", "1"),
        )
        super().__init__(*args, **kwargs)


class DocToLoraMetaSelectLateOracleBaseHandler(
    DocToLoraMetaSelectBindBaseHandler
):
    """Oracle router call + exact raw late-schema bind on the frozen base."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "oracle_route",
            os.environ.get("D2L_ORACLE_ROUTE", "1"),
        )
        kwargs["use_baseline_prompts"] = False
        super().__init__(*args, **kwargs)


class DocToLoraMetaSelectBindOnBaseHandler(DocToLoraMetaSelectBindHandler):
    """D2L selection + frozen-base bind.

    Stage-1 internalizes all tools and calls ``select_tool`` with the D2L
    adapter active (better routing than the raw base). Before stage-2 the
    all-tools LoRA is reset so the bind pass runs on the frozen base with
    the selected schema in the user payload. This keeps D2L's routing edge
    while avoiding the bind-time overfill the all-tools adapter introduces.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "bind_on_base",
            os.environ.get("D2L_BIND_ON_BASE", "1"),
        )
        kwargs.setdefault(
            "use_baseline_prompts",
            os.environ.get("D2L_BASELINE_PROMPTS", "1"),
        )
        super().__init__(*args, **kwargs)


class DocToLoraMetaSelectNativeBinderHandler(DocToLoraMetaSelectBindHandler):
    """D2L ``select_tool`` + separate native-FC binder (query + one schema).

    Stage-1 matches Variant B (name-only meta tool). Stage-2 matches the
    oracle single-tool Qwen FC setup: original user query in chat messages and
    the selected schema via native ``tools=`` / XGrammar, on a separate
    non-thinking binder checkpoint (0.6B / 1.7B / 4B, etc.).
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault(
            "native_fc_binder",
            os.environ.get("D2L_NATIVE_FC_BINDER", "1"),
        )
        super().__init__(*args, **kwargs)


class DocToLoraICLNativeHandler(DocToLoraMetaSelectNativeBinderHandler):
    """ICL ceiling: frozen Qwen3-4B, every schema inline, no D2L at all.

    Same binder model, chat template and lexical constraint as the native-binder
    runs, so the only difference from them is what the model can see: all N
    schemas inline rather than the one D2L routed to. There is no routing stage
    -- it is bypassed outright, not made free with an oracle -- so no ground
    truth touches this path. Internalization is skipped: no adapter takes part.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("icl_all_tools", os.environ.get("D2L_ICL_ALL_TOOLS", "1"))
        kwargs.setdefault(
            "skip_internalize", os.environ.get("D2L_SKIP_INTERNALIZE", "1")
        )
        super().__init__(*args, **kwargs)
