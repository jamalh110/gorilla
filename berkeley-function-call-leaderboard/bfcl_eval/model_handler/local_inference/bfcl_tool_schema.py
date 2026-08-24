"""BFCL-side tool schema normalize + hash (parity with tool-lora/bfcl_schema_utils)."""

from __future__ import annotations

import hashlib
import json


def normalize_bfcl_function(func: dict) -> dict:
    """Normalize a BFCL function definition to OpenAI-style JSON Schema types."""
    func = json.loads(json.dumps(func))  # deep copy
    func.pop("name_original", None)

    _TYPE_MAP = {
        "dict": "object",
        "tuple": "array",
        "float": "number",
        "hashmap": "object",
        "arraylist": "array",
        "any": "string",
    }
    _JSON_TYPES = {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }

    def _normalize_schema(schema: dict):
        t = schema.get("type")
        if isinstance(t, str):
            lowered = t.casefold()
            schema["type"] = _TYPE_MAP.get(
                lowered, lowered if lowered in _JSON_TYPES else t
            )
        elif isinstance(t, list):
            schema["type"] = [
                _TYPE_MAP.get(
                    item.casefold(),
                    item.casefold() if item.casefold() in _JSON_TYPES else item,
                )
                if isinstance(item, str)
                else item
                for item in t
            ]
        schema.pop("optional", None)
        schema.pop("name_original", None)
        for prop in schema.get("properties", {}).values():
            _normalize_schema(prop)
        if "items" in schema and isinstance(schema["items"], dict):
            _normalize_schema(schema["items"])

    params = func.get("parameters")
    if not isinstance(params, dict):
        params = {"type": "object", "properties": {}}
        func["parameters"] = params
    _normalize_schema(params)
    params.setdefault("type", "object")
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    params.pop("optional", None)
    return func


def normalize_functions(functions: list[dict]) -> list[dict]:
    return [normalize_bfcl_function(f) for f in functions]


def build_tools_json(functions: list[dict], *, already_normalized: bool = False) -> str:
    norms = functions if already_normalized else normalize_functions(functions)
    tools = [{"type": "function", "function": f} for f in norms]
    return json.dumps(tools, indent=2)


def tools_hash(functions: list[dict], *, already_normalized: bool = False) -> str:
    norms = functions if already_normalized else normalize_functions(functions)
    blob = json.dumps(norms, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
