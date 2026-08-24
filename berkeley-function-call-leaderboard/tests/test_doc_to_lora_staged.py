import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from bfcl_eval.constants.supported_models import SUPPORTED_MODELS
from bfcl_eval.model_handler.local_inference import binder_worker, d2l_worker
from bfcl_eval.model_handler.local_inference.binder_worker import (
    apply_binder_chat_template,
)
from bfcl_eval.model_handler.local_inference.doc_to_lora import DocToLoraHandler
from bfcl_eval.model_handler.local_inference.doc_to_lora_staged import (
    BASELINE_BIND_SYSTEM_MESSAGE,
    BASELINE_SELECT_SYSTEM_PREFIX,
    BINDER_SYSTEM_MESSAGE,
    B_JOINT_BIND_SYSTEM_MESSAGE,
    META_CALL_SYSTEM_PREFIX,
    DocToLoraMetaIntentBinderHandler,
    DocToLoraMetaSelectBindBaseHandler,
    DocToLoraMetaSelectBindHandler,
    StagedValidationError,
    binder_prompt_contains_original_query,
    build_binder_messages,
    build_plain_schema_bind_messages,
    build_route_and_plan_tool,
    build_select_tool,
    parse_exactly_one_tool_call,
    validate_call_against_function,
    _append_missing_end_tag,
    _build_stage_one_messages,
    _build_variant_b_messages,
    _is_supported_staged_entry_id,
    _repair_schema_key_syntax,
)


def tagged(call):
    return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"


def result(call, enable_thinking=None):
    text = tagged(call)
    value = {
        "text": text,
        "decoded_text": text,
        "input_tokens": 10,
        "output_tokens": 5,
        "token_ids": [101, 102, 103],
        "constraint_mode": "none",
    }
    if enable_thinking is not None:
        value["enable_thinking"] = enable_thinking
    return value


def weather_function():
    return {
        "name": "weather.lookup",
        "description": "Look up weather.",
        "parameters": {
            "type": "dict",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["c", "f"]},
                "options": {
                    "type": "dict",
                    "properties": {
                        "days": {"type": "integer"},
                    },
                    "required": ["days"],
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "optional": True,
                },
            },
            "required": ["location", "units", "options"],
        },
    }


class FakeWorker:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, command, args=None):
        self.requests.append((command, args or {}))
        if command in ("internalize", "reset"):
            return {}
        if command in (
            "generate",
            "score_router_candidates",
            "generate_late_schema",
            "start_late_session",
            "append_late_schema_session",
        ):
            response = self.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            response.setdefault("constraint_mode", "none")
            if command == "start_late_session":
                response.setdefault("session_id", "session-1")
            if command in ("generate_late_schema", "append_late_schema_session"):
                response.setdefault(
                    "messages",
                    [
                        *args.get("stage1_messages", []),
                        {"role": "assistant", "content": "router"},
                        {
                            "role": "tool",
                            "content": json.dumps(
                                {
                                    "selected_tool_schema": {
                                        "type": "function",
                                        "function": args["raw_function"],
                                    }
                                }
                            ),
                        },
                    ],
                )
                response.setdefault(
                    "transcript",
                    {
                        "p_hash": "p",
                        "r_hash": "r",
                        "s_hash": "s",
                        "full_hash": "full",
                    },
                )
            return response
        if command == "close_session":
            return {"closed": True}
        raise AssertionError(f"unexpected command: {command}")

    @property
    def is_alive(self):
        return True


def bare_handler(
    handler_class,
    main_responses,
    binder_responses=None,
    *,
    skip_internalize=False,
    use_baseline_prompts=False,
    bind_on_base=False,
    stateful_continuation=False,
    plain_schema_bind=False,
    router_score_candidates=False,
    router_schema_ablation="full",
    routing_only=False,
):
    handler = object.__new__(handler_class)
    handler.variant = handler_class.variant
    handler.temperature = 0
    handler.chunk_size = 128
    handler.max_new_tokens = 128
    handler.binder_max_new_tokens = 128
    handler.skip_internalize = skip_internalize
    handler.use_baseline_prompts = use_baseline_prompts
    handler.bind_on_base = bind_on_base
    handler.native_fc_binder = False
    handler.stateful_continuation = stateful_continuation
    handler.oracle_route = False
    handler.plain_schema_bind = plain_schema_bind
    handler.router_score_candidates = router_score_candidates
    handler.router_schema_ablation = router_schema_ablation
    handler.routing_only = routing_only
    handler._main_worker = FakeWorker(main_responses)
    handler._binder_worker = (
        FakeWorker(binder_responses) if binder_responses is not None else None
    )
    handler._raw_log = None
    handler._log_lock = threading.Lock()
    handler.last_trace = None
    return handler


class StagedSchemaTests(unittest.TestCase):
    def test_raw_trace_records_are_valid_jsonl(self):
        handler = object.__new__(DocToLoraMetaSelectBindHandler)
        handler._log_lock = threading.Lock()
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            handler._raw_log_fd = os.open(
                path, os.O_WRONLY | os.O_TRUNC | os.O_APPEND
            )
            handler._write_trace({"id": "one", "text": '"quoted"'})
            handler._write_trace({"id": "two", "nested": {"ok": True}})
            os.close(handler._raw_log_fd)
            handler._raw_log_fd = None
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["id"] for row in rows], ["one", "two"])
        finally:
            os.unlink(path)

    def test_worker_dispatch_forwards_strict_grammar_flag(self):
        with patch.object(d2l_worker, "_generate", return_value={}) as generate:
            d2l_worker._DISPATCH["generate"](
                {"messages": [], "strict_json_schema": True}
            )
        self.assertIs(generate.call_args.args[-1], True)

        with patch.object(binder_worker, "_generate", return_value={}) as generate:
            binder_worker._DISPATCH["generate"](
                {
                    "messages": [],
                    "constraint_tools": [{}],
                    "strict_json_schema": False,
                }
            )
        self.assertIs(generate.call_args.args[-1], False)

    def test_dynamic_real_name_enum(self):
        names = ["maps.shortest_path", "WeatherAPI.getForecast"]
        self.assertEqual(
            build_route_and_plan_tool(names)["parameters"]["properties"]["tool_name"][
                "enum"
            ],
            names,
        )
        self.assertEqual(
            build_select_tool(names)["parameters"]["properties"]["tool_name"]["enum"],
            names,
        )
        with self.assertRaises(StagedValidationError):
            build_select_tool(["duplicate", "duplicate"])

    def test_staged_single_turn_entry_ids(self):
        self.assertTrue(_is_supported_staged_entry_id("multiple_1"))
        self.assertTrue(_is_supported_staged_entry_id("live_simple_1"))
        self.assertFalse(_is_supported_staged_entry_id("simple_1"))

    def test_exactly_one_call_enforced(self):
        call = {"name": "x", "arguments": {}}
        self.assertEqual(parse_exactly_one_tool_call(tagged(call)), call)
        for bad in (
            json.dumps(call),
            "prose " + tagged(call),
            tagged(call) + tagged(call),
            "<tool_call>{bad}</tool_call>",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(StagedValidationError):
                    parse_exactly_one_tool_call(bad)

    def test_missing_end_tag_repair_preserves_raw_json(self):
        raw = '<tool_call>\n{"name":"x","arguments":{}}'
        repaired = _append_missing_end_tag(raw)
        self.assertNotEqual(raw, repaired)
        self.assertEqual(
            parse_exactly_one_tool_call(repaired),
            {"name": "x", "arguments": {}},
        )

    def test_meta_enum_duplicate_quote_is_repaired(self):
        raw = (
            '<tool_call>\n{"name":"select_tool","arguments":'
            '{"tool_name":"weather.lookup""}}\n</tool_call>'
        )
        repaired = _append_missing_end_tag(raw)
        self.assertEqual(
            parse_exactly_one_tool_call(repaired),
            {
                "name": "select_tool",
                "arguments": {"tool_name": "weather.lookup"},
            },
        )
        multiple = tagged({"name": "x", "arguments": {}}) * 2
        self.assertEqual(_append_missing_end_tag(multiple), multiple)

    def test_schema_key_quote_and_colon_repairs_are_bounded(self):
        function = weather_function()
        missing_colon = (
            '<tool_call>\n{"name":"weather.lookup","arguments":'
            '{"location""Paris","units":"c","options":{days:3}}}'
        )
        repaired = _append_missing_end_tag(
            _repair_schema_key_syntax(missing_colon, function)
        )
        self.assertEqual(
            parse_exactly_one_tool_call(repaired)["arguments"]["options"],
            {"days": 3},
        )
        assignment = '{"arguments":{"location"="Paris"}}'
        self.assertEqual(
            _repair_schema_key_syntax(assignment, function),
            '{"arguments":{"location":"Paris"}}',
        )
        unknown = '{"arguments":{unknown:3}}'
        self.assertEqual(_repair_schema_key_syntax(unknown, function), unknown)

    def test_nested_schema_and_optional_omission(self):
        function = weather_function()
        valid = {
            "name": "weather.lookup",
            "arguments": {
                "location": "Paris",
                "units": "c",
                "options": {"days": 3},
            },
        }
        validate_call_against_function(valid, function)
        invalid = json.loads(json.dumps(valid))
        invalid["arguments"]["options"]["extra"] = True
        with self.assertRaises(StagedValidationError):
            validate_call_against_function(invalid, function)
        invalid = json.loads(json.dumps(valid))
        invalid["arguments"]["units"] = "kelvin"
        with self.assertRaises(StagedValidationError):
            validate_call_against_function(invalid, function)

    def test_free_form_object_allows_undeclared_keys(self):
        function = {
            "name": "grades.average",
            "parameters": {
                "type": "object",
                "properties": {
                    "gradeDict": {"type": "object"},
                },
                "required": ["gradeDict"],
            },
        }
        validate_call_against_function(
            {
                "name": "grades.average",
                "arguments": {
                    "gradeDict": {
                        "math": 90,
                        "science": 75,
                    }
                },
            },
            function,
        )

    def test_bfcl_schema_normalization_matches_training_shape(self):
        function = weather_function()
        function["name_original"] = "Weather Lookup"
        function["parameters"]["properties"]["location"]["type"] = "String"
        normalized = DocToLoraHandler._normalize_bfcl_function(function)
        self.assertNotIn("name_original", normalized)
        self.assertEqual(normalized["parameters"]["type"], "object")
        self.assertEqual(
            normalized["parameters"]["properties"]["location"]["type"],
            "string",
        )
        self.assertNotIn(
            "optional", normalized["parameters"]["properties"]["tags"]
        )

    def test_binder_boundary_excludes_original_query(self):
        messages = build_binder_messages(
            weather_function(),
            {
                "action": "look up weather",
                "facts": [{"meaning": "place", "value": "Paris"}],
            },
        )
        original = [{"role": "user", "content": "Will it rain in Paris tomorrow?"}]
        self.assertFalse(binder_prompt_contains_original_query(messages, original))
        self.assertEqual(messages[0]["content"], BINDER_SYSTEM_MESSAGE)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(set(payload), {"intent"})
        self.assertNotIn(": ", messages[1]["content"])

    def test_stage_one_prompt_matches_training_contract(self):
        meta = build_select_tool(["weather.lookup", "maps.route"])
        messages = _build_stage_one_messages(
            meta, [{"role": "user", "content": "weather"}]
        )
        self.assertTrue(messages[0]["content"].startswith(META_CALL_SYSTEM_PREFIX))
        schema = json.loads(
            messages[0]["content"][len(META_CALL_SYSTEM_PREFIX) :]
        )
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"], meta)

    def test_variant_b_prompt_matches_training_contract(self):
        messages = _build_variant_b_messages(
            weather_function(),
            [{"role": "user", "content": "weather in Paris"}],
        )
        self.assertEqual(messages[0]["content"], B_JOINT_BIND_SYSTEM_MESSAGE)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["original_query"], "weather in Paris")
        self.assertEqual(
            payload["selected_tool_schema"]["function"]["name"],
            "weather.lookup",
        )
        self.assertNotIn(": ", messages[1]["content"])

    def test_plain_schema_bind_messages_drop_router_and_preserve_raw_schema(self):
        stage1_messages = _build_stage_one_messages(
            build_select_tool(["weather.lookup", "maps.route"]),
            [{"role": "user", "content": "weather in Paris"}],
            system_prefix=META_CALL_SYSTEM_PREFIX,
        )
        messages = build_plain_schema_bind_messages(
            stage1_messages,
            weather_function(),
        )
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "user"],
        )
        self.assertEqual(
            sum(
                message.get("content") == "weather in Paris"
                for message in messages
            ),
            1,
        )
        rendered = json.dumps(messages)
        self.assertNotIn('"name": "select_tool", "arguments"', rendered)
        payload = messages[-1]["content"]
        self.assertNotIn("<tool_response>", payload)
        self.assertIn('"type":"dict"', payload)
        self.assertIn('"optional":true', payload)

    def test_binder_template_disables_thinking(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "ids"

        tokenizer = Tokenizer()
        self.assertEqual(
            apply_binder_chat_template(
                tokenizer,
                [{"role": "user", "content": "{}"}],
                [{"type": "function", "function": weather_function()}],
                "<tool_call>\n",
            ),
            "ids",
        )
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)
        self.assertIs(tokenizer.kwargs["continue_final_message"], True)


class StagedFlowTests(unittest.TestCase):
    def setUp(self):
        self.functions = [
            weather_function(),
            {
                "name": "maps.route",
                "description": "Find a route.",
                "parameters": {
                    "type": "dict",
                    "properties": {"destination": {"type": "string"}},
                    "required": ["destination"],
                },
            },
        ]
        self.entry = {
            "id": "multiple_123",
            "function": self.functions,
        }
        self.original = [
            {"role": "user", "content": "Weather in Paris in celsius for three days"}
        ]
        self.final_call = {
            "name": "weather.lookup",
            "arguments": {
                "location": "Paris",
                "units": "c",
                "options": {"days": 3},
            },
        }

    def test_variant_a_uses_schema_intent_only_binder(self):
        route = {
            "name": "route_and_plan",
            "arguments": {
                "tool_name": "weather.lookup",
                "intent": {
                    "action": "look up weather",
                    "facts": [
                        {"meaning": "place", "value": "Paris"},
                        {"meaning": "temperature scale", "value": "celsius"},
                        {"meaning": "forecast duration", "value": 3},
                    ],
                },
            },
        }
        handler = bare_handler(
            DocToLoraMetaIntentBinderHandler,
            [result(route)],
            [result(self.final_call, enable_thinking=False)],
        )
        output, usage = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        binder_request = handler._binder_worker.requests[-1][1]
        self.assertFalse(
            binder_prompt_contains_original_query(
                binder_request["messages"], self.original
            )
        )
        self.assertEqual(
            binder_request["constraint_tools"][0]["function"]["name"],
            "weather.lookup",
        )
        self.assertIs(
            handler._main_worker.requests[1][1]["enable_thinking"], False
        )
        self.assertEqual(usage["input_tokens"], 20)

    def test_variant_b_keeps_adapter_and_appends_raw_schema_after_router(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [result(selection), result(self.final_call)],
        )
        output, _ = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        commands = [command for command, _ in handler._main_worker.requests]
        self.assertEqual(
            commands, ["internalize", "generate", "generate_late_schema"]
        )
        late_args = handler._main_worker.requests[-1][1]
        self.assertEqual(
            late_args["stage1_messages"][1]["content"],
            self.original[0]["content"],
        )
        self.assertEqual(late_args["raw_function"], self.functions[0])
        self.assertTrue(late_args["raw_function"]["parameters"]["properties"]["tags"]["optional"])
        self.assertIs(
            handler.last_trace["stage2"]["attempts"][0]["force_first_required"],
            False,
        )
        self.assertEqual(
            handler.last_trace["stage2"]["constraint_mode"], "none"
        )

    def test_variant_b_stateful_path_appends_without_resending_query(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [result(selection), result(self.final_call)],
            stateful_continuation=True,
        )
        output, _ = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        commands = [command for command, _ in handler._main_worker.requests]
        self.assertEqual(
            commands,
            ["internalize", "start_late_session", "append_late_schema_session"],
        )
        append_args = handler._main_worker.requests[-1][1]
        self.assertEqual(set(append_args), {"session_id", "raw_function", "max_new_tokens"})
        self.assertEqual(append_args["session_id"], "session-1")

    def test_variant_b_scores_candidates_and_externalizes_selector_syntax(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        scored = result(selection)
        scored["constraint_mode"] = "candidate_likelihood"
        scored["candidate_scores"] = [
            {"name": "weather.lookup", "score": -1.0},
            {"name": "maps.route", "score": -2.0},
        ]
        scored["choice_margin"] = 1.0
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [scored, result(self.final_call)],
            router_score_candidates=True,
        )
        output, _ = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        command, request = handler._main_worker.requests[1]
        self.assertEqual(command, "score_router_candidates")
        self.assertEqual(
            request["candidate_names"],
            ["weather.lookup", "maps.route"],
        )
        self.assertEqual(handler.last_trace["stage1"]["choice_margin"], 1.0)

    def test_names_only_ablation_removes_schema_semantics(self):
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [],
            router_schema_ablation="names_only",
        )
        normalized = [
            DocToLoraHandler._normalize_bfcl_function(function)
            for function in self.functions
        ]
        handler._internalize(normalized)
        command, request = handler._main_worker.requests[-1]
        self.assertEqual(command, "internalize")
        tools = json.loads(request["tool_defs"])
        self.assertEqual(tools[0]["function"]["description"], "")
        self.assertEqual(
            tools[0]["function"]["parameters"]["properties"],
            {},
        )

    def test_routing_only_skips_bind_generation(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [result(selection)],
            routing_only=True,
        )
        output, usage = handler._execute_stages(self.entry, self.original)
        self.assertEqual(
            parse_exactly_one_tool_call(output),
            {"name": "weather.lookup", "arguments": {}},
        )
        self.assertEqual(
            [command for command, _ in handler._main_worker.requests],
            ["internalize", "generate"],
        )
        self.assertTrue(handler.last_trace["stage2"]["skipped"])
        self.assertEqual(usage["input_tokens"], 10)

    def test_variant_b_plain_schema_bind_drops_router_turn_and_keeps_lora(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [result(selection), result(self.final_call)],
            plain_schema_bind=True,
        )
        output, _ = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        commands = [command for command, _ in handler._main_worker.requests]
        self.assertEqual(commands, ["internalize", "generate", "generate"])
        bind_request = handler._main_worker.requests[-1][1]
        bind_messages = bind_request["messages"]
        self.assertEqual(
            [message["role"] for message in bind_messages],
            ["system", "user", "user"],
        )
        self.assertTrue(
            bind_messages[0]["content"].startswith(
                META_CALL_SYSTEM_PREFIX
            )
        )
        rendered = json.dumps(bind_messages)
        self.assertNotIn(tagged(selection), rendered)
        self.assertNotIn("<tool_response>", bind_messages[-1]["content"])
        self.assertIn('"type":"dict"', bind_messages[-1]["content"])
        self.assertIn('"optional":true', bind_messages[-1]["content"])
        self.assertFalse(bind_request["restrict_toolgen"])
        self.assertIs(bind_request["enable_thinking"], False)
        self.assertEqual(
            handler.last_trace["protocol"],
            "router-free-plain-schema-bind",
        )

    def test_variant_b_base_uses_same_late_transcript_without_lora(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindBaseHandler,
            [result(selection), result(self.final_call)],
            skip_internalize=True,
            use_baseline_prompts=True,
        )
        output, _ = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        commands = [command for command, _ in handler._main_worker.requests]
        self.assertEqual(
            commands, ["reset", "generate", "generate_late_schema"]
        )
        stage1 = handler._main_worker.requests[1][1]["messages"]
        self.assertTrue(stage1[0]["content"].startswith(BASELINE_SELECT_SYSTEM_PREFIX))
        stage2 = handler._main_worker.requests[2][1]
        self.assertEqual(stage2["raw_function"], self.functions[0])
        self.assertEqual(
            stage2["stage1_messages"][1]["content"],
            self.original[0]["content"],
        )
        self.assertTrue(handler.last_trace["skip_internalize"])
        self.assertEqual(
            handler.last_trace["stage2"]["worker"], "base-qwen3-4b-no-d2l"
        )

    def test_variant_b_base_can_route_with_d2l_meta_prompt_and_no_lora(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindBaseHandler,
            [result(selection), result(self.final_call)],
            skip_internalize=True,
            use_baseline_prompts=False,
        )
        output, _ = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), self.final_call)
        stage1 = handler._main_worker.requests[1][1]["messages"]
        self.assertTrue(stage1[0]["content"].startswith(META_CALL_SYSTEM_PREFIX))
        self.assertEqual(
            [command for command, _ in handler._main_worker.requests],
            ["reset", "generate", "generate_late_schema"],
        )

    def test_stage_two_does_not_retry_after_schema_invalid_call(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        empty_call = {"name": "weather.lookup", "arguments": {}}
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler, [result(selection), result(empty_call)]
        )
        output, usage = handler._execute_stages(self.entry, self.original)
        self.assertEqual(parse_exactly_one_tool_call(output), empty_call)
        attempts = handler.last_trace["stage2"]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertIn("schema_validation_error", attempts[0])
        self.assertEqual(usage["input_tokens"], 20)

    def test_stage_two_runtime_error_is_not_retried(self):
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [result(selection), RuntimeError("generation failed")],
        )
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            handler._execute_stages(self.entry, self.original)
        attempts = handler.last_trace["stage2"]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertIn("validation_error", attempts[0])

    def test_stage_one_is_unconstrained_and_not_retried(self):
        malformed = {
            "text": "<tool_call>\nnot-json\n</tool_call>",
            "input_tokens": 10,
            "output_tokens": 10,
        }
        selection = {
            "name": "select_tool",
            "arguments": {"tool_name": "weather.lookup"},
        }
        handler = bare_handler(
            DocToLoraMetaSelectBindHandler,
            [malformed],
        )
        with self.assertRaises(StagedValidationError):
            handler._execute_stages(self.entry, self.original)
        attempts = handler.last_trace["stage1"]["attempts"]
        self.assertEqual(len(attempts), 1)
        generate_requests = [
            args
            for command, args in handler._main_worker.requests
            if command == "generate"
        ]
        self.assertIs(generate_requests[0]["strict_json_schema"], False)
        self.assertFalse(generate_requests[0]["restrict_toolgen"])
        self.assertEqual(generate_requests[0]["assistant_prefix"], "")

    def test_models_are_in_supported_index(self):
        self.assertIn(
            "doc-to-lora/qwen3-4b-meta-intent-0.6b", SUPPORTED_MODELS
        )
        self.assertIn(
            "doc-to-lora/qwen3-4b-meta-select-bind", SUPPORTED_MODELS
        )
        self.assertIn(
            "doc-to-lora/qwen3-4b-meta-select-bind-base", SUPPORTED_MODELS
        )


if __name__ == "__main__":
    unittest.main()
