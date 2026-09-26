"""The training format must be exactly what BFCL shows the model at test time."""
import pytest

from nanodistill.data import bfcl_tool, completion, parse_calls, prompt, same_calls

XLAM_TOOL = {"name": "live_giveaways_by_type", "description": "Retrieve live giveaways.",
             "parameters": {"type": {"description": "Giveaway type.", "type": "str", "default": "game"},
                            "limit": {"description": "How many.", "type": "int"},
                            "ids": {"description": "Ids.", "type": "List[int]"},
                            "tag": {"description": "Tag.", "type": "str, optional"}}}


def test_bfcl_tool_types_and_required():
    tool = bfcl_tool(XLAM_TOOL)
    props = tool["parameters"]["properties"]
    assert props["type"]["type"] == "string" and props["limit"]["type"] == "integer"
    assert props["ids"] == {"type": "array", "items": {"type": "integer"}, "description": "Ids."}
    assert tool["parameters"]["required"] == ["limit", "ids"]           # defaults and optionals are not required
    assert tool["description"].endswith("Note that the provided function is in Python 3 syntax.")


def test_prompt_matches_bfcl_qwen_handler():
    handler = pytest.importorskip("bfcl_eval.model_handler.local_inference.qwen_fc")
    tools = [bfcl_tool(XLAM_TOOL)]
    query = "Show me beta giveaways."
    expected = handler.QwenFCHandler._format_prompt(None, [{"role": "user", "content": query}], tools)
    assert prompt(tools, query) == expected


def test_completion_round_trips_through_the_bfcl_parser():
    calls = [{"name": "f", "arguments": {"x": 1}}, {"name": "g", "arguments": {"y": "a"}}]
    text = completion(calls, reasoning="The user wants two things.")
    assert text.startswith("<think>\nThe user wants two things.\n</think>\n\n") and text.endswith("<|im_end|>")
    handler = pytest.importorskip("bfcl_eval.model_handler.local_inference.qwen_fc")
    assert handler.QwenFCHandler._extract_tool_calls(text) == calls
    assert parse_calls(text) == calls


def test_same_calls_ignores_order_and_int_float():
    a = [{"name": "f", "arguments": {"x": 5}}, {"name": "g", "arguments": {}}]
    b = [{"name": "g", "arguments": {}}, {"name": "f", "arguments": {"x": 5.0}}]
    assert same_calls(a, b) and not same_calls(a, b[:1])
