import json

import pytest

from mlx_vlm.tool_parsers import _infer_tool_parser
from mlx_vlm.tool_parsers.minimax_m3 import (
    NS_TOKEN,
    parse_tool_call,
    tool_call_end,
    tool_call_start,
)

NS = NS_TOKEN


def _wire(name, arg_xml):
    """Reconstruct exactly what M3's chat_template emits inside an <invoke>."""
    return f'{NS}<invoke name="{name}">\n{arg_xml}{NS}</invoke>\n'


def _inner(*invokes):
    """The text process_tool_calls() hands to parse_tool_call: the block with the
    outer start/end markers already stripped."""
    return "\n" + "".join(invokes)


def test_envelope_markers():
    assert tool_call_start == "]<]minimax[>[<tool_call>"
    assert tool_call_end == "]<]minimax[>[</tool_call>"


def test_single_call_scalar_args():
    arg_xml = f"{NS}<ticker>NVDA{NS}</ticker>\n{NS}<qty>100{NS}</qty>\n"
    result = parse_tool_call(_inner(_wire("buy_stock", arg_xml)))
    assert result["name"] == "buy_stock"
    assert json.loads(result["arguments"]) == {"ticker": "NVDA", "qty": 100}


def test_types_bool_number_string():
    arg_xml = (
        f"{NS}<flag>true{NS}</flag>\n"
        f"{NS}<price>12.5{NS}</price>\n"
        f"{NS}<note>hold for now{NS}</note>\n"
    )
    result = parse_tool_call(_inner(_wire("t", arg_xml)))
    assert json.loads(result["arguments"]) == {
        "flag": True,
        "price": 12.5,
        "note": "hold for now",
    }


def test_multiple_invokes_returns_list():
    a = _wire("f1", f"{NS}<x>1{NS}</x>\n")
    b = _wire("f2", f"{NS}<y>2{NS}</y>\n")
    result = parse_tool_call(_inner(a, b))
    assert isinstance(result, list)
    assert [c["name"] for c in result] == ["f1", "f2"]
    assert json.loads(result[0]["arguments"]) == {"x": 1}
    assert json.loads(result[1]["arguments"]) == {"y": 2}


def test_nested_object():
    inner = f"{NS}<a>1{NS}</a>\n{NS}<b>hi{NS}</b>\n"
    arg_xml = f"{NS}<opts>{inner}{NS}</opts>\n"
    result = parse_tool_call(_inner(_wire("cfg", arg_xml)))
    assert json.loads(result["arguments"]) == {"opts": {"a": 1, "b": "hi"}}


def test_array_of_scalars():
    items = f"{NS}<item>a{NS}</item>\n{NS}<item>b{NS}</item>\n"
    arg_xml = f"{NS}<tags>{items}{NS}</tags>\n"
    result = parse_tool_call(_inner(_wire("tag", arg_xml)))
    assert json.loads(result["arguments"]) == {"tags": ["a", "b"]}


def test_no_arg_call():
    result = parse_tool_call(_inner(_wire("ping", "")))
    assert result["name"] == "ping"
    assert json.loads(result["arguments"]) == {}


def test_raises_without_invoke():
    with pytest.raises(ValueError):
        parse_tool_call("no invoke here")


def test_infer_routes_minimax_m3():
    # Any template carrying the M3 namespace marker must route to minimax_m3,
    # even though it also contains the '<tool_call>' substring mlx_lm keys on.
    tmpl = "prefix ]<]minimax[>[<tool_call> ... tool_call.name <tool_call> suffix"
    assert _infer_tool_parser(tmpl) == "minimax_m3"


def test_integration_with_process_tool_calls():
    """End-to-end through the server's block-slicing logic."""
    from mlx_vlm.server import responses_state as rs
    from mlx_vlm.tool_parsers import minimax_m3 as mod

    full = (
        "Sure, buying now.\n"
        + tool_call_start
        + "\n"
        + _wire("buy_stock", f"{NS}<ticker>NVDA{NS}</ticker>\n")
        + tool_call_end
    )
    out = rs.process_tool_calls(full, mod, tools=None)
    assert len(out["calls"]) == 1
    call = out["calls"][0]
    assert call["function"]["name"] == "buy_stock"
    assert json.loads(call["function"]["arguments"]) == {"ticker": "NVDA"}
    assert "buying now" in out["remaining_text"]
