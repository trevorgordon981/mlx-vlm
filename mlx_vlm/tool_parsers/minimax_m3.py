"""Tool-call parser for MiniMax-M3 (``minimax_m3_vl``).

MiniMax-M3 emits an XML tool-call envelope in which every tag is prefixed by the
model's namespace token ``]<]minimax[>[`` (see the model's ``chat_template.jinja``
``to_xml`` macro). A block the model generates looks like::

    ]<]minimax[>[<tool_call>
    ]<]minimax[>[<invoke name="get_weather">
    ]<]minimax[>[<location>Boston]<]minimax[>[</location>
    ]<]minimax[>[<units>celsius]<]minimax[>[</units>
    ]<]minimax[>[</invoke>
    ]<]minimax[>[</tool_call>

Scalar argument values are rendered bare (``{{ val }}``), booleans via ``tojson``.
Nested objects render as nested ``<key>...</key>`` tags; arrays render as repeated
``<item>...</item>`` tags. This parser inverts that encoding into the OpenAI
tool-call shape ``{"name", "arguments": <json string>}``.

Without this parser mlx_lm's ``_infer_tool_parser`` mis-detects M3 as ``json_tools``
(both templates contain ``<tool_call>``), which cannot parse M3's XML output.
"""

import json
import re

# Namespace token that M3 prefixes onto every tool-call XML tag.
NS_TOKEN = "]<]minimax[>["

# Envelope markers used by the server's process_tool_calls() to slice a block out
# of the raw model output. After slicing, the inner text (containing one or more
# <invoke> elements, each tag still ns-prefixed) is handed to parse_tool_call().
tool_call_start = NS_TOKEN + "<tool_call>"
tool_call_end = NS_TOKEN + "</tool_call>"

_INVOKE_RE = re.compile(r'<invoke\s+name="([^"]*)"\s*>(.*?)</invoke>', re.DOTALL)
# An opening element tag with no attributes, e.g. <location> or <item>.
_OPEN_TAG_RE = re.compile(r"<([A-Za-z_][\w.-]*)\s*>")


def _strip_ns(text: str) -> str:
    return text.replace(NS_TOKEN, "")


def _coerce_scalar(text: str):
    """Bare scalar -> python value. Mirrors sibling parsers: try JSON, else str."""
    s = text.strip()
    if s == "":
        return ""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _find_close(body: str, key: str, start: int):
    """Return (inner_start, inner_end, after) for the balanced </key> matching the
    <key> whose content begins at *start*, honoring same-name nesting. Returns None
    if unbalanced."""
    tag_re = re.compile(r"<(/?)" + re.escape(key) + r"\s*>")
    depth = 1
    cur = start
    while True:
        m = tag_re.search(body, cur)
        if not m:
            return None
        if m.group(1) == "":  # another <key>
            depth += 1
        else:  # a </key>
            depth -= 1
            if depth == 0:
                return start, m.start(), m.end()
        cur = m.end()


def _parse_children(body: str):
    """Parse a tag body into a python value.

    - repeated <item>..</item>        -> list
    - one or more distinct <key>..</key> -> dict
    - no child element tags           -> coerced scalar
    """
    children = []  # list[(key, inner_text)]
    pos = 0
    while True:
        m = _OPEN_TAG_RE.search(body, pos)
        if not m:
            break
        key = m.group(1)
        found = _find_close(body, key, m.end())
        if found is None:
            # Unbalanced; treat remainder as scalar tail and stop.
            break
        inner_start, inner_end, after = found
        children.append((key, body[inner_start:inner_end]))
        pos = after

    if not children:
        return _coerce_scalar(body)
    if all(k == "item" for k, _ in children):
        return [_parse_children(inner) for _, inner in children]
    return {k: _parse_children(inner) for k, inner in children}


def parse_tool_call(text: str, tools=None):
    """Parse one M3 tool-call block (ns markers may still be present) into OpenAI
    tool-call dict(s). Returns a single dict for one invoke, or a list for many."""
    body = _strip_ns(text)
    calls = []
    for m in _INVOKE_RE.finditer(body):
        name = m.group(1).strip()
        parsed = _parse_children(m.group(2))
        if isinstance(parsed, dict):
            arguments = parsed
        elif parsed in ("", None):
            arguments = {}
        else:
            arguments = {"value": parsed}
        calls.append(
            {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}
        )
    if not calls:
        raise ValueError("No <invoke> element found in MiniMax-M3 tool call.")
    return calls if len(calls) > 1 else calls[0]
