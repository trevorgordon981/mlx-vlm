"""
Tool parser utilities for mlx-vlm.

Re-exports mlx_lm's _infer_tool_parser with additional model support,
and loads parsers from both mlx_lm.tool_parsers and mlx_vlm.tool_parsers.
"""

import importlib

from mlx_lm.tokenizer_utils import _infer_tool_parser as _mlx_lm_infer_tool_parser

# Additional patterns not covered by mlx_lm
_EXTRA_PATTERNS = [
    ("<|tool_call>", "gemma4"),
    ("<|START_ACTION|>", "cohere2_moe"),
]

# Patterns that must win over mlx_lm's inference. MiniMax-M3 wraps its tool-call
# XML in the namespace token ']<]minimax[>[' and its template also contains the
# bare '<tool_call>' substring, which mlx_lm otherwise mis-detects as 'json_tools'
# (a JSON parser that cannot read M3's XML envelope). Check the distinctive M3
# marker first so M3 routes to its own parser.
_PRIORITY_PATTERNS = [
    ("]<]minimax[>[", "minimax_m3"),
]


def _infer_tool_parser(chat_template):
    """Infer tool parser type: model-specific overrides, then mlx_lm, then extras."""
    if isinstance(chat_template, str):
        for marker, parser_type in _PRIORITY_PATTERNS:
            if marker in chat_template:
                return parser_type

    result = _mlx_lm_infer_tool_parser(chat_template)
    if result is not None:
        return result

    if not isinstance(chat_template, str):
        return None

    for marker, parser_type in _EXTRA_PATTERNS:
        if marker in chat_template:
            return parser_type

    return None


def _infer_tool_parser_from_processor(processor):
    """Infer tool parser type from processor's chat template."""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return _infer_tool_parser(tokenizer.chat_template)

    return None


def load_tool_module(tool_parser_type):
    """Load a tool parser module from mlx_vlm.tool_parsers or mlx_lm.tool_parsers."""
    if importlib.util.find_spec(f"mlx_vlm.tool_parsers.{tool_parser_type}"):
        return importlib.import_module(f"mlx_vlm.tool_parsers.{tool_parser_type}")
    return importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
