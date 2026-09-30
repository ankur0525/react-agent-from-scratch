"""Tools the agent is allowed to call. Kept deliberately free of external APIs
so the whole thing runs offline except for the actual LLM call."""

import ast
import operator

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str):
    """Arithmetic only, no variables, no function calls - parsed with ast
    instead of eval() so the agent can't be tricked into running arbitrary code."""
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


_CONVERSIONS = {
    ("celsius", "fahrenheit"): lambda c: c * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda f: (f - 32) * 5 / 9,
    ("km", "miles"): lambda k: k * 0.621371,
    ("miles", "km"): lambda m: m / 0.621371,
}


def convert_units(value: float, from_unit: str, to_unit: str):
    key = (from_unit.lower(), to_unit.lower())
    if key not in _CONVERSIONS:
        return f"No conversion available from {from_unit} to {to_unit}"
    return round(_CONVERSIONS[key](value), 4)


_NOTES = {
    "react pattern": "Thought -> Action -> Observation, repeated until the model has enough to answer.",
    "pandas multiindex": "An index with multiple levels, created with set_index() on a list of columns.",
    "bm25": "Sparse ranking function for search/retrieval, scores by term frequency and rarity.",
}


def lookup_note(topic: str):
    """Tiny hardcoded knowledge base, standing in for a real notes/vector store."""
    return _NOTES.get(topic.lower(), f"No note saved for '{topic}' yet.")


TOOLS = {
    "calculate": calculate,
    "convert_units": convert_units,
    "lookup_note": lookup_note,
}

TOOL_DESCRIPTIONS = """\
- calculate(expression: str) -> evaluates a basic arithmetic expression, e.g. "12 * (3 + 1)"
- convert_units(value: float, from_unit: str, to_unit: str) -> converts celsius/fahrenheit or km/miles
- lookup_note(topic: str) -> looks up a short saved note ("react pattern", "pandas multiindex", "bm25")\
"""
