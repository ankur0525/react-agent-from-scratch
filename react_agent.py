"""A ReAct-style agent loop, written from scratch to understand how the
pattern actually works under the hood before reaching for a framework.

ReAct = the model alternates between reasoning ("Thought") and acting
("Action"), and gets fed the result of each action ("Observation") before
deciding what to do next. It keeps going until it's confident enough to give
a final "Answer".
"""

import argparse
import json
import re

from llm_client import chat
from tools import TOOLS, TOOL_DESCRIPTIONS

SYSTEM_PROMPT = f"""You are a reasoning agent that solves problems step by step.

Work in a loop of Thought, Action, Observation, and finish with an Answer.

Rules:
- Thought: one line reasoning about what to do next.
- Action: call exactly one tool as JSON on its own line: {{"tool": "<name>", "args": {{...}}}}
  Then stop - do not guess the Observation yourself, wait for it to be given to you.
- Once you have enough information, skip Thought/Action and respond with just:
  Answer: <your final answer>

Available tools:
{TOOL_DESCRIPTIONS}

Example:
Thought: I need to convert 100 celsius to fahrenheit before I can answer.
Action: {{"tool": "convert_units", "args": {{"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}}}}
"""


def extract_action(model_reply: str):
    match = re.search(r"Action:\s*(\{.*\})", model_reply, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def run(question: str, max_turns: int = 6, mock: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for turn in range(1, max_turns + 1):
        print(f"--- turn {turn} ---")

        reply = _mock_reply(turn) if mock else chat(messages)
        print(reply)
        messages.append({"role": "assistant", "content": reply})

        if reply.strip().startswith("Answer:"):
            return reply

        action = extract_action(reply)
        if action is None:
            # Model didn't call a tool and didn't answer - nothing useful to
            # feed back, so stop instead of looping on the same reply.
            return reply

        tool_name = action.get("tool")
        tool_args = action.get("args", {})

        if tool_name not in TOOLS:
            observation = f"Unknown tool: {tool_name}"
        else:
            try:
                observation = TOOLS[tool_name](**tool_args)
            except Exception as exc:
                observation = f"Tool error: {exc}"

        print(f"Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Gave up after max_turns without reaching a final answer."


def _mock_reply(turn: int) -> str:
    """Scripted response for --mock so the loop mechanics can be demoed/tested
    without an API key or spending any tokens. Only covers the temperature
    example below - swap the script if you try a different question."""
    script = [
        'Thought: I should convert the temperature first.\n'
        'Action: {"tool": "convert_units", "args": {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"}}',
        "Answer: 100 Celsius is 212 Fahrenheit.",
    ]
    return script[min(turn - 1, len(script) - 1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ReAct agent on a question.")
    parser.add_argument("question", help="e.g. \"What is 100 celsius in fahrenheit?\"")
    parser.add_argument("--mock", action="store_true", help="Skip the real LLM call, use a scripted reply")
    parsed = parser.parse_args()

    run(parsed.question, mock=parsed.mock)
