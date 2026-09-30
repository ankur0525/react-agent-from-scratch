"""Thin wrapper around the OpenAI client so react_agent.py doesn't need to
know which provider is behind `chat()`. Swap this file out to point at a
different provider without touching the agent loop."""

import os

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily so --mock runs without this installed
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def chat(messages, model="gpt-4o-mini"):
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content
