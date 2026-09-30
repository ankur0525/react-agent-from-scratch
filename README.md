# ReAct agent from scratch

After going through a course on building AI agents, I wanted to actually write my own version of a ReAct-style agent instead of keeping the course's code around - so this is a clean rebuild of the same idea, my own code, no dependency on anything from that course.

## What ReAct actually means

It's short for Reasoning + Acting. The model doesn't just answer straight away - it runs in a loop:

1. **Thought** - the model reasons out loud about what to do next
2. **Action** - it picks a tool and calls it, formatted as JSON
3. **Observation** - the tool's actual output gets fed back into the conversation
4. **Answer** - once it has enough, it stops looping and gives a final answer

The loop can run several times before the model is ready to answer - that's the whole point, it lets the model gather information it didn't have up front instead of hallucinating an answer.

## What's in here

- `tools.py` - three small tools the agent can call: a safe arithmetic evaluator (built with `ast`, not `eval()`, so it can't be tricked into running arbitrary code), a unit converter (celsius/fahrenheit, km/miles), and a tiny hardcoded notes lookup
- `llm_client.py` - a one-function wrapper around the OpenAI API, kept separate so swapping providers later doesn't mean touching the agent loop itself
- `react_agent.py` - the actual loop: builds the prompt, calls the model, pulls the JSON action out of the reply with a regex, runs the tool, feeds the result back, repeats

## Running it

```
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python react_agent.py "What is 100 celsius in fahrenheit?"
```

There's also a `--mock` flag if you just want to see the loop mechanics run without spending any API tokens:

```
python react_agent.py "What is 100 celsius in fahrenheit?" --mock
```

It replays a scripted response instead of calling the real model - only really makes sense for that exact temperature question above since the script isn't dynamic, but it's enough to see the Thought/Action/Observation/Answer flow end to end.

## Honest limitations

- Only one tool call per turn, no parallel tool calls
- If the model's reply doesn't match the expected `Action: {...}` format, the loop just gives up instead of trying to recover - a real production agent would probably retry with a nudge
- `lookup_note` is a hardcoded dict, obviously a stand-in for a real notes store or vector DB
