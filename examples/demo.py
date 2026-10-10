"""Two isolated agents share one cached system prompt.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/demo.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from vonnneumann import AgentStateRouter, SchemaValidationError

# Long, shared instructions (prompt caching needs >= 512 tokens on Sonnet 5.5).
SHARED_RULES = "\n".join(
    f"Rule {i}: Be concise, factual, and never invent numbers. "
    f"If data is missing, say so explicitly. Answer in English."
    for i in range(1, 61)
)


async def main() -> None:
    client = anthropic.AsyncAnthropic()
    router = AgentStateRouter(system_prompt=SHARED_RULES)

    # Agent 1: free text
    print(await router.run(client, "researcher",
                           "In two sentences, what is prompt caching?"))

    # Agent 2: must return JSON; its context is separate from agent 1
    raw = await router.run(
        client, "scorer",
        'Return only JSON: {"summary": string, "score": number 1-10} '
        "rating how useful prompt caching is for multi-agent systems.")
    try:
        print(router.validate_json(raw, {"summary": "string", "score": "number"}))
    except SchemaValidationError as e:
        print("Validation failed:", e)

    print("\n" + router.report())


if __name__ == "__main__":
    asyncio.run(main())
