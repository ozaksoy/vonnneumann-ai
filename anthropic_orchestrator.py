"""Minimal orchestrator example - the same code shown on vonnneumann.com.

For the full multi-agent demo and caching benchmark, see examples/.
"""
import asyncio

import anthropic
from vonnneumann import AgentStateRouter

client = anthropic.AsyncAnthropic()
router = AgentStateRouter(
    system_prompt="You are a concise, factual assistant.",
    cache_ttl_sec=300,
)


async def run_claude_agent(prompt: str, context: dict | None = None) -> str:
    # Direct Anthropic API call with prompt caching on the shared system prompt.
    # Note: caching only applies when the system prompt is >= 512 tokens;
    # examples/demo.py uses a long shared prompt to show this.
    response = await client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        system=router.system_block(),
        messages=[{"role": "user", "content": prompt}],
    )
    return router.process_agent_state(response)


if __name__ == "__main__":
    print(asyncio.run(run_claude_agent("Say hello in one sentence.")))
