"""Compare latency and token usage with and without prompt caching.

Run:  python examples/benchmark_cache.py
Paste the printed table into README.md under "Benchmark results".
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic
from vonnneumann import AgentStateRouter
from demo import SHARED_RULES

RUNS = 5
PROMPT = "Reply with one short sentence: what is 2 + 2?"


async def bench(use_cache: bool) -> AgentStateRouter:
    client = anthropic.AsyncAnthropic()
    router = AgentStateRouter(system_prompt=SHARED_RULES)
    for i in range(RUNS):
        # new agent each time so only the system prompt is shared
        await router.run(client, f"agent{i}", PROMPT, max_tokens=50,
                         use_cache=use_cache)
    return router


async def main() -> None:
    for label, flag in (("WITHOUT cache", False), ("WITH cache", True)):
        r = await bench(flag)
        avg = sum(s["latency_s"] for s in r.stats) / len(r.stats)
        reads = sum(s["cache_read_tokens"] for s in r.stats)
        print(f"\n## {label}  (avg latency {avg:.2f}s, cache-read tokens {reads})")
        print(r.report())


if __name__ == "__main__":
    asyncio.run(main())
