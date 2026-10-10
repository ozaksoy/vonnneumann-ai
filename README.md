# Vonn Neumann AI

An experimental, non-commercial state router for multi-agent workflows on the
Anthropic Claude API (`claude-sonnet-5-5`). Independent developer project.

## What it does
- **Shared cached system prompt**: sent with `cache_control` so Claude prompt
  caching can reuse it across agents.
- **Isolated agent contexts**: each agent keeps its own message history.
- **JSON schema checks**: agent outputs are validated before use downstream.

## Quick start
```bash
pip install -r requirements.txt
python tests_offline.py              # no API key needed
export ANTHROPIC_API_KEY=sk-ant-...
python examples/demo.py
python examples/benchmark_cache.py
```

## Benchmark results
_To be added after running `examples/benchmark_cache.py`._

## Status
Early proof of concept. Feedback: aksoy@vonnneumann.com
