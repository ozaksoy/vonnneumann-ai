# Vonn Neumann AI - Developer Preview

Vonn Neumann AI is an experimental orchestration layer designed for multi-agent LLM workflows. Our primary goal is to address the state synchronization overhead and context isolation issues prevalent in standard agent frameworks.

## Current Focus
* **Claude Ecosystem Optimization**: Native integration with the `anthropic` Python SDK, utilizing the latest `claude-5-5` model family.
* **Prompt Caching**: Aggressively leveraging Anthropic's ephemeral prompt caching to reduce token latency in repetitive agent loops.
* **Deterministic Execution**: Architectural design aimed at preventing JSON validation failures through strict schema proxies.

## Status
We are currently in the **Alpha / Developer Preview** phase. The repository contains proof-of-concept scripts (`anthropic_orchestrator.py`) demonstrating our caching and routing approach. 

> Note: Access to the full deployment infrastructure is currently invite-only. To request access to the Alpha preview, please contact us through the [Vonn Neumann AI Website](https://vonnneumann.com).

## Contact
Based in Çanakkale, Turkey.  
Email: contact@vonnneumann.com
