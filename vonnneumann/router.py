"""Core state router.

Three ideas:
1. Shared system prompt, sent with cache_control so Claude prompt caching
   can reuse it across agents (min. cacheable length on Sonnet 5.5: 512 tokens).
2. Isolated per-agent context: each agent keeps its own message history,
   so parallel agents never see each other's turns.
3. Schema validation: agent outputs that must be JSON are checked against
   a simple schema before they are passed downstream.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5-5"

_TYPES = {"string": str, "number": (int, float), "integer": int,
          "boolean": bool, "object": dict, "array": list}


class SchemaValidationError(ValueError):
    """Raised when an agent's JSON output does not match its schema."""


@dataclass
class AgentContext:
    """Isolated message history for one agent."""
    name: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


class AgentStateRouter:
    def __init__(self, system_prompt: str = "", cache_ttl_sec: int = 300,
                 model: str = DEFAULT_MODEL):
        self.system_prompt = system_prompt
        self.cache_ttl_sec = cache_ttl_sec  # Claude's ephemeral cache lives ~5 min
        self.model = model
        self._agents: dict[str, AgentContext] = {}
        self.stats: list[dict[str, Any]] = []

    # ---- shared, cacheable system prompt -------------------------------
    def get_cached_system_prompt(self) -> str:
        return self.system_prompt

    def system_block(self) -> list[dict[str, Any]]:
        return [{"type": "text", "text": self.system_prompt,
                 "cache_control": {"type": "ephemeral"}}]

    # ---- isolated agent contexts ---------------------------------------
    def agent(self, name: str) -> AgentContext:
        if name not in self._agents:
            self._agents[name] = AgentContext(name)
        return self._agents[name]

    # ---- response handling ---------------------------------------------
    @staticmethod
    def extract_text(response: Any) -> str:
        """Return only text blocks (Sonnet 5.5 may also return thinking blocks)."""
        return "".join(b.text for b in response.content
                       if getattr(b, "type", None) == "text")

    def process_agent_state(self, response: Any, agent_name: str = "default",
                            latency_s: float | None = None) -> str:
        text = self.extract_text(response)
        self.agent(agent_name).add_assistant(text)
        u = response.usage
        self.stats.append({
            "agent": agent_name,
            "latency_s": latency_s,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        })
        return text

    # ---- one call ------------------------------------------------------
    async def run(self, client: Any, agent_name: str, prompt: str,
                  max_tokens: int = 1024, use_cache: bool = True) -> str:
        ctx = self.agent(agent_name)
        ctx.add_user(prompt)
        system = self.system_block() if use_cache else self.system_prompt
        start = time.perf_counter()
        # Note: temperature/top_p/top_k are intentionally not set
        # (non-default values return a 400 on Sonnet 5.5).
        response = await client.messages.create(
            model=self.model, max_tokens=max_tokens,
            system=system, messages=ctx.messages)
        return self.process_agent_state(response, agent_name,
                                        time.perf_counter() - start)

    # ---- schema validation ---------------------------------------------
    @staticmethod
    def validate_json(text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Parse text as JSON and check required keys and basic types.

        schema example: {"summary": "string", "score": "number"}
        """
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"Not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise SchemaValidationError("Top-level JSON must be an object")
        for key, type_name in schema.items():
            if key not in data:
                raise SchemaValidationError(f"Missing key: {key}")
            expected = _TYPES[type_name]
            value = data[key]
            if isinstance(value, bool) and type_name in ("number", "integer"):
                raise SchemaValidationError(f"{key}: expected {type_name}")
            if not isinstance(value, expected):
                raise SchemaValidationError(f"{key}: expected {type_name}")
        return data

    # ---- summary -------------------------------------------------------
    def report(self) -> str:
        lines = ["agent | latency_s | input | output | cache_write | cache_read"]
        for s in self.stats:
            lat = f"{s['latency_s']:.2f}" if s["latency_s"] is not None else "-"
            lines.append(f"{s['agent']} | {lat} | {s['input_tokens']} | "
                         f"{s['output_tokens']} | {s['cache_write_tokens']} | "
                         f"{s['cache_read_tokens']}")
        return "\n".join(lines)
