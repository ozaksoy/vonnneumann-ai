"""Offline tests (no API key needed): python tests_offline.py"""
from types import SimpleNamespace as NS
from vonnneumann import AgentStateRouter, SchemaValidationError

r = AgentStateRouter(system_prompt="x")
a, b = r.agent("a"), r.agent("b")
a.add_user("hi")
assert b.messages == [], "contexts must be isolated"

resp = NS(content=[NS(type="thinking", thinking="..."), NS(type="text", text='{"k": 1}')],
          usage=NS(input_tokens=10, output_tokens=5,
                   cache_creation_input_tokens=0, cache_read_input_tokens=8))
assert r.process_agent_state(resp, "a") == '{"k": 1}'
assert r.validate_json('{"k": 1}', {"k": "number"}) == {"k": 1}
for bad in ('nope', '{"x": 1}', '{"k": "1"}'):
    try:
        r.validate_json(bad, {"k": "number"}); raise AssertionError(bad)
    except SchemaValidationError:
        pass
print("All offline tests passed.")
