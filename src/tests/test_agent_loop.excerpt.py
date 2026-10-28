# Excerpt: tests/test_agent_loop.py and tests/fakes.py
# Trimmed for the case study. "..." marks omitted code.
#
# The agent is tested without a database, a GPU or an API key. FakeStore is an
# in-memory 5-page annual report behind the same DocStore interface as Postgres.
# ScriptedLLM replaces services.llm.chat and plays a fixed script per role
# (planner, mapper, answer, checker, router), so every test is deterministic.
# The whole suite (60 tests) runs in under a second.

import json

from fakes import FakeStore, ScriptedLLM, text_turn, tool_turn

from services import llm
from services.agent.deep import run_deep
from services.agent.workspace import Budget, Workspace


# --- fakes.py ---------------------------------------------------------------

class ScriptedLLM:
    """Replaces services.llm.chat. Routes by model/tools so each role gets its own script."""

    async def __call__(self, messages, *, model=None, max_tokens=1024, temperature=0.0, tools=None):
        prompt = messages[-1]["content"] if isinstance(messages[-1].get("content"), str) else ""
        self.calls.append({"model": model, "tools": bool(tools), "prompt": prompt, "n_messages": len(messages)})
        if tools:
            return self.planner.pop(0) if self.planner else text_turn("done")
        if "Reply with JSON only: {\"unsupported\"" in prompt:
            return text_turn(json.dumps(self.check))
        if "Reply with JSON only: {\"findings\"" in prompt:
            return text_turn(json.dumps(self.mapper(prompt)), tokens=200)
        ...
        return text_turn(self.answer, tokens=300)


# --- test_agent_loop.py -----------------------------------------------------

async def _collect(gen):
    return [item async for item in gen]


async def test_deep_loop_runs_tools_keeps_evidence_and_finishes(monkeypatch):
    script = ScriptedLLM(planner=[
        tool_turn(("outline", {"document": "D1"}), ("search", {"query": "revenue 2024"})),
        tool_turn(("add_evidence", {"passages": ["P1"], "note": "revenue"}), ("finish", {})),
        tool_turn(("search", {"query": "never reached"})),
    ])
    monkeypatch.setattr(llm, "chat", script)
    ws, budget = Workspace(), Budget(max_steps=8, max_tokens=10_000, time_limit_s=60)

    events = await _collect(run_deep("What was revenue in 2024?", FakeStore(), ws, budget))

    tools = [d["tool"] for _, d in events]
    assert tools == ["outline", "search", "add_evidence"]
    assert [d["n"] for _, d in events] == [1, 2, 3]
    assert ws.evidence == ["P1"] and "$12.4M" in ws.passages["P1"].text
    assert budget.steps == 3 and budget.tokens == 200
    # The planner saw tool results for every call it made (OpenAI tool-call protocol).
    assert script.calls[1]["n_messages"] == 2 + 1 + 2  # system, user, assistant, 2 tool results


async def test_deep_loop_stops_at_step_limit(monkeypatch):
    many = [tool_turn(("search", {"query": f"q{i}"}), ("search", {"query": f"r{i}"})) for i in range(10)]
    monkeypatch.setattr(llm, "chat", ScriptedLLM(planner=many))
    ws, budget = Workspace(), Budget(max_steps=3, max_tokens=10**6, time_limit_s=60)

    events = await _collect(run_deep("anything about steel", FakeStore(), ws, budget))

    tool_steps = [d for _, d in events if d["tool"] == "search"]
    assert len(tool_steps) == 3
    assert any(d["tool"] == "budget" and "step limit" in d["summary"] for _, d in events)


async def test_deep_loop_stops_at_token_budget(monkeypatch):
    many = [tool_turn(("search", {"query": "steel"}), tokens=600) for _ in range(10)]
    monkeypatch.setattr(llm, "chat", ScriptedLLM(planner=many))
    ws, budget = Workspace(), Budget(max_steps=8, max_tokens=1000, time_limit_s=60)

    events = await _collect(run_deep("steel?", FakeStore(), ws, budget))

    assert budget.steps == 2
    assert events[-2][1]["tool"] == "budget" and "token budget" in events[-2][1]["summary"]


async def test_deep_loop_auto_keeps_top_passages_when_planner_keeps_none(monkeypatch):
    monkeypatch.setattr(llm, "chat", ScriptedLLM(planner=[tool_turn(("search", {"query": "risks currency"}))]))
    ws = Workspace()
    events = await _collect(run_deep("risks?", FakeStore(), ws, Budget()))
    assert ws.evidence
    assert events[-1][1]["summary"].startswith("auto-kept")
