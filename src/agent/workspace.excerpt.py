# Excerpt: src/services/agent/workspace.py
# Trimmed for the case study. "..." marks omitted code.
#
# Per-query scratch state for the agent: passages it has seen, evidence it kept,
# short references for documents/sections, and the budget it is spending.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from config import settings


@dataclass
class Budget:
    # Defaults: 8 tool calls, 120,000 tokens, 120 seconds. One budget is shared
    # by every LLM call in a run: planner turns, map batches, the answer and the check.
    max_steps: int = field(default_factory=lambda: settings.agent_max_steps)
    max_tokens: int = field(default_factory=lambda: settings.agent_token_budget)
    time_limit_s: float = field(default_factory=lambda: settings.agent_time_limit_s)
    steps: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    def add_tokens(self, n: int) -> None:
        self.tokens += max(0, int(n))

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def steps_left(self) -> int:
        return max(0, self.max_steps - self.steps)

    def exhausted(self) -> str | None:
        """A human-readable reason, so the stream can say *why* the agent stopped."""
        if self.steps >= self.max_steps:
            return f"step limit ({self.max_steps}) reached"
        if self.tokens >= self.max_tokens:
            return f"token budget ({self.max_tokens}) spent"
        if self.elapsed() >= self.time_limit_s:
            return f"time limit ({self.time_limit_s:.0f}s) reached"
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"steps": self.steps, "tokens": self.tokens, "seconds": round(self.elapsed(), 1)}


class Workspace:
    def __init__(self) -> None:
        self.passages: dict[str, Passage] = {}
        self.evidence: list[str] = []
        self._dedupe: dict[tuple[Any, ...], str] = {}
        self.refs: dict[str, str] = {}       # "S4" / "D1" -> UUID
        self._ref_of: dict[str, str] = {}    # UUID -> "S4"

    def ref(self, prefix: str, uuid: str) -> str:
        """Short ids (D1, S4) instead of UUIDs: fewer tokens, fewer copy mistakes by the model."""
        ...

    def add_passage(self, **fields: Any) -> Passage:
        # The same text seen twice (search, then read_pages) keeps one id.
        key = (fields.get("document_id"), fields.get("chunk_id"), fields.get("page_start"), fields["text"][:200])
        if key in self._dedupe:
            return self.passages[self._dedupe[key]]
        pid = f"P{len(self.passages) + 1}"
        passage = Passage(id=pid, **fields)
        self.passages[pid] = passage
        self._dedupe[key] = pid
        return passage

    def add_evidence(self, ids: list[str], note: str | None = None) -> tuple[list[str], list[str]]:
        added, unknown = [], []
        for raw in ids:
            pid = str(raw).strip().upper()
            if pid not in self.passages:
                unknown.append(str(raw))   # an invented id can't become evidence
                continue
            if pid not in self.evidence:
                self.evidence.append(pid)
                if note:
                    self.passages[pid].note = note
            added.append(pid)
        return added, unknown
