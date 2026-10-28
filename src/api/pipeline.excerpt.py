# Excerpt: src/services/rag_pipeline.py
# Trimmed for the case study. "..." marks omitted code.
#
# Query orchestration for all modes.
#
#   fast        retrieve top-k -> stream an answer that cites [n] passages
#   deep        bounded tool loop (agent/deep.py) -> answer -> check
#   exhaustive  map-reduce over sections (agent/exhaustive.py) -> answer -> check
#   auto        router picks one of the above
#
# stream_answer() yields (event, data) pairs used directly by the SSE endpoint:
#
#   mode       {"mode", "requested", "reason"}
#   step       {"n", "tool", "args", "summary"}             (deep / exhaustive)
#   sources    {"sources": [...]}                            evidence the answer may cite
#   token      {"text"}                                      answer text
#   citations  {"citations": [...], "removed": [...]}        markers used + sentences the checker dropped
#   done       {"latency_ms", "mode", "budget"}
#
# answer_query() runs the same generator and collects a QueryResponse, so the
# blocking and streaming endpoints can't drift apart.

from __future__ import annotations

import time
from typing import Any, AsyncIterator

import openai

from config import settings
from services.agent import answer as answer_mod
from services.agent.deep import run_deep
from services.agent.exhaustive import run_exhaustive
from services.agent.router import route
from services.agent.store import DocStore, PgStore
from services.agent.workspace import Budget, Workspace
from services.llm_client import get_openrouter_client

Event = tuple[str, dict[str, Any]]


async def stream_answer(
    query: str,
    document_ids: list[str] | None,
    k: int,
    session: AsyncSession,
    mode: str = "fast",
    store: DocStore | None = None,
) -> AsyncIterator[Event]:
    start_time = time.perf_counter()
    ...

    if _is_meta_query(query):
        # "Who are you?" gets a fixed intro, no retrieval, no model call.
        ...
        return

    # PgStore is scoped to the requested documents: tools can't read outside them.
    store = store or PgStore(session, document_ids)
    requested = mode
    reason = "requested"
    if mode == "auto":
        mode, reason = await route(query)
    yield "mode", {"mode": mode, "requested": requested, "reason": reason}

    ws = Workspace()
    budget = Budget()

    # ---- fast: one retrieval pass, streamed answer -----------------------
    if mode == "fast":
        passages = await _fast_passages(store, query, k, ws)
        yield "sources", {"sources": _sources(passages)}
        stream = await get_openrouter_client().chat.completions.create(
            model=settings.openrouter_model,
            max_tokens=1024,
            stream=True,
            messages=answer_mod.build_messages(query, passages),
        )
        text = ""
        async for event in stream:
            if event.choices and event.choices[0].delta.content:
                piece = event.choices[0].delta.content
                text += piece
                yield "token", {"text": piece}
        yield "citations", {"citations": answer_mod.citations(text, passages), "removed": []}
        yield "done", {"latency_ms": latency(), "mode": mode, "budget": budget.snapshot()}
        return

    # ---- deep / exhaustive: research, then answer, then check -------------
    runner = run_exhaustive if mode == "exhaustive" else run_deep
    try:
        async for event in runner(query, store, ws, budget):
            yield event
    except openai.BadRequestError:
        # e.g. the configured model doesn't support tool calls — degrade to fast retrieval.
        yield "step", {"n": budget.steps, "tool": "fallback", "args": {}, "summary": "agent unavailable, using fast retrieval"}
        await _fast_passages(store, query, max(k, 8), ws)

    if not ws.evidence:
        await _fast_passages(store, query, k, ws)

    passages = answer_mod.select_evidence(ws.evidence_passages())
    yield "sources", {"sources": _sources(passages)}

    text, used = await answer_mod.generate(query, passages)
    budget.add_tokens(used)
    removed: list[str] = []
    if settings.verify_answers and passages:
        text, removed, used = await answer_mod.check(text, passages)
        budget.add_tokens(used)
        yield "step", {
            "n": budget.steps, "tool": "check", "args": {},
            "summary": f"checked answer: removed {len(removed)} unsupported sentence(s)",
        }

    # Deep and exhaustive answers are sent only after the check, so a removed
    # sentence is never shown and then taken back.
    for piece in _paragraphs(text):
        yield "token", {"text": piece}
    yield "citations", {"citations": answer_mod.citations(text, passages), "removed": removed}
    yield "done", {"latency_ms": latency(), "mode": mode, "budget": budget.snapshot()}
