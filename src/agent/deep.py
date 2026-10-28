# From: src/services/agent/deep.py (complete, imports trimmed)
#
# Deep mode: a bounded tool-calling loop.
#
# The planner model sees the question and the document list, then calls tools
# (outline, search, keyword_scan, read_pages, read_section, add_evidence, finish)
# until it finishes or the budget (steps, tokens, wall time) runs out. Each tool
# call is yielded as a `step` event so the UI can show progress.

from __future__ import annotations

from typing import Any, AsyncIterator

from config import settings
from services import llm
from services.agent.store import DocStore
from services.agent.tools import TOOL_SPECS, ToolBox
from services.agent.workspace import Budget, Workspace

Event = tuple[str, dict[str, Any]]

_SYSTEM = """\
You are the research planner for DocMind, a document Q&A system. Your job is to \
find the passages in the uploaded documents that answer the user's question. You \
do not write the final answer; another step writes it from the evidence you keep.

How to work:
- Start from the document list. Use `outline` when the question names a chapter, \
section or topic, or when you need to know where things are.
- Use `search` for concepts, `keyword_scan` for exact names, numbers, codes or defined terms.
- Use `read_section` / `read_pages` when a passage is cut off or the answer needs \
surrounding context (tables, lists, definitions that continue on the next page).
- Call `add_evidence` with the passage ids (P1, P2…) that directly support the \
answer. Keep everything the answer will need, including passages that show \
something is absent or contradicted. Do not keep irrelevant passages.
- Call `finish` as soon as the evidence is sufficient. Prefer several tools in one \
turn when they are independent.
- You have at most {max_steps} tool calls. Spend them wisely.\
"""


async def run_deep(query: str, store: DocStore, ws: Workspace, budget: Budget) -> AsyncIterator[Event]:
    tools = ToolBox(store, ws)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM.format(max_steps=budget.max_steps)},
        {
            "role": "user",
            "content": f"Documents:\n{await tools.doc_ref_list()}\n\nQuestion: {query}",
        },
    ]

    finished = False
    while not finished:
        # Checked before every model call: steps, tokens and wall time.
        reason = budget.exhausted()
        if reason:
            yield "step", {"n": budget.steps, "tool": "budget", "args": {}, "summary": f"Stopped: {reason}"}
            break

        result = await llm.chat(messages, model=settings.openrouter_model, tools=TOOL_SPECS, max_tokens=1200)
        budget.add_tokens(result.tokens)
        messages.append(result.message)
        if not result.tool_calls:
            break

        for call in result.tool_calls:
            if call.name == "finish":
                finished = True
                messages.append({"role": "tool", "tool_call_id": call.id, "content": "ok"})
                continue
            if budget.steps_left() <= 0 or finished:
                # Every tool call still gets a result, so the conversation stays valid
                # for the provider, but nothing past the limit actually runs.
                messages.append({
                    "role": "tool", "tool_call_id": call.id,
                    "content": "Skipped: tool-call budget exhausted.",
                })
                continue
            budget.steps += 1
            output, summary = await tools.run(call.name, call.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            yield "step", {"n": budget.steps, "tool": call.name, "args": call.arguments, "summary": summary}

    # The planner may stop without keeping anything; fall back to what it found.
    if not ws.evidence and ws.passages:
        ranked = sorted(ws.passages.values(), key=lambda p: p.score, reverse=True)[:8]
        ws.add_evidence([p.id for p in ranked], "auto-selected: planner kept no evidence")
        yield "step", {
            "n": budget.steps, "tool": "add_evidence", "args": {},
            "summary": f"auto-kept {len(ranked)} top passages",
        }
