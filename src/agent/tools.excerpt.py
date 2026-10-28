# Excerpt: src/services/agent/tools.py
# Trimmed for the case study. "..." marks omitted code.
#
# Tools the deep-mode agent can call. Each returns (text for the model, one-line
# summary for the UI step event). Every passage a tool shows the model is
# registered in the workspace under an id like P7 so it can be kept as evidence.
#
# All seven tools are read-only. The agent can look things up and keep
# evidence; it cannot write, delete, fetch URLs or run code.

from __future__ import annotations

from typing import Any

from services.agent.store import DocStore
from services.agent.workspace import Workspace

MAX_PAGES_PER_READ = 5
PAGE_CHARS = 6_000
SECTION_CHARS = 14_000
SEARCH_CHARS = 1_500
OUTPUT_CHARS = 24_000   # hard cap on any single tool result sent back to the model

TOOL_SPECS: list[dict[str, Any]] = [
    _fn("outline", "Show a document's section tree: section refs (S1, S2…), titles, page ranges and summaries. "
                   "Use it to find where a topic lives before reading.", ...),
    _fn("search", "Semantic + keyword search over all chunks; returns the best passages with pages and sections.", ...),
    _fn("keyword_scan", "Find exact occurrences of terms (names, codes, numbers, defined terms) across all pages.", ...),
    _fn("read_pages", f"Read the full text of up to {MAX_PAGES_PER_READ} consecutive pages of a document.", ...),
    _fn("read_section", "Read the full text of one section (including its subsections).", ...),
    _fn("add_evidence", "Keep passages that support the answer. Only kept evidence can be cited in the final answer.", ...),
    _fn("finish", "Stop: the kept evidence is enough to answer (or nothing more can be found).", {}, []),
]


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " …[truncated]"


class ToolBox:
    def __init__(self, store: DocStore, ws: Workspace) -> None:
        self.store = store
        self.ws = ws

    async def run(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return f"Unknown tool '{name}'.", f"unknown tool {name}"
        try:
            text, summary = await handler(**args)
        except TypeError as exc:
            return f"Bad arguments for {name}: {exc}", f"{name}: bad arguments"
        except Exception as exc:  # tool failures are reported to the model, not raised
            return f"{name} failed: {exc}", f"{name} failed"
        return _clip(text, OUTPUT_CHARS), summary

    async def _t_search(self, query: str, k: int = 6) -> tuple[str, str]:
        k = max(1, min(int(k or 6), 10))
        hits = await self.store.search(query, k)
        blocks = []
        for h in hits:
            p = self.ws.add_passage(
                document_id=h["document_id"], filename=h["filename"], text=h["text"],
                page_start=h.get("page_start"), page_end=h.get("page_end"),
                section_path=h.get("section_path"), chunk_id=h.get("chunk_id"),
                chunk_index=h.get("chunk_index", -1),
                score=float(h.get("rerank_score", h.get("score", 0.0)) or 0.0),
            )
            blocks.append(f"[{p.id}] {p.location()}\n{_clip(p.text, SEARCH_CHARS)}")
        text = "\n\n".join(blocks) or "No passages found."
        return text, f"search “{query}”: {len(hits)} passages"

    async def _t_read_pages(self, document: str, start: int, end: int | None = None) -> tuple[str, str]:
        doc_id = self.ws.resolve(document)   # "D1" -> UUID; unknown refs resolve to None
        start = max(1, int(start))
        end = int(end or start)
        end = max(start, min(end, start + MAX_PAGES_PER_READ - 1))
        pages = await self.store.read_pages(doc_id, start, end) if doc_id else []
        ...

    async def _t_add_evidence(self, passages: list[str] | str, note: str | None = None) -> tuple[str, str]:
        if isinstance(passages, str):
            passages = [passages]
        # Only ids a tool actually showed the model can become evidence.
        added, unknown = self.ws.add_evidence(list(passages), note)
        msg = f"Kept {len(added)} passage(s) as evidence ({len(self.ws.evidence)} total)."
        if unknown:
            msg += f" Unknown ids ignored: {', '.join(unknown)}."
        return msg, f"kept {', '.join(added) or 'nothing'} as evidence"

    # _t_outline, _t_keyword_scan, _t_read_section, _t_finish ...
