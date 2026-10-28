# Excerpt: src/services/agent/answer.py
# Trimmed for the case study. "..." marks omitted code.
#
# Answer from evidence only, then check it.
#
# - build_messages numbers the evidence [1]..[n] with file, pages and section.
# - The answer model must cite every factual sentence with [n].
# - check() asks the cheap model which sentences the evidence does not support
#   and removes them (sentences, not whole paragraphs, so the rest survives).
# - citations() maps the [n] markers actually used to their pages and quotes.

from __future__ import annotations

import re
from typing import Any

from config import settings
from services import llm
from services.agent.workspace import Passage

EVIDENCE_CHARS = 2_500
MAX_EVIDENCE = 40

SYSTEM_PROMPT = """\
You are DocMind, an assistant that answers questions strictly from uploaded documents.

You receive numbered evidence passages [1], [2], … each labelled with its file, \
pages and section. Passages may contain Markdown tables, [IMAGE] descriptions \
(treat as factual) and [HYPERLINKS].

Rules:
- Use only the evidence. No outside knowledge, no guessing.
- Put a citation like [2] or [1][4] right after every sentence that states a fact \
from the documents. Only cite passages that actually support that sentence.
- Synthesise across passages; reason carefully over table values.
- If the evidence does not fully answer the question, answer what it supports \
and say plainly what the documents do not cover.
...
"""

NO_EVIDENCE_NOTE = (
    "No relevant passages were found in the uploaded documents. Say briefly that the "
    "documents do not contain this information. Do not answer from general knowledge."
)

_CHECK_PROMPT = """\
You are checking an answer against its evidence. For each numbered sentence of the \
answer, decide whether the cited evidence (or any evidence below) supports it. \
Sentences that only say information is missing, or that introduce/structure the \
answer without stating facts, count as supported.

Evidence:
{evidence}

Answer sentences:
{sentences}

Reply with JSON only: {{"unsupported": [<sentence numbers>]}}"""


def evidence_block(passages: list[Passage]) -> str:
    return "\n\n".join(
        f"[{i}] {p.location()}\n{p.text[:EVIDENCE_CHARS]}" for i, p in enumerate(passages, start=1)
    )


# Splits into alternating [text, sep, text, sep, …] so the answer can be rebuilt
# exactly, minus the sentences the checker rejects.
_SPLIT_RE = re.compile(r"(\n+|(?<=[.!?])\s+(?=[A-Z0-9\"'(*\-]))")


def remove_sentences(answer: str, numbers: set[int]) -> tuple[str, list[str]]:
    """Drop claim sentences by 1-based number (as shown to the checker)."""
    ...


async def check(answer: str, passages: list[Passage]) -> tuple[str, list[str], int]:
    """Return (answer without unsupported sentences, removed sentences, tokens used)."""
    claims = numbered_claims(answer)
    if not claims or not passages:
        return answer, [], 0
    prompt = _CHECK_PROMPT.format(
        evidence=evidence_block(passages),
        sentences="\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1)),
    )
    try:
        res = await llm.chat([{"role": "user", "content": prompt}], model=settings.fast_model, max_tokens=200)
        data = llm.parse_json(res.content)
        bad = {int(x) for x in (data.get("unsupported", []) if isinstance(data, dict) else []) if str(x).isdigit()}
    except Exception:
        # A checker outage keeps the answer; it never blanks it.
        return answer, [], 0
    bad = {b for b in bad if 1 <= b <= len(claims)}   # ignore numbers the checker made up
    if not bad:
        return answer, [], res.tokens
    cleaned, removed = remove_sentences(answer, bad)
    if not cleaned.strip():
        cleaned = "I couldn't verify an answer to this question against the documents."
    return cleaned, removed, res.tokens


_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def citations(answer: str, passages: list[Passage]) -> list[dict[str, Any]]:
    """Only markers that point at a real passage become citations; [99] is ignored."""
    used: list[int] = []
    for m in _MARKER_RE.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(passages) and n not in used:
            used.append(n)
    out = []
    for n in sorted(used):
        p = passages[n - 1]
        out.append({
            "marker": n,
            "document_id": p.document_id,
            "filename": p.filename,
            "page_start": p.page_start,
            "page_end": p.page_end,
            "section_path": p.section_path,
            "quote": p.text[:300],
        })
    return out
