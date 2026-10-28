# Excerpt: src/services/agent/exhaustive.py
# Trimmed for the case study. "..." marks omitted code.
#
# Exhaustive mode: map-reduce over the whole document.
#
# Sections are packed into "map units" of roughly EXHAUSTIVE_UNIT_TOKENS tokens
# (at most EXHAUSTIVE_MAX_UNITS). The cheap model reads each unit and extracts
# verbatim findings relevant to the question; findings become evidence and the
# answer step reduces them. Each batch of MAP_CONCURRENCY units is one step, so
# the same step / token / time budget applies as in deep mode.

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from config import settings
from services import llm

MAX_UNIT_GROWTH = 4  # units may grow up to 4x the target size to fit under the unit cap

_MAP_PROMPT = """\
Question: {query}

Below is part of the document "{filename}" — section "{path}", pages {p1}-{p2}.
Extract every fact in it that helps answer the question, including partial or \
negative information. Copy a short verbatim quote (under 300 characters) for \
each. If nothing is relevant, return an empty list.

Reply with JSON only: {{"findings": [{{"fact": "...", "quote": "...", "page": <page number or null>}}]}}

---
{text}
"""


def build_map_units(
    sections: list[dict[str, Any]], unit_tokens: int, max_units: int
) -> list[dict[str, Any]]:
    """Pack section bodies into units; grow the unit size until they fit under max_units."""
    target = unit_tokens
    units = _pack(sections, target)
    while len(units) > max_units and target < unit_tokens * MAX_UNIT_GROWTH:
        target = int(target * 1.5)
        units = _pack(sections, target)
    if len(units) > max_units:
        # Said out loud in the logs; known limit for very long documents.
        units = units[:max_units]
    return units


def _pack(sections: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Neighbouring small sections share a unit; long ones split on paragraph breaks."""
    ...


async def run_exhaustive(query: str, store: DocStore, ws: Workspace, budget: Budget) -> AsyncIterator[Event]:
    units = build_map_units(
        await store.section_bodies(), settings.exhaustive_unit_tokens, settings.exhaustive_max_units
    )
    yield "step", {
        "n": 0, "tool": "plan", "args": {"units": len(units)},
        "summary": f"reading {len(units)} parts of the document",
    }

    size = max(1, settings.map_concurrency)  # 6 units read in parallel
    for start in range(0, len(units), size):
        reason = budget.exhausted()
        if reason:
            yield "step", {
                "n": budget.steps, "tool": "budget", "args": {},
                "summary": f"Stopped after {start}/{len(units)} parts: {reason}",
            }
            break
        batch = units[start : start + size]
        budget.steps += 1
        results = await asyncio.gather(*[_map_one(query, u, budget) for u in batch])

        kept = 0
        for unit, findings in zip(batch, results):
            for f in findings:
                # The model's page number is only trusted if it falls inside the unit.
                page = f.get("page")
                if not isinstance(page, int) or not unit["page_start"] <= page <= unit["page_end"]:
                    page = None
                quote = str(f.get("quote") or "").strip()
                fact = str(f.get("fact") or "").strip()
                p = ws.add_passage(
                    document_id=unit["document_id"], filename=unit["filename"],
                    text=quote or fact,
                    page_start=page or unit["page_start"], page_end=page or unit["page_end"],
                    section_path=unit["paths"][0],
                )
                ws.add_evidence([p.id], fact or None)
                kept += 1
        yield "step", {
            "n": budget.steps, "tool": "map",
            "args": {"parts": f"{start + 1}-{start + len(batch)} of {len(units)}"},
            "summary": f"read parts {start + 1}-{start + len(batch)} of {len(units)}: {kept} findings",
        }
